"""COCO detection accuracy (mAP50-95 / mAP50) for any registered backend.

This is the number the quantization promotion gate turns on: "INT8 mAP50-95 drop
<= 3% absolute vs the FP32 baseline". Agreement-with-FP32 (see
``benchmark_frontier.py``) answers a different, softer question -- this one is
comparable to published figures and to the baseline measured on the *same class
set*, which is why the evaluated class set is recorded in the output JSON.

Two decisions are load-bearing and easy to get silently wrong:

1. **Detection, not tracking.** ``Detector.track`` runs ``model.track``, and
   Ultralytics *replaces* the result boxes with the tracker's output -- boxes
   ByteTrack did not associate simply disappear. mAP integrates the whole
   precision/recall curve, so those boxes matter. This module therefore drives
   the loaded model's ``predict`` path directly and reuses ``Detector`` only for
   backend resolution and model loading.
2. **Confidence floor.** The serving default (0.25) is a UI decision. Evaluating
   at 0.25 truncates the PR curve and understates AP by a large margin, so the
   default here is 0.001 with ``max_det=100``, matching the COCO protocol.

Usage:
    python ml/eval/eval_coco.py --engine fp32_gpu --classes prd6
    python ml/eval/eval_coco.py --engine openvino_cpu --imgsz 640 --limit 500
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.backends import BACKENDS, availability, get_backend  # noqa: E402
from app.config import get_settings, probe_gpu  # noqa: E402
from app.detector import Detector  # noqa: E402

REPORTS_DIR = REPO_ROOT / "ml" / "eval" / "reports"
DEFAULT_IMAGES_DIR = REPO_ROOT / "ml" / "data" / "raw" / "coco" / "val2017"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "ml" / "data" / "raw" / "coco" / "annotations" / "instances_val2017.json"
)
DOWNLOAD_HINT = "run ml/data/scripts/download_coco.py to fetch the val2017 subset"

#: The class set the PRD pins the baseline and the INT8 target to. The deployed
#: detector is fine-tuned down to these six, so its baseline and its target must
#: both be measured here -- an 80-class number is not comparable.
PRD_CLASS_SUBSET: tuple[str, ...] = ("person", "bicycle", "car", "motorcycle", "bus", "truck")

#: COCO caps detections per image at 100 when computing the headline AP.
COCO_MAX_DET = 100

#: Low enough not to truncate the precision/recall curve; see the module docstring.
EVAL_CONF_THRESHOLD = 0.001


# --------------------------------------------------------------------- convert


def xyxy_to_coco_bbox(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Convert a corner box to COCO's ``[x, y, width, height]``.

    Ultralytics reports ``xyxy`` in pixels of the *original* image (it undoes its
    own letterboxing), so no rescaling is needed here -- only the corner-to-size
    change. Feeding xyxy straight into a COCO result file produces boxes whose
    width is the right edge, which overlaps nothing and scores a flat zero mAP
    without ever raising.
    """
    return [
        round(float(x1), 2),
        round(float(y1), 2),
        round(float(x2) - float(x1), 2),
        round(float(y2) - float(y1), 2),
    ]


def normalise_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def build_category_map(
    class_names: Mapping[int, str],
    categories: Iterable[Mapping[str, Any]],
    required: Iterable[str] | None = None,
) -> dict[int, int]:
    """Map the model's contiguous class indices onto COCO category ids.

    The model numbers its classes 0..N-1. COCO category ids are **not**
    contiguous -- ``instances_val2017.json`` uses ids in 1..90 with gaps, so
    ``person`` is 1 but ``truck`` is 8 while the 80-class model calls them 0 and
    7. Submitting the model index as ``category_id`` matches the wrong category
    (or none at all) and yields a silent zero AP, which is the single most common
    way a COCO harness lies. Matching on the class *name* is also what keeps this
    correct for the fine-tuned 6-class model, whose indices are 0..5.

    Args:
        class_names: model index -> class name.
        categories: the ``categories`` list from the COCO annotation file.
        required: class names that must resolve; a miss raises. Names outside
            this set that do not resolve are omitted, and their detections are
            dropped and counted by the caller rather than mis-assigned.

    Raises:
        ValueError: a required class name has no COCO category.
    """
    by_name = {normalise_name(str(c["name"])): int(c["id"]) for c in categories}
    required_set = {normalise_name(n) for n in required} if required is not None else None

    mapping: dict[int, int] = {}
    missing: list[str] = []
    for index, name in class_names.items():
        category_id = by_name.get(normalise_name(str(name)))
        if category_id is not None:
            mapping[int(index)] = category_id
        elif required_set is None or normalise_name(str(name)) in required_set:
            missing.append(str(name))

    if missing:
        raise ValueError(
            "these model classes have no matching COCO category, so their "
            f"detections cannot be scored: {sorted(set(missing))}"
        )
    return mapping


def select_class_indices(
    class_names: Mapping[int, str], requested: Sequence[str] | None
) -> list[int]:
    """Resolve requested class names to model indices, preserving model order.

    Raises:
        ValueError: a requested name is not one of the model's classes.
    """
    if not requested:
        return sorted(int(i) for i in class_names)
    wanted = {normalise_name(n) for n in requested}
    chosen = [int(i) for i, name in class_names.items() if normalise_name(str(name)) in wanted]
    found = {normalise_name(str(class_names[i])) for i in chosen}
    unknown = sorted(wanted - found)
    if unknown:
        raise ValueError(
            f"the model does not have these classes: {unknown}; "
            f"available: {sorted(normalise_name(str(n)) for n in class_names.values())}"
        )
    return sorted(chosen)


# ----------------------------------------------------------------- data access


def load_annotations(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"COCO annotations not found: {path} ({DOWNLOAD_HINT})")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_images(
    annotations: Mapping[str, Any], images_dir: Path, limit: int | None
) -> list[tuple[int, Path]]:
    """Pair annotation image ids with files on disk.

    The pairing goes through ``file_name`` rather than parsing the id out of the
    filename: the id is authoritative in the JSON, and a wrong id maps every
    prediction to the wrong image (another silent-zero path).
    """
    if not images_dir.is_dir():
        raise SystemExit(f"COCO images directory not found: {images_dir} ({DOWNLOAD_HINT})")

    entries = sorted(annotations["images"], key=lambda item: int(item["id"]))
    pairs: list[tuple[int, Path]] = []
    for entry in entries:
        candidate = images_dir / str(entry["file_name"])
        if candidate.exists():
            pairs.append((int(entry["id"]), candidate))
        if limit is not None and len(pairs) >= limit:
            break

    if not pairs:
        raise SystemExit(
            f"no annotated image files found in {images_dir} ({DOWNLOAD_HINT}); "
            "the directory and the annotation file must describe the same split"
        )
    return pairs


def model_class_names(detector: Detector) -> dict[int, str]:
    """Read the loaded model's index -> name table.

    Reaches into the Ultralytics object because ``Detector`` deliberately exposes
    only the detect+track contract the API needs; the class table is metadata
    that every export format carries, and duplicating an 80-name list here would
    be wrong the moment the model is fine-tuned.
    """
    model = detector._model
    names = getattr(model, "names", None)
    if not names:
        raise SystemExit(f"backend '{detector.backend.key}' exposes no class names")
    return {int(index): str(name) for index, name in dict(names).items()}


# ------------------------------------------------------------------ prediction


def detach_tracker(detector: Detector) -> None:
    """Guarantee this module measures the detector, not the tracker.

    ``Detector`` exists to serve the streaming API, so its warmup calls
    ``model.track``. Ultralytics' ``Model.track`` registers persistent
    ``on_predict_start`` / ``on_predict_postprocess_end`` callbacks on the model,
    and ``Model.predict`` reuses that same predictor and callback dict. The
    tracker therefore keeps firing on subsequent ``predict`` calls, replacing the
    detector's boxes with Kalman-smoothed track boxes and discarding everything
    ByteTrack did not associate.

    Measured on one 1080p image: 19 boxes with a 0.264 score floor through the
    leaked path, versus 100 boxes down to 0.015 from a clean model. COCO needs
    that low-confidence tail -- truncating it understates mAP badly -- and track
    state carried between unrelated images makes the result order-dependent.

    So: warm up through ``predict``, and strip the tracker callbacks. The
    assertion in :func:`predict_dataset` is the real guard; this is the fix.
    """
    model = detector._model
    if model is None:
        raise SystemExit("detector was not loaded")

    callbacks = getattr(model, "callbacks", None)
    if isinstance(callbacks, dict):
        for hook in ("on_predict_start", "on_predict_postprocess_end"):
            handlers = callbacks.get(hook)
            if isinstance(handlers, list):
                handlers.clear()

    import numpy as np

    blank = np.zeros((detector.imgsz, detector.imgsz, 3), dtype=np.uint8)
    for _ in range(2):
        model.predict(blank, imgsz=detector.imgsz, device=detector.backend.device, verbose=False)


def predict_dataset(
    detector: Detector,
    images: Sequence[tuple[int, Path]],
    class_indices: Sequence[int],
    category_map: Mapping[int, int],
    conf: float,
    iou: float,
) -> tuple[list[dict[str, Any]], int]:
    """Run the model over every image and emit COCO-format results.

    Returns:
        ``(detections, dropped)`` where *dropped* counts boxes whose class had no
        COCO category. Reported rather than swallowed: a non-zero count means the
        class mapping is incomplete and the mAP is understated.
    """
    import cv2

    # The tracking path would hide boxes ByteTrack did not associate; see the
    # module docstring. This is the same loaded model, driven as a detector.
    model = detector._model
    results: list[dict[str, Any]] = []
    dropped = 0

    for position, (image_id, path) in enumerate(images, start=1):
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not decode {path}")

        prediction = model.predict(
            image,
            imgsz=detector.imgsz,
            conf=conf,
            iou=iou,
            device=detector.backend.device,
            classes=list(class_indices),
            max_det=COCO_MAX_DET,
            verbose=False,
        )[0]

        boxes = getattr(prediction, "boxes", None)
        # If ids are present the tracker is still attached and these are track
        # boxes, not detections. One line, and it is the line that would have
        # caught a whole run of silently wrong mAP numbers.
        if boxes is not None and getattr(boxes, "id", None) is not None:
            raise SystemExit(
                "tracker callbacks are attached: predictions carry track ids, so this would "
                "measure ByteTrack rather than the detector. Call detach_tracker() first."
            )
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for row in range(len(xyxy)):
                category_id = category_map.get(int(classes[row]))
                if category_id is None:
                    dropped += 1
                    continue
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": xyxy_to_coco_bbox(*xyxy[row]),
                        "score": round(float(scores[row]), 5),
                    }
                )

        if position % 100 == 0 or position == len(images):
            print(f"  {position}/{len(images)} images, {len(results)} detections")

    return results, dropped


# ------------------------------------------------------------------ evaluation


def evaluate(
    annotations_path: Path,
    detections: list[dict[str, Any]],
    image_ids: Sequence[int],
    category_ids: Sequence[int],
) -> dict[str, Any]:
    """Score detections with pycocotools, restricted to the evaluated subset."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # pycocotools prints index-building chatter to stdout; keep the console useful.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotations_path))
        coco_dt = coco_gt.loadRes(list(detections))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.catIds = list(category_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = [float(value) for value in evaluator.stats]
    return {
        "map50_95": round(stats[0], 5),
        "map50": round(stats[1], 5),
        "map75": round(stats[2], 5),
        "map_small": round(stats[3], 5),
        "map_medium": round(stats[4], 5),
        "map_large": round(stats[5], 5),
        "ar_max100": round(stats[8], 5),
        "per_class_ap50_95": per_class_ap(evaluator),
    }


def per_class_ap(evaluator: Any) -> dict[str, float]:
    """AP50-95 per category id, read out of the accumulated precision tensor."""
    precision = evaluator.eval.get("precision") if evaluator.eval else None
    if precision is None:
        return {}
    # precision is [iou_threshold, recall, category, area_range, max_dets];
    # area index 0 is "all" and max-dets index 2 is 100, matching stats[0].
    output: dict[str, float] = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        slice_ = precision[:, :, index, 0, 2]
        valid = slice_[slice_ > -1]
        output[str(category_id)] = round(float(valid.mean()), 5) if valid.size else 0.0
    return output


def empty_metrics() -> dict[str, Any]:
    return {
        "map50_95": 0.0,
        "map50": 0.0,
        "map75": 0.0,
        "map_small": 0.0,
        "map_medium": 0.0,
        "map_large": 0.0,
        "ar_max100": 0.0,
        "per_class_ap50_95": {},
    }


# ------------------------------------------------------------------------ main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--engine",
        "--precision",
        dest="engine",
        default="fp32_gpu",
        choices=sorted(BACKENDS),
        help="inference backend to evaluate",
    )
    parser.add_argument("--imgsz", "--resolution", dest="imgsz", type=int, default=640)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--limit", type=int, help="evaluate only the first N annotated images")
    parser.add_argument(
        "--classes",
        nargs="+",
        help=(
            "class names to evaluate, or the shorthand 'prd6' for "
            f"{', '.join(PRD_CLASS_SUBSET)}. Default: every class the model has."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=EVAL_CONF_THRESHOLD,
        help="confidence floor (COCO protocol wants this near zero)",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--json", dest="json_out", type=Path, help="output path for the report")
    return parser.parse_args(argv)


def resolve_class_request(requested: Sequence[str] | None) -> tuple[list[str] | None, str]:
    """Expand the ``prd6`` shorthand and label the class set for the report."""
    if not requested:
        return None, "model-all"
    if len(requested) == 1 and requested[0].lower() == "prd6":
        return list(PRD_CLASS_SUBSET), "prd6"
    return list(requested), "custom"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_classes, class_set_label = resolve_class_request(args.classes)

    base = get_settings()
    gpu = probe_gpu()
    backend = get_backend(args.engine)

    runnable, reason = availability(backend, base, gpu.available)
    if not runnable:
        raise SystemExit(f"backend '{args.engine}' cannot run here: {reason}")
    if not backend.supports_imgsz(args.imgsz):
        raise SystemExit(
            f"backend '{args.engine}' is a fixed-shape artifact exported at "
            f"{backend.export_imgsz} px and cannot run at {args.imgsz} px"
        )

    annotations = load_annotations(args.annotations)
    images = resolve_images(annotations, args.images, args.limit)
    print(f"evaluating {len(images)} images from {args.images}")

    # The serving thresholds are a UI trade-off; evaluation needs its own.
    settings = base.model_copy(update={"conf_threshold": args.conf, "iou_threshold": args.iou})
    detector = Detector(backend, args.imgsz, settings)
    detector.load()
    detach_tracker(detector)

    try:
        class_names = model_class_names(detector)
        class_indices = select_class_indices(class_names, requested_classes)
        selected_names = [class_names[i] for i in class_indices]
        category_map = build_category_map(
            class_names, annotations["categories"], required=selected_names
        )
        category_ids = sorted({category_map[i] for i in class_indices})
        print(f"class set: {class_set_label} -> {selected_names}")
        print(f"model index -> COCO category id: {[(i, category_map[i]) for i in class_indices]}")

        detections, dropped = predict_dataset(
            detector, images, class_indices, category_map, args.conf, args.iou
        )
    finally:
        detector.close()

    if detections:
        metrics = evaluate(args.annotations, detections, [i for i, _ in images], category_ids)
    else:
        print("no detections were produced; reporting zeros")
        metrics = empty_metrics()

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "engine": backend.key,
        # Duplicated as "backend" because the promotion gate
        # (ml/quantization/benchmark_precision.py) keys records on that name.
        "backend": backend.key,
        "label": backend.label,
        "device": backend.device,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "max_det": COCO_MAX_DET,
        "images": len(images),
        "annotations": str(args.annotations),
        # The PRD makes this the deciding context: a baseline and a target are
        # only comparable when both were measured on the same class set.
        "class_set": class_set_label,
        "classes": selected_names,
        "category_ids": category_ids,
        "detections": len(detections),
        "detections_dropped_unmapped": dropped,
        "gpu": {"name": gpu.name, "available": gpu.available},
        **metrics,
    }

    out = args.json_out or REPORTS_DIR / f"coco_{backend.key}_{args.imgsz}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\nmAP50-95 {payload['map50_95']:.4f}  mAP50 {payload['map50']:.4f}  "
        f"({class_set_label}, {len(images)} images, {backend.key} @ {args.imgsz}px)"
    )
    if dropped:
        print(f"warning: {dropped} detections had no COCO category and were dropped")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

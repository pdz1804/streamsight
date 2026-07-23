"""CI smoke test for the COCO evaluation path (PRD NFR-8), on CPU only.

``eval_coco.py`` needs a ~1 GB val2017 download and a GPU-shaped runtime to be
worth anything as an accuracy number. Neither is available on a hosted CI
runner, so this script proves something narrower but load-bearing: that the
*pipeline* -- dataset load, inference, xyxy->xywh conversion, class-index ->
COCO-category-id mapping, pycocotools scoring -- runs end to end without
raising and produces a plausible score. Those conversions are exactly where a
COCO harness goes silently wrong (see ``eval_coco``'s module docstring); a
smoke test that only checks "the script exited 0" would not catch a single one
of them, because a broken conversion still exits 0 with mAP 0.0.

Dataset: `COCO8 <https://docs.ultralytics.com/datasets/detect/coco8>`_, an
Ultralytics-hosted 8-image subset of COCO train2017 (~1 MB) with YOLO-format
``.txt`` labels. This module converts the val half (4 images) to a minimal
COCO annotation file so the exact same ``eval_coco`` functions that drive the
real benchmark -- ``build_category_map``, ``detach_tracker``,
``predict_dataset``, ``evaluate`` -- run unmodified against it. The YOLO ->
COCO box conversion is new here (val2017 already ships COCO-format boxes) and
gets its own unit tests in ``test_smoke_coco8.py``.

Assertions are deliberately loose (``0 < mAP <= 1``, detections non-empty):
8 images is not enough to pin an exact number without flaking on every
ultralytics point release, and pinning one would test today's model weights
rather than the pipeline.

Failure modes are meant to be told apart by reading the message: a setup
failure (network down, upstream moved the asset, dataset layout changed) says
"SETUP FAILURE" and is not a code problem. An assertion failure at the bottom
says "SMOKE TEST FAILED" and means the eval pipeline itself regressed. Neither
path skips -- a missing dataset or model is fatal, never a silent pass.

Usage:
    python ml/eval/smoke_coco8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval_coco
sys.path.insert(0, str(REPO_ROOT / "ml" / "data" / "scripts"))  # dataset_integrity, download_coco
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))  # app.*

from app.core.config import get_settings  # noqa: E402
from app.inference.backends import get_backend  # noqa: E402
from app.inference.detector import Detector  # noqa: E402
from dataset_integrity import (  # noqa: E402
    RAW_DIR,
    count_files,
    download_with_resume,
    human_bytes,
    load_manifest,
    pin_or_verify_sha256,
    save_manifest,
    verify_size,
)
from download_coco import extract  # noqa: E402 - reuse the path-traversal-safe unzip
from eval_coco import (  # noqa: E402
    EVAL_CONF_THRESHOLD,
    build_category_map,
    detach_tracker,
    evaluate,
    model_class_names,
    predict_dataset,
)

MANIFEST_NAME = "coco8"

COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
#: Observed 2026-07-23; a mismatch is a warning (see verify_size), not fatal.
COCO8_EXPECTED_BYTES = 443_158
#: Well below the real size -- anything smaller is a truncated transfer or an
#: HTML error page served with a 200.
COCO8_MIN_BYTES = 200_000

YOLO11N_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
YOLO11N_EXPECTED_BYTES = 5_613_764
YOLO11N_MIN_BYTES = 3_000_000

#: The zip's own top-level folder is "coco8/", so extracting into RAW_DIR
#: reproduces the same "one level under raw/" layout the COCO val2017 download
#: uses (see download_coco.py).
ARCHIVE_PATH = RAW_DIR / "coco8.zip"
DATASET_DIR = RAW_DIR / "coco8"
VAL_IMAGE_COUNT = 4

#: COCO protocol NMS threshold, matching eval_coco.py's own default.
SMOKE_IOU = 0.7

#: coco8.yaml's class table (contiguous 0..79, the standard pretrained-YOLO
#: ordering) paired with the real COCO category ids each name maps to
#: (``ultralytics.data.converter.coco80_to_coco91_class()``, copied verbatim
#: rather than imported so this module has no dependency on an internal,
#: not-designed-for-reuse ultralytics helper). Both tables are fixed COCO
#: spec, not something a future ultralytics release changes.
COCO8_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)  # fmt: skip
COCO8_CATEGORY_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)  # fmt: skip


# --------------------------------------------------------------------- assets


def _download_or_die(url: str, target: Path) -> None:
    """Download, turning a network failure into a message that names its own cause."""
    try:
        download_with_resume(url, target)
    except OSError as exc:
        raise SystemExit(
            f"SETUP FAILURE: could not download {url}: {exc}\n"
            "This is a network or upstream-availability problem, not a pipeline regression."
        ) from exc


def ensure_model_weight(path: Path, manifest: dict[str, Any]) -> None:
    """Fetch the pretrained yolo11n weights the smoke test detects with.

    Shares ``path`` with the real ``fp32_cpu`` backend on purpose: a smoke-test
    run on a dev machine that already has the weight (e.g. for ``eval_coco.py``)
    reuses it instead of downloading a second copy.
    """
    if not path.exists():
        print(f"downloading {YOLO11N_URL}")
        _download_or_die(YOLO11N_URL, path)
    else:
        print(f"model weight already present: {path} ({human_bytes(path.stat().st_size)})")

    verify_size(path, expected=YOLO11N_EXPECTED_BYTES, minimum=YOLO11N_MIN_BYTES)
    _, newly_pinned = pin_or_verify_sha256(manifest, "yolo11n.pt", path, source=YOLO11N_URL)
    print(f"  sha256 {'pinned' if newly_pinned else 'matches the pin'}")


def _dataset_ready(dataset_dir: Path) -> bool:
    images = count_files(dataset_dir / "images" / "val", (".jpg",))
    labels = count_files(dataset_dir / "labels" / "val", (".txt",))
    return images == VAL_IMAGE_COUNT and labels == VAL_IMAGE_COUNT


def ensure_coco8(manifest: dict[str, Any]) -> None:
    """Fetch and extract coco8's val split, verifying size + pinned SHA256."""
    if _dataset_ready(DATASET_DIR):
        print(f"coco8 already extracted at {DATASET_DIR}")
        return

    if not ARCHIVE_PATH.exists():
        print(f"downloading {COCO8_URL}")
        _download_or_die(COCO8_URL, ARCHIVE_PATH)
    else:
        size = human_bytes(ARCHIVE_PATH.stat().st_size)
        print(f"archive already present: {ARCHIVE_PATH} ({size})")

    verify_size(ARCHIVE_PATH, expected=COCO8_EXPECTED_BYTES, minimum=COCO8_MIN_BYTES)
    _, newly_pinned = pin_or_verify_sha256(manifest, "coco8.zip", ARCHIVE_PATH, source=COCO8_URL)
    print(f"  sha256 {'pinned' if newly_pinned else 'matches the pin'}")

    extract(ARCHIVE_PATH, RAW_DIR)
    if not _dataset_ready(DATASET_DIR):
        raise SystemExit(
            f"SETUP FAILURE: extracted {ARCHIVE_PATH.name} but did not find {VAL_IMAGE_COUNT} "
            f"val images/labels under {DATASET_DIR} - the archive layout may have changed upstream"
        )


# ------------------------------------------------------------- yolo -> coco gt


def yolo_label_to_coco_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> list[float]:
    """Convert a normalized YOLO box (center x/y, width, height) to COCO pixels.

    YOLO's four numbers are all fractions of the image size, centered on the
    box; COCO wants absolute pixels, corner-anchored. A width/height swap here
    still produces a box of the right *area* in roughly the right place -- it
    does not error, and on a scene with roughly-square objects it can even
    score a nonzero IoU -- which is exactly the "silent, plausible-looking
    wrong number" failure class this whole smoke test exists to catch. See
    ``test_smoke_coco8.py`` for the pinned regression test.
    """
    box_w = w * img_w
    box_h = h * img_h
    x = cx * img_w - box_w / 2
    y = cy * img_h - box_h / 2
    return [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)]


def coco8_categories() -> list[dict[str, Any]]:
    return [
        {"id": category_id, "name": name}
        for category_id, name in zip(COCO8_CATEGORY_IDS, COCO8_NAMES, strict=True)
    ]


def build_ground_truth(dataset_dir: Path) -> tuple[dict[str, Any], list[tuple[int, Path]]]:
    """Read coco8's YOLO-format val split into a minimal COCO annotation dict.

    Returns:
        ``(coco_dict, images)`` where ``images`` is the same ``(image_id, path)``
        pairing ``eval_coco.resolve_images`` produces from a real annotation
        file, so ``predict_dataset`` needs no coco8-specific code path.
    """
    import cv2

    images_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels" / "val"
    image_paths = sorted(images_dir.glob("*.jpg"))
    if len(image_paths) != VAL_IMAGE_COUNT:
        raise SystemExit(
            f"SETUP FAILURE: expected {VAL_IMAGE_COUNT} images in {images_dir}, "
            f"found {len(image_paths)}"
        )

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    pairs: list[tuple[int, Path]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise SystemExit(f"SETUP FAILURE: could not decode {image_path}")
        height, width = frame.shape[:2]
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        pairs.append((image_id, image_path))

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise SystemExit(f"SETUP FAILURE: no label file for {image_path.name}: {label_path}")
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            class_index, cx, cy, w, h = (float(v) for v in line.split())
            bbox = yolo_label_to_coco_bbox(cx, cy, w, h, width, height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": COCO8_CATEGORY_IDS[int(class_index)],
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": "coco8 val split, converted from YOLO labels for the CI smoke test"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": coco8_categories(),
    }
    return coco, pairs


# ------------------------------------------------------------------------ main


def main() -> int:
    settings = get_settings()
    manifest = load_manifest(MANIFEST_NAME)

    print("== ensuring assets (model weight + coco8 dataset) ==")
    ensure_model_weight(settings.weights_dir / "yolo11n.pt", manifest)
    ensure_coco8(manifest)
    save_manifest(MANIFEST_NAME, manifest)

    print("== converting YOLO labels to a minimal COCO annotation file ==")
    coco_gt, images = build_ground_truth(DATASET_DIR)
    annotations_path = DATASET_DIR / "instances_coco8_val.json"
    annotations_path.write_text(json.dumps(coco_gt), encoding="utf-8")
    print(
        f"wrote {annotations_path} "
        f"({len(coco_gt['images'])} images, {len(coco_gt['annotations'])} boxes)"
    )

    print("== running the real detect -> convert -> score pipeline (CPU) ==")
    # fp32_cpu is the always-available, never-fails-to-load backend (see
    # backends.py) and its device is hardcoded to "cpu" -- this cannot touch a
    # GPU regardless of what else is running on the host.
    backend = get_backend("fp32_cpu")
    eval_settings = settings.model_copy(
        update={"conf_threshold": EVAL_CONF_THRESHOLD, "iou_threshold": SMOKE_IOU}
    )
    detector = Detector(backend, settings.default_imgsz, eval_settings)
    detector.load()
    # Same leak eval_coco.py guards against: warmup drives model.track, whose
    # callbacks would otherwise silently replace detections with track output
    # on every later predict() call. See detach_tracker's docstring.
    detach_tracker(detector)

    try:
        class_names = model_class_names(detector)
        class_indices = sorted(class_names)
        category_map = build_category_map(class_names, coco_gt["categories"])
        category_ids = sorted(set(category_map.values()))
        detections, dropped = predict_dataset(
            detector, images, class_indices, category_map, EVAL_CONF_THRESHOLD, SMOKE_IOU
        )
    finally:
        detector.close()

    print(f"{len(detections)} detections, {dropped} dropped (class had no COCO category)")
    if not detections:
        raise AssertionError(
            "SMOKE TEST FAILED: zero detections survived the pipeline. Either the model "
            "produced nothing on 4 real images with a pretrained checkpoint (implausible), "
            "or the class-index -> COCO-category-id mapping dropped everything - this is a "
            "pipeline regression, not a dataset problem."
        )

    image_ids = [image_id for image_id, _ in images]
    metrics = evaluate(annotations_path, detections, image_ids, category_ids)
    print(
        f"mAP50-95 {metrics['map50_95']:.4f}  mAP50 {metrics['map50']:.4f}  "
        f"mAP75 {metrics['map75']:.4f}"
    )

    # Loose bounds only, deliberately: 4 images is too few to pin an exact
    # number without flaking on the next ultralytics point release. mAP50 is
    # NOT checked here even though it is in the report: it is forgiving enough
    # (IoU >= 0.5) that a measured width/height swap in the box conversion
    # still scored 0.86 on this dataset -- it would not have caught the bug
    # this test exists to catch. mAP75 is: the same swap drove it to exactly
    # 0.0, because a pretrained model on 4 easy images should localize tightly
    # enough to clear a 0.75 IoU bar on *something* when the boxes are right.
    for name in ("map50_95", "map75"):
        value = metrics[name]
        if not (0.0 < value <= 1.0):
            raise AssertionError(
                f"SMOKE TEST FAILED: {name}={value!r} is outside (0, 1]. On 4 real images "
                "with a known-good pretrained model this means the eval pipeline itself is "
                "broken (box format, category-id mapping, or image/annotation alignment) - "
                "not model quality."
            )

    print("coco8 smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

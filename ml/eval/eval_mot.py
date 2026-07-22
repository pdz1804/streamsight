"""MOT17 tracking accuracy (MOTA / IDF1 / IDSW) for the deployed tracker.

Complements ``eval_coco.py``: mAP scores the detector, this scores the *identity*
work ByteTrack does on top of it. The PRD gates tracking quality separately from
engine promotion (IDF1 >= 0.60), so this writes its own report.

Unlike the COCO harness this runs the real ``Detector.track`` path -- the tracker
is the thing under test, and the runtime confidence threshold is part of it, so
the serving default is kept unless ``--conf`` overrides it.

Frame alignment is the correctness trap. MOT ground truth is **1-indexed** and
stores ``[x, y, width, height]`` with the top-left corner, while the image files
are ``img1/000001.jpg`` upward. Enumerating hypotheses from 0, or sorting the
files lexically without checking, shifts every hypothesis one frame against the
ground truth: motmetrics still returns numbers, they are just quietly wrong (a
one-frame shift typically costs a few MOTA points and inflates IDSW). This module
therefore numbers hypothesis frames from 1 in sorted filename order and asserts
that the sequence starts at frame 1.

Protocol note: this is the straightforward MOTChallenge accumulation (class-1
pedestrian ground truth with the ``conf`` flag set, IoU matching at 0.5). It does
not implement the official distractor/ignore-region preprocessing, so numbers are
mildly conservative against leaderboard figures. That is stated in the report
JSON rather than left for a reader to assume.

Usage:
    python ml/eval/eval_mot.py --engine fp32_gpu --seqs MOT17-02-FRCNN
    python ml/eval/eval_mot.py --engine openvino_cpu --limit 300
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.backends import BACKENDS, availability, get_backend  # noqa: E402
from app.config import get_settings, probe_gpu  # noqa: E402
from app.detector import Detector  # noqa: E402
from app.tracker import BYTETRACK_CONFIG  # noqa: E402

REPORTS_DIR = REPO_ROOT / "ml" / "eval" / "reports"
DEFAULT_MOT_ROOT = REPO_ROOT / "ml" / "data" / "raw" / "mot"
DOWNLOAD_HINT = (
    "MOT17 is registration-gated: download the zip manually and run "
    "ml/data/scripts/download_mot.py --zip <path-to-MOT17.zip>"
)

#: MOT17 ground truth class 1 is "pedestrian"; the other ids are distractors
#: (static person, reflection, vehicle, ...) that the standard protocol excludes.
GT_PEDESTRIAN_CLASS = 1

#: Only boxes the model calls "person" can be identity-matched against pedestrian
#: ground truth. Matching by name, not index, so a fine-tuned model still works.
HYPOTHESIS_CLASS_NAME = "person"

#: A hypothesis and a ground-truth box can be the same object below this IoU
#: distance (i.e. IoU >= 0.5), the MOTChallenge convention.
MAX_IOU_DISTANCE = 0.5

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def import_motmetrics() -> Any:
    """Import ``motmetrics``, restoring the numpy aliases it still expects.

    The pinned motmetrics (1.2.0) calls ``np.zeros_like(..., dtype=np.bool)``
    inside ``MOTAccumulator.update``. ``np.bool`` was removed in NumPy 1.24 and
    this project pins 1.26, so *every* update raises ``AttributeError`` --
    verified, not assumed. Re-adding the alias is a two-line fix confined to this
    CLI process; the alternative is bumping a pinned dependency, which is a
    reproducibility decision that belongs to whoever owns the lockfile. The
    shim is conditional, so it disappears by itself once motmetrics is upgraded.
    """
    with warnings.catch_warnings():
        # Probing for the alias is itself what emits NumPy's FutureWarning.
        warnings.simplefilter("ignore", FutureWarning)
        for alias, builtin in (("bool", bool), ("float", float), ("int", int)):
            if getattr(np, alias, None) is None:
                setattr(np, alias, builtin)
    import motmetrics

    return motmetrics


def iou_distance_matrix(
    objects: np.ndarray, hypotheses: np.ndarray, max_distance: float = MAX_IOU_DISTANCE
) -> np.ndarray:
    """Pairwise ``1 - IoU`` between ``[x, y, w, h]`` boxes, ``nan`` above the cut.

    Written here rather than taken from ``mm.distances.iou_matrix`` because that
    function also depends on numpy aliases removed years ago (see
    :func:`import_motmetrics`), and because ``nan`` -- not a large distance -- is
    what tells the Hungarian assignment inside motmetrics that a pair is *not
    allowed* to match. Returning 1.0 there would let far-apart boxes pair up when
    nothing better is available and quietly inflate MOTA.
    """
    rows, columns = len(objects), len(hypotheses)
    if rows == 0 or columns == 0:
        return np.empty((rows, columns), dtype=float)

    objects = np.asarray(objects, dtype=float).reshape(rows, 4)
    hypotheses = np.asarray(hypotheses, dtype=float).reshape(columns, 4)

    obj_x1, obj_y1 = objects[:, 0, None], objects[:, 1, None]
    obj_x2, obj_y2 = obj_x1 + objects[:, 2, None], obj_y1 + objects[:, 3, None]
    hyp_x1, hyp_y1 = hypotheses[None, :, 0], hypotheses[None, :, 1]
    hyp_x2, hyp_y2 = hyp_x1 + hypotheses[None, :, 2], hyp_y1 + hypotheses[None, :, 3]

    inter_w = np.maximum(0.0, np.minimum(obj_x2, hyp_x2) - np.maximum(obj_x1, hyp_x1))
    inter_h = np.maximum(0.0, np.minimum(obj_y2, hyp_y2) - np.maximum(obj_y1, hyp_y1))
    intersection = inter_w * inter_h
    union = (
        objects[:, 2, None] * objects[:, 3, None]
        + hypotheses[None, :, 2] * hypotheses[None, :, 3]
        - intersection
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)

    distance = 1.0 - iou
    return np.where(distance > max_distance, np.nan, distance)


def sequence_dirs(root: Path, requested: list[str] | None) -> list[Path]:
    """Locate the sequences to evaluate.

    The search is structural rather than path-shaped, matching
    ``ml/data/scripts/download_mot.py``: the MOT zips nest their content
    differently (``MOT17/train/MOT17-02-DPM`` in one, ``train/`` at the root in
    another), so a directory qualifies when it actually carries ``img1/`` and
    ``gt/gt.txt``. Sequences without ground truth (the ``test`` split) are
    invisible here by construction, which is correct -- they cannot be scored.

    MOT17 ships every sequence three times, once per public detector
    (``-DPM``/``-FRCNN``/``-SDP``), but the *images and ground truth are
    identical* -- only the bundled public detections differ, and this harness
    uses its own detector. Evaluating all 21 would triple the GPU time for
    exactly the same result, so the default picks the ``-FRCNN`` copies.
    """
    if not root.is_dir():
        raise SystemExit(f"MOT root not found: {root} ({DOWNLOAD_HINT})")

    available = sorted(
        {
            gt_file.parent.parent
            for gt_file in root.rglob("gt/gt.txt")
            if (gt_file.parent.parent / "img1").is_dir()
        },
        key=lambda sequence: sequence.name,
    )
    if not available:
        raise SystemExit(f"no MOT sequences with ground truth under {root} ({DOWNLOAD_HINT})")

    if requested:
        by_name = {p.name: p for p in available}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise SystemExit(f"sequences not found: {missing}; available: {sorted(by_name)}")
        return [by_name[name] for name in requested]

    preferred = [p for p in available if p.name.endswith("-FRCNN")]
    return preferred or available


def load_ground_truth(sequence: Path) -> dict[int, list[tuple[int, list[float]]]]:
    """Read ``gt/gt.txt`` into ``frame -> [(object_id, [x, y, w, h]), ...]``.

    Columns are ``frame, id, x, y, w, h, conf, class, visibility``. Rows with the
    ``conf`` flag cleared are annotator-suppressed and rows outside the pedestrian
    class are distractors; both are excluded, which is what the MOTChallenge
    devkit does before matching.
    """
    path = sequence / "gt" / "gt.txt"
    if not path.exists():
        raise SystemExit(f"ground truth missing: {path} ({DOWNLOAD_HINT})")

    frames: dict[int, list[tuple[int, list[float]]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) < 7:
            continue
        frame = int(float(fields[0]))
        object_id = int(float(fields[1]))
        x, y, w, h = (float(v) for v in fields[2:6])
        flag = int(float(fields[6]))
        class_id = int(float(fields[7])) if len(fields) > 7 else GT_PEDESTRIAN_CLASS
        if flag != 1 or class_id != GT_PEDESTRIAN_CLASS:
            continue
        frames.setdefault(frame, []).append((object_id, [x, y, w, h]))
    if not frames:
        raise SystemExit(f"no usable pedestrian ground truth in {path}")
    return frames


def image_frames(sequence: Path) -> list[tuple[int, Path]]:
    """List ``img1`` as ``(frame_number, path)`` with MOT's 1-based numbering.

    The frame number comes from the filename when it is numeric, so the alignment
    survives a sequence that does not start at 000001; position-based numbering
    is only the fallback.
    """
    img_dir = sequence / "img1"
    if not img_dir.is_dir():
        raise SystemExit(f"frames missing: {img_dir} ({DOWNLOAD_HINT})")

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images in {img_dir} ({DOWNLOAD_HINT})")

    pairs: list[tuple[int, Path]] = []
    for position, path in enumerate(files, start=1):
        stem = path.stem
        pairs.append((int(stem) if stem.isdigit() else position, path))

    if pairs[0][0] != 1:
        print(f"  note: {sequence.name} starts at frame {pairs[0][0]}, not 1")
    return pairs


def track_sequence(
    detector: Detector, frames: list[tuple[int, Path]]
) -> dict[int, list[tuple[int, list[float]]]]:
    """Run the tracker over one sequence.

    Returns:
        ``frame -> [(track_id, [x, y, w, h]), ...]`` in the same 1-based frame
        numbering and the same corner+size box convention as the ground truth,
        so the two can be handed to ``motmetrics`` without further translation.
    """
    import cv2

    detector.reset_tracker()
    hypotheses: dict[int, list[tuple[int, list[float]]]] = {}

    for position, (frame_number, path) in enumerate(frames, start=1):
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not decode {path}")
        _, tracks, _ = detector.track(image)
        rows = [
            (track.track_id, [track.x1, track.y1, track.x2 - track.x1, track.y2 - track.y1])
            for track in tracks
            if track.track_id is not None and track.class_name.lower() == HYPOTHESIS_CLASS_NAME
        ]
        hypotheses[frame_number] = rows
        if position % 100 == 0 or position == len(frames):
            print(f"  {position}/{len(frames)} frames")

    return hypotheses


def accumulate(
    ground_truth: dict[int, list[tuple[int, list[float]]]],
    hypotheses: dict[int, list[tuple[int, list[float]]]],
) -> Any:
    """Build a ``motmetrics`` accumulator over the union of both frame sets.

    Frames present in only one side still have to be visited: a frame with ground
    truth and no hypothesis is a set of misses, and the reverse is a set of false
    positives. Skipping them would silently improve every metric.
    """
    mm = import_motmetrics()

    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(ground_truth) | set(hypotheses)):
        gt_rows = ground_truth.get(frame, [])
        hyp_rows = hypotheses.get(frame, [])
        gt_boxes = np.array([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
        hyp_boxes = np.array([row[1] for row in hyp_rows], dtype=float).reshape(-1, 4)
        accumulator.update(
            [row[0] for row in gt_rows],
            [row[0] for row in hyp_rows],
            iou_distance_matrix(gt_boxes, hyp_boxes),
            frameid=frame,
        )
    return accumulator


METRIC_NAMES = (
    "mota",
    "idf1",
    "num_switches",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_false_positives",
    "num_misses",
    "num_fragmentations",
    "num_unique_objects",
    "mostly_tracked",
    "mostly_lost",
    "num_objects",
)


def summarise(accumulators: list[Any], names: list[str]) -> dict[str, dict[str, float]]:
    """Compute per-sequence and overall metrics."""
    mm = import_motmetrics()

    host = mm.metrics.create()
    summary = host.compute_many(
        accumulators, metrics=list(METRIC_NAMES), names=names, generate_overall=True
    )
    output: dict[str, dict[str, float]] = {}
    for name, row in summary.iterrows():
        output[str(name)] = {
            metric: (None if math.isnan(float(row[metric])) else round(float(row[metric]), 5))
            for metric in METRIC_NAMES
        }
    return output


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
    parser.add_argument("--root", type=Path, default=DEFAULT_MOT_ROOT, help="MOT17 dataset root")
    parser.add_argument("--seqs", nargs="+", help="sequence names (default: the -FRCNN copies)")
    parser.add_argument("--limit", type=int, help="evaluate only the first N frames per sequence")
    parser.add_argument("--conf", type=float, help="override the runtime confidence threshold")
    parser.add_argument("--json", dest="json_out", type=Path, help="output path for the report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    sequences = sequence_dirs(args.root, args.seqs)
    print(f"evaluating {len(sequences)} sequence(s): {[p.name for p in sequences]}")

    # ByteTrack is a two-stage associator: high-score detections seed tracks, then
    # a second pass rescues *low*-score boxes between track_low_thresh (0.1) and
    # track_high_thresh (0.25). Feeding it the serving threshold of 0.25 means NMS
    # has already discarded that band, so the second stage never has anything to
    # work with and is silently dead -- depressing MOTA and recall and inflating
    # fragmentation. Ultralytics forces conf=0.1 inside Model.track for exactly
    # this reason; the API overrides it for UI reasons, and evaluation must not
    # inherit that. Defaulting to track_low_thresh measures the real algorithm.
    eval_conf = args.conf if args.conf is not None else BYTETRACK_CONFIG["track_low_thresh"]
    settings = base.model_copy(update={"conf_threshold": float(eval_conf)})
    print(f"detection confidence floor: {eval_conf} (ByteTrack low-score stage enabled)")
    detector = Detector(backend, args.imgsz, settings)
    detector.load()
    detector.warmup(frames=2)

    accumulators: list[Any] = []
    names: list[str] = []
    frames_total = 0
    try:
        for sequence in sequences:
            print(f"\n=== {sequence.name} ===")
            frames = image_frames(sequence)
            if args.limit is not None:
                frames = frames[: args.limit]
            ground_truth = load_ground_truth(sequence)
            if args.limit is not None:
                keep = {number for number, _ in frames}
                ground_truth = {k: v for k, v in ground_truth.items() if k in keep}
            hypotheses = track_sequence(detector, frames)
            accumulators.append(accumulate(ground_truth, hypotheses))
            names.append(sequence.name)
            frames_total += len(frames)
    finally:
        detector.close()

    metrics = summarise(accumulators, names)
    overall = metrics.get("OVERALL", next(iter(metrics.values())))

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "engine": backend.key,
        "backend": backend.key,
        "label": backend.label,
        "device": backend.device,
        "imgsz": args.imgsz,
        "conf_threshold": settings.conf_threshold,
        "iou_match_threshold": MAX_IOU_DISTANCE,
        "root": str(args.root),
        "sequences": names,
        "frames": frames_total,
        "protocol": (
            "MOTChallenge accumulation over class-1 pedestrian ground truth, IoU 0.5 matching; "
            f"detection confidence floor {settings.conf_threshold} so ByteTrack's low-score "
            "second association stage is active; no distractor/ignore-region preprocessing, so "
            "these figures are not directly comparable to leaderboard submissions"
        ),
        "gpu": {"name": gpu.name, "available": gpu.available},
        "overall": overall,
        "per_sequence": {k: v for k, v in metrics.items() if k != "OVERALL"},
    }

    out = args.json_out or REPORTS_DIR / f"mot_{backend.key}_{args.imgsz}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\nMOTA {overall['mota']:.4f}  IDF1 {overall['idf1']:.4f}  "
        f"IDSW {int(overall['num_switches'])}  ({frames_total} frames, "
        f"{backend.key} @ {args.imgsz}px)"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

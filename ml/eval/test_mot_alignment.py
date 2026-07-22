"""Guards for the two ways a MOT harness silently reports the wrong number.

`eval_coco.py` got conversion tests because its failure modes (box format,
category ids) are famous. `eval_mot.py` has its own pair and had none:

1. **Frame alignment.** MOT ground truth is 1-based. Enumerating image files
   from 0, or trusting lexical order without checking, shifts every hypothesis
   one frame against its ground truth. Nothing errors; MOTA just drops a few
   points and IDSW inflates.
2. **Distance matrix conventions.** motmetrics expects `nan` for "these two
   cannot be matched", not a large finite distance. A large number is still a
   candidate the Hungarian solver may pick, which fabricates matches between
   boxes that do not overlap at all.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mot import (
    GT_PEDESTRIAN_CLASS,
    MAX_IOU_DISTANCE,
    iou_distance_matrix,
    load_ground_truth,
)

# --------------------------------------------------------------- distances


def test_identical_boxes_have_zero_distance() -> None:
    matrix = iou_distance_matrix([[10.0, 10.0, 20.0, 20.0]], [[10.0, 10.0, 20.0, 20.0]])
    assert matrix[0][0] == pytest.approx(0.0, abs=1e-6)


def test_non_overlapping_boxes_are_nan_not_a_large_number() -> None:
    """`nan` means 'not a candidate'. A large finite cost is still selectable."""
    matrix = iou_distance_matrix([[0.0, 0.0, 10.0, 10.0]], [[500.0, 500.0, 10.0, 10.0]])
    assert math.isnan(matrix[0][0]), (
        "non-overlapping pairs must be nan; a finite distance lets the solver "
        "match boxes that do not overlap"
    )


def test_overlap_below_the_iou_threshold_is_rejected() -> None:
    """The cut is at IoU 0.5, i.e. distance 0.5. Just past it must not match."""
    # Two 10x10 boxes offset by 8px overlap on 2/10 of each axis -> IoU ~0.02.
    matrix = iou_distance_matrix([[0.0, 0.0, 10.0, 10.0]], [[8.0, 0.0, 10.0, 10.0]])
    assert math.isnan(matrix[0][0])


def test_strong_overlap_is_kept_as_a_candidate() -> None:
    matrix = iou_distance_matrix([[0.0, 0.0, 10.0, 10.0]], [[1.0, 0.0, 10.0, 10.0]])
    value = matrix[0][0]
    assert not math.isnan(value)
    assert 0.0 < value <= MAX_IOU_DISTANCE


def test_empty_inputs_produce_an_empty_matrix() -> None:
    assert len(iou_distance_matrix([], [[0.0, 0.0, 1.0, 1.0]])) == 0


# ---------------------------------------------------------- ground truth


def _write_gt(tmp_path: Path, rows: list[str]) -> Path:
    sequence = tmp_path / "MOT17-02-FRCNN"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return sequence


def test_ground_truth_frames_are_one_based(tmp_path: Path) -> None:
    """The first MOT frame is 1. Reading it as 0 shifts the whole sequence."""
    sequence = _write_gt(
        tmp_path,
        [
            f"1,1,10,20,30,40,1,{GT_PEDESTRIAN_CLASS},1.0",
            f"2,1,11,21,30,40,1,{GT_PEDESTRIAN_CLASS},1.0",
        ],
    )
    frames = load_ground_truth(sequence)
    assert min(frames) == 1
    assert 0 not in frames


def test_ground_truth_keeps_xywh_not_corners(tmp_path: Path) -> None:
    sequence = _write_gt(tmp_path, [f"1,7,10,20,30,40,1,{GT_PEDESTRIAN_CLASS},1.0"])
    (object_id, box) = load_ground_truth(sequence)[1][0]
    assert object_id == 7
    assert box == [10.0, 20.0, 30.0, 40.0], "columns are x,y,width,height"


def test_non_pedestrian_classes_are_excluded(tmp_path: Path) -> None:
    """MOT17 gt contains vehicles and distractors; scoring them inflates FP."""
    sequence = _write_gt(
        tmp_path,
        [
            f"1,1,10,20,30,40,1,{GT_PEDESTRIAN_CLASS},1.0",
            "1,2,50,60,30,40,1,3,1.0",
        ],
    )
    assert [oid for oid, _ in load_ground_truth(sequence)[1]] == [1]


def test_zero_confidence_rows_are_excluded(tmp_path: Path) -> None:
    """conf=0 marks an ignored box in the MOT format."""
    sequence = _write_gt(
        tmp_path,
        [
            f"1,1,10,20,30,40,1,{GT_PEDESTRIAN_CLASS},1.0",
            f"1,2,50,60,30,40,0,{GT_PEDESTRIAN_CLASS},1.0",
        ],
    )
    assert [oid for oid, _ in load_ground_truth(sequence)[1]] == [1]

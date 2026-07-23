"""Guard on smoke_coco8's own conversion: normalized YOLO box -> pixel COCO bbox.

Same failure class ``eval_coco``'s ``xyxy_to_coco_bbox`` tests guard against: a
swapped or mis-scaled box still produces *a* number instead of an error, so
pinning the conversion here is what catches a broken smoke test rather than a
broken model. Pure-function tests: no model, no dataset, no network.
"""

from __future__ import annotations

import pytest
from smoke_coco8 import yolo_label_to_coco_bbox


def test_centered_box_converts_to_corner_and_size():
    # 100x100 image, box centered at (50, 50) sized 40 wide x 20 tall.
    assert yolo_label_to_coco_bbox(0.5, 0.5, 0.4, 0.2, 100, 100) == [30.0, 40.0, 40.0, 20.0]


def test_width_and_height_are_not_swapped():
    # Width comes from the third YOLO value against image width, height from
    # the fourth against image height. A swap would report 160/40 instead.
    box = yolo_label_to_coco_bbox(0.5, 0.5, 0.8, 0.2, 100, 200)
    assert box[2] == 80.0
    assert box[3] == 40.0


def test_full_image_box_matches_image_bounds():
    assert yolo_label_to_coco_bbox(0.5, 0.5, 1.0, 1.0, 640, 480) == [0.0, 0.0, 640.0, 480.0]


def test_corner_box_near_origin():
    assert yolo_label_to_coco_bbox(0.05, 0.05, 0.1, 0.1, 200, 200) == [0.0, 0.0, 20.0, 20.0]


def test_bbox_accepts_numpy_scalars():
    numpy = pytest.importorskip("numpy")
    row = numpy.array([0.5, 0.5, 0.4, 0.2], dtype=numpy.float32)
    box = yolo_label_to_coco_bbox(*row, 100, 100)
    assert box == [30.0, 40.0, 40.0, 20.0]

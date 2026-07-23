"""Guards on the two conversions that fail silently.

A COCO harness that gets either the box format or the category ids wrong still
runs to completion and still prints a number -- it just prints ~0.0 mAP, or worse,
a plausible-looking wrong one. These tests pin both conversions so a refactor
cannot reintroduce that failure mode. They are pure-function tests: no model, no
dataset, no GPU.
"""

from __future__ import annotations

import pytest
from eval_coco import (
    PRD_CLASS_SUBSET,
    build_category_map,
    resolve_class_request,
    select_class_indices,
    xyxy_to_coco_bbox,
)

#: A faithful excerpt of ``instances_val2017.json``'s category list. The gaps are
#: the point: ids 12, 26, 29... do not exist, so the ids are not the model's
#: indices shifted by one.
COCO_CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 13, "name": "stop sign"},
    {"id": 14, "name": "parking meter"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
]

#: The first entries of the pretrained YOLO11n table, contiguous from 0.
YOLO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
}


def test_xyxy_becomes_corner_plus_size():
    assert xyxy_to_coco_bbox(10.0, 20.0, 110.0, 220.0) == [10.0, 20.0, 100.0, 200.0]


def test_width_and_height_are_not_the_far_corner():
    box = xyxy_to_coco_bbox(300.0, 400.0, 350.0, 480.0)
    assert box[2] == 50.0
    assert box[3] == 80.0


def test_bbox_accepts_numpy_scalars():
    numpy = pytest.importorskip("numpy")
    row = numpy.array([1.5, 2.5, 4.0, 6.5], dtype=numpy.float32)
    assert xyxy_to_coco_bbox(*row) == [1.5, 2.5, 2.5, 4.0]


def test_zero_area_box_survives_conversion():
    assert xyxy_to_coco_bbox(5.0, 5.0, 5.0, 5.0) == [5.0, 5.0, 0.0, 0.0]


def test_category_ids_are_not_the_model_indices():
    mapping = build_category_map(YOLO_NAMES, COCO_CATEGORIES)
    # The off-by-one that "works" for the first eleven classes and then does not.
    assert mapping[0] == 1
    assert mapping[7] == 8
    assert mapping[11] == 13
    assert mapping[13] == 15
    assert mapping[11] != 12


def test_every_prd_class_maps():
    mapping = build_category_map(YOLO_NAMES, COCO_CATEGORIES, required=PRD_CLASS_SUBSET)
    indices = select_class_indices(YOLO_NAMES, PRD_CLASS_SUBSET)
    assert sorted(mapping[i] for i in indices) == [1, 2, 3, 4, 6, 8]


def test_fine_tuned_six_class_model_maps_to_the_same_categories():
    """The deployed model renumbers its classes 0..5; the ids must not follow."""
    fine_tuned = dict(enumerate(PRD_CLASS_SUBSET))
    mapping = build_category_map(fine_tuned, COCO_CATEGORIES, required=PRD_CLASS_SUBSET)
    assert mapping == {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 8}


def test_required_class_without_a_category_is_an_error():
    with pytest.raises(ValueError, match="no matching COCO category"):
        build_category_map({0: "person", 1: "forklift"}, COCO_CATEGORIES, required=["forklift"])


def test_unrequested_unmappable_class_is_omitted_not_mismapped():
    mapping = build_category_map({0: "person", 1: "forklift"}, COCO_CATEGORIES, required=["person"])
    assert mapping == {0: 1}


def test_names_match_case_and_underscore_insensitively():
    mapping = build_category_map({0: "Traffic_Light"}, COCO_CATEGORIES, required=["traffic light"])
    assert mapping == {0: 10}


def test_select_class_indices_defaults_to_every_class():
    assert select_class_indices(YOLO_NAMES, None) == sorted(YOLO_NAMES)


def test_select_class_indices_rejects_unknown_names():
    with pytest.raises(ValueError, match="does not have these classes"):
        select_class_indices(YOLO_NAMES, ["unicorn"])


def test_prd6_shorthand_expands_and_is_labelled():
    classes, label = resolve_class_request(["prd6"])
    assert classes == list(PRD_CLASS_SUBSET)
    assert label == "prd6"


def test_no_class_request_is_labelled_model_all():
    assert resolve_class_request(None) == (None, "model-all")

"""Result parsing and ByteTrack configuration."""

from __future__ import annotations

import pytest
from app.vision.tracker import BYTETRACK_CONFIG, ensure_tracker_config, parse_results
from fakes import FakeBoxes, FakeResults


def test_empty_results_yield_nothing() -> None:
    detections, tracks = parse_results(FakeResults(None))
    assert detections == []
    assert tracks == []


def test_boxes_without_ids_are_detections_only() -> None:
    """A box ByteTrack has not confirmed is still a detection, but not a track."""
    result = FakeResults(FakeBoxes(xyxy=[[10, 20, 110, 220]], conf=[0.9], cls=[0]))
    detections, tracks = parse_results(result)

    assert len(detections) == 1
    assert tracks == []
    assert detections[0].class_name == "person"
    assert (detections[0].x1, detections[0].y1) == (10.0, 20.0)


def test_ids_produce_tracks_alongside_detections() -> None:
    result = FakeResults(
        FakeBoxes(
            xyxy=[[0, 0, 50, 50], [60, 60, 120, 120]],
            conf=[0.8, 0.7],
            cls=[0, 2],
            ids=[7, 12],
        )
    )
    detections, tracks = parse_results(result)

    assert len(detections) == 2
    assert [t.track_id for t in tracks] == [7, 12]
    assert [t.class_name for t in tracks] == ["person", "car"]


def test_unknown_class_id_falls_back_to_its_number() -> None:
    result = FakeResults(FakeBoxes(xyxy=[[0, 0, 5, 5]], conf=[0.5], cls=[99]), names={0: "person"})
    detections, _ = parse_results(result)
    assert detections[0].class_name == "99"


def test_tracker_config_is_written_from_the_python_source(tmp_path) -> None:
    """The YAML is generated, so documented values cannot drift from used values."""
    target = tmp_path / "nested" / "bytetrack.yaml"
    written = ensure_tracker_config(target)

    assert written.exists()
    body = written.read_text(encoding="utf-8")
    assert "tracker_type: bytetrack" in body
    for key, value in BYTETRACK_CONFIG.items():
        assert key in body
        if isinstance(value, bool):
            assert f"{key}: {str(value).lower()}" in body


def test_existing_tracker_config_is_not_overwritten(tmp_path) -> None:
    target = tmp_path / "bytetrack.yaml"
    target.write_text("custom: true\n", encoding="utf-8")
    ensure_tracker_config(target)
    assert target.read_text(encoding="utf-8") == "custom: true\n"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("track_high_thresh", 0.25),
        ("track_low_thresh", 0.1),
        ("new_track_thresh", 0.25),
        ("track_buffer", 30),
        ("match_thresh", 0.8),
    ],
)
def test_bytetrack_hyperparameters_match_the_plan(key: str, expected: float) -> None:
    assert BYTETRACK_CONFIG[key] == expected

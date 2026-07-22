"""Telemetry aggregation and threaded video capture."""

from __future__ import annotations

import time

import numpy as np
import pytest
from app.capture import FrameSource, classify_source
from app.config import Settings
from app.exceptions import SourceUnavailableError
from app.metrics import UNIQUE_TRACK_CAP, WINDOW, MetricsCollector

# --------------------------------------------------------------------- metrics


def test_snapshot_is_safe_before_any_frame(settings: Settings) -> None:
    snapshot = MetricsCollector(settings).snapshot("fp32_cpu", 640)
    assert snapshot.fps == 0.0
    assert snapshot.frames_processed == 0
    assert snapshot.avg_latency_ms == 0.0


def test_latency_percentiles_reflect_recorded_frames(settings: Settings) -> None:
    collector = MetricsCollector(settings)
    for latency in range(1, 101):
        collector.record_frame(float(latency), [])

    snapshot = collector.snapshot("fp32_cpu", 640)
    assert snapshot.frames_processed == 100
    assert 40 <= snapshot.p50_latency_ms <= 60
    assert snapshot.p95_latency_ms >= snapshot.p50_latency_ms


def test_fps_is_derived_from_arrival_times_not_latency(settings: Settings) -> None:
    """FPS must include queueing and encode cost, so it is measured on the clock."""
    collector = MetricsCollector(settings)
    for _ in range(5):
        collector.record_frame(1.0, [])
        time.sleep(0.02)

    snapshot = collector.snapshot("fp32_cpu", 640)
    # Latency claims 1000 FPS; wall-clock spacing says roughly 50.
    assert 10 < snapshot.fps < 200


def test_unique_track_ids_are_counted_once(settings: Settings) -> None:
    collector = MetricsCollector(settings)
    for _ in range(10):
        collector.record_frame(5.0, [1, 2, 3])
    collector.record_frame(5.0, [4])

    snapshot = collector.snapshot("fp32_cpu", 640)
    assert snapshot.unique_tracks == 4
    assert snapshot.track_count == 1


def test_none_track_ids_are_ignored(settings: Settings) -> None:
    collector = MetricsCollector(settings)
    collector.record_frame(5.0, [None, None, 1])
    assert collector.snapshot("fp32_cpu", 640).unique_tracks == 1


def test_rolling_buffers_stay_bounded(settings: Settings) -> None:
    """A multi-hour soak must not grow the collector without limit."""
    collector = MetricsCollector(settings)
    for index in range(WINDOW * 3):
        collector.record_frame(float(index), [index])

    assert len(collector._latencies) == WINDOW
    assert len(collector._frame_times) == WINDOW
    assert len(collector._unique_tracks) <= UNIQUE_TRACK_CAP


def test_degraded_flag_and_reason_travel_together(settings: Settings) -> None:
    collector = MetricsCollector(settings)
    collector.set_degraded(True, "out of VRAM")
    assert collector.snapshot("fp32_cpu", 480).degrade_reason == "out of VRAM"

    collector.set_degraded(False)
    assert collector.snapshot("fp32_cpu", 640).degrade_reason is None


def test_reset_stream_keeps_lifetime_frame_count(settings: Settings) -> None:
    collector = MetricsCollector(settings)
    for _ in range(20):
        collector.record_frame(4.0, [1])
    collector.reset_stream()

    snapshot = collector.snapshot("fp32_cpu", 640)
    assert snapshot.frames_processed == 20
    assert snapshot.fps == 0.0


# --------------------------------------------------------------------- capture


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0", "webcam"),
        ("1", "webcam"),
        ("rtsp://camera.local/stream", "rtsp"),
        ("http://host/feed.mjpg", "rtsp"),
        (r"D:\videos\clip.mp4", "file"),
        ("clip.mp4", "file"),
    ],
)
def test_source_classification(spec: str, expected: str) -> None:
    assert classify_source(spec) == expected


def test_missing_file_fails_before_opening_a_capture(tmp_path) -> None:
    source = FrameSource(str(tmp_path / "nope.mp4"))
    with pytest.raises(SourceUnavailableError, match="not found"):
        source.open()


def test_file_source_streams_frames(settings: Settings) -> None:
    clip = settings.assets_dir / "sample.mp4"
    if not clip.exists():
        pytest.skip("sample clip missing - run ml/scripts/fetch_assets.py")

    with FrameSource(str(clip), ring_size=8, loop=False) as source:
        frames = []
        deadline = time.perf_counter() + 10.0
        while len(frames) < 5 and time.perf_counter() < deadline:
            frame = source.read(timeout=2.0)
            if frame is not None:
                frames.append(frame)
            elif source.finished:
                break

        assert len(frames) == 5
        assert all(isinstance(f, np.ndarray) and f.ndim == 3 for f in frames)
        assert source.width > 0 and source.height > 0
        assert 1.0 <= source.source_fps <= 240.0


def test_buffer_never_exceeds_its_ring_size(settings: Settings) -> None:
    """Drop-oldest keeps memory flat when the consumer is slower than the source."""
    clip = settings.assets_dir / "sample.mp4"
    if not clip.exists():
        pytest.skip("sample clip missing - run ml/scripts/fetch_assets.py")

    source = FrameSource(str(clip), ring_size=4, loop=True, pace_files=False)
    source.open()
    try:
        time.sleep(1.0)  # let the producer outrun the (idle) consumer
        assert len(source._buffer) <= 4
        assert source.produced_frames > 4
    finally:
        source.close()

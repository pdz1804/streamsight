"""The degradation ladder.

This path is a headline reliability claim and it was completely untested. A code
review found that exhausting the ladder left the runtime with no detector at
all, which 503s every route -- including the one an operator would use to
recover -- with no way back short of a process restart. These tests pin the
invariant that broke.

A fake detector is used rather than the real model: the behaviour under test is
the runtime's ladder-walking and recovery logic, and loading a real model on
every rung would make the suite minutes long while testing Ultralytics instead.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from app.backends import BACKENDS
from app.config import Settings
from app.detector import OutOfVramError
from app.exceptions import NoBackendError
from app.runtime import InferenceRuntime


class FakeDetector:
    """Loads and runs successfully, unless told otherwise."""

    #: Backend keys that should raise on load, simulating an unusable artifact.
    unloadable: ClassVar[set[str]] = set()
    #: Backend keys that should raise OOM during warmup.
    oom_on_warmup: ClassVar[set[str]] = set()
    instances: ClassVar[list[FakeDetector]] = []

    def __init__(self, backend, imgsz, settings) -> None:
        self.backend = backend
        self.imgsz = imgsz
        self.closed = False
        FakeDetector.instances.append(self)

    def load(self) -> None:
        if self.backend.key in FakeDetector.unloadable:
            raise RuntimeError(f"{self.backend.key} cannot load here")

    def warmup(self, frames: int = 2) -> None:
        if self.backend.key in FakeDetector.oom_on_warmup:
            raise OutOfVramError("CUDA out of memory")

    def close(self) -> None:
        self.closed = True

    def reset_tracker(self) -> None:
        pass

    @property
    def device(self) -> str:
        return self.backend.device

    @property
    def model_file(self) -> str:
        return self.backend.artifact

    def track(self, frame):
        return [], [], 1.0


@pytest.fixture
def runtime(tmp_path, monkeypatch) -> InferenceRuntime:
    """A runtime over stub artifacts, with every CPU backend available."""
    FakeDetector.unloadable = set()
    FakeDetector.oom_on_warmup = set()
    FakeDetector.instances = []

    settings = Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        assets_dir=tmp_path / "assets",
    )
    # Presence is all `availability()` checks; contents are never read here.
    for backend in BACKENDS.values():
        path = backend.path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        if backend.key == "openvino_cpu":
            path.mkdir(exist_ok=True)
        elif not path.exists():
            path.write_bytes(b"stub artifact")

    monkeypatch.setattr("app.runtime.Detector", FakeDetector)
    instance = InferenceRuntime(settings)
    # Force the CPU-only ladder so the test does not depend on the host GPU.
    monkeypatch.setattr("app.runtime.probe_gpu", lambda: instance._gpu)
    instance.startup()
    return instance


def test_startup_selects_a_backend(runtime: InferenceRuntime) -> None:
    assert runtime.ready
    assert runtime.imgsz == 640


def test_repeated_degradation_never_leaves_the_service_without_a_detector(
    runtime: InferenceRuntime,
) -> None:
    """The regression this file exists for.

    Previously the third drill emptied the fallback chain, skipped the rescue,
    and left `_detector = None` permanently.
    """
    for step in range(12):
        runtime.simulate_oom()
        assert runtime.ready, f"runtime lost its detector after degradation step {step + 1}"
        assert runtime.precision != "none"


def test_degradation_still_serves_frames_after_the_ladder_bottoms_out(
    runtime: InferenceRuntime, blank_frame
) -> None:
    for _ in range(10):
        runtime.simulate_oom()
    # The whole point of staying loaded: requests keep working.
    detections, tracks, timing, frame_id = runtime.process(blank_frame)
    assert frame_id > 0
    assert timing.total_ms >= 0


def test_first_degradation_step_reduces_resolution(runtime: InferenceRuntime) -> None:
    """Resolution is the first lever, being cheaper than swapping backends.

    The backend may still change on this step: an artifact exported at 640 px
    cannot serve 480 px, so a shape-fixed backend is necessarily replaced by one
    that accepts the new size. Whatever ends up loaded must actually support it.
    """
    runtime.simulate_oom()
    assert runtime.imgsz == 480
    assert BACKENDS[runtime.precision].supports_imgsz(480)


def test_degradation_sets_a_reason_the_ui_can_show(runtime: InferenceRuntime) -> None:
    runtime.simulate_oom()
    assert runtime.metrics.degraded is True
    assert runtime.metrics.degrade_reason


def test_config_remains_readable_after_the_ladder_is_exhausted(
    runtime: InferenceRuntime,
) -> None:
    """An operator must still be able to inspect and fix the service."""
    for _ in range(10):
        runtime.simulate_oom()
    config = runtime.config_response()
    assert config.precision != "none"
    assert config.available_backends


def test_switch_recovers_the_service_after_degradation(runtime: InferenceRuntime) -> None:
    for _ in range(10):
        runtime.simulate_oom()
    restored = runtime.switch("fp32_cpu", 640)
    assert restored.precision == "fp32_cpu"
    assert restored.imgsz == 640
    assert runtime.ready


def test_transient_oom_does_not_permanently_blacklist_a_backend(
    runtime: InferenceRuntime,
) -> None:
    """OOM describes the machine at an instant, not the backend's capability."""
    FakeDetector.oom_on_warmup = {"fp32_cpu"}
    runtime._load_first_working([BACKENDS["fp32_cpu"]], 320)
    assert ("fp32_cpu", 320) not in runtime._unusable

    # A genuine capability failure is remembered.
    FakeDetector.oom_on_warmup = set()
    FakeDetector.unloadable = {"fp32_cpu"}
    runtime._load_first_working([BACKENDS["fp32_cpu"]], 320)
    assert ("fp32_cpu", 320) in runtime._unusable


def test_startup_fails_loudly_when_nothing_can_run(tmp_path, monkeypatch) -> None:
    """An empty model tree is a startup error, not a silent half-working service."""
    monkeypatch.setattr("app.runtime.Detector", FakeDetector)
    settings = Settings(models_dir=tmp_path / "empty", data_dir=tmp_path / "d")
    with pytest.raises(NoBackendError):
        InferenceRuntime(settings).startup()


def test_shape_fixed_backends_are_skipped_at_other_resolutions(
    runtime: InferenceRuntime,
) -> None:
    """A 640 px export cannot serve 480 px, and must not be offered for it."""
    chain = [BACKENDS["openvino_cpu"]]
    assert runtime._load_first_working(chain, 480) is None
    assert runtime._load_first_working(chain, 640) is not None

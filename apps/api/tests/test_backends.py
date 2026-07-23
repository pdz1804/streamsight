"""Backend registry, availability rules, and the degradation ladder."""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.inference.backends import (
    BACKENDS,
    FALLBACK_LADDER,
    availability,
    candidate_chain,
    get_backend,
)


@pytest.fixture
def isolated_settings(tmp_path) -> Settings:
    """Settings pointed at an empty model tree, so nothing is available yet."""
    return Settings(models_dir=tmp_path / "models")


def test_every_ladder_entry_is_a_registered_backend() -> None:
    assert set(FALLBACK_LADDER) <= set(BACKENDS)


def test_ladder_ends_on_the_always_available_cpu_path() -> None:
    """The last rung must never require a GPU or an export step."""
    last = get_backend(FALLBACK_LADDER[-1])
    assert last.requires_gpu is False
    assert last.exported is False


def test_gpu_backend_is_unavailable_without_a_gpu(isolated_settings: Settings) -> None:
    runnable, reason = availability(get_backend("int8_trt"), isolated_settings, gpu_available=False)
    assert runnable is False
    assert "GPU" in reason


def test_missing_artifact_is_reported_with_a_next_step(isolated_settings: Settings) -> None:
    runnable, reason = availability(get_backend("fp32_cpu"), isolated_settings, gpu_available=False)
    assert runnable is False
    assert "yolo11n.pt" in reason


def test_present_artifact_becomes_available(isolated_settings: Settings) -> None:
    weights = isolated_settings.weights_dir / "yolo11n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"not a real model, only presence is checked here")

    runnable, reason = availability(get_backend("fp32_cpu"), isolated_settings, gpu_available=False)
    assert runnable is True
    assert reason == ""


def test_chain_is_empty_when_nothing_can_run(isolated_settings: Settings) -> None:
    assert candidate_chain(None, isolated_settings, gpu_available=False) == []


def test_preferred_backend_leads_the_chain(isolated_settings: Settings) -> None:
    weights = isolated_settings.weights_dir / "yolo11n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"stub")

    chain = candidate_chain("fp32_cpu", isolated_settings, gpu_available=False)
    assert next(b.key for b in chain) == "fp32_cpu"


def test_chain_never_repeats_a_backend(isolated_settings: Settings) -> None:
    weights = isolated_settings.weights_dir / "yolo11n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"stub")

    keys = [b.key for b in candidate_chain("fp32_cpu", isolated_settings, gpu_available=False)]
    assert len(keys) == len(set(keys))


def test_auto_preference_is_not_treated_as_a_backend_name(isolated_settings: Settings) -> None:
    weights = isolated_settings.weights_dir / "yolo11n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"stub")

    chain = candidate_chain("auto", isolated_settings, gpu_available=False)
    assert [b.key for b in chain] == ["fp32_cpu"]

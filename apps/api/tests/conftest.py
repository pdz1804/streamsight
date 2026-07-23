"""Shared fixtures.

Tests are split into two tiers: pure-logic tests that need no model, and a
smaller set that boots the real runtime. The heavy fixtures are session-scoped so
the model loads at most once per run, and they skip cleanly when no inference
artifact is present -- a fresh clone should be able to run the fast tests before
downloading any weights.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.core.config import Settings, get_settings, probe_gpu  # noqa: E402
from app.inference.backends import candidate_chain  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def has_backend(settings: Settings) -> bool:
    """True when at least one inference artifact exists on this machine."""
    return bool(candidate_chain(None, settings, probe_gpu().available))


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A plain BGR frame. Content does not matter for shape and plumbing tests."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def sample_frame(settings: Settings) -> np.ndarray:
    """A real frame from the bundled clip, so detections are actually non-empty."""
    import cv2

    clip = settings.assets_dir / "sample.mp4"
    if not clip.exists():
        pytest.skip("sample clip missing - run ml/scripts/fetch_assets.py")
    capture = cv2.VideoCapture(str(clip))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip("could not read a frame from the sample clip")
    return frame


@pytest.fixture(scope="session")
def client(has_backend: bool) -> Iterator[object]:
    """TestClient with the real lifespan, so the model is genuinely loaded."""
    if not has_backend:
        pytest.skip("no inference artifact available - run ml/scripts/fetch_assets.py")
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client

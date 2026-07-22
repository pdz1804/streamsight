"""Source resolution and upload handling.

These paths take untrusted input and reach the filesystem and the network, so
they are tested adversarially rather than only on the happy path.
"""

from __future__ import annotations

import io

import pytest
from app.config import Settings
from app.exceptions import InvalidFrameError, SourceUnavailableError
from app.models import SourceInfo
from app.sources import SAMPLE_ID, WEBCAM_ID, SourceRegistry


@pytest.fixture
def registry(tmp_path) -> SourceRegistry:
    settings = Settings(data_dir=tmp_path / "data", assets_dir=tmp_path / "assets")
    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    return SourceRegistry(settings)


def _upload(
    registry: SourceRegistry, name: str, payload: bytes, limit: int = 1024 * 1024
) -> SourceInfo:
    return registry.save_upload(name, io.BytesIO(payload), limit)


# ------------------------------------------------------------------- traversal


@pytest.mark.parametrize(
    "hostile",
    [
        "../../assets/calibration",
        "..\\..\\assets\\sample",
        "../../../../windows/system32/config",
        "a/../../b",
        "*",
        "?" * 12,
        "....//....//etc/passwd",
    ],
)
def test_resolve_rejects_path_traversal(registry: SourceRegistry, hostile: str) -> None:
    """`Path.glob` expands `..`, so ids must be validated, not just globbed."""
    with pytest.raises(SourceUnavailableError):
        registry.resolve(hostile)


def test_resolve_rejects_glob_metacharacters(registry: SourceRegistry) -> None:
    """A wildcard id must not match a real upload."""
    created = _upload(registry, "clip.mp4", b"x" * 2048)
    assert registry.resolve(created.id)  # the real id still works
    with pytest.raises(SourceUnavailableError):
        registry.resolve("*" * 12)


def test_resolved_upload_stays_inside_the_uploads_directory(registry: SourceRegistry) -> None:
    created = _upload(registry, "clip.mp4", b"x" * 2048)
    resolved = registry.resolve(created.id)
    uploads = (registry._settings.data_dir / "uploads").resolve()
    assert str(resolved).startswith(str(uploads))


# ------------------------------------------------------------------ url schemes


@pytest.mark.parametrize(
    "hostile",
    [
        "file:///C:/Windows/win.ini",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
        "https://example.com/secret",
        "gopher://internal/",
    ],
)
def test_resolve_rejects_non_stream_schemes(registry: SourceRegistry, hostile: str) -> None:
    """Otherwise the server reads host files or cloud metadata and renders it back."""
    with pytest.raises(SourceUnavailableError, match="scheme"):
        registry.resolve(hostile)


@pytest.mark.parametrize("url", ["rtsp://camera.local/stream", "rtmp://host/live"])
def test_resolve_accepts_real_stream_schemes(registry: SourceRegistry, url: str) -> None:
    assert registry.resolve(url) == url


# ---------------------------------------------------------------------- basics


def test_webcam_resolves_to_a_device_index(registry: SourceRegistry) -> None:
    assert registry.resolve(WEBCAM_ID) == "0"


def test_missing_sample_clip_is_reported(registry: SourceRegistry) -> None:
    with pytest.raises(SourceUnavailableError, match="sample"):
        registry.resolve(SAMPLE_ID)


# --------------------------------------------------------------------- uploads


def test_upload_rejects_a_non_video_extension(registry: SourceRegistry) -> None:
    with pytest.raises(InvalidFrameError, match="unsupported video type"):
        _upload(registry, "payload.exe", b"MZ" * 100)


def test_oversized_upload_is_rejected_and_leaves_nothing_behind(
    registry: SourceRegistry,
) -> None:
    """The cap must be enforced while copying, not after the disk is consumed."""
    uploads = registry._settings.data_dir / "uploads"
    with pytest.raises(InvalidFrameError, match="limit"):
        _upload(registry, "huge.mp4", b"x" * 5000, limit=1024)
    assert list(uploads.iterdir()) == []


def test_empty_upload_is_rejected_and_leaves_nothing_behind(registry: SourceRegistry) -> None:
    uploads = registry._settings.data_dir / "uploads"
    with pytest.raises(InvalidFrameError, match="empty"):
        _upload(registry, "empty.mp4", b"")
    assert list(uploads.iterdir()) == []


def test_upload_appears_in_the_catalogue_with_its_original_name(
    registry: SourceRegistry,
) -> None:
    created = _upload(registry, "My Holiday Clip.mp4", b"x" * 4096)
    listed = {s.id: s for s in registry.list()}
    assert created.id in listed
    assert listed[created.id].label == "My Holiday Clip.mp4"


def test_name_sidecars_are_not_offered_as_sources(registry: SourceRegistry) -> None:
    _upload(registry, "clip.mp4", b"x" * 2048)
    kinds = {s.id for s in registry.list()}
    assert not any(k.endswith(".name") for k in kinds)

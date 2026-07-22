"""Video source catalogue.

Tracks what the viewer can stream from: the bundled sample clip, any uploaded
files, and the local webcam. Uploads are stored under ``data/uploads`` with
generated names -- the client never gets to influence a filesystem path, and the
opaque id it receives is what gets resolved back to disk here.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from .config import Settings
from .exceptions import InvalidFrameError, SourceUnavailableError
from .models import SourceInfo

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".webm"}
)

SAMPLE_ID = "sample"
WEBCAM_ID = "webcam"


class SourceRegistry:
    """Resolves source ids to concrete OpenCV specs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._uploads_dir = settings.data_dir / "uploads"
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._sample_path = settings.assets_dir / "sample.mp4"

    @property
    def sample_path(self) -> Path:
        return self._sample_path

    def list(self) -> list[SourceInfo]:
        """Every currently selectable source, newest upload first."""
        items: list[SourceInfo] = []
        if self._sample_path.exists():
            items.append(
                SourceInfo(
                    id=SAMPLE_ID,
                    kind="sample",
                    label="Bundled sample clip",
                    detail=self._sample_path.name,
                )
            )
        items.append(
            SourceInfo(
                id=WEBCAM_ID,
                kind="webcam",
                label="Local webcam",
                detail="device 0",
            )
        )
        uploads = sorted(
            (p for p in self._uploads_dir.iterdir() if _is_video(p)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in uploads:
            items.append(
                SourceInfo(
                    id=path.stem,
                    kind="file",
                    label=_display_name(path),
                    detail=f"{path.stat().st_size / 1024**2:.1f} MB",
                )
            )
        return items

    def resolve(self, source: str) -> str:
        """Turn a source id or raw spec into something OpenCV can open.

        Raises:
            SourceUnavailableError: the id is unknown or the file has vanished.
        """
        spec = source.strip()
        if not spec:
            spec = SAMPLE_ID
        if spec == SAMPLE_ID:
            if not self._sample_path.exists():
                raise SourceUnavailableError(
                    "no bundled sample clip - run scripts/fetch_sample_video.py"
                )
            return str(self._sample_path)
        if spec == WEBCAM_ID:
            return "0"
        if spec.isdigit() or "://" in spec:
            return spec

        matches = [p for p in self._uploads_dir.glob(f"{spec}.*") if _is_video(p)]
        if matches:
            return str(matches[0])
        raise SourceUnavailableError(f"unknown source '{source}'")

    def save_upload(self, filename: str, stream, max_bytes: int) -> SourceInfo:  # noqa: ANN001
        """Persist an uploaded video and return its catalogue entry.

        Raises:
            InvalidFrameError: unsupported extension or the file exceeds the cap.
        """
        suffix = Path(filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise InvalidFrameError(
                f"unsupported video type '{suffix or filename}'; allowed: "
                + ", ".join(sorted(ALLOWED_SUFFIXES))
            )
        source_id = uuid.uuid4().hex[:12]
        target = self._uploads_dir / f"{source_id}{suffix}"
        with target.open("wb") as handle:
            shutil.copyfileobj(stream, handle, length=1024 * 1024)
        size = target.stat().st_size
        if size == 0 or size > max_bytes:
            target.unlink(missing_ok=True)
            raise InvalidFrameError(
                f"upload rejected: {size} bytes (limit {max_bytes} bytes, must be non-empty)"
            )
        # Keep the original name for display without letting it touch the path.
        (self._uploads_dir / f"{source_id}.name").write_text(Path(filename).name, encoding="utf-8")
        logger.info("stored upload %s (%s, %d bytes)", source_id, filename, size)
        return SourceInfo(
            id=source_id,
            kind="file",
            label=Path(filename).name,
            detail=f"{size / 1024**2:.1f} MB",
        )


def _is_video(path: Path) -> bool:
    """Filter out the ``.name`` sidecars that live alongside uploads."""
    return path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES


def _display_name(path: Path) -> str:
    """Prefer the uploader's original filename when we recorded it."""
    sidecar = path.with_suffix(".name")
    if sidecar.exists():
        try:
            return sidecar.read_text(encoding="utf-8").strip() or path.name
        except OSError:
            return path.name
    return path.name

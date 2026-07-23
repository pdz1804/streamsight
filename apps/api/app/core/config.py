"""Runtime settings and GPU capability probing.

Settings are environment-overridable (prefix ``STREAMSIGHT_``) so the same image
runs on a 4 GB laptop, a CPU-only CI box, or a bigger dev machine without code
changes. ``probe_gpu`` is the single place that decides whether this host can
carry the 640 px pipeline or must be capped to 480 px.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Repo root, by parent depth from this file:
#   apps/api/app/core/config.py -> core -> app -> apps/api -> apps -> <root>
#
# The depth is load-bearing -- weights, the demo clip, the SQLite log and .env
# are all resolved from it, and moving this module changes the count silently.
# tests/test_repo_layout.py asserts the result still looks like the repo root so
# that a move fails a test rather than a deployment.
REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Application configuration.

    Every field can be overridden by an environment variable with the
    ``STREAMSIGHT_`` prefix, e.g. ``STREAMSIGHT_DEFAULT_IMGSZ=480``.
    """

    model_config = SettingsConfigDict(
        env_prefix="STREAMSIGHT_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "StreamSight"
    api_port: int = 8100

    # --- artifact locations -------------------------------------------------
    models_dir: Path = REPO_ROOT / "ml" / "models"
    data_dir: Path = REPO_ROOT / "apps" / "api" / "data"
    assets_dir: Path = REPO_ROOT / "apps" / "api" / "assets"

    # --- inference defaults -------------------------------------------------
    default_precision: str = "auto"
    default_imgsz: int = 640
    degraded_imgsz: int = 480
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    # --- VRAM policy --------------------------------------------------------
    # Free VRAM measured immediately after the model + runtime workspace load.
    # Below this floor there is not enough headroom for 640 px frame buffers,
    # so the pipeline starts at `degraded_imgsz` instead of crashing later.
    vram_free_floor_mb: int = 1200
    vram_budget_mb: int = 3500

    # --- streaming ----------------------------------------------------------
    ring_buffer_size: int = 30
    jpeg_quality: int = 80
    #: Cap applied to decoded frames on the capture thread. The detector
    #: letterboxes to `default_imgsz` anyway, so anything wider is cost without
    #: accuracy. Set to 0 to disable and process frames at source resolution.
    capture_max_width: int = 1280
    #: Second guard on what actually reaches the browser. Normally redundant with
    #: `capture_max_width`, and the one that matters for sources already narrower
    #: than the capture cap.
    stream_max_width: int = 1280
    target_fps: int = 30
    gpu_poll_interval_frames: int = 100
    gc_interval_frames: int = 1000
    capture_open_timeout_s: float = 10.0

    # --- api ----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3100", "http://127.0.0.1:3100"]
    max_upload_bytes: int = 32 * 1024 * 1024

    # --- mlflow registry (optional, FR-16's closing clause) -----------------
    # Empty by default: this is the single switch `app.registry` checks before
    # doing anything mlflow-related, so an unconfigured (or CI) process never
    # imports mlflow and never makes a network call.
    mlflow_tracking_uri: str = ""
    #: Registry model to consult once a tracking URI is set, e.g.
    #: "streamsight-detector" (see `ml/quantization/benchmark_precision.py`).
    mlflow_model_name: str = ""
    #: Stage to resolve; empty defaults to "Production" inside the resolver.
    mlflow_stage: str = ""

    @property
    def weights_dir(self) -> Path:
        return self.models_dir / "weights"

    @property
    def engines_dir(self) -> Path:
        return self.models_dir / "engines"

    @property
    def model_config_dir(self) -> Path:
        return self.models_dir / "config"

    @property
    def tracker_config_path(self) -> Path:
        return self.model_config_dir / "bytetrack.yaml"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "stream.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


@dataclass(frozen=True)
class GpuProbe:
    """Result of interrogating the local NVIDIA device.

    ``free_mb`` is meaningful only when ``available`` is true; on CPU-only hosts
    every numeric field is zero and callers must take the CPU path.
    """

    available: bool
    name: str = "cpu"
    total_mb: int = 0
    free_mb: int = 0
    used_mb: int = 0

    @property
    def summary(self) -> str:
        if not self.available:
            return "no NVIDIA GPU detected - CPU inference path"
        return f"{self.name} ({self.total_mb} MiB total, {self.free_mb} MiB free)"


def probe_gpu() -> GpuProbe:
    """Read NVIDIA memory via NVML, falling back to ``nvidia-smi``.

    Never raises: a host without an NVIDIA driver is a supported configuration,
    not an error.
    """
    probe = _probe_via_nvml()
    if probe is not None:
        return probe
    probe = _probe_via_smi()
    if probe is not None:
        return probe
    return GpuProbe(available=False)


def _probe_via_nvml() -> GpuProbe | None:
    try:
        import pynvml
    except ImportError:  # pragma: no cover - dependency always installed in prod
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - NVML raises driver-specific errors
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        return GpuProbe(
            available=True,
            name=name,
            total_mb=int(info.total // 1024**2),
            free_mb=int(info.free // 1024**2),
            used_mb=int(info.used // 1024**2),
        )
    except Exception:  # noqa: BLE001
        return None
    finally:
        # Shutdown failures are irrelevant: the reading is already taken, and
        # NVML is re-initialised on the next probe regardless.
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _probe_via_smi() -> GpuProbe | None:
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                exe,
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 4:
        return None
    try:
        return GpuProbe(
            available=True,
            name=parts[0],
            total_mb=int(float(parts[1])),
            free_mb=int(float(parts[2])),
            used_mb=int(float(parts[3])),
        )
    except ValueError:
        return None


def resolve_start_imgsz(settings: Settings, free_mb_after_load: int, gpu_available: bool) -> int:
    """Choose the starting inference resolution from post-load VRAM headroom.

    On CPU there is no VRAM ceiling to respect, so the full 640 px path is used.
    """
    if not gpu_available:
        return settings.default_imgsz
    if free_mb_after_load < settings.vram_free_floor_mb:
        logger.warning(
            "only %d MiB free after model load (floor %d MiB) - capping to %d px",
            free_mb_after_load,
            settings.vram_free_floor_mb,
            settings.degraded_imgsz,
        )
        return settings.degraded_imgsz
    return settings.default_imgsz

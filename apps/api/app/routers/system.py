"""Health, metrics, model configuration, and source management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..dependencies import RegistryDep, RuntimeDep, SettingsDep
from ..models import (
    GpuInfo,
    HealthResponse,
    MetricsResponse,
    ModelConfigRequest,
    ModelConfigResponse,
    SourceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness + active configuration")
def health(request: Request) -> HealthResponse:
    """Always answers, even before the model is ready, so probes can tell them apart."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        gpu = GpuInfo(available=False, name="unknown", total_mb=0, used_mb=0, free_mb=0)
        return HealthResponse(
            status="ok", app="StreamSight", version=__version__, gpu=gpu, precision="none", imgsz=0
        )
    return HealthResponse(
        status="ok",
        app="StreamSight",
        version=__version__,
        gpu=runtime.metrics.gpu_info(),
        precision=runtime.precision,
        imgsz=runtime.imgsz,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Live telemetry snapshot")
def metrics(runtime: RuntimeDep) -> MetricsResponse:
    return runtime.metrics_response()


@router.get("/config/model", response_model=ModelConfigResponse, summary="Active model config")
def get_model_config(runtime: RuntimeDep) -> ModelConfigResponse:
    return runtime.config_response()


@router.post("/config/model", response_model=ModelConfigResponse, summary="Hot-swap the model")
def set_model_config(payload: ModelConfigRequest, runtime: RuntimeDep) -> ModelConfigResponse:
    """Swap precision and/or resolution without restarting the process.

    Track identities are reset by the swap: ids from the previous model are not
    comparable to ids from the new one.
    """
    return runtime.switch(payload.precision, payload.imgsz)


@router.post(
    "/config/degrade",
    response_model=ModelConfigResponse,
    summary="Trigger one degradation step (reliability drill)",
)
def force_degrade(runtime: RuntimeDep) -> ModelConfigResponse:
    """Exercise the auto-degrade ladder on demand.

    The fallback path is a reliability claim; this endpoint makes it observable
    without waiting for a real out-of-memory event.
    """
    runtime.simulate_oom()
    return runtime.config_response()


@router.get("/sources", response_model=list[SourceInfo], summary="Selectable video sources")
def list_sources(registry: RegistryDep) -> list[SourceInfo]:
    return registry.list()


@router.post("/sources/upload", response_model=SourceInfo, summary="Upload a video file")
async def upload_source(
    registry: RegistryDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
) -> SourceInfo:
    """Store an uploaded video.

    The copy runs in a worker thread: writing tens of megabytes to disk from the
    event loop would stall every live stream for the duration of the upload.
    """
    return await run_in_threadpool(
        registry.save_upload, file.filename or "", file.file, settings.max_upload_bytes
    )

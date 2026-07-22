"""FastAPI application factory.

The model is loaded once in the lifespan handler rather than per request: a
TensorRT engine takes seconds to build its execution context, and every request
paying that cost would make the FPS numbers meaningless.

Startup is deliberately tolerant -- if no backend can load, the app still boots
and reports the problem through ``/health`` and a 503 on inference routes,
because a service that refuses to start is much harder to diagnose than one that
explains itself.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .exceptions import StreamSightError
from .routers import detect, system
from .runtime import InferenceRuntime
from .sources import SourceRegistry

logger = logging.getLogger(__name__)

DESCRIPTION = """
Real-time object detection and multi-object tracking on a 4 GB laptop GPU.

* `POST /detect/frame` - single image, returns detections + track ids
* `WS /detect/stream` - annotated live stream from file, webcam, or RTSP
* `GET /metrics` - FPS, latency percentiles, VRAM, track counts
* `POST /config/model` - hot-swap precision / resolution without a restart
"""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ultralytics is chatty per-frame at INFO even with verbose=False on some paths.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    app.state.settings = settings
    app.state.registry = SourceRegistry(settings)
    runtime = InferenceRuntime(settings)
    app.state.runtime = runtime
    try:
        runtime.startup()
    except Exception as exc:  # noqa: BLE001 - boot anyway so /health can explain
        logger.error("inference runtime failed to start: %s", exc)
        app.state.startup_error = str(exc)
    else:
        app.state.startup_error = None
    try:
        yield
    finally:
        runtime.shutdown()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StreamSightError)
    async def handle_domain_error(_: Request, exc: StreamSightError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "detail": exc.message},
        )

    app.include_router(system.router)
    app.include_router(detect.router)
    return app


app = create_app()

"""Single-frame and streaming detection endpoints."""

from __future__ import annotations

import contextlib
import logging
from typing import Literal

from fastapi import APIRouter, File, Query, UploadFile, WebSocket
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..dependencies import RegistryWsDep, RuntimeDep, RuntimeWsDep, SettingsDep
from ..models import FrameResponse
from ..preprocess import decode_base64_frame, decode_image_bytes
from ..streaming import StreamSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["detect"])


class FrameRequest(BaseModel):
    """Base64 image payload; accepts a bare string or a ``data:`` URI."""

    image: str = Field(min_length=8, description="Base64-encoded JPEG/PNG, data URI optional")


@router.post("/detect/frame", response_model=FrameResponse, summary="Detect + track one frame")
def detect_frame(payload: FrameRequest, runtime: RuntimeDep) -> FrameResponse:
    frame = decode_base64_frame(payload.image)
    return _run(frame, runtime)


@router.post(
    "/detect/image",
    response_model=FrameResponse,
    summary="Detect + track one uploaded image file",
)
async def detect_image(runtime: RuntimeDep, file: UploadFile = File(...)) -> FrameResponse:
    """Decode and run one uploaded image.

    Inference is pushed to a worker thread. This handler has to be ``async`` to
    await the upload, which means its body runs on the event loop -- and a
    synchronous inference call there would stall every active WebSocket send and
    metrics poll for its full duration. (``/detect/frame`` is a plain ``def``, so
    Starlette already runs it in a threadpool.)
    """
    raw = await file.read()
    frame = decode_image_bytes(raw)
    return await run_in_threadpool(_run, frame, runtime)


def _run(frame, runtime) -> FrameResponse:  # noqa: ANN001 - numpy array / runtime
    detections, tracks, timing, frame_id = runtime.process(frame)
    height, width = frame.shape[:2]
    snapshot = runtime.metrics_response()
    return FrameResponse(
        frame_id=frame_id,
        width=width,
        height=height,
        detections=detections,
        tracks=tracks,
        timing=timing,
        fps=snapshot.fps,
        precision=runtime.precision,
        imgsz=runtime.imgsz,
        degraded_mode=snapshot.degraded_mode,
    )


@router.websocket("/detect/stream")
async def detect_stream(
    websocket: WebSocket,
    runtime: RuntimeWsDep,
    registry: RegistryWsDep,
    settings: SettingsDep,
    source: str = Query("sample", description="Source id, device index, or rtsp:// URL"),
    loop: bool = Query(True, description="Restart file sources when they end"),
    encoding: Literal["binary", "base64"] = Query(
        "binary",
        description="Wire format: length-prefixed binary frames, or base64 data URIs in JSON",
    ),
) -> None:
    """Stream annotated frames for the requested source until the client leaves.

    ``encoding=base64`` keeps the original all-JSON transport for clients that
    cannot read binary messages. It costs a third more bytes plus an encode and
    a decode, so it is not the default.
    """
    await websocket.accept()
    try:
        spec = registry.resolve(source)
    except Exception as exc:  # noqa: BLE001 - report to the client, never crash the socket
        await websocket.send_json({"kind": "status", "phase": "error", "message": str(exc)})
        await websocket.close()
        return

    session = StreamSession(websocket, runtime, settings, spec, loop_source=loop, encoding=encoding)
    try:
        await session.run()
    finally:
        # The client may already have closed the socket; that is the normal way
        # a stream ends, not an error.
        with contextlib.suppress(RuntimeError):
            await websocket.close()

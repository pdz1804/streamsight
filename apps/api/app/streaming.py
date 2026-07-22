"""WebSocket streaming session.

Bridges a :class:`~app.capture.FrameSource` to a websocket client. The blocking
parts -- frame decode, inference, JPEG encode -- run in worker threads via
``asyncio.to_thread`` so the event loop stays free to notice client disconnects
promptly; otherwise a stalled send would be discovered only after the next
inference finished.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import cv2
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from .annotate import draw_tracks
from .capture import FrameSource
from .config import Settings
from .exceptions import SourceUnavailableError, StreamSightError
from .models import StreamFrame, StreamStatus
from .preprocess import encode_jpeg_data_uri
from .runtime import InferenceRuntime

logger = logging.getLogger(__name__)


def _fit_for_transport(frame, max_width: int):  # noqa: ANN001, ANN202 - numpy array
    """Downscale wide frames before encoding.

    Detection runs on the full-resolution frame, but shipping that resolution to
    a browser that displays it in a ~1000 px pane is pure cost: a 1080p JPEG is
    several times more expensive to encode, base64, and decode than a 1280 px
    one, and none of those pixels reach the viewer. Frames narrower than the cap
    are passed through untouched.
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA
    )


class StreamSession:
    """One client streaming one source."""

    def __init__(
        self,
        websocket: WebSocket,
        runtime: InferenceRuntime,
        settings: Settings,
        source_spec: str,
        *,
        loop_source: bool = True,
        annotate: bool = True,
    ) -> None:
        self._ws = websocket
        self._runtime = runtime
        self._settings = settings
        self._spec = source_spec
        self._loop_source = loop_source
        self._annotate = annotate
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Open the source and pump annotated frames until the client leaves."""
        await self._send_status("opening", f"opening {self._spec}", source=self._spec)

        source = FrameSource(
            self._spec,
            ring_size=self._settings.ring_buffer_size,
            loop=self._loop_source,
            max_width=self._settings.capture_max_width or None,
        )
        try:
            await asyncio.to_thread(source.open)
        except SourceUnavailableError as exc:
            await self._send_status("error", exc.message, source=self._spec)
            return
        except Exception as exc:
            logger.exception("failed to open source %s", self._spec)
            await self._send_status("error", f"could not open source: {exc}", source=self._spec)
            return

        self._runtime.new_session()
        await self._send_status(
            "streaming",
            f"{source.kind} {source.width}x{source.height} @ {source.source_fps:.0f} fps",
            source=self._spec,
            total_frames=source.total_frames,
        )

        listener = asyncio.create_task(self._listen_for_client())
        try:
            await self._pump(source)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener
            await asyncio.to_thread(source.close)

        if self._ws.client_state is WebSocketState.CONNECTED:
            await self._send_status("ended", "stream finished", source=self._spec)

    # ------------------------------------------------------------------ pump

    async def _pump(self, source: FrameSource) -> None:
        while not self._stop.is_set():
            frame = await asyncio.to_thread(source.read, 1.0)
            if frame is None:
                if source.finished:
                    return
                continue
            try:
                payload = await asyncio.to_thread(self._process, frame)
            except StreamSightError as exc:
                await self._send_status("error", exc.message, source=self._spec)
                return
            except Exception as exc:
                logger.exception("frame processing failed")
                await self._send_status("error", str(exc), source=self._spec)
                return
            try:
                await self._ws.send_text(payload.model_dump_json())
            except (WebSocketDisconnect, RuntimeError):
                return

    def _process(self, frame) -> StreamFrame:  # noqa: ANN001 - numpy array
        """Blocking pipeline stage: detect, track, annotate, downscale, encode."""
        detections, tracks, timing, frame_id = self._runtime.process(frame)
        canvas = draw_tracks(frame, tracks) if self._annotate else frame
        encode_started = time.perf_counter()
        image = encode_jpeg_data_uri(
            _fit_for_transport(canvas, self._settings.stream_max_width),
            self._settings.jpeg_quality,
        )
        timing.encode_ms = round((time.perf_counter() - encode_started) * 1000.0, 2)
        timing.total_ms = round(timing.total_ms + timing.encode_ms, 2)
        height, width = frame.shape[:2]
        return StreamFrame(
            frame_id=frame_id,
            image=image,
            width=width,
            height=height,
            tracks=tracks,
            timing=timing,
            fps=self._runtime.metrics.current_fps(),
            server_ts=time.time() * 1000.0,
            precision=self._runtime.precision,
            imgsz=self._runtime.imgsz,
            degraded_mode=self._runtime.metrics.degraded,
        )

    # --------------------------------------------------------------- control

    async def _listen_for_client(self) -> None:
        """Watch for a client stop command or disconnect."""
        try:
            while True:
                message = await self._ws.receive_text()
                if "stop" in message:
                    self._stop.set()
                    return
        except (WebSocketDisconnect, RuntimeError):
            self._stop.set()

    async def _send_status(
        self,
        phase: str,
        message: str,
        *,
        source: str = "",
        total_frames: int | None = None,
    ) -> None:
        if self._ws.client_state is not WebSocketState.CONNECTED:
            return
        status = StreamStatus(
            phase=phase,  # type: ignore[arg-type]
            message=message,
            source=source,
            total_frames=total_frames,
        )
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await self._ws.send_text(status.model_dump_json())

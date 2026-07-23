"""WebSocket streaming session.

Bridges a :class:`~app.capture.FrameSource` to a websocket client.

The session runs as two cooperating tasks rather than one loop, because a single
``read -> infer -> encode -> send`` loop serializes stages that have no reason to
wait for each other: inference for frame N+1 cannot start until frame N is on the
wire, so the frame period is the *sum* of every stage instead of the slowest one.
Splitting it puts capture+inference on one task and encode+send on another, with
a depth-1 queue between them, so the two overlap.

The queue is depth-1 and the producer **waits** for the consumer to drain it.
That coupling is deliberate and load-bearing: without it a client whose socket
has stalled leaves the producer running capture, inference, tracking and
annotation at full rate forever, burning the GPU and writing detections for
frames nobody will ever see. Freshness is not this queue's job -- the capture
ring buffer upstream already drops the oldest frame and bounds how far behind
live the stream can fall.

Only the consumer task ever writes to the socket. Starlette's WebSocket is not
safe to send from two tasks at once, so a producer failure travels through the
queue as an error sentinel instead of being reported directly.

The blocking parts -- frame decode, inference, JPEG encode -- run in worker
threads via ``asyncio.to_thread`` so the event loop stays free to notice client
disconnects promptly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import cv2
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from .annotate import draw_tracks
from .capture import FrameSource
from .config import Settings
from .exceptions import SourceUnavailableError, StreamSightError
from .models import FrameTiming, StreamFrame, StreamStatus, Track
from .preprocess import encode_jpeg, encode_jpeg_data_uri
from .runtime import InferenceRuntime
from .wire import encode_stream_frame_raw

logger = logging.getLogger(__name__)

StreamEncoding = Literal["binary", "base64"]


def _fit_for_transport(frame, max_width: int):  # noqa: ANN001, ANN202 - numpy array
    """Downscale wide frames before encoding.

    Detection runs on the full-resolution frame, but shipping that resolution to
    a browser that displays it in a ~1000 px pane is pure cost: a 1080p JPEG is
    several times more expensive to encode and decode than a 1280 px one, and
    none of those pixels reach the viewer. Frames narrower than the cap are
    passed through untouched.
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA
    )


@dataclass(slots=True)
class _Rendered:
    """An annotated frame waiting to be encoded and sent."""

    canvas: object  # numpy array
    frame_id: int
    width: int
    height: int
    tracks: list[Track]
    timing: FrameTiming
    fps: float
    precision: str
    imgsz: int
    degraded_mode: bool


@dataclass(slots=True)
class _EndOfStream:
    """Producer sentinel. ``message`` is set only when the producer failed."""

    message: str = ""
    fatal: bool = field(default=False)


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
        encoding: StreamEncoding = "binary",
    ) -> None:
        self._ws = websocket
        self._runtime = runtime
        self._settings = settings
        self._spec = source_spec
        self._loop_source = loop_source
        self._annotate = annotate
        self._encoding: StreamEncoding = encoding
        self._stop = asyncio.Event()
        self._last_send_ms = 0.0

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
        """Run the producer and consumer until the stream or the client ends."""
        queue: asyncio.Queue[_Rendered | _EndOfStream] = asyncio.Queue(maxsize=1)
        producer = asyncio.create_task(self._produce(source, queue), name="stream-produce")
        consumer = asyncio.create_task(self._consume(queue), name="stream-consume")
        try:
            # The consumer owns termination: it stops on the sentinel, on a
            # client disconnect, or on a send failure. Awaiting it (not both)
            # means a client that walks away is noticed without waiting for the
            # producer to finish whatever frame it is holding.
            await consumer
        finally:
            self._stop.set()
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    async def _produce(
        self, source: FrameSource, queue: asyncio.Queue[_Rendered | _EndOfStream]
    ) -> None:
        """Capture, detect, track and annotate; hand results to the consumer."""
        end = _EndOfStream()
        try:
            while not self._stop.is_set():
                waited = time.perf_counter()
                frame = await asyncio.to_thread(source.read, 1.0)
                wait_ms = round((time.perf_counter() - waited) * 1000.0, 2)
                if frame is None:
                    if source.finished:
                        return
                    continue
                try:
                    rendered = await asyncio.to_thread(self._render, frame, wait_ms)
                except StreamSightError as exc:
                    end = _EndOfStream(exc.message, fatal=True)
                    return
                except Exception as exc:
                    logger.exception("frame processing failed")
                    end = _EndOfStream(str(exc), fatal=True)
                    return
                # Blocking, not drop-oldest: this await is what paces inference
                # to what the client can actually take.
                await queue.put(rendered)
        finally:
            # The consumer blocks on this queue, so the sentinel has to arrive on
            # every exit path -- including cancellation -- or the session hangs.
            self._force_sentinel(queue, end)

    def _force_sentinel(
        self, queue: asyncio.Queue[_Rendered | _EndOfStream], end: _EndOfStream
    ) -> None:
        """Enqueue *end* without awaiting, displacing a pending frame if needed.

        Everywhere else the producer awaits the queue, but the shutdown path
        cannot: it runs from a ``finally`` that may already be unwinding a
        cancellation, where awaiting is not guaranteed to resume. Displacing one
        undelivered frame is the price, and it is the right way round -- the
        alternative is a consumer parked on ``get()`` forever.
        """
        try:
            queue.put_nowait(end)
            return
        except asyncio.QueueFull:
            pass
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(end)

    async def _consume(self, queue: asyncio.Queue[_Rendered | _EndOfStream]) -> None:
        """Encode and send whatever the producer has most recently finished."""
        while True:
            item = await queue.get()
            if isinstance(item, _EndOfStream):
                if item.fatal:
                    await self._send_status("error", item.message, source=self._spec)
                return
            try:
                payload, jpeg = await asyncio.to_thread(self._encode, item)
            except Exception as exc:
                logger.exception("frame encoding failed")
                await self._send_status("error", str(exc), source=self._spec)
                return
            started = time.perf_counter()
            try:
                if self._encoding == "binary":
                    header = payload.model_dump_json(exclude={"image"}).encode("utf-8")
                    await self._ws.send_bytes(encode_stream_frame_raw(header, jpeg))
                else:
                    await self._ws.send_text(payload.model_dump_json())
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception as exc:
                # Serialization can fail on its own -- an oversized header, a
                # value Pydantic cannot encode -- and that is not a disconnect.
                # Without this the session would die silently, leaving the
                # client with no status message explaining why.
                logger.exception("frame send failed")
                await self._send_status("error", str(exc), source=self._spec)
                return
            self._last_send_ms = round((time.perf_counter() - started) * 1000.0, 2)

    # -------------------------------------------------------------- stages

    def _render(self, frame, wait_ms: float) -> _Rendered:  # noqa: ANN001 - numpy array
        """Blocking stage one: detect, track, annotate."""
        detections, tracks, timing, frame_id = self._runtime.process(frame)
        timing.wait_ms = wait_ms
        canvas = draw_tracks(frame, tracks) if self._annotate else frame
        height, width = frame.shape[:2]
        return _Rendered(
            canvas=canvas,
            frame_id=frame_id,
            width=width,
            height=height,
            tracks=tracks,
            timing=timing,
            fps=self._runtime.metrics.current_fps(),
            precision=self._runtime.precision,
            imgsz=self._runtime.imgsz,
            degraded_mode=self._runtime.metrics.degraded,
        )

    def _encode(self, item: _Rendered) -> tuple[StreamFrame, bytes]:
        """Blocking stage two: downscale for transport and JPEG-encode.

        Returns the payload plus the raw JPEG. In base64 mode the bytes are
        folded into ``image`` and the second element goes unused; in binary mode
        ``image`` stays ``None`` and the bytes ride after the header.
        """
        # The downscale is timed as part of encoding, as it always has been:
        # it exists only to make the encode cheaper, so charging it elsewhere
        # would flatter this stage and make the two figures incomparable across
        # versions.
        started = time.perf_counter()
        fitted = _fit_for_transport(item.canvas, self._settings.stream_max_width)
        if self._encoding == "binary":
            jpeg = encode_jpeg(fitted, self._settings.jpeg_quality)
            image: str | None = None
        else:
            jpeg = b""
            image = encode_jpeg_data_uri(fitted, self._settings.jpeg_quality)
        timing = item.timing
        timing.encode_ms = round((time.perf_counter() - started) * 1000.0, 2)
        # A frame cannot report how long it took to send -- that is only known
        # once it has already been serialized. This carries the previous frame's
        # send cost instead, which is what makes the stage visible at all.
        timing.send_ms = self._last_send_ms
        timing.total_ms = round(timing.total_ms + timing.encode_ms, 2)
        payload = StreamFrame(
            frame_id=item.frame_id,
            image=image,
            width=item.width,
            height=item.height,
            tracks=item.tracks,
            timing=timing,
            fps=item.fps,
            server_ts=time.time() * 1000.0,
            precision=item.precision,
            imgsz=item.imgsz,
            degraded_mode=item.degraded_mode,
        )
        return payload, jpeg

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

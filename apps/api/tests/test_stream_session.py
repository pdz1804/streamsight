"""Streaming session lifecycle: delivery, termination, and backpressure.

The session is two concurrent tasks sharing a depth-1 queue, and its failure
modes are the ones unit tests of the pieces cannot see: a sentinel that never
arrives leaves the consumer parked on ``get()`` forever, and a producer with no
coupling to delivery keeps running inference for frames nobody receives. Both
are hangs or silent waste rather than exceptions, so nothing else would catch
them.

Everything here runs on fakes -- no model, no GPU, no socket -- because the
concurrency is what is under test, not the inference.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from app.core.config import Settings
from app.core.models import FrameTiming
from app.streaming.session import StreamSession
from app.streaming.wire import decode_stream_frame
from starlette.websockets import WebSocketDisconnect, WebSocketState

FRAME = np.zeros((16, 24, 3), dtype=np.uint8)


class FakeMetrics:
    degraded = False

    def current_fps(self) -> float:
        return 12.5


class FakeRuntime:
    """Counts how much inference actually happened."""

    precision = "fp32_gpu"
    imgsz = 640

    def __init__(self, fail_after: int | None = None) -> None:
        self.metrics = FakeMetrics()
        self.processed = 0
        self.sessions = 0
        self._fail_after = fail_after

    def new_session(self) -> None:
        self.sessions += 1

    def process(self, frame: np.ndarray) -> tuple[list, list, FrameTiming, int]:
        self.processed += 1
        if self._fail_after is not None and self.processed > self._fail_after:
            raise RuntimeError("inference exploded")
        return [], [], FrameTiming(inference_ms=1.0, total_ms=1.0), self.processed


class FakeSource:
    """A frame source that yields *total* frames and then finishes."""

    kind = "file"
    width = 24
    height = 16
    source_fps = 30.0
    total_frames = None

    def __init__(self, spec: str, **_: object) -> None:
        self.spec = spec
        self.produced = 0
        self.closed = False
        self._total = FakeSource.total_frames

    def open(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if self._total is not None and self.produced >= self._total:
            return None
        self.produced += 1
        return FRAME.copy()

    @property
    def finished(self) -> bool:
        return self._total is not None and self.produced >= self._total


class FakeWebSocket:
    """Records what the session sent; optionally stalls or drops the client."""

    def __init__(self, *, stall: asyncio.Event | None = None, fail_send: bool = False) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.text: list[str] = []
        self.binary: list[bytes] = []
        self._stall = stall
        self._fail_send = fail_send

    async def send_text(self, data: str) -> None:
        self.text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        if self._fail_send:
            raise WebSocketDisconnect(1006)
        if self._stall is not None:
            await self._stall.wait()
        self.binary.append(data)

    async def receive_text(self) -> str:
        # A client that never speaks; the session must not depend on it.
        await asyncio.Event().wait()
        return ""


@pytest.fixture
def patched_source(monkeypatch: pytest.MonkeyPatch) -> type[FakeSource]:
    created: list[FakeSource] = []

    def factory(spec: str, **kwargs: object) -> FakeSource:
        source = FakeSource(spec, **kwargs)
        created.append(source)
        return source

    monkeypatch.setattr("app.streaming.session.FrameSource", factory)
    FakeSource.instances = created  # type: ignore[attr-defined]
    return FakeSource


def _phases(socket: FakeWebSocket) -> list[str]:
    import json

    return [json.loads(t)["phase"] for t in socket.text if '"status"' in t]


def test_finite_source_delivers_every_frame_then_ends(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeSource, "total_frames", 5, raising=False)
    socket = FakeWebSocket()
    runtime = FakeRuntime()
    session = StreamSession(socket, runtime, Settings(), "clip.mp4")  # type: ignore[arg-type]

    asyncio.run(asyncio.wait_for(session.run(), timeout=10))

    assert len(socket.binary) == 5
    assert _phases(socket) == ["opening", "streaming", "ended"]
    assert runtime.sessions == 1


def test_frames_carry_a_decodable_header_and_jpeg(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeSource, "total_frames", 1, raising=False)
    socket = FakeWebSocket()
    session = StreamSession(socket, FakeRuntime(), Settings(), "clip.mp4")  # type: ignore[arg-type]

    asyncio.run(asyncio.wait_for(session.run(), timeout=10))

    header, jpeg = decode_stream_frame(socket.binary[0])
    assert header["kind"] == "frame"
    # Omitted, not null: the pixels are the bytes that follow, and repeating a
    # dead field on every frame is payload nobody reads.
    assert "image" not in header
    assert jpeg.startswith(b"\xff\xd8")  # a real JPEG, not a placeholder


def test_base64_encoding_sends_text_frames_carrying_a_data_uri(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy transport has to keep working; it is the documented fallback."""
    monkeypatch.setattr(FakeSource, "total_frames", 1, raising=False)
    socket = FakeWebSocket()
    session = StreamSession(
        socket,  # type: ignore[arg-type]
        FakeRuntime(),
        Settings(),
        "clip.mp4",
        encoding="base64",
    )

    asyncio.run(asyncio.wait_for(session.run(), timeout=10))

    assert not socket.binary
    frames = [t for t in socket.text if '"frame"' in t]
    assert len(frames) == 1
    assert "data:image/jpeg;base64," in frames[0]


def test_a_disconnected_client_ends_the_session_instead_of_hanging(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An endless source plus a dead socket is the classic hang; it must not."""
    monkeypatch.setattr(FakeSource, "total_frames", None, raising=False)
    socket = FakeWebSocket(fail_send=True)
    session = StreamSession(socket, FakeRuntime(), Settings(), "rtsp://cam")  # type: ignore[arg-type]

    asyncio.run(asyncio.wait_for(session.run(), timeout=10))

    assert socket.binary == []


def test_a_failing_producer_reports_an_error_and_terminates(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer exception must reach the client through the consumer.

    The producer never touches the socket -- two tasks sending concurrently is
    a data race -- so the error travels as a queue sentinel instead.
    """
    monkeypatch.setattr(FakeSource, "total_frames", None, raising=False)
    socket = FakeWebSocket()
    session = StreamSession(socket, FakeRuntime(fail_after=3), Settings(), "rtsp://cam")  # type: ignore[arg-type]

    asyncio.run(asyncio.wait_for(session.run(), timeout=10))

    assert "error" in _phases(socket)


def test_a_stalled_client_stops_inference_instead_of_burning_the_gpu(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pacing guarantee, and the reason the queue put is not drop-oldest.

    With an unbounded producer a client whose socket has stalled would leave
    capture, inference, tracking and annotation running at full rate forever,
    discarding every result and still writing detections for frames nobody
    receives. The depth-1 queue exists to pipeline, not to absorb; the
    producer must block behind a consumer that is not draining.
    """
    monkeypatch.setattr(FakeSource, "total_frames", None, raising=False)
    stall = asyncio.Event()
    socket = FakeWebSocket(stall=stall)
    runtime = FakeRuntime()
    session = StreamSession(socket, runtime, Settings(), "rtsp://cam")  # type: ignore[arg-type]

    async def scenario() -> int:
        task = asyncio.create_task(session.run())
        await asyncio.sleep(0.3)  # ample time to run away, if it could
        processed = runtime.processed
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return processed

    processed = asyncio.run(scenario())

    # One frame in the consumer's hands, one in the queue, one in the
    # producer: bounded by the pipeline's depth, not by elapsed time.
    assert processed <= 3, f"producer ran ahead of a stalled client: {processed} frames inferred"


def test_the_source_is_always_closed(
    patched_source: type[FakeSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capture thread holds a device handle; leaking it survives the session."""
    monkeypatch.setattr(FakeSource, "total_frames", 2, raising=False)
    socket = FakeWebSocket(fail_send=True)
    session = StreamSession(socket, FakeRuntime(), Settings(), "clip.mp4")  # type: ignore[arg-type]

    asyncio.run(asyncio.wait_for(session.run(), timeout=10))

    assert FakeSource.instances[-1].closed  # type: ignore[attr-defined]

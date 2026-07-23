"""Binary stream framing.

The framing is the contract between `app/streaming/wire.py` and the browser's parser in
`apps/web/hooks/use-stream.ts`. Both sides compute an offset from a length prefix,
so an off-by-one here does not raise -- it silently hands the image decoder a
JPEG with a few stray bytes of JSON on the front, or lops off its header. These
tests pin the boundaries.
"""

from __future__ import annotations

import asyncio
import json
import struct

import pytest
from app.core.config import Settings
from app.core.models import FrameTiming
from app.streaming.session import StreamSession, _EndOfStream, _Rendered
from app.streaming.wire import (
    MAX_HEADER_BYTES,
    WireFormatError,
    decode_stream_frame,
    encode_stream_frame,
    encode_stream_frame_raw,
)

JPEG = b"\xff\xd8\xff\xe0garbage-but-opaque-to-the-framing\xff\xd9"


def _rendered(frame_id: int) -> _Rendered:
    return _Rendered(
        canvas=None,
        frame_id=frame_id,
        width=8,
        height=8,
        tracks=[],
        timing=FrameTiming(),
        fps=0.0,
        precision="fp32_gpu",
        imgsz=640,
        degraded_mode=False,
    )


def test_round_trip_preserves_header_and_payload() -> None:
    header = {"kind": "frame", "frame_id": 7, "tracks": [{"track_id": 3}]}
    decoded, jpeg = decode_stream_frame(encode_stream_frame(header, JPEG))
    assert decoded == header
    assert jpeg == JPEG


def test_round_trip_preserves_non_ascii_header() -> None:
    """Class names are not guaranteed ASCII, and the prefix counts *bytes*."""
    header = {"class_name": "vélo", "note": "日本語"}
    decoded, jpeg = decode_stream_frame(encode_stream_frame(header, JPEG))
    assert decoded == header
    assert jpeg == JPEG


def test_empty_payload_round_trips() -> None:
    decoded, jpeg = decode_stream_frame(encode_stream_frame({"frame_id": 1}, b""))
    assert decoded == {"frame_id": 1}
    assert jpeg == b""


def test_payload_starting_with_a_brace_is_not_absorbed_into_the_header() -> None:
    """The split is by length, never by scanning for JSON's end."""
    payload = b'{"not":"json-really"}\xff\xd9'
    decoded, jpeg = decode_stream_frame(encode_stream_frame({"frame_id": 2}, payload))
    assert decoded == {"frame_id": 2}
    assert jpeg == payload


def test_raw_and_dict_encoders_agree() -> None:
    header = {"frame_id": 4, "fps": 12.5}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    assert encode_stream_frame_raw(blob, JPEG) == encode_stream_frame(header, JPEG)


def test_truncated_prefix_is_rejected() -> None:
    with pytest.raises(WireFormatError):
        decode_stream_frame(b"\x00\x00")


def test_truncated_header_is_rejected() -> None:
    message = encode_stream_frame({"frame_id": 5}, JPEG)
    with pytest.raises(WireFormatError):
        decode_stream_frame(message[:6])


def test_absurd_declared_length_is_rejected_without_allocating() -> None:
    with pytest.raises(WireFormatError):
        decode_stream_frame(struct.pack(">I", MAX_HEADER_BYTES + 1) + b"{}")


def test_non_object_header_is_rejected() -> None:
    with pytest.raises(WireFormatError):
        decode_stream_frame(encode_stream_frame_raw(b"[1,2,3]", JPEG))


def test_invalid_json_header_is_rejected() -> None:
    with pytest.raises(WireFormatError):
        decode_stream_frame(encode_stream_frame_raw(b"{not json", JPEG))


def test_oversized_header_is_refused_at_encode_time() -> None:
    with pytest.raises(WireFormatError):
        encode_stream_frame_raw(b"x" * (MAX_HEADER_BYTES + 1), JPEG)


def _session() -> StreamSession:
    """A session whose collaborators are never touched by the queue logic."""
    return StreamSession(None, None, Settings(), "sample")  # type: ignore[arg-type]


def test_sentinel_reaches_an_empty_queue() -> None:
    async def scenario() -> object:
        session = _session()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        session._force_sentinel(queue, _EndOfStream())
        return queue.get_nowait()

    assert isinstance(asyncio.run(scenario()), _EndOfStream)


def test_sentinel_displaces_a_pending_frame_rather_than_waiting() -> None:
    """The consumer parks on ``get()``, so a lost sentinel is a hung session.

    The shutdown path cannot await, so it evicts an undelivered frame instead.
    Losing one frame at teardown beats never terminating.
    """

    async def scenario() -> object:
        session = _session()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        queue.put_nowait(_rendered(0))
        session._force_sentinel(queue, _EndOfStream("boom", fatal=True))
        return queue.get_nowait()

    item = asyncio.run(scenario())
    assert isinstance(item, _EndOfStream)
    assert item.fatal and item.message == "boom"

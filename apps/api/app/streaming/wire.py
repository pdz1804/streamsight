"""Binary framing for the streaming WebSocket.

A streamed frame is two things that must never be separated: the JPEG pixels and
the metadata describing them (frame id, boxes, timings). The obvious encoding --
a JSON message with the image as a base64 data URI -- keeps them together but
costs more than it looks:

* base64 inflates the payload by a third,
* the server pays an encode, and then pays *again* when the JSON serializer
  escapes and copies that ~130 KB string,
* the browser pays a decode, and a ``data:`` URI handed to ``new Image()``
  decodes on the main thread.

None of that work moves a pixel. This module replaces it with a single binary
message carrying both parts:

``[4-byte big-endian uint32 header length][UTF-8 JSON header][raw JPEG bytes]``

One message rather than two deliberately. Sending the header and the image as
separate WebSocket messages would be simpler to write, but two concurrent sends
could interleave and pair a header with the wrong image -- a failure that would
show up as boxes lagging the picture, intermittently, under load only. A single
message makes that unrepresentable.

Status messages stay text JSON, so the client discriminates on the frame's type
alone and never has to sniff content.
"""

from __future__ import annotations

import json
import struct

#: Big-endian uint32. Network byte order, and wide enough that the length field
#: can never be the thing that limits frame size.
_HEADER_LEN = struct.Struct(">I")

#: Guards against a malformed or hostile length prefix causing a huge slice.
#: Headers carry boxes and timings; anything approaching a megabyte is a bug.
MAX_HEADER_BYTES = 1 << 20


class WireFormatError(ValueError):
    """A binary stream message could not be parsed."""


def encode_stream_frame_raw(header_json: bytes, jpeg: bytes) -> bytes:
    """Pack an already-serialized *header_json* and *jpeg* into one message.

    The hot path uses this so the header can come straight from Pydantic's
    serializer instead of being round-tripped through a dict.
    """
    if len(header_json) > MAX_HEADER_BYTES:
        raise WireFormatError(f"header too large: {len(header_json)} bytes")
    return _HEADER_LEN.pack(len(header_json)) + header_json + jpeg


def encode_stream_frame(header: dict[str, object], jpeg: bytes) -> bytes:
    """Pack *header* and *jpeg* into one length-prefixed binary message."""
    return encode_stream_frame_raw(json.dumps(header, separators=(",", ":")).encode("utf-8"), jpeg)


def decode_stream_frame(message: bytes) -> tuple[dict[str, object], bytes]:
    """Unpack a message produced by :func:`encode_stream_frame`.

    Raises:
        WireFormatError: the buffer is truncated, mis-prefixed, or not JSON.
    """
    if len(message) < _HEADER_LEN.size:
        raise WireFormatError("message shorter than its length prefix")
    (header_len,) = _HEADER_LEN.unpack_from(message, 0)
    if header_len > MAX_HEADER_BYTES:
        raise WireFormatError(f"header length {header_len} exceeds the cap")
    start = _HEADER_LEN.size
    end = start + header_len
    if len(message) < end:
        raise WireFormatError("message truncated inside its header")
    try:
        header = json.loads(message[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireFormatError(f"invalid header: {exc}") from exc
    if not isinstance(header, dict):
        raise WireFormatError("header is not a JSON object")
    return header, message[end:]

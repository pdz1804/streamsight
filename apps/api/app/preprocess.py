"""Frame decoding.

Deliberately thin: Ultralytics performs letterbox resize, normalization and NMS
inside ``model.track``, so the only job here is turning whatever the client sent
into a BGR ``ndarray``. Adding a manual CHW/float32 step would duplicate work the
inference backend already does.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import cv2
import numpy as np

from .exceptions import InvalidFrameError

_DATA_URI_PREFIX = "data:"


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/...) into a BGR frame."""
    if not raw:
        raise InvalidFrameError("empty image payload")
    buffer = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise InvalidFrameError("could not decode image - unsupported or corrupt format")
    return frame


def decode_base64_frame(payload: str) -> np.ndarray:
    """Decode a base64 string, with or without a ``data:image/...;base64,`` prefix."""
    data = payload.strip()
    if data.startswith(_DATA_URI_PREFIX):
        _, _, data = data.partition(",")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidFrameError(f"invalid base64 image payload: {exc}") from exc
    return decode_image_bytes(raw)


def read_image_file(path: str | Path) -> np.ndarray:
    """Read an image from disk into a BGR frame."""
    file_path = Path(path)
    if not file_path.exists():
        raise InvalidFrameError(f"image not found: {file_path}")
    frame = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise InvalidFrameError(f"could not decode image file: {file_path}")
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame to JPEG bytes for transport."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise InvalidFrameError("JPEG encoding failed")
    return buffer.tobytes()


def encode_jpeg_data_uri(frame: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR frame as a ``data:image/jpeg;base64,...`` URI."""
    payload = base64.b64encode(encode_jpeg(frame, quality)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"

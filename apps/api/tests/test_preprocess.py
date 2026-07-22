"""Frame decoding and encoding."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from app.exceptions import InvalidFrameError
from app.preprocess import (
    decode_base64_frame,
    decode_image_bytes,
    encode_jpeg,
    encode_jpeg_data_uri,
    read_image_file,
)


@pytest.fixture
def jpeg_bytes(blank_frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", blank_frame)
    assert ok
    return buffer.tobytes()


def test_decode_image_bytes_round_trips_dimensions(jpeg_bytes: bytes) -> None:
    frame = decode_image_bytes(jpeg_bytes)
    assert frame.shape == (480, 640, 3)


def test_decode_rejects_empty_payload() -> None:
    with pytest.raises(InvalidFrameError, match="empty"):
        decode_image_bytes(b"")


def test_decode_rejects_non_image_bytes() -> None:
    with pytest.raises(InvalidFrameError, match="decode"):
        decode_image_bytes(b"this is definitely not a jpeg")


def test_bare_base64_is_accepted(jpeg_bytes: bytes) -> None:
    frame = decode_base64_frame(base64.b64encode(jpeg_bytes).decode())
    assert frame.shape == (480, 640, 3)


def test_data_uri_prefix_is_stripped(jpeg_bytes: bytes) -> None:
    payload = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
    frame = decode_base64_frame(payload)
    assert frame.shape == (480, 640, 3)


def test_malformed_base64_is_reported_clearly() -> None:
    with pytest.raises(InvalidFrameError, match="base64"):
        decode_base64_frame("!!!not base64 at all!!!")


def test_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(InvalidFrameError, match="not found"):
        read_image_file(tmp_path / "absent.jpg")


def test_encode_jpeg_produces_a_decodable_image(blank_frame: np.ndarray) -> None:
    raw = encode_jpeg(blank_frame, quality=70)
    assert raw[:2] == b"\xff\xd8"  # JPEG start-of-image marker
    assert decode_image_bytes(raw).shape == blank_frame.shape


def test_data_uri_encoding_is_self_describing(blank_frame: np.ndarray) -> None:
    uri = encode_jpeg_data_uri(blank_frame)
    assert uri.startswith("data:image/jpeg;base64,")
    assert decode_base64_frame(uri).shape == blank_frame.shape


def test_lower_quality_produces_smaller_output(sample_frame: np.ndarray) -> None:
    """Guards against the quality argument being silently ignored."""
    assert len(encode_jpeg(sample_frame, quality=20)) < len(encode_jpeg(sample_frame, quality=95))

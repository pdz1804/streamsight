"""HTTP and WebSocket contract tests against a genuinely loaded model.

These are the slow tier: the session-scoped `client` fixture runs the real
lifespan, so a passing run proves the model loads, serves, hot-swaps and streams
on this machine, not merely that the routes are wired up.
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from app.streaming.wire import decode_stream_frame

pytestmark = pytest.mark.slow


def _b64_jpeg(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode()


def _collect_frames(socket, count, *, binary) -> list[tuple[dict, bytes | None]]:
    """Read until `count` frame messages arrive, failing fast if the stream dies.

    Status messages are text JSON in both encodings; only frames differ, so the
    branch is on the transport of the payload rather than on the message kind.
    """
    frames = []
    while len(frames) < count:
        message = socket.receive()
        if message.get("bytes") is not None:
            header, jpeg = decode_stream_frame(message["bytes"])
            assert binary, "binary frame arrived on a base64 stream"
            frames.append((header, jpeg))
            continue
        payload = json.loads(message["text"])
        if payload["kind"] == "frame":
            assert not binary, "text frame arrived on a binary stream"
            frames.append((payload, None))
        elif payload["phase"] in {"error", "ended"}:
            pytest.fail(f"stream ended early: {payload['message']}")
    return frames


# ------------------------------------------------------------------- health


def test_health_reports_the_active_configuration(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["app"] == "StreamSight"
    assert body["imgsz"] > 0
    assert body["precision"] != "none"
    assert "available" in body["gpu"]


# ------------------------------------------------------------------- detect


def test_detect_frame_returns_wellformed_detections(client, sample_frame) -> None:
    response = client.post("/detect/frame", json={"image": _b64_jpeg(sample_frame)})
    assert response.status_code == 200
    body = response.json()

    height, width = sample_frame.shape[:2]
    assert (body["width"], body["height"]) == (width, height)
    assert body["timing"]["inference_ms"] > 0
    assert body["detections"], "the sample clip should contain detectable objects"

    for detection in body["detections"]:
        assert detection["x1"] < detection["x2"]
        assert detection["y1"] < detection["y2"]
        assert 0.0 <= detection["confidence"] <= 1.0
        assert isinstance(detection["class_name"], str) and detection["class_name"]


def test_detect_frame_accepts_a_data_uri(client, sample_frame) -> None:
    payload = f"data:image/jpeg;base64,{_b64_jpeg(sample_frame)}"
    assert client.post("/detect/frame", json={"image": payload}).status_code == 200


def test_detect_frame_rejects_undecodable_input(client) -> None:
    response = client.post("/detect/frame", json={"image": base64.b64encode(b"junk").decode()})
    assert response.status_code == 400
    assert response.json()["error"] == "InvalidFrameError"


def test_tracks_accumulate_identities_across_frames(client, settings) -> None:
    """Track ids only appear once ByteTrack has seen an object over several frames."""
    capture = cv2.VideoCapture(str(settings.assets_dir / "sample.mp4"))
    try:
        seen: set[int] = set()
        for _ in range(12):
            ok, frame = capture.read()
            if not ok:
                break
            body = client.post("/detect/frame", json={"image": _b64_jpeg(frame)}).json()
            seen.update(t["track_id"] for t in body["tracks"] if t["track_id"] is not None)
    finally:
        capture.release()
    assert seen, "no persistent track ids were assigned over 12 consecutive frames"


def test_detect_image_accepts_a_multipart_upload(client, sample_frame) -> None:
    ok, buffer = cv2.imencode(".jpg", sample_frame)
    assert ok
    response = client.post(
        "/detect/image",
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["detections"]


# ------------------------------------------------------------------ metrics


def test_metrics_expose_the_full_telemetry_contract(client, sample_frame) -> None:
    client.post("/detect/frame", json={"image": _b64_jpeg(sample_frame)})
    body = client.get("/metrics").json()

    for field in (
        "fps",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "frames_processed",
        "track_count",
        "unique_tracks",
        "gpu",
        "cpu_percent",
        "degraded_mode",
        "precision",
        "imgsz",
        "uptime_s",
    ):
        assert field in body, f"missing metric: {field}"
    assert body["frames_processed"] > 0


# ------------------------------------------------------------------- config


def test_model_config_lists_backends_with_reasons(client) -> None:
    body = client.get("/config/model").json()
    assert body["available_backends"]
    assert body["supported_imgsz"]
    for backend in body["available_backends"]:
        # An unavailable backend must always explain itself.
        if not backend["available"]:
            assert backend["reason"], f"{backend['precision']} is unavailable with no reason"


def test_resolution_hot_swap_takes_effect(client) -> None:
    original = client.get("/config/model").json()["imgsz"]
    target = 480 if original != 480 else 640

    switched = client.post("/config/model", json={"imgsz": target})
    assert switched.status_code == 200
    assert switched.json()["imgsz"] == target
    assert client.get("/health").json()["imgsz"] == target

    restored = client.post("/config/model", json={"imgsz": original})
    assert restored.json()["imgsz"] == original


def test_unknown_precision_is_rejected_without_disturbing_service(client) -> None:
    before = client.get("/config/model").json()["precision"]
    response = client.post("/config/model", json={"precision": "int4_quantum"})
    assert response.status_code == 409
    assert client.get("/config/model").json()["precision"] == before


def test_unsupported_resolution_is_rejected(client) -> None:
    response = client.post("/config/model", json={"imgsz": 123})
    assert response.status_code == 409
    assert "unsupported resolution" in response.json()["detail"]


def test_unknown_fields_are_refused(client) -> None:
    """`extra="forbid"` on the request model guards against silent typos."""
    assert client.post("/config/model", json={"precison": "fp32_cpu"}).status_code == 422


# ------------------------------------------------------------------ sources


def test_sources_always_offer_the_webcam_entry(client) -> None:
    ids = {source["id"] for source in client.get("/sources").json()}
    assert "webcam" in ids


def test_upload_rejects_a_non_video_extension(client) -> None:
    response = client.post(
        "/sources/upload",
        files={"file": ("notes.txt", b"definitely not a video", "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported video type" in response.json()["detail"]


# ------------------------------------------------------------------- stream


def test_stream_reports_an_unknown_source_instead_of_hanging(client) -> None:
    with client.websocket_connect("/detect/stream?source=does-not-exist") as socket:
        message = socket.receive_json()
    assert message["kind"] == "status"
    assert message["phase"] == "error"


def test_stream_delivers_annotated_frames(client) -> None:
    """End-to-end proof over the default transport: real frames, real overlays.

    Binary is what the browser negotiates, so it is what this asserts. The header
    carries no ``image`` key at all -- the pixels are the second half of the same
    message -- which is exactly the difference an earlier version of this test
    missed by calling ``receive_json``.
    """
    with client.websocket_connect("/detect/stream?source=sample&loop=true") as socket:
        frames = _collect_frames(socket, 3, binary=True)

    for header, jpeg in frames:
        assert "image" not in header
        assert jpeg.startswith(b"\xff\xd8"), "payload is not a JPEG"
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.size > 0
        assert header["width"] > 0 and header["height"] > 0
        assert header["timing"]["inference_ms"] > 0
        assert header["server_ts"] > 0

    ids = [h["frame_id"] for h, _ in frames]
    assert ids == sorted(ids)


def test_stream_base64_encoding_still_carries_a_data_uri(client) -> None:
    """The legacy transport stays available for non-browser consumers."""
    url = "/detect/stream?source=sample&loop=true&encoding=base64"
    with client.websocket_connect(url) as socket:
        frames = _collect_frames(socket, 2, binary=False)

    for payload, _ in frames:
        assert payload["image"].startswith("data:image/jpeg;base64,")
        assert payload["width"] > 0 and payload["height"] > 0

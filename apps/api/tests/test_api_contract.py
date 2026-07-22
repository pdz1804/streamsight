"""Spec-vocabulary aliases on the public API contract.

The PRD names `latency_ms`, `gpu_mem_mb`, `resolution` and the precision words
`int8|fp16|fp32`; the implementation grew a nested `timing{}`/`gpu{}` shape and
concrete backend keys instead. The aliases are additive, and the risk they carry
is exactly that: two spellings of one value can drift apart, and a new accepted
field can weaken `extra="forbid"`. These tests pin both halves -- the alias
answers, and the original spellings still answering unchanged.

The mapping tests build a runtime over stub artifacts instead of loading a model:
precision resolution is pure host-capability logic, so a real engine would only
make the suite slow.
"""

from __future__ import annotations

import base64

import cv2
import pytest
from app.backends import BACKENDS
from app.config import GpuProbe, Settings
from app.exceptions import BackendUnavailableError
from app.models import (
    FrameResponse,
    FrameTiming,
    GpuInfo,
    MetricsResponse,
    ModelConfigRequest,
)
from app.runtime import InferenceRuntime
from pydantic import ValidationError


def _b64_jpeg(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode()


def _metrics(**overrides: object) -> MetricsResponse:
    payload = {
        "fps": 30.0,
        "fps_rolling": [],
        "avg_latency_ms": 12.0,
        "p50_latency_ms": 11.0,
        "p95_latency_ms": 20.0,
        "frames_processed": 5,
        "track_count": 2,
        "unique_tracks": 3,
        "gpu": GpuInfo(available=True, name="stub", total_mb=4096, used_mb=917, free_mb=3179),
        "cpu_percent": 4.0,
        "ram_used_mb": 8000,
        "process_ram_mb": 900,
        "degraded_mode": False,
        "degrade_reason": None,
        "precision": "fp32_gpu",
        "imgsz": 640,
        "uptime_s": 1.0,
    }
    payload.update(overrides)
    return MetricsResponse(**payload)


@pytest.fixture
def cpu_only_runtime(tmp_path) -> InferenceRuntime:
    """A runtime whose host has every artifact present but no GPU.

    Stub files are enough: `availability()` only checks for existence, and no
    model is ever loaded here.
    """
    settings = Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        assets_dir=tmp_path / "assets",
    )
    for backend in BACKENDS.values():
        path = backend.path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        if backend.key == "openvino_cpu":
            path.mkdir(exist_ok=True)
        elif not path.exists():
            path.write_bytes(b"stub artifact")

    runtime = InferenceRuntime(settings)
    runtime._gpu = GpuProbe(available=False)
    return runtime


# --------------------------------------------------------------- FR-5 latency


def test_frame_response_mirrors_total_ms_as_flat_latency() -> None:
    response = FrameResponse(
        frame_id=1,
        width=640,
        height=480,
        detections=[],
        tracks=[],
        timing=FrameTiming(inference_ms=30.5, total_ms=41.25),
        fps=24.0,
        precision="fp32_cpu",
        imgsz=640,
        degraded_mode=False,
    )
    body = response.model_dump()
    assert body["latency_ms"] == 41.25
    # The nested breakdown is unchanged: it remains the source of truth.
    assert body["timing"]["total_ms"] == 41.25


# --------------------------------------------------------------- FR-7 gpu mem


def test_metrics_mirror_gpu_used_mb_as_flat_gpu_mem_mb() -> None:
    body = _metrics().model_dump()
    assert body["gpu_mem_mb"] == 917
    assert body["gpu"]["used_mb"] == 917


def test_flat_gpu_mem_follows_the_nested_value() -> None:
    """A mirror that can be set independently would be a second source of truth."""
    gpu = GpuInfo(available=False, name="cpu", total_mb=0, used_mb=0, free_mb=0)
    assert _metrics(gpu=gpu).gpu_mem_mb == 0


# ------------------------------------------------------- FR-8 resolution alias


def test_resolution_is_accepted_as_a_synonym_for_imgsz() -> None:
    assert ModelConfigRequest(resolution=480).imgsz == 480


def test_imgsz_still_works_on_its_own() -> None:
    request = ModelConfigRequest(imgsz=320)
    assert request.imgsz == 320
    assert request.resolution is None


def test_matching_imgsz_and_resolution_are_accepted() -> None:
    assert ModelConfigRequest(imgsz=640, resolution=640).imgsz == 640


def test_contradictory_imgsz_and_resolution_are_refused() -> None:
    """Guessing which one the caller meant would silently ignore half the request."""
    with pytest.raises(ValidationError):
        ModelConfigRequest(imgsz=640, resolution=480)


def test_unknown_fields_are_still_forbidden() -> None:
    """Adding an accepted field must not relax the typo guard."""
    with pytest.raises(ValidationError):
        ModelConfigRequest(resolition=480)


# ------------------------------------------------------- FR-8 precision words


def test_int8_resolves_to_the_cpu_graph_when_tensorrt_is_absent(
    cpu_only_runtime: InferenceRuntime,
) -> None:
    assert cpu_only_runtime._resolve_precision("int8", 640) == "int8_onnx_cpu"


def test_fp32_resolves_to_the_cpu_weights_without_a_gpu(
    cpu_only_runtime: InferenceRuntime,
) -> None:
    assert cpu_only_runtime._resolve_precision("fp32", 640) == "fp32_cpu"


def test_fp16_has_no_cpu_fallback_and_says_what_it_tried(
    cpu_only_runtime: InferenceRuntime,
) -> None:
    """Both FP16 artifacts are CUDA-only, so a GPU-less host genuinely cannot serve it."""
    with pytest.raises(BackendUnavailableError) as excinfo:
        cpu_only_runtime._resolve_precision("fp16", 640)
    message = str(excinfo.value)
    assert "fp16_trt" in message
    assert "fp16_onnx" in message


def test_precision_words_respect_the_requested_resolution(
    cpu_only_runtime: InferenceRuntime,
) -> None:
    """The INT8 CPU graph is exported at 640 px, so 480 px has nothing left to offer."""
    with pytest.raises(BackendUnavailableError) as excinfo:
        cpu_only_runtime._resolve_precision("int8", 480)
    assert "int8_onnx_cpu" in str(excinfo.value)


def test_concrete_backend_keys_pass_through_unchanged(
    cpu_only_runtime: InferenceRuntime,
) -> None:
    assert cpu_only_runtime._resolve_precision("fp32_cpu", 640) == "fp32_cpu"
    assert cpu_only_runtime._resolve_precision("openvino_cpu", 640) == "openvino_cpu"


def test_unknown_precision_is_left_for_the_caller_to_reject(
    cpu_only_runtime: InferenceRuntime,
) -> None:
    """Resolution only widens the vocabulary; validation stays in `switch`."""
    assert cpu_only_runtime._resolve_precision("int4_quantum", 640) == "int4_quantum"


# ------------------------------------------------------------- over HTTP


@pytest.mark.slow
def test_detect_frame_serves_both_latency_spellings(client, sample_frame) -> None:
    body = client.post("/detect/frame", json={"image": _b64_jpeg(sample_frame)}).json()
    assert body["latency_ms"] == body["timing"]["total_ms"]
    assert body["latency_ms"] > 0


@pytest.mark.slow
def test_metrics_serve_both_gpu_memory_spellings(client, sample_frame) -> None:
    client.post("/detect/frame", json={"image": _b64_jpeg(sample_frame)})
    body = client.get("/metrics").json()
    assert body["gpu_mem_mb"] == body["gpu"]["used_mb"]


@pytest.mark.slow
def test_config_accepts_resolution_over_http(client) -> None:
    original = client.get("/config/model").json()["imgsz"]
    target = 480 if original != 480 else 640
    try:
        switched = client.post("/config/model", json={"resolution": target})
        assert switched.status_code == 200
        assert switched.json()["imgsz"] == target
    finally:
        client.post("/config/model", json={"imgsz": original})


@pytest.mark.slow
def test_config_accepts_a_precision_word_over_http(client) -> None:
    before = client.get("/config/model").json()
    try:
        response = client.post("/config/model", json={"precision": "fp32"})
        assert response.status_code == 200
        # Which concrete key wins depends on the host; it must be an fp32 one.
        assert response.json()["precision"] in {"fp32_gpu", "fp32_cpu"}
    finally:
        client.post(
            "/config/model",
            json={"precision": before["precision"], "imgsz": before["imgsz"]},
        )


@pytest.mark.slow
def test_contradictory_resolution_request_is_refused_over_http(client) -> None:
    assert client.post("/config/model", json={"imgsz": 640, "resolution": 480}).status_code == 422

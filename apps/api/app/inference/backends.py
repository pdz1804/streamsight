"""Inference backend registry and the degradation ladder.

A *backend* pairs an exported artifact with the device it runs on. Keeping the
catalogue in one declarative table means the API, the model selector UI, the
benchmark harness, and the export scripts all agree on the same identifiers
(``int8_trt``, ``fp16_onnx``, ...) without duplicating path logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.config import Settings

#: Ladder walked when the preferred backend is unavailable or runs out of VRAM.
#: Each step is strictly cheaper in memory than the one before it.
FALLBACK_LADDER: tuple[str, ...] = (
    "int8_trt",
    "fp16_trt",
    "fp16_onnx",
    "fp32_gpu",
    "openvino_cpu",
    "int8_onnx_cpu",
    "fp32_cpu",
)


@dataclass(frozen=True)
class Backend:
    """Declarative description of one runnable inference artifact."""

    key: str
    label: str
    description: str
    #: Path relative to ``settings.models_dir``. Directories are valid (OpenVINO).
    artifact: str
    device: str
    requires_gpu: bool
    #: True when the artifact must be produced by an export step first.
    exported: bool = True
    #: Exported graphs bake in their input resolution; only the PyTorch weights
    #: accept an arbitrary ``imgsz``. Requesting another size from a fixed
    #: artifact is a hard error at inference, so it is rejected up front.
    dynamic_shape: bool = False
    #: Resolution a fixed-shape artifact was exported at.
    export_imgsz: int = 640

    def path(self, settings: Settings) -> Path:
        return settings.models_dir / self.artifact

    def supports_imgsz(self, imgsz: int) -> bool:
        return self.dynamic_shape or imgsz == self.export_imgsz


BACKENDS: dict[str, Backend] = {
    "int8_trt": Backend(
        key="int8_trt",
        label="INT8 - TensorRT",
        description="8-bit quantized TensorRT engine. Fastest and smallest on NVIDIA.",
        artifact="engines/yolo11n_int8.engine",
        device="cuda",
        requires_gpu=True,
    ),
    "fp16_trt": Backend(
        key="fp16_trt",
        label="FP16 - TensorRT",
        description="Half-precision TensorRT engine. Full accuracy, more VRAM than INT8.",
        artifact="engines/yolo11n_fp16.engine",
        device="cuda",
        requires_gpu=True,
    ),
    "fp16_onnx": Backend(
        key="fp16_onnx",
        label="FP16 - ONNX Runtime",
        description="Portable half-precision GPU path. Works without TensorRT.",
        artifact="engines/yolo11n_fp16.onnx",
        device="cuda",
        requires_gpu=True,
    ),
    "fp32_gpu": Backend(
        key="fp32_gpu",
        label="FP32 - PyTorch GPU",
        description="Unquantized reference on GPU. Accuracy baseline for the frontier.",
        artifact="weights/yolo11n.pt",
        device="cuda",
        requires_gpu=True,
        exported=False,
        dynamic_shape=True,
    ),
    "openvino_cpu": Backend(
        key="openvino_cpu",
        label="OpenVINO - CPU",
        description="Intel-optimized CPU runtime. Best no-GPU throughput.",
        artifact="engines/yolo11n_openvino_model",
        device="cpu",
        requires_gpu=False,
    ),
    "int8_onnx_cpu": Backend(
        key="int8_onnx_cpu",
        label="INT8 - ONNX CPU",
        description="Quantized CPU path for GPU-less deployment.",
        artifact="engines/yolo11n_int8.onnx",
        device="cpu",
        requires_gpu=False,
    ),
    "fp32_cpu": Backend(
        key="fp32_cpu",
        label="FP32 - PyTorch CPU",
        description="Always-available last resort. Slow but never fails to load.",
        artifact="weights/yolo11n.pt",
        device="cpu",
        requires_gpu=False,
        exported=False,
        dynamic_shape=True,
    ),
}


def get_backend(key: str) -> Backend:
    """Look up a backend by key.

    Raises:
        KeyError: if the key is not a known backend.
    """
    return BACKENDS[key]


def availability(backend: Backend, settings: Settings, gpu_available: bool) -> tuple[bool, str]:
    """Return ``(is_runnable, human_reason)`` for one backend on this host."""
    if backend.requires_gpu and not gpu_available:
        return False, "no NVIDIA GPU on this host"
    path = backend.path(settings)
    if not path.exists():
        hint = "run ml/quantization exports" if backend.exported else "download yolo11n.pt"
        return False, f"artifact missing ({hint})"
    if backend.key in {"int8_trt", "fp16_trt"} and not _tensorrt_installed():
        return False, "tensorrt not installed"
    return True, ""


def candidate_chain(
    preferred: str | None, settings: Settings, gpu_available: bool
) -> list[Backend]:
    """Build the ordered list of backends to attempt.

    The preferred backend goes first (when named), then the standard ladder with
    duplicates removed. Unavailable entries are filtered out, so the caller can
    simply take the first element -- and an empty result means nothing at all can
    run, which is a genuine startup failure.
    """
    order: list[str] = []
    if preferred and preferred != "auto":
        order.append(preferred)
    order.extend(k for k in FALLBACK_LADDER if k not in order)

    chain: list[Backend] = []
    for key in order:
        backend = BACKENDS.get(key)
        if backend is None:
            continue
        runnable, _ = availability(backend, settings, gpu_available)
        if runnable:
            chain.append(backend)
    return chain


def _tensorrt_installed() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:  # noqa: BLE001 - import can fail on missing CUDA libs too
        return False
    return True

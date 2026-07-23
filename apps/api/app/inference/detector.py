"""The detection + tracking engine.

Wraps a single Ultralytics model instance. One ``Detector`` owns one loaded
backend at one resolution; swapping either means constructing a new instance,
which keeps the object immutable enough to reason about under concurrency (the
hot-swap lock lives in :mod:`app.runtime`, not here).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Settings
from ..core.models import Detection, Track
from ..vision.tracker import ensure_tracker_config, parse_results
from .backends import Backend

logger = logging.getLogger(__name__)


class OutOfVramError(RuntimeError):
    """Raised when inference exhausts device memory and degradation is required."""


def is_oom(exc: BaseException) -> bool:
    """Heuristic: does this exception mean 'the GPU ran out of memory'?

    Backends report OOM differently -- PyTorch raises ``torch.cuda.OutOfMemoryError``
    while TensorRT and ONNX Runtime surface plain runtime errors with driver text --
    so message matching is the only portable signal.
    """
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("out of memory", "cuda_error_out_of_memory", "cudnn_status_alloc_failed")
    )


_dll_directories_added = False


def _expose_torch_cuda_libraries() -> None:
    """Put torch's bundled CUDA and cuDNN DLLs on the loader path (Windows).

    ONNX Runtime advertises a CUDA provider whenever the package is the GPU
    build, but it loads the CUDA libraries itself and does not see the copies
    torch ships inside ``torch/lib``. Without this the provider is listed,
    accepted, and then fails at the first bind with an opaque device error.
    Idempotent, and a no-op on platforms without ``add_dll_directory``.
    """
    global _dll_directories_added
    if _dll_directories_added or not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            # ONNX Runtime resolves its provider DLLs through the process PATH
            # rather than Python's added-DLL-directory list, so set both.
            os.environ["PATH"] = f"{torch_lib}{os.pathsep}{os.environ.get('PATH', '')}"
            logger.debug("added %s to the DLL search path", torch_lib)
    except Exception as exc:  # noqa: BLE001 - best effort; the ladder covers failure
        logger.debug("could not expose torch CUDA libraries: %s", exc)
    _dll_directories_added = True


class Detector:
    """A loaded inference backend plus its ByteTrack state."""

    def __init__(self, backend: Backend, imgsz: int, settings: Settings) -> None:
        self.backend = backend
        self.imgsz = imgsz
        self._settings = settings
        self._tracker_cfg = ensure_tracker_config(settings.tracker_config_path)
        self._model: Any | None = None
        self._model_path: Path = backend.path(settings)
        self._warm = False

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Instantiate the Ultralytics model for this backend.

        Raises:
            FileNotFoundError: the artifact is missing.
            Exception: propagated from Ultralytics if the artifact is unusable,
                so the caller can step down the fallback ladder.
        """
        from ultralytics import YOLO

        if not self._model_path.exists():
            raise FileNotFoundError(f"artifact not found: {self._model_path}")

        if self.backend.device == "cuda" and self._model_path.suffix == ".onnx":
            _expose_torch_cuda_libraries()

        started = time.perf_counter()
        self._model = YOLO(str(self._model_path), task="detect")
        logger.info(
            "loaded backend=%s artifact=%s device=%s in %.2fs",
            self.backend.key,
            self._model_path.name,
            self.backend.device,
            time.perf_counter() - started,
        )

    def warmup(self, frames: int = 2) -> None:
        """Run throwaway inferences so the first real frame is not the slow one.

        TensorRT and ONNX Runtime both defer allocation and kernel selection to
        the first call; without this the UI would show a multi-second first frame.

        Warmup doubles as the real usability check. A backend can load happily and
        still fail on execution -- onnxruntime-gpu advertises a CUDA provider it
        cannot bind when the installed cuDNN is too old, for instance. Failing
        here lets the caller step down the ladder instead of serving every request
        from a backend that cannot run.

        Raises:
            Exception: whatever the backend raised, for the caller to act on.
        """
        if self._model is None:
            raise RuntimeError("warmup called before load()")
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(frames):
            self._predict(blank)
        self.reset_tracker()
        self._warm = True

    @property
    def is_warm(self) -> bool:
        return self._warm

    @property
    def model_file(self) -> str:
        return self._model_path.name

    @property
    def device(self) -> str:
        return self.backend.device

    # ----------------------------------------------------------------- infer

    def track(self, frame: np.ndarray) -> tuple[list[Detection], list[Track], float]:
        """Detect and associate objects in one BGR frame.

        Returns:
            ``(detections, tracks, inference_ms)``.

        Raises:
            OutOfVramError: device memory was exhausted; the caller should degrade.
        """
        if self._model is None:
            raise RuntimeError("detector used before load()")
        started = time.perf_counter()
        try:
            result = self._predict(frame)
        except Exception as exc:
            if is_oom(exc):
                raise OutOfVramError(str(exc)) from exc
            raise
        inference_ms = (time.perf_counter() - started) * 1000.0
        detections, tracks = parse_results(result)
        return detections, tracks, inference_ms

    def _predict(self, frame: np.ndarray) -> Any:
        if self._model is None:
            raise RuntimeError("detector used before load()")
        results = self._model.track(
            frame,
            tracker=str(self._tracker_cfg),
            persist=True,
            imgsz=self.imgsz,
            conf=self._settings.conf_threshold,
            iou=self._settings.iou_threshold,
            device=self.backend.device,
            verbose=False,
        )
        return results[0]

    # ----------------------------------------------------------------- state

    def reset_tracker(self) -> None:
        """Drop accumulated track identities.

        Called when switching video sources: carrying IDs across unrelated
        footage would produce nonsensical identity continuity.
        """
        model = self._model
        if model is None:
            return
        predictor = getattr(model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if not trackers:
            return
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def close(self) -> None:
        """Release the model and any device memory it holds."""
        self._model = None
        self._warm = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            logger.debug("could not empty the CUDA cache on close: %s", exc)

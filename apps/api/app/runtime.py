"""Inference runtime: the single owner of the loaded model.

Everything that mutates model state -- startup selection, hot-swap, and automatic
degradation -- goes through here under one lock. Routes and the streaming loop
never touch a :class:`~app.detector.Detector` directly, which is what makes a
mid-stream precision switch safe.

Degradation policy, applied on CUDA OOM:

1. If running above the degraded resolution, reload the same backend at 480 px
   (roughly 40 % less activation memory).
2. Otherwise step to the next, cheaper backend on the ladder.
3. If the ladder is exhausted, the service is unrecoverable and says so.

Each step sets ``degraded_mode`` with a human-readable reason, which the UI shows
rather than silently serving worse results.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
import uuid

import numpy as np

from .backends import BACKENDS, Backend, availability, candidate_chain, get_backend
from .config import GpuProbe, Settings, probe_gpu, resolve_start_imgsz
from .detector import Detector, OutOfVramError, is_oom
from .exceptions import BackendUnavailableError, NoBackendError
from .metrics import MetricsCollector
from .models import (
    BackendInfo,
    Detection,
    FrameTiming,
    MetricsResponse,
    ModelConfigResponse,
    Track,
)
from .store import DetectionStore

logger = logging.getLogger(__name__)

SUPPORTED_IMGSZ: tuple[int, ...] = (640, 480, 320)


def _short(exc: BaseException, limit: int = 200) -> str:
    """One-line exception text; backend errors can be paragraphs of driver detail."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else f"{text[:limit]}..."


class InferenceRuntime:
    """Owns the active detector, the metrics collector, and the telemetry store."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._detector: Detector | None = None
        self._gpu: GpuProbe = GpuProbe(available=False)
        self._frame_counter = 0
        self._session_id = uuid.uuid4().hex[:12]
        # Configurations whose artifact exists but which proved unrunnable here.
        # Static availability cannot detect these -- a missing cuDNN or a driver
        # mismatch only surfaces on execution -- so the first failure is
        # remembered and reported instead of the UI offering a dead option.
        #
        # Keyed by (backend, imgsz), not by backend alone: exported artifacts
        # carry a fixed input shape, so a 640 px ONNX genuinely cannot serve a
        # 480 px request. Blacklisting the whole backend for that would discard a
        # perfectly good configuration.
        self._unusable: dict[tuple[str, int], str] = {}
        self.metrics = MetricsCollector(settings)
        self.store = DetectionStore(settings.db_path)

    # ------------------------------------------------------------------ setup

    def startup(self) -> None:
        """Probe the host, load the best runnable backend, and warm it up.

        Raises:
            NoBackendError: nothing on the ladder can run here.
        """
        self._gpu = probe_gpu()
        logger.info("gpu probe: %s", self._gpu.summary)
        self.store.start()

        chain = candidate_chain(
            self._settings.default_precision, self._settings, self._gpu.available
        )
        if not chain:
            raise NoBackendError(
                "no inference backend available - export an engine or place "
                f"yolo11n.pt in {self._settings.weights_dir}"
            )

        detector = self._load_first_working(chain, self._settings.default_imgsz)
        if detector is None:
            raise NoBackendError("every candidate backend failed to load")

        # Resolution policy is decided from headroom measured *after* the model and
        # its runtime workspace are resident -- before frame buffers exist.
        after_load = probe_gpu()
        imgsz = resolve_start_imgsz(self._settings, after_load.free_mb, self._gpu.available)
        if imgsz != detector.imgsz:
            logger.info("re-loading at %d px based on %d MiB free", imgsz, after_load.free_mb)
            detector.close()
            detector = self._load_first_working(chain, imgsz)
            if detector is None:
                raise NoBackendError("backend failed to load at reduced resolution")
            self.metrics.set_degraded(True, f"low VRAM headroom ({after_load.free_mb} MiB free)")

        self._detector = detector
        self.metrics.refresh_gpu()
        logger.info("runtime ready: %s @ %d px", detector.backend.key, detector.imgsz)

    def shutdown(self) -> None:
        with self._lock:
            if self._detector is not None:
                self._detector.close()
                self._detector = None
        self.store.stop()

    def _load_first_working(
        self, chain: list[Backend], imgsz: int, warmup_frames: int = 2
    ) -> Detector | None:
        """Try each backend in order; return the first that loads *and runs*.

        Loading is not proof of usability, so warmup happens here and a backend
        that throws during it is rejected exactly like one that failed to load.
        """
        for backend in chain:
            if (backend.key, imgsz) in self._unusable or not backend.supports_imgsz(imgsz):
                continue
            detector = Detector(backend, imgsz, self._settings)
            try:
                detector.load()
                detector.warmup(frames=warmup_frames)
            except Exception as exc:  # noqa: BLE001 - any failure means "try the next one"
                reason = _short(exc)
                logger.warning("backend %s at %d px unusable: %s", backend.key, imgsz, reason)
                # Only *capability* failures are permanent. Running out of memory
                # says something about the machine at this instant, not about the
                # backend, so blacklisting on OOM would retire a perfectly good
                # configuration because one frame arrived at a bad moment.
                if not is_oom(exc):
                    self._unusable[(backend.key, imgsz)] = reason
                detector.close()
                continue
            return detector
        return None

    # -------------------------------------------------------------- inference

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @property
    def precision(self) -> str:
        return self._detector.backend.key if self._detector else "none"

    @property
    def imgsz(self) -> int:
        return self._detector.imgsz if self._detector else 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self) -> str:
        """Start a new logging session and clear tracker identities."""
        with self._lock:
            self._session_id = uuid.uuid4().hex[:12]
            if self._detector is not None:
                self._detector.reset_tracker()
            self.metrics.reset_stream()
        return self._session_id

    def process(self, frame: np.ndarray) -> tuple[list[Detection], list[Track], FrameTiming, int]:
        """Run one frame through detect + track, degrading on OOM.

        Returns:
            ``(detections, tracks, timing, frame_id)``.

        Raises:
            NoBackendError: no detector is loaded, or degradation exhausted the ladder.
        """
        started = time.perf_counter()
        with self._lock:
            detector = self._detector
            if detector is None:
                raise NoBackendError("inference runtime is not ready")
            try:
                detections, tracks, inference_ms = detector.track(frame)
            except OutOfVramError as exc:
                logger.error("out of VRAM on %s: %s", detector.backend.key, exc)
                self._degrade(str(exc))
                detector = self._detector
                if detector is None:
                    raise NoBackendError("out of VRAM and no fallback remains") from exc
                detections, tracks, inference_ms = detector.track(frame)

            self._frame_counter += 1
            frame_id = self._frame_counter
            precision = detector.backend.key
            imgsz = detector.imgsz
            if self._frame_counter % self._settings.gc_interval_frames == 0:
                gc.collect()

        total_ms = (time.perf_counter() - started) * 1000.0
        timing = FrameTiming(
            inference_ms=round(inference_ms, 2),
            total_ms=round(total_ms, 2),
        )
        self.metrics.record_frame(total_ms, [t.track_id for t in tracks])
        self.store.record(
            session_id=self._session_id,
            frame_id=frame_id,
            precision=precision,
            imgsz=imgsz,
            latency_ms=total_ms,
            detections=len(detections),
            tracks=tracks,
        )
        return detections, tracks, timing, frame_id

    # ------------------------------------------------------------ degradation

    def _degrade(self, reason: str) -> None:
        """Step down one rung of the ladder. Caller must hold the lock.

        The current detector is kept alive until a replacement has loaded *and*
        warmed up. Releasing it first is the obvious ordering and it is wrong: if
        every remaining rung fails, the service is left with no detector at all
        and every route -- including the one that would let an operator fix it --
        starts returning 503 with no way back short of a restart.

        The cost of holding both briefly is one extra model in memory for a few
        seconds, which the 4 GB budget absorbs (the model is ~130 MiB resident).
        """
        current = self._detector
        if current is None:
            return
        backend = current.backend
        imgsz = current.imgsz

        if imgsz > self._settings.degraded_imgsz:
            target_imgsz = self._settings.degraded_imgsz
            chain = [backend]
            note = f"reduced to {target_imgsz} px after VRAM exhaustion"
        else:
            target_imgsz = imgsz
            full = candidate_chain(None, self._settings, self._gpu.available)
            keys = [b.key for b in full]
            # Keep only rungs strictly cheaper than the one that just failed.
            cutoff = keys.index(backend.key) + 1 if backend.key in keys else 0
            chain = full[cutoff:]
            note = f"fell back from {backend.key} after VRAM exhaustion"

        # An OOM is by definition a memory-pressure event, so free the old model
        # before trying to load a new one -- but remember enough to put it back.
        current.close()
        self._detector = None
        gc.collect()

        detector = self._load_first_working(chain, target_imgsz, warmup_frames=1)
        if detector is None and not any(b.key == "fp32_cpu" for b in chain):
            # Last resort: the CPU path needs no GPU memory and no export step.
            detector = self._load_first_working(
                [get_backend("fp32_cpu")], target_imgsz, warmup_frames=1
            )
        if detector is None:
            # Nothing cheaper works. Restore what was running rather than leaving
            # the service dead: it was serving traffic a moment ago.
            restored = self._load_first_working([backend], imgsz, warmup_frames=1)
            if restored is not None:
                self._detector = restored
                self.metrics.set_degraded(
                    True, f"{note}; ladder exhausted, kept {backend.key} ({reason})"
                )
                logger.error("degradation ladder exhausted; restored %s", backend.key)
            else:
                self.metrics.set_degraded(True, f"{note}; no fallback loaded ({reason})")
                logger.error("degradation ladder exhausted and %s did not reload", backend.key)
            return
        self._detector = detector
        self.metrics.set_degraded(True, note)
        logger.warning(
            "degraded: %s -> %s @ %d px", backend.key, detector.backend.key, target_imgsz
        )

    def simulate_oom(self) -> None:
        """Force one degradation step. Used by tests and the /settings drill.

        Exposed deliberately: the auto-degrade path is a headline reliability
        claim, and a claim that cannot be exercised on demand cannot be trusted.
        """
        with self._lock:
            self._degrade("manually triggered degradation drill")

    # ------------------------------------------------------------- hot swap

    def switch(self, precision: str | None, imgsz: int | None) -> ModelConfigResponse:
        """Rebuild the detector with a new precision and/or resolution.

        Raises:
            BackendUnavailableError: the requested combination cannot run here.
        """
        with self._lock:
            current = self._detector
            target_precision = precision or (current.backend.key if current else "auto")
            target_imgsz = imgsz or (current.imgsz if current else self._settings.default_imgsz)

            if target_imgsz not in SUPPORTED_IMGSZ:
                raise BackendUnavailableError(
                    f"unsupported resolution {target_imgsz}; choose one of {list(SUPPORTED_IMGSZ)}"
                )
            if target_precision not in BACKENDS:
                raise BackendUnavailableError(f"unknown precision '{target_precision}'")

            backend = get_backend(target_precision)
            runnable, why = availability(backend, self._settings, self._gpu.available)
            if not runnable:
                raise BackendUnavailableError(f"{backend.label} unavailable: {why}")
            if not backend.supports_imgsz(target_imgsz):
                raise BackendUnavailableError(
                    f"{backend.label} was exported at {backend.export_imgsz} px and cannot run "
                    f"at {target_imgsz} px; re-export it or use a PyTorch backend"
                )
            previous_failure = self._unusable.get((target_precision, target_imgsz))
            if previous_failure:
                raise BackendUnavailableError(
                    f"{backend.label} already failed at {target_imgsz} px: {previous_failure}"
                )

            if current is not None:
                current.close()
                self._detector = None
                gc.collect()

            detector = self._load_first_working([backend], target_imgsz, warmup_frames=1)
            if detector is None:
                # Restoring service beats honouring the request: reload what worked.
                fallback = candidate_chain(None, self._settings, self._gpu.available)
                self._detector = self._load_first_working(fallback, target_imgsz, warmup_frames=1)
                raise BackendUnavailableError(
                    f"{backend.label} loaded but could not run on this host"
                )

            self._detector = detector
            self.metrics.set_degraded(False)
            self.metrics.reset_stream()
            logger.info("switched to %s @ %d px", detector.backend.key, detector.imgsz)
            return self.config_response()

    # ---------------------------------------------------------------- reporting

    def config_response(self) -> ModelConfigResponse:
        detector = self._detector
        backends: list[BackendInfo] = []
        active_imgsz = detector.imgsz if detector else self._settings.default_imgsz
        for backend in BACKENDS.values():
            runnable, why = availability(backend, self._settings, self._gpu.available)
            failure = self._unusable.get((backend.key, active_imgsz))
            if runnable and failure:
                runnable, why = False, f"failed at {active_imgsz} px: {failure}"
            backends.append(
                BackendInfo(
                    precision=backend.key,
                    label=backend.label,
                    description=backend.description,
                    device=backend.device,
                    available=runnable,
                    reason=why,
                    artifact=backend.artifact,
                )
            )
        return ModelConfigResponse(
            precision=detector.backend.key if detector else "none",
            imgsz=detector.imgsz if detector else 0,
            device=detector.device if detector else "none",
            model_file=detector.model_file if detector else "",
            degraded_mode=self.metrics.degraded,
            degrade_reason=self.metrics.degrade_reason,
            available_backends=backends,
            supported_imgsz=list(SUPPORTED_IMGSZ),
        )

    def metrics_response(self) -> MetricsResponse:
        return self.metrics.snapshot(self.precision, self.imgsz)

    def gpu_probe(self) -> GpuProbe:
        return self._gpu

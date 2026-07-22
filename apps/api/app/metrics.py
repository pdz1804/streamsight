"""Rolling telemetry for the dashboard and the stability soak.

Everything is bounded: fixed-length deques for the time series and a capped set
for unique track ids, so a multi-hour stream cannot grow this collector without
limit. That matters -- NFR-6 asks for a 4 h run with no memory creep, and an
unbounded metrics buffer would be the first thing to break it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from statistics import fmean

import psutil

from .config import GpuProbe, Settings, probe_gpu
from .models import GpuInfo, MetricsResponse

#: Number of recent frames used for the rolling FPS/latency figures.
WINDOW = 120
#: Cap on remembered track ids; ids are monotonic so the count stays informative.
UNIQUE_TRACK_CAP = 10_000


class MetricsCollector:
    """Thread-safe aggregator fed by every processed frame."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self._latencies: deque[float] = deque(maxlen=WINDOW)
        self._frame_times: deque[float] = deque(maxlen=WINDOW)
        self._fps_series: deque[float] = deque(maxlen=60)
        self._frames = 0
        self._track_count = 0
        self._unique_tracks: set[int] = set()
        self._unique_overflow = 0
        self._degraded = False
        self._degrade_reason: str | None = None
        self._gpu: GpuProbe = probe_gpu()
        self._process = psutil.Process()
        # Prime psutil's CPU sampler so the first read is not a meaningless 0.0.
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    # ---------------------------------------------------------------- record

    def record_frame(self, latency_ms: float, track_ids: list[int | None]) -> None:
        """Register one completed frame."""
        now = time.perf_counter()
        with self._lock:
            self._frames += 1
            self._latencies.append(latency_ms)
            self._frame_times.append(now)
            self._track_count = len(track_ids)
            for track_id in track_ids:
                if track_id is None:
                    continue
                if len(self._unique_tracks) >= UNIQUE_TRACK_CAP:
                    self._unique_overflow += 1
                    continue
                self._unique_tracks.add(track_id)
            if self._frames % self._settings.gpu_poll_interval_frames == 0:
                self._gpu = probe_gpu()
            self._fps_series.append(self._fps_locked())

    def set_degraded(self, degraded: bool, reason: str | None = None) -> None:
        with self._lock:
            self._degraded = degraded
            self._degrade_reason = reason if degraded else None

    def reset_stream(self) -> None:
        """Clear per-stream counters while keeping process-lifetime totals."""
        with self._lock:
            self._latencies.clear()
            self._frame_times.clear()
            self._fps_series.clear()
            self._track_count = 0

    def refresh_gpu(self) -> None:
        with self._lock:
            self._gpu = probe_gpu()

    # ----------------------------------------------------------------- query

    def current_fps(self) -> float:
        """Just the rolling FPS.

        The streaming loop needs this per frame, and building a full
        :class:`MetricsResponse` for it would run several psutil syscalls at the
        frame rate -- measurably more expensive than the inference it reports on.
        """
        with self._lock:
            return round(self._fps_locked(), 2)

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def degrade_reason(self) -> str | None:
        return self._degrade_reason

    @property
    def frames_processed(self) -> int:
        return self._frames

    def gpu_info(self) -> GpuInfo:
        with self._lock:
            gpu = self._gpu
        return GpuInfo(
            available=gpu.available,
            name=gpu.name,
            total_mb=gpu.total_mb,
            used_mb=gpu.used_mb,
            free_mb=gpu.free_mb,
        )

    def snapshot(self, precision: str, imgsz: int) -> MetricsResponse:
        """Build the dashboard payload."""
        with self._lock:
            fps = self._fps_locked()
            latencies = sorted(self._latencies)
            avg = fmean(latencies) if latencies else 0.0
            p50 = _percentile(latencies, 0.50)
            p95 = _percentile(latencies, 0.95)
            series = list(self._fps_series)
            frames = self._frames
            track_count = self._track_count
            unique = len(self._unique_tracks) + self._unique_overflow
            degraded = self._degraded
            reason = self._degrade_reason
        gpu = self.gpu_info()
        virtual = psutil.virtual_memory()
        return MetricsResponse(
            fps=round(fps, 2),
            fps_rolling=[round(v, 2) for v in series],
            avg_latency_ms=round(avg, 2),
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            frames_processed=frames,
            track_count=track_count,
            unique_tracks=unique,
            gpu=gpu,
            cpu_percent=round(psutil.cpu_percent(interval=None), 1),
            ram_used_mb=int(virtual.used // 1024**2),
            process_ram_mb=int(self._process.memory_info().rss // 1024**2),
            degraded_mode=degraded,
            degrade_reason=reason,
            precision=precision,
            imgsz=imgsz,
            uptime_s=round(time.perf_counter() - self._started, 1),
        )

    # -------------------------------------------------------------- internal

    def _fps_locked(self) -> float:
        """Frames per second over the current window. Caller must hold the lock.

        Measured from wall-clock arrival times rather than 1000/latency so that
        queueing and encode overhead are included -- this is the number a user
        actually perceives.
        """
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]

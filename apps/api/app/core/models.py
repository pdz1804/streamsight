"""Pydantic schemas shared by every API surface.

These types are the public contract of the service: the Next.js client mirrors
them in ``apps/web/lib/types.ts``, and the measurement harnesses in ``ml/eval``
consumes the same field names. Change them deliberately.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

BBox = tuple[float, float, float, float]

SourceKind = Literal["file", "webcam", "rtsp", "sample"]


class Detection(BaseModel):
    """One detected object in image pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int
    class_name: str

    @property
    def box(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)


class Track(Detection):
    """A detection carrying a persistent ByteTrack identity.

    ``track_id`` is ``None`` for detections the tracker has not yet confirmed;
    the UI renders those without an ID badge rather than inventing one.
    """

    track_id: int | None = None


class FrameTiming(BaseModel):
    """Per-stage wall-clock breakdown, milliseconds.

    ``wait_ms`` and ``send_ms`` cover the stages *outside* inference. They exist
    because a frame period and the sum of the inference stages disagreed by more
    than half the budget, and no field accounted for the difference -- so the
    cost could be argued about but not measured. They are reported separately
    from ``total_ms``, which remains the server-side processing cost, so the
    published latency figure keeps its original meaning.

    ``total_ms`` is the work spent on *this* frame, not the interval between
    frames. Those were never the same number, and since capture+inference and
    encode+send run as overlapping stages they are not even the same shape:
    throughput is set by the slowest stage, while ``total_ms`` is their sum.
    Read ``fps`` for rate; do not derive it from this.
    """

    decode_ms: float = 0.0
    inference_ms: float = 0.0
    encode_ms: float = 0.0
    total_ms: float = 0.0
    #: Time the pump blocked waiting for a frame from the capture ring buffer.
    wait_ms: float = 0.0
    #: Time to serialize and hand a frame to the socket. Reports the *previous*
    #: frame's cost: a frame is already serialized by the time its own send
    #: finishes, so it can never carry that figure itself.
    send_ms: float = 0.0


class FrameResponse(BaseModel):
    """Result of running one frame through detect + track."""

    frame_id: int
    width: int
    height: int
    detections: list[Detection]
    tracks: list[Track]
    timing: FrameTiming
    fps: float
    precision: str
    imgsz: int
    degraded_mode: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def latency_ms(self) -> float:
        """Flat mirror of ``timing.total_ms``.

        Derived rather than stored so the two can never disagree: the nested
        breakdown stays the single source of truth while callers that only want
        one number do not have to reach into it.
        """
        return self.timing.total_ms


class StreamFrame(BaseModel):
    """WebSocket payload: annotated frame metadata, and optionally the pixels.

    Two transports carry this model, and ``image`` is what distinguishes them.

    * **binary** (default): the pixels travel as raw JPEG bytes appended after
      this object in one length-prefixed message (see :mod:`app.streaming.wire`), so
      ``image`` is ``None``.
    * **base64**: ``image`` holds a ``data:image/jpeg;base64,...`` URI and the
      whole thing ships as one JSON text frame.

    Either way the pixels and the boxes arrive together in a single message, so
    the client never has to correlate them.
    """

    kind: Literal["frame"] = "frame"
    frame_id: int
    image: str | None = None
    width: int
    height: int
    tracks: list[Track]
    timing: FrameTiming
    fps: float
    server_ts: float
    precision: str
    imgsz: int
    degraded_mode: bool


class StreamStatus(BaseModel):
    """WebSocket control message (lifecycle + error reporting)."""

    kind: Literal["status"] = "status"
    phase: Literal["opening", "streaming", "ended", "error"]
    message: str = ""
    source: str = ""
    total_frames: int | None = None


class GpuInfo(BaseModel):
    available: bool
    name: str
    total_mb: int
    used_mb: int
    free_mb: int


class MetricsResponse(BaseModel):
    """Snapshot consumed by the /metrics dashboard, polled ~1 Hz."""

    fps: float
    fps_rolling: list[float]
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    frames_processed: int
    track_count: int
    unique_tracks: int
    gpu: GpuInfo
    cpu_percent: float
    ram_used_mb: int
    process_ram_mb: int
    degraded_mode: bool
    degrade_reason: str | None
    precision: str
    imgsz: int
    uptime_s: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gpu_mem_mb(self) -> int:
        """Flat mirror of ``gpu.used_mb`` for dashboards that want a single gauge.

        Derived from the nested block for the same reason as ``latency_ms``:
        one owner of the value, no chance of the two drifting apart.
        """
        return self.gpu.used_mb


class BackendInfo(BaseModel):
    """One selectable inference backend and whether it can run here."""

    precision: str
    label: str
    description: str
    device: str
    available: bool
    reason: str = ""
    artifact: str = ""


class ModelConfigResponse(BaseModel):
    """Active inference configuration plus the full selectable menu."""

    # `model_file` collides with Pydantic's reserved `model_` prefix; the field
    # name is part of the public API contract, so the guard is relaxed instead.
    model_config = ConfigDict(protected_namespaces=())

    precision: str
    imgsz: int
    device: str
    model_file: str
    degraded_mode: bool
    degrade_reason: str | None
    available_backends: list[BackendInfo]
    supported_imgsz: list[int]


class ModelConfigRequest(BaseModel):
    """Hot-swap request. Omitted fields keep their current value.

    ``resolution`` is an accepted synonym for ``imgsz`` so the endpoint speaks the
    vocabulary of the product spec as well as the one the codebase and the web
    client already use. ``extra="forbid"`` is kept: a typo must still 422 rather
    than silently do nothing.

    ``precision`` accepts both concrete backend keys (``fp32_gpu``) and the
    spec's abstract words (``int8``/``fp16``/``fp32``); the abstract form is
    resolved against what this host can actually run, in
    :mod:`app.inference.runtime`.
    """

    model_config = ConfigDict(extra="forbid")

    precision: str | None = None
    # Constrained positive: without `gt=0`, a `0` is falsy and the runtime's
    # "keep the current value" fallback swallows it, returning 200 having changed
    # nothing -- while an out-of-range value like 123 correctly 409s. Same class
    # of mistake, two different outcomes, is worse than either.
    imgsz: int | None = Field(default=None, gt=0)
    resolution: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _fold_resolution_into_imgsz(self) -> ModelConfigRequest:
        """Collapse the alias so downstream code only ever reads ``imgsz``.

        Sending both with different values is a contradiction the caller has to
        resolve, not something to guess at, so it is refused.
        """
        if self.resolution is None:
            return self
        if self.imgsz is not None and self.imgsz != self.resolution:
            raise ValueError(
                f"imgsz ({self.imgsz}) and resolution ({self.resolution}) disagree; send one"
            )
        self.imgsz = self.resolution
        return self


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    version: str
    gpu: GpuInfo
    precision: str
    imgsz: int


class SourceInfo(BaseModel):
    """A video source the viewer can stream from."""

    id: str
    kind: SourceKind
    label: str
    detail: str = ""

/**
 * Client mirror of the FastAPI schemas in `apps/api/app/models.py`.
 * Keep the two files in step: these are the service's public contract.
 */

export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface Track extends Detection {
  track_id: number | null;
}

export interface FrameTiming {
  decode_ms: number;
  inference_ms: number;
  encode_ms: number;
  total_ms: number;
}

export interface FrameResponse {
  frame_id: number;
  width: number;
  height: number;
  detections: Detection[];
  tracks: Track[];
  timing: FrameTiming;
  fps: number;
  precision: string;
  imgsz: number;
  degraded_mode: boolean;
}

export interface StreamFrame {
  kind: "frame";
  frame_id: number;
  image: string;
  width: number;
  height: number;
  tracks: Track[];
  timing: FrameTiming;
  fps: number;
  server_ts: number;
  precision: string;
  imgsz: number;
  degraded_mode: boolean;
}

export type StreamPhase = "opening" | "streaming" | "ended" | "error";

export interface StreamStatus {
  kind: "status";
  phase: StreamPhase;
  message: string;
  source: string;
  total_frames: number | null;
}

export type StreamMessage = StreamFrame | StreamStatus;

export interface GpuInfo {
  available: boolean;
  name: string;
  total_mb: number;
  used_mb: number;
  free_mb: number;
}

export interface MetricsResponse {
  fps: number;
  fps_rolling: number[];
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  frames_processed: number;
  track_count: number;
  unique_tracks: number;
  gpu: GpuInfo;
  cpu_percent: number;
  ram_used_mb: number;
  process_ram_mb: number;
  degraded_mode: boolean;
  degrade_reason: string | null;
  precision: string;
  imgsz: number;
  uptime_s: number;
}

export interface BackendInfo {
  precision: string;
  label: string;
  description: string;
  device: string;
  available: boolean;
  reason: string;
  artifact: string;
}

export interface ModelConfigResponse {
  precision: string;
  imgsz: number;
  device: string;
  model_file: string;
  degraded_mode: boolean;
  degrade_reason: string | null;
  available_backends: BackendInfo[];
  supported_imgsz: number[];
}

export interface HealthResponse {
  status: "ok";
  app: string;
  version: string;
  gpu: GpuInfo;
  precision: string;
  imgsz: number;
}

export type SourceKind = "file" | "webcam" | "rtsp" | "sample";

export interface SourceInfo {
  id: string;
  kind: SourceKind;
  label: string;
  detail: string;
}

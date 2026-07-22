/**
 * Typed HTTP client for the StreamSight API.
 *
 * Every call funnels through `request` so error shapes are handled once: the API
 * returns `{error, detail}` for domain failures, and surfacing `detail` verbatim
 * is what lets the UI explain *why* a backend is unavailable instead of showing
 * a generic failure.
 */

import type {
  FrameResponse,
  HealthResponse,
  MetricsResponse,
  ModelConfigResponse,
  SourceInfo,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8100";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Is uvicorn running on port 8100?`,
      0,
    );
  }

  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as T;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join(", ");
    return JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  metrics: () => request<MetricsResponse>("/metrics"),

  modelConfig: () => request<ModelConfigResponse>("/config/model"),

  /**
   * Hot-swap the model. `precision` accepts a concrete backend key (`fp32_gpu`)
   * or an abstract word (`int8` | `fp16` | `fp32`), which the server resolves
   * against what this host can actually run. `resolution` is a server-side
   * synonym for `imgsz`; send one or the other, never both with different values.
   */
  setModelConfig: (payload: { precision?: string; imgsz?: number; resolution?: number }) =>
    request<ModelConfigResponse>("/config/model", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  forceDegrade: () => request<ModelConfigResponse>("/config/degrade", { method: "POST" }),

  sources: () => request<SourceInfo[]>("/sources"),

  uploadSource: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<SourceInfo>("/sources/upload", { method: "POST", body: form });
  },

  detectImage: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<FrameResponse>("/detect/image", { method: "POST", body: form });
  },
};

/** WebSocket URL for the annotated live stream. */
export function streamUrl(source: string, loop = true): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/detect/stream?source=${encodeURIComponent(source)}&loop=${loop}`;
}

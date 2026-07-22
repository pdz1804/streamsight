"use client";

/**
 * Telemetry dashboard, polled once a second.
 *
 * Polling beats a second WebSocket here: these numbers are already aggregated
 * server-side, a stale reading is harmless, and a plain interval survives the
 * API restarting under it without any reconnect logic.
 */

import { useEffect, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";

import { Sparkline } from "./sparkline";
import { Chip, ErrorNote, Panel, Skeleton, Stat } from "./ui";

const POLL_MS = 1000;
const HISTORY = 90;
const FPS_TARGET = 30;
const VRAM_BUDGET_MB = 3500;

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  const [vramHistory, setVramHistory] = useState<number[]>([]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const snapshot = await api.metrics();
        if (cancelled) return;
        setMetrics(snapshot);
        setError(null);
        setFpsHistory((h) => [...h, snapshot.fps].slice(-HISTORY));
        setLatencyHistory((h) => [...h, snapshot.avg_latency_ms].slice(-HISTORY));
        setVramHistory((h) => [...h, snapshot.gpu.used_mb].slice(-HISTORY));
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof ApiError ? cause.message : String(cause));
      }
    };
    void poll();
    const timer = window.setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  if (error && !metrics) {
    return (
      <div className="p-4 md:p-6">
        <ErrorNote>{error}</ErrorNote>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="grid gap-4 p-4 md:grid-cols-2 md:p-6 xl:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <Skeleton key={index} className="h-52" />
        ))}
      </div>
    );
  }

  const vramPercent = metrics.gpu.total_mb
    ? Math.round((metrics.gpu.used_mb / metrics.gpu.total_mb) * 100)
    : 0;

  return (
    <div className="flex flex-col gap-4 p-4 md:p-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-[15px] font-semibold text-text">Runtime telemetry</h1>
        <Chip tone={metrics.degraded_mode ? "warn" : "ok"}>
          {metrics.degraded_mode ? "Degraded" : "Nominal"}
        </Chip>
        <span className="font-mono text-[11px] text-text-mute">
          {metrics.precision} · {metrics.imgsz}px · up {formatUptime(metrics.uptime_s)}
        </span>
      </div>

      {metrics.degrade_reason ? (
        <p className="rounded-[var(--radius)] border border-warn/40 bg-warn/8 px-3 py-2 text-[12px] text-warn">
          {metrics.degrade_reason}
        </p>
      ) : null}
      {error ? <ErrorNote>{error}</ErrorNote> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Panel title="Frames per second">
          <div className="divide-y divide-line">
            <Stat
              label="Current"
              value={metrics.fps.toFixed(1)}
              tone={metrics.fps >= FPS_TARGET ? "ok" : metrics.fps > 0 ? "warn" : "neutral"}
            />
            <Stat label="Frames processed" value={metrics.frames_processed.toLocaleString()} />
          </div>
          <div className="border-t border-line px-3 pb-3 pt-3">
            <Sparkline values={fpsHistory} label="Frames per second" floorAt={FPS_TARGET} />
          </div>
        </Panel>

        <Panel title="Latency">
          <div className="divide-y divide-line">
            <Stat label="Mean" value={metrics.avg_latency_ms.toFixed(1)} unit="ms" />
            <Stat label="p50" value={metrics.p50_latency_ms.toFixed(1)} unit="ms" />
            <Stat
              label="p95"
              value={metrics.p95_latency_ms.toFixed(1)}
              unit="ms"
              tone={metrics.p95_latency_ms > 100 ? "warn" : "ok"}
            />
          </div>
          <div className="border-t border-line px-3 pb-3 pt-3">
            <Sparkline values={latencyHistory} label="Mean latency" color="var(--warn)" />
          </div>
        </Panel>

        <Panel title="GPU memory">
          {metrics.gpu.available ? (
            <>
              <div className="divide-y divide-line">
                <Stat
                  label="Used"
                  value={metrics.gpu.used_mb.toLocaleString()}
                  unit="MiB"
                  tone={metrics.gpu.used_mb > VRAM_BUDGET_MB ? "danger" : "ok"}
                />
                <Stat label="Free" value={metrics.gpu.free_mb.toLocaleString()} unit="MiB" />
                <Stat label="Device total" value={metrics.gpu.total_mb.toLocaleString()} unit="MiB" />
              </div>
              <div className="border-t border-line px-3 pb-3 pt-3">
                <Sparkline
                  values={vramHistory}
                  label="GPU memory used"
                  color="var(--ok)"
                  floorAt={VRAM_BUDGET_MB}
                />
                <p className="mt-1 px-1 text-[10px] text-text-mute">
                  {vramPercent}% of {metrics.gpu.name}, dashed line is the {VRAM_BUDGET_MB} MiB budget
                </p>
              </div>
            </>
          ) : (
            <p className="px-4 py-8 text-center text-[12px] text-text-mute">
              No NVIDIA device on this host. Inference is running on the CPU path.
            </p>
          )}
        </Panel>

        <Panel title="Tracking">
          <div className="divide-y divide-line">
            <Stat label="Active tracks" value={metrics.track_count} />
            <Stat
              label="Unique ids"
              value={metrics.unique_tracks.toLocaleString()}
              hint="Distinct ByteTrack identities seen since the process started."
            />
          </div>
        </Panel>

        <Panel title="Host">
          <div className="divide-y divide-line">
            <Stat label="CPU" value={metrics.cpu_percent.toFixed(0)} unit="%" />
            <Stat label="System RAM" value={metrics.ram_used_mb.toLocaleString()} unit="MiB" />
            <Stat label="API process" value={metrics.process_ram_mb.toLocaleString()} unit="MiB" />
          </div>
        </Panel>
      </div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

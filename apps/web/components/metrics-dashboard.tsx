"use client";

/**
 * Telemetry dashboard, polled once a second.
 *
 * Polling beats a second WebSocket here: these numbers are already aggregated
 * server-side, a stale reading is harmless, and a plain interval survives the
 * API restarting under it without any reconnect logic.
 *
 * Layout: three headline KPIs across the top carrying the numbers an operator
 * actually watches, then supporting detail below. The previous version rendered
 * five equal cards in a uniform grid, which gave the eye nowhere to land and
 * left the lower half of the viewport empty.
 */

import { useEffect, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";

import { Sparkline } from "./sparkline";
import { Chip, ErrorNote, KpiTile, Panel, Skeleton, Stat } from "./ui";

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
      <div className="mx-auto max-w-[1500px] p-5 md:p-7">
        <ErrorNote>{error}</ErrorNote>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="mx-auto max-w-[1500px] p-5 md:p-7">
        <div className="grid gap-5 md:grid-cols-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-44" />
          ))}
        </div>
        <div className="mt-5 grid gap-5 lg:grid-cols-[1.4fr_1fr]">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </div>
    );
  }

  const vramPercent = metrics.gpu.total_mb
    ? (metrics.gpu.used_mb / metrics.gpu.total_mb) * 100
    : 0;

  // Nothing has been measured until frames have flowed. Colouring a zero as
  // "good" would report an idle service as a healthy one.
  const idle = metrics.frames_processed === 0 || metrics.fps === 0;
  const fpsTone = idle ? "neutral" : metrics.fps >= FPS_TARGET ? "ok" : "warn";
  const latencyTone = idle ? "neutral" : metrics.p95_latency_ms > 100 ? "warn" : "ok";

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-5 p-5 md:p-7">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <h1 className="text-[22px] font-semibold tracking-tight text-text">Runtime telemetry</h1>
        <Chip tone={metrics.degraded_mode ? "warn" : "ok"} live={!metrics.degraded_mode}>
          {metrics.degraded_mode ? "Degraded" : "Nominal"}
        </Chip>
        <span className="font-mono text-[11px] text-text-mute">
          {metrics.precision} · {metrics.imgsz}px · up {formatUptime(metrics.uptime_s)}
        </span>
      </header>

      {metrics.degrade_reason ? (
        <p className="rounded-[var(--radius)] border border-warn/35 bg-warn/8 px-3.5 py-2.5 text-[12px] text-warn">
          {metrics.degrade_reason}
        </p>
      ) : null}
      {error ? <ErrorNote>{error}</ErrorNote> : null}

      {/* Headline figures. These are what someone glances at from across a desk. */}
      <div className="grid gap-5 md:grid-cols-3">
        <KpiTile
          index={0}
          label="Throughput"
          value={metrics.fps.toFixed(1)}
          unit="fps"
          tone={fpsTone}
          caption={`${metrics.frames_processed.toLocaleString()} frames processed`}
          chart={
            <Sparkline
              values={fpsHistory}
              label="Frames per second"
              floorAt={FPS_TARGET}
              height={64}
            />
          }
        />
        <KpiTile
          index={1}
          label="Latency p95"
          value={metrics.p95_latency_ms.toFixed(0)}
          unit="ms"
          tone={latencyTone}
          caption={`${metrics.avg_latency_ms.toFixed(1)} ms mean · ${metrics.p50_latency_ms.toFixed(1)} ms median`}
          chart={
            <Sparkline
              values={latencyHistory}
              label="Mean latency"
              color="var(--warn)"
              height={64}
            />
          }
        />
        <KpiTile
          index={2}
          label="GPU memory"
          value={metrics.gpu.available ? metrics.gpu.used_mb.toLocaleString() : "n/a"}
          unit={metrics.gpu.available ? "MiB" : undefined}
          tone={metrics.gpu.used_mb > VRAM_BUDGET_MB ? "danger" : "ok"}
          caption={
            metrics.gpu.available
              ? `${vramPercent.toFixed(0)}% of ${metrics.gpu.total_mb.toLocaleString()} MiB · budget ${VRAM_BUDGET_MB.toLocaleString()}`
              : "No NVIDIA device; running the CPU path"
          }
          chart={
            metrics.gpu.available ? (
              <Sparkline
                values={vramHistory}
                label="GPU memory used"
                color="var(--ok)"
                floorAt={VRAM_BUDGET_MB}
                height={64}
              />
            ) : undefined
          }
        />
      </div>

      {/* Supporting detail, deliberately not the same shape as the row above. */}
      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        <Panel index={3} title="Tracking">
          <div className="flex flex-col gap-1 px-2 pb-4">
            <Stat label="Active tracks in frame" value={metrics.track_count} />
            <Stat
              label="Unique identities seen"
              value={metrics.unique_tracks.toLocaleString()}
              hint="Distinct ByteTrack ids since the process started."
            />
            <Stat label="Frames processed" value={metrics.frames_processed.toLocaleString()} />
          </div>
          <div className="mx-5 mb-5 mt-1 rounded-[var(--radius)] border border-line bg-surface-2 px-4 py-3">
            <p className="text-pretty text-[12px] leading-relaxed text-text-dim">
              Identities persist while ByteTrack keeps matching an object across frames. A
              rising unique count with a steady active count means objects are entering and
              leaving, not that tracking is dropping them.
            </p>
          </div>
        </Panel>

        <Panel index={4} title="Host">
          <div className="flex flex-col gap-1 px-2 pb-4">
            <Stat label="CPU" value={metrics.cpu_percent.toFixed(0)} unit="%" />
            <Stat label="System RAM" value={metrics.ram_used_mb.toLocaleString()} unit="MiB" />
            <Stat label="API process" value={metrics.process_ram_mb.toLocaleString()} unit="MiB" />
            <Stat label="Uptime" value={formatUptime(metrics.uptime_s)} />
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

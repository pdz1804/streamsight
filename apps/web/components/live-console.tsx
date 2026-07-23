"use client";

/**
 * The live viewer: source selection, the annotated stream, and its telemetry.
 *
 * The right column leads with throughput as a single large figure, because that
 * is the number being watched; the timing breakdown that explains it sits below
 * at supporting weight.
 */

import { Play, Stop, UploadSimple } from "@phosphor-icons/react/dist/ssr";
import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { SourceInfo } from "@/lib/types";
import { useStream } from "@/hooks/use-stream";

import { TrackLegend } from "./track-legend";
import { Button, Chip, ErrorNote, Panel, Sparkline, Stat } from "./ui";
import { VideoStage } from "./video-stage";

const FPS_TARGET = 30;
const FPS_HISTORY = 90;

export function LiveConsole() {
  const { canvasRef, phase, message, telemetry, connected, start, stop } = useStream();
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [selected, setSelected] = useState("sample");
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshSources = useCallback(async () => {
    try {
      const list = await api.sources();
      setSources(list);
      setSelected((current) => (list.some((s) => s.id === current) ? current : (list[0]?.id ?? "")));
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refreshSources();
  }, [refreshSources]);

  // Keep a short FPS history for the sparkline; the API's own rolling series is
  // sampled at request time and would be sparse while a stream is running.
  useEffect(() => {
    if (!telemetry.fps) return;
    setFpsHistory((history) => [...history, telemetry.fps].slice(-FPS_HISTORY));
  }, [telemetry.fps, telemetry.frameId]);

  const streaming = connected && phase === "streaming";
  const busy = streaming || phase === "opening";

  const handleToggle = () => {
    if (busy) {
      stop();
      return;
    }
    setFpsHistory([]);
    start(selected);
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const created = await api.uploadSource(file);
      await refreshSources();
      setSelected(created.id);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const fpsTone = telemetry.fps >= FPS_TARGET ? "ok" : telemetry.fps > 0 ? "warn" : "neutral";

  return (
    <div className="mx-auto flex h-full min-h-0 max-w-[1600px] flex-col gap-4 p-4 md:p-6">
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="source" className="sr-only">
          Video source
        </label>
        <select
          id="source"
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
          disabled={streaming}
          className="h-9 min-w-[230px] rounded-[var(--radius)] border border-line bg-surface px-3 text-[13px] text-text shadow-[var(--shadow-sm)] transition-colors hover:border-line-strong disabled:opacity-50"
        >
          {sources.length === 0 ? <option value="">no sources found</option> : null}
          {sources.map((source) => (
            <option key={source.id} value={source.id}>
              {source.label}
              {source.detail ? ` (${source.detail})` : ""}
            </option>
          ))}
        </select>

        <Button variant="primary" onClick={handleToggle} disabled={!selected}>
          {busy ? <Stop size={15} weight="fill" /> : <Play size={15} weight="fill" />}
          {busy ? "Stop stream" : "Start stream"}
        </Button>

        <Button onClick={() => fileInputRef.current?.click()} disabled={uploading || streaming}>
          <UploadSimple size={15} />
          {uploading ? "Uploading" : "Upload video"}
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void handleUpload(file);
          }}
        />

        <div className="ml-auto flex items-center gap-2">
          {telemetry.degradedMode ? <Chip tone="warn">Degraded</Chip> : null}
          {streaming ? (
            <Chip tone="ok" live>
              Streaming
            </Chip>
          ) : (
            <Chip>{phase === "ended" ? "Ended" : "Idle"}</Chip>
          )}
        </div>
      </div>

      {error ? <ErrorNote>{error}</ErrorNote> : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <VideoStage
          canvasRef={canvasRef}
          phase={phase}
          message={message}
          hasFrame={telemetry.frameId > 0}
          precision={telemetry.precision}
          imgsz={telemetry.imgsz}
          resolution={
            telemetry.width && telemetry.height
              ? `${telemetry.width}x${telemetry.height}`
              : undefined
          }
        />

        <div className="flex min-h-0 flex-col gap-5 overflow-y-auto pb-1">
          {/* Headline figure, then the breakdown that explains it. */}
          <section className="surface-panel rise shrink-0 overflow-hidden" style={{ "--i": 0 } as React.CSSProperties}>
            <div className="relative flex items-start justify-between gap-3 px-5 pt-4">
              <div>
                <h2 className="text-[11px] font-medium uppercase tracking-[0.09em] text-text-mute">
                  Throughput
                </h2>
                <p className="mt-2 flex items-baseline gap-1.5">
                  <span
                    className={`display-num font-mono text-[44px] font-semibold ${
                      fpsTone === "ok"
                        ? "text-ok"
                        : fpsTone === "warn"
                          ? "text-warn"
                          : "text-text"
                    }`}
                  >
                    {telemetry.fps.toFixed(1)}
                  </span>
                  <span className="text-[13px] font-medium text-text-mute">fps</span>
                </p>
              </div>
              <span className="mt-1 font-mono text-[11px] text-text-mute">
                target {FPS_TARGET}
              </span>
            </div>

            <div className="relative mt-3">
              <Sparkline
                values={fpsHistory}
                label="Frames per second"
                floorAt={FPS_TARGET}
                height={56}
              />
            </div>

            <div className="relative flex flex-col gap-0.5 px-2 pb-3 pt-1">
              <Stat
                label="Server latency"
                value={telemetry.serverLatencyMs.toFixed(1)}
                unit="ms"
                hint="Decode, inference, tracking and JPEG encode inside the API process."
              />
              <Stat
                label="Inference"
                value={telemetry.inferenceMs.toFixed(1)}
                unit="ms"
                hint="Detection plus ByteTrack association only."
              />
              <Stat
                label="End to end"
                value={telemetry.clientLatencyMs.toFixed(0)}
                unit="ms"
                tone={telemetry.clientLatencyMs <= 100 ? "ok" : "warn"}
                hint="Server send timestamp to browser receipt. Localhost clocks, so indicative."
              />
            </div>
          </section>

          <Panel
            index={1}
            className="min-h-[220px] flex-1"
            title="Tracks"
            action={
              <span className="tnum font-mono text-[11px] text-text-mute">
                {telemetry.tracks.length} active
              </span>
            }
            bodyClassName="overflow-y-auto"
          >
            <TrackLegend tracks={telemetry.tracks} />
          </Panel>
        </div>
      </div>
    </div>
  );
}

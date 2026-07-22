"use client";

/**
 * The live viewer: source selection, the annotated stream, and its telemetry.
 */

import { Play, Stop, UploadSimple } from "@phosphor-icons/react/dist/ssr";
import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { SourceInfo } from "@/lib/types";
import { useStream } from "@/lib/use-stream";

import { Sparkline } from "./sparkline";
import { TrackLegend } from "./track-legend";
import { Button, Chip, ErrorNote, Panel, Stat } from "./ui";
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

  const handleToggle = () => {
    if (streaming || phase === "opening") {
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
    <div className="flex h-full min-h-0 flex-col gap-4 p-4 md:p-6">
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="source" className="sr-only">
          Video source
        </label>
        <select
          id="source"
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
          disabled={streaming}
          className="h-9 min-w-[220px] rounded-[var(--radius)] border border-line bg-surface px-3 text-[13px] text-text disabled:opacity-50"
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
          {streaming || phase === "opening" ? <Stop size={15} weight="fill" /> : <Play size={15} weight="fill" />}
          {streaming || phase === "opening" ? "Stop stream" : "Start stream"}
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

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <VideoStage
          canvasRef={canvasRef}
          phase={phase}
          message={message}
          hasFrame={telemetry.frameId > 0}
        />

        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          <Panel
            className="shrink-0"
            title="Throughput"
            action={
              <span className="font-mono text-[11px] text-text-mute">
                {telemetry.precision || "-"} · {telemetry.imgsz || "-"}px
              </span>
            }
          >
            <div className="divide-y divide-line">
              <Stat label="FPS" value={telemetry.fps.toFixed(1)} tone={fpsTone} />
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
                hint="Server send timestamp to browser receipt. Localhost clocks, so this is indicative."
              />
            </div>
            <div className="border-t border-line px-3 pb-3 pt-3">
              <Sparkline values={fpsHistory} label="Frames per second" floorAt={FPS_TARGET} />
              <p className="mt-1.5 px-1 text-[10px] leading-normal text-text-mute">
                dashed line marks the {FPS_TARGET} FPS target
              </p>
            </div>
          </Panel>

          <Panel
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

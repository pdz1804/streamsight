"use client";

/**
 * Single-image detection.
 *
 * Unlike the live stream, boxes here are drawn client-side: the response is JSON
 * only, and the browser already holds the source pixels, so re-encoding an
 * annotated JPEG server-side would be wasted work for a still frame.
 */

import { ImageSquare, UploadSimple } from "@phosphor-icons/react/dist/ssr";
import { useEffect, useRef, useState } from "react";

import { api, ApiError } from "@/lib/api";
import { trackColor } from "@/lib/palette";
import type { FrameResponse, Track } from "@/lib/types";

import { TrackLegend } from "./track-legend";
import { Button, EmptyState, ErrorNote, Panel, Stat } from "./ui";

export function SingleFrame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [result, setResult] = useState<FrameResponse | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  }, [imageUrl]);

  // Drawing has to happen *after* React commits, because the canvas is only
  // mounted once `result` is set. Painting straight from the fetch handler runs
  // while canvasRef is still null, which silently produces a blank stage.
  useEffect(() => {
    if (!result || !imageUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const tracks = result.tracks.length ? result.tracks : toTracks(result);
    const image = new Image();
    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(image, 0, 0);
      drawBoxes(context, tracks, image.naturalWidth);
    };
    image.src = imageUrl;
  }, [result, imageUrl]);

  const handleFile = async (file: File) => {
    setBusy(true);
    setError(null);
    const url = URL.createObjectURL(file);
    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return url;
    });
    try {
      setResult(await api.detectImage(file));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
      setResult(null);
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-5 p-5 md:p-7">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-[22px] font-semibold tracking-tight text-text">Single frame detection</h1>
        <Button variant="primary" onClick={() => fileInputRef.current?.click()} disabled={busy}>
          <UploadSimple size={15} />
          {busy ? "Detecting" : "Choose image"}
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void handleFile(file);
          }}
        />
      </div>

      {error ? <ErrorNote>{error}</ErrorNote> : null}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="rise flex min-h-[420px] items-center justify-center overflow-hidden rounded-[var(--radius-lg)] border border-line bg-stage shadow-[var(--shadow-lg)]">
          {result ? (
            <canvas ref={canvasRef} className="max-h-[68vh] max-w-full object-contain" />
          ) : (
            <EmptyState
              icon={<ImageSquare size={28} />}
              title="No image analysed yet"
              body="Pick any photo and the API will run the same detect and track pipeline the live stream uses, on one frame."
            />
          )}
        </div>

        <div className="flex flex-col gap-5">
          <Panel title="Result" index={1}>
            {result ? (
              <div className="flex flex-col gap-0.5 px-2 pb-4">
                <Stat label="Detections" value={result.detections.length} />
                <Stat label="Tracks" value={result.tracks.length} />
                <Stat label="Inference" value={result.timing.inference_ms.toFixed(1)} unit="ms" />
                <Stat label="Total" value={result.timing.total_ms.toFixed(1)} unit="ms" />
                <Stat label="Backend" value={result.precision} />
                <Stat label="Input size" value={result.imgsz} unit="px" />
              </div>
            ) : (
              <p className="px-4 py-6 text-center text-[12px] text-text-mute">
                Results appear here after an image is analysed.
              </p>
            )}
          </Panel>

          {result ? (
            <Panel title="Objects" index={2} bodyClassName="max-h-[42vh] overflow-y-auto">
              <TrackLegend tracks={result.tracks.length ? result.tracks : toTracks(result)} />
            </Panel>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/** A still frame may produce detections before ByteTrack confirms any identity. */
function toTracks(response: FrameResponse): Track[] {
  return response.detections.map((detection) => ({ ...detection, track_id: null }));
}

function drawBoxes(context: CanvasRenderingContext2D, tracks: Track[], width: number): void {
  const lineWidth = Math.max(2, Math.round(width / 480));
  const fontSize = Math.max(12, Math.round(width / 60));
  context.lineWidth = lineWidth;
  context.font = `${fontSize}px ui-monospace, monospace`;
  context.textBaseline = "top";

  for (const track of tracks) {
    const color = trackColor(track.track_id);
    context.strokeStyle = color;
    context.strokeRect(track.x1, track.y1, track.x2 - track.x1, track.y2 - track.y1);

    const label =
      track.track_id === null
        ? `${track.class_name} ${track.confidence.toFixed(2)}`
        : `#${track.track_id} ${track.class_name} ${track.confidence.toFixed(2)}`;
    const padding = fontSize * 0.3;
    const textWidth = context.measureText(label).width;
    const chipHeight = fontSize + padding * 2;
    const top = track.y1 - chipHeight < 0 ? track.y1 : track.y1 - chipHeight;

    context.fillStyle = color;
    context.fillRect(track.x1, top, textWidth + padding * 2, chipHeight);
    context.fillStyle = readableOn(color);
    context.fillText(label, track.x1 + padding, top + padding);
  }
}

/** Black or white label text, whichever survives on the chip colour. */
function readableOn(hex: string): string {
  const red = parseInt(hex.slice(1, 3), 16);
  const green = parseInt(hex.slice(3, 5), 16);
  const blue = parseInt(hex.slice(5, 7), 16);
  const luminance = 0.299 * red + 0.587 * green + 0.114 * blue;
  return luminance > 140 ? "#000000" : "#ffffff";
}


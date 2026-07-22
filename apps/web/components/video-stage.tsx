"use client";

/**
 * The video surface.
 *
 * The canvas is always mounted, never conditionally rendered: the streaming hook
 * holds a ref to it and starts painting the moment the first frame lands, so
 * unmounting it between states would drop frames. Overlays sit above it instead.
 *
 * The stage is deliberately darker than the surrounding page in both themes.
 * Video needs a neutral, low-luminance surround to read correctly, and a bright
 * frame around footage washes out the detection overlays.
 */

import { Broadcast, WarningCircle } from "@phosphor-icons/react/dist/ssr";
import type { RefObject } from "react";

import type { StreamPhase } from "@/lib/types";

export function VideoStage({
  canvasRef,
  phase,
  message,
  hasFrame,
  precision,
  imgsz,
  resolution,
}: {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  phase: StreamPhase | "idle";
  message: string;
  hasFrame: boolean;
  precision?: string;
  imgsz?: number;
  resolution?: string;
}) {
  const showOverlay = !hasFrame || phase === "error";

  return (
    <div className="rise relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-[var(--radius-lg)] border border-line bg-stage shadow-[var(--shadow-lg)]">
      {/* Inner edge, so the video sits in a frame rather than floating on a fill. */}
      <div
        className="pointer-events-none absolute inset-0 z-10 rounded-[var(--radius-lg)] shadow-[inset_0_0_0_1px_var(--stage-edge)]"
        aria-hidden="true"
      />

      <canvas
        ref={canvasRef}
        className="max-h-full max-w-full object-contain"
        aria-label="Annotated detection stream"
      />

      {/* Runtime badge, only once there is something to describe. */}
      {hasFrame && !showOverlay ? (
        <div className="pointer-events-none absolute left-3 top-3 z-20 flex items-center gap-2 rounded-full border border-white/10 bg-black/55 px-3 py-1.5 font-mono text-[11px] text-white/85 backdrop-blur-sm">
          {precision ? <span>{precision}</span> : null}
          {imgsz ? <span className="text-white/50">{imgsz}px</span> : null}
          {resolution ? <span className="text-white/50">{resolution}</span> : null}
        </div>
      ) : null}

      {showOverlay ? (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 px-8 text-center">
          {phase === "error" ? (
            <>
              <div className="rounded-full border border-danger/35 bg-danger/10 p-3">
                <WarningCircle size={26} className="text-danger" />
              </div>
              <p className="text-[15px] font-semibold text-text">Stream failed</p>
              <p className="max-w-[52ch] text-pretty text-[13px] leading-relaxed text-text-dim">
                {message || "The server could not open this source."}
              </p>
            </>
          ) : phase === "opening" ? (
            <>
              <div className="rounded-full border border-accent/30 bg-accent-soft p-3">
                <Broadcast size={26} className="animate-pulse text-accent" />
              </div>
              <p className="text-[13px] text-text-dim">{message || "opening source"}</p>
            </>
          ) : (
            <>
              <div className="rounded-full border border-line bg-surface-2 p-3">
                <Broadcast size={26} className="text-text-mute" />
              </div>
              <p className="text-[15px] font-semibold text-text">No stream running</p>
              <p className="max-w-[54ch] text-pretty text-[13px] leading-relaxed text-text-dim">
                Pick a source and start the stream. Boxes and track ids are drawn server-side,
                so the overlay can never drift out of sync with the frame it describes.
              </p>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

"use client";

/**
 * The video surface.
 *
 * The canvas is always mounted, never conditionally rendered: the streaming hook
 * holds a ref to it and starts painting the moment the first frame lands, so
 * unmounting it between states would drop frames. Overlays sit above it instead.
 */

import { Broadcast, WarningCircle } from "@phosphor-icons/react/dist/ssr";
import type { RefObject } from "react";

import type { StreamPhase } from "@/lib/types";

export function VideoStage({
  canvasRef,
  phase,
  message,
  hasFrame,
}: {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  phase: StreamPhase | "idle";
  message: string;
  hasFrame: boolean;
}) {
  const showOverlay = !hasFrame || phase === "error";

  return (
    <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-[var(--radius)] border border-line bg-stage">
      <canvas
        ref={canvasRef}
        className="max-h-full max-w-full object-contain"
        aria-label="Annotated detection stream"
      />

      {showOverlay ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center">
          {phase === "error" ? (
            <>
              <WarningCircle size={28} className="text-danger" />
              <p className="text-[14px] font-medium text-text">Stream failed</p>
              <p className="max-w-[52ch] text-[13px] leading-relaxed text-text-dim">
                {message || "The server could not open this source."}
              </p>
            </>
          ) : phase === "opening" ? (
            <>
              <Broadcast size={28} className="animate-pulse text-accent" />
              <p className="text-[13px] text-text-dim">{message || "opening source"}</p>
            </>
          ) : (
            <>
              <Broadcast size={28} className="text-text-mute" />
              <p className="text-[14px] font-medium text-text">No stream running</p>
              <p className="max-w-[52ch] text-[13px] leading-relaxed text-text-dim">
                Pick a source and start the stream. Detection boxes and track ids are drawn
                server-side, so the overlay can never drift out of sync with the frame.
              </p>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

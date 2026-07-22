/**
 * Live list of the tracks in the current frame.
 *
 * Sorted by id rather than by confidence or position so a given object holds its
 * row as long as ByteTrack holds its identity. A list that reorders every frame
 * is unreadable at 30 FPS.
 */

import { trackColor } from "@/lib/palette";
import type { Track } from "@/lib/types";

export function TrackLegend({ tracks }: { tracks: Track[] }) {
  if (tracks.length === 0) {
    return (
      <p className="px-4 py-6 text-center text-[12px] text-text-mute">
        No confirmed tracks in this frame.
      </p>
    );
  }

  const sorted = [...tracks].sort((a, b) => (a.track_id ?? 0) - (b.track_id ?? 0));

  return (
    <ul className="flex flex-col gap-0.5 px-2 pb-3">
      {sorted.map((track, index) => (
        <li
          key={track.track_id ?? `anon-${index}`}
          className="flex items-center gap-2.5 rounded-[var(--radius)] px-3 py-1.5 transition-colors duration-150 hover:bg-surface-2"
        >
          {/* Swatch matches the burned-in box colour exactly, so the legend maps
              to the frame without the viewer having to guess. */}
          <span
            className="size-2.5 shrink-0 rounded-[3px] ring-1 ring-inset ring-black/15"
            style={{ backgroundColor: trackColor(track.track_id) }}
            aria-hidden="true"
          />
          <span className="tnum w-11 shrink-0 font-mono text-[12px] font-medium text-text">
            {track.track_id === null ? "-" : `#${track.track_id}`}
          </span>
          <span className="min-w-0 flex-1 truncate text-[12px] text-text-dim">
            {track.class_name}
          </span>
          <span className="tnum shrink-0 font-mono text-[11px] text-text-mute">
            {track.confidence.toFixed(2)}
          </span>
        </li>
      ))}
    </ul>
  );
}

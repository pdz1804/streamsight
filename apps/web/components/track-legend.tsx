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
    <ul className="divide-y divide-line">
      {sorted.map((track, index) => (
        <li
          key={track.track_id ?? `anon-${index}`}
          className="flex items-center gap-2.5 px-4 py-1.5"
        >
          <span
            className="size-2.5 shrink-0 rounded-[2px]"
            style={{ backgroundColor: trackColor(track.track_id) }}
            aria-hidden="true"
          />
          <span className="tnum w-10 shrink-0 font-mono text-[12px] text-text">
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

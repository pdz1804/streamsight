/**
 * Track colour palette.
 *
 * These are the exact colours `apps/api/app/annotate.py` burns into the frame,
 * converted from its BGR tuples to RGB hex. The legend has to match the boxes
 * pixel for pixel, so the two lists are kept deliberately in sync -- if you
 * change one, change the other.
 */

export const TRACK_PALETTE = [
  "#00b0ff",
  "#e9b456",
  "#739e00",
  "#42e4f0",
  "#b27200",
  "#005ed5",
  "#a779cc",
  "#b5ff94",
  "#916cff",
  "#ffc874",
] as const;

export const UNIDENTIFIED_COLOR = "#a0a0a0";

export function trackColor(trackId: number | null): string {
  if (trackId === null) return UNIDENTIFIED_COLOR;
  return TRACK_PALETTE[trackId % TRACK_PALETTE.length];
}

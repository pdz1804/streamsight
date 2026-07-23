import type { ReactNode } from "react";

import { TONE_BG, TONE_BORDER, TONE_TEXT, type Tone } from "./tone";

/** The single pill-shaped element in the system: a compact state badge. */
export function Chip({
  tone = "neutral",
  children,
  live = false,
}: {
  tone?: Tone;
  children: ReactNode;
  live?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${TONE_BORDER[tone]} ${TONE_BG[tone]} ${TONE_TEXT[tone]}`}
    >
      {live ? <span className="live-dot size-1.5 rounded-full bg-current" aria-hidden="true" /> : null}
      {children}
    </span>
  );
}

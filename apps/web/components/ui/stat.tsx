import type { ReactNode } from "react";

import { TONE_TEXT, type Tone } from "./tone";

/**
 * A supporting label/value pair.
 *
 * Rows are separated by spacing rather than a hairline under every one. A
 * divider on each row of a list is the fastest way to make a panel read as a
 * spec sheet; the eye groups fine without them.
 */
export function Stat({
  label,
  value,
  unit,
  tone = "neutral",
  hint,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: Tone;
  hint?: string;
}) {
  const valueTone = tone === "neutral" ? "text-text" : TONE_TEXT[tone];
  return (
    <div
      className="flex items-baseline justify-between gap-4 rounded-[var(--radius)] px-3 py-2 transition-colors duration-200 hover:bg-surface-2"
      title={hint}
    >
      <span className="text-[12px] font-medium text-text-dim">{label}</span>
      <span className={`tnum font-mono text-[15px] font-semibold ${valueTone}`}>
        {value}
        {unit ? <span className="ml-1 text-[11px] font-normal text-text-mute">{unit}</span> : null}
      </span>
    </div>
  );
}

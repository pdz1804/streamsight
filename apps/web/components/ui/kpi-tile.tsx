import type { ReactNode } from "react";

import { TONE_TEXT, type Tone } from "./tone";

/**
 * A headline figure. Large, tight-tracked, with the unit and label subordinate.
 */
export function KpiTile({
  label,
  value,
  unit,
  tone = "neutral",
  caption,
  chart,
  index,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: Tone;
  caption?: ReactNode;
  chart?: ReactNode;
  index?: number;
}) {
  const valueTone = tone === "neutral" ? "text-text" : TONE_TEXT[tone];
  return (
    <section
      className="surface-panel rise flex flex-col overflow-hidden"
      style={index === undefined ? undefined : ({ "--i": index } as React.CSSProperties)}
    >
      <div className="relative px-5 pb-4 pt-4">
        {/* A real heading: this labels a section, so it belongs in the document
            outline rather than being styled text. */}
        <h2 className="text-[11px] font-medium uppercase tracking-[0.09em] text-text-mute">
          {label}
        </h2>
        <p className="mt-2.5 flex items-baseline gap-1.5">
          <span className={`display-num font-mono text-[40px] font-semibold ${valueTone}`}>
            {value}
          </span>
          {unit ? (
            <span className="text-[13px] font-medium text-text-mute">{unit}</span>
          ) : null}
        </p>
        {caption ? (
          <p className="mt-2 text-[12px] leading-snug text-text-dim">{caption}</p>
        ) : null}
      </div>
      {chart ? <div className="relative mt-auto">{chart}</div> : null}
    </section>
  );
}

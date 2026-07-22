/**
 * Shared primitives.
 *
 * Hierarchy comes from type scale first and surface treatment second. The
 * earlier version rendered every figure at the same size, so a headline metric
 * and a supporting one were visually identical and the eye had nowhere to land.
 * `KpiTile` exists to carry the number that matters; `Stat` carries the rest.
 *
 * Shape scale: containers 14px (`rounded-[var(--radius-lg)]`), controls and
 * inner surfaces 9px (`rounded-[var(--radius)]`), chips full. Chips are the only
 * pill in the system.
 */

import type { ReactNode } from "react";

export type Tone = "neutral" | "ok" | "warn" | "danger" | "accent";

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-text-dim",
  ok: "text-ok",
  warn: "text-warn",
  danger: "text-danger",
  accent: "text-accent",
};

const TONE_BORDER: Record<Tone, string> = {
  neutral: "border-line",
  ok: "border-ok/35",
  warn: "border-warn/35",
  danger: "border-danger/35",
  accent: "border-accent/35",
};

const TONE_BG: Record<Tone, string> = {
  neutral: "bg-surface-2",
  ok: "bg-ok/8",
  warn: "bg-warn/8",
  danger: "bg-danger/8",
  accent: "bg-accent-soft",
};

export function Panel({
  title,
  action,
  children,
  className = "",
  bodyClassName = "",
  index,
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** Position in a group, used to stagger the entry animation. */
  index?: number;
}) {
  return (
    <section
      className={`surface-panel rise flex min-h-0 flex-col ${className}`}
      style={index === undefined ? undefined : ({ "--i": index } as React.CSSProperties)}
    >
      {title ? (
        <header className="relative flex shrink-0 items-center justify-between gap-3 px-5 pb-3 pt-4">
          <h2 className="text-[13px] font-semibold tracking-tight text-text">{title}</h2>
          {action}
        </header>
      ) : null}
      <div className={`relative min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

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

export function Button({
  variant = "secondary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger";
}) {
  const base =
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius)] px-3.5 py-2 text-[13px] font-medium transition-[background-color,color,border-color,box-shadow,transform] duration-200 active:translate-y-px disabled:pointer-events-none disabled:opacity-40";
  const variants = {
    primary:
      "bg-accent text-accent-fg shadow-[var(--shadow-sm)] hover:bg-accent-hover hover:shadow-[var(--shadow)]",
    secondary:
      "border border-line bg-surface text-text shadow-[var(--shadow-sm)] hover:border-line-strong hover:bg-surface-2",
    danger: "border border-danger/40 bg-danger/8 text-danger hover:bg-danger/14",
  } as const;
  return <button className={`${base} ${variants[variant]} ${className}`} {...props} />;
}

export function Field({
  label,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  hint?: string;
  htmlFor?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor={htmlFor}
        className="text-[11px] font-medium uppercase tracking-[0.09em] text-text-mute"
      >
        {label}
      </label>
      {children}
      {hint ? <p className="text-[12px] leading-relaxed text-text-dim">{hint}</p> : null}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon?: ReactNode;
  title: string;
  body: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-8 py-12 text-center">
      {icon ? (
        <div className="mb-1 rounded-full border border-line bg-surface-2 p-3 text-text-mute">
          {icon}
        </div>
      ) : null}
      <h3 className="text-[15px] font-semibold text-text">{title}</h3>
      <p className="max-w-[46ch] text-pretty text-[13px] leading-relaxed text-text-dim">{body}</p>
      {action}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="rounded-[var(--radius)] border border-danger/35 bg-danger/8 px-3.5 py-2.5 text-[12px] leading-relaxed text-danger"
    >
      {children}
    </p>
  );
}

/** Skeleton that matches the shape of the content it replaces. */
export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-[var(--radius-lg)] border border-line bg-surface-2 ${className}`}
      aria-hidden="true"
    />
  );
}

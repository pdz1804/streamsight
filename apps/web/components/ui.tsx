/**
 * Shared primitives.
 *
 * The console is dense by design, so hierarchy comes from hairlines and type
 * weight rather than nested cards and shadows. Every surface uses the single 8px
 * radius; only `Chip` is pill-shaped, and that is the one documented exception.
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
  ok: "border-ok/40",
  warn: "border-warn/40",
  danger: "border-danger/40",
  accent: "border-accent/40",
};

export function Panel({
  title,
  action,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-[var(--radius)] border border-line bg-surface ${className}`}
    >
      {title ? (
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <h2 className="text-[13px] font-semibold tracking-tight text-text">{title}</h2>
          {action}
        </header>
      ) : null}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

/** Label + value row. Values are monospace and tabular so they do not jitter. */
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
    <div className="flex items-baseline justify-between gap-3 px-4 py-2" title={hint}>
      <span className="text-[11px] uppercase tracking-wider text-text-mute">{label}</span>
      <span className={`tnum font-mono text-[15px] font-medium ${valueTone}`}>
        {value}
        {unit ? <span className="ml-1 text-[11px] text-text-mute">{unit}</span> : null}
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
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${TONE_BORDER[tone]} ${TONE_TEXT[tone]}`}
    >
      {live ? (
        <span className={`live-dot size-1.5 rounded-full bg-current`} aria-hidden="true" />
      ) : null}
      {children}
    </span>
  );
}

export function Button({
  variant = "secondary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "danger" }) {
  const base =
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius)] px-3.5 py-2 text-[13px] font-medium transition-[background-color,color,border-color,transform] duration-150 active:translate-y-px disabled:pointer-events-none disabled:opacity-45";
  const variants = {
    primary: "bg-accent text-accent-fg hover:bg-accent-hover",
    secondary: "border border-line bg-surface-2 text-text hover:border-line-strong",
    danger: "border border-danger/50 bg-transparent text-danger hover:bg-danger/10",
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
      <label htmlFor={htmlFor} className="text-[11px] uppercase tracking-wider text-text-mute">
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
      {icon ? <div className="text-text-mute">{icon}</div> : null}
      <h3 className="text-[15px] font-semibold text-text">{title}</h3>
      <p className="max-w-[46ch] text-[13px] leading-relaxed text-text-dim">{body}</p>
      {action}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="rounded-[var(--radius)] border border-danger/40 bg-danger/8 px-3 py-2 text-[12px] leading-relaxed text-danger"
    >
      {children}
    </p>
  );
}

/** Shimmerless skeleton that matches the shape of the content it replaces. */
export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-surface-2 ${className}`} aria-hidden="true" />;
}

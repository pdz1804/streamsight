export type Tone = "neutral" | "ok" | "warn" | "danger" | "accent";

export const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-text-dim",
  ok: "text-ok",
  warn: "text-warn",
  danger: "text-danger",
  accent: "text-accent",
};

export const TONE_BORDER: Record<Tone, string> = {
  neutral: "border-line",
  ok: "border-ok/35",
  warn: "border-warn/35",
  danger: "border-danger/35",
  accent: "border-accent/35",
};

export const TONE_BG: Record<Tone, string> = {
  neutral: "bg-surface-2",
  ok: "bg-ok/8",
  warn: "bg-warn/8",
  danger: "bg-danger/8",
  accent: "bg-accent-soft",
};

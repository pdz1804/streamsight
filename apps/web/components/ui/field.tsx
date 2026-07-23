import type { ReactNode } from "react";

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

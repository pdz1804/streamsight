import type { ReactNode } from "react";

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

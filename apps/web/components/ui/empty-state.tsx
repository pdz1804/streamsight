import type { ReactNode } from "react";

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

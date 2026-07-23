import type { ReactNode } from "react";

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

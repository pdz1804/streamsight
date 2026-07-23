/** Skeleton that matches the shape of the content it replaces. */
export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-[var(--radius-lg)] border border-line bg-surface-2 ${className}`}
      aria-hidden="true"
    />
  );
}

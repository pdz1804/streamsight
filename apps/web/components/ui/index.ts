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

export type { Tone } from "./tone";
export { Panel } from "./panel";
export { KpiTile } from "./kpi-tile";
export { Stat } from "./stat";
export { Chip } from "./chip";
export { Button } from "./button";
export { Field } from "./field";
export { EmptyState } from "./empty-state";
export { ErrorNote } from "./error-note";
export { Skeleton } from "./skeleton";
export { Sparkline } from "./sparkline";
export { ThemeToggle } from "./theme-toggle";

"use client";

/**
 * Minimal SVG sparkline.
 *
 * Dependency-free is the right call for one series with no axes and no
 * interaction: a charting library would add bundle weight and a second theming
 * surface for a shape that is a handful of path maths.
 *
 * The gradient fill and end-point marker are what stop it reading as a
 * placeholder rule. The gradient id is derived from the label so two charts on
 * one page cannot collide in the SVG id namespace.
 */

export function Sparkline({
  values,
  height = 56,
  label,
  color = "var(--accent)",
  floorAt,
  showLast = true,
}: {
  values: number[];
  height?: number;
  label: string;
  color?: string;
  /** Optional reference line, e.g. the 30 FPS acceptance target. */
  floorAt?: number;
  showLast?: boolean;
}) {
  const width = 100;
  const gradientId = `spark-${label.replace(/[^a-z0-9]/gi, "-").toLowerCase()}`;

  // A flat line at zero would read as a measurement rather than an absence, so
  // the empty state says so in words instead of drawing a misleading baseline.
  const hasSignal = values.length >= 2 && values.some((value) => value > 0);
  if (!hasSignal) {
    return (
      <div
        className="flex items-center justify-center border-t border-dashed border-line text-[11px] text-text-mute"
        style={{ height }}
        role="img"
        aria-label={`${label}: no samples yet`}
      >
        awaiting samples
      </div>
    );
  }

  // Scale to the data range, not to zero. A latency series hovering around 40 ms
  // plotted on a 0-based axis is a solid block with a flat lid: all fill, no
  // signal.
  const dataMax = Math.max(...values);
  const dataMin = Math.min(...values);
  const dataSpan = dataMax - dataMin;

  // A reference line only earns a place in the scale if it is near the data.
  // Forcing a 3500 MiB budget into a series sitting at 488 flattens the trace
  // into a rule at the bottom of the box, which reports nothing. When the
  // reference is that far away the number itself is the story, and it is already
  // in the caption.
  const referenceInRange =
    floorAt !== undefined &&
    floorAt <= dataMax + Math.max(dataSpan * 3, dataMax * 0.6) &&
    floorAt >= dataMin - Math.max(dataSpan * 3, dataMax * 0.6);
  const reference = referenceInRange ? floorAt : undefined;

  const candidates = reference === undefined ? values : [...values, reference];
  const rawMax = Math.max(...candidates);
  const rawMin = Math.min(...candidates);
  const pad = (rawMax - rawMin) * 0.22 || Math.abs(rawMax) * 0.12 || 1;
  const max = rawMax + pad;
  const min = Math.max(0, rawMin - pad);
  const span = max - min || 1;

  const toY = (value: number) => height - ((value - min) / span) * height;
  const step = width / (values.length - 1);
  const points = values.map((value, index) => `${index * step},${toY(value)}`);
  const line = `M ${points.join(" L ")}`;
  const area = `${line} L ${width},${height} L 0,${height} Z`;

  const latest = values[values.length - 1];
  const lastX = (values.length - 1) * step;
  const lastY = toY(latest);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
      role="img"
      aria-label={`${label}: latest ${latest.toFixed(1)}, range ${min.toFixed(1)} to ${max.toFixed(1)}`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.30" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      <path d={area} fill={`url(#${gradientId})`} />

      {reference !== undefined ? (
        <line
          x1={0}
          x2={width}
          y1={toY(reference)}
          y2={toY(reference)}
          stroke="var(--text-mute)"
          strokeWidth={1}
          strokeDasharray="2 4"
          opacity={0.55}
          vectorEffect="non-scaling-stroke"
        />
      ) : null}

      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth={1.75}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />

      {/* Marks where the series is now, so a live chart has a clear "you are here". */}
      {showLast ? (
        <circle
          cx={lastX}
          cy={lastY}
          r={2.5}
          fill={color}
          stroke="var(--surface)"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
    </svg>
  );
}

/**
 * Minimal SVG sparkline.
 *
 * A dependency-free chart is the right call here: one series, no axes, no
 * interaction. Pulling in a charting library for this would add bundle weight
 * and a theming surface for a shape that is nine lines of path maths.
 */

export function Sparkline({
  values,
  height = 44,
  label,
  color = "var(--accent)",
  floorAt,
}: {
  values: number[];
  height?: number;
  label: string;
  color?: string;
  /** Optional reference line, e.g. the 30 FPS acceptance target. */
  floorAt?: number;
}) {
  const width = 100;
  if (values.length < 2) {
    return (
      <div
        className="flex items-center justify-center text-[11px] text-text-mute"
        style={{ height }}
        role="img"
        aria-label={`${label}: not enough samples yet`}
      >
        collecting samples
      </div>
    );
  }

  // Scale to the data range, not to zero. A latency series hovering around 40 ms
  // plotted on a 0-based axis is a solid block with a flat lid: all the fill, none
  // of the signal. Any reference line is included so it stays on screen, and a
  // little padding keeps the trace off the edges.
  const candidates = floorAt === undefined ? values : [...values, floorAt];
  const rawMax = Math.max(...candidates);
  const rawMin = Math.min(...candidates);
  const pad = (rawMax - rawMin) * 0.18 || Math.abs(rawMax) * 0.1 || 1;
  const max = rawMax + pad;
  const min = Math.max(0, rawMin - pad);
  const span = max - min || 1;

  const toY = (value: number) => height - ((value - min) / span) * height;
  const step = width / (values.length - 1);
  const points = values.map((value, index) => `${index * step},${toY(value)}`);
  const line = `M ${points.join(" L ")}`;
  const area = `${line} L ${width},${height} L 0,${height} Z`;

  const latest = values[values.length - 1];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
      role="img"
      aria-label={`${label}: latest ${latest.toFixed(1)}, range ${min.toFixed(1)} to ${max.toFixed(1)}`}
    >
      <path d={area} fill={color} opacity={0.09} />
      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      {floorAt !== undefined ? (
        <line
          x1={0}
          x2={width}
          y1={toY(floorAt)}
          y2={toY(floorAt)}
          stroke="var(--text-mute)"
          strokeWidth={1}
          strokeDasharray="3 3"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
    </svg>
  );
}

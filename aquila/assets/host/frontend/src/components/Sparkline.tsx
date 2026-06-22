import { useRef, useState, useCallback } from "react";

type SparklineProps = {
  data: number[];
  height?: number;
  color?: string;
  area?: boolean;
  yMin?: number;
  yMax?: number;
};

export function Sparkline({
  data,
  height = 48,
  color = "var(--accent)",
  area = false,
  yMin,
  yMax,
}: SparklineProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    value: number;
  } | null>(null);

  const width = 160;
  const pad = 1;

  const lo = yMin ?? Math.min(...data);
  const hi = yMax ?? Math.max(...data);
  const range = hi - lo || 1;

  const points = data.map((v, i) => {
    const x = pad + (i / Math.max(data.length - 1, 1)) * (width - 2 * pad);
    const y = pad + (1 - (v - lo) / range) * (height - 2 * pad);
    return { x, y, value: v };
  });

  const lineD = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const areaD = lineD + ` L${points[points.length - 1].x},${height} L${points[0].x},${height} Z`;

  const onMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg || points.length === 0) return;
      const rect = svg.getBoundingClientRect();
      const relX = ((e.clientX - rect.left) / rect.width) * width;
      let closest = points[0];
      let minDist = Math.abs(relX - closest.x);
      for (let i = 1; i < points.length; i++) {
        const d = Math.abs(relX - points[i].x);
        if (d < minDist) {
          minDist = d;
          closest = points[i];
        }
      }
      setTooltip({ x: closest.x, y: closest.y, value: closest.value });
    },
    [points, width]
  );

  const onLeave = useCallback(() => setTooltip(null), []);

  if (data.length === 0) return null;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      preserveAspectRatio="none"
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{ display: "block" }}
    >
      {area && (
        <path d={areaD} fill={color} opacity={0.15} />
      )}
      <path d={lineD} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      {tooltip && (
        <>
          <circle cx={tooltip.x} cy={tooltip.y} r={3} fill={color} />
          <title>{tooltip.value.toFixed(1)}</title>
        </>
      )}
    </svg>
  );
}

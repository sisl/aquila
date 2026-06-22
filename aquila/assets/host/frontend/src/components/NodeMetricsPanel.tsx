import { useMemo, useState } from "react";
import {
  Box,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  useTheme
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";

import { Sparkline } from "./Sparkline";

import { fetchNodeMetricsHistory } from "../services/api";
import { EmptyState } from "./EmptyState";

type NodeMetricsPanelProps = {
  nodeId: number;
};

const RANGES: { minutes: number; label: string }[] = [
  { minutes: 60, label: "1h" },
  { minutes: 360, label: "6h" },
  { minutes: 1440, label: "24h" }
];

// Samples arrive every ~10s; downsample server-side to ~180 points per range.
function stepFor(minutes: number): number {
  return Math.max(1, Math.round((minutes * 6) / 180));
}

type Series = {
  label: string;
  unit: string;
  values: number[];
  max?: number;
};

export function NodeMetricsPanel({ nodeId }: NodeMetricsPanelProps) {
  const muiTheme = useTheme();
  const [minutes, setMinutes] = useState(60);

  const historyQuery = useQuery({
    queryKey: ["node-metrics-history", nodeId, minutes],
    queryFn: () => fetchNodeMetricsHistory(nodeId, minutes, stepFor(minutes)),
    refetchInterval: 30000,
    placeholderData: (previous) => previous
  });

  const series = useMemo<Series[]>(() => {
    const points = historyQuery.data?.points ?? [];
    if (points.length === 0) {
      return [];
    }
    const gpuIndices = new Set<number>();
    for (const point of points) {
      for (const gpu of point.gpus ?? []) {
        gpuIndices.add(gpu.index);
      }
    }
    const result: Series[] = [];
    for (const index of [...gpuIndices].sort((a, b) => a - b)) {
      const util: number[] = [];
      const mem: number[] = [];
      let memTotal = 0;
      for (const point of points) {
        const gpu = (point.gpus ?? []).find((g) => g.index === index);
        util.push(gpu?.utilization ?? 0);
        mem.push((gpu?.memory_used_mb ?? 0) / 1024);
        memTotal = Math.max(memTotal, (gpu?.memory_total_mb ?? 0) / 1024);
      }
      result.push({ label: `GPU ${index} util`, unit: "%", values: util, max: 100 });
      result.push({
        label: `GPU ${index} mem`,
        unit: "GB",
        values: mem,
        max: memTotal || undefined
      });
    }
    result.push({
      label: "CPU",
      unit: "%",
      values: points.map((p) => p.cpu_percent ?? 0),
      max: 100
    });
    result.push({
      label: "RAM",
      unit: "%",
      values: points.map((p) => p.memory_percent ?? 0),
      max: 100
    });
    return result;
  }, [historyQuery.data]);

  return (
    <Box sx={{ py: 1.5, px: 1 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
        <Typography variant="body2" className="muted">
          Utilization history
        </Typography>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={minutes}
          onChange={(_, value) => value && setMinutes(value as number)}
        >
          {RANGES.map((range) => (
            <ToggleButton key={range.minutes} value={range.minutes} sx={{ px: 1.2, py: 0.2 }}>
              {range.label}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>
      {historyQuery.isLoading ? (
        <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
          <CircularProgress size={20} />
        </Box>
      ) : series.length === 0 ? (
        <EmptyState
          primary="No samples recorded yet."
          hint="History accumulates while the node is online."
        />
      ) : (
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: 1.5
          }}
        >
          {series.map((entry) => {
            const last = entry.values[entry.values.length - 1] ?? 0;
            return (
              <Box
                key={entry.label}
                sx={{
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius)",
                  px: 1,
                  pt: 0.5
                }}
              >
                <Typography variant="caption" className="muted">
                  {entry.label} — {last.toFixed(entry.unit === "GB" ? 1 : 0)}
                  {entry.unit}
                </Typography>
                <Sparkline
                  data={entry.values}
                  height={48}
                  area
                  color={muiTheme.palette.primary.main}
                  yMin={0}
                  yMax={entry.max}
                />
              </Box>
            );
          })}
        </Box>
      )}
    </Box>
  );
}

import { Fragment, useState } from "react";
import ClearIcon from "@mui/icons-material/Clear";
import DnsOutlined from "@mui/icons-material/DnsOutlined";
import SearchIcon from "@mui/icons-material/Search";
import WarningAmberRounded from "@mui/icons-material/WarningAmberRounded";
import {
  Chip,
  Collapse,
  IconButton,
  InputAdornment,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  Box,
  Tooltip,
  Typography,
  useMediaQuery,
  useTheme
} from "@mui/material";

import type { Node } from "../services/api";
import { AppButton } from "./AppButton";
import { EmptyState } from "./EmptyState";
import { NodeMetricsPanel } from "./NodeMetricsPanel";

type NodeTableProps = {
  nodes: Node[];
  loading?: boolean;
  onManage?: (node: Node) => void;
  onToggleMaintenance?: (node: Node) => void;
};

const LOW_DISK_GB = 30;

const toGb = (mb?: number) => {
  if (mb === undefined) {
    return null;
  }
  return (mb / 1024).toFixed(1);
};

function statusColor(node: Node): "success" | "warning" | "default" {
  if (node.maintenance) return "warning";
  if (node.status === "healthy") return "success";
  return "default";
}

type GpuUsage = NonNullable<Node["gpu_usage"]>[number];

function gpuLine(nodeId: number, gpu: GpuUsage) {
  const used = toGb(gpu.memory_used_mb);
  const total = toGb(gpu.memory_total_mb);
  const memLabel = used && total ? `${used}/${total} GB` : "?";
  const sourceLabel = gpu.source === "unified" ? " (unified)" : "";
  return (
    <Typography
      key={`${nodeId}-gpu-${gpu.index}`}
      variant="body2"
      sx={{ whiteSpace: "nowrap" }}
    >
      GPU{gpu.index}
      {/* Opacity-based muting so the labels read on the light cell AND the
          dark tooltip surface. */}
      <Box component="span" sx={{ opacity: 0.65 }}>
        {" "}· compute{" "}
      </Box>
      {gpu.utilization != null ? `${gpu.utilization}%` : "n/a"}
      <Box component="span" sx={{ opacity: 0.65 }}>
        {" "}· memory{" "}
      </Box>
      {memLabel}
      {sourceLabel}
    </Typography>
  );
}

const GPU_EXPLAINER =
  "compute: how busy the GPU's compute units are; memory: VRAM in use / total";

// Single GPU: one labelled line. Multiple GPUs: a one-line summary with the
// per-GPU breakdown in the tooltip, so multi-GPU nodes keep row density.
function GpuCell({ nodeId, gpus }: { nodeId: number; gpus: GpuUsage[] }) {
  if (gpus.length === 1) {
    return (
      <Tooltip title={GPU_EXPLAINER} enterDelay={500}>
        <Box>{gpuLine(nodeId, gpus[0])}</Box>
      </Tooltip>
    );
  }

  const utils = gpus
    .map((gpu) => gpu.utilization)
    .filter((value): value is number => typeof value === "number");
  const avgUtil = utils.length
    ? Math.round(utils.reduce((sum, value) => sum + value, 0) / utils.length)
    : null;
  const usedMb = gpus.reduce((sum, gpu) => sum + (gpu.memory_used_mb ?? 0), 0);
  const totalMb = gpus.reduce((sum, gpu) => sum + (gpu.memory_total_mb ?? 0), 0);
  const memLabel = totalMb > 0 ? `${toGb(usedMb)}/${toGb(totalMb)} GB` : "?";

  return (
    <Tooltip
      title={
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
          {gpus.map((gpu) => gpuLine(nodeId, gpu))}
        </Box>
      }
      enterDelay={300}
    >
      <Typography variant="body2" sx={{ whiteSpace: "nowrap" }}>
        {gpus.length} GPUs
        <Box component="span" sx={{ opacity: 0.65 }}>
          {" "}· compute{" "}
        </Box>
        {avgUtil != null ? `${avgUtil}% avg` : "n/a"}
        <Box component="span" sx={{ opacity: 0.65 }}>
          {" "}· memory{" "}
        </Box>
        {memLabel}
      </Typography>
    </Tooltip>
  );
}

type SortKey = "hostname" | "ip" | "port" | "status" | "heartbeat";

export function NodeTable({ nodes, loading = false, onManage, onToggleMaintenance }: NodeTableProps) {
  const [expandedNodeId, setExpandedNodeId] = useState<number | null>(null);

  // Collapse low-priority columns below "lg" so the table fits without
  // horizontal scrolling on tablets/laptops.
  const muiTheme = useTheme();
  const compact = useMediaQuery(muiTheme.breakpoints.down("lg"));
  const columnCount = compact ? 5 : 7;

  // Search + sort are purely presentational (same pattern as the
  // deployments table).
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const handleSort = (key: SortKey) => {
    if (sortBy === key) {
      setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(key);
      setSortDir("asc");
    }
  };

  const sortValue = (node: Node): string | number => {
    switch (sortBy) {
      case "hostname":
        return node.hostname.toLowerCase();
      case "ip":
        return node.ip_address;
      case "port":
        return node.port ?? Number.MAX_SAFE_INTEGER;
      case "status":
        return node.maintenance ? "maintenance" : node.status;
      case "heartbeat":
        return node.last_heartbeat_at ?? "";
      default:
        return 0;
    }
  };

  const query = search.trim().toLowerCase();
  const visibleRows = nodes.filter((node) => {
    if (!query) return true;
    const haystack = [
      node.hostname,
      node.ip_address,
      node.maintenance ? "maintenance" : node.status,
      String(node.port ?? "")
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
  if (sortBy !== null) {
    visibleRows.sort((a, b) => {
      const va = sortValue(a);
      const vb = sortValue(b);
      const cmp =
        typeof va === "number" && typeof vb === "number"
          ? va - vb
          : String(va).localeCompare(String(vb));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }

  const sortableHeader = (key: SortKey, label: string, align?: "right") => (
    <TableCell align={align} sortDirection={sortBy === key ? sortDir : false}>
      <TableSortLabel
        active={sortBy === key}
        direction={sortBy === key ? sortDir : "asc"}
        onClick={() => handleSort(key)}
      >
        {label}
      </TableSortLabel>
    </TableCell>
  );

  return (
    <>
    <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1.5 }}>
      <TextField
        size="small"
        placeholder="Search hostname, IP, status…"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        sx={{ width: { xs: "100%", sm: 280 } }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon sx={{ fontSize: 16, color: "text.secondary" }} />
            </InputAdornment>
          ),
          endAdornment: search ? (
            <InputAdornment position="end">
              <IconButton
                size="small"
                aria-label="Clear search"
                onClick={() => setSearch("")}
              >
                <ClearIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </InputAdornment>
          ) : undefined
        }}
      />
    </Box>
    <TableContainer
      component={Box}
      className="scroll-thin"
      sx={{ minHeight: 200, maxHeight: "70vh", overflowX: "auto" }}
    >
      <Table size="small" stickyHeader sx={{ minWidth: compact ? 560 : 820 }}>
        <TableHead>
          <TableRow>
            {sortableHeader("hostname", "Hostname")}
            {sortableHeader("ip", "IP Address")}
            {!compact && sortableHeader("port", "Client Port")}
            <TableCell>GPUs</TableCell>
            {sortableHeader("status", "Status")}
            {!compact && sortableHeader("heartbeat", "Last Heartbeat", "right")}
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {visibleRows.map((node) => {
            const freeGb = node.disk_usage?.free_gb;
            const lowDisk = typeof freeGb === "number" && freeGb < LOW_DISK_GB;
            const expanded = expandedNodeId === node.id;
            return (
              <Fragment key={node.id}>
                <TableRow hover>
                  <TableCell>{node.hostname}</TableCell>
                  <TableCell>{node.ip_address}</TableCell>
                  {!compact && <TableCell>{node.port ?? "-"}</TableCell>}
                  <TableCell>
                    {node.gpu_usage && node.gpu_usage.length > 0 ? (
                      <GpuCell nodeId={node.id} gpus={node.gpu_usage} />
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap" }}>
                      <Chip
                        label={node.maintenance ? "maintenance" : node.status}
                        size="small"
                        color={statusColor(node)}
                      />
                      {node.rogue_container_count != null && node.rogue_container_count > 0 && (
                        <Chip
                          icon={<WarningAmberRounded sx={{ fontSize: 13 }} />}
                          label={`${node.rogue_container_count} rogue`}
                          size="small"
                          color="warning"
                          onClick={onManage ? () => onManage(node) : undefined}
                          clickable={Boolean(onManage)}
                          title="Untracked vLLM container(s) detected — click to manage"
                        />
                      )}
                      {lowDisk && (
                        <Chip
                          icon={<WarningAmberRounded sx={{ fontSize: 13 }} />}
                          label={`${freeGb!.toFixed(0)} GB disk free`}
                          size="small"
                          color="warning"
                          onClick={onManage ? () => onManage(node) : undefined}
                          clickable={Boolean(onManage)}
                          title="Low disk space — click to manage the model/image caches"
                        />
                      )}
                    </Box>
                  </TableCell>
                  {!compact && (
                    <TableCell align="right">{node.last_heartbeat_at ?? "-"}</TableCell>
                  )}
                  <TableCell align="right">
                    <Box sx={{ display: "flex", gap: 1, justifyContent: "flex-end" }}>
                      <AppButton
                        type="button"
                        className="app-button--small"
                        ghost
                        onClick={() =>
                          setExpandedNodeId(expanded ? null : node.id)
                        }
                      >
                        {expanded ? "Hide Metrics" : "Metrics"}
                      </AppButton>
                      {onToggleMaintenance && (
                        <AppButton
                          type="button"
                          className="app-button--small"
                          ghost
                          onClick={() => onToggleMaintenance(node)}
                        >
                          {node.maintenance ? "End Maintenance" : "Maintenance"}
                        </AppButton>
                      )}
                      <AppButton
                        type="button"
                        className="app-button--small"
                        ghost
                        onClick={() => onManage?.(node)}
                      >
                        Manage
                      </AppButton>
                    </Box>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell
                    colSpan={columnCount}
                    sx={{ p: 0, borderBottom: expanded ? undefined : "none" }}
                  >
                    <Collapse in={expanded} timeout={180} unmountOnExit>
                      <NodeMetricsPanel nodeId={node.id} />
                    </Collapse>
                  </TableCell>
                </TableRow>
              </Fragment>
            );
          })}
          {loading &&
            nodes.length === 0 &&
            [0, 1, 2].map((row) => (
              <TableRow key={`skeleton-${row}`}>
                {Array.from({ length: columnCount }, (_, column) => (
                  <TableCell key={column}>
                    <Skeleton variant="text" />
                  </TableCell>
                ))}
              </TableRow>
            ))}
          {!loading && nodes.length === 0 && (
            <TableRow>
              <TableCell colSpan={columnCount} sx={{ borderBottom: "none" }}>
                <EmptyState
                  icon={DnsOutlined}
                  primary="No registered nodes yet."
                  hint="Start a client agent and it will appear here within seconds."
                />
              </TableCell>
            </TableRow>
          )}
          {!loading && nodes.length > 0 && visibleRows.length === 0 && (
            <TableRow>
              <TableCell colSpan={columnCount} sx={{ borderBottom: "none" }}>
                <EmptyState
                  icon={SearchIcon}
                  primary={`No nodes match "${search.trim()}".`}
                  hint="Clear the search to see all nodes."
                />
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
    </>
  );
}

import { Fragment, useState } from "react";
import BarChartOutlined from "@mui/icons-material/BarChartOutlined";
import BuildOutlined from "@mui/icons-material/BuildOutlined";
import ClearIcon from "@mui/icons-material/Clear";
import DnsOutlined from "@mui/icons-material/DnsOutlined";
import EngineeringOutlined from "@mui/icons-material/EngineeringOutlined";
import SearchIcon from "@mui/icons-material/Search";
import WarningAmberRounded from "@mui/icons-material/WarningAmberRounded";
import {
  Button,
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
} from "@mui/material";

import type { Node } from "../services/api";
import { useColumnVisibility, type ColumnDef } from "../hooks/useColumnVisibility";
import { ColumnPicker } from "./ColumnPicker";
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

const HEARTBEAT_STALE_SECONDS = 60;

function isStale(node: Node): boolean {
  if (!node.last_heartbeat_at) return true;
  const age = (Date.now() - new Date(node.last_heartbeat_at).getTime()) / 1000;
  return age > HEARTBEAT_STALE_SECONDS;
}

function statusColor(node: Node): "success" | "warning" | "error" | "default" {
  if (node.maintenance) return "warning";
  if (node.partial_maintenance) return "warning";
  if (isStale(node)) return "error";
  if (node.status === "no-runtime") return "error";
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
      sx={{ whiteSpace: { xs: "normal", md: "nowrap" } }}
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
      <Typography variant="body2" sx={{ whiteSpace: { xs: "normal", md: "nowrap" } }}>
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

const NODE_ICON_SX = {
  fontSize: 16,
  opacity: 0.45,
  "@media (hover: none)": { fontSize: 20 },
} as const;

const NODE_BTN_SX = {
  minWidth: "unset",
  px: 0.75,
  pt: 0.5,
  pb: 0.25,
  position: "relative",
  "&:hover": { backgroundColor: "transparent" },
  "@media (hover: hover)": {
    "&:hover .MuiSvgIcon-root": { opacity: 1, transform: "translateY(-4px)" },
    "& .MuiSvgIcon-root": { transition: "opacity 120ms ease, transform 120ms ease" },
    "& .act-label": {
      position: "absolute",
      top: "calc(100% - 2px)",
      left: "50%",
      transform: "translateX(-50%)",
      fontSize: "0.55rem",
      lineHeight: 1,
      letterSpacing: "0.02em",
      opacity: 0,
      whiteSpace: "nowrap",
      pointerEvents: "none",
      transition: "opacity 100ms ease 80ms",
    },
    "&:hover .act-label": {
      opacity: 0.7,
      transition: "opacity 120ms ease-out",
    },
  },
  "@media (hover: none)": {
    minWidth: 36,
    minHeight: 36,
    px: 0.75,
    "& .act-label": { display: "none" },
  },
} as const;

function NodeActionBtn({
  tooltip,
  label,
  icon,
  onClick,
}: {
  tooltip: string;
  label?: string;
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <Tooltip title={tooltip} enterDelay={2000}>
      <Button
        variant="text"
        size="small"
        color="primary"
        aria-label={tooltip}
        onClick={onClick}
        sx={NODE_BTN_SX}
      >
        {icon}
        <span className="act-label" aria-hidden>{label ?? tooltip}</span>
      </Button>
    </Tooltip>
  );
}

type SortKey = "hostname" | "ip" | "port" | "status" | "heartbeat";

const NODE_COLUMNS: ColumnDef[] = [
  { key: "hostname",  label: "Hostname",       alwaysVisible: true },
  { key: "ip",        label: "IP Address" },
  { key: "port",      label: "Client Port" },
  { key: "gpus",      label: "GPUs" },
  { key: "status",    label: "Status" },
  { key: "heartbeat", label: "Last Heartbeat" },
  { key: "actions",   label: "Actions",        alwaysVisible: true },
];

export function NodeTable({ nodes, loading = false, onManage, onToggleMaintenance }: NodeTableProps) {
  const [expandedNodeId, setExpandedNodeId] = useState<number | null>(null);

  const { visibleKeys, userHidden, toggle: toggleColumn, reset: resetColumns, isCustomized } =
    useColumnVisibility("athanor:columns:nodes", NODE_COLUMNS);
  const columnCount = visibleKeys.size;

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
        return node.maintenance
          ? "maintenance"
          : node.partial_maintenance
            ? "partial maintenance"
            : node.status;
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
      node.maintenance || node.partial_maintenance ? "maintenance" : node.status,
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
    <Box sx={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 1, mb: 1.5, flexWrap: "wrap" }}>
      <ColumnPicker
        columns={NODE_COLUMNS}
        userHidden={userHidden}
        onToggle={toggleColumn}
        onReset={resetColumns}
        isCustomized={isCustomized}
      />
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
      <Table size="small" stickyHeader sx={{ minWidth: 820 }}>
        <TableHead>
          <TableRow>
            {sortableHeader("hostname", "Hostname")}
            {visibleKeys.has("ip") && sortableHeader("ip", "IP Address")}
            {visibleKeys.has("port") && sortableHeader("port", "Client Port")}
            {visibleKeys.has("gpus") && <TableCell>GPUs</TableCell>}
            {visibleKeys.has("status") && sortableHeader("status", "Status")}
            {visibleKeys.has("heartbeat") && sortableHeader("heartbeat", "Last Heartbeat", "right")}
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
                  {visibleKeys.has("ip") && <TableCell>{node.ip_address}</TableCell>}
                  {visibleKeys.has("port") && <TableCell>{node.port ?? "-"}</TableCell>}
                  {visibleKeys.has("gpus") && (
                    <TableCell>
                      {node.gpu_usage && node.gpu_usage.length > 0 ? (
                        <GpuCell nodeId={node.id} gpus={node.gpu_usage} />
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          -
                        </Typography>
                      )}
                    </TableCell>
                  )}
                  {visibleKeys.has("status") && (
                    <TableCell>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap" }}>
                        <Chip
                          label={
                            node.maintenance
                              ? "maintenance"
                              : node.partial_maintenance
                                ? `maint. GPU ${(node.maintenance_gpus ?? []).join(", ")}`
                                : isStale(node)
                                  ? "unreachable"
                                  : node.status === "no-runtime"
                                    ? "no runtime"
                                    : node.status
                          }
                          size="small"
                          color={statusColor(node)}
                          title={
                            isStale(node)
                              ? "Last heartbeat was over 60 seconds ago — node may be down."
                              : node.status === "no-runtime"
                                ? "No container runtime detected — install Docker or enable the Podman socket."
                                : undefined
                          }
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
                        {node.rogue_process_count != null && node.rogue_process_count > 0 && (
                          <Chip
                            icon={<WarningAmberRounded sx={{ fontSize: 13 }} />}
                            label={`${node.rogue_process_count} orphan GPU`}
                            size="small"
                            color="warning"
                            onClick={onManage ? () => onManage(node) : undefined}
                            clickable={Boolean(onManage)}
                            title="Orphaned vLLM GPU process(es) holding VRAM with no container — click to manage"
                          />
                        )}
                        {node.rogue_artifact_count != null && node.rogue_artifact_count > 0 && (
                          <Chip
                            icon={<WarningAmberRounded sx={{ fontSize: 13 }} />}
                            label={`${node.rogue_artifact_count} orphan cache`}
                            size="small"
                            color="warning"
                            onClick={onManage ? () => onManage(node) : undefined}
                            clickable={Boolean(onManage)}
                            title="Orphaned warm-cache artifacts (RAM sleepers / disk compile caches) — click to manage"
                          />
                        )}
                        {node.warm_offload_enabled &&
                          node.ram_cache_used_mb != null &&
                          node.ram_cache_used_mb > 0 && (
                            <Chip
                              label={`RAM cache ${(node.ram_cache_used_mb / 1024).toFixed(1)}${
                                node.ram_cache_limit_mb
                                  ? `/${(node.ram_cache_limit_mb / 1024).toFixed(0)}`
                                  : ""
                              } GB`}
                              size="small"
                              color="info"
                              onClick={onManage ? () => onManage(node) : undefined}
                              clickable={Boolean(onManage)}
                              title="CPU RAM held by paused (RAM) models — click to manage warm cache"
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
                  )}
                  {visibleKeys.has("heartbeat") && (
                    <TableCell align="right">{node.last_heartbeat_at ?? "-"}</TableCell>
                  )}
                  <TableCell align="right">
                    <Box sx={{ display: "flex", gap: 0.5, justifyContent: "flex-end", alignItems: "center" }}>
                      <NodeActionBtn
                        tooltip={expanded ? "Hide Metrics" : "Metrics"}
                        label="Metrics"
                        icon={<BarChartOutlined sx={NODE_ICON_SX} />}
                        onClick={() => setExpandedNodeId(expanded ? null : node.id)}
                      />
                      {onToggleMaintenance && (
                        <NodeActionBtn
                          tooltip={node.maintenance || node.partial_maintenance ? "Edit Maintenance" : "Maintenance"}
                          label="Maint."
                          icon={<EngineeringOutlined sx={NODE_ICON_SX} />}
                          onClick={() => onToggleMaintenance(node)}
                        />
                      )}
                      <NodeActionBtn
                        tooltip="Manage"
                        icon={<BuildOutlined sx={NODE_ICON_SX} />}
                        onClick={() => onManage?.(node)}
                      />
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

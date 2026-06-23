import { MouseEvent, useEffect, useState } from "react";
import AddIcon from "@mui/icons-material/Add";
import ClearIcon from "@mui/icons-material/Clear";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import SearchIcon from "@mui/icons-material/Search";
import {
  Button,
  Chip,
  IconButton,
  InputAdornment,
  Menu,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  Tooltip,
  Box,
  Skeleton,
  Typography
} from "@mui/material";

import type { Deployment, DeploymentExtension } from "../services/api";
import { useColumnVisibility, type ColumnDef } from "../hooks/useColumnVisibility";
import { AppButton } from "./AppButton";
import { AppDialog } from "./AppDialog";
import { ColumnPicker } from "./ColumnPicker";
import { EmptyState } from "./EmptyState";
import { Mono } from "./Mono";
import { ConfirmDialog } from "./ConfirmDialog";
import { DeploymentActions } from "./DeploymentActions";

type DeploymentTableProps = {
  deployments: Deployment[];
  loading?: boolean;
  onStop: (deploymentId: number) => void;
  onDelete: (deploymentId: number) => void;
  onLogs: (deploymentId: number) => void;
  onSettings: (deployment: Deployment) => void;
  onRestart: (deployment: Deployment) => void;
  onExtend?: (deployment: Deployment, extension: DeploymentExtension) => void;
  onEndpoint?: (deployment: Deployment) => void;
  // Warm cache (only shown for deployments on a warm-offload-enabled node).
  onPause?: (deployment: Deployment) => void;
  onResume?: (deployment: Deployment) => void;
  onPin?: (deployment: Deployment, pinned: boolean) => void;
  isWarmNode?: (nodeId: number) => boolean;
  isUnifiedNode?: (nodeId: number) => boolean;
  nodeNameById: Record<number, string>;
};

const EXTEND_CHOICES: { hours: number; label: string }[] = [
  { hours: 1, label: "+1 hour" },
  { hours: 4, label: "+4 hours" },
  { hours: 12, label: "+12 hours" },
  { hours: 24, label: "+24 hours" }
];

type SortKey =
  | "model"
  | "owner"
  | "node"
  | "port"
  | "vllm"
  | "fraction"
  | "usage"
  | "remaining"
  | "status";

const DEPLOYMENT_COLUMNS: ColumnDef[] = [
  { key: "model",     label: "Model",       alwaysVisible: true },
  { key: "owner",     label: "Owner" },
  { key: "node",      label: "Node" },
  { key: "port",      label: "Port" },
  { key: "vllm",      label: "vLLM" },
  { key: "fraction",  label: "GPU Fraction" },
  { key: "gpus",      label: "GPUs" },
  { key: "args",      label: "Args" },
  { key: "usage",     label: "Usage" },
  { key: "remaining", label: "Remaining" },
  { key: "status",    label: "Status" },
  { key: "actions",   label: "Actions",     alwaysVisible: true },
];

type Remaining = { text: string; urgent: boolean };

const HOUR_MS = 60 * 60 * 1000;

function formatRemaining(deployment: Deployment, now: number): Remaining {
  const status = deployment.status;
  if (status !== "running" && status !== "loading" && status !== "paused_ram") {
    return { text: "—", urgent: false };
  }
  if (deployment.duration_seconds == null) {
    return { text: "∞", urgent: false };
  }
  if (!deployment.expires_at) {
    return { text: "starting…", urgent: false };
  }
  const remainingMs = new Date(deployment.expires_at).getTime() - now;
  if (remainingMs <= 0) {
    return { text: "expiring…", urgent: true };
  }
  const totalMinutes = Math.floor(remainingMs / 60000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  const text = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  return { text, urgent: remainingMs < HOUR_MS };
}

function statusColor(
  status: string
): "success" | "warning" | "error" | "info" | "default" {
  if (status === "running") return "success";
  if (status === "expired") return "warning";
  if (status === "error" || status === "unreachable") return "error";
  if (status === "paused_ram" || status === "offloading") return "info";
  return "default";
}

// Compact token formatting: 1234567 -> "1.2M".
export function formatTokens(value?: number): string {
  if (!value) return "0";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

// Sub-1000 rates keep one decimal; beyond that the decimal is noise.
function formatRate(value: number): string {
  return value >= 1000 ? formatTokens(Math.round(value)) : value.toFixed(1);
}

function usageTooltip(deployment: Deployment): string {
  const lines = [
    `${deployment.total_requests} requests — prompt / completion tokens`
  ];
  const perRequest: string[] = [];
  if (typeof deployment.prompt_tps === "number") {
    perRequest.push(`read ${formatRate(deployment.prompt_tps)}`);
  }
  if (typeof deployment.generation_tps === "number") {
    perRequest.push(`generation ${formatRate(deployment.generation_tps)}`);
  }
  if (perRequest.length > 0) {
    lines.push(
      `Avg per request: ${perRequest.join(" · ")} tok/s (read: prefill · generation: first → last token; excludes queue wait and warm-up request)`
    );
  }
  const throughput: string[] = [];
  if (typeof deployment.prompt_throughput === "number") {
    throughput.push(`read ${formatRate(deployment.prompt_throughput)}`);
  }
  if (typeof deployment.generation_throughput === "number") {
    throughput.push(`generation ${formatRate(deployment.generation_throughput)}`);
  }
  if (throughput.length > 0) {
    lines.push(`Engine throughput: ${throughput.join(" · ")} tok/s (last window)`);
  }
  const queue: string[] = [];
  if (typeof deployment.requests_running === "number") {
    queue.push(`${deployment.requests_running} running`);
  }
  if (typeof deployment.requests_waiting === "number") {
    queue.push(`${deployment.requests_waiting} queued`);
  }
  if (queue.length > 0) {
    lines.push(queue.join(" · "));
  }
  return lines.join("\n");
}

// Show the client-reported load phase while a deployment is loading, so a
// multi-minute start reads as progress rather than a stall.
function statusLabel(deployment: Deployment): string {
  if (
    (deployment.status === "loading" || deployment.status === "starting") &&
    deployment.detail
  ) {
    // Image-pull progress rides along inside the parentheses, e.g.
    // "starting (pulling image · 3.5/21.6 GB)".
    const pull =
      deployment.pull_total_mb && deployment.pull_total_mb > 0
        ? ` · ${((deployment.pull_downloaded_mb ?? 0) / 1024).toFixed(1)}/${(
            deployment.pull_total_mb / 1024
          ).toFixed(1)} GB`
        : "";
    return `${deployment.status} (${deployment.detail.replace(/_/g, " ")}${pull})`;
  }
  if (deployment.status === "paused_ram") return "paused (RAM)";
  if (deployment.status === "offloading") return "offloading to RAM…";
  return deployment.status;
}

export function DeploymentTable({
  deployments,
  loading = false,
  onStop,
  onDelete,
  onLogs,
  onSettings,
  onRestart,
  onExtend,
  onEndpoint,
  onPause,
  onResume,
  onPin,
  isWarmNode,
  isUnifiedNode,
  nodeNameById
}: DeploymentTableProps) {
  const [extendMenu, setExtendMenu] = useState<{
    anchor: HTMLElement;
    deployment: Deployment;
  } | null>(null);

  const openExtendMenu = (event: MouseEvent<HTMLElement>, deployment: Deployment) => {
    setExtendMenu({ anchor: event.currentTarget, deployment });
  };
  const [customExtend, setCustomExtend] = useState<Deployment | null>(null);
  const [customExtendHours, setCustomExtendHours] = useState("6");
  const [confirmStop, setConfirmStop] = useState<Deployment | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<Deployment | null>(null);

  const { visibleKeys, userHidden, toggle: toggleColumn, reset: resetColumns, isCustomized } =
    useColumnVisibility("aquila:columns:deployments", DEPLOYMENT_COLUMNS);
  const columnCount = visibleKeys.size;

  // Tick once a second so the remaining-time countdown and the <1h highlight
  // stay live regardless of how often the deployments query refetches.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const formatArgs = (args?: string[]) => {
    if (!args || args.length === 0) {
      return [];
    }
    const rows: string[] = [];
    let i = 0;
    while (i < args.length) {
      const token = String(args[i]);
      const next = args[i + 1] ? String(args[i + 1]) : "";
      const isFlag = token.startsWith("-");
      const hasValue = next && !next.startsWith("-");
      if (isFlag && hasValue) {
        rows.push(`${token} ${next}`);
        i += 2;
      } else {
        rows.push(token);
        i += 1;
      }
    }
    return rows;
  };

  const canRestart = (status: string) =>
    status === "stopped" || status === "expired" || status === "error";

  // Search + sort are purely presentational, so they live here rather than
  // in the query layer.
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

  const sortValue = (deployment: Deployment): string | number => {
    switch (sortBy) {
      case "model": {
        const served = deployment.engine_args?.served_model_name;
        return (typeof served === "string" && served ? served : deployment.model_name).toLowerCase();
      }
      case "owner":
        return (deployment.owner ?? "").toLowerCase();
      case "node":
        return (
          nodeNameById[deployment.node_id] ?? String(deployment.node_id)
        ).toLowerCase();
      case "port":
        return deployment.port;
      case "vllm":
        return deployment.vllm_version ?? "";
      case "fraction":
        return deployment.gpu_memory_fraction;
      case "usage":
        return (
          (deployment.total_prompt_tokens ?? 0) +
          (deployment.total_completion_tokens ?? 0)
        );
      case "remaining":
        // No expiry (infinite/inactive) sorts after every real deadline.
        return deployment.expires_at
          ? new Date(deployment.expires_at).getTime()
          : Number.MAX_SAFE_INTEGER;
      case "status":
        return deployment.status;
      default:
        return 0;
    }
  };

  const query = search.trim().toLowerCase();
  const visibleRows = deployments.filter((deployment) => {
    if (!query) return true;
    const haystack = [
      deployment.model_name,
      (typeof deployment.engine_args?.served_model_name === "string" && deployment.engine_args.served_model_name) || "",
      deployment.owner ?? "",
      nodeNameById[deployment.node_id] ?? String(deployment.node_id),
      deployment.status,
      String(deployment.port),
      deployment.vllm_version ?? ""
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

  const sortableHeader = (key: SortKey, label: string) => (
    <TableCell sortDirection={sortBy === key ? sortDir : false}>
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
        columns={DEPLOYMENT_COLUMNS}
        userHidden={userHidden}
        onToggle={toggleColumn}
        onReset={resetColumns}
        isCustomized={isCustomized}
      />
      <TextField
        size="small"
        placeholder="Search model, owner, node, status…"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        sx={{ width: { xs: "100%", sm: 280 } }}
        slotProps={{
          input: {
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
          }
        }}
      />
    </Box>
    <TableContainer
      component={Box}
      className="scroll-thin"
      sx={{ minHeight: 200, maxHeight: "70vh", overflowX: "auto" }}
    >
      <Table size="small" stickyHeader sx={{ minWidth: 1040 }}>
        <TableHead>
          <TableRow>
            {sortableHeader("model", "Model")}
            {visibleKeys.has("owner") && sortableHeader("owner", "Owner")}
            {visibleKeys.has("node") && sortableHeader("node", "Node")}
            {visibleKeys.has("port") && sortableHeader("port", "Port")}
            {visibleKeys.has("vllm") && sortableHeader("vllm", "vLLM")}
            {visibleKeys.has("fraction") && sortableHeader("fraction", "GPU Fraction")}
            {visibleKeys.has("gpus") && <TableCell>GPUs</TableCell>}
            {visibleKeys.has("args") && <TableCell>Args</TableCell>}
            {visibleKeys.has("usage") && sortableHeader("usage", "Usage")}
            {visibleKeys.has("remaining") && sortableHeader("remaining", "Remaining")}
            {visibleKeys.has("status") && sortableHeader("status", "Status")}
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {visibleRows.map((deployment) => {
            const argRows = formatArgs(deployment.extra_args);
            const remaining = formatRemaining(deployment, now);
            return (
              <TableRow key={deployment.id} hover>
                <TableCell>
                  <Tooltip title={deployment.model_name} enterDelay={500}>
                    <Box
                      sx={{
                        maxWidth: { xs: 140, sm: 240 },
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap"
                      }}
                    >
                      {(typeof deployment.engine_args?.served_model_name === "string" &&
                        deployment.engine_args.served_model_name) ||
                        deployment.model_name}
                    </Box>
                  </Tooltip>
                </TableCell>
                {visibleKeys.has("owner") && <TableCell>{deployment.owner || "-"}</TableCell>}
                {visibleKeys.has("node") && <TableCell>{nodeNameById[deployment.node_id] ?? deployment.node_id}</TableCell>}
                {visibleKeys.has("port") && <TableCell>{deployment.port}</TableCell>}
                {visibleKeys.has("vllm") && (
                  <TableCell>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.75 }}>
                      {deployment.vllm_version || "-"}
                      {deployment.extra_packages && deployment.extra_packages.length > 0 && (
                        <Chip label={`+${deployment.extra_packages.length} pkg`} size="small" />
                      )}
                    </Box>
                  </TableCell>
                )}
                {visibleKeys.has("fraction") && <TableCell>{deployment.gpu_memory_fraction}</TableCell>}
                {visibleKeys.has("gpus") && (
                  <TableCell>
                    {deployment.gpu_ids && deployment.gpu_ids.length > 0
                      ? deployment.gpu_ids.join(", ")
                      : "-"}
                  </TableCell>
                )}
                {visibleKeys.has("args") && (
                  <TableCell>
                    {argRows.length > 0 ? (
                      <Mono block sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
                        {argRows.map((row, index) => (
                          <Box key={`${deployment.id}-arg-${index}`}>{row}</Box>
                        ))}
                      </Mono>
                    ) : (
                      "-"
                    )}
                  </TableCell>
                )}
                {visibleKeys.has("usage") && (
                  <TableCell>
                    {deployment.total_requests ? (
                      <Tooltip
                        title={
                          <Box sx={{ whiteSpace: "pre-line" }}>
                            {usageTooltip(deployment)}
                          </Box>
                        }
                        enterDelay={500}
                      >
                        <Box>
                          <Typography variant="body2" sx={{ whiteSpace: "nowrap" }}>
                            {formatTokens(deployment.total_prompt_tokens)} /{" "}
                            {formatTokens(deployment.total_completion_tokens)}
                          </Typography>
                          <Typography
                            variant="caption"
                            className="muted"
                            sx={{ whiteSpace: "nowrap" }}
                          >
                            {formatTokens(deployment.total_requests)} req
                            {(typeof deployment.prompt_tps === "number" ||
                              typeof deployment.generation_tps === "number") &&
                              ` · ${[
                                typeof deployment.prompt_tps === "number"
                                  ? `read ${formatRate(deployment.prompt_tps)}`
                                  : null,
                                typeof deployment.generation_tps === "number"
                                  ? `gen ${formatRate(deployment.generation_tps)}`
                                  : null
                              ]
                                .filter(Boolean)
                                .join(" · ")} tok/s`}
                          </Typography>
                        </Box>
                      </Tooltip>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                )}
                {visibleKeys.has("remaining") && (
                  <TableCell>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                      <Typography
                        variant="body2"
                        sx={{
                          color: remaining.urgent ? "error.main" : "inherit",
                          fontWeight: remaining.urgent ? 600 : 400,
                          whiteSpace: "nowrap",
                          fontVariantNumeric: "tabular-nums",
                          minWidth: 64
                        }}
                      >
                        {remaining.text}
                      </Typography>
                      {onExtend &&
                        deployment.duration_seconds != null &&
                        (deployment.status === "running" ||
                          deployment.status === "loading" ||
                          deployment.status === "paused_ram") && (
                          <Tooltip title="Extend serve time" enterDelay={500}>
                            <IconButton
                              size="small"
                              aria-label="Extend serve time"
                              onClick={(event) => openExtendMenu(event, deployment)}
                              sx={{ p: 0.25, color: "text.secondary" }}
                            >
                              <AddIcon sx={{ fontSize: 15 }} />
                            </IconButton>
                          </Tooltip>
                        )}
                    </Box>
                  </TableCell>
                )}
                {visibleKeys.has("status") && (
                  <TableCell>
                    <Tooltip
                      title={deployment.last_error ?? ""}
                      arrow
                      disableHoverListener={!deployment.last_error}
                    >
                      <Box
                        sx={{
                          display: "inline-flex",
                          flexDirection: "column",
                          alignItems: "flex-start",
                          gap: 0.25
                        }}
                      >
                        <Chip
                          label={statusLabel(deployment)}
                          size="small"
                          color={statusColor(deployment.status)}
                        />
                        {deployment.status === "error" && deployment.last_error && (
                          <Typography
                            variant="caption"
                            onClick={() => onLogs(deployment.id)}
                            sx={{
                              color: "error.main",
                              maxWidth: 180,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                              cursor: "pointer",
                              "&:hover": { textDecoration: "underline" }
                            }}
                          >
                            {deployment.last_error}
                          </Typography>
                        )}
                      </Box>
                    </Tooltip>
                  </TableCell>
                )}
                <TableCell align="right" sx={{ py: 0.5 }}>
                  <DeploymentActions
                    deployment={deployment}
                    isWarmNode={isWarmNode}
                    isUnifiedMemory={isUnifiedNode?.(deployment.node_id)}
                    onSettings={() => onSettings(deployment)}
                    onLogs={() => onLogs(deployment.id)}
                    onEndpoint={onEndpoint ? () => onEndpoint(deployment) : undefined}
                    onPin={onPin}
                    onPause={onPause ? () => onPause(deployment) : undefined}
                    onResume={onResume ? () => onResume(deployment) : undefined}
                    onStopClick={() => setConfirmStop(deployment)}
                    onRestart={() => onRestart(deployment)}
                    onDeleteClick={() => setConfirmDelete(deployment)}
                  />
                </TableCell>
              </TableRow>
            );
          })}
          {loading &&
            deployments.length === 0 &&
            [0, 1, 2].map((row) => (
              <TableRow key={`skeleton-${row}`}>
                {Array.from({ length: columnCount }, (_, column) => (
                  <TableCell key={column}>
                    <Skeleton variant="text" />
                  </TableCell>
                ))}
              </TableRow>
            ))}
          {!loading && deployments.length === 0 && (
            <TableRow>
              <TableCell colSpan={columnCount} sx={{ borderBottom: "none" }}>
                <EmptyState
                  icon={RocketLaunchOutlined}
                  primary="No deployments yet."
                  hint="Launch one from the Deploy Model panel."
                />
              </TableCell>
            </TableRow>
          )}
          {!loading && deployments.length > 0 && visibleRows.length === 0 && (
            <TableRow>
              <TableCell colSpan={columnCount} sx={{ borderBottom: "none" }}>
                <EmptyState
                  icon={SearchIcon}
                  primary={`No deployments match "${search.trim()}".`}
                  hint="Clear the search to see all deployments."
                />
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      <Menu
        anchorEl={extendMenu?.anchor ?? null}
        open={extendMenu !== null}
        onClose={() => setExtendMenu(null)}
      >
        {EXTEND_CHOICES.map((choice) => (
          <MenuItem
            key={choice.hours}
            onClick={() => {
              if (extendMenu && onExtend) {
                onExtend(extendMenu.deployment, { hours: choice.hours });
              }
              setExtendMenu(null);
            }}
          >
            {choice.label}
          </MenuItem>
        ))}
        <MenuItem
          onClick={() => {
            if (extendMenu) {
              setCustomExtend(extendMenu.deployment);
              setCustomExtendHours("6");
            }
            setExtendMenu(null);
          }}
        >
          Custom…
        </MenuItem>
        <MenuItem
          onClick={() => {
            if (extendMenu && onExtend) {
              onExtend(extendMenu.deployment, { infinite: true });
            }
            setExtendMenu(null);
          }}
        >
          Infinite
        </MenuItem>
      </Menu>
      <AppDialog
        open={customExtend !== null}
        onClose={() => setCustomExtend(null)}
        title={`Extend ${customExtend?.model_name ?? ""}`}
        maxWidth="xs"
        actions={
          <>
            <AppButton type="button" onClick={() => setCustomExtend(null)}>
              Cancel
            </AppButton>
            <Button
              variant="contained"
              disabled={!(Number(customExtendHours) > 0)}
              onClick={() => {
                if (customExtend && onExtend) {
                  onExtend(customExtend, { hours: Number(customExtendHours) });
                }
                setCustomExtend(null);
              }}
            >
              Extend
            </Button>
          </>
        }
      >
        <TextField
          fullWidth
          autoFocus
          label="Additional hours"
          type="number"
          slotProps={{ htmlInput: { step: 0.5, min: 0.1 } }}
          value={customExtendHours}
          onChange={(event) => setCustomExtendHours(event.target.value)}
          sx={{ mt: 0.5 }}
        />
      </AppDialog>
      <ConfirmDialog
        open={confirmStop !== null}
        title={`Stop ${confirmStop?.model_name ?? ""}?`}
        body={
          confirmStop
            ? `The container on port ${confirmStop.port} will be stopped and removed. ` +
              "The model weights and image stay cached for a fast restart."
            : undefined
        }
        confirmLabel="Stop"
        danger
        onConfirm={() => {
          if (confirmStop) {
            onStop(confirmStop.id);
          }
          setConfirmStop(null);
        }}
        onCancel={() => setConfirmStop(null)}
      />
      <ConfirmDialog
        open={confirmDelete !== null}
        title={`Delete deployment ${confirmDelete?.model_name ?? ""}?`}
        body="This removes the deployment record (and its saved arguments) permanently."
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (confirmDelete) {
            onDelete(confirmDelete.id);
          }
          setConfirmDelete(null);
        }}
        onCancel={() => setConfirmDelete(null)}
      />
    </TableContainer>
    </>
  );
}

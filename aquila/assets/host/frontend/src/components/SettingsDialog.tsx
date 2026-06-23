import { useEffect, useState } from "react";
import {
  Autocomplete,
  Box,
  Checkbox,
  Chip,
  Collapse,
  FormControlLabel,
  IconButton,
  MenuItem,
  Stack,
  Switch,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography
} from "@mui/material";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import EditOutlined from "@mui/icons-material/EditOutlined";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import TuneOutlined from "@mui/icons-material/TuneOutlined";
import VpnKeyOutlined from "@mui/icons-material/VpnKeyOutlined";
import WarningAmber from "@mui/icons-material/WarningAmber";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  createApiKey,
  deleteApiKey,
  fetchApiKeys,
  fetchDeployments,
  fetchSettings,
  purgeDatabase,
  updateApiKey,
  updateSettings,
  type ApiKeyCreated,
  type ApiKeyInfo,
  type Deployment,
  type RuntimeSettings
} from "../services/api";
import { copyToClipboard } from "../services/clipboard";
import { AppButton } from "./AppButton";
import { AppDialog } from "./AppDialog";
import { ConfirmDialog } from "./ConfirmDialog";
import { DialogSection } from "./DialogSection";
import { useToast } from "./ToastProvider";

type SettingsDialogProps = {
  open: boolean;
  onClose: () => void;
};

const DURATION_OPTIONS: { value: string; label: string }[] = [
  { value: "3600", label: "1 hour" },
  { value: "7200", label: "2 hours" },
  { value: "14400", label: "4 hours" },
  { value: "28800", label: "8 hours" },
  { value: "43200", label: "12 hours" },
  { value: "86400", label: "24 hours" },
  { value: "172800", label: "48 hours" },
  { value: "inf", label: "Infinite" }
];

const PURGE_OPTIONS: { key: string; label: string; hint: string }[] = [
  { key: "deployments", label: "Deployment records", hint: "running models re-adopt" },
  { key: "nodes", label: "Nodes", hint: "includes deployments + metric history" },
  { key: "metrics", label: "Metric history", hint: "node charts start fresh" },
  { key: "configs", label: "Saved configurations", hint: "" }
];

const TAB_ICON_SX = { fontSize: 18 } as const;

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

function timeAgo(iso: string): string {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function timeRemaining(iso: string): string {
  const seconds = Math.floor((new Date(iso).getTime() - Date.now()) / 1000);
  if (seconds <= 0) return "expired";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function deploymentDisplayName(d: Deployment): string {
  const served = d.engine_args?.served_model_name;
  const name = typeof served === "string" && served ? served : d.model_name;
  const short = name.split("/").pop() ?? name;
  return `${short} (#${d.id})`;
}

function scopeLabel(
  ids: number[] | null,
  deployments: Deployment[],
): string {
  if (ids === null) return "all";
  if (ids.length === 0) return "none";
  const names = ids.map((id) => {
    const d = deployments.find((dep) => dep.id === id);
    if (!d) return `#${id}`;
    const served = d.engine_args?.served_model_name;
    const name = typeof served === "string" && served ? served : d.model_name;
    return name.split("/").pop() ?? name;
  });
  return names.join(", ");
}

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  const queryClient = useQueryClient();
  const toast = useToast();
  const [draft, setDraft] = useState<Partial<RuntimeSettings>>({});
  const [actionError, setActionError] = useState("");
  const [activeTab, setActiveTab] = useState(0);
  const [purgeTargets, setPurgeTargets] = useState<string[]>([]);
  const [confirmPurge, setConfirmPurge] = useState(false);
  const [createKeyOpen, setCreateKeyOpen] = useState(false);
  const [newKeyLabel, setNewKeyLabel] = useState("");
  const [newKeyScopeAll, setNewKeyScopeAll] = useState(true);
  const [newKeyScopeIds, setNewKeyScopeIds] = useState<Deployment[]>([]);
  const [createdKey, setCreatedKey] = useState<ApiKeyCreated | null>(null);
  const [confirmDeleteKey, setConfirmDeleteKey] = useState<ApiKeyInfo | null>(null);
  const [deleteAck, setDeleteAck] = useState(false);
  const [tempKeysOpen, setTempKeysOpen] = useState(false);
  const [editKey, setEditKey] = useState<ApiKeyInfo | null>(null);
  const [editLabel, setEditLabel] = useState("");
  const [editScopeAll, setEditScopeAll] = useState(true);
  const [editScopeIds, setEditScopeIds] = useState<Deployment[]>([]);

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: fetchSettings,
    enabled: open
  });

  const deploymentsQuery = useQuery({
    queryKey: ["deployments"],
    queryFn: fetchDeployments,
    enabled: open
  });

  useEffect(() => {
    if (open && settingsQuery.data) {
      setDraft(settingsQuery.data);
    }
  }, [open, settingsQuery.data]);

  const saveMutation = useMutation({
    mutationFn: () => updateSettings(draft),
    onSuccess: (data) => {
      queryClient.setQueryData(["settings"], data);
      setActionError("");
      toast.success("Settings saved. Changes apply immediately.");
      onClose();
    },
    onError: (error) => {
      const msg = errorMessage(error);
      setActionError(msg);
      toast.error(msg);
    }
  });

  const purgeMutation = useMutation({
    mutationFn: () => purgeDatabase(purgeTargets),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      queryClient.invalidateQueries({ queryKey: ["configs"] });
      setConfirmPurge(false);
      setPurgeTargets([]);
      setActionError("");
      toast.success("Selected records purged.");
    },
    onError: (error) => {
      setConfirmPurge(false);
      setActionError(errorMessage(error));
    }
  });

  const apiKeysQuery = useQuery({
    queryKey: ["api-keys"],
    queryFn: fetchApiKeys,
    enabled: open
  });

  const createKeyMutation = useMutation({
    mutationFn: (args: { label: string; deploymentIds?: number[] }) =>
      createApiKey(args.label, undefined, args.deploymentIds),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
      setCreatedKey(data);
    },
    onError: (error) => toast.error(errorMessage(error))
  });

  const updateKeyMutation = useMutation({
    mutationFn: (args: { id: number; label?: string; deployment_ids?: number[] | null }) =>
      updateApiKey(args.id, { label: args.label, deployment_ids: args.deployment_ids }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
      setEditKey(null);
      toast.success("API key updated.");
    },
    onError: (error) => {
      toast.error(errorMessage(error));
    }
  });

  const deleteKeyMutation = useMutation({
    mutationFn: (id: number) => deleteApiKey(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
      setConfirmDeleteKey(null);
      toast.success("API key deleted.");
    },
    onError: (error) => {
      setConfirmDeleteKey(null);
      toast.error(errorMessage(error));
    }
  });

  const set = <K extends keyof RuntimeSettings>(key: K, value: RuntimeSettings[K]) =>
    setDraft((current) => ({ ...current, [key]: value }));

  const num = (value: string): number => Number(value);

  const togglePurgeTarget = (key: string, checked: boolean) => {
    setPurgeTargets((current) => {
      let next = checked
        ? [...current, key]
        : current.filter((item) => item !== key);
      if (key === "nodes" && checked) {
        next = Array.from(new Set([...next, "deployments", "metrics"]));
      }
      if ((key === "deployments" || key === "metrics") && !checked) {
        next = next.filter((item) => item !== "nodes");
      }
      return next;
    });
  };

  const handleClose = () => {
    setActionError("");
    setPurgeTargets([]);
    setConfirmPurge(false);
    setActiveTab(0);
    onClose();
  };

  const selectedPurgeLabels = PURGE_OPTIONS.filter((option) =>
    purgeTargets.includes(option.key)
  ).map((option) => option.label.toLowerCase());

  const permanentKeys = (apiKeysQuery.data ?? []).filter((k) => !k.expires_at);
  const isLastPermanentKey =
    confirmDeleteKey !== null &&
    !confirmDeleteKey.expires_at &&
    permanentKeys.length === 1;

  const allDeployments = deploymentsQuery.data ?? [];

  const openCreateDialog = () => {
    setNewKeyLabel("");
    setNewKeyScopeAll(true);
    setNewKeyScopeIds([]);
    setCreatedKey(null);
    setCreateKeyOpen(true);
  };

  const openEditDialog = (k: ApiKeyInfo) => {
    setEditKey(k);
    setEditLabel(k.label);
    if (k.allowed_deployment_ids === null) {
      setEditScopeAll(true);
      setEditScopeIds([]);
    } else {
      setEditScopeAll(false);
      setEditScopeIds(
        allDeployments.filter((d) => k.allowed_deployment_ids!.includes(d.id))
      );
    }
  };

  const handleCreateSubmit = () => {
    const deploymentIds = newKeyScopeAll
      ? undefined
      : newKeyScopeIds.map((d) => d.id);
    createKeyMutation.mutate({ label: newKeyLabel.trim(), deploymentIds });
  };

  const handleEditSubmit = () => {
    if (!editKey) return;
    const deploymentIds = editScopeAll
      ? null
      : editScopeIds.map((d) => d.id);
    updateKeyMutation.mutate({
      id: editKey.id,
      label: editLabel.trim() || undefined,
      deployment_ids: deploymentIds,
    });
  };

  return (
    <>
      <AppDialog
        open={open}
        onClose={handleClose}
        title="Settings"
        meta="Saved values override the backend's environment defaults and apply live — no restart needed."
        contentSx={{ pt: 0 }}
        actions={
          <>
            <AppButton type="button" ghost onClick={handleClose}>
              Cancel
            </AppButton>
            <AppButton
              type="button"
              disabled={saveMutation.isPending || settingsQuery.isLoading}
              onClick={() => saveMutation.mutate()}
            >
              Save
            </AppButton>
          </>
        }
      >
        {actionError && (
          <Typography variant="body2" color="error" sx={{ mt: 2, mb: -1 }}>
            {actionError}
          </Typography>
        )}

        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            borderBottom: "1px solid var(--line)",
            mx: -3,
            px: 3,
          }}
        >
          <Tab icon={<VpnKeyOutlined sx={TAB_ICON_SX} />} iconPosition="start" label="Gateway & Keys" />
          <Tab icon={<RocketLaunchOutlined sx={TAB_ICON_SX} />} iconPosition="start" label="Deployments" />
          <Tab icon={<TuneOutlined sx={TAB_ICON_SX} />} iconPosition="start" label="System" />
          <Tab
            icon={<WarningAmber sx={TAB_ICON_SX} />}
            iconPosition="start"
            label="Danger Zone"
            sx={{ color: "error.main", "&.Mui-selected": { color: "error.main" } }}
          />
        </Tabs>

        <Box sx={{ pt: 2 }}>
          {/* ── Tab 0: Gateway & Keys ── */}
          {activeTab === 0 && (
            <Box>
              <DialogSection
                first
                title="Gateway"
                hint="When disabled, /v1 requests get a 503; direct node URLs keep working."
                action={
                  <Switch
                    checked={draft.gateway_enabled ?? true}
                    onChange={(event) => set("gateway_enabled", event.target.checked)}
                    slotProps={{ input: { "aria-label": "OpenAI gateway enabled" } }}
                  />
                }
              >
                <TextField
                  size="small"
                  label="Request timeout (s)"
                  type="number"
                  value={draft.gateway_timeout_seconds ?? ""}
                  onChange={(event) => set("gateway_timeout_seconds", num(event.target.value))}
                  helperText="Non-streaming requests; streams are never read-limited."
                  sx={{ width: { xs: "100%", sm: 220 } }}
                />
              </DialogSection>

              <DialogSection
                title="API Keys"
                hint={(() => {
                  const permanent = (apiKeysQuery.data ?? []).filter((k) => !k.expires_at);
                  return permanent.length > 0
                    ? `${permanent.length} active key${permanent.length > 1 ? "s" : ""}. All /v1 requests require a valid key.`
                    : "No API keys — the gateway is open to all requests.";
                })()}
                action={
                  <AppButton type="button" onClick={openCreateDialog}>
                    Create Key
                  </AppButton>
                }
              >
                {(() => {
                  const allKeys = apiKeysQuery.data ?? [];
                  const permanent = allKeys.filter((k) => !k.expires_at);
                  const temporary = allKeys.filter((k) => !!k.expires_at);

                  return (
                    <>
                      {permanent.length > 0 ? (
                        <Stack spacing={0.5}>
                          {permanent.map((k) => (
                            <Box
                              key={k.id}
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 1.5,
                                py: 0.5,
                                flexWrap: "wrap"
                              }}
                            >
                              <Typography variant="body2" sx={{ fontWeight: 500, minWidth: { xs: "auto", sm: 100 } }}>
                                {k.label}
                              </Typography>
                              <Typography
                                variant="body2"
                                className="muted"
                                sx={{ fontFamily: "monospace", fontSize: "0.8rem" }}
                              >
                                {k.prefix}...
                              </Typography>
                              <Typography variant="caption" className="muted">
                                {scopeLabel(k.allowed_deployment_ids, allDeployments)}
                              </Typography>
                              <Typography variant="caption" className="muted" sx={{ ml: { xs: 0, sm: "auto" } }}>
                                {k.last_used_at
                                  ? `used ${timeAgo(k.last_used_at)}`
                                  : "never used"}
                              </Typography>
                              <Tooltip title="Edit key">
                                <IconButton
                                  size="small"
                                  onClick={() => openEditDialog(k)}
                                >
                                  <EditOutlined fontSize="small" />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Delete key">
                                <IconButton
                                  size="small"
                                  onClick={() => setConfirmDeleteKey(k)}
                                >
                                  <DeleteOutlineIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          ))}
                        </Stack>
                      ) : (
                        <Typography variant="body2" className="muted">
                          Create an API key to require authentication on /v1 gateway requests.
                        </Typography>
                      )}

                      {temporary.length > 0 && (
                        <Box sx={{ mt: 1.5 }}>
                          <Box
                            onClick={() => setTempKeysOpen((v) => !v)}
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 0.5,
                              cursor: "pointer",
                              userSelect: "none",
                              "&:hover": { opacity: 0.8 }
                            }}
                          >
                            <ExpandMoreIcon
                              fontSize="small"
                              className="muted"
                              sx={{
                                transform: tempKeysOpen ? "rotate(0deg)" : "rotate(-90deg)",
                                transition: "transform 150ms"
                              }}
                            />
                            <Typography variant="caption" className="muted">
                              {temporary.length} temporary key{temporary.length > 1 ? "s" : ""}
                            </Typography>
                          </Box>
                          <Collapse in={tempKeysOpen} timeout={150}>
                            <Stack spacing={0.5} sx={{ mt: 0.5, pl: 3 }}>
                              {temporary.map((k) => (
                                <Box
                                  key={k.id}
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 1.5,
                                    py: 0.25,
                                    flexWrap: "wrap"
                                  }}
                                >
                                  <Typography variant="caption" sx={{ minWidth: { xs: "auto", sm: 80 } }}>
                                    {k.label}
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    className="muted"
                                    sx={{ fontFamily: "monospace", fontSize: "0.75rem" }}
                                  >
                                    {k.prefix}...
                                  </Typography>
                                  {k.allowed_deployment_ids !== null && (
                                    <Typography variant="caption" className="muted">
                                      {scopeLabel(k.allowed_deployment_ids, allDeployments)}
                                    </Typography>
                                  )}
                                  <Typography variant="caption" className="muted" sx={{ ml: { xs: 0, sm: "auto" } }}>
                                    {timeRemaining(k.expires_at!)} left
                                  </Typography>
                                  <Tooltip title="Delete key">
                                    <IconButton
                                      size="small"
                                      onClick={() => setConfirmDeleteKey(k)}
                                      sx={{ p: 0.25 }}
                                    >
                                      <DeleteOutlineIcon sx={{ fontSize: 14 }} />
                                    </IconButton>
                                  </Tooltip>
                                </Box>
                              ))}
                            </Stack>
                          </Collapse>
                        </Box>
                      )}

                      <TextField
                        size="small"
                        label="Snippet key lifetime (s)"
                        type="number"
                        slotProps={{ htmlInput: { min: 0, max: 3600 } }}
                        value={draft.temp_api_key_ttl_seconds ?? ""}
                        onChange={(event) => set("temp_api_key_ttl_seconds", num(event.target.value))}
                        helperText="Temporary key lifespan for endpoint code snippets. 0 = disabled."
                        sx={{ width: { xs: "100%", sm: 220 }, mt: 2 }}
                      />
                    </>
                  );
                })()}
              </DialogSection>
            </Box>
          )}

          {/* ── Tab 1: Deployments ── */}
          {activeTab === 1 && (
            <Box>
              <DialogSection
                first
                title="Deployments"
                hint="Start watchdog, runtime preference, and the deploy form's pre-filled defaults."
              >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
                  <TextField
                    size="small"
                    label="Start timeout (s)"
                    type="number"
                    value={draft.start_timeout_seconds ?? ""}
                    onChange={(event) => set("start_timeout_seconds", num(event.target.value))}
                    helperText="Mark a deployment as errored if it isn't running by then."
                    sx={{ width: { xs: "100%", sm: 220 } }}
                  />
                  <TextField
                    size="small"
                    select
                    label="Preferred runtime"
                    value={draft.preferred_container_runtime ?? "docker"}
                    onChange={(event) =>
                      set("preferred_container_runtime", event.target.value)
                    }
                    helperText="Used when a node has both and no per-node override."
                    sx={{ width: { xs: "100%", sm: 200 } }}
                  >
                    <MenuItem value="docker">Docker</MenuItem>
                    <MenuItem value="podman">Podman</MenuItem>
                  </TextField>
                </Stack>
                <Typography variant="body2" className="muted" sx={{ mb: 1 }}>
                  Defaults pre-filled in the deploy form:
                </Typography>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
                  <TextField
                    size="small"
                    label="Port"
                    type="number"
                    value={draft.default_port ?? ""}
                    onChange={(event) => set("default_port", num(event.target.value))}
                    sx={{ width: { xs: "100%", sm: 130 } }}
                  />
                  <TextField
                    size="small"
                    label="GPU fraction"
                    type="number"
                    slotProps={{ htmlInput: { step: 0.05, min: 0.05, max: 1 } }}
                    value={draft.default_gpu_fraction ?? ""}
                    onChange={(event) => set("default_gpu_fraction", num(event.target.value))}
                    sx={{ width: { xs: "100%", sm: 130 } }}
                  />
                  <TextField
                    size="small"
                    select
                    label="Serve for"
                    value={draft.default_duration_choice ?? "43200"}
                    onChange={(event) => set("default_duration_choice", event.target.value)}
                    sx={{ width: { xs: "100%", sm: 150 } }}
                  >
                    {DURATION_OPTIONS.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Stack>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                  <TextField
                    size="small"
                    label="vLLM version"
                    placeholder="empty = latest stable"
                    value={draft.default_vllm_version ?? ""}
                    onChange={(event) => set("default_vllm_version", event.target.value)}
                    sx={{ width: { xs: "100%", sm: 220 } }}
                  />
                  <TextField
                    size="small"
                    label="Max failed restarts"
                    type="number"
                    placeholder="client default"
                    value={draft.default_max_failed_restarts ?? ""}
                    onChange={(event) =>
                      set(
                        "default_max_failed_restarts",
                        event.target.value === "" ? null : num(event.target.value)
                      )
                    }
                    sx={{ width: { xs: "100%", sm: 180 } }}
                  />
                </Stack>
              </DialogSection>

              <DialogSection
                title="Notifications"
                hint="Webhook messages for ready / failed / expiring deployments. Slack URLs get Slack formatting automatically."
              >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                  <TextField
                    size="small"
                    label="Webhook URL"
                    placeholder="empty = notifications off"
                    value={draft.webhook_url ?? ""}
                    onChange={(event) => set("webhook_url", event.target.value)}
                    helperText="Slack webhook URLs get Slack formatting automatically."
                    sx={{ flex: 1, minWidth: { xs: 180, sm: 260 } }}
                  />
                  <TextField
                    size="small"
                    label="Expiry warning (min)"
                    type="number"
                    value={draft.expiry_warning_minutes ?? ""}
                    onChange={(event) => set("expiry_warning_minutes", num(event.target.value))}
                    sx={{ width: { xs: "100%", sm: 180 } }}
                  />
                </Stack>
              </DialogSection>
            </Box>
          )}

          {/* ── Tab 2: System ── */}
          {activeTab === 2 && (
            <Box>
              <DialogSection
                first
                title="Data"
                hint="How long node metric history is kept for the charts."
              >
                <TextField
                  size="small"
                  label="Metric history retention (h)"
                  type="number"
                  value={draft.node_metrics_retention_hours ?? ""}
                  onChange={(event) =>
                    set("node_metrics_retention_hours", num(event.target.value))
                  }
                  sx={{ width: { xs: "100%", sm: 220 } }}
                />
              </DialogSection>

              <DialogSection
                title="Warm Cache"
                hint="Controls how GPU models are swapped in and out of VRAM."
              >
                <FormControlLabel
                  control={
                    <Switch
                      checked={Boolean(draft.default_warm_offload_enabled)}
                      onChange={(event) =>
                        set("default_warm_offload_enabled", event.target.checked)
                      }
                    />
                  }
                  label="Enable warm cache by default on new nodes"
                />
                <TextField
                  size="small"
                  label="Busy guard (s)"
                  type="number"
                  slotProps={{ htmlInput: { min: 0, max: 300 } }}
                  value={draft.busy_guard_seconds ?? ""}
                  onChange={(event) => set("busy_guard_seconds", num(event.target.value))}
                  helperText="Seconds after a model's last request before it can be auto-evicted. 0 = evict immediately when idle."
                  sx={{ width: { xs: "100%", sm: 220 } }}
                />
              </DialogSection>

              <DialogSection
                title="Sync & Thresholds"
                hint="Sync tuning — the defaults are sensible; changes apply live."
              >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
                  <TextField
                    size="small"
                    label="Node sync (s)"
                    type="number"
                    value={draft.nodes_sync_interval_seconds ?? ""}
                    onChange={(event) =>
                      set("nodes_sync_interval_seconds", num(event.target.value))
                    }
                    sx={{ width: { xs: "100%", sm: 150 } }}
                  />
                  <TextField
                    size="small"
                    label="Deployment sync (s)"
                    type="number"
                    value={draft.deployments_sync_interval_seconds ?? ""}
                    onChange={(event) =>
                      set("deployments_sync_interval_seconds", num(event.target.value))
                    }
                    sx={{ width: { xs: "100%", sm: 170 } }}
                  />
                  <TextField
                    size="small"
                    label="Expiry check (s)"
                    type="number"
                    value={draft.expiry_check_interval_seconds ?? ""}
                    onChange={(event) =>
                      set("expiry_check_interval_seconds", num(event.target.value))
                    }
                    sx={{ width: { xs: "100%", sm: 150 } }}
                  />
                </Stack>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                  <TextField
                    size="small"
                    label="Node failure threshold"
                    type="number"
                    value={draft.node_failure_threshold ?? ""}
                    onChange={(event) =>
                      set("node_failure_threshold", num(event.target.value))
                    }
                    helperText="Consecutive failures before a node turns critical."
                    sx={{ width: { xs: "100%", sm: 220 } }}
                  />
                  <TextField
                    size="small"
                    label="Deployment failure threshold"
                    type="number"
                    value={draft.deployment_failure_threshold ?? ""}
                    onChange={(event) =>
                      set("deployment_failure_threshold", num(event.target.value))
                    }
                    helperText="Unreachable polls before deployments degrade."
                    sx={{ width: { xs: "100%", sm: 250 } }}
                  />
                </Stack>
              </DialogSection>
            </Box>
          )}

          {/* ── Tab 3: Danger Zone ── */}
          {activeTab === 3 && (
            <Box>
              <DialogSection
                first
                title="Purge"
                hint="Purge selected records. Running models are not stopped — active nodes re-register and their deployments are re-adopted automatically."
              >
                <Box sx={{ display: "flex", flexDirection: "column", mb: 1 }}>
                  {PURGE_OPTIONS.map((option) => (
                    <FormControlLabel
                      key={option.key}
                      control={
                        <Checkbox
                          size="small"
                          checked={purgeTargets.includes(option.key)}
                          onChange={(event) =>
                            togglePurgeTarget(option.key, event.target.checked)
                          }
                        />
                      }
                      label={
                        <Typography variant="body2">
                          {option.label}
                          {option.hint && (
                            <Typography component="span" variant="caption" className="muted">
                              {" "}
                              — {option.hint}
                            </Typography>
                          )}
                        </Typography>
                      }
                    />
                  ))}
                </Box>
                <AppButton
                  type="button"
                  variant="stop"
                  disabled={purgeTargets.length === 0 || purgeMutation.isPending}
                  onClick={() => setConfirmPurge(true)}
                >
                  Purge Selected
                </AppButton>
              </DialogSection>
            </Box>
          )}
        </Box>
      </AppDialog>

      <ConfirmDialog
        open={confirmPurge}
        title="Purge selected records?"
        body={`This deletes: ${selectedPurgeLabels.join(", ")}. It cannot be undone. Running models are not stopped.`}
        confirmLabel="Purge"
        danger
        onConfirm={() => purgeMutation.mutate()}
        onCancel={() => setConfirmPurge(false)}
      />

      {/* ── Create Key Dialog ── */}
      <AppDialog
        open={createKeyOpen}
        onClose={() => setCreateKeyOpen(false)}
        maxWidth="xs"
        title={createdKey ? "API Key Created" : "Create API Key"}
        actions={
          createdKey ? (
            <AppButton type="button" onClick={() => setCreateKeyOpen(false)}>
              Done
            </AppButton>
          ) : (
            <>
              <AppButton type="button" ghost onClick={() => setCreateKeyOpen(false)}>
                Cancel
              </AppButton>
              <AppButton
                type="button"
                disabled={!newKeyLabel.trim() || createKeyMutation.isPending}
                onClick={handleCreateSubmit}
              >
                Create
              </AppButton>
            </>
          )
        }
      >
        {createdKey ? (
          <Stack spacing={2}>
            <Typography variant="body2">
              Copy this key now. It will not be shown again.
            </Typography>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                p: 1.5,
                borderRadius: 1,
                bgcolor: "action.hover",
                fontFamily: "monospace",
                fontSize: "0.85rem",
                wordBreak: "break-all"
              }}
            >
              <Typography sx={{ fontFamily: "inherit", fontSize: "inherit", flex: 1 }}>
                {createdKey.key}
              </Typography>
              <Tooltip title="Copy to clipboard">
                <IconButton
                  size="small"
                  onClick={async () => {
                    await copyToClipboard(createdKey.key);
                    toast.success("Key copied to clipboard.");
                  }}
                >
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            {createdKey.allowed_deployment_ids !== null && (
              <Typography variant="caption" className="muted">
                Scoped to: {scopeLabel(createdKey.allowed_deployment_ids, allDeployments)}
              </Typography>
            )}
          </Stack>
        ) : (
          <Stack spacing={2}>
            <TextField
              autoFocus
              size="small"
              label="Label"
              placeholder='e.g. "laptop", "CI pipeline"'
              value={newKeyLabel}
              onChange={(e) => setNewKeyLabel(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && newKeyLabel.trim()) {
                  handleCreateSubmit();
                }
              }}
              fullWidth
              helperText="A short name to identify this key."
            />
            <FormControlLabel
              control={
                <Switch
                  checked={newKeyScopeAll}
                  onChange={(e) => setNewKeyScopeAll(e.target.checked)}
                />
              }
              label="Access all deployments"
            />
            {!newKeyScopeAll && (
              <Autocomplete
                multiple
                size="small"
                options={allDeployments}
                getOptionLabel={deploymentDisplayName}
                value={newKeyScopeIds}
                onChange={(_, value) => setNewKeyScopeIds(value)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Allowed deployments"
                    placeholder="Select deployments"
                  />
                )}
              />
            )}
          </Stack>
        )}
      </AppDialog>

      {/* ── Edit Key Dialog ── */}
      <AppDialog
        open={editKey !== null}
        onClose={() => setEditKey(null)}
        maxWidth="xs"
        title="Edit API Key"
        actions={
          <>
            <AppButton type="button" ghost onClick={() => setEditKey(null)}>
              Cancel
            </AppButton>
            <AppButton
              type="button"
              disabled={!editLabel.trim() || updateKeyMutation.isPending}
              onClick={handleEditSubmit}
            >
              Save
            </AppButton>
          </>
        }
      >
        <Stack spacing={2}>
          <TextField
            autoFocus
            size="small"
            label="Label"
            value={editLabel}
            onChange={(e) => setEditLabel(e.target.value)}
            fullWidth
          />
          <FormControlLabel
            control={
              <Switch
                checked={editScopeAll}
                onChange={(e) => setEditScopeAll(e.target.checked)}
              />
            }
            label="Access all deployments"
          />
          {!editScopeAll && (
            <Autocomplete
              multiple
              size="small"
              options={allDeployments}
              getOptionLabel={deploymentDisplayName}
              value={editScopeIds}
              onChange={(_, value) => setEditScopeIds(value)}
              isOptionEqualToValue={(option, value) => option.id === value.id}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Allowed deployments"
                  placeholder="Select deployments"
                />
              )}
            />
          )}
        </Stack>
      </AppDialog>

      {/* ── Delete Key Confirm ── */}
      <AppDialog
        open={confirmDeleteKey !== null}
        onClose={() => { setConfirmDeleteKey(null); setDeleteAck(false); }}
        maxWidth="xs"
        title="Delete API key?"
        actions={
          <>
            <AppButton type="button" onClick={() => { setConfirmDeleteKey(null); setDeleteAck(false); }}>
              Cancel
            </AppButton>
            <AppButton
              type="button"
              variant="stop"
              disabled={isLastPermanentKey && !deleteAck}
              onClick={() => {
                if (confirmDeleteKey) deleteKeyMutation.mutate(confirmDeleteKey.id);
                setDeleteAck(false);
              }}
            >
              Delete
            </AppButton>
          </>
        }
      >
        {isLastPermanentKey ? (
          <Stack spacing={1.5}>
            <Typography variant="body2">
              This is the last API key (&ldquo;{confirmDeleteKey?.label}&rdquo;).
              Deleting it will leave the gateway completely unprotected &mdash;
              anyone with network access can send requests without authentication.
            </Typography>
            <FormControlLabel
              control={
                <Checkbox
                  size="small"
                  checked={deleteAck}
                  onChange={(e) => setDeleteAck(e.target.checked)}
                />
              }
              label={
                <Typography variant="body2">
                  I understand the gateway will be open to all requests
                </Typography>
              }
            />
          </Stack>
        ) : (
          <Typography variant="body2">
            {confirmDeleteKey
              ? `Delete key "${confirmDeleteKey.label}" (${confirmDeleteKey.prefix}...)? Clients using this key will lose access.`
              : ""}
          </Typography>
        )}
      </AppDialog>
    </>
  );
}

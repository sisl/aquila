import { useEffect, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Checkbox,
  FormControlLabel,
  MenuItem,
  Stack,
  Switch,
  TextField,
  Typography
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  fetchSettings,
  purgeDatabase,
  updateSettings,
  type RuntimeSettings
} from "../services/api";
import { AppButton } from "./AppButton";
import { AppDialog } from "./AppDialog";
import { ConfirmDialog } from "./ConfirmDialog";
import { SectionLabel } from "./SectionLabel";
import { useToast } from "./ToastProvider";

type SettingsDialogProps = {
  open: boolean;
  onClose: () => void;
};

// Mirrors the deploy form's options minus "custom" (a default must be concrete).
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

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  const queryClient = useQueryClient();
  const toast = useToast();
  const [draft, setDraft] = useState<Partial<RuntimeSettings>>({});
  const [actionError, setActionError] = useState("");
  const [purgeTargets, setPurgeTargets] = useState<string[]>([]);
  const [confirmPurge, setConfirmPurge] = useState(false);

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: fetchSettings,
    enabled: open
  });

  // Re-seed the draft each time the dialog opens with fresh data.
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
    onError: (error) => setActionError(errorMessage(error))
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

  const set = <K extends keyof RuntimeSettings>(key: K, value: RuntimeSettings[K]) =>
    setDraft((current) => ({ ...current, [key]: value }));

  const num = (value: string): number => Number(value);

  const togglePurgeTarget = (key: string, checked: boolean) => {
    setPurgeTargets((current) => {
      let next = checked
        ? [...current, key]
        : current.filter((item) => item !== key);
      // Nodes can't outlive their children: force-select what the backend
      // will delete anyway so the confirmation is honest.
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
    onClose();
  };

  const selectedPurgeLabels = PURGE_OPTIONS.filter((option) =>
    purgeTargets.includes(option.key)
  ).map((option) => option.label.toLowerCase());

  return (
    <>
      <AppDialog
        open={open}
        onClose={handleClose}
        title="Settings"
        meta="Saved values override the backend's environment defaults and apply live — no restart needed."
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
          <Typography variant="body2" color="error" sx={{ mb: 2 }}>
            {actionError}
          </Typography>
        )}

        <Stack spacing={3}>
          {/* Gateway ----------------------------------------------------- */}
          <Box>
            <SectionLabel sx={{ mb: 1 }}>Gateway</SectionLabel>
            <FormControlLabel
              control={
                <Switch
                  checked={draft.gateway_enabled ?? true}
                  onChange={(event) => set("gateway_enabled", event.target.checked)}
                />
              }
              label="OpenAI gateway enabled"
            />
            <Typography variant="caption" className="muted" sx={{ display: "block", mb: 1.5 }}>
              When disabled, /v1 requests get a 503; direct node URLs keep working.
            </Typography>
            <TextField
              size="small"
              label="Request timeout (s)"
              type="number"
              value={draft.gateway_timeout_seconds ?? ""}
              onChange={(event) => set("gateway_timeout_seconds", num(event.target.value))}
              helperText="Non-streaming requests; streams are never read-limited."
              sx={{ width: 220 }}
            />
          </Box>

          {/* Deployments --------------------------------------------------- */}
          <Box>
            <SectionLabel sx={{ mb: 1 }}>Deployments</SectionLabel>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
              <TextField
                size="small"
                label="Start timeout (s)"
                type="number"
                value={draft.start_timeout_seconds ?? ""}
                onChange={(event) => set("start_timeout_seconds", num(event.target.value))}
                helperText="Mark a deployment as errored if it isn't running by then."
                sx={{ width: 220 }}
              />
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
                sx={{ width: 130 }}
              />
              <TextField
                size="small"
                label="GPU fraction"
                type="number"
                inputProps={{ step: 0.05, min: 0.05, max: 1 }}
                value={draft.default_gpu_fraction ?? ""}
                onChange={(event) => set("default_gpu_fraction", num(event.target.value))}
                sx={{ width: 130 }}
              />
              <TextField
                size="small"
                select
                label="Serve for"
                value={draft.default_duration_choice ?? "43200"}
                onChange={(event) => set("default_duration_choice", event.target.value)}
                sx={{ width: 150 }}
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
                sx={{ width: 220 }}
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
                sx={{ width: 180 }}
              />
            </Stack>
          </Box>

          {/* Notifications ----------------------------------------------- */}
          <Box>
            <SectionLabel sx={{ mb: 1 }}>Notifications</SectionLabel>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <TextField
                size="small"
                label="Webhook URL"
                placeholder="empty = notifications off"
                value={draft.webhook_url ?? ""}
                onChange={(event) => set("webhook_url", event.target.value)}
                helperText="Slack webhook URLs get Slack formatting automatically."
                sx={{ flex: 1, minWidth: 260 }}
              />
              <TextField
                size="small"
                label="Expiry warning (min)"
                type="number"
                value={draft.expiry_warning_minutes ?? ""}
                onChange={(event) => set("expiry_warning_minutes", num(event.target.value))}
                sx={{ width: 180 }}
              />
            </Stack>
          </Box>

          {/* Data ----------------------------------------------------------- */}
          <Box>
            <SectionLabel sx={{ mb: 1 }}>Data</SectionLabel>
            <TextField
              size="small"
              label="Metric history retention (h)"
              type="number"
              value={draft.node_metrics_retention_hours ?? ""}
              onChange={(event) =>
                set("node_metrics_retention_hours", num(event.target.value))
              }
              sx={{ width: 220, mb: 2 }}
            />
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
              Danger zone
            </Typography>
            <Typography variant="caption" className="muted" sx={{ display: "block", mb: 1 }}>
              Purge selected records. Running models are not stopped — active nodes
              re-register and their deployments are re-adopted automatically.
            </Typography>
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
          </Box>

          {/* Advanced ------------------------------------------------------ */}
          <Accordion disableGutters elevation={0}>
            <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />}>
              <SectionLabel>Advanced</SectionLabel>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="caption" className="muted" sx={{ display: "block", mb: 1.5 }}>
                Sync tuning — the defaults are sensible; changes apply live.
              </Typography>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
                <TextField
                  size="small"
                  label="Node sync (s)"
                  type="number"
                  value={draft.nodes_sync_interval_seconds ?? ""}
                  onChange={(event) =>
                    set("nodes_sync_interval_seconds", num(event.target.value))
                  }
                  sx={{ width: 150 }}
                />
                <TextField
                  size="small"
                  label="Deployment sync (s)"
                  type="number"
                  value={draft.deployments_sync_interval_seconds ?? ""}
                  onChange={(event) =>
                    set("deployments_sync_interval_seconds", num(event.target.value))
                  }
                  sx={{ width: 170 }}
                />
                <TextField
                  size="small"
                  label="Expiry check (s)"
                  type="number"
                  value={draft.expiry_check_interval_seconds ?? ""}
                  onChange={(event) =>
                    set("expiry_check_interval_seconds", num(event.target.value))
                  }
                  sx={{ width: 150 }}
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
                  sx={{ width: 220 }}
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
                  sx={{ width: 250 }}
                />
              </Stack>
            </AccordionDetails>
          </Accordion>
        </Stack>
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
    </>
  );
}

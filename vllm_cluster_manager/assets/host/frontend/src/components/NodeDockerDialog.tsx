import { useEffect, useRef, useState } from "react";
import {
  Box,
  Chip,
  CircularProgress,
  Divider,
  FormControlLabel,
  IconButton,
  LinearProgress,
  MenuItem,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography
} from "@mui/material";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  deleteLocalModel,
  deleteNode,
  deleteNodeImage,
  setNodeRuntime,
  setNodeWarmCache,
  deleteNodeModelCache,
  deleteNodeWarmCache,
  fetchLocalModels,
  fetchLocalModelTransfers,
  fetchNodeContainers,
  fetchNodeGpuProcesses,
  fetchNodeImages,
  fetchNodeModelCache,
  fetchNodeWarmArtifacts,
  killNodeGpuProcess,
  killNodeRamSleeper,
  pruneNodeImages,
  pullLocalModel,
  stopNodeContainer,
  uploadLocalModelArchive,
  uploadLocalModelFolder,
  type GpuProcess,
  type ImagePruneResult,
  type LocalModelUploadResult,
  type Node
} from "../services/api";
import { copyToClipboard } from "../services/clipboard";
import { AppButton } from "./AppButton";
import { AppDialog } from "./AppDialog";
import { Mono } from "./Mono";
import { ConfirmDialog } from "./ConfirmDialog";
import { DialogSection } from "./DialogSection";
import { useToast } from "./ToastProvider";

type NodeDockerDialogProps = {
  node: Node | null;
  open: boolean;
  onClose: () => void;
};

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

const MODEL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

// Mirror of the agent's name rule so 400s are caught before any bytes move.
function suggestModelName(raw: string): string {
  const cleaned = raw
    .replace(/\.(tar\.gz|tgz|zip)$/i, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^[._-]+/, "")
    .slice(0, 64);
  return cleaned;
}

type PendingUpload = {
  kind: "folder" | "archive";
  files: File[];
  name: string;
};

export function NodeDockerDialog({ node, open, onClose }: NodeDockerDialogProps) {
  const queryClient = useQueryClient();
  const toast = useToast();
  const copyName = async (name: string) => {
    try {
      await copyToClipboard(name);
      toast.success("Model name copied.");
    } catch {
      toast.error("Clipboard unavailable.");
    }
  };
  const nodeId = node?.id ?? null;
  const enabled = open && nodeId !== null;
  const [actionError, setActionError] = useState("");
  const [pruneResult, setPruneResult] = useState<ImagePruneResult | null>(null);
  const [confirmStopId, setConfirmStopId] = useState<string | null>(null);
  const [confirmKillPid, setConfirmKillPid] = useState<number | null>(null);
  const [confirmSleeperPid, setConfirmSleeperPid] = useState<number | null>(null);
  const [confirmCacheName, setConfirmCacheName] = useState<string | null>(null);
  // RAM-cache limit draft (GB; "" = unlimited), synced from the node.
  const [ramLimitGb, setRamLimitGb] = useState("");
  const [confirmImageId, setConfirmImageId] = useState<string | null>(null);
  const [confirmPrune, setConfirmPrune] = useState(false);
  const [confirmModelName, setConfirmModelName] = useState<string | null>(null);
  const [confirmLocalModel, setConfirmLocalModel] = useState<string | null>(null);
  const [confirmRemoveNode, setConfirmRemoveNode] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);
  const [uploadProgress, setUploadProgress] = useState<
    { name: string; fraction: number } | null
  >(null);
  const [uploadNotice, setUploadNotice] = useState("");
  const [showPullForm, setShowPullForm] = useState(false);
  const [pullUrl, setPullUrl] = useState("");
  const [pullName, setPullName] = useState("");
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const archiveInputRef = useRef<HTMLInputElement | null>(null);

  const containersQuery = useQuery({
    queryKey: ["node-containers", nodeId],
    queryFn: () => fetchNodeContainers(nodeId as number),
    enabled,
    refetchInterval: 5000
  });

  const gpuProcessesQuery = useQuery({
    queryKey: ["node-gpu-processes", nodeId],
    queryFn: () => fetchNodeGpuProcesses(nodeId as number),
    enabled,
    refetchInterval: 5000
  });

  const warmArtifactsQuery = useQuery({
    queryKey: ["node-warm-artifacts", nodeId],
    queryFn: () => fetchNodeWarmArtifacts(nodeId as number),
    enabled: enabled && Boolean(node?.warm_offload_enabled),
    refetchInterval: 10000
  });

  const imagesQuery = useQuery({
    queryKey: ["node-images", nodeId],
    queryFn: () => fetchNodeImages(nodeId as number),
    enabled,
    refetchInterval: 5000
  });

  const modelCacheQuery = useQuery({
    queryKey: ["node-model-cache", nodeId],
    queryFn: () => fetchNodeModelCache(nodeId as number),
    enabled,
    refetchInterval: 30000
  });

  const localModelsQuery = useQuery({
    queryKey: ["node-local-models", nodeId],
    queryFn: () => fetchLocalModels(nodeId as number),
    enabled,
    refetchInterval: 30000
  });

  const transfersQuery = useQuery({
    queryKey: ["node-local-model-transfers", nodeId],
    queryFn: () => fetchLocalModelTransfers(nodeId as number),
    enabled,
    refetchInterval: (query) =>
      (query.state.data ?? []).some(
        (t) => t.status === "downloading" || t.status === "extracting"
      )
        ? 2000
        : 15000
  });

  // Keep the RAM-limit field in sync with the node's persisted value.
  useEffect(() => {
    setRamLimitGb(
      node?.ram_cache_limit_mb ? String(Math.round(node.ram_cache_limit_mb / 1024)) : ""
    );
  }, [nodeId, node?.ram_cache_limit_mb]);

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["node-containers", nodeId] });
    queryClient.invalidateQueries({ queryKey: ["node-gpu-processes", nodeId] });
    queryClient.invalidateQueries({ queryKey: ["node-warm-artifacts", nodeId] });
    queryClient.invalidateQueries({ queryKey: ["node-images", nodeId] });
    queryClient.invalidateQueries({ queryKey: ["node-model-cache", nodeId] });
    queryClient.invalidateQueries({ queryKey: ["node-local-models", nodeId] });
    queryClient.invalidateQueries({ queryKey: ["node-local-model-transfers", nodeId] });
  };

  const stopMutation = useMutation({
    mutationFn: (containerId: string) => stopNodeContainer(nodeId as number, containerId),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const killProcessMutation = useMutation({
    mutationFn: (pid: number) => killNodeGpuProcess(nodeId as number, pid),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const warmCacheMutation = useMutation({
    mutationFn: (vars: { enabled: boolean; ramCacheLimitMb: number | null }) =>
      setNodeWarmCache(nodeId as number, vars.enabled, vars.ramCacheLimitMb),
    onSuccess: (updated) => {
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      setActionError("");
      toast.success(
        updated.warm_offload_enabled
          ? `Warm-cache auto-offload enabled on ${updated.hostname}.`
          : `Warm-cache auto-offload disabled on ${updated.hostname}.`
      );
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const killSleeperMutation = useMutation({
    mutationFn: (pid: number) => killNodeRamSleeper(nodeId as number, pid),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const deleteWarmCacheMutation = useMutation({
    mutationFn: (name: string) => deleteNodeWarmCache(nodeId as number, name),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const deleteImageMutation = useMutation({
    mutationFn: (imageId: string) => deleteNodeImage(nodeId as number, imageId),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const pruneMutation = useMutation({
    mutationFn: () => pruneNodeImages(nodeId as number),
    onSuccess: (result) => {
      setActionError("");
      setPruneResult(result);
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const deleteModelMutation = useMutation({
    mutationFn: (name: string) => deleteNodeModelCache(nodeId as number, name),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const runtimeMutation = useMutation({
    mutationFn: (runtime: string | null) => setNodeRuntime(nodeId as number, runtime),
    onSuccess: (updated) => {
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      setActionError("");
      toast.success(
        updated.container_runtime
          ? `New deployments on ${updated.hostname} will use ${updated.container_runtime}.`
          : `${updated.hostname} follows the preferred runtime again.`
      );
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const removeNodeMutation = useMutation({
    mutationFn: () => deleteNode(nodeId as number),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      toast.success(
        `Removed ${result.hostname}. An active node re-registers automatically.`
      );
      handleClose();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const deleteLocalModelMutation = useMutation({
    mutationFn: (name: string) => deleteLocalModel(nodeId as number, name),
    onSuccess: () => {
      setActionError("");
      refresh();
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const pullMutation = useMutation({
    mutationFn: () =>
      pullLocalModel(nodeId as number, {
        url: pullUrl.trim(),
        ...(pullName.trim() ? { name: pullName.trim() } : {})
      }),
    onSuccess: () => {
      setActionError("");
      setShowPullForm(false);
      setPullUrl("");
      setPullName("");
      queryClient.invalidateQueries({ queryKey: ["node-local-model-transfers", nodeId] });
    },
    onError: (error) => setActionError(errorMessage(error))
  });

  const handleUploadPicked = (kind: "folder" | "archive", fileList: FileList | null) => {
    const files = Array.from(fileList ?? []);
    if (files.length === 0) {
      return;
    }
    const raw =
      kind === "archive"
        ? files[0].name
        : ((files[0] as File & { webkitRelativePath?: string }).webkitRelativePath ||
            files[0].name
          ).split("/")[0];
    setUploadNotice("");
    setPendingUpload({ kind, files, name: suggestModelName(raw) });
  };

  const startUpload = async () => {
    if (!pendingUpload || nodeId === null) {
      return;
    }
    const { kind, files, name } = pendingUpload;
    setPendingUpload(null);
    setActionError("");
    setUploadNotice("");
    setUploadProgress({ name, fraction: 0 });
    try {
      const onProgress = (fraction: number) =>
        setUploadProgress({ name, fraction: Math.min(1, fraction) });
      const result: LocalModelUploadResult =
        kind === "folder"
          ? await uploadLocalModelFolder(nodeId, name, files, onProgress)
          : await uploadLocalModelArchive(nodeId, name, files[0], onProgress);
      setUploadNotice(
        `Uploaded ${result.name} (${(result.size_mb / 1024).toFixed(1)} GB).` +
          (result.warnings.length > 0 ? ` Warning: ${result.warnings.join(" ")}` : "")
      );
      refresh();
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setUploadProgress(null);
    }
  };

  const busy =
    stopMutation.isPending ||
    deleteImageMutation.isPending ||
    pruneMutation.isPending ||
    deleteModelMutation.isPending ||
    deleteLocalModelMutation.isPending ||
    pullMutation.isPending ||
    removeNodeMutation.isPending ||
    runtimeMutation.isPending ||
    warmCacheMutation.isPending ||
    uploadProgress !== null;

  const containers = containersQuery.data ?? [];
  const gpuProcesses = gpuProcessesQuery.data ?? [];
  const ramSleepers = warmArtifactsQuery.data?.ram_sleepers ?? [];
  const diskCaches = warmArtifactsQuery.data?.disk_caches ?? [];
  const images = imagesQuery.data ?? [];
  const cachedModels = modelCacheQuery.data ?? [];
  const localModels = localModelsQuery.data ?? [];
  const transfers = (transfersQuery.data ?? []).filter((t) => t.status !== "done");
  const pendingNameValid =
    pendingUpload !== null && MODEL_NAME_RE.test(pendingUpload.name);
  const totalImageMb = images.reduce((sum, image) => sum + (image.size_mb || 0), 0);
  const totalModelMb = cachedModels.reduce((sum, model) => sum + (model.size_mb || 0), 0);
  const disk = node?.disk_usage ?? null;
  const diskUsedFraction =
    disk && disk.total_gb && disk.free_gb !== undefined
      ? Math.min(1, Math.max(0, (disk.total_gb - disk.free_gb) / disk.total_gb))
      : null;

  const handleClose = () => {
    setActionError("");
    setPruneResult(null);
    setPendingUpload(null);
    setUploadNotice("");
    setShowPullForm(false);
    setConfirmRemoveNode(false);
    onClose();
  };

  return (
    <>
      <AppDialog
        open={open}
        onClose={handleClose}
        title={`Manage${node ? ` — ${node.hostname}` : ""}`}
        actions={
          <>
            <AppButton
              type="button"
              variant="stop"
              ghost
              disabled={busy}
              onClick={() => setConfirmRemoveNode(true)}
            >
              Remove Node
            </AppButton>
            <AppButton type="button" onClick={handleClose}>
              Close
            </AppButton>
          </>
        }
      >
        {disk && diskUsedFraction !== null && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" className="muted" sx={{ mb: 0.5 }}>
              Disk: {disk.free_gb?.toFixed(0)} GB free of {disk.total_gb?.toFixed(0)} GB
              {typeof disk.hf_cache_gb === "number"
                ? ` — model cache uses ${disk.hf_cache_gb.toFixed(1)} GB`
                : ""}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={diskUsedFraction * 100}
              color={diskUsedFraction > 0.9 ? "error" : diskUsedFraction > 0.75 ? "warning" : "primary"}
              sx={{ borderRadius: 1, height: 6 }}
            />
          </Box>
        )}
        {actionError && (
          <Typography variant="body2" color="error" sx={{ mb: 2 }}>
            {actionError}
          </Typography>
        )}

        <DialogSection
          first
          title="Container Runtime"
          hint={
            node?.available_runtimes && node.available_runtimes.length > 0
              ? `Detected: ${node.available_runtimes.join(", ")}. Applies to new deployments; running containers keep the runtime they started with.`
              : "None detected — install Docker or enable the Podman socket on this node."
          }
          action={
            <TextField
              size="small"
              select
              label="Runtime"
              // Sentinel value: MUI renders nothing for "", which left the
              // select looking permanently unselected.
              value={node?.container_runtime ?? "auto"}
              disabled={busy || !(node?.available_runtimes ?? []).length}
              onChange={(event) =>
                runtimeMutation.mutate(
                  event.target.value === "auto" ? null : event.target.value
                )
              }
              sx={{ width: 170 }}
            >
              <MenuItem value="auto">Auto (preferred)</MenuItem>
              {(node?.available_runtimes ?? []).map((runtime) => (
                <MenuItem key={runtime} value={runtime}>
                  {runtime}
                </MenuItem>
              ))}
            </TextField>
          }
        />

        <DialogSection
          title="Warm Cache"
          hint={
            "When enabled, new deployments launch pausable: the agent auto-offloads the least-recently-used unpinned model (GPU → RAM) to fit new or woken models, and a request to a paused model wakes it. Running deployments keep their mode until redeployed."
          }
          action={
            <FormControlLabel
              control={
                <Switch
                  checked={Boolean(node?.warm_offload_enabled)}
                  disabled={busy}
                  onChange={(event) =>
                    warmCacheMutation.mutate({
                      enabled: event.target.checked,
                      ramCacheLimitMb: node?.ram_cache_limit_mb ?? null
                    })
                  }
                />
              }
              label="Auto-offload"
            />
          }
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <TextField
              size="small"
              type="number"
              label="RAM cache limit (GB)"
              placeholder="unlimited"
              value={ramLimitGb}
              disabled={busy || !node?.warm_offload_enabled}
              onChange={(event) => setRamLimitGb(event.target.value)}
              sx={{ width: 200 }}
            />
            <AppButton
              type="button"
              className="app-button--small"
              ghost
              disabled={busy || !node?.warm_offload_enabled}
              onClick={() => {
                const gb = ramLimitGb.trim() === "" ? null : Number(ramLimitGb);
                warmCacheMutation.mutate({
                  enabled: Boolean(node?.warm_offload_enabled),
                  ramCacheLimitMb:
                    gb && Number.isFinite(gb) && gb > 0 ? Math.round(gb * 1024) : null
                });
              }}
            >
              Apply
            </AppButton>
            <Typography variant="body2" className="muted">
              Maximum host RAM for paused model weights. Blank = unlimited.
            </Typography>
          </Box>
        </DialogSection>

        <DialogSection
          title="vLLM Containers"
          hint={'"Active" containers back a tracked deployment (stop them from the Deployments table). "Rogue" containers are untracked leftovers safe to stop & remove here.'}
        >
        {containersQuery.isLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
            <CircularProgress size={20} />
          </Box>
        ) : containersQuery.isError ? (
          <Typography variant="body2" color="error">
            {errorMessage(containersQuery.error)}
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Image</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Tracking</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {containers.map((container) => (
                <TableRow key={container.id} hover>
                  <TableCell>{container.name}</TableCell>
                  <TableCell>
                    <Mono>{container.image}</Mono>
                    {container.runtime && (
                      <Typography variant="caption" className="muted" sx={{ display: "block" }}>
                        {container.runtime}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>{container.status}</TableCell>
                  <TableCell>
                    <Chip
                      label={container.tracked ? "Active" : "Rogue"}
                      size="small"
                      color={container.tracked ? "success" : "warning"}
                    />
                  </TableCell>
                  <TableCell align="right">
                    {container.tracked ? (
                      <Typography variant="body2" className="muted">
                        managed
                      </Typography>
                    ) : (
                      <AppButton
                        type="button"
                        variant="stop"
                        className="app-button--small"
                        ghost
                        disabled={busy}
                        onClick={() => setConfirmStopId(container.id)}
                      >
                        Stop &amp; Remove
                      </AppButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {containers.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} sx={{ borderBottom: "none" }}>
                    <Typography variant="body2" className="muted" sx={{ py: 1 }}>
                      No vLLM containers on this node.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
        </DialogSection>

        <DialogSection
          title="Orphaned GPU Processes"
          hint={'vLLM GPU workers (e.g. "VLLM::EngineCore") that outlived their container and still pin VRAM, with no container left to stop. Killing one frees its GPU memory; a root-owned orphan (from rootful Docker) is killed via a one-shot root container, and if that still fails the error shows a manual cleanup command.'}
        >
        {gpuProcessesQuery.isLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
            <CircularProgress size={20} />
          </Box>
        ) : gpuProcessesQuery.isError ? (
          <Typography variant="body2" color="error">
            {errorMessage(gpuProcessesQuery.error)}
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>PID</TableCell>
                <TableCell>Process</TableCell>
                <TableCell>GPU memory</TableCell>
                <TableCell>GPU</TableCell>
                <TableCell>Tracking</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {gpuProcesses.map((proc: GpuProcess) => (
                <TableRow key={proc.pid} hover>
                  <TableCell>
                    <Mono>{proc.pid}</Mono>
                  </TableCell>
                  <TableCell>
                    <Mono>{proc.process_name || "—"}</Mono>
                  </TableCell>
                  <TableCell>
                    {proc.gpu_memory_mb != null
                      ? `${(proc.gpu_memory_mb / 1024).toFixed(1)} GB`
                      : "—"}
                  </TableCell>
                  <TableCell>{proc.gpu_index != null ? proc.gpu_index : "—"}</TableCell>
                  <TableCell>
                    <Chip
                      label={proc.tracked ? "Active" : "Rogue"}
                      size="small"
                      color={proc.tracked ? "success" : "warning"}
                    />
                  </TableCell>
                  <TableCell align="right">
                    {proc.tracked ? (
                      <Typography variant="body2" className="muted">
                        managed
                      </Typography>
                    ) : (
                      <AppButton
                        type="button"
                        variant="stop"
                        className="app-button--small"
                        ghost
                        disabled={busy}
                        onClick={() => setConfirmKillPid(proc.pid)}
                      >
                        Kill
                      </AppButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {gpuProcesses.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} sx={{ borderBottom: "none" }}>
                    <Typography variant="body2" className="muted" sx={{ py: 1 }}>
                      No vLLM GPU processes on this node.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
        </DialogSection>

        {node?.warm_offload_enabled && (
          <DialogSection
            title="Orphaned Warm-Cache Artifacts"
            hint={
              "RAM sleepers are paused-to-RAM vLLM processes holding CPU memory with no tracked deployment; disk caches are leftover torch.compile directories. Both are safe to reclaim here."
            }
          >
            {warmArtifactsQuery.isLoading ? (
              <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
                <CircularProgress size={20} />
              </Box>
            ) : warmArtifactsQuery.isError ? (
              <Typography variant="body2" color="error">
                {errorMessage(warmArtifactsQuery.error)}
              </Typography>
            ) : ramSleepers.length === 0 && diskCaches.length === 0 ? (
              <Typography variant="body2" className="muted" sx={{ py: 1 }}>
                No orphaned warm-cache artifacts on this node.
              </Typography>
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Artifact</TableCell>
                    <TableCell>Detail</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell align="right">Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {ramSleepers.map((sleeper) => (
                    <TableRow key={`pid-${sleeper.pid}`} hover>
                      <TableCell>
                        <Chip label="RAM sleeper" size="small" color="warning" />
                      </TableCell>
                      <TableCell>
                        <Mono>
                          {sleeper.process_name || "vLLM"} (pid {sleeper.pid})
                        </Mono>
                      </TableCell>
                      <TableCell>{(sleeper.rss_mb / 1024).toFixed(1)} GB RAM</TableCell>
                      <TableCell align="right">
                        <AppButton
                          type="button"
                          variant="stop"
                          className="app-button--small"
                          ghost
                          disabled={busy}
                          onClick={() => setConfirmSleeperPid(sleeper.pid)}
                        >
                          Kill
                        </AppButton>
                      </TableCell>
                    </TableRow>
                  ))}
                  {diskCaches.map((cache) => (
                    <TableRow key={`cache-${cache.name}`} hover>
                      <TableCell>
                        <Chip label="Disk cache" size="small" color="warning" />
                      </TableCell>
                      <TableCell>
                        <Mono>{cache.path}</Mono>
                      </TableCell>
                      <TableCell>{(cache.size_mb / 1024).toFixed(1)} GB</TableCell>
                      <TableCell align="right">
                        <AppButton
                          type="button"
                          variant="stop"
                          className="app-button--small"
                          ghost
                          disabled={busy}
                          onClick={() => setConfirmCacheName(cache.name)}
                        >
                          Delete
                        </AppButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </DialogSection>
        )}

        <DialogSection
          title="vLLM Image Cache"
          hint={`Cached images total ${(totalImageMb / 1024).toFixed(1)} GB. "Prune unused" removes images (and dangling derived layers) not backing any container.`}
          action={
            <AppButton
              type="button"
              className="app-button--small"
              disabled={busy || images.length === 0}
              onClick={() => setConfirmPrune(true)}
            >
              {pruneMutation.isPending ? "Pruning..." : "Prune Unused"}
            </AppButton>
          }
        >
        {pruneResult && (
          <Typography variant="body2" className="muted" sx={{ mb: 1 }}>
            Pruned {pruneResult.removed.length} image(s), freed{" "}
            {(pruneResult.freed_mb / 1024).toFixed(1)} GB
            {pruneResult.skipped.length > 0
              ? `; ${pruneResult.skipped.length} in use / skipped`
              : ""}
            .
          </Typography>
        )}
        {imagesQuery.isLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
            <CircularProgress size={20} />
          </Box>
        ) : imagesQuery.isError ? (
          <Typography variant="body2" color="error">
            {errorMessage(imagesQuery.error)}
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Tags</TableCell>
                <TableCell align="right">Size</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {images.map((image) => (
                <TableRow key={image.id} hover>
                  <TableCell>
                    <Mono>{image.tags.length > 0 ? image.tags.join(", ") : image.id}</Mono>
                    {(image.runtimes?.length || image.runtime) && (
                      <Typography variant="caption" className="muted" sx={{ display: "block" }}>
                        {image.runtimes?.join(" \u00b7 ") ?? image.runtime}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell align="right">{(image.size_mb / 1024).toFixed(1)} GB</TableCell>
                  <TableCell align="right">
                    <AppButton
                      type="button"
                      variant="stop"
                      className="app-button--small"
                      ghost
                      disabled={busy}
                      onClick={() => setConfirmImageId(image.id)}
                    >
                      Delete
                    </AppButton>
                  </TableCell>
                </TableRow>
              ))}
              {images.length === 0 && (
                <TableRow>
                  <TableCell colSpan={3} sx={{ borderBottom: "none" }}>
                    <Typography variant="body2" className="muted" sx={{ py: 1 }}>
                      No cached vLLM images on this node.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
        </DialogSection>

        <DialogSection
          title="Local Models"
          hint="Checkpoints uploaded here live in ~/.vllm-client/.models on the node and are deployable without MODEL_DIRS configuration. Keep this dialog open while an upload runs."
          action={
            <Box sx={{ display: "flex", gap: 0.5 }}>
            <AppButton
              type="button"
              className="app-button--small"
              disabled={busy}
              onClick={() => folderInputRef.current?.click()}
            >
              Upload Folder
            </AppButton>
            <AppButton
              type="button"
              className="app-button--small"
              disabled={busy}
              onClick={() => archiveInputRef.current?.click()}
            >
              Upload Archive
            </AppButton>
            <AppButton
              type="button"
              className="app-button--small"
              disabled={busy}
              onClick={() => setShowPullForm((value) => !value)}
            >
              Pull from URL
            </AppButton>
            </Box>
          }
        >
        {/* Hidden pickers; webkitdirectory is non-standard but universal. */}
        <input
          ref={folderInputRef}
          type="file"
          multiple
          style={{ display: "none" }}
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          onChange={(event) => {
            handleUploadPicked("folder", event.target.files);
            event.target.value = "";
          }}
        />
        <input
          ref={archiveInputRef}
          type="file"
          accept=".tar.gz,.tgz,.zip"
          style={{ display: "none" }}
          onChange={(event) => {
            handleUploadPicked("archive", event.target.files);
            event.target.value = "";
          }}
        />
        {pendingUpload && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <TextField
              size="small"
              label="Model name on the node"
              value={pendingUpload.name}
              error={!pendingNameValid}
              helperText={
                pendingNameValid
                  ? `${pendingUpload.files.length} file(s), ${(
                      pendingUpload.files.reduce((sum, f) => sum + f.size, 0) /
                      1024 ** 3
                    ).toFixed(2)} GB`
                  : "Letters, digits, '.', '_' or '-' only."
              }
              onChange={(event) =>
                setPendingUpload({ ...pendingUpload, name: event.target.value })
              }
              sx={{ minWidth: 260 }}
            />
            <AppButton
              type="button"
              className="app-button--small"
              disabled={!pendingNameValid || busy}
              onClick={() => void startUpload()}
            >
              Start Upload
            </AppButton>
            <AppButton
              type="button"
              className="app-button--small"
              ghost
              onClick={() => setPendingUpload(null)}
            >
              Cancel
            </AppButton>
          </Box>
        )}
        {showPullForm && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1, flexWrap: "wrap" }}>
            <TextField
              size="small"
              label="Archive or file URL (http/https)"
              value={pullUrl}
              onChange={(event) => setPullUrl(event.target.value)}
              sx={{ flex: 1, minWidth: 280 }}
            />
            <TextField
              size="small"
              label="Name (optional)"
              value={pullName}
              onChange={(event) => setPullName(event.target.value)}
              sx={{ width: 180 }}
            />
            <AppButton
              type="button"
              className="app-button--small"
              disabled={busy || !pullUrl.trim()}
              onClick={() => pullMutation.mutate()}
            >
              Start Pull
            </AppButton>
          </Box>
        )}
        {uploadProgress && (
          <Box sx={{ mb: 1 }}>
            <Typography variant="body2" className="muted" sx={{ mb: 0.5 }}>
              Uploading {uploadProgress.name} — {(uploadProgress.fraction * 100).toFixed(0)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={uploadProgress.fraction * 100}
              sx={{ borderRadius: 1, height: 6 }}
            />
          </Box>
        )}
        {transfers.map((transfer) => (
          <Box key={transfer.id} sx={{ mb: 1 }}>
            <Typography
              variant="body2"
              className="muted"
              color={transfer.status === "error" ? "error" : undefined}
              sx={{ mb: 0.5 }}
            >
              {transfer.status === "error"
                ? `Pull of ${transfer.name} failed: ${transfer.error}`
                : transfer.status === "extracting"
                  ? `Extracting ${transfer.name}…`
                  : `Pulling ${transfer.name} — ${(transfer.received_bytes / 1024 ** 3).toFixed(2)} GB` +
                    (transfer.total_bytes
                      ? ` of ${(transfer.total_bytes / 1024 ** 3).toFixed(2)} GB`
                      : "")}
            </Typography>
            {transfer.status !== "error" && (
              <LinearProgress
                variant={
                  transfer.total_bytes && transfer.status === "downloading"
                    ? "determinate"
                    : "indeterminate"
                }
                value={
                  transfer.total_bytes
                    ? (transfer.received_bytes / transfer.total_bytes) * 100
                    : undefined
                }
                sx={{ borderRadius: 1, height: 6 }}
              />
            )}
          </Box>
        ))}
        {uploadNotice && (
          <Typography variant="body2" className="muted" sx={{ mb: 1 }}>
            {uploadNotice}
          </Typography>
        )}
        {localModelsQuery.isLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
            <CircularProgress size={20} />
          </Box>
        ) : localModelsQuery.isError ? (
          <Typography variant="body2" color="error">
            {errorMessage(localModelsQuery.error)}
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell>Source</TableCell>
                <TableCell>Status</TableCell>
                <TableCell align="right">Size</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {localModels.map((model) => (
                <TableRow
                  key={model.path}
                  hover
                  sx={{ "& .copy-icon": { opacity: 0, transition: "opacity 0.15s" }, "&:hover .copy-icon": { opacity: 1 } }}
                >
                  <TableCell>
                    <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.5 }}>
                      <Box>
                        {model.name}
                        <Mono block>{model.path}</Mono>
                      </Box>
                      <Tooltip title="Copy model name" enterDelay={500}>
                        <IconButton
                          className="copy-icon"
                          size="small"
                          aria-label={`Copy ${model.name}`}
                          onClick={() => void copyName(model.name)}
                          sx={{ alignSelf: "center" }}
                        >
                          <ContentCopyIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={model.source === "managed" ? "managed" : "MODEL_DIRS"}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {model.in_use ? (
                      <Chip label="In use" size="small" color="success" />
                    ) : (
                      <Chip label="Idle" size="small" />
                    )}
                  </TableCell>
                  <TableCell align="right">{(model.size_mb / 1024).toFixed(1)} GB</TableCell>
                  <TableCell align="right">
                    {model.deletable && !model.in_use ? (
                      <AppButton
                        type="button"
                        variant="stop"
                        className="app-button--small"
                        ghost
                        disabled={busy}
                        onClick={() => setConfirmLocalModel(model.name)}
                      >
                        Delete
                      </AppButton>
                    ) : (
                      <Typography variant="body2" className="muted">
                        {model.in_use ? "serving" : "external"}
                      </Typography>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {localModels.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} sx={{ borderBottom: "none" }}>
                    <Typography variant="body2" className="muted" sx={{ py: 1 }}>
                      No local models on this node yet.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
        </DialogSection>

        <DialogSection
          title="Model Cache"
          hint={`Downloaded model weights total ${(totalModelMb / 1024).toFixed(1)} GB. They are shared across deployments; deleting one forces a re-download on next use.`}
        >
        {modelCacheQuery.isLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
            <CircularProgress size={20} />
          </Box>
        ) : modelCacheQuery.isError ? (
          <Typography variant="body2" color="error">
            {errorMessage(modelCacheQuery.error)}
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell>Status</TableCell>
                <TableCell align="right">Size</TableCell>
                <TableCell align="right">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {cachedModels.map((model) => (
                <TableRow
                  key={model.name}
                  hover
                  sx={{ "& .copy-icon": { opacity: 0, transition: "opacity 0.15s" }, "&:hover .copy-icon": { opacity: 1 } }}
                >
                  <TableCell>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Mono>{model.name}</Mono>
                      <Tooltip title="Copy model name" enterDelay={500}>
                        <IconButton
                          className="copy-icon"
                          size="small"
                          aria-label={`Copy ${model.name}`}
                          onClick={() => void copyName(model.name)}
                        >
                          <ContentCopyIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                  <TableCell>
                    {model.in_use ? (
                      <Chip label="In use" size="small" color="success" />
                    ) : (
                      <Chip label="Idle" size="small" />
                    )}
                  </TableCell>
                  <TableCell align="right">{(model.size_mb / 1024).toFixed(1)} GB</TableCell>
                  <TableCell align="right">
                    {model.in_use ? (
                      <Typography variant="body2" className="muted">
                        serving
                      </Typography>
                    ) : (
                      <AppButton
                        type="button"
                        variant="stop"
                        className="app-button--small"
                        ghost
                        disabled={busy}
                        onClick={() => setConfirmModelName(model.name)}
                      >
                        Delete
                      </AppButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {cachedModels.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} sx={{ borderBottom: "none" }}>
                    <Typography variant="body2" className="muted" sx={{ py: 1 }}>
                      No cached models on this node.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
        </DialogSection>
      </AppDialog>

      <ConfirmDialog
        open={confirmStopId !== null}
        title="Stop & remove this container?"
        body="The rogue container will be stopped and removed from the node."
        confirmLabel="Stop & Remove"
        danger
        onConfirm={() => {
          if (confirmStopId) {
            stopMutation.mutate(confirmStopId);
          }
          setConfirmStopId(null);
        }}
        onCancel={() => setConfirmStopId(null)}
      />
      <ConfirmDialog
        open={confirmKillPid !== null}
        title="Kill this GPU process?"
        body="The orphaned vLLM process will be terminated (SIGTERM, then SIGKILL) and its GPU memory freed. This cannot be undone."
        confirmLabel="Kill"
        danger
        onConfirm={() => {
          if (confirmKillPid !== null) {
            killProcessMutation.mutate(confirmKillPid);
          }
          setConfirmKillPid(null);
        }}
        onCancel={() => setConfirmKillPid(null)}
      />
      <ConfirmDialog
        open={confirmSleeperPid !== null}
        title="Kill this RAM sleeper?"
        body="The orphaned paused-to-RAM vLLM process will be terminated and its CPU memory freed. This cannot be undone."
        confirmLabel="Kill"
        danger
        onConfirm={() => {
          if (confirmSleeperPid !== null) {
            killSleeperMutation.mutate(confirmSleeperPid);
          }
          setConfirmSleeperPid(null);
        }}
        onCancel={() => setConfirmSleeperPid(null)}
      />
      <ConfirmDialog
        open={confirmCacheName !== null}
        title="Delete this compile cache?"
        body="Removes the orphaned torch.compile cache directory. A future deployment with the same config will recompile from scratch."
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (confirmCacheName) {
            deleteWarmCacheMutation.mutate(confirmCacheName);
          }
          setConfirmCacheName(null);
        }}
        onCancel={() => setConfirmCacheName(null)}
      />
      <ConfirmDialog
        open={confirmImageId !== null}
        title="Delete this image?"
        body="Removes it from every container runtime on this node. The next deployment using it will pull it again from the registry."
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (confirmImageId) {
            deleteImageMutation.mutate(confirmImageId);
          }
          setConfirmImageId(null);
        }}
        onCancel={() => setConfirmImageId(null)}
      />
      <ConfirmDialog
        open={confirmPrune}
        title="Prune unused images?"
        body="Removes all cached vLLM images (and dangling derived layers) not backing any container."
        confirmLabel="Prune"
        danger
        onConfirm={() => {
          pruneMutation.mutate();
          setConfirmPrune(false);
        }}
        onCancel={() => setConfirmPrune(false)}
      />
      <ConfirmDialog
        open={confirmRemoveNode}
        title={`Remove node ${node?.hostname ?? ""}?`}
        body="Removes this node and its deployment records from the dashboard. Running containers on the node are not stopped — an active node re-registers within seconds and its deployments are re-adopted; a stale node disappears for good."
        confirmLabel="Remove"
        danger
        onConfirm={() => {
          setConfirmRemoveNode(false);
          removeNodeMutation.mutate();
        }}
        onCancel={() => setConfirmRemoveNode(false)}
      />
      <ConfirmDialog
        open={confirmLocalModel !== null}
        title={`Delete local model ${confirmLocalModel ?? ""}?`}
        body="The checkpoint files are removed from the node. Deployments configured with this path will fail to redeploy."
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (confirmLocalModel) {
            deleteLocalModelMutation.mutate(confirmLocalModel);
          }
          setConfirmLocalModel(null);
        }}
        onCancel={() => setConfirmLocalModel(null)}
      />
      <ConfirmDialog
        open={confirmModelName !== null}
        title={`Delete cached weights for ${confirmModelName ?? ""}?`}
        body="The next deployment of this model will re-download the weights from Hugging Face."
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (confirmModelName) {
            deleteModelMutation.mutate(confirmModelName);
          }
          setConfirmModelName(null);
        }}
        onCancel={() => setConfirmModelName(null)}
      />
    </>
  );
}

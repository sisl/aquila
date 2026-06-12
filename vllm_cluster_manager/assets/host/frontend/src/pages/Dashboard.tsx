import { useEffect, useMemo, useRef, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Checkbox,
  Divider,
  FormControlLabel,
  IconButton,
  LinearProgress,
  MenuItem,
  Paper,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import ClearIcon from "@mui/icons-material/Clear";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  fetchDeployments,
  fetchConfigs,
  fetchLocalModels,
  fetchNodes,
  fetchLatestVllmVersion,
  type Deployment,
  type DeploymentConfig,
  type DeploymentExtension,
  type Node,
  deleteDeployment,
  deploymentLogsDownloadUrl,
  extendDeployment,
  fetchDeploymentLogs,
  checkNodePort,
  createConfig,
  deleteConfig,
  fetchManifest,
  setNodeMaintenance,
  startDeployment,
  stopDeployment,
  purgeDatabase,
  restartDeployment,
  uploadPackage
} from "../services/api";
import { connectWebSocket } from "../services/ws";
import { StatusCard } from "../components/StatusCard";
import { SectionLabel } from "../components/SectionLabel";
import { NodeTable } from "../components/NodeTable";
import { NodeDockerDialog } from "../components/NodeDockerDialog";
import { DeploymentTable } from "../components/DeploymentTable";
import { EndpointDialog } from "../components/EndpointDialog";
import { AppButton } from "../components/AppButton";
import { AppDialog } from "../components/AppDialog";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EmptyState } from "../components/EmptyState";
import { useToast } from "../components/ToastProvider";

const DTYPE_OPTIONS = ["", "auto", "bfloat16", "float16", "float32"];
const QUANTIZATION_OPTIONS = ["", "awq", "gptq", "fp8", "bitsandbytes"];

const DURATION_OPTIONS: { value: string; label: string }[] = [
  { value: "3600", label: "1 hour" },
  { value: "7200", label: "2 hours" },
  { value: "14400", label: "4 hours" },
  { value: "28800", label: "8 hours" },
  { value: "43200", label: "12 hours" },
  { value: "86400", label: "24 hours" },
  { value: "172800", label: "48 hours" },
  { value: "inf", label: "Infinite" },
  { value: "custom", label: "Custom…" }
];

const DEFAULT_DURATION_CHOICE = "43200"; // 12 hours

// null => infinite; null is also returned for an invalid custom value (guarded
// against by isDurationValid before deploy/restart is enabled).
function durationChoiceToSeconds(choice: string, customHours: string): number | null {
  if (choice === "inf") return null;
  if (choice === "custom") {
    const hours = parseFloat(customHours);
    return Number.isFinite(hours) && hours > 0 ? Math.round(hours * 3600) : null;
  }
  return Number(choice);
}

function isDurationValid(choice: string, customHours: string): boolean {
  if (choice !== "custom") return true;
  const hours = parseFloat(customHours);
  return Number.isFinite(hours) && hours > 0;
}

// Map a stored duration (seconds | null) back onto the dropdown + custom field.
function secondsToChoice(seconds: number | null): { choice: string; customHours: string } {
  if (seconds === null) return { choice: "inf", customHours: "6" };
  const preset = DURATION_OPTIONS.find((option) => option.value === String(seconds));
  if (preset) return { choice: preset.value, customHours: "6" };
  return { choice: "custom", customHours: String(seconds / 3600) };
}

export function Dashboard() {
  const [nodeId, setNodeId] = useState("");
  const [modelName, setModelName] = useState("");
  const [owner, setOwner] = useState("");
  const [durationChoice, setDurationChoice] = useState(DEFAULT_DURATION_CHOICE);
  const [customHours, setCustomHours] = useState("6");
  const [port, setPort] = useState(8001);
  const [gpuFraction, setGpuFraction] = useState(0.5);
  const [gpuIds, setGpuIds] = useState<number[]>([]);
  const [advancedArgs, setAdvancedArgs] = useState<
    Array<{ id: number; key: string; value: string }>
  >([]);
  const [rawArgs, setRawArgs] = useState("");
  // Which optional form section (accordion) is open; exclusive expansion.
  const [expandedSection, setExpandedSection] = useState<string | false>(false);
  const advancedArgId = useRef(1);
  const [envVars, setEnvVars] = useState<Array<{ id: number; key: string; value: string }>>(
    []
  );
  const envVarId = useRef(1);
  const [loraModules, setLoraModules] = useState<
    Array<{ id: number; name: string; path: string }>
  >([]);
  const loraModuleId = useRef(1);
  const [vllmVersion, setVllmVersion] = useState("");
  // Structured engine options (translated to vLLM CLI flags by the client).
  const [maxModelLen, setMaxModelLen] = useState("");
  const [revision, setRevision] = useState("");
  const [seed, setSeed] = useState("");
  const [dtype, setDtype] = useState("");
  const [quantization, setQuantization] = useState("");
  const [servedModelName, setServedModelName] = useState("");
  const [maxNumSeqs, setMaxNumSeqs] = useState("");
  const [enforceEager, setEnforceEager] = useState(false);
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [maxFailedRestarts, setMaxFailedRestarts] = useState("");
  const [skipResourceCheck, setSkipResourceCheck] = useState(false);
  const [extraPackagesText, setExtraPackagesText] = useState("");
  const [uploadingPackage, setUploadingPackage] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [configName, setConfigName] = useState("");
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [logsDeploymentId, setLogsDeploymentId] = useState<number | null>(null);
  const [settingsDeployment, setSettingsDeployment] = useState<Deployment | null>(null);
  const [dockerNode, setDockerNode] = useState<Node | null>(null);
  const [restartTarget, setRestartTarget] = useState<Deployment | null>(null);
  const [restartOwner, setRestartOwner] = useState("");
  const [restartDurationChoice, setRestartDurationChoice] = useState(DEFAULT_DURATION_CHOICE);
  const [restartCustomHours, setRestartCustomHours] = useState("6");
  const [maintenanceTarget, setMaintenanceTarget] = useState<Node | null>(null);
  const [maintenanceDrain, setMaintenanceDrain] = useState(false);
  const [manifestDeployment, setManifestDeployment] = useState<Deployment | null>(null);
  const [endpointDeployment, setEndpointDeployment] = useState<Deployment | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [confirmPurge, setConfirmPurge] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const queryClient = useQueryClient();
  const toast = useToast();

  // Push channel: the backend broadcasts nodes_changed/deployments_changed
  // after every sync tick and user action, so status updates land in
  // sub-second time. Polling stays on as a slow fallback while connected.
  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let retryDelay = 1000;
    let retryTimer: number | undefined;

    const open = () => {
      socket = connectWebSocket((message) => {
        if (message.type === "deployments_changed") {
          queryClient.invalidateQueries({ queryKey: ["deployments"] });
        } else if (message.type === "nodes_changed") {
          queryClient.invalidateQueries({ queryKey: ["nodes"] });
        }
      });
      socket.onopen = () => {
        retryDelay = 1000;
        setWsConnected(true);
      };
      socket.onclose = () => {
        setWsConnected(false);
        if (!closed) {
          retryTimer = window.setTimeout(open, retryDelay);
          retryDelay = Math.min(retryDelay * 2, 30000);
        }
      };
    };
    open();

    return () => {
      closed = true;
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
      }
      socket?.close();
    };
  }, [queryClient]);

  const pollInterval = wsConnected ? 30000 : 5000;

  const nodesQuery = useQuery({
    queryKey: ["nodes"],
    queryFn: fetchNodes,
    refetchInterval: pollInterval,
    refetchIntervalInBackground: true,
    staleTime: 4000,
    placeholderData: (previous) => previous
  });

  const deploymentsQuery = useQuery({
    queryKey: ["deployments"],
    queryFn: fetchDeployments,
    refetchInterval: pollInterval,
    refetchIntervalInBackground: true,
    staleTime: 4000,
    placeholderData: (previous) => previous
  });

  const configsQuery = useQuery({
    queryKey: ["configs"],
    queryFn: fetchConfigs
  });

  // Local checkpoints on the selected node, offered as a deploy-form picker.
  const localModelsQuery = useQuery({
    queryKey: ["node-local-models", nodeId === "" ? null : Number(nodeId)],
    queryFn: () => fetchLocalModels(Number(nodeId)),
    enabled: nodeId !== "",
    staleTime: 30000
  });

  const latestVllmQuery = useQuery({
    queryKey: ["latest-vllm-version"],
    queryFn: fetchLatestVllmVersion,
    staleTime: 3600000,
    refetchInterval: 3600000
  });

  const portCheckQuery = useQuery({
    queryKey: ["port-check", nodeId, port],
    queryFn: () => checkNodePort(Number(nodeId), port),
    enabled: nodeId !== "" && !Number.isNaN(Number(nodeId)) && port > 0,
    refetchInterval: 5000,
    refetchIntervalInBackground: true,
    staleTime: 4000,
    placeholderData: (previous) => previous
  });

  const startMutation = useMutation({
    mutationFn: startDeployment,
    onSuccess: (deployment) => {
      queryClient.setQueryData(["deployments"], (existing: Deployment[] | undefined) => {
        if (!existing) {
          return [deployment];
        }
        return [deployment, ...existing];
      });
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      setLogsDeploymentId(deployment.id);
      toast.success(`Deployment of ${deployment.model_name} started.`);
      setModelName("");
      setPort(8001);
      setGpuFraction(0.5);
      setGpuIds([]);
      setAdvancedArgs([]);
      setRawArgs("");
      setEnvVars([]);
      setLoraModules([]);
      setVllmVersion("");
      setMaxModelLen("");
      setRevision("");
      setSeed("");
      setDtype("");
      setQuantization("");
      setServedModelName("");
      setMaxNumSeqs("");
      setEnforceEager(false);
      setTrustRemoteCode(false);
      setMaxFailedRestarts("");
      setSkipResourceCheck(false);
      setExtraPackagesText("");
      setExpandedSection(false);
      setConfigName("");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to deploy.");
    }
  });

  const stopMutation = useMutation({
    mutationFn: stopDeployment,
    onSuccess: (deployment) => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      toast.success(`Stopped ${deployment.model_name}.`);
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to stop deployment.");
    }
  });

  const deleteMutation = useMutation({
    mutationFn: deleteDeployment,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      toast.success("Deployment deleted.");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to delete deployment.");
    }
  });

  const extendMutation = useMutation({
    mutationFn: ({ id, extension }: { id: number; extension: DeploymentExtension }) =>
      extendDeployment(id, extension),
    onSuccess: (deployment, variables) => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      toast.success(
        "infinite" in variables.extension
          ? `${deployment.model_name} now serves until stopped.`
          : `Extended ${deployment.model_name}.`
      );
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to extend deployment.");
    }
  });

  const purgeMutation = useMutation({
    mutationFn: purgeDatabase,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      queryClient.invalidateQueries({ queryKey: ["configs"] });
      setConfirmPurge(false);
      setSettingsOpen(false);
      toast.success("Database purged. Nodes and running models re-register shortly.");
    },
    onError: (error) => {
      setConfirmPurge(false);
      toast.error(error instanceof Error ? error.message : "Failed to purge database.");
    }
  });

  const maintenanceMutation = useMutation({
    mutationFn: ({ node, enabled, drain }: { node: Node; enabled: boolean; drain: boolean }) =>
      setNodeMaintenance(node.id, enabled, drain),
    onSuccess: (node) => {
      queryClient.invalidateQueries({ queryKey: ["nodes"] });
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      setMaintenanceTarget(null);
      toast.success(
        node.maintenance
          ? `${node.hostname} is now in maintenance mode.`
          : `${node.hostname} is back in rotation.`
      );
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to update maintenance mode.");
    }
  });

  const openMaintenance = (node: Node) => {
    setMaintenanceTarget(node);
    setMaintenanceDrain(false);
  };

  const restartMutation = useMutation({
    mutationFn: ({
      id,
      owner: restartBy,
      durationSeconds
    }: {
      id: number;
      owner: string;
      durationSeconds: number | null;
    }) => restartDeployment(id, restartBy, durationSeconds),
    onSuccess: (deployment) => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
      setRestartTarget(null);
      setLogsDeploymentId(deployment.id);
      toast.success(`Restarted ${deployment.model_name}.`);
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to restart deployment.");
    }
  });

  const openRestart = (deployment: Deployment) => {
    setRestartTarget(deployment);
    setRestartOwner(deployment.owner ?? "");
    setRestartDurationChoice(DEFAULT_DURATION_CHOICE);
    setRestartCustomHours("6");
  };

  const createConfigMutation = useMutation({
    mutationFn: ({ name, payload }: { name: string; payload: Record<string, unknown> }) =>
      createConfig(name, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["configs"] });
      setConfigName("");
      toast.success("Configuration saved.");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to save configuration.");
    }
  });

  const deleteConfigMutation = useMutation({
    mutationFn: (configId: number) => deleteConfig(configId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["configs"] });
      toast.success("Configuration deleted.");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to delete configuration.");
    }
  });

  const manifestQuery = useQuery({
    queryKey: ["manifest", manifestDeployment?.id],
    queryFn: () => fetchManifest(manifestDeployment?.id ?? 0),
    enabled: manifestDeployment !== null
  });

  const manifestJson = manifestQuery.data
    ? JSON.stringify(manifestQuery.data, null, 2)
    : "";

  const copyManifest = async () => {
    try {
      await navigator.clipboard.writeText(manifestJson);
      toast.success("Manifest copied to clipboard.");
    } catch {
      toast.error("Clipboard unavailable.");
    }
  };

  const downloadManifest = () => {
    if (!manifestDeployment) return;
    const blob = new Blob([manifestJson], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `deployment-${manifestDeployment.id}-manifest.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const loadManifestIntoForm = () => {
    const manifest = manifestQuery.data;
    if (!manifest) return;
    const gpu = (manifest.gpu ?? {}) as Record<string, unknown>;
    // Reshape into the saved-config payload format and reuse its loader.
    loadConfig({
      id: 0,
      name: "manifest",
      payload: {
        model_name: manifest.model,
        gpu_memory_fraction: gpu.memory_fraction,
        gpu_ids: gpu.ids,
        extra_args: manifest.extra_args,
        env_vars: [],
        vllm_version: manifest.vllm_version,
        extra_packages: manifest.extra_packages,
        engine_args: manifest.engine_args,
        lora_modules: manifest.lora_modules,
        duration_seconds: manifest.duration_seconds
      }
    });
    const envKeys = Array.isArray(manifest.env_var_keys) ? manifest.env_var_keys : [];
    if (envKeys.length > 0) {
      toast.info(
        `Re-enter values for environment variable(s): ${envKeys.join(", ")} (not stored in manifests).`
      );
      setEnvVars(
        envKeys.map((key, index) => ({
          id: envVarId.current + index,
          key: String(key),
          value: ""
        }))
      );
      envVarId.current += envKeys.length;
    }
    setManifestDeployment(null);
    setSettingsDeployment(null);
    toast.success("Manifest loaded into the launch form — pick a node and port.");
  };

  const logsQuery = useQuery({
    queryKey: ["deployment-logs", logsDeploymentId],
    queryFn: () => fetchDeploymentLogs(logsDeploymentId ?? 0, 400),
    enabled: logsDeploymentId !== null,
    refetchInterval: logsDeploymentId !== null ? 2000 : false,
    refetchIntervalInBackground: true,
    staleTime: 1500,
    placeholderData: (previous) => previous
  });

  // Logs auto-scroll: pinned to the bottom while the user is at (or near)
  // the bottom; scrolling up releases the pin, scrolling back re-arms it.
  const logsBodyRef = useRef<HTMLDivElement | null>(null);
  const logsStickRef = useRef(true);
  const handleLogsScroll = () => {
    const el = logsBodyRef.current;
    if (el) {
      logsStickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    }
  };
  useEffect(() => {
    if (logsDeploymentId !== null) {
      logsStickRef.current = true;
    }
  }, [logsDeploymentId]);
  useEffect(() => {
    const el = logsBodyRef.current;
    if (el && logsStickRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logsQuery.data]);

  const durationSeconds = durationChoiceToSeconds(durationChoice, customHours);
  const canDeploy =
    nodeId !== "" &&
    modelName.trim().length > 0 &&
    owner.trim().length > 0 &&
    isDurationValid(durationChoice, customHours) &&
    !Number.isNaN(Number(nodeId));

  const visibleDeployments = useMemo(() => {
    return (deploymentsQuery.data ?? []).filter(
      (deployment) => deployment.status !== "unreachable"
    );
  }, [deploymentsQuery.data]);

  const statusSummary = useMemo(() => {
    const nodes = nodesQuery.data ?? [];
    const running = visibleDeployments.filter(
      (deployment) => deployment.status === "running"
    );

    return {
      totalNodes: nodes.length,
      healthyNodes: nodes.filter((node) => node.status === "healthy").length,
      activeModels: running.length
    };
  }, [nodesQuery.data, visibleDeployments]);

  const nodeNameById = useMemo(() => {
    const mapping: Record<number, string> = {};
    for (const node of nodesQuery.data ?? []) {
      mapping[node.id] = node.hostname;
    }
    return mapping;
  }, [nodesQuery.data]);

  const selectedNode = useMemo(() => {
    const parsedId = Number(nodeId);
    if (Number.isNaN(parsedId)) {
      return null;
    }
    return (nodesQuery.data ?? []).find((node) => node.id === parsedId) ?? null;
  }, [nodeId, nodesQuery.data]);

  const availableGpuIds = useMemo(() => {
    return selectedNode?.gpu_usage?.map((gpu) => gpu.index) ?? [];
  }, [selectedNode]);

  const parsedRawArgs = useMemo(() => {
    const tokens: string[] = [];
    const regex = /"([^"]*)"|'([^']*)'|(\S+)/g;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(rawArgs)) !== null) {
      const value = match[1] ?? match[2] ?? match[3];
      if (value) {
        tokens.push(value);
      }
    }
    return tokens;
  }, [rawArgs]);

  const extraArgs = useMemo(() => {
    const args: string[] = [];
    for (const entry of advancedArgs) {
      const rawKey = entry.key.trim();
      if (!rawKey) {
        continue;
      }
      const key = rawKey.startsWith("--") ? rawKey : `--${rawKey}`;
      args.push(key);
      const value = entry.value.trim();
      if (value) {
        args.push(value);
      }
    }
    for (const token of parsedRawArgs) {
      if (token.trim().length > 0) {
        args.push(token.trim());
      }
    }
    return args;
  }, [advancedArgs, parsedRawArgs]);

  const shellQuote = (value: string) => {
    if (value === "") {
      return "''";
    }
    if (
      /^[A-Za-z0-9_./:=@+-]+$/.test(value) ||
      (/^'.*'$/.test(value) || /^".*"$/.test(value))
    ) {
      return value;
    }
    return `'${value.replace(/'/g, `'\"'\"'`)}'`;
  };

  const resolveImageTag = (version?: string | null) => {
    const repo = "vllm/vllm-openai";
    const raw = (version ?? "").trim();
    if (!raw) return `${repo}:latest`;
    if (raw.toLowerCase() === "nightly") return `${repo}:nightly`;
    if (/^[0-9a-f]{40}$/.test(raw)) return `${repo}:nightly-${raw}`;
    if (/^\d+\.\d+(\.\d+)?/.test(raw)) return `${repo}:v${raw}`;
    return `${repo}:${raw}`;
  };

  const buildVllmCommand = (deployment: Deployment) => {
    const lines: string[] = [];
    if (deployment.extra_packages && deployment.extra_packages.length > 0) {
      lines.push(`# extra_packages: ${deployment.extra_packages.join(", ")}`);
      lines.push(`# (a derived image is built FROM the base tag and cached)`);
    }

    const image = resolveImageTag(deployment.vllm_version);
    const gpus =
      deployment.gpu_ids && deployment.gpu_ids.length > 0
        ? `'"device=${deployment.gpu_ids.join(",")}"'`
        : "all";

    const dockerParts = [
      "docker", "run", "-d",
      "--gpus", gpus,
      "--ipc=host",
      "--network", "host",
      "-v", "~/.cache/huggingface:/root/.cache/huggingface"
    ];
    if (deployment.env_vars && deployment.env_vars.length > 0) {
      for (const pair of deployment.env_vars) {
        if (!pair.key) {
          continue;
        }
        dockerParts.push("-e", `${pair.key}=${shellQuote(String(pair.value ?? ""))}`);
      }
    }
    dockerParts.push(image);

    const vllmArgs = [
      "--model",
      deployment.model_name,
      "--port",
      String(deployment.port),
      "--gpu-memory-utilization",
      String(deployment.gpu_memory_fraction)
    ];
    if (deployment.tensor_parallel_size) {
      vllmArgs.push("--tensor-parallel-size", String(deployment.tensor_parallel_size));
    }
    if (deployment.engine_args) {
      for (const [key, value] of Object.entries(deployment.engine_args)) {
        const flag = `--${key.replace(/_/g, "-")}`;
        if (value === true) {
          vllmArgs.push(flag);
        } else if (value !== false && value !== null && value !== "") {
          vllmArgs.push(flag, String(value));
        }
      }
    }
    if (deployment.extra_args && deployment.extra_args.length > 0) {
      vllmArgs.push(...deployment.extra_args);
    }

    lines.push([...dockerParts, ...vllmArgs.map(shellQuote)].join(" "));
    return lines.join("\n");
  };

  const extraEnvVars = useMemo(() => {
    return envVars
      .map((entry) => ({ key: entry.key.trim(), value: entry.value }))
      .filter((entry) => entry.key.length > 0);
  }, [envVars]);

  const cleanedExtraPackages = useMemo(() => {
    return extraPackagesText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#"));
  }, [extraPackagesText]);

  const engineArgs = useMemo(() => {
    const args: Record<string, unknown> = {};
    const maxLen = parseInt(maxModelLen, 10);
    if (Number.isFinite(maxLen) && maxLen > 0) args.max_model_len = maxLen;
    if (revision.trim()) args.revision = revision.trim();
    const seedValue = parseInt(seed, 10);
    if (Number.isFinite(seedValue)) args.seed = seedValue;
    if (dtype) args.dtype = dtype;
    if (quantization) args.quantization = quantization;
    if (servedModelName.trim()) args.served_model_name = servedModelName.trim();
    const maxSeqs = parseInt(maxNumSeqs, 10);
    if (Number.isFinite(maxSeqs) && maxSeqs > 0) args.max_num_seqs = maxSeqs;
    if (enforceEager) args.enforce_eager = true;
    if (trustRemoteCode) args.trust_remote_code = true;
    return args;
  }, [maxModelLen, revision, seed, dtype, quantization, servedModelName, maxNumSeqs, enforceEager, trustRemoteCode]);

  const parsedMaxFailedRestarts = useMemo(() => {
    const value = parseInt(maxFailedRestarts, 10);
    return Number.isFinite(value) && value > 0 ? value : null;
  }, [maxFailedRestarts]);

  const cleanedLoraModules = useMemo(() => {
    return loraModules
      .map((entry) => ({ name: entry.name.trim(), path: entry.path.trim() }))
      .filter((entry) => entry.name.length > 0 && entry.path.length > 0);
  }, [loraModules]);

  // Live counts surfaced in the accordion summaries so collapsed optional
  // sections still show what's configured.
  const argsEnvCount =
    advancedArgs.filter((entry) => entry.key.trim()).length +
    envVars.filter((entry) => entry.key.trim()).length +
    (rawArgs.trim() ? 1 : 0);
  const engineOptionsCount =
    Object.keys(engineArgs).length +
    (parsedMaxFailedRestarts != null ? 1 : 0) +
    (skipResourceCheck ? 1 : 0);
  const runtimeSummary = vllmVersion.trim() ? `v${vllmVersion.trim()}` : "latest";

  const handlePackageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !nodeId) return;
    setUploadingPackage(true);
    setUploadError("");
    setUploadSuccess("");
    try {
      const result = await uploadPackage(Number(nodeId), file);
      if (result.type === "plugin") {
        // .py plugin files are passed via CLI flags (e.g. --reasoning-parser-plugin)
        // Add to raw args so user can reference it
        setRawArgs((prev) => {
          const trimmed = prev.trim();
          const sep = trimmed.length > 0 ? " " : "";
          return `${trimmed}${sep}${result.install_path}`;
        });
        setExpandedSection("args");
        setUploadSuccess(
          `Plugin uploaded to ${result.install_path} — added to raw args. ` +
          `Prepend the appropriate flag (e.g. --reasoning-parser-plugin).`
        );
      } else {
        setExtraPackagesText((prev) => {
          const trimmed = prev.trimEnd();
          const sep = trimmed.length > 0 ? "\n" : "";
          return `${trimmed}${sep}${result.install_path}`;
        });
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadingPackage(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const loadConfig = (config: DeploymentConfig) => {
    const payload = config.payload ?? {};
    const modelNameValue = String(payload.model_name ?? "");
    const portValue = Number(payload.port ?? 8001);
    const fractionValue = Number(payload.gpu_memory_fraction ?? 0.5);
    const gpuIdsValue = Array.isArray(payload.gpu_ids)
      ? payload.gpu_ids.map((id) => Number(id)).filter((id) => !Number.isNaN(id))
      : [];
    const extraArgsValue = Array.isArray(payload.extra_args) ? payload.extra_args : [];
    const advancedArgsValue = Array.isArray(payload.advanced_args)
      ? payload.advanced_args
      : [];
    const rawArgsValue = typeof payload.raw_args === "string" ? payload.raw_args : "";
    const envVarsValue = Array.isArray(payload.env_vars) ? payload.env_vars : [];

    setModelName(modelNameValue);
    setPort(Number.isNaN(portValue) ? 8001 : portValue);
    setGpuFraction(Number.isNaN(fractionValue) ? 0.5 : fractionValue);
    setGpuIds(gpuIdsValue);
    if (advancedArgsValue.length > 0) {
      setAdvancedArgs(
        advancedArgsValue.map((entry: { key?: string; value?: string }, index: number) => ({
          id: advancedArgId.current + index,
          key: String(entry.key ?? ""),
          value: String(entry.value ?? "")
        }))
      );
      advancedArgId.current += advancedArgsValue.length;
    } else if (extraArgsValue.length > 0) {
      const parsedPairs: Array<{ key: string; value: string }> = [];
      let i = 0;
      while (i < extraArgsValue.length) {
        const token = String(extraArgsValue[i] ?? "");
        if (!token.startsWith("--")) {
          i += 1;
          continue;
        }
        const next = String(extraArgsValue[i + 1] ?? "");
        if (next && !next.startsWith("--")) {
          parsedPairs.push({ key: token, value: next });
          i += 2;
        } else {
          parsedPairs.push({ key: token, value: "" });
          i += 1;
        }
      }
      setAdvancedArgs(
        parsedPairs.map((entry, index) => ({
          id: advancedArgId.current + index,
          key: entry.key,
          value: entry.value
        }))
      );
      advancedArgId.current += parsedPairs.length;
    } else {
      setAdvancedArgs([]);
    }
    setRawArgs(rawArgsValue);
    setEnvVars(
      envVarsValue.map((env, index) => ({
        id: envVarId.current + index,
        key: String(env.key ?? ""),
        value: String(env.value ?? "")
      }))
    );
    envVarId.current += envVarsValue.length;
    const loraValue = Array.isArray(payload.lora_modules) ? payload.lora_modules : [];
    setLoraModules(
      loraValue.map((entry: { name?: string; path?: string }, index: number) => ({
        id: loraModuleId.current + index,
        name: String(entry.name ?? ""),
        path: String(entry.path ?? "")
      }))
    );
    loraModuleId.current += loraValue.length;
    setVllmVersion(typeof payload.vllm_version === "string" ? payload.vllm_version : "");
    const engineArgsValue =
      payload.engine_args && typeof payload.engine_args === "object"
        ? (payload.engine_args as Record<string, unknown>)
        : {};
    setMaxModelLen(engineArgsValue.max_model_len != null ? String(engineArgsValue.max_model_len) : "");
    setRevision(typeof engineArgsValue.revision === "string" ? engineArgsValue.revision : "");
    setSeed(engineArgsValue.seed != null ? String(engineArgsValue.seed) : "");
    setDtype(typeof engineArgsValue.dtype === "string" ? engineArgsValue.dtype : "");
    setQuantization(
      typeof engineArgsValue.quantization === "string" ? engineArgsValue.quantization : ""
    );
    setServedModelName(
      typeof engineArgsValue.served_model_name === "string"
        ? engineArgsValue.served_model_name
        : ""
    );
    setMaxNumSeqs(engineArgsValue.max_num_seqs != null ? String(engineArgsValue.max_num_seqs) : "");
    setEnforceEager(engineArgsValue.enforce_eager === true);
    setTrustRemoteCode(engineArgsValue.trust_remote_code === true);
    const loadedPackages = Array.isArray(payload.extra_packages) ? payload.extra_packages.join("\n") : "";
    setExtraPackagesText(loadedPackages);
    // Accordion summaries surface the loaded values via their live counts.
    if ("duration_seconds" in payload) {
      const stored = payload.duration_seconds;
      const seconds = stored === null ? null : Number(stored);
      const { choice, customHours: hours } = secondsToChoice(
        seconds === null || Number.isNaN(seconds as number) ? null : (seconds as number)
      );
      setDurationChoice(choice);
      setCustomHours(hours);
    }
  };

  const gpuAllocationWarning = useMemo(() => {
    if (!selectedNode || gpuIds.length === 0) {
      return null;
    }

    const deployments = deploymentsQuery.data ?? [];
    const activeStatuses = new Set(["running", "loading", "stopping"]);
    const available = availableGpuIds.length > 0 ? availableGpuIds : gpuIds;

    const usageByGpu = new Map<number, number>();
    for (const deployment of deployments) {
      if (deployment.node_id !== selectedNode.id) {
        continue;
      }
      if (!activeStatuses.has(deployment.status)) {
        continue;
      }
      const targetGpus =
        deployment.gpu_ids && deployment.gpu_ids.length > 0 ? deployment.gpu_ids : available;
      for (const gpuId of targetGpus) {
        usageByGpu.set(
          gpuId,
          (usageByGpu.get(gpuId) ?? 0) + deployment.gpu_memory_fraction
        );
      }
    }

    const overAllocated = gpuIds.filter(
      (gpuId) => (usageByGpu.get(gpuId) ?? 0) + gpuFraction > 1
    );

    if (overAllocated.length === 0) {
      return null;
    }

    return `GPU memory fraction exceeds 1.0 on GPU ${overAllocated.join(", ")}.`;
  }, [availableGpuIds, deploymentsQuery.data, gpuFraction, gpuIds, selectedNode]);

  const portInUseWarning = useMemo(() => {
    if (!selectedNode || portCheckQuery.isLoading) {
      return null;
    }
    if (portCheckQuery.isError) {
      return "Unable to verify port availability on this node.";
    }
    if (portCheckQuery.data && portCheckQuery.data.available === false) {
      return `Port ${port} is already in use on ${selectedNode.hostname}.`;
    }
    return null;
  }, [port, portCheckQuery.data, portCheckQuery.isError, portCheckQuery.isLoading, selectedNode]);

  return (
    <Box className="app">
      <Box className="brand">
        {/* Plain h1: MUI Typography's body1 font-size would override the
            .brand-title CSS (its runtime styles land after the stylesheet). */}
        <h1 className="brand-title">vLLM CLUSTER MANAGER</h1>
        <Tooltip title="Settings" enterDelay={500}>
          <IconButton aria-label="Settings" onClick={() => setSettingsOpen(true)}>
            <SettingsOutlinedIcon sx={{ fontSize: 20 }} />
          </IconButton>
        </Tooltip>
      </Box>
      <Box className="app-shell">
        {/* Compact stat strip — three numbers don't warrant a full panel. */}
        <Paper className="panel overview-strip" sx={{ py: 2 }}>
            <Box className="stack">
              <StatusCard label="Nodes" value={statusSummary.totalNodes} />
              <StatusCard
                label="Healthy"
                value={statusSummary.healthyNodes}
                accent="success"
              />
              <StatusCard
                label="Models"
                value={statusSummary.activeModels}
                accent="warning"
              />
            </Box>
          </Paper>

          <Paper className="panel deploy-rail">
            <Box className="panel-header">
              <Box>
                <Typography variant="h6">Deploy Model</Typography>
                <Typography variant="body2" className="muted">
                  Launch new vLLM instances.
                </Typography>
              </Box>
              <AppButton type="button" onClick={() => setConfigDialogOpen(true)}>
                Manage Configurations
              </AppButton>
            </Box>
            <Stack spacing={2}>
              {(configsQuery.data ?? []).length > 0 && (
                <TextField
                  fullWidth
                  select
                  label="Load saved config"
                  value=""
                  onChange={(event) => {
                    const config = (configsQuery.data ?? []).find(
                      (entry) => String(entry.id) === event.target.value
                    );
                    if (config) {
                      loadConfig(config);
                      toast.success(`Loaded "${config.name}".`);
                    }
                  }}
                >
                  {(configsQuery.data ?? []).map((config) => (
                    <MenuItem key={config.id} value={String(config.id)}>
                      {config.name}
                    </MenuItem>
                  ))}
                </TextField>
              )}
              <TextField
                fullWidth
                select
                label="Node"
                value={nodeId}
                onChange={(event) => setNodeId(event.target.value)}
              >
                {(nodesQuery.data ?? []).map((node) => (
                  <MenuItem key={node.id} value={node.id} disabled={Boolean(node.maintenance)}>
                    {node.hostname} ({node.ip_address})
                    {node.maintenance ? " — maintenance" : ""}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                fullWidth
                label="Model Name"
                value={modelName}
                onChange={(event) => setModelName(event.target.value)}
              />
              {nodeId !== "" && (localModelsQuery.data ?? []).length > 0 && (
                <TextField
                  fullWidth
                  select
                  label="Local model on this node (optional)"
                  value=""
                  onChange={(event) => {
                    if (event.target.value) {
                      setModelName(event.target.value);
                    }
                  }}
                  helperText="Fills the model name with the checkpoint's path."
                >
                  {(localModelsQuery.data ?? []).map((model) => (
                    <MenuItem key={model.path} value={model.path}>
                      {model.name} ({(model.size_mb / 1024).toFixed(1)} GB)
                      {model.source === "managed" ? "" : ` — ${model.source}`}
                    </MenuItem>
                  ))}
                </TextField>
              )}
              <TextField
                fullWidth
                required
                label="Owner (your name/ID)"
                value={owner}
                onChange={(event) => setOwner(event.target.value)}
              />
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                <TextField
                  fullWidth
                  label="Port"
                  type="number"
                  value={port}
                  onChange={(event) => setPort(Number(event.target.value))}
                />
                <TextField
                  fullWidth
                  label="GPU Fraction"
                  type="number"
                  inputProps={{ step: 0.1, min: 0.1, max: 1.0 }}
                  value={gpuFraction}
                  onChange={(event) => setGpuFraction(Number(event.target.value))}
                />
                <TextField
                  fullWidth
                  select
                  label="Serve for"
                  value={durationChoice}
                  onChange={(event) => setDurationChoice(event.target.value)}
                >
                  {DURATION_OPTIONS.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </TextField>
              </Stack>
              {durationChoice === "custom" && (
                <TextField
                  label="Custom hours"
                  type="number"
                  inputProps={{ step: 0.5, min: 0.1 }}
                  value={customHours}
                  onChange={(event) => setCustomHours(event.target.value)}
                  helperText="Hours to serve before the model is auto-stopped."
                  sx={{ maxWidth: 240 }}
                />
              )}
              <Box>
                <SectionLabel sx={{ mb: 0.5 }}>GPUs</SectionLabel>
                {availableGpuIds.length > 0 ? (
                  <ToggleButtonGroup
                    value={gpuIds}
                    onChange={(_, newIds) => setGpuIds(newIds as number[])}
                    size="small"
                    sx={{ flexWrap: "wrap", gap: 0.5 }}
                  >
                    {availableGpuIds.map((gpuId) => (
                      <ToggleButton
                        key={gpuId}
                        value={gpuId}
                        sx={{ px: 1.5, py: 0.5, fontSize: "0.8rem" }}
                      >
                        GPU {gpuId}
                      </ToggleButton>
                    ))}
                  </ToggleButtonGroup>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    {selectedNode ? "No GPUs reported on this node." : "Select a node first."}
                  </Typography>
                )}
              </Box>
              <Box>
                {/* Optional sections: slim exclusive accordions with live counts. */}
                <Divider />
                <Accordion
                  expanded={expandedSection === "args"}
                  onChange={(_, open) => setExpandedSection(open ? "args" : false)}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />}>
                    <SectionLabel>Arguments & Environment</SectionLabel>
                    {argsEnvCount > 0 && (
                      <Typography variant="caption" className="muted">
                        {argsEnvCount} set
                      </Typography>
                    )}
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={1.5}>
                      {advancedArgs.map((entry, index) => (
                        <Stack key={entry.id} direction="row" spacing={1.5} alignItems="center">
                          <TextField
                            fullWidth
                            label="Flag"
                            placeholder="--max-model-len"
                            value={entry.key}
                            onChange={(event) => {
                              const next = [...advancedArgs];
                              next[index] = { ...entry, key: event.target.value };
                              setAdvancedArgs(next);
                            }}
                          />
                          <TextField
                            fullWidth
                            label="Value"
                            placeholder="4096"
                            value={entry.value}
                            onChange={(event) => {
                              const next = [...advancedArgs];
                              next[index] = { ...entry, value: event.target.value };
                              setAdvancedArgs(next);
                            }}
                          />
                          <IconButton
                            size="small"
                            aria-label="Remove option"
                            onClick={() => {
                              setAdvancedArgs(advancedArgs.filter((_, i) => i !== index));
                            }}
                            sx={{ color: "text.secondary" }}
                          >
                            <ClearIcon fontSize="inherit" />
                          </IconButton>
                        </Stack>
                      ))}
                      <Button
                        variant="text"
                        size="small"
                        startIcon={<AddIcon />}
                        sx={{ alignSelf: "flex-start" }}
                        onClick={() =>
                          setAdvancedArgs([
                            ...advancedArgs,
                            { id: advancedArgId.current++, key: "", value: "" }
                          ])
                        }
                      >
                        Add option
                      </Button>
                      <TextField
                        fullWidth
                        label="Raw Args"
                        placeholder="--dtype bf16 --max-model-len 8192"
                        value={rawArgs}
                        onChange={(event) => setRawArgs(event.target.value)}
                        helperText="Raw CLI args are appended after advanced options."
                      />
                      <SectionLabel>Environment Variables</SectionLabel>
                      {envVars.map((entry, index) => (
                        <Stack key={entry.id} direction="row" spacing={1.5} alignItems="center">
                          <TextField
                            fullWidth
                            label="Name"
                            placeholder="HF_HOME"
                            value={entry.key}
                            onChange={(event) => {
                              const next = [...envVars];
                              next[index] = { ...entry, key: event.target.value };
                              setEnvVars(next);
                            }}
                          />
                          <TextField
                            fullWidth
                            label="Value"
                            placeholder="/mnt/models"
                            value={entry.value}
                            onChange={(event) => {
                              const next = [...envVars];
                              next[index] = { ...entry, value: event.target.value };
                              setEnvVars(next);
                            }}
                          />
                          <IconButton
                            size="small"
                            aria-label="Remove environment variable"
                            onClick={() => {
                              setEnvVars(envVars.filter((_, i) => i !== index));
                            }}
                            sx={{ color: "text.secondary" }}
                          >
                            <ClearIcon fontSize="inherit" />
                          </IconButton>
                        </Stack>
                      ))}
                      <Button
                        variant="text"
                        size="small"
                        startIcon={<AddIcon />}
                        sx={{ alignSelf: "flex-start" }}
                        onClick={() =>
                          setEnvVars([...envVars, { id: envVarId.current++, key: "", value: "" }])
                        }
                      >
                        Add environment variable
                      </Button>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
                <Divider />
                <Accordion
                  expanded={expandedSection === "lora"}
                  onChange={(_, open) => setExpandedSection(open ? "lora" : false)}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />}>
                    <SectionLabel>LoRA Adapters</SectionLabel>
                    {cleanedLoraModules.length > 0 && (
                      <Typography variant="caption" className="muted">
                        {cleanedLoraModules.length} set
                      </Typography>
                    )}
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={1.5}>
                      {loraModules.map((entry, index) => (
                        <Stack key={entry.id} direction="row" spacing={1.5} alignItems="center">
                          <TextField
                            fullWidth
                            label="Name"
                            placeholder="my-adapter"
                            value={entry.name}
                            onChange={(event) => {
                              const next = [...loraModules];
                              next[index] = { ...entry, name: event.target.value };
                              setLoraModules(next);
                            }}
                          />
                          <TextField
                            fullWidth
                            label="Path or HF id"
                            placeholder="/models/loras/my-adapter"
                            value={entry.path}
                            onChange={(event) => {
                              const next = [...loraModules];
                              next[index] = { ...entry, path: event.target.value };
                              setLoraModules(next);
                            }}
                          />
                          <IconButton
                            size="small"
                            aria-label="Remove LoRA adapter"
                            onClick={() => {
                              setLoraModules(loraModules.filter((_, i) => i !== index));
                            }}
                            sx={{ color: "text.secondary" }}
                          >
                            <ClearIcon fontSize="inherit" />
                          </IconButton>
                        </Stack>
                      ))}
                      <Button
                        variant="text"
                        size="small"
                        startIcon={<AddIcon />}
                        sx={{ alignSelf: "flex-start" }}
                        onClick={() =>
                          setLoraModules([
                            ...loraModules,
                            { id: loraModuleId.current++, name: "", path: "" }
                          ])
                        }
                      >
                        Add LoRA adapter
                      </Button>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
                <Divider />
                <Accordion
                  expanded={expandedSection === "engine"}
                  onChange={(_, open) => setExpandedSection(open ? "engine" : false)}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />}>
                    <SectionLabel>Engine Options</SectionLabel>
                    {engineOptionsCount > 0 && (
                      <Typography variant="caption" className="muted">
                        {engineOptionsCount} set
                      </Typography>
                    )}
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={1.5}>
                      <Stack direction="row" spacing={1.5}>
                        <TextField
                          fullWidth
                          label="HF revision"
                          placeholder="--revision"
                          value={revision}
                          onChange={(event) => setRevision(event.target.value)}
                        />
                        <TextField
                          fullWidth
                          label="Seed"
                          placeholder="--seed"
                          type="number"
                          value={seed}
                          onChange={(event) => setSeed(event.target.value)}
                        />
                      </Stack>
                      <Stack direction="row" spacing={1.5}>
                        <TextField
                          fullWidth
                          label="Max model len"
                          placeholder="--max-model-len"
                          type="number"
                          inputProps={{ min: 1 }}
                          value={maxModelLen}
                          onChange={(event) => setMaxModelLen(event.target.value)}
                        />
                        <TextField
                          fullWidth
                          label="Max num seqs"
                          placeholder="--max-num-seqs"
                          type="number"
                          inputProps={{ min: 1 }}
                          value={maxNumSeqs}
                          onChange={(event) => setMaxNumSeqs(event.target.value)}
                        />
                      </Stack>
                      <Stack direction="row" spacing={1.5}>
                        <TextField
                          fullWidth
                          select
                          label="Dtype"
                          value={dtype}
                          onChange={(event) => setDtype(event.target.value)}
                        >
                          {DTYPE_OPTIONS.map((option) => (
                            <MenuItem key={option || "default"} value={option}>
                              {option || "default"}
                            </MenuItem>
                          ))}
                        </TextField>
                        <TextField
                          fullWidth
                          select
                          label="Quantization"
                          value={quantization}
                          onChange={(event) => setQuantization(event.target.value)}
                        >
                          {QUANTIZATION_OPTIONS.map((option) => (
                            <MenuItem key={option || "none"} value={option}>
                              {option || "none"}
                            </MenuItem>
                          ))}
                        </TextField>
                      </Stack>
                      <TextField
                        fullWidth
                        label="Served model name"
                        placeholder="Name exposed on the OpenAI API"
                        value={servedModelName}
                        onChange={(event) => setServedModelName(event.target.value)}
                      />
                      <Stack direction="row" spacing={2}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={enforceEager}
                              onChange={(event) => setEnforceEager(event.target.checked)}
                            />
                          }
                          label="Enforce eager"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={trustRemoteCode}
                              onChange={(event) => setTrustRemoteCode(event.target.checked)}
                            />
                          }
                          label="Trust remote code"
                        />
                      </Stack>
                      <Stack direction="row" spacing={2} alignItems="center">
                        <TextField
                          label="Max failed restarts"
                          placeholder="3"
                          type="number"
                          inputProps={{ min: 1 }}
                          value={maxFailedRestarts}
                          onChange={(event) => setMaxFailedRestarts(event.target.value)}
                          sx={{ maxWidth: 220 }}
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={skipResourceCheck}
                              onChange={(event) => setSkipResourceCheck(event.target.checked)}
                            />
                          }
                          label="Skip GPU memory check"
                        />
                      </Stack>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
                <Divider />
                <Accordion
                  expanded={expandedSection === "runtime"}
                  onChange={(_, open) => setExpandedSection(open ? "runtime" : false)}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />}>
                    <SectionLabel>Runtime</SectionLabel>
                    <Typography variant="caption" className="muted">
                      {runtimeSummary}
                      {cleanedExtraPackages.length > 0
                        ? ` · ${cleanedExtraPackages.length} pkg`
                        : ""}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={1.5}>
                      <TextField
                        fullWidth
                        label="vLLM version"
                        placeholder={
                          latestVllmQuery.data?.version
                            ? `${latestVllmQuery.data.version} (latest), nightly, or commit hash`
                            : "version, nightly, or commit hash"
                        }
                        value={vllmVersion}
                        onChange={(e) => setVllmVersion(e.target.value)}
                      />
                      <TextField
                        fullWidth
                        multiline
                        minRows={2}
                        maxRows={6}
                        label="Extra packages"
                        placeholder={"vllm-flash-attn\ncustom-plugin==1.0.0"}
                        value={extraPackagesText}
                        onChange={(e) => setExtraPackagesText(e.target.value)}
                        helperText="One package per line (requirements.txt format). Or upload a .py, .whl, or .tar.gz below."
                      />
                      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept=".py,.whl,.tar.gz,.zip"
                          style={{ display: "none" }}
                          onChange={handlePackageUpload}
                        />
                        <AppButton
                          type="button"
                          className="app-button--small"
                          disabled={!nodeId || uploadingPackage}
                          onClick={() => fileInputRef.current?.click()}
                        >
                          {uploadingPackage ? "Uploading..." : "Upload Package"}
                        </AppButton>
                        {uploadError && (
                          <Typography variant="caption" color="error">
                            {uploadError}
                          </Typography>
                        )}
                        {uploadSuccess && (
                          <Typography variant="caption" color="success.main">
                            {uploadSuccess}
                          </Typography>
                        )}
                      </Stack>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
                <Divider />
              </Box>
              {/* Sticky footer: blocking warnings always sit next to the
                  action they block, even while the long form scrolls. */}
              <Box
                sx={{
                  position: "sticky",
                  // Compensates the -20px bottom margin: sticky pins the
                  // margin box, so without this the footer floats 20px up
                  // and form rows peek out beneath it.
                  bottom: "-20px",
                  zIndex: 2,
                  bgcolor: "var(--panel)",
                  borderTop: "1px solid var(--line)",
                  mx: "-24px",
                  mb: "-20px",
                  px: "24px",
                  py: 2,
                  borderBottomLeftRadius: "var(--radius-lg)",
                  borderBottomRightRadius: "var(--radius-lg)"
                }}
              >
                {gpuAllocationWarning && (
                  <Alert severity="warning" sx={{ mb: 1.5 }}>
                    {gpuAllocationWarning}
                  </Alert>
                )}
                {portInUseWarning && (
                  <Alert severity="warning" sx={{ mb: 1.5 }}>
                    {portInUseWarning}
                  </Alert>
                )}
                {startMutation.isError && (
                  <Alert severity="error" sx={{ mb: 1.5 }}>
                    {startMutation.error?.message ?? "Failed to deploy. Check backend logs."}
                  </Alert>
                )}
                <Button
                  fullWidth
                  variant="contained"
                  disabled={
                    !canDeploy || startMutation.isPending || gpuAllocationWarning !== null
                  }
                  onClick={() =>
                    startMutation.mutate({
                      node_id: Number(nodeId),
                      model_name: modelName.trim(),
                      owner: owner.trim(),
                      duration_seconds: durationSeconds,
                      port,
                      gpu_memory_fraction: gpuFraction,
                      gpu_ids: gpuIds,
                      extra_args: extraArgs.length > 0 ? extraArgs : undefined,
                      env_vars: extraEnvVars.length > 0 ? extraEnvVars : undefined,
                      vllm_version: vllmVersion.trim() || undefined,
                      extra_packages: cleanedExtraPackages.length > 0 ? cleanedExtraPackages : undefined,
                      engine_args: Object.keys(engineArgs).length > 0 ? engineArgs : undefined,
                      lora_modules:
                        cleanedLoraModules.length > 0 ? cleanedLoraModules : undefined,
                      max_failed_restarts: parsedMaxFailedRestarts,
                      skip_resource_check: skipResourceCheck || undefined
                    })
                  }
                >
                  {startMutation.isPending ? "Starting..." : "Deploy Model"}
                </Button>
                {startMutation.isPending && (
                  <Box sx={{ mt: 1.5 }}>
                    <LinearProgress sx={{ mb: 0.5, borderRadius: 1 }} />
                    <Typography variant="caption" className="muted">
                      {vllmVersion.trim()
                        ? `Installing vLLM ${vllmVersion.trim()}`
                        : "Installing latest stable vLLM"}
                      {cleanedExtraPackages.length > 0
                        ? ` + ${cleanedExtraPackages.length} extra package(s)`
                        : ""}
                      {" — creating isolated environment. This may take a few minutes."}
                    </Typography>
                  </Box>
                )}
              </Box>
            </Stack>
          </Paper>

        <Box className="stage">
          <Paper className="panel">
            <Box className="panel-header">
              <Box>
                <Typography variant="h6">Model Deployments</Typography>
                <Typography variant="body2" className="muted">
                  Active and recent vLLM models.
                </Typography>
              </Box>
              <Tooltip
                title={
                  wsConnected
                    ? "Live updates over WebSocket."
                    : "WebSocket down — falling back to 5s polling."
                }
                enterDelay={300}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      bgcolor: wsConnected ? "success.main" : "warning.main"
                    }}
                  />
                  <Typography variant="caption" className="muted">
                    {wsConnected ? "Live" : "Polling"}
                  </Typography>
                </Box>
              </Tooltip>
            </Box>
            <DeploymentTable
              deployments={visibleDeployments}
              loading={deploymentsQuery.isLoading}
              onStop={(id) => stopMutation.mutate(id)}
              onDelete={(id) => deleteMutation.mutate(id)}
              onLogs={(id) => setLogsDeploymentId(id)}
              onSettings={(deployment) => setSettingsDeployment(deployment)}
              onRestart={openRestart}
              onExtend={(deployment, extension) =>
                extendMutation.mutate({ id: deployment.id, extension })
              }
              onEndpoint={setEndpointDeployment}
              nodeNameById={nodeNameById}
            />
          </Paper>

          <Paper className="panel">
            <Box className="panel-header">
              <Box>
                <Typography variant="h6">Nodes</Typography>
                <Typography variant="body2" className="muted">
                  Healthy nodes ready for deployments.
                </Typography>
              </Box>
            </Box>
            <NodeTable
              nodes={nodesQuery.data ?? []}
              loading={nodesQuery.isLoading}
              onManage={setDockerNode}
              onToggleMaintenance={openMaintenance}
            />
          </Paper>
        </Box>
      </Box>

      <AppDialog
        open={logsDeploymentId !== null}
        onClose={() => setLogsDeploymentId(null)}
        title="Terminal Output"
        meta={
          logsDeploymentId
            ? `Deployment #${logsDeploymentId} — live tail; the download holds the full run`
            : undefined
        }
        paperClassName="terminal-shell"
        actions={
          <>
            <AppButton
              type="button"
              ghost
              onClick={() => {
                if (logsDeploymentId !== null) {
                  window.open(deploymentLogsDownloadUrl(logsDeploymentId), "_blank");
                }
              }}
            >
              Download full log
            </AppButton>
            <AppButton type="button" onClick={() => setLogsDeploymentId(null)}>
              Close
            </AppButton>
          </>
        }
      >
        <Box
          ref={logsBodyRef}
          onScroll={handleLogsScroll}
          className="code-area terminal-input"
          sx={{ minHeight: 280 }}
        >
          {(logsQuery.data?.lines ?? []).join("\n")}
        </Box>
      </AppDialog>

      <AppDialog
        open={settingsDeployment !== null}
        onClose={() => setSettingsDeployment(null)}
        title="Deployment Settings"
        meta={
          settingsDeployment
            ? `${settingsDeployment.model_name} (port ${settingsDeployment.port})`
            : undefined
        }
        paperClassName="terminal-shell"
        actions={
          <>
            <AppButton type="button" onClick={() => setSettingsDeployment(null)}>
              Close
            </AppButton>
            <AppButton
              type="button"
              onClick={() => setManifestDeployment(settingsDeployment)}
            >
              Manifest
            </AppButton>
          </>
        }
      >
        <Box className="code-area terminal-input" sx={{ minHeight: 160 }}>
          {settingsDeployment ? buildVllmCommand(settingsDeployment) : ""}
        </Box>
      </AppDialog>

      <AppDialog
        open={manifestDeployment !== null}
        onClose={() => setManifestDeployment(null)}
        title="Deployment Manifest"
        meta="Env var values are omitted; redeploys pin vLLM version + HF revision + seed."
        paperClassName="terminal-shell"
        actions={
          <>
            <AppButton type="button" onClick={() => setManifestDeployment(null)}>
              Close
            </AppButton>
            <AppButton type="button" disabled={!manifestJson} onClick={downloadManifest}>
              Download
            </AppButton>
            <AppButton
              type="button"
              disabled={!manifestQuery.data}
              onClick={loadManifestIntoForm}
            >
              Load into Form
            </AppButton>
          </>
        }
      >
        <Box sx={{ position: "relative" }}>
          <Box className="code-area terminal-input" sx={{ minHeight: 200, pr: 5 }}>
            {manifestQuery.isLoading ? "Loading…" : manifestJson}
          </Box>
          <Tooltip title="Copy" enterDelay={500}>
            <span style={{ position: "absolute", top: 0, right: 0 }}>
              <IconButton
                size="small"
                aria-label="Copy manifest"
                disabled={!manifestJson}
                onClick={copyManifest}
              >
                <ContentCopyIcon fontSize="inherit" />
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </AppDialog>

      <AppDialog
        open={configDialogOpen}
        onClose={() => setConfigDialogOpen(false)}
        title="Manage Configurations"
        maxWidth="sm"
        actions={
          <AppButton type="button" onClick={() => setConfigDialogOpen(false)}>
            Close
          </AppButton>
        }
      >
        <Stack spacing={1.5}>
          <Stack direction="row" spacing={1} alignItems="center">
            <TextField
              fullWidth
              label="Configuration Name"
              size="small"
              value={configName}
              onChange={(event) => setConfigName(event.target.value)}
            />
            <AppButton
              type="button"
              className="app-button--small"
              disabled={configName.trim().length === 0}
              onClick={() =>
                createConfigMutation.mutate({
                  name: configName.trim(),
                  payload: {
                    model_name: modelName.trim(),
                    port,
                    gpu_memory_fraction: gpuFraction,
                    gpu_ids: gpuIds,
                    extra_args: extraArgs,
                    advanced_args: advancedArgs.map((entry) => ({
                      key: entry.key.trim(),
                      value: entry.value
                    })),
                    raw_args: rawArgs,
                    env_vars: extraEnvVars,
                    vllm_version: vllmVersion.trim(),
                    extra_packages: cleanedExtraPackages,
                    engine_args: engineArgs,
                    lora_modules: cleanedLoraModules,
                    duration_seconds: durationSeconds
                  }
                })
              }
            >
              Save
            </AppButton>
          </Stack>
          <Box>
            {(configsQuery.data ?? []).map((config) => (
              <Box
                key={config.id}
                className="list-row"
                onClick={() => {
                  loadConfig(config);
                  setConfigDialogOpen(false);
                }}
              >
                <Typography variant="body2">{config.name}</Typography>
                <Tooltip title="Delete" enterDelay={500}>
                  <IconButton
                    size="small"
                    aria-label={`Delete config ${config.name}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      deleteConfigMutation.mutate(config.id);
                    }}
                    sx={{ color: "text.secondary", "&:hover": { color: "error.main" } }}
                  >
                    <DeleteOutlineIcon fontSize="inherit" />
                  </IconButton>
                </Tooltip>
              </Box>
            ))}
          </Box>
          {(configsQuery.data ?? []).length === 0 && (
            <EmptyState
              primary="No saved configs yet."
              hint="Fill in the launch form, name it above, and press Save."
            />
          )}
        </Stack>
      </AppDialog>

      <NodeDockerDialog
        node={dockerNode}
        open={dockerNode !== null}
        onClose={() => setDockerNode(null)}
      />

      <EndpointDialog
        deployment={endpointDeployment}
        node={
          (nodesQuery.data ?? []).find(
            (node) => node.id === endpointDeployment?.node_id
          ) ?? null
        }
        open={endpointDeployment !== null}
        onClose={() => setEndpointDeployment(null)}
      />

      <AppDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Settings"
        maxWidth="xs"
        actions={
          <AppButton type="button" onClick={() => setSettingsOpen(false)}>
            Close
          </AppButton>
        }
      >
        <SectionLabel sx={{ mb: 0.5 }}>Danger zone</SectionLabel>
        <Typography variant="body2" className="muted" sx={{ mb: 1.5 }}>
          Purging deletes all deployments, nodes, metric history, and saved
          configurations from the database. Running models are not stopped —
          nodes re-register and their deployments are re-adopted automatically
          within ~15 seconds.
        </Typography>
        <AppButton
          type="button"
          variant="stop"
          onClick={() => setConfirmPurge(true)}
          disabled={purgeMutation.isPending}
        >
          Purge database
        </AppButton>
      </AppDialog>

      <ConfirmDialog
        open={confirmPurge}
        title="Purge the database?"
        body="All deployments, nodes, metric history, and saved configurations will be deleted. This cannot be undone."
        confirmLabel="Purge"
        danger
        onConfirm={() => purgeMutation.mutate()}
        onCancel={() => setConfirmPurge(false)}
      />

      <AppDialog
        open={maintenanceTarget !== null}
        onClose={() => setMaintenanceTarget(null)}
        title={
          maintenanceTarget?.maintenance
            ? `End maintenance on ${maintenanceTarget?.hostname}?`
            : `Put ${maintenanceTarget?.hostname} into maintenance?`
        }
        maxWidth="xs"
        actions={
          <>
            <AppButton type="button" onClick={() => setMaintenanceTarget(null)}>
              Cancel
            </AppButton>
            <Button
              variant="contained"
              color={maintenanceTarget?.maintenance ? "primary" : "warning"}
              disabled={maintenanceMutation.isPending}
              onClick={() =>
                maintenanceTarget &&
                maintenanceMutation.mutate({
                  node: maintenanceTarget,
                  enabled: !maintenanceTarget.maintenance,
                  drain: maintenanceDrain
                })
              }
            >
              {maintenanceMutation.isPending
                ? "Applying..."
                : maintenanceTarget?.maintenance
                  ? "End Maintenance"
                  : maintenanceDrain
                    ? "Cordon & Drain"
                    : "Cordon"}
            </Button>
          </>
        }
      >
        {maintenanceTarget?.maintenance ? (
          <Typography variant="body2">
            The node returns to rotation and accepts new deployments again.
          </Typography>
        ) : (
          <Stack spacing={1}>
            <Typography variant="body2">
              No new deployments can be started on the node, and health flaps during the
              maintenance window are ignored.
            </Typography>
            <FormControlLabel
              control={
                <Checkbox
                  checked={maintenanceDrain}
                  onChange={(event) => setMaintenanceDrain(event.target.checked)}
                />
              }
              label="Also stop all active deployments on this node (drain)"
            />
          </Stack>
        )}
      </AppDialog>

      <AppDialog
        open={restartTarget !== null}
        onClose={() => setRestartTarget(null)}
        title={`Restart ${restartTarget?.model_name ?? ""}`}
        maxWidth="xs"
        actions={
          <>
            <AppButton type="button" onClick={() => setRestartTarget(null)}>
              Cancel
            </AppButton>
            <Button
              variant="contained"
              disabled={
                restartOwner.trim().length === 0 ||
                !isDurationValid(restartDurationChoice, restartCustomHours) ||
                restartMutation.isPending
              }
              onClick={() =>
                restartTarget &&
                restartMutation.mutate({
                  id: restartTarget.id,
                  owner: restartOwner.trim(),
                  durationSeconds: durationChoiceToSeconds(
                    restartDurationChoice,
                    restartCustomHours
                  )
                })
              }
            >
              {restartMutation.isPending ? "Starting..." : "Start"}
            </Button>
          </>
        }
      >
        <Stack spacing={2} sx={{ mt: 0.5 }}>
            <TextField
              fullWidth
              required
              label="Owner (your name/ID)"
              value={restartOwner}
              onChange={(event) => setRestartOwner(event.target.value)}
            />
            <TextField
              fullWidth
              select
              label="Serve for"
              value={restartDurationChoice}
              onChange={(event) => setRestartDurationChoice(event.target.value)}
            >
              {DURATION_OPTIONS.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </TextField>
            {restartDurationChoice === "custom" && (
              <TextField
                fullWidth
                label="Custom hours"
                type="number"
                inputProps={{ step: 0.5, min: 0.1 }}
                value={restartCustomHours}
                onChange={(event) => setRestartCustomHours(event.target.value)}
              />
            )}
            {restartMutation.isError && (
              <Typography variant="body2" color="error">
                {(restartMutation.error as Error).message}
              </Typography>
            )}
        </Stack>
      </AppDialog>
    </Box>
  );
}

import { useMemo, useRef, useState } from "react";
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  MenuItem,
  Paper,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  fetchDeployments,
  fetchConfigs,
  fetchNodes,
  type Deployment,
  type DeploymentConfig,
  deleteDeployment,
  fetchDeploymentLogs,
  checkNodePort,
  createConfig,
  deleteConfig,
  startDeployment,
  stopDeployment
} from "../services/api";
import { StatusCard } from "../components/StatusCard";
import { NodeTable } from "../components/NodeTable";
import { DeploymentTable } from "../components/DeploymentTable";
import { AppButton } from "../components/AppButton";

export function Dashboard() {
  const [nodeId, setNodeId] = useState("");
  const [modelName, setModelName] = useState("");
  const [port, setPort] = useState(8001);
  const [gpuFraction, setGpuFraction] = useState(0.5);
  const [gpuIds, setGpuIds] = useState<number[]>([]);
  const [advancedArgs, setAdvancedArgs] = useState<
    Array<{ id: number; key: string; value: string }>
  >([]);
  const [rawArgs, setRawArgs] = useState("");
  const [showRawArgs, setShowRawArgs] = useState(false);
  const advancedArgId = useRef(1);
  const [envVars, setEnvVars] = useState<Array<{ id: number; key: string; value: string }>>(
    []
  );
  const envVarId = useRef(1);
  const [configName, setConfigName] = useState("");
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [logsDeploymentId, setLogsDeploymentId] = useState<number | null>(null);
  const [settingsDeployment, setSettingsDeployment] = useState<Deployment | null>(null);
  const queryClient = useQueryClient();

  const nodesQuery = useQuery({
    queryKey: ["nodes"],
    queryFn: fetchNodes,
    refetchInterval: 5000
  });

  const deploymentsQuery = useQuery({
    queryKey: ["deployments"],
    queryFn: fetchDeployments,
    refetchInterval: 5000
  });

  const configsQuery = useQuery({
    queryKey: ["configs"],
    queryFn: fetchConfigs
  });

  const portCheckQuery = useQuery({
    queryKey: ["port-check", nodeId, port],
    queryFn: () => checkNodePort(Number(nodeId), port),
    enabled: nodeId !== "" && !Number.isNaN(Number(nodeId)) && port > 0,
    refetchInterval: 5000
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
      setModelName("");
      setPort(8001);
      setGpuFraction(0.5);
      setGpuIds([]);
      setAdvancedArgs([]);
      setRawArgs("");
      setShowRawArgs(false);
      setEnvVars([]);
      setConfigName("");
    }
  });

  const stopMutation = useMutation({
    mutationFn: stopDeployment,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
    }
  });

  const deleteMutation = useMutation({
    mutationFn: deleteDeployment,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments"] });
    }
  });

  const createConfigMutation = useMutation({
    mutationFn: ({ name, payload }: { name: string; payload: Record<string, unknown> }) =>
      createConfig(name, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["configs"] });
      setConfigName("");
    }
  });

  const deleteConfigMutation = useMutation({
    mutationFn: (configId: number) => deleteConfig(configId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["configs"] });
    }
  });

  const logsQuery = useQuery({
    queryKey: ["deployment-logs", logsDeploymentId],
    queryFn: () => fetchDeploymentLogs(logsDeploymentId ?? 0, 400),
    enabled: logsDeploymentId !== null,
    refetchInterval: logsDeploymentId !== null ? 2000 : false
  });

  const canDeploy = nodeId !== "" && modelName.trim().length > 0 && !Number.isNaN(Number(nodeId));

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
    if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
      return value;
    }
    return `'${value.replace(/'/g, `'\"'\"'`)}'`;
  };

  const buildVllmCommand = (deployment: Deployment) => {
    const envParts: string[] = [];
    if (deployment.gpu_ids && deployment.gpu_ids.length > 0) {
      envParts.push(`CUDA_VISIBLE_DEVICES=${shellQuote(deployment.gpu_ids.join(","))}`);
    }
    if (deployment.env_vars && deployment.env_vars.length > 0) {
      for (const pair of deployment.env_vars) {
        if (!pair.key) {
          continue;
        }
        envParts.push(`${pair.key}=${shellQuote(String(pair.value ?? ""))}`);
      }
    }

    const cmdParts = [
      "python",
      "-m",
      "vllm.entrypoints.openai.api_server",
      "--model",
      deployment.model_name,
      "--port",
      String(deployment.port),
      "--gpu-memory-utilization",
      String(deployment.gpu_memory_fraction)
    ];
    if (deployment.extra_args && deployment.extra_args.length > 0) {
      cmdParts.push(...deployment.extra_args);
    }

    const envPrefix = envParts.length > 0 ? `${envParts.join(" ")} ` : "";
    return `${envPrefix}${cmdParts.map(shellQuote).join(" ")}`;
  };

  const extraEnvVars = useMemo(() => {
    return envVars
      .map((entry) => ({ key: entry.key.trim(), value: entry.value }))
      .filter((entry) => entry.key.length > 0);
  }, [envVars]);

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
        <Typography
          className="brand-title"
          sx={{ fontSize: 44, fontWeight: 300, marginBottom: 0 }}
        >
          vLLM Cluster
        </Typography>
        <Typography
          className="brand-subtitle"
          sx={{
            color: "#9aa5b1",
            fontWeight: 200,
            letterSpacing: "0.22em",
            marginTop: -0.5,
            fontSize: 20
          }}
        >
          Admin Dashboard
        </Typography>
      </Box>
      <Box className="app-shell">
        <Box className="side-rail">
          <Paper className="panel">
            <Box className="panel-header">
              <Box>
                <Typography variant="h6">Overview</Typography>
                <Typography variant="body2" className="muted">
                  Snapshot of active capacity.
                </Typography>
              </Box>
            </Box>
            <Box className="stack">
              <StatusCard label="Total Nodes" value={statusSummary.totalNodes} />
              <StatusCard
                label="Healthy Nodes"
                value={statusSummary.healthyNodes}
                accent="#1f9d55"
              />
              <StatusCard
                label="Active Models"
                value={statusSummary.activeModels}
                accent="#ff9f1c"
              />
            </Box>
          </Paper>

          <Paper className="panel">
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
              <TextField
                fullWidth
                select
                label="Node"
                value={nodeId}
                onChange={(event) => setNodeId(event.target.value)}
              >
                {(nodesQuery.data ?? []).map((node) => (
                  <MenuItem key={node.id} value={node.id}>
                    {node.hostname} ({node.ip_address})
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                fullWidth
                label="Model Name"
                value={modelName}
                onChange={(event) => setModelName(event.target.value)}
              />
              <Stack direction="row" spacing={2}>
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
              </Stack>
              <TextField
                fullWidth
                select
                label="GPU IDs"
                SelectProps={{
                  multiple: true,
                  value: gpuIds,
                  onChange: (event) => {
                    const value = event.target.value;
                    setGpuIds(Array.isArray(value) ? (value as number[]) : []);
                  }
                }}
                disabled={!selectedNode || availableGpuIds.length === 0}
                helperText={
                  availableGpuIds.length === 0
                    ? "No GPUs reported on this node."
                    : "Select one or more GPUs."
                }
              >
                {availableGpuIds.map((gpuId) => (
                  <MenuItem key={gpuId} value={gpuId}>
                    GPU {gpuId}
                  </MenuItem>
                ))}
              </TextField>
              <Stack spacing={1}>
                <Typography
                  variant="body2"
                  className="muted"
                  sx={{ textTransform: "uppercase", letterSpacing: "0.16em", fontSize: "0.7rem" }}
                >
                  Advanced Options
                </Typography>
                {advancedArgs.map((entry, index) => (
                  <Stack key={entry.id} direction="row" spacing={2} sx={{ mt: 0.5 }}>
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
                    <AppButton
                      type="button"
                      className="app-button--small"
                      onClick={() => {
                        const next = advancedArgs.filter((_, i) => i !== index);
                        setAdvancedArgs(next);
                      }}
                    >
                      Remove
                    </AppButton>
                  </Stack>
                ))}
                <Box>
                  <AppButton
                    type="button"
                    onClick={() =>
                      setAdvancedArgs([
                        ...advancedArgs,
                        { id: advancedArgId.current++, key: "", value: "" }
                      ])
                    }
                  >
                    Add Option
                  </AppButton>
                </Box>
                <Box>
                  <AppButton type="button" onClick={() => setShowRawArgs((prev) => !prev)}>
                    {showRawArgs ? "Hide Raw Args" : "Add Raw Args"}
                  </AppButton>
                </Box>
                {showRawArgs && (
                  <TextField
                    fullWidth
                    label="Raw Args"
                    placeholder="--dtype bf16 --max-model-len 8192"
                    value={rawArgs}
                    onChange={(event) => setRawArgs(event.target.value)}
                    helperText="Raw CLI args are appended after advanced options."
                  />
                )}
                <Typography
                  variant="body2"
                  className="muted"
                  sx={{ textTransform: "uppercase", letterSpacing: "0.16em", fontSize: "0.7rem" }}
                >
                  Environment Variables
                </Typography>
                {envVars.map((entry, index) => (
                  <Stack key={entry.id} direction="row" spacing={2} sx={{ mt: 0.5 }}>
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
                    <AppButton
                      type="button"
                      className="app-button--small"
                      onClick={() => {
                        const next = envVars.filter((_, i) => i !== index);
                        setEnvVars(next);
                      }}
                    >
                      Remove
                    </AppButton>
                  </Stack>
                ))}
                <Box>
                  <AppButton
                    type="button"
                    onClick={() =>
                      setEnvVars([...envVars, { id: envVarId.current++, key: "", value: "" }])
                    }
                  >
                    Add Environment Variable
                  </AppButton>
                </Box>
              </Stack>
              <Box>
                <Button
                  variant="contained"
                  disabled={
                    !canDeploy || startMutation.isPending || gpuAllocationWarning !== null
                  }
                  onClick={() =>
                    startMutation.mutate({
                      node_id: Number(nodeId),
                      model_name: modelName.trim(),
                      port,
                      gpu_memory_fraction: gpuFraction,
                      gpu_ids: gpuIds,
                      extra_args: extraArgs.length > 0 ? extraArgs : undefined,
                      env_vars: extraEnvVars.length > 0 ? extraEnvVars : undefined
                    })
                  }
                >
                  {startMutation.isPending ? "Starting..." : "Deploy Model"}
                </Button>
                {gpuAllocationWarning && (
                  <Typography variant="body2" color="warning.main" sx={{ mt: 1 }}>
                    {gpuAllocationWarning}
                  </Typography>
                )}
                {portInUseWarning && (
                  <Typography variant="body2" color="warning.main" sx={{ mt: 1 }}>
                    {portInUseWarning}
                  </Typography>
                )}
                {startMutation.isError && (
                  <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                    {startMutation.error?.message ?? "Failed to deploy. Check backend logs."}
                  </Typography>
                )}
              </Box>
            </Stack>
          </Paper>
        </Box>

        <Box className="stage">
          <Paper className="panel">
            <Box className="panel-header">
              <Box>
                <Typography variant="h6">Model Deployments</Typography>
                <Typography variant="body2" className="muted">
                  Active and recent vLLM models.
                </Typography>
              </Box>
            </Box>
            <DeploymentTable
              deployments={visibleDeployments}
              onStop={(id) => stopMutation.mutate(id)}
              onDelete={(id) => deleteMutation.mutate(id)}
              onLogs={(id) => setLogsDeploymentId(id)}
              onSettings={(deployment) => setSettingsDeployment(deployment)}
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
            <NodeTable nodes={nodesQuery.data ?? []} />
          </Paper>
        </Box>
      </Box>

      <Dialog
        open={logsDeploymentId !== null}
        onClose={() => setLogsDeploymentId(null)}
        fullWidth
        maxWidth="md"
        PaperProps={{ className: "panel terminal-shell" }}
      >
        <DialogTitle>Terminal Output</DialogTitle>
        <DialogContent dividers className="terminal-body">
          <Box
            className="code-area terminal-input"
            sx={{
              minHeight: 280,
              whiteSpace: "pre-wrap",
              fontFamily: "SFMono-Regular, ui-monospace, monospace",
              fontSize: "12.5px",
              lineHeight: 1.5,
              color: "inherit",
              background: "transparent",
              border: "none",
              padding: 0
            }}
          >
            {(logsQuery.data?.lines ?? []).join("\n")}
          </Box>
        </DialogContent>
        <DialogActions className="terminal-controls">
          <Typography variant="body2" className="muted">
            {logsDeploymentId ? `Deployment #${logsDeploymentId}` : ""}
          </Typography>
          <AppButton type="button" onClick={() => setLogsDeploymentId(null)}>
            Close
          </AppButton>
        </DialogActions>
      </Dialog>

      <Dialog
        open={settingsDeployment !== null}
        onClose={() => setSettingsDeployment(null)}
        fullWidth
        maxWidth="md"
        PaperProps={{ className: "panel terminal-shell" }}
      >
        <DialogTitle>Deployment Settings</DialogTitle>
        <DialogContent dividers className="terminal-body">
          <Box
            className="code-area terminal-input"
            sx={{
              minHeight: 160,
              whiteSpace: "pre-wrap",
              fontFamily: "SFMono-Regular, ui-monospace, monospace",
              fontSize: "12.5px",
              lineHeight: 1.5,
              color: "inherit",
              background: "transparent",
              border: "none",
              padding: 0
            }}
          >
            {settingsDeployment ? buildVllmCommand(settingsDeployment) : ""}
          </Box>
        </DialogContent>
        <DialogActions className="terminal-controls">
          <Typography variant="body2" className="muted">
            {settingsDeployment
              ? `${settingsDeployment.model_name} (port ${settingsDeployment.port})`
              : ""}
          </Typography>
          <AppButton type="button" onClick={() => setSettingsDeployment(null)}>
            Close
          </AppButton>
        </DialogActions>
      </Dialog>

      <Dialog
        open={configDialogOpen}
        onClose={() => setConfigDialogOpen(false)}
        fullWidth
        maxWidth="sm"
        PaperProps={{ className: "panel" }}
      >
        <DialogTitle>Manage Configurations</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField
                fullWidth
                label="Configuration Name"
                size="small"
                value={configName}
                onChange={(event) => setConfigName(event.target.value)}
                sx={{
                  "& .MuiInputBase-input": {
                    fontSize: "0.875rem"
                  },
                  "& .MuiOutlinedInput-notchedOutline": {
                    borderColor: "rgba(148, 163, 184, 0.6)"
                  },
                  "&:hover .MuiOutlinedInput-notchedOutline": {
                    borderColor: "rgba(148, 163, 184, 0.6)"
                  },
                  "& .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline": {
                    borderColor: "rgba(148, 163, 184, 0.6)"
                  }
                }}
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
                      env_vars: extraEnvVars
                    }
                  })
                }
              >
                Save
              </AppButton>
            </Stack>
            {(configsQuery.data ?? []).map((config) => (
              <Stack key={config.id} direction="row" spacing={1} alignItems="center">
                <AppButton
                  type="button"
                  className="app-button--small"
                  onClick={() => {
                    loadConfig(config);
                    setConfigDialogOpen(false);
                  }}
                  style={{ flex: 1, justifyContent: "flex-start" }}
                >
                  {config.name}
                </AppButton>
                <AppButton
                  type="button"
                  className="app-button--small"
                  onClick={() => deleteConfigMutation.mutate(config.id)}
                >
                  Delete
                </AppButton>
              </Stack>
            ))}
            {(configsQuery.data ?? []).length === 0 && (
              <Typography variant="body2" className="muted">
                No saved configs yet.
              </Typography>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <AppButton type="button" onClick={() => setConfigDialogOpen(false)}>
            Close
          </AppButton>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

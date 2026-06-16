export type NodeDiskUsage = {
  total_gb?: number;
  free_gb?: number;
  hf_cache_gb?: number;
};

export type Node = {
  id: number;
  hostname: string;
  ip_address: string;
  status: string;
  maintenance?: boolean;
  port?: number | null;
  gpu_usage?: {
    index: number;
    name?: string;
    source?: string;
    utilization?: number;
    memory_used_mb?: number;
    memory_total_mb?: number;
  }[];
  disk_usage?: NodeDiskUsage | null;
  default_pip_packages?: string[];
  installed_packages?: string[];
  rogue_container_count?: number | null;
  // Orphaned vLLM GPU processes (workers that outlived their container).
  rogue_process_count?: number | null;
  // Orphaned warm-cache artifacts (RAM sleepers + disk compile caches).
  rogue_artifact_count?: number | null;
  // Warm cache: opt-in auto-offload toggle + RAM-cache budget/usage (MB).
  warm_offload_enabled?: boolean;
  ram_cache_limit_mb?: number | null;
  ram_cache_used_mb?: number | null;
  // Detected container runtimes ("docker"/"podman"); empty = node unusable.
  available_runtimes?: string[];
  // Per-node runtime override; null = auto (preferred runtime).
  container_runtime?: string | null;
  last_heartbeat_at?: string | null;
};

export type Deployment = {
  id: number;
  node_id: number;
  container_runtime?: string | null;
  model_name: string;
  port: number;
  gpu_memory_fraction: number;
  gpu_ids?: number[];
  tensor_parallel_size?: number | null;
  extra_args?: string[];
  env_vars?: { key: string; value: string }[];
  pip_packages?: string[];
  vllm_version?: string | null;
  extra_packages?: string[];
  engine_args?: Record<string, unknown> | null;
  lora_modules?: { name: string; path: string }[] | null;
  max_failed_restarts?: number | null;
  owner?: string | null;
  duration_seconds?: number | null;
  expires_at?: string | null;
  // Warm cache: protect this deployment from automatic eviction.
  pinned?: boolean;
  status: string;
  // Cumulative usage from the vLLM instance's Prometheus counters.
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  total_requests?: number;
  // Live metrics from the latest scrape (running deployments only).
  // *_tps: per-request, idle-free speeds; *_throughput: engine-wide window rate.
  prompt_tps?: number | null;
  generation_tps?: number | null;
  prompt_throughput?: number | null;
  generation_throughput?: number | null;
  requests_running?: number | null;
  requests_waiting?: number | null;
  // Image-pull progress while starting (transient).
  pull_percent?: number | null;
  pull_downloaded_mb?: number | null;
  pull_total_mb?: number | null;
  // Failure reason (client error or watchdog timeout) for error states.
  last_error?: string | null;
  // Load phase while status is "loading" (downloading/loading_weights/compiling).
  detail?: string | null;
  status_changed_at?: string | null;
  created_at?: string | null;
};

const baseUrl = withBase("api");

// Absolute OpenAI-compatible gateway base URL, honoring the UI base path
// (the dev/preview server and reverse-proxy configs forward /v1 to the backend).
export function gatewayBaseUrl(): string {
  return `${window.location.origin}${withBase("v1")}`;
}

function withBase(path: string): string {
  const base = import.meta.env.BASE_URL || "/";
  const normalized = base.endsWith("/") ? base : `${base}/`;
  return `${normalized}${path}`.replace(/\/{2,}/g, "/").replace(/:\//, "://");
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchNodes(): Promise<Node[]> {
  return request<Node[]>("/nodes/");
}

export function fetchLatestVllmVersion(): Promise<{ version: string }> {
  return request<{ version: string }>("/vllm-version");
}

export function checkNodePort(
  nodeId: number,
  port: number
): Promise<{ available: boolean }> {
  return request<{ available: boolean }>(`/nodes/${nodeId}/ports/check?port=${port}`);
}

export type ServedNameCheck = {
  available: boolean;
  conflict_id?: number;
  conflict_model?: string;
  suggestion?: string;
};

export function checkServedName(
  name: string,
  excludeId?: number
): Promise<ServedNameCheck> {
  const params = new URLSearchParams({ name });
  if (excludeId !== undefined) params.set("exclude_id", String(excludeId));
  return request<ServedNameCheck>(`/deployments/served-name/check?${params.toString()}`);
}

export function fetchDeployments(): Promise<Deployment[]> {
  return request<Deployment[]>("/deployments/");
}

export function fetchDiscovered(): Promise<{ nodes: { node: string; address: string }[] }> {
  return request<{ nodes: { node: string; address: string }[] }>("/nodes/discovered");
}

export type DeploymentStart = {
  node_id: number;
  model_name: string;
  port: number;
  gpu_memory_fraction: number;
  gpu_ids?: number[];
  tensor_parallel_size?: number | null;
  extra_args?: string[];
  env_vars?: { key: string; value: string }[];
  vllm_version?: string;
  extra_packages?: string[];
  engine_args?: Record<string, unknown>;
  lora_modules?: { name: string; path: string }[];
  max_failed_restarts?: number | null;
  skip_resource_check?: boolean;
  owner: string;
  duration_seconds?: number | null;
  // Warm cache: launch protected from automatic eviction.
  pinned?: boolean;
};

export async function startDeployment(payload: DeploymentStart): Promise<Deployment> {
  const response = await fetch(`${baseUrl}/deployments/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as Deployment;
}

export async function stopDeployment(deploymentId: number): Promise<Deployment> {
  const response = await fetch(`${baseUrl}/deployments/stop/${deploymentId}`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as Deployment;
}

export async function restartDeployment(
  deploymentId: number,
  owner: string,
  durationSeconds: number | null
): Promise<Deployment> {
  const response = await fetch(`${baseUrl}/deployments/${deploymentId}/restart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ owner, duration_seconds: durationSeconds })
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as Deployment;
}

export type DeploymentExtension = { hours: number } | { infinite: true };

export async function extendDeployment(
  deploymentId: number,
  extension: DeploymentExtension
): Promise<Deployment> {
  const response = await fetch(`${baseUrl}/deployments/${deploymentId}/extend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(extension)
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as Deployment;
}

export async function deleteDeployment(deploymentId: number): Promise<void> {
  const response = await fetch(`${baseUrl}/deployments/${deploymentId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
}

export type DeploymentLogs = {
  key: string;
  lines: string[];
};

export async function fetchDeploymentLogs(
  deploymentId: number,
  tail = 200
): Promise<DeploymentLogs> {
  const response = await fetch(`${baseUrl}/deployments/${deploymentId}/logs?tail=${tail}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as DeploymentLogs;
}

// Full persisted log of the deployment's current run (streamed attachment).
export function deploymentLogsDownloadUrl(deploymentId: number): string {
  return `${baseUrl}/deployments/${deploymentId}/logs/download`;
}

export type DeploymentConfig = {
  id: number;
  name: string;
  payload: Record<string, unknown>;
  created_at?: string | null;
};

export async function fetchConfigs(): Promise<DeploymentConfig[]> {
  return request<DeploymentConfig[]>("/configs/");
}

export async function createConfig(
  name: string,
  payload: Record<string, unknown>
): Promise<DeploymentConfig> {
  const response = await fetch(`${baseUrl}/configs/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, payload })
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as DeploymentConfig;
}

export async function deleteConfig(configId: number): Promise<void> {
  const response = await fetch(`${baseUrl}/configs/${configId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
}

export type PackageUploadResult = {
  package_id: string;
  filename: string;
  install_path: string;
  type: "package" | "plugin";
};

export async function uploadPackage(
  nodeId: number,
  file: File
): Promise<PackageUploadResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${baseUrl}/nodes/${nodeId}/packages/upload`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `Upload failed: ${response.status}`);
  }
  return (await response.json()) as PackageUploadResult;
}

export type NodePackage = {
  package_id: string;
  filename: string;
};

export async function fetchNodePackages(nodeId: number): Promise<NodePackage[]> {
  return request<NodePackage[]>(`/nodes/${nodeId}/packages`);
}

// ---------------------------------------------------------------------------
// Per-node container runtime management (containers + image cache)
// ---------------------------------------------------------------------------

export type NodeContainer = {
  id: string;
  name: string;
  image: string;
  status: string;
  runtime?: string;
  managed: boolean;
  tracked: boolean;
  key?: string | null;
};

export type NodeImage = {
  id: string;
  tags: string[];
  size_mb: number;
  runtime?: string;
  // Every runtime store holding a copy; the cache is one logical store.
  runtimes?: string[];
};

export type GpuProcess = {
  pid: number;
  gpu_index?: number | null;
  gpu_memory_mb?: number | null;
  process_name: string;
  tracked: boolean;
  key?: string | null;
};

export type ImagePruneResult = {
  removed: string[];
  freed_mb: number;
  skipped: string[];
};

async function requestWithDetail<T>(
  path: string,
  method: "GET" | "POST" | "PUT" | "DELETE" = "GET",
  jsonBody?: unknown
): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    ...(jsonBody !== undefined
      ? {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(jsonBody)
        }
      : {})
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchNodeContainers(nodeId: number): Promise<NodeContainer[]> {
  return requestWithDetail<NodeContainer[]>(`/nodes/${nodeId}/containers`);
}

export function stopNodeContainer(
  nodeId: number,
  containerId: string
): Promise<{ status: string; id: string }> {
  return requestWithDetail(
    `/nodes/${nodeId}/containers/${encodeURIComponent(containerId)}/stop`,
    "POST"
  );
}

export function fetchNodeGpuProcesses(nodeId: number): Promise<GpuProcess[]> {
  return requestWithDetail<GpuProcess[]>(`/nodes/${nodeId}/gpu-processes`);
}

export function killNodeGpuProcess(
  nodeId: number,
  pid: number
): Promise<{ status: string; pid: number }> {
  return requestWithDetail(`/nodes/${nodeId}/gpu-processes/${pid}/kill`, "POST");
}

// --- Warm cache (pause/resume) ---------------------------------------------

export type RamSleeper = {
  pid: number;
  process_name: string;
  rss_mb: number;
};

export type DiskCache = {
  name: string;
  path: string;
  size_mb: number;
};

export type WarmArtifacts = {
  ram_sleepers: RamSleeper[];
  disk_caches: DiskCache[];
};

// Enable/disable warm-cache auto-offload and set the RAM-cache budget (MB; null
// = unlimited). Applies to new deployments; running ones keep their mode.
export function setNodeWarmCache(
  nodeId: number,
  enabled: boolean,
  ramCacheLimitMb: number | null
): Promise<Node> {
  return requestWithDetail(`/nodes/${nodeId}/warm-cache`, "POST", {
    enabled,
    ram_cache_limit_mb: ramCacheLimitMb
  });
}

export function pinDeployment(
  deploymentId: number,
  pinned: boolean
): Promise<Deployment> {
  return requestWithDetail(`/deployments/${deploymentId}/pin`, "POST", { pinned });
}

export function pauseDeployment(
  deploymentId: number,
  tier?: "ram" | "disk"
): Promise<Deployment> {
  return requestWithDetail(`/deployments/${deploymentId}/pause`, "POST", {
    tier: tier ?? null
  });
}

export function resumeDeployment(deploymentId: number): Promise<Deployment> {
  return requestWithDetail(`/deployments/${deploymentId}/resume`, "POST");
}

export function fetchNodeWarmArtifacts(nodeId: number): Promise<WarmArtifacts> {
  return requestWithDetail<WarmArtifacts>(`/nodes/${nodeId}/warm-artifacts`);
}

export function killNodeRamSleeper(
  nodeId: number,
  pid: number
): Promise<{ status: string; pid: number }> {
  return requestWithDetail(
    `/nodes/${nodeId}/warm-artifacts/sleepers/${pid}/kill`,
    "POST"
  );
}

export function deleteNodeWarmCache(
  nodeId: number,
  name: string
): Promise<{ status: string; name: string }> {
  return requestWithDetail(`/nodes/${nodeId}/warm-artifacts/caches/${name}`, "DELETE");
}

export function fetchNodeImages(nodeId: number): Promise<NodeImage[]> {
  return requestWithDetail<NodeImage[]>(`/nodes/${nodeId}/images`);
}

export function deleteNodeImage(
  nodeId: number,
  imageId: string
): Promise<{ status: string; id: string }> {
  return requestWithDetail(
    `/nodes/${nodeId}/images/${encodeURIComponent(imageId)}`,
    "DELETE"
  );
}

export function pruneNodeImages(nodeId: number): Promise<ImagePruneResult> {
  return requestWithDetail(`/nodes/${nodeId}/images/prune`, "POST");
}

// ---------------------------------------------------------------------------
// Node maintenance, metrics history, HF model cache
// ---------------------------------------------------------------------------

export async function setNodeMaintenance(
  nodeId: number,
  enabled: boolean,
  drain: boolean
): Promise<Node> {
  const response = await fetch(`${baseUrl}/nodes/${nodeId}/maintenance`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled, drain })
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as Node;
}

export type NodeMetricPoint = {
  recorded_at: string | null;
  gpus: {
    index: number;
    utilization?: number;
    memory_used_mb?: number;
    memory_total_mb?: number;
  }[];
  cpu_percent?: number | null;
  memory_percent?: number | null;
};

export type NodeMetricsHistory = {
  node_id: number;
  points: NodeMetricPoint[];
};

export function fetchNodeMetricsHistory(
  nodeId: number,
  minutes: number,
  step: number
): Promise<NodeMetricsHistory> {
  return request<NodeMetricsHistory>(
    `/nodes/${nodeId}/metrics/history?minutes=${minutes}&step=${step}`
  );
}

export type CachedModel = {
  name: string;
  size_mb: number;
  last_used_at: number;
  in_use: boolean;
};

export function fetchNodeModelCache(nodeId: number): Promise<CachedModel[]> {
  return requestWithDetail<CachedModel[]>(`/nodes/${nodeId}/models/cache`);
}

export type DeploymentManifest = Record<string, unknown>;

export function fetchManifest(deploymentId: number): Promise<DeploymentManifest> {
  return request<DeploymentManifest>(`/deployments/${deploymentId}/manifest`);
}

export function deleteNodeModelCache(
  nodeId: number,
  name: string
): Promise<{ status: string; name: string }> {
  return requestWithDetail(`/nodes/${nodeId}/models/cache/${name}`, "DELETE");
}

// Set the node's runtime override (null = auto). New deployments only.
export function setNodeRuntime(
  nodeId: number,
  runtime: string | null
): Promise<Node> {
  return requestWithDetail(`/nodes/${nodeId}/runtime`, "POST", { runtime });
}

// Removes a node and its deployment records. Containers are untouched: a live
// node re-registers within seconds; a stale node disappears for good.
export function deleteNode(
  nodeId: number
): Promise<{ status: string; hostname: string; deployments_deleted: number }> {
  return requestWithDetail(`/nodes/${nodeId}`, "DELETE");
}

// Selective purge: targets ⊆ {deployments, nodes, metrics, configs}; empty =
// everything. Running containers are untouched and re-register/re-adopt.
export function purgeDatabase(
  targets: string[]
): Promise<{ purged: Record<string, number> }> {
  return requestWithDetail(`/admin/purge`, "POST", { targets });
}

// ---------------------------------------------------------------------------
// Global runtime settings (DB-backed, editable in the Settings dialog)
// ---------------------------------------------------------------------------

export type RuntimeSettings = {
  gateway_enabled: boolean;
  gateway_timeout_seconds: number;
  start_timeout_seconds: number;
  preferred_container_runtime: string;
  default_port: number;
  default_gpu_fraction: number;
  default_duration_choice: string;
  default_vllm_version: string;
  default_max_failed_restarts: number | null;
  webhook_url: string;
  expiry_warning_minutes: number;
  node_metrics_retention_hours: number;
  nodes_sync_interval_seconds: number;
  deployments_sync_interval_seconds: number;
  expiry_check_interval_seconds: number;
  node_failure_threshold: number;
  deployment_failure_threshold: number;
};

export function fetchSettings(): Promise<RuntimeSettings> {
  return requestWithDetail<RuntimeSettings>(`/settings`);
}

export function updateSettings(
  partial: Partial<RuntimeSettings>
): Promise<RuntimeSettings> {
  return requestWithDetail<RuntimeSettings>(`/settings`, "PUT", partial);
}

// ---------------------------------------------------------------------------
// Managed local models (streamed uploads / URL pulls)
// ---------------------------------------------------------------------------

export type LocalModel = {
  name: string;
  path: string;
  source: string;
  size_mb: number;
  in_use: boolean;
  deletable: boolean;
  last_modified_at: number;
};

export type LocalModelTransfer = {
  id: string;
  kind: string;
  name: string;
  status: "downloading" | "extracting" | "done" | "error";
  total_bytes: number | null;
  received_bytes: number;
  error: string | null;
  path: string | null;
};

export type LocalModelUploadResult = {
  name: string;
  path: string;
  size_mb: number;
  warnings: string[];
};

export function fetchLocalModels(nodeId: number): Promise<LocalModel[]> {
  return requestWithDetail<LocalModel[]>(`/nodes/${nodeId}/local-models`);
}

export function fetchLocalModelTransfers(nodeId: number): Promise<LocalModelTransfer[]> {
  return requestWithDetail<LocalModelTransfer[]>(`/nodes/${nodeId}/local-models/transfers`);
}

export function deleteLocalModel(
  nodeId: number,
  name: string
): Promise<{ status: string; name: string }> {
  return requestWithDetail(
    `/nodes/${nodeId}/local-models/${encodeURIComponent(name)}`,
    "DELETE"
  );
}

export function pullLocalModel(
  nodeId: number,
  payload: { url: string; name?: string }
): Promise<{ transfer_id: string; name: string }> {
  return requestWithDetail(`/nodes/${nodeId}/local-models/pull`, "POST", payload);
}

// Raw-body XHR upload: xhr.send(file) streams the File from disk (no JS-side
// buffering) and, unlike fetch, exposes upload progress events.
function uploadLocalModelStream(
  path: string,
  method: "PUT" | "POST",
  file: File,
  onProgress: (loadedBytes: number) => void
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, `${baseUrl}${path}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          resolve(null);
        }
      } else {
        let detail = "";
        try {
          detail = (JSON.parse(xhr.responseText) as { detail?: string }).detail ?? "";
        } catch {
          detail = "";
        }
        reject(new Error(detail || `Upload failed: ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed: network error."));
    xhr.onabort = () => reject(new Error("Upload cancelled."));
    xhr.send(file);
  });
}

// Strip the picked folder's own name from webkitRelativePath so the model
// root holds config.json directly.
function relativePathInsideFolder(file: File): string {
  const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
  const parts = rel.split("/");
  return parts.length > 1 ? parts.slice(1).join("/") : rel;
}

export async function uploadLocalModelFolder(
  nodeId: number,
  name: string,
  files: File[],
  onProgress: (fraction: number) => void
): Promise<LocalModelUploadResult> {
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  const { session_id } = await requestWithDetail<{ session_id: string }>(
    `/nodes/${nodeId}/local-models/upload/begin`,
    "POST",
    { name, total_bytes: totalBytes, file_count: files.length }
  );
  let doneBytes = 0;
  try {
    // Sequential: keeps disk writes contiguous and the progress math exact;
    // multi-GB transfers are bandwidth-bound anyway.
    for (const file of files) {
      const rel = relativePathInsideFolder(file);
      await uploadLocalModelStream(
        `/nodes/${nodeId}/local-models/upload/${session_id}/file?path=${encodeURIComponent(rel)}`,
        "PUT",
        file,
        (loaded) => onProgress(totalBytes ? (doneBytes + loaded) / totalBytes : 0)
      );
      doneBytes += file.size;
      onProgress(totalBytes ? doneBytes / totalBytes : 1);
    }
    return await requestWithDetail<LocalModelUploadResult>(
      `/nodes/${nodeId}/local-models/upload/${session_id}/finish`,
      "POST",
      { flatten: false }
    );
  } catch (error) {
    await requestWithDetail(
      `/nodes/${nodeId}/local-models/upload/${session_id}/abort`,
      "POST"
    ).catch(() => undefined);
    throw error;
  }
}

export async function uploadLocalModelArchive(
  nodeId: number,
  name: string,
  file: File,
  onProgress: (fraction: number) => void
): Promise<LocalModelUploadResult> {
  const result = await uploadLocalModelStream(
    `/nodes/${nodeId}/local-models/archive?name=${encodeURIComponent(name)}&filename=${encodeURIComponent(file.name)}`,
    "POST",
    file,
    (loaded) => onProgress(file.size ? loaded / file.size : 0)
  );
  return result as LocalModelUploadResult;
}

export type Node = {
  id: number;
  hostname: string;
  ip_address: string;
  status: string;
  port?: number | null;
  gpu_usage?: {
    index: number;
    name?: string;
    source?: string;
    utilization?: number;
    memory_used_mb?: number;
    memory_total_mb?: number;
  }[];
  last_heartbeat_at?: string | null;
};

export type Deployment = {
  id: number;
  node_id: number;
  model_name: string;
  port: number;
  gpu_memory_fraction: number;
  gpu_ids?: number[];
  tensor_parallel_size?: number | null;
  extra_args?: string[];
  env_vars?: { key: string; value: string }[];
  status: string;
  created_at?: string | null;
};

const baseUrl = withBase("api");

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

export function checkNodePort(
  nodeId: number,
  port: number
): Promise<{ available: boolean }> {
  return request<{ available: boolean }>(`/nodes/${nodeId}/ports/check?port=${port}`);
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

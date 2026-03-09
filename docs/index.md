# vLLM Cluster Manager

<div class="hero-card"  style="margin-bottom: 2rem;">
  <h2>Operate multi-node vLLM deployments</h2>
  <p>Spin up a host with a web dashboard, then add GPU nodes with the client agent. The UI lets you launch, monitor, and troubleshoot model deployments without building a full MLOps stack.</p>
  <div class="grid-2">
    <div>
      <strong>Best for</strong>
      <ul>
        <li>Research labs</li>
        <li>Small teams</li>
        <li>Multi-model serving</li>
      </ul>
    </div>
    <div>
      <strong>Built-in</strong>
      <ul>
        <li>Service discovery</li>
        <li>Web UI</li>
        <li>Systemd support</li>
      </ul>
    </div>
  </div>
</div>

<div class="hero-image">
  <img alt="vLLM Cluster Manager dashboard" src="assets/img/vllm-cluster-manager-screenshot.png" />
</div>

## What you can do
- Register and manage GPU nodes that run vLLM workloads.
- Deploy models with a specific vLLM version, nightly build, or commit hash — each deployment gets its own isolated venv.
- Install extra pip packages and upload vLLM plugins (`.py`, `.whl`) per deployment.
- Select GPUs with toggle buttons and configure tensor parallelism.
- Save and reload deployment configurations for one-click redeployment.
- Monitor node health, GPU utilization, and deployment status in real time.
- Stream logs from running processes for quick troubleshooting.
- Automatic deployment recovery after backend restarts.

## Supported platforms
- Python 3.10–3.14
- Ubuntu 22.04 and 24.04
- NVIDIA GPUs (H100, A100, L40, RTX 4090, DGX Spark)

!!! tip
    New here? Start with the [Getting Started](getting-started.md) guide, then review the [Deployments](deployments.md) page for version selection and configuration options.

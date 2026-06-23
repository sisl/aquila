"""Transient, in-memory per-node state derived by the sync loops.

These values are *not* persisted (the schema is created with
``Base.metadata.create_all`` and has no migrations, so adding DB columns would
break existing databases). They are recomputed every sync cycle and merged into
the node read responses, so a backend restart simply rebuilds them.
"""

# node_id -> number of rogue (untracked) vLLM containers currently on the node.
rogue_container_counts: dict[int, int] = {}

# node_id -> number of rogue (orphaned) vLLM GPU processes — workers that
# outlived their container and still pin VRAM with no container to find.
rogue_process_counts: dict[int, int] = {}

# node_id -> number of orphaned warm-cache artifacts (RAM sleepers + disk
# compile caches) not attributable to any tracked deployment.
rogue_artifact_counts: dict[int, int] = {}

# node_id -> CPU RAM (MB) currently held by RAM-paused models on the node.
ram_cache_used_mb: dict[int, float] = {}

# node_id -> Aquila version string reported by the client agent.
aquila_versions: dict[int, str] = {}

"""Transient, in-memory per-node state derived by the sync loops.

These values are *not* persisted (the schema is created with
``Base.metadata.create_all`` and has no migrations, so adding DB columns would
break existing databases). They are recomputed every sync cycle and merged into
the node read responses, so a backend restart simply rebuilds them.
"""

# node_id -> number of rogue (untracked) vLLM containers currently on the node.
rogue_container_counts: dict[int, int] = {}

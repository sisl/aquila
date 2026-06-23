"""Shared model-naming helpers.

A deployment's *effective served name* is what clients address at the gateway:
the explicit ``served_model_name`` engine arg, or the model name when unset.
Its *primary aliases* (effective served name + LoRA adapter names) are the
directly-addressable identities that must be unique across active deployments,
so the gateway never has to guess between replicas. The raw model name is only
a fallback alias and may be shared (the same base model served several times
under distinct served names).
"""


def effective_served_name(deployment) -> str:
    value = (getattr(deployment, "engine_args", None) or {}).get("served_model_name")
    if isinstance(value, str) and value:
        return value
    return deployment.model_name


def lora_names(deployment) -> list[str]:
    modules = getattr(deployment, "lora_modules", None) or []
    names = []
    for module in modules:
        if isinstance(module, dict) and module.get("name"):
            names.append(str(module["name"]))
    return names


def primary_aliases(deployment) -> set[str]:
    """Names that must be globally unique among active deployments."""
    return {effective_served_name(deployment), *lora_names(deployment)}

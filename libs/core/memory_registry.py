from __future__ import annotations

from typing import Dict, Iterable, List

from libs.core.models import MemoryScope, MemorySpec


class MemoryRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, MemorySpec] = {}

    def register(self, spec: MemorySpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("MemorySpec name must be non-empty")
        if name in self._specs:
            raise ValueError(f"MemorySpec already registered: {name}")
        if spec.ttl_seconds is not None and spec.ttl_seconds <= 0:
            raise ValueError("MemorySpec ttl_seconds must be positive when provided")
        self._specs[name] = spec.model_copy(update={"name": name})

    def get(self, name: str) -> MemorySpec:
        return self._specs[name]

    def has(self, name: str) -> bool:
        return name in self._specs

    def list(self) -> List[MemorySpec]:
        return list(self._specs.values())


DEFAULT_MEMORY_SPECS: List[MemorySpec] = [
    MemorySpec(
        name="job_context",
        description="Per-job context payloads and validated inputs.",
        scope=MemoryScope.session,
        schema_def={"type": "object"},
        ttl_seconds=7 * 24 * 60 * 60,
        read_roles=["planner", "worker", "api"],
        write_roles=["api"],
    ),
    MemorySpec(
        name="task_outputs",
        description="Task outputs and intermediate artifacts for a job session.",
        scope=MemoryScope.session,
        schema_def={"type": "object"},
        ttl_seconds=7 * 24 * 60 * 60,
        read_roles=["planner", "worker", "api"],
        write_roles=["worker", "api"],
    ),
    MemorySpec(
        name="user_profile",
        description="Stable user preferences and profile attributes.",
        scope=MemoryScope.user,
        schema_def={"type": "object"},
        ttl_seconds=None,
        read_roles=["planner", "worker", "api"],
        write_roles=["api"],
    ),
    MemorySpec(
        name="project_preferences",
        description="Project-level defaults such as style guides or quality bars.",
        scope=MemoryScope.project,
        schema_def={"type": "object"},
        ttl_seconds=30 * 24 * 60 * 60,
        read_roles=["planner", "worker", "api"],
        write_roles=["api"],
    ),
    MemorySpec(
        name="global_reference",
        description="Global canonical references or policy summaries.",
        scope=MemoryScope.global_,
        schema_def={"type": "object"},
        ttl_seconds=None,
        read_roles=["planner", "worker", "api"],
        write_roles=["api"],
    ),
]


def default_memory_registry(extra_specs: Iterable[MemorySpec] | None = None) -> MemoryRegistry:
    registry = MemoryRegistry()
    for spec in DEFAULT_MEMORY_SPECS:
        registry.register(spec)
    if extra_specs:
        for spec in extra_specs:
            registry.register(spec)
    return registry

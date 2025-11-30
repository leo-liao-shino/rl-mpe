"""Factories and presets for PettingZoo MPE environments."""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from importlib import import_module
from typing import Any, Dict, Mapping, Optional

from pettingzoo.utils.env import AECEnv, ParallelEnv


@dataclass(frozen=True)
class MPEEnvironmentSpec:
    """Configuration describing how to build an MPE PettingZoo environment."""

    dotted_path: str
    max_cycles: int = 100
    render_mode: Optional[str] = None
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, **overrides: Any) -> "MPEEnvironmentSpec":
        """Return a copy of the spec with overrides applied."""

        merge_kwargs = {**self.env_kwargs, **overrides.pop("env_kwargs", {})}
        return replace(self, env_kwargs=merge_kwargs, **overrides)

    def build(self) -> AECEnv:
        """Instantiate the PettingZoo environment defined by this spec."""

        module = import_module(self.dotted_path)
        env_fn = getattr(module, "env")
        return env_fn(max_cycles=self.max_cycles, render_mode=self.render_mode, **self.env_kwargs)


PRESET_SPECS: Dict[str, MPEEnvironmentSpec] = {
    "simple_reference_v3": MPEEnvironmentSpec(
        dotted_path="pettingzoo.mpe.simple_reference_v3", max_cycles=100
    ),
    "simple_world_comm_v3": MPEEnvironmentSpec(
        dotted_path="pettingzoo.mpe.simple_world_comm_v3", max_cycles=200
    ),
}


def build_mpe_env(name_or_spec: str | MPEEnvironmentSpec, **overrides: Any) -> AECEnv:
    """Build an MPE environment from a preset name or explicit spec."""

    if isinstance(name_or_spec, str):
        try:
            base_spec = PRESET_SPECS[name_or_spec]
        except KeyError as exc:
            raise KeyError(
                f"Unknown environment preset '{name_or_spec}'. Available: {list(PRESET_SPECS)}"
            ) from exc
    else:
        base_spec = name_or_spec

    spec = base_spec.resolve(**overrides)
    return spec.build()


def build_parallel_mpe_env(
    name_or_spec: str | MPEEnvironmentSpec, **overrides: Any
) -> ParallelEnv:
    """Build the parallel-API version of an MPE environment."""

    if isinstance(name_or_spec, str):
        try:
            base_spec = PRESET_SPECS[name_or_spec]
        except KeyError as exc:
            raise KeyError(
                f"Unknown environment preset '{name_or_spec}'. Available: {list(PRESET_SPECS)}"
            ) from exc
    else:
        base_spec = name_or_spec

    spec = base_spec.resolve(**overrides)
    module = import_module(spec.dotted_path)
    env_fn = getattr(module, "parallel_env")
    return env_fn(max_cycles=spec.max_cycles, render_mode=spec.render_mode, **spec.env_kwargs)

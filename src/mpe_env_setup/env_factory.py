"""Factories and presets for PettingZoo MPE environments."""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Optional

from pettingzoo.utils.env import AECEnv, ParallelEnv
from gymnasium import spaces


@dataclass(frozen=True)
class MPEEnvironmentSpec:
    """Configuration describing how to build an MPE PettingZoo environment."""

    dotted_path: str
    max_cycles: int = 100
    render_mode: Optional[str] = None
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    post_build: Optional[Callable[[AECEnv], AECEnv]] = None
    post_build_parallel: Optional[Callable[[ParallelEnv], ParallelEnv]] = None

    def resolve(self, **overrides: Any) -> "MPEEnvironmentSpec":
        """Return a copy of the spec with overrides applied."""

        merge_kwargs = {**self.env_kwargs, **overrides.pop("env_kwargs", {})}
        return replace(self, env_kwargs=merge_kwargs, **overrides)

    def build(self) -> AECEnv:
        """Instantiate the PettingZoo environment defined by this spec."""

        module = import_module(self.dotted_path)
        env_fn = getattr(module, "env")
        env = env_fn(max_cycles=self.max_cycles, render_mode=self.render_mode, **self.env_kwargs)
        if self.post_build:
            env = self.post_build(env)
        return env


def _unwrap_base_env(env: AECEnv | ParallelEnv) -> AECEnv:
    base_env: Any = getattr(env, "aec_env", env)
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def _enable_world_comm_good_agents(env: AECEnv | ParallelEnv) -> AECEnv | ParallelEnv:
    """Allow cooperative agents in simple_world_comm_v3 to speak."""

    base_env: AECEnv = _unwrap_base_env(env)
    world = getattr(getattr(base_env, "unwrapped", base_env), "world", None)
    if world is None:
        return env

    dim_c = int(getattr(world, "dim_c", 0) or 0)
    if dim_c <= 0:
        return env

    updated: list[str] = []

    for agent in world.agents:
        if agent.adversary and not getattr(agent, "leader", False):
            continue
        if getattr(agent, "silent", False):
            agent.silent = False
        updated.append(agent.name)

    if not updated:
        return env

    for agent in world.agents:
        if agent.name not in updated:
            continue
        if base_env.continuous_actions:
            move_dim = (world.dim_p * 2 + 1) if agent.movable else 0
            space_dim = move_dim + dim_c
            base_env.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(space_dim,))
        else:
            move_dim = (world.dim_p * 2 + 1) if agent.movable else 1
            base_env.action_spaces[agent.name] = spaces.Discrete(move_dim * dim_c)

    return env


PRESET_SPECS: Dict[str, MPEEnvironmentSpec] = {
    "simple_reference_v3": MPEEnvironmentSpec(
        dotted_path="pettingzoo.mpe.simple_reference_v3", max_cycles=100
    ),
    "simple_world_comm_v3": MPEEnvironmentSpec(
        dotted_path="pettingzoo.mpe.simple_world_comm_v3",
        max_cycles=200,
        post_build=_enable_world_comm_good_agents,
        post_build_parallel=_enable_world_comm_good_agents,
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
    env = env_fn(max_cycles=spec.max_cycles, render_mode=spec.render_mode, **spec.env_kwargs)
    hook = spec.post_build_parallel or spec.post_build
    if hook:
        env = hook(env)
    return env

"""Helpers to build supervised datasets for communicative agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from pettingzoo.utils.env import ParallelEnv

from .env_factory import PRESET_SPECS, build_parallel_mpe_env, _unwrap_base_env


@dataclass
class AgentDataset:
    """Container for per-agent supervision data."""

    observations: np.ndarray
    labels: np.ndarray


def _extract_target_indices(env: ParallelEnv) -> Dict[str, int]:
    """Return the landmark index each agent is tasked with, if available."""

    base_env = _unwrap_base_env(getattr(env, "aec_env", env))
    world = getattr(base_env, "world", None)
    if world is None:
        return {}

    landmarks = list(getattr(world, "landmarks", []))
    if not landmarks:
        return {}

    indices: Dict[str, int] = {}
    for agent in getattr(world, "agents", []):
        landmark = getattr(agent, "goal_b", None)
        if landmark is None:
            continue
        try:
            idx = landmarks.index(landmark)
        except ValueError:
            continue
        indices[getattr(agent, "name", str(agent))] = idx
    return indices


def _sample_random_actions(env: ParallelEnv) -> Dict[str, object]:
    return {agent: env.action_space(agent).sample() for agent in env.possible_agents}


def collect_speaker_dataset(
    env_name: str,
    *,
    episodes: int,
    max_steps: Optional[int] = None,
    agents: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, AgentDataset], Mapping[str, int]]:
    """Collect (observation, target) pairs for communicative agents.

    Args:
        env_name: Name of the preset environment.
        episodes: Number of episodes to roll out with random policies.
        max_steps: Optional cap on episode length; defaults to preset max cycles.
        agents: Optional subset of agent names to keep. When omitted, every
            agent with a detectable `goal_b` landmark is included.
        seed: Optional RNG seed affecting environment resets.

    Returns:
        Tuple where the first element maps agent names to numpy arrays
        containing observations and labels, and the second element stores
        the number of examples collected per agent.
    """

    if env_name not in PRESET_SPECS:
        raise KeyError(
            f"Unknown environment '{env_name}'. Expected one of: {list(PRESET_SPECS)}"
        )

    env = build_parallel_mpe_env(env_name)
    rng = np.random.default_rng(seed)
    episode_cap = max_steps or PRESET_SPECS[env_name].max_cycles
    requested_agents = set(agents) if agents else None

    obs_buffers: Dict[str, List[np.ndarray]] = {}
    label_buffers: Dict[str, List[int]] = {}

    try:
        for episode in range(episodes):
            reset_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
            observations, _ = env.reset(seed=reset_seed)
            steps = 0

            while steps < episode_cap and observations:
                target_indices = _extract_target_indices(env)
                if not target_indices:
                    raise RuntimeError(
                        "Could not find target landmarks for any agent; "
                        "warm-start supervision is unavailable for this environment."
                    )

                for agent_name, label in target_indices.items():
                    if requested_agents and agent_name not in requested_agents:
                        continue
                    obs_vec = observations.get(agent_name)
                    if obs_vec is None:
                        continue
                    obs_buffers.setdefault(agent_name, []).append(
                        np.asarray(obs_vec, dtype=np.float32)
                    )
                    label_buffers.setdefault(agent_name, []).append(int(label))

                actions = _sample_random_actions(env)
                observations, _, terminations, truncations, _ = env.step(actions)
                steps += 1
                if all(terminations.values()) or all(truncations.values()):
                    break
    finally:
        env.close()

    if not obs_buffers:
        raise RuntimeError(
            "No supervision examples were collected. Ensure the selected agents "
            "exist and expose a goal landmark."
        )

    datasets: Dict[str, AgentDataset] = {}
    counts: Dict[str, int] = {}
    for agent_name, obs_list in obs_buffers.items():
        labels = label_buffers.get(agent_name, [])
        if not obs_list or not labels:
            continue
        datasets[agent_name] = AgentDataset(
            observations=np.stack(obs_list, axis=0),
            labels=np.asarray(labels, dtype=np.int64),
        )
        counts[agent_name] = len(labels)

    if not datasets:
        raise RuntimeError(
            "Supervision buffers were populated but empty after filtering; "
            "this likely indicates varying observation shapes per agent."
        )

    return datasets, counts

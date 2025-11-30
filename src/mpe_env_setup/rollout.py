"""Utility routines for running simple PettingZoo rollouts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from pettingzoo.utils.env import AECEnv

from .env_factory import PRESET_SPECS, build_mpe_env


def run_random_episode(
    env_name: str,
    *,
    max_agent_steps: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a short random-action episode in the requested environment.

    Returns summary statistics that can be reported to the CLI or tests.
    """

    if env_name not in PRESET_SPECS:
        raise KeyError(
            f"Unknown environment '{env_name}'. Expected one of: {list(PRESET_SPECS)}"
        )

    env: AECEnv = build_mpe_env(env_name)
    env.reset(seed=seed)

    max_iter = max_agent_steps * max(1, env.num_agents)
    reward_totals = defaultdict(float)
    observation_shapes: Dict[str, List[int]] = {}
    action_spaces: Dict[str, Any] = {}

    for idx, agent in enumerate(env.agent_iter(max_iter)):
        obs, reward, termination, truncation, info = env.last(observe=True)
        reward_totals[agent] += reward

        if agent not in observation_shapes:
            obs_shape = list(getattr(obs, "shape", [])) if hasattr(obs, "shape") else []
            if not obs_shape and isinstance(obs, (list, tuple)):
                obs_shape = [len(obs)]
            observation_shapes[agent] = obs_shape
            action_spaces[agent] = env.action_space(agent)

        done = termination or truncation
        action = None if done else env.action_space(agent).sample()
        env.step(action)

    env.close()

    return {
        "env_name": env_name,
        "steps_taken": idx + 1 if "idx" in locals() else 0,
        "reward_totals": dict(reward_totals),
        "num_agents": env.num_agents,
        "observation_shapes": observation_shapes,
        "action_space_types": {agent: type(space).__name__ for agent, space in action_spaces.items()},
    }

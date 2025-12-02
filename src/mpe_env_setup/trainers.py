"""Training utilities for PettingZoo MPE presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env_factory import (
    PRESET_SPECS,
    MPEEnvironmentSpec,
    build_mpe_env,
    build_parallel_mpe_env,
)

DeviceLike = Optional[str]


@dataclass(frozen=True)
class PolicyOutput:
    move_dist: Optional[Categorical]
    comm_dist: Optional[Categorical]


@dataclass(frozen=True)
class AgentActionLayout:
    move_dim: int
    comm_dim: int

    @property
    def has_comm(self) -> bool:
        return self.comm_dim > 0

    @property
    def action_space_n(self) -> int:
        move_choices = max(1, self.move_dim)
        comm_choices = max(1, self.comm_dim)
        return move_choices * comm_choices

    def flatten(self, move_idx: int, comm_idx: Optional[int] = None) -> int:
        move_choices = max(1, self.move_dim)
        if move_idx >= move_choices or move_idx < 0:
            raise ValueError(
                f"Move index {move_idx} outside valid range [0, {move_choices})"
            )
        if not self.has_comm:
            return move_idx
        if comm_idx is None:
            raise ValueError("Communication index is required for communicative agents")
        if comm_idx < 0 or comm_idx >= self.comm_dim:
            raise ValueError(
                f"Comm index {comm_idx} outside valid range [0, {self.comm_dim})"
            )
        return move_idx * self.comm_dim + comm_idx


class PolicyNetwork(nn.Module):
    """MLP policy with separate movement and communication heads."""

    def __init__(
        self,
        input_dim: int,
        move_dim: int,
        comm_dim: int,
        hidden_sizes: Iterable[int],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        self.body = nn.Sequential(*layers) if layers else nn.Identity()

        move_out = max(1, move_dim)
        self.move_head = nn.Linear(prev, move_out)
        self.move_dim = move_out

        self.comm_dim = comm_dim
        if comm_dim > 0:
            self.comm_head = nn.Linear(prev, comm_dim)
        else:
            self.comm_head = None

    def forward(self, x: torch.Tensor) -> PolicyOutput:
        hidden = self.body(x)
        move_logits = self.move_head(hidden)
        move_dist = Categorical(logits=move_logits)
        comm_dist = None
        if self.comm_head is not None:
            comm_logits = self.comm_head(hidden)
            comm_dist = Categorical(logits=comm_logits)
        return PolicyOutput(move_dist=move_dist, comm_dist=comm_dist)


@dataclass
class EpisodeStats:
    episode: int
    mean_return: float
    per_agent_return: Dict[str, float]
    steps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndependentPolicyTrainer:
    """Trains one policy per agent with REINFORCE using the parallel PettingZoo API."""

    def __init__(
        self,
        env_name: str,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_sizes: Iterable[int] = (128, 128),
        device: DeviceLike = None,
        entropy_coef: float = 0.01,
        grad_clip: Optional[float] = 0.5,
        baseline_momentum: Optional[float] = 0.95,
    ) -> None:
        if env_name not in PRESET_SPECS:
            raise KeyError(
                f"Unknown environment preset '{env_name}'. Expected one of: {list(PRESET_SPECS)}"
            )

        self.env_name = env_name
        self.spec: MPEEnvironmentSpec = PRESET_SPECS[env_name]
        self.lr = lr
        self.gamma = gamma
        self.hidden_sizes = tuple(hidden_sizes)
        self.device = _resolve_device(device)
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.baseline_momentum = baseline_momentum

        self.models: Dict[str, PolicyNetwork] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.possible_agents: List[str] = []
        self.running_baselines: Dict[str, float] = {}
        self.action_layouts: Dict[str, AgentActionLayout] = {}
        self._initialize_models()

    def load_checkpoints(
        self,
        checkpoint_dir: Path,
        *,
        strict: bool = True,
        component: str = "all",
    ) -> Dict[str, Path]:
        """Load the most recent checkpoint per agent from a directory.

        Args:
            checkpoint_dir: Directory containing files saved via `_save_checkpoint`.
            strict: When True, raise `FileNotFoundError` if any agent is missing.
            component: Which parameters to load ("all" or "comm_head").

        Returns:
            Mapping from agent name to the checkpoint path that was loaded.
        """

        if component not in {"all", "comm_head"}:
            raise ValueError("component must be 'all' or 'comm_head'")

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist")

        episode_pattern = re.compile(r"_ep(\d+)\.pt$")
        loaded: Dict[str, Path] = {}

        def _episode_num(path: Path) -> int:
            match = episode_pattern.search(path.name)
            return int(match.group(1)) if match else -1

        for agent in self.possible_agents:
            matches = sorted(checkpoint_dir.glob(f"*_{agent}_ep*.pt"))
            if not matches:
                if strict:
                    raise FileNotFoundError(
                        f"No checkpoint files for agent '{agent}' in '{checkpoint_dir}'"
                    )
                continue
            best_path = max(matches, key=lambda p: (_episode_num(p), p.stat().st_mtime))
            state = torch.load(best_path, map_location=self.device)
            model = self.models[agent]
            if component == "all":
                model.load_state_dict(state)
            else:  # comm_head only
                if model.comm_head is None:
                    if strict:
                        raise ValueError(
                            f"Agent '{agent}' has no communication head; cannot load comm weights"
                        )
                    continue
                current = model.state_dict()
                updated = False
                for key, value in state.items():
                    if key.startswith("comm_head"):
                        current[key] = value
                        updated = True
                if updated:
                    model.load_state_dict(current)
            loaded[agent] = best_path

        return loaded

    def freeze_heads(self, *, move: bool = False, comm: bool = False) -> None:
        for model in self.models.values():
            if move:
                for param in model.move_head.parameters():
                    param.requires_grad = False
            if comm and model.comm_head is not None:
                for param in model.comm_head.parameters():
                    param.requires_grad = False

    def _initialize_models(self) -> None:
        raw_env = build_mpe_env(self.env_name)
        world = getattr(raw_env.unwrapped, "world", None)
        world_dim_c = int(getattr(world, "dim_c", 0) or 0) if world else 0
        agent_traits: Dict[str, Any] = {}
        if world:
            for agent in getattr(world, "agents", []):
                agent_traits[getattr(agent, "name", str(agent))] = agent
        raw_env.close()

        env = build_parallel_mpe_env(self.env_name)
        self.possible_agents = list(env.possible_agents)
        for agent in self.possible_agents:
            obs_space = env.observation_space(agent)
            act_space = env.action_space(agent)
            obs_dim = int(np.prod(obs_space.shape or (1,)))
            action_dim = int(getattr(act_space, "n", 0))
            if action_dim <= 0:
                raise ValueError(
                    f"Agent '{agent}' in {self.env_name} does not expose a discrete action space"
                )

            traits = agent_traits.get(agent, None)
            silent = bool(getattr(traits, "silent", False)) if traits else False
            comm_dim = world_dim_c if (world_dim_c > 0 and not silent) else 0
            if comm_dim > 0:
                move_dim = max(1, action_dim // comm_dim)
            else:
                move_dim = action_dim
            layout = AgentActionLayout(move_dim=move_dim, comm_dim=comm_dim)
            if layout.action_space_n != action_dim:
                raise ValueError(
                    f"Action space mismatch for agent '{agent}': env={action_dim}, layout={layout.action_space_n}"
                )
            self.action_layouts[agent] = layout

            model = PolicyNetwork(obs_dim, move_dim, comm_dim, self.hidden_sizes).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            self.models[agent] = model
            self.optimizers[agent] = optimizer
        env.close()

    def _sample_policy_action(
        self, agent: str, obs_tensor: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        layout = self.action_layouts[agent]
        output = self.models[agent](obs_tensor)

        log_prob = torch.zeros((), dtype=torch.float32, device=self.device)
        entropy = torch.zeros((), dtype=torch.float32, device=self.device)

        move_action = output.move_dist.sample()
        log_prob = log_prob + output.move_dist.log_prob(move_action)
        entropy = entropy + output.move_dist.entropy()
        move_idx = int(move_action.item())

        comm_idx: Optional[int] = None
        if output.comm_dist is not None:
            comm_action = output.comm_dist.sample()
            log_prob = log_prob + output.comm_dist.log_prob(comm_action)
            entropy = entropy + output.comm_dist.entropy()
            comm_idx = int(comm_action.item())

        action_idx = layout.flatten(move_idx, comm_idx)
        return action_idx, log_prob, entropy

    def _greedy_policy_action(self, agent: str, obs_tensor: torch.Tensor) -> int:
        layout = self.action_layouts[agent]
        output = self.models[agent](obs_tensor)
        move_idx = int(output.move_dist.probs.argmax().item())
        comm_idx = (
            int(output.comm_dist.probs.argmax().item())
            if output.comm_dist is not None
            else None
        )
        return layout.flatten(move_idx, comm_idx)

    def train(
        self,
        *,
        episodes: int = 200,
        seed: Optional[int] = None,
        log_interval: int = 10,
        checkpoint_dir: Optional[Path] = None,
        episode_callback: Optional[Callable[[EpisodeStats], None]] = None,
    ) -> List[EpisodeStats]:
        history: List[EpisodeStats] = []
        rng = np.random.default_rng(seed)
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_path:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for episode in range(1, episodes + 1):
            env = build_parallel_mpe_env(self.env_name)
            ep_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None
            obs, _ = env.reset(seed=ep_seed)

            trajectories = self._empty_trajectory_buffer()
            steps = 0

            while env.agents:
                steps += 1
                actions = {}
                for agent in env.agents:
                    agent_obs = obs[agent]
                    obs_tensor = torch.as_tensor(
                        agent_obs, dtype=torch.float32, device=self.device
                    )
                    action_idx, log_prob, entropy = self._sample_policy_action(
                        agent, obs_tensor
                    )
                    actions[agent] = action_idx
                    trajectories[agent]["log_probs"].append(log_prob)
                    trajectories[agent]["entropies"].append(entropy)

                obs, rewards, terminations, truncations, _ = env.step(actions)

                for agent in actions:
                    reward_value = float(rewards.get(agent, 0.0))
                    trajectories[agent]["rewards"].append(reward_value)
                    if terminations.get(agent, False) or truncations.get(agent, False):
                        trajectories[agent]["dones"].append(True)
                    else:
                        trajectories[agent]["dones"].append(False)

                if steps > self.spec.max_cycles * len(self.possible_agents):
                    break

            env.close()

            stats = self._update_policies(trajectories, episode, steps)
            history.append(stats)
            if episode_callback:
                episode_callback(stats)

            if log_interval and episode % log_interval == 0:
                mean_return = stats.mean_return
                print(
                    f"[{self.env_name}] Episode {episode}/{episodes} - "
                    f"mean return: {mean_return:.3f}"
                )
                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path, episode)

        return history

    def _empty_trajectory_buffer(self) -> Dict[str, Dict[str, List]]:
        return {
            agent: {"log_probs": [], "rewards": [], "dones": [], "entropies": []}
            for agent in self.possible_agents
        }

    def _update_policies(
        self,
        trajectories: Mapping[str, Dict[str, List]],
        episode: int,
        steps: int,
    ) -> EpisodeStats:
        per_agent_returns: Dict[str, float] = {}

        for agent, data in trajectories.items():
            log_probs: List[torch.Tensor] = data["log_probs"]
            rewards: List[float] = data["rewards"]
            if not log_probs:
                per_agent_returns[agent] = 0.0
                continue

            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + self.gamma * running
                returns.insert(0, running)

            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            baseline = self.running_baselines.get(agent, 0.0)
            advantages = returns_tensor - baseline
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            log_prob_tensor = torch.stack(log_probs)
            loss = -(log_prob_tensor * advantages).sum()

            entropies: List[torch.Tensor] = data["entropies"]
            if entropies and self.entropy_coef:
                entropy_tensor = torch.stack(entropies)
                loss -= self.entropy_coef * entropy_tensor.sum()

            self.optimizers[agent].zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.models[agent].parameters(), self.grad_clip)
            self.optimizers[agent].step()

            per_agent_returns[agent] = float(np.sum(rewards))

            if self.baseline_momentum is not None:
                avg_return = float(returns_tensor.mean().item())
                prev = self.running_baselines.get(agent, 0.0)
                momentum = float(np.clip(self.baseline_momentum, 0.0, 0.9999))
                self.running_baselines[agent] = momentum * prev + (1 - momentum) * avg_return

        mean_return = float(np.mean(list(per_agent_returns.values()) or [0.0]))
        return EpisodeStats(
            episode=episode,
            mean_return=mean_return,
            per_agent_return=per_agent_returns,
            steps=steps,
        )

    def _save_checkpoint(self, checkpoint_dir: Path, episode: int) -> None:
        for agent, model in self.models.items():
            out_path = checkpoint_dir / f"{self.env_name}_{agent}_ep{episode}.pt"
            torch.save(model.state_dict(), out_path)


def _resolve_device(preferred: DeviceLike) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if _cuda_is_usable():
            return torch.device("cuda")
        print("CUDA requested but unusable; defaulting to CPU.")
        return torch.device("cpu")

    # Auto
    target = "cuda" if _cuda_is_usable() else "cpu"
    return torch.device(target)


def _cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError:
        return False
    return True
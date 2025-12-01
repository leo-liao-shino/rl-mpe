"""Training utilities for PettingZoo MPE presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env_factory import PRESET_SPECS, MPEEnvironmentSpec, build_parallel_mpe_env

DeviceLike = Optional[str]


class PolicyNetwork(nn.Module):
    """Simple MLP policy producing categorical action distributions."""

    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.model(x)
        return Categorical(logits=logits)


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
        self._initialize_models()

    def load_checkpoints(
        self, checkpoint_dir: Path, *, strict: bool = True
    ) -> Dict[str, Path]:
        """Load the most recent checkpoint per agent from a directory.

        Args:
            checkpoint_dir: Directory containing files saved via `_save_checkpoint`.
            strict: When True, raise `FileNotFoundError` if any agent is missing.

        Returns:
            Mapping from agent name to the checkpoint path that was loaded.
        """

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
            self.models[agent].load_state_dict(state)
            loaded[agent] = best_path

        return loaded

    def _initialize_models(self) -> None:
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
            model = PolicyNetwork(obs_dim, action_dim, self.hidden_sizes).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            self.models[agent] = model
            self.optimizers[agent] = optimizer
        env.close()

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
                    obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device)
                    dist = self.models[agent](obs_tensor)
                    action = dist.sample()
                    actions[agent] = int(action.item())
                    trajectories[agent]["log_probs"].append(dist.log_prob(action))
                    trajectories[agent]["entropies"].append(dist.entropy())

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
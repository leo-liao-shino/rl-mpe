"""Supervised warm-start trainer for communicative MPE agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .supervised_dataset import AgentDataset
from .trainers import DeviceLike, IndependentPolicyTrainer


@dataclass(frozen=True)
class SpeakerPretrainConfig:
    """Hyper-parameters controlling supervised warm-start training."""

    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    shuffle: bool = True
    patience: Optional[int] = None
    min_improvement: float = 1e-4
    train_body: bool = True
    train_move_head: bool = False
    train_comm_heads: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class SpeakerPretrainStats:
    agent: str
    epochs_completed: int
    loss_history: List[float]
    best_loss: float
    accuracy: float
    samples: int


class _AgentTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: AgentDataset) -> None:
        observations = np.asarray(data.observations, dtype=np.float32)
        labels = np.asarray(data.labels, dtype=np.int64)
        if observations.shape[0] != labels.shape[0]:
            raise ValueError(
                "Observation and label counts do not match for supervised dataset"
            )
        self._observations = torch.from_numpy(observations)
        self._labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._observations[idx], self._labels[idx]


def load_npz_speaker_dataset(
    npz_path: Path, *, agents: Optional[Iterable[str]] = None
) -> Tuple[Dict[str, AgentDataset], Mapping[str, object]]:
    """Load speaker supervision data saved by `generate_speaker_dataset.py`."""

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file '{npz_path}' does not exist")

    selected = set(agents) if agents else None
    datasets: Dict[str, AgentDataset] = {}

    with np.load(npz_path, allow_pickle=False) as payload:
        meta_raw = payload.get("meta_json")
        if meta_raw is None:
            meta: Mapping[str, object] = {}
        else:
            meta_text = meta_raw.item() if hasattr(meta_raw, "item") else str(meta_raw)
            meta = json.loads(meta_text)

        for key in payload.files:
            if not key.endswith("_obs"):
                continue
            agent = key[: -len("_obs")]
            if selected and agent not in selected:
                continue
            label_key = f"{agent}_labels"
            if label_key not in payload:
                continue
            obs = np.asarray(payload[key], dtype=np.float32)
            labels = np.asarray(payload[label_key], dtype=np.int64)
            datasets[agent] = AgentDataset(observations=obs, labels=labels)

    if not datasets:
        raise RuntimeError(
            "No matching agents were found in the provided dataset; check the agents list."
        )

    return datasets, meta


class SpeakerPretrainer:
    """Train communication heads to predict target landmarks before RL."""

    def __init__(
        self,
        env_name: str,
        *,
        base_trainer: Optional[IndependentPolicyTrainer] = None,
        hidden_sizes: Sequence[int] = (128, 128),
        language_width: int = 128,
        language_arch: str = "simple",
        device: DeviceLike = None,
    ) -> None:
        if base_trainer is not None and base_trainer.env_name != env_name:
            raise ValueError("base_trainer.env_name does not match requested env")

        self.trainer = base_trainer or IndependentPolicyTrainer(
            env_name,
            hidden_sizes=hidden_sizes,
            language_width=language_width,
            language_arch=language_arch,
            device=device,
        )
        self.device = self.trainer.device
        self.env_name = env_name

    def train(
        self,
        datasets: Mapping[str, AgentDataset],
        *,
        config: Optional[SpeakerPretrainConfig] = None,
        agents: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, SpeakerPretrainStats]:
        cfg = config or SpeakerPretrainConfig()
        requested = set(agents) if agents else None
        rng = torch.Generator(device="cpu")
        if seed is not None:
            rng.manual_seed(int(seed))

        stats: Dict[str, SpeakerPretrainStats] = {}

        for agent_name, dataset in datasets.items():
            if requested and agent_name not in requested:
                continue
            layout = self.trainer.action_layouts.get(agent_name)
            if layout is None:
                continue
            if layout.comm_dim <= 0:
                continue
            if agent_name not in self.trainer.models:
                continue

            model = self.trainer.models[agent_name]
            trainable_params = self._collect_trainable_parameters(model, cfg)
            if not trainable_params:
                raise RuntimeError(
                    "No parameters were selected for optimization; check config." )

            optimizer = torch.optim.Adam(
                trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            loader = DataLoader(
                _AgentTensorDataset(dataset),
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                generator=rng,
            )
            loss_history: List[float] = []
            best_loss = float("inf")
            epochs_no_improve = 0
            completed_epochs = 0

            for epoch in range(cfg.epochs):
                epoch_losses: List[float] = []
                model.train()
                for obs_batch, label_batch in loader:
                    obs_batch = obs_batch.to(self.device)
                    label_batch = label_batch.to(self.device)
                    output = model(obs_batch)
                    if output.comm_dist is None:
                        raise RuntimeError(
                            "Agent does not expose a communication distribution during training"
                        )
                    logits = output.comm_dist.logits
                    loss = F.cross_entropy(logits, label_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.grad_clip is not None and cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    optimizer.step()
                    epoch_losses.append(float(loss.item()))

                completed_epochs += 1
                mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                loss_history.append(mean_loss)

                if mean_loss + cfg.min_improvement < best_loss:
                    best_loss = mean_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if cfg.verbose:
                    print(
                        f"[{self.env_name}][{agent_name}] Epoch {epoch + 1}/{cfg.epochs} "
                        f"loss={mean_loss:.4f}"
                    )

                if cfg.patience and epochs_no_improve >= cfg.patience:
                    if cfg.verbose:
                        print(
                            f"[{self.env_name}][{agent_name}] Early stopping after {completed_epochs} epochs"
                        )
                    break

            accuracy = self._evaluate_accuracy(model, dataset, cfg.batch_size)
            stats[agent_name] = SpeakerPretrainStats(
                agent=agent_name,
                epochs_completed=completed_epochs,
                loss_history=loss_history,
                best_loss=best_loss,
                accuracy=accuracy,
                samples=len(dataset.labels),
            )

        if not stats:
            raise RuntimeError(
                "No agents were trained; ensure dataset agents match the trainer configuration."
            )

        return stats

    def save_checkpoints(self, output_dir: Path, episode: int = 0) -> None:
        """Persist current model weights using the standard checkpoint layout."""

        output_dir.mkdir(parents=True, exist_ok=True)
        for agent, model in self.trainer.models.items():
            checkpoint = output_dir / f"{self.env_name}_{agent}_ep{episode}.pt"
            torch.save(model.state_dict(), checkpoint)

    def _collect_trainable_parameters(
        self, model: torch.nn.Module, cfg: SpeakerPretrainConfig
    ) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        if cfg.train_body and hasattr(model, "body"):
            params.extend(p for p in model.body.parameters())
        if cfg.train_move_head and hasattr(model, "move_head"):
            params.extend(p for p in model.move_head.parameters())
        if cfg.train_comm_heads:
            if hasattr(model, "comm_encoder") and model.comm_encoder is not None:
                params.extend(p for p in model.comm_encoder.parameters())
            if hasattr(model, "comm_decoder") and model.comm_decoder is not None:
                params.extend(p for p in model.comm_decoder.parameters())
            if hasattr(model, "comm_head") and model.comm_head is not None:
                params.extend(p for p in model.comm_head.parameters())
        # Remove duplicates while preserving order
        seen: set[int] = set()
        unique_params: List[torch.nn.Parameter] = []
        for param in params:
            if id(param) in seen:
                continue
            seen.add(id(param))
            unique_params.append(param)
        return unique_params

    def _evaluate_accuracy(
        self, model: torch.nn.Module, dataset: AgentDataset, batch_size: int
    ) -> float:
        loader = DataLoader(
            _AgentTensorDataset(dataset),
            batch_size=batch_size,
            shuffle=False,
        )
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for obs_batch, label_batch in loader:
                obs_batch = obs_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                output = model(obs_batch)
                if output.comm_dist is None:
                    continue
                logits = output.comm_dist.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += int((predictions == label_batch).sum().item())
                total += int(label_batch.numel())
        return float(correct / total) if total else 0.0
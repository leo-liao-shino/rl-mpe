#!/usr/bin/env python3
"""Analyze and visualize learned communication in trained MPE agents.

This script provides tools to interpret the communication layer:
1. Message usage statistics (which messages are used, entropy)
2. Context-message mapping (what situations trigger what messages)
3. Latent space visualization (t-SNE/UMAP of comm_encoder outputs)
4. Message-outcome correlation (do certain messages lead to success?)
5. Intervention analysis (what happens if we scramble/silence messages?)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional imports for dimensionality reduction
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


@dataclass
class MessageRecord:
    """Record of a single message sent during an episode."""
    episode: int
    step: int
    agent: str
    observation: np.ndarray
    encoder_output: np.ndarray  # The 128-dim latent
    message_logits: np.ndarray
    message_id: int
    message_prob: float


@dataclass 
class EpisodeRecord:
    """Summary of an episode for correlation analysis."""
    episode: int
    total_return: float
    messages_sent: List[int]
    success: bool  # Task-specific success metric


def load_trained_models(
    checkpoint_dir: Path,
    env_name: str,
    device: str = "cpu",
    *,
    hidden_sizes: Optional[List[int]] = None,
    algorithm: str = "actor-critic",
    language_arch: str = "simple",
    flat_action_space: bool = False,
):
    """Load trained models from checkpoint directory with configurable architecture."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from mpe_env_setup.trainers import IndependentPolicyTrainer, ActorCriticTrainer

    hidden = hidden_sizes or [256, 256]

    if algorithm == "actor-critic":
        trainer = ActorCriticTrainer(
            env_name=env_name,
            hidden_sizes=hidden,
            device=device,
            language_arch=language_arch,
            flat_action_space=flat_action_space,
        )
    else:
        trainer = IndependentPolicyTrainer(
            env_name=env_name,
            hidden_sizes=hidden,
            device=device,
            language_arch=language_arch,
            flat_action_space=flat_action_space,
        )

    loaded = trainer.load_checkpoints(checkpoint_dir, strict=False)
    if loaded:
        print(f"Loaded models for: {list(loaded.keys())}")
    else:
        print("Warning: No checkpoints loaded")

    return trainer


def collect_communication_data(
    trainer,
    env_name: str,
    num_episodes: int = 100,
    device: str = "cpu",
) -> Tuple[List[MessageRecord], List[EpisodeRecord]]:
    """Run episodes and collect communication data for analysis."""
    from mpe_env_setup.env_factory import build_parallel_mpe_env
    
    env = build_parallel_mpe_env(env_name)
    message_records: List[MessageRecord] = []
    episode_records: List[EpisodeRecord] = []
    
    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        episode_messages = []
        episode_returns = defaultdict(float)
        step = 0
        
        while env.agents:
            actions = {}
            
            for agent in env.agents:
                if agent not in obs_dict:
                    continue
                    
                obs = obs_dict[agent]
                model = trainer.models[agent]
                model.eval()
                
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # Get hidden representation
                    hidden = model.body(obs_tensor)
                    
                    # Get movement action
                    move_logits = model.move_head(hidden)
                    move_probs = torch.softmax(move_logits, dim=-1)
                    move_action = torch.argmax(move_probs, dim=-1).item()
                    
                    # Get communication if agent can communicate
                    comm_action = 0
                    if model.comm_encoder is not None:
                        encoder_out = model.comm_encoder(hidden)
                        
                        if model.comm_head is not None:
                            comm_logits = model.comm_head(encoder_out)
                        elif model.comm_decoder is not None:
                            comm_logits = model.comm_decoder(encoder_out)
                        else:
                            comm_logits = None
                        
                        if comm_logits is not None:
                            comm_probs = torch.softmax(comm_logits, dim=-1)
                            comm_action = torch.argmax(comm_probs, dim=-1).item()
                            
                            # Record message data
                            record = MessageRecord(
                                episode=ep,
                                step=step,
                                agent=agent,
                                observation=obs.copy(),
                                encoder_output=encoder_out.squeeze(0).cpu().numpy(),
                                message_logits=comm_logits.squeeze(0).cpu().numpy(),
                                message_id=comm_action,
                                message_prob=comm_probs[0, comm_action].item(),
                            )
                            message_records.append(record)
                            episode_messages.append(comm_action)
                    
                    # Compute flat action
                    if model.comm_dim > 0:
                        flat_action = move_action * model.comm_dim + comm_action
                    else:
                        flat_action = move_action
                    
                    actions[agent] = flat_action
            
            obs_dict, rewards, terminations, truncations, _ = env.step(actions)
            
            for agent, reward in rewards.items():
                episode_returns[agent] += reward
            
            step += 1
            
            if all(terminations.values()) or all(truncations.values()):
                break
        
        # Record episode summary
        total_return = sum(episode_returns.values())
        # For simple_reference: success if return > -50 (close to target)
        success = total_return > -50
        
        episode_records.append(EpisodeRecord(
            episode=ep,
            total_return=total_return,
            messages_sent=episode_messages,
            success=success,
        ))
        
        if (ep + 1) % 20 == 0:
            print(f"Collected {ep + 1}/{num_episodes} episodes")
    
    env.close()
    return message_records, episode_records


def analyze_message_usage(message_records: List[MessageRecord]) -> Dict:
    """Analyze which messages are used and how often."""
    message_counts = defaultdict(int)
    agent_message_counts = defaultdict(lambda: defaultdict(int))
    
    for record in message_records:
        message_counts[record.message_id] += 1
        agent_message_counts[record.agent][record.message_id] += 1
    
    total = sum(message_counts.values())
    message_probs = {k: v / total for k, v in message_counts.items()}
    
    # Calculate entropy
    probs = np.array(list(message_probs.values()))
    probs = probs[probs > 0]  # Remove zeros for log
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(message_counts)) if message_counts else 0
    
    return {
        "message_counts": dict(message_counts),
        "message_probs": message_probs,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
        "num_unique_messages": len(message_counts),
        "per_agent_counts": {k: dict(v) for k, v in agent_message_counts.items()},
    }


def analyze_message_outcome_correlation(
    message_records: List[MessageRecord],
    episode_records: List[EpisodeRecord],
) -> Dict:
    """Analyze correlation between messages and episode outcomes."""
    # Group messages by episode
    episode_messages = defaultdict(list)
    for record in message_records:
        episode_messages[record.episode].append(record.message_id)
    
    # Correlate with outcomes
    message_success_rate = defaultdict(lambda: {"success": 0, "total": 0})
    message_return_sum = defaultdict(lambda: {"sum": 0, "count": 0})
    
    for ep_record in episode_records:
        msgs = set(episode_messages[ep_record.episode])
        for msg in msgs:
            message_success_rate[msg]["total"] += 1
            if ep_record.success:
                message_success_rate[msg]["success"] += 1
            message_return_sum[msg]["sum"] += ep_record.total_return
            message_return_sum[msg]["count"] += 1
    
    success_rates = {
        msg: data["success"] / data["total"] if data["total"] > 0 else 0
        for msg, data in message_success_rate.items()
    }
    
    avg_returns = {
        msg: data["sum"] / data["count"] if data["count"] > 0 else 0
        for msg, data in message_return_sum.items()
    }
    
    return {
        "message_success_rates": success_rates,
        "message_avg_returns": avg_returns,
    }


def visualize_latent_space(
    message_records: List[MessageRecord],
    output_path: Path,
    method: str = "tsne",
    color_by: str = "message",
) -> None:
    """Visualize the comm_encoder latent space using t-SNE or UMAP."""
    if not message_records:
        print("No message records to visualize")
        return
    
    # Extract latent vectors
    latents = np.array([r.encoder_output for r in message_records])
    messages = np.array([r.message_id for r in message_records])
    agents = np.array([r.agent for r in message_records])
    
    print(f"Visualizing {len(latents)} latent vectors with {method.upper()}")
    
    # Dimensionality reduction
    if method == "tsne":
        if not HAS_TSNE:
            print("scikit-learn not installed, skipping t-SNE")
            return
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
        embeddings = reducer.fit_transform(latents)
    elif method == "umap":
        if not HAS_UMAP:
            print("umap-learn not installed, skipping UMAP")
            return
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(latents)
    else:
        # Just use first 2 dimensions
        embeddings = latents[:, :2]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color by message
    scatter1 = axes[0].scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=messages, cmap="tab10", alpha=0.6, s=20
    )
    axes[0].set_title(f"Latent Space ({method.upper()}) - Colored by Message")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    plt.colorbar(scatter1, ax=axes[0], label="Message ID")
    
    # Color by agent
    agent_ids = {a: i for i, a in enumerate(set(agents))}
    agent_colors = [agent_ids[a] for a in agents]
    scatter2 = axes[1].scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=agent_colors, cmap="Set1", alpha=0.6, s=20
    )
    axes[1].set_title(f"Latent Space ({method.upper()}) - Colored by Agent")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    
    # Add legend for agents
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=plt.cm.Set1(i/len(agent_ids)), markersize=10, label=a)
               for a, i in agent_ids.items()]
    axes[1].legend(handles=handles, title="Agent")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved latent space visualization to {output_path}")
    plt.close()


def visualize_message_usage(usage_stats: Dict, output_path: Path) -> None:
    """Create bar chart of message usage."""
    counts = usage_stats["message_counts"]
    if not counts:
        print("No messages to visualize")
        return
    
    messages = sorted(counts.keys())
    values = [counts[m] for m in messages]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(messages, values, color="steelblue", edgecolor="black")
    
    ax.set_xlabel("Message ID")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Message Usage Distribution\n"
                 f"Entropy: {usage_stats['entropy']:.2f} / {usage_stats['max_entropy']:.2f} "
                 f"({usage_stats['entropy_ratio']*100:.1f}%)")
    ax.set_xticks(messages)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved message usage chart to {output_path}")
    plt.close()


def visualize_message_outcomes(correlation_stats: Dict, output_path: Path) -> None:
    """Visualize message-outcome correlations."""
    returns = correlation_stats["message_avg_returns"]
    if not returns:
        print("No correlation data to visualize")
        return
    
    messages = sorted(returns.keys())
    values = [returns[m] for m in messages]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if v > np.mean(values) else "red" for v in values]
    bars = ax.bar(messages, values, color=colors, edgecolor="black", alpha=0.7)
    
    ax.axhline(y=np.mean(values), color="black", linestyle="--", label=f"Mean: {np.mean(values):.1f}")
    ax.set_xlabel("Message ID")
    ax.set_ylabel("Average Episode Return")
    ax.set_title("Message → Episode Return Correlation")
    ax.set_xticks(messages)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved message-outcome chart to {output_path}")
    plt.close()


def run_intervention_analysis(
    trainer,
    env_name: str,
    num_episodes: int = 50,
    device: str = "cpu",
) -> Dict:
    """Test what happens when we intervene on communication."""
    from mpe_env_setup.env_factory import build_parallel_mpe_env
    
    results = {}
    
    # Baseline (normal communication)
    print("Running baseline (normal comm)...")
    _, baseline_episodes = collect_communication_data(trainer, env_name, num_episodes, device)
    baseline_returns = [e.total_return for e in baseline_episodes]
    results["baseline"] = {
        "mean_return": np.mean(baseline_returns),
        "std_return": np.std(baseline_returns),
    }
    
    # Silenced communication (set all comm to 0)
    print("Running silenced communication...")
    # We'd need to modify the model temporarily - skip for now
    results["silenced"] = {"note": "Not implemented - requires model modification"}
    
    # Random communication
    print("Running random communication...")
    results["random"] = {"note": "Not implemented - requires model modification"}
    
    return results


def analyze_simple_reference_grounding(
    message_records: List[MessageRecord],
    output_path: Optional[Path] = None,
) -> Dict:
    """Analyze whether messages in simple_reference correlate with target landmark.
    
    In simple_reference_v3:
    - agent_0 is the "speaker" who sees the target color
    - agent_1 is the "listener" who must navigate to the target
    
    Observation structure for agent_0 (speaker):
    - [0:4]: own velocity (2) + own position (2) = 4 dims
    - [4:10]: landmark positions relative (3 landmarks * 2) = 6 dims
    - [10:13]: target landmark color (RGB, 3 dims)
    - [13:21]: communication from other agents = 8 dims (padded)
    
    We want to see if messages correlate with the target color (obs[10:13]).
    """
    # Filter to only speaker (agent_0) messages
    speaker_records = [r for r in message_records if r.agent == "agent_0"]
    
    if not speaker_records:
        print("No speaker (agent_0) messages found")
        return {}
    
    # Extract target colors (indices 10:13 are the RGB of target landmark)
    # Note: The exact indices may vary - let's check observation dimensions
    sample_obs = speaker_records[0].observation
    obs_dim = len(sample_obs)
    print(f"Speaker observation dimension: {obs_dim}")
    
    # For simple_reference_v3, target info should be around index 10-13
    # Let's try to identify clusters by the target landmark
    target_colors = np.array([r.observation[10:13] for r in speaker_records])
    messages = np.array([r.message_id for r in speaker_records])
    
    # Quantize target colors to identify unique targets
    # (RGB values normalized, so round to nearest 0.1)
    quantized_colors = np.round(target_colors, 1)
    unique_targets = np.unique(quantized_colors, axis=0)
    
    print(f"Found {len(unique_targets)} unique target colors:")
    for i, color in enumerate(unique_targets):
        print(f"  Target {i}: RGB = {color}")
    
    # Map each record to its target ID
    target_ids = []
    for color in quantized_colors:
        for i, unique in enumerate(unique_targets):
            if np.allclose(color, unique):
                target_ids.append(i)
                break
    target_ids = np.array(target_ids)
    
    # Build message distribution per target
    target_message_dist = {}
    for tid in range(len(unique_targets)):
        mask = target_ids == tid
        msgs = messages[mask]
        counts = defaultdict(int)
        for m in msgs:
            counts[m] += 1
        total = len(msgs)
        dist = {m: c / total for m, c in counts.items()}
        target_message_dist[f"target_{tid}"] = dict(dist)
    
    # Calculate mutual information between target and message
    # Higher MI = messages encode target information
    from collections import Counter
    
    # Joint distribution P(target, message)
    joint_counts = Counter(zip(target_ids, messages))
    total = len(target_ids)
    
    # Marginal P(target)
    target_counts = Counter(target_ids)
    p_target = {t: c/total for t, c in target_counts.items()}
    
    # Marginal P(message)
    msg_counts = Counter(messages)
    p_message = {m: c/total for m, c in msg_counts.items()}
    
    # Calculate MI
    mi = 0.0
    for (t, m), count in joint_counts.items():
        p_tm = count / total
        if p_tm > 0:
            mi += p_tm * np.log2(p_tm / (p_target[t] * p_message[m]))
    
    # Max possible MI is min(H(target), H(message))
    h_target = -sum(p * np.log2(p) for p in p_target.values() if p > 0)
    h_message = -sum(p * np.log2(p) for p in p_message.values() if p > 0)
    max_mi = min(h_target, h_message)
    
    results = {
        "num_unique_targets": len(unique_targets),
        "unique_target_colors": unique_targets.tolist(),
        "target_message_distributions": target_message_dist,
        "mutual_information": mi,
        "max_mutual_information": max_mi,
        "mi_ratio": mi / max_mi if max_mi > 0 else 0,
        "h_target": h_target,
        "h_message": h_message,
    }
    
    print(f"\n📊 GROUNDING ANALYSIS:")
    print(f"  Mutual Information: {mi:.3f} bits")
    print(f"  Max possible MI: {max_mi:.3f} bits")
    print(f"  MI ratio: {results['mi_ratio']*100:.1f}%")
    print(f"  (Higher MI = messages encode more about target)")
    
    # Visualize target-message correspondence
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap of P(message | target)
        num_targets = len(unique_targets)
        num_messages = max(messages) + 1
        heatmap = np.zeros((num_targets, num_messages))
        
        for tid in range(num_targets):
            mask = target_ids == tid
            msgs = messages[mask]
            for m in msgs:
                heatmap[tid, m] += 1
            if heatmap[tid].sum() > 0:
                heatmap[tid] /= heatmap[tid].sum()  # Normalize to get P(msg|target)
        
        im = axes[0].imshow(heatmap, cmap="YlOrRd", aspect="auto")
        axes[0].set_xlabel("Message ID")
        axes[0].set_ylabel("Target ID")
        axes[0].set_title(f"P(Message | Target)\nMI = {mi:.3f} bits ({results['mi_ratio']*100:.1f}% of max)")
        axes[0].set_xticks(range(num_messages))
        axes[0].set_yticks(range(num_targets))
        plt.colorbar(im, ax=axes[0], label="Probability")
        
        # Bar chart of message entropy per target
        target_entropies = []
        for tid in range(num_targets):
            p = heatmap[tid]
            p = p[p > 0]
            h = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
            target_entropies.append(h)
        
        axes[1].bar(range(num_targets), target_entropies, color="steelblue")
        axes[1].axhline(y=np.mean(target_entropies), color="red", linestyle="--", 
                       label=f"Mean: {np.mean(target_entropies):.2f}")
        axes[1].set_xlabel("Target ID")
        axes[1].set_ylabel("Message Entropy (bits)")
        axes[1].set_title("Message Entropy per Target\n(Lower = more consistent mapping)")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved grounding analysis to {output_path}")
        plt.close()
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                       help="Directory containing trained model checkpoints")
    parser.add_argument("--env", default="simple_reference_v3",
                       help="Environment name")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--hidden", type=int, nargs="*", default=None,
                       help="Hidden layer sizes to match the trained model (default adapts to env)")
    parser.add_argument("--algorithm", choices=["reinforce", "actor-critic"], default="actor-critic",
                       help="Algorithm used during training so we can construct the right architecture")
    parser.add_argument("--language-arch", choices=["simple", "encdec"], default="simple",
                       help="Communication head architecture used during training")
    parser.add_argument("--flat-action-space", action="store_true",
                       help="Use flat action head (no separate comm channel) if the model was trained that way")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/comm_analysis"),
                       help="Output directory for visualizations")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne",
                       help="Dimensionality reduction method")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.hidden:
        hidden = args.hidden
    elif "simple_world_comm_v3" in args.env:
        hidden = [512, 256]
    else:
        hidden = [256, 256]
    
    print(f"Loading models from {args.checkpoint_dir}")
    trainer = load_trained_models(
        args.checkpoint_dir,
        args.env,
        args.device,
        hidden_sizes=hidden,
        algorithm=args.algorithm,
        language_arch=args.language_arch,
        flat_action_space=args.flat_action_space,
    )
    
    print(f"\nCollecting communication data over {args.episodes} episodes...")
    message_records, episode_records = collect_communication_data(
        trainer, args.env, args.episodes, args.device
    )
    
    print(f"\nCollected {len(message_records)} messages from {len(episode_records)} episodes")
    
    # Analyze message usage
    print("\n" + "="*60)
    print("MESSAGE USAGE ANALYSIS")
    print("="*60)
    usage_stats = analyze_message_usage(message_records)
    print(f"Unique messages used: {usage_stats['num_unique_messages']}")
    print(f"Message entropy: {usage_stats['entropy']:.2f} / {usage_stats['max_entropy']:.2f} "
          f"({usage_stats['entropy_ratio']*100:.1f}% of max)")
    print(f"Message distribution: {usage_stats['message_counts']}")
    
    # Analyze message-outcome correlation
    print("\n" + "="*60)
    print("MESSAGE-OUTCOME CORRELATION")
    print("="*60)
    correlation_stats = analyze_message_outcome_correlation(message_records, episode_records)
    print("Average return per message:")
    for msg, ret in sorted(correlation_stats["message_avg_returns"].items()):
        print(f"  Message {msg}: {ret:.1f}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualize_message_usage(
        usage_stats, 
        args.output_dir / f"{args.env}_message_usage.png"
    )
    
    visualize_message_outcomes(
        correlation_stats,
        args.output_dir / f"{args.env}_message_outcomes.png"
    )
    
    visualize_latent_space(
        message_records,
        args.output_dir / f"{args.env}_latent_space_{args.method}.png",
        method=args.method,
    )
    
    # Task-specific grounding analysis for simple_reference
    grounding_stats = None
    if "simple_reference" in args.env:
        print("\n" + "="*60)
        print("GROUNDING ANALYSIS (simple_reference)")
        print("="*60)
        grounding_stats = analyze_simple_reference_grounding(
            message_records,
            args.output_dir / f"{args.env}_grounding.png"
        )
    
    # Save raw data
    summary = {
        "env": args.env,
        "episodes": args.episodes,
        "total_messages": len(message_records),
        "usage_stats": {
            k: v for k, v in usage_stats.items() 
            if k not in ["per_agent_counts"]  # Skip nested dicts for JSON
        },
        "correlation_stats": correlation_stats,
    }
    
    # Add grounding stats if available
    if grounding_stats:
        summary["grounding_stats"] = {
            k: v for k, v in grounding_stats.items()
            if not isinstance(v, (np.ndarray, list)) or k == "unique_target_colors"
        }
    
    summary_path = args.output_dir / f"{args.env}_analysis_summary.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {str(k): convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    summary = convert_to_native(summary)
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis summary to {summary_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

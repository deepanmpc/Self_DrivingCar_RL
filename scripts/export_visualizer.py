"""Export the configured actor and critic to visualizer/model JSON."""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.visualization import (  # noqa: E402
    build_networks,
    load_checkpoint,
    load_yaml,
    serialize_model,
    write_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "network.yaml",
        help="Network architecture configuration.",
    )
    parser.add_argument(
        "--observation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "observation.yaml",
        help="Observation feature configuration.",
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment.yaml",
        help="Environment configuration used for physical display ranges.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional actor/critic checkpoint to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "visualizer" / "models" / "latest.json",
        help="Output model JSON path.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable model label shown in the visualizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when generating untrained weights.",
    )
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Allow torch.load without weights_only for trusted checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network_config = load_yaml(args.config)
    observation_config = load_yaml(args.observation_config)
    environment_config = load_yaml(args.environment_config)

    torch.manual_seed(args.seed)
    actor, critic = build_networks(network_config)

    label = args.label
    if args.checkpoint is not None:
        loaded = load_checkpoint(
            actor,
            critic,
            args.checkpoint,
            allow_unsafe_load=args.allow_unsafe_load,
        )
        if label is None:
            label = args.checkpoint.stem
        print(f"Loaded checkpoint weights for: {', '.join(loaded)}")
    elif label is None:
        label = "untrained"

    actor.eval()
    if critic is not None:
        critic.eval()

    model = serialize_model(
        actor,
        critic,
        observation_config,
        environment_config,
        label=label,
        checkpoint=str(args.checkpoint) if args.checkpoint is not None else None,
        seed=args.seed if args.checkpoint is None else None,
    )
    write_model(model, args.output)

    actor_layers = len(actor.mean_network.hidden_dims)
    summary = f"Exported {label} ({model['metadata']['parameters']:,} parameters)"
    summary += f" with {actor_layers} hidden layers"
    if critic is not None:
        summary += f" and {len(critic.value_network.hidden_dims)} critic hidden layers"
    print(summary)
    print(f"  model: {args.output}")
    print(f"  max PyTorch/serialized error: {model['validation']['maxTorchError']:.3g}")


if __name__ == "__main__":
    main()

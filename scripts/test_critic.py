"""Test Critic value predictions, architecture, and gradient flow."""

import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.networks import Actor, Critic


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "network.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    critic = Critic.from_config(config)
    observations = torch.randn(32, 8)
    values = critic(observations)

    assert values.shape == (32,)
    assert torch.isfinite(values).all()
    assert critic.value_network.hidden_dims == (256, 256, 256, 128, 128, 128, 64)

    value_loss = torch.nn.functional.mse_loss(values, torch.randn_like(values))
    value_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in critic.parameters()
    )

    actor = Actor.from_config(config)
    actor_parameters = dict(actor.named_parameters())
    critic_parameters = dict(critic.named_parameters())
    assert actor_parameters.keys() != critic_parameters.keys()
    assert not any(
        actor_parameter is critic_parameter
        for actor_parameter in actor.parameters()
        for critic_parameter in critic.parameters()
    ), "Actor and Critic must not share parameter objects."

    try:
        critic(torch.randn(4, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("Critic accepted an observation with the wrong feature dimension.")

    single_value = critic(torch.randn(8))
    assert single_value.ndim == 0
    assert torch.isfinite(single_value)

    print("Critic tests passed.")
    print(f"  stages: 1 input + {len(critic.value_network.hidden_dims)} hidden + 1 output")
    print(f"  batch value shape: {tuple(values.shape)}")
    print(f"  MSE value loss: {value_loss.item():.4f}")


if __name__ == "__main__":
    main()

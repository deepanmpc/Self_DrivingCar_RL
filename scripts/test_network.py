"""Test Actor and Critic architecture, outputs, and gradient flow."""

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
    actor = Actor.from_config(config)

    observations = torch.randn(32, 8)
    means = actor(observations)
    assert means.shape == (32, 2)

    actions, log_probabilities = actor.sample_action(observations)
    assert actions.shape == (32, 2)
    assert log_probabilities.shape == (32,)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    assert torch.isfinite(log_probabilities).all()

    reevaluated_log_probabilities = actor.evaluate_actions(observations, actions.detach())
    assert reevaluated_log_probabilities.shape == (32,)
    assert torch.allclose(reevaluated_log_probabilities, log_probabilities.detach(), atol=1e-5)

    deterministic_actions, deterministic_log_probabilities = actor.sample_action(
        observations, deterministic=True
    )
    assert torch.allclose(deterministic_actions, torch.tanh(means))
    assert deterministic_log_probabilities.shape == (32,)

    loss = -log_probabilities.mean() + means.square().mean()
    loss.backward()
    assert actor.log_std.grad is not None
    assert torch.isfinite(actor.log_std.grad).all()
    trainable_parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable_parameters)

    try:
        actor(torch.randn(4, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("Actor accepted an observation with the wrong feature dimension.")

    critic = Critic.from_config(config)
    values = critic(observations)
    assert values.shape == (32,)
    assert torch.isfinite(values).all()

    value_loss = values.square().mean()
    value_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in critic.parameters()
    )

    try:
        critic(torch.randn(4, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("Critic accepted an observation with the wrong feature dimension.")

    print("Actor and Critic network tests passed.")
    print(f"  actor stages: 1 input + {len(actor.mean_network.hidden_dims)} hidden + 1 output")
    print(f"  action range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    print(f"  mean log probability: {log_probabilities.mean().item():.4f}")
    print(f"  critic stages: 1 input + {len(critic.value_network.hidden_dims)} hidden + 1 output")
    print(f"  value range: [{values.min().item():.4f}, {values.max().item():.4f}]")


if __name__ == "__main__":
    main()

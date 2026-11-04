"""Test policy selection, truncation bootstrapping, and live rollout collection."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent
from memory.rollout_buffer import RolloutBuffer
from simulator.environment import DrivingEnvironment
from training.collector import RolloutCollector


class BoundaryEnvironment:
    """Tiny deterministic environment to verify collector boundary behavior."""

    def __init__(self):
        self.step_index = 0
        self.reset_count = 0

    def reset(self, seed=None):
        self.reset_count += 1
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        observation = np.full(8, self.step_index, dtype=np.float32)
        if self.step_index == 1:
            return observation, 1.0, False, True, {}
        return observation, 1.0, True, False, {}


def load_agent():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "network.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Network config must be a YAML mapping: {config_path}")
    return Agent.from_config(config)


def test_boundary_bootstrapping():
    buffer = RolloutBuffer()
    zero_observation = np.zeros(8, dtype=np.float32)
    buffer.add(
        observation=zero_observation,
        action=np.zeros(2, dtype=np.float32),
        reward=1.0,
        next_observation=np.ones(8, dtype=np.float32),
        terminated=False,
        truncated=True,
        old_log_probability=-0.5,
        value=0.5,
        next_value=10.0,
    )
    buffer.add(
        observation=np.ones(8, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        reward=2.0,
        next_observation=np.full(8, 2.0, dtype=np.float32),
        terminated=True,
        truncated=False,
        old_log_probability=-0.4,
        value=1.0,
        next_value=99.0,
    )
    returns, advantages = buffer.compute_returns_and_advantages(gamma=0.9, gae_lambda=0.95)
    assert np.allclose(advantages, [9.5, 1.0])
    assert np.allclose(returns, [10.0, 2.0])


def test_auto_reset_and_terminal_value():
    agent = load_agent()
    environment = BoundaryEnvironment()
    collector = RolloutCollector(environment, agent)
    buffer = collector.collect(2)
    assert len(buffer) == 2
    assert buffer.truncated == [True, False]
    assert buffer.terminated == [False, True]
    assert buffer.dones == [True, True]
    assert buffer.episode_boundaries == [True, True]
    assert buffer.next_values[0] != 0.0
    assert buffer.next_values[1] == 0.0
    assert environment.reset_count == 3


def test_live_environment_collection():
    agent = load_agent()
    environment = DrivingEnvironment(render=False)
    try:
        collector = RolloutCollector(environment, agent)
        buffer = collector.collect(4, seed=0)
        assert len(buffer) == 4
        assert np.stack(buffer.observations).shape == (4, 8)
        assert np.stack(buffer.actions).shape == (4, 2)
        assert np.all(np.isfinite(buffer.old_log_probabilities))
        assert np.all(np.isfinite(buffer.values))
        assert np.all(np.isfinite(buffer.next_values))

        returns, advantages = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        assert returns.shape == advantages.shape == (4,)
        batch = buffer.as_tensors()
        assert batch.observations.shape == (4, 8)
        assert batch.actions.shape == (4, 2)

        evaluated_log_probabilities, entropy, values = agent.evaluate_actions(
            batch.observations, batch.actions
        )
        assert torch.allclose(
            evaluated_log_probabilities,
            batch.old_log_probabilities,
            atol=1e-5,
        )
        assert entropy.shape == values.shape == (4,)
        assert torch.isfinite(entropy).all() and torch.isfinite(values).all()
    finally:
        environment.close()


def main():
    test_boundary_bootstrapping()
    test_auto_reset_and_terminal_value()
    test_live_environment_collection()
    print("Agent and rollout collection tests passed.")


if __name__ == "__main__":
    main()

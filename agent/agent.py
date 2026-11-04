"""High-level controller that exposes policy actions to rollout collection."""

from collections.abc import Mapping
from typing import Optional

import torch

from agent.policy import ActionSelection, Policy


class Agent:
    """Select environment actions and expose the policy for PPO evaluation."""

    def __init__(self, policy: Policy):
        self.policy = policy

    @classmethod
    def from_config(cls, config: Mapping, device: Optional[torch.device] = None):
        return cls(Policy.from_config(config, device=device))

    @classmethod
    def from_yaml(cls, path, device: Optional[torch.device] = None):
        import yaml

        with open(path, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, Mapping):
            raise ValueError("Network configuration must be a YAML mapping.")
        return cls.from_config(config, device=device)

    def select_action(self, observation, deterministic: bool = False) -> ActionSelection:
        return self.policy.select_action(observation, deterministic=deterministic)

    def estimate_value(self, observation) -> float:
        return self.policy.estimate_value(observation)

    def evaluate_actions(self, observations, actions):
        return self.policy.evaluate_actions(observations, actions)

"""Coordinate actor sampling, critic values, and PPO training evaluations."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
import torch

from agent.networks import Actor, Critic


@dataclass(frozen=True)
class ActionSelection:
    """Action and policy statistics for one observation."""

    action: np.ndarray
    log_probability: float
    value: float


class Policy:
    """Inference and training interface over an Actor and an independent Critic."""

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        device: Optional[torch.device] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

    @classmethod
    def from_config(cls, config: Mapping, device: Optional[torch.device] = None):
        return cls(
            actor=Actor.from_config(config),
            critic=Critic.from_config(config),
            device=device,
        )

    def select_action(self, observation, deterministic: bool = False) -> ActionSelection:
        """Select one action and return its old-policy log probability and value."""
        observation_tensor = self._observation_tensor(observation)
        with torch.no_grad():
            action, log_probability = self.actor.sample_action(
                observation_tensor, deterministic=deterministic
            )
            value = self.critic(observation_tensor)
        return ActionSelection(
            action=action.cpu().numpy().astype(np.float32, copy=True),
            log_probability=float(log_probability.item()),
            value=float(value.item()),
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log probabilities, sampled entropy estimates, and state values."""
        observations = observations.to(device=self.device, dtype=torch.float32)
        actions = actions.to(device=self.device, dtype=torch.float32)
        log_probabilities = self.actor.evaluate_actions(observations, actions)
        entropy = self.actor.distribution(observations).entropy()
        values = self.critic(observations)
        return log_probabilities, entropy, values

    def estimate_value(self, observation) -> float:
        """Estimate V(s) without retaining a gradient graph."""
        observation_tensor = self._observation_tensor(observation)
        with torch.no_grad():
            value = self.critic(observation_tensor)
        return float(value.item())

    def _observation_tensor(self, observation) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tensor.shape != (self.actor.observation_dim,):
            raise ValueError(
                f"Expected one observation with shape ({self.actor.observation_dim},), "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

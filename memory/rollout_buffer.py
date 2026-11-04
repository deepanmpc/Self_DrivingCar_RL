"""Rollout storage and GAE computation for PPO training."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_boundaries: torch.Tensor
    old_log_probabilities: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    """Store transitions without conflating terminal states and time limits."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.terminated = []
        self.truncated = []
        self.episode_boundaries = []
        self.old_log_probabilities = []
        self.values = []
        self.next_values = []
        self._observation_shape = None
        self._action_shape = None
        self._returns = None
        self._advantages = None

    def add(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        terminated: bool,
        truncated: bool,
        old_log_probability: float,
        value: float,
        next_value: float,
    ) -> None:
        observation_array = self._vector(observation, "observation")
        action_array = self._vector(action, "action")
        next_observation_array = self._vector(next_observation, "next_observation")
        if observation_array.shape != next_observation_array.shape:
            raise ValueError("observation and next_observation must have the same shape.")
        if self._observation_shape is not None and observation_array.shape != self._observation_shape:
            raise ValueError("All observations in a rollout must have a consistent shape.")
        if self._action_shape is not None and action_array.shape != self._action_shape:
            raise ValueError("All actions in a rollout must have a consistent shape.")
        scalars = {
            "reward": reward,
            "old_log_probability": old_log_probability,
            "value": value,
            "next_value": next_value,
        }
        scalar_values = {name: self._finite_scalar(value, name) for name, value in scalars.items()}

        self._observation_shape = observation_array.shape
        self._action_shape = action_array.shape
        self.observations.append(observation_array)
        self.actions.append(action_array)
        self.rewards.append(scalar_values["reward"])
        self.next_observations.append(next_observation_array)
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.dones.append(bool(terminated or truncated))
        self.episode_boundaries.append(bool(terminated or truncated))
        self.old_log_probabilities.append(scalar_values["old_log_probability"])
        self.values.append(scalar_values["value"])
        self.next_values.append(scalar_values["next_value"])
        self._returns = None
        self._advantages = None

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE; truncated states bootstrap, but do not leak across episodes."""
        gamma = self._discount(gamma, "gamma")
        gae_lambda = self._discount(gae_lambda, "gae_lambda")
        if not self:
            raise ValueError("Cannot calculate returns for an empty rollout.")

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        terminated = np.asarray(self.terminated, dtype=np.bool_)
        episode_boundaries = np.asarray(self.episode_boundaries, dtype=np.bool_)
        advantages = np.zeros_like(rewards)
        gae = 0.0

        for index in range(len(self) - 1, -1, -1):
            bootstrap_mask = 0.0 if terminated[index] else 1.0
            continue_mask = 0.0 if episode_boundaries[index] else 1.0
            delta = rewards[index] + gamma * next_values[index] * bootstrap_mask - values[index]
            gae = delta + gamma * gae_lambda * continue_mask * gae
            advantages[index] = gae

        self._advantages = advantages
        self._returns = advantages + values
        return self._returns.copy(), self._advantages.copy()

    def as_tensors(self, device: Optional[torch.device] = None) -> RolloutBatch:
        """Convert the rollout and computed returns to tensors for PPO minibatches."""
        if not self:
            raise ValueError("Cannot convert an empty rollout to tensors.")
        if self._returns is None or self._advantages is None:
            raise RuntimeError("Compute returns and advantages before converting the rollout.")

        return RolloutBatch(
            observations=torch.as_tensor(np.stack(self.observations), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.stack(self.actions), dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            next_observations=torch.as_tensor(
                np.stack(self.next_observations), dtype=torch.float32, device=device
            ),
            dones=torch.as_tensor(self.dones, dtype=torch.bool, device=device),
            terminated=torch.as_tensor(self.terminated, dtype=torch.bool, device=device),
            truncated=torch.as_tensor(self.truncated, dtype=torch.bool, device=device),
            episode_boundaries=torch.as_tensor(
                self.episode_boundaries, dtype=torch.bool, device=device
            ),
            old_log_probabilities=torch.as_tensor(
                self.old_log_probabilities, dtype=torch.float32, device=device
            ),
            values=torch.as_tensor(self.values, dtype=torch.float32, device=device),
            next_values=torch.as_tensor(self.next_values, dtype=torch.float32, device=device),
            returns=torch.as_tensor(self._returns, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(self._advantages, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.rewards)

    def __bool__(self) -> bool:
        return bool(self.rewards)

    @staticmethod
    def _vector(value, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array.copy()

    @staticmethod
    def _finite_scalar(value, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite scalar.") from error
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite scalar.")
        return result

    @staticmethod
    def _discount(value: float, name: str) -> float:
        value = RolloutBuffer._finite_scalar(value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
        return value

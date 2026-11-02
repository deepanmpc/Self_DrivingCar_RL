"""Squashed-Gaussian Actor for continuous bounded driving actions."""

import math
from collections.abc import Mapping
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from agent.networks.layers import NeuralNetLayers


class Actor(nn.Module):
    """Map observations to tanh-bounded Gaussian actions and their log probabilities."""

    def __init__(
        self,
        observation_dim: int = 8,
        action_dim: int = 2,
        hidden_dims: Sequence[int] = (256, 256, 256, 128, 128, 128, 64),
        activation: str = "relu",
        layer_norm: bool = False,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be a positive integer.")
        if not all(math.isfinite(value) for value in (initial_log_std, min_log_std, max_log_std)):
            raise ValueError("Log standard deviation settings must be finite.")
        if min_log_std >= max_log_std:
            raise ValueError("min_log_std must be less than max_log_std.")
        if not min_log_std <= initial_log_std <= max_log_std:
            raise ValueError("initial_log_std must be within the configured standard deviation bounds.")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.mean_network = NeuralNetLayers(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=activation,
            layer_norm=layer_norm,
            output_gain=0.01,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    @classmethod
    def from_config(cls, config: Mapping):
        """Create an Actor from the `actor` mapping in network.yaml."""
        actor_config = config.get("actor", config)
        if not isinstance(actor_config, Mapping):
            raise ValueError("Actor configuration must be a mapping.")
        return cls(**dict(actor_config))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the unsquashed Gaussian mean for each observation."""
        self._validate_observations(observations)
        return self.mean_network(observations)

    def distribution(self, observations: torch.Tensor) -> TransformedDistribution:
        """Return a tanh-transformed diagonal Gaussian with actions in (-1, 1)."""
        means = self(observations)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        base_distribution = Normal(means, log_std.exp().expand_as(means))
        return TransformedDistribution(
            base_distribution,
            [TanhTransform(cache_size=1)],
        )

    def sample_action(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or select the mean action) and return its corrected log probability."""
        distribution = self.distribution(observations)
        if deterministic:
            actions = torch.tanh(self(observations))
        else:
            actions = distribution.rsample()
        log_probabilities = distribution.log_prob(actions).sum(dim=-1)
        return actions, log_probabilities

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate stored bounded actions, summing log probabilities by action dimension."""
        distribution = self.distribution(observations)
        if actions.shape[-1:] != (self.action_dim,):
            raise ValueError(
                f"Expected actions with final dimension {self.action_dim}, "
                f"got shape {tuple(actions.shape)}."
            )
        if not torch.is_floating_point(actions):
            raise TypeError("actions must have a floating-point dtype.")
        epsilon = torch.finfo(actions.dtype).eps
        bounded_actions = actions.clamp(-1.0 + epsilon, 1.0 - epsilon)
        return distribution.log_prob(bounded_actions).sum(dim=-1)

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor.")
        if observations.ndim == 0 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observations with final dimension {self.observation_dim}, "
                f"got shape {tuple(observations.shape)}."
            )
        if not torch.is_floating_point(observations):
            raise TypeError("observations must have a floating-point dtype.")

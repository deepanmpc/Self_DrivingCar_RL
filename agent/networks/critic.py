"""State-value critic for continuous-control PPO."""

from collections.abc import Mapping
from typing import Sequence

import torch
import torch.nn as nn

from agent.networks.layers import NeuralNetLayers


class Critic(nn.Module):
    """Map observations to a scalar estimate of expected future return."""

    def __init__(
        self,
        observation_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 256, 256, 128, 128, 128, 64),
        activation: str = "relu",
        layer_norm: bool = False,
        output_gain: float = 1.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.value_network = NeuralNetLayers(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm,
            output_gain=output_gain,
        )

    @classmethod
    def from_config(cls, config: Mapping):
        """Create a Critic from the `critic` mapping in network.yaml."""
        critic_config = config.get("critic", config)
        if not isinstance(critic_config, Mapping):
            raise ValueError("Critic configuration must be a mapping.")
        return cls(**dict(critic_config))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one state-value estimate per observation."""
        self._validate_observations(observations)
        return self.value_network(observations).squeeze(-1)

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor.")
        if observations.ndim == 0 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observations with final dimension {self.observation_dim}, "
                f"got shape {tuple(observations.shape)}."
            )
        if not torch.is_floating_point(observations):
            raise TypeError("observations must be a floating-point dtype.")

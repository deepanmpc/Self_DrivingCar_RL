"""Action distributions and probability calculations for continuous policies."""

import math
from typing import Tuple

import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class SquashedGaussian:
    """Diagonal Gaussian transformed by tanh to produce actions in (-1, 1)."""

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        if mean.ndim == 0 or mean.shape[-1] == 0:
            raise ValueError("Gaussian means must have a non-empty action dimension.")
        if not torch.is_floating_point(mean):
            raise TypeError("Gaussian means must have a floating-point dtype.")
        if not torch.is_floating_point(log_std):
            raise TypeError("Gaussian log standard deviations must be floating point.")
        if min_log_std >= max_log_std:
            raise ValueError("min_log_std must be less than max_log_std.")
        if not math.isfinite(min_log_std) or not math.isfinite(max_log_std):
            raise ValueError("Log standard deviation bounds must be finite.")

        bounded_log_std = log_std.clamp(min_log_std, max_log_std)
        try:
            standard_deviation = bounded_log_std.exp().expand_as(mean)
        except RuntimeError as error:
            raise ValueError("log_std must broadcast to the shape of mean.") from error
        self.base_distribution = Normal(mean, standard_deviation)
        self.transformed_distribution = TransformedDistribution(
            self.base_distribution,
            [TanhTransform(cache_size=1)],
        )
        self.action_dim = mean.shape[-1]

    def sample(self, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return reparameterized bounded actions and corrected joint log probabilities."""
        if deterministic:
            actions = torch.tanh(self.base_distribution.mean)
        else:
            actions = self.transformed_distribution.rsample()
        return actions, self.log_prob(actions)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate corrected joint log probabilities over the action dimensions."""
        if actions.shape[-1:] != (self.action_dim,):
            raise ValueError(
                f"Expected actions with final dimension {self.action_dim}, "
                f"got shape {tuple(actions.shape)}."
            )
        if not torch.is_floating_point(actions):
            raise TypeError("actions must have a floating-point dtype.")
        epsilon = torch.finfo(actions.dtype).eps
        bounded_actions = actions.clamp(-1.0 + epsilon, 1.0 - epsilon)
        return self.transformed_distribution.log_prob(bounded_actions).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Estimate transformed-distribution entropy with a reparameterized sample."""
        actions = self.transformed_distribution.rsample()
        return -self.log_prob(actions)

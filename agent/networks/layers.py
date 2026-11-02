"""Reusable neural network layer and initialization helpers."""

from typing import Sequence

import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def make_activation(name: str) -> nn.Module:
    """Create one of the supported activation functions by name."""
    try:
        activation_type = _ACTIVATIONS[name.lower()]
    except (AttributeError, KeyError) as error:
        valid_names = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unsupported activation {name!r}; choose from: {valid_names}.") from error
    return activation_type()


def initialize_linear(layer: nn.Linear, activation: str, gain: float = 1.0) -> None:
    """Initialize a dense layer consistently with its following activation."""
    activation = activation.lower()
    if activation == "relu":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    elif activation in {"tanh", "silu", "gelu"}:
        initialization_gain = nn.init.calculate_gain("tanh") if activation == "tanh" else gain
        nn.init.xavier_uniform_(layer.weight, gain=initialization_gain)
    else:
        make_activation(activation)
    nn.init.zeros_(layer.bias)


class NeuralNetLayers(nn.Module):
    """Configurable multilayer perceptron with optional hidden layer normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        layer_norm: bool = False, 
        output_gain: float = 0.01,
    ):
        super().__init__()
        dimensions = (input_dim, *hidden_dims, output_dim)
        if any(not isinstance(size, int) or size <= 0 for size in dimensions):
            raise ValueError("All network dimensions must be positive integers.")
        if not output_gain > 0:
            raise ValueError("output_gain must be positive.")
        make_activation(activation)

        modules = []
        for index, (input_size, output_size) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            linear = nn.Linear(input_size, output_size)
            is_output = index == len(dimensions) - 2
            if is_output:
                nn.init.orthogonal_(linear.weight, gain=output_gain)
                nn.init.zeros_(linear.bias)
            else:
                initialize_linear(linear, activation)
            modules.append(linear)

            if not is_output:
                if layer_norm:
                    modules.append(nn.LayerNorm(output_size))
                modules.append(make_activation(activation))

        self.network = nn.Sequential(*modules)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)

    def forward(self, inputs):
        if inputs.ndim == 0 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected inputs with final dimension {self.input_dim}, "
                f"got shape {tuple(inputs.shape)}."
            )
        return self.network(inputs)

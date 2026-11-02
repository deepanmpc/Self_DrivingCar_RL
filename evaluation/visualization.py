"""Export the actor and critic networks into a browser-friendly JSON model."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml

from agent.networks import Actor, Critic
from agent.networks.layers import NeuralNetLayers


ACTIVATION_NAMES = {
    nn.ReLU: "relu",
    nn.Tanh: "tanh",
    nn.SiLU: "silu",
    nn.GELU: "gelu",
}

FEATURE_METADATA = {
    "speed": {
        "label": "Speed",
        "unit": "m/s",
        "description": "Current vehicle speed, normalized by the configured maximum.",
    },
    "lane_offset": {
        "label": "Lane offset",
        "unit": "m",
        "description": "Signed distance from lane center. Positive is right of center.",
    },
    "heading_error": {
        "label": "Heading error",
        "unit": "rad",
        "description": "Vehicle heading minus lane heading, wrapped to [-pi, pi].",
    },
    "left_boundary_distance": {
        "label": "Left boundary",
        "unit": "m",
        "description": "Distance to the left road boundary.",
    },
    "right_boundary_distance": {
        "label": "Right boundary",
        "unit": "m",
        "description": "Distance to the right road boundary.",
    },
    "front_obstacle_distance": {
        "label": "Front obstacle",
        "unit": "m",
        "description": "Nearest forward gap. A value of 1.0 means clear through sensor range.",
    },
    "road_curvature": {
        "label": "Road curvature",
        "unit": "1/m",
        "description": "Signed lane heading change over the configured lookahead.",
    },
    "previous_steering": {
        "label": "Previous steering",
        "unit": "",
        "description": "Steering command applied on the previous step.",
    },
}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def build_networks(network_config: Mapping) -> tuple[Actor, Optional[Critic]]:
    actor = Actor.from_config(network_config)
    critic = Critic.from_config(network_config) if "critic" in network_config else None
    return actor, critic


def load_checkpoint(
    actor: Actor,
    critic: Optional[Critic],
    checkpoint_path: Path,
    allow_unsafe_load: bool = False,
) -> list[str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        if not allow_unsafe_load:
            raise RuntimeError(
                f"Could not safely load {checkpoint_path}. "
                "Pass --allow-unsafe-load only for checkpoints you trust."
            ) from None
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    targets = {"actor": actor}
    if critic is not None:
        targets["critic"] = critic

    explicit: dict[str, dict[str, torch.Tensor]] = {}
    for name in targets:
        state = _extract_named_state(payload, name)
        if state:
            explicit[name] = state

    if not explicit:
        global_state = _as_state_dict(payload)
        if global_state is None:
            for wrapper in ("state_dict", "model_state_dict", "networks", "model"):
                if isinstance(payload, Mapping) and wrapper in payload:
                    global_state = _as_state_dict(payload[wrapper])
                    if global_state is not None:
                        break
        if global_state is None:
            raise RuntimeError(
                f"No recognisable actor or critic weights found in {checkpoint_path}."
            )
        explicit = _split_global_state(global_state)

    loaded = []
    for name, state in explicit.items():
        network = targets[name]
        _load_state(network, state, name)
        loaded.append(name)
    if not loaded:
        raise RuntimeError(f"No actor or critic weights found in {checkpoint_path}.")
    return loaded


def _extract_named_state(payload: Any, name: str) -> Optional[dict[str, torch.Tensor]]:
    if isinstance(payload, nn.Module):
        state = payload.state_dict()
        return state if any(_belongs_to(key, name) for key in state) else None
    if not isinstance(payload, Mapping):
        return None

    for key in (name, f"{name}_state_dict", f"{name}_model_state_dict", f"{name}_network"):
        if key in payload:
            state = _as_state_dict(payload[key])
            if state:
                return state

    for wrapper in ("state_dict", "model_state_dict", "networks", "models", "model"):
        if wrapper in payload:
            state = _extract_named_state(payload[wrapper], name)
            if state:
                return state
    return None


def _as_state_dict(value: Any) -> Optional[dict[str, torch.Tensor]]:
    if isinstance(value, nn.Module):
        return value.state_dict()
    if not isinstance(value, Mapping):
        return None
    state = {str(key): tensor for key, tensor in value.items() if isinstance(tensor, torch.Tensor)}
    return state or None


def _split_global_state(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {"actor": {}, "critic": {}}
    for key, value in state.items():
        if key.startswith("actor."):
            groups["actor"][key[len("actor."):]] = value
        elif key.startswith("critic."):
            groups["critic"][key[len("critic."):]] = value
        elif key.startswith("mean_network.") or key == "log_std":
            groups["actor"][key] = value
        elif key.startswith("value_network."):
            groups["critic"][key] = value
    return {name: values for name, values in groups.items() if values}


def _belongs_to(key: str, name: str) -> bool:
    if name == "actor":
        return key == "log_std" or key.startswith("mean_network.")
    return key.startswith("value_network.")


def _load_state(network: nn.Module, state: Mapping[str, torch.Tensor], name: str) -> None:
    model_keys = set(network.state_dict())
    filtered = {
        key: value
        for key, value in state.items()
        if key in model_keys and isinstance(value, torch.Tensor)
    }
    if not filtered:
        raise RuntimeError(
            f"Checkpoint does not contain any {name} parameters expected by the configured network."
        )
    missing = sorted(model_keys - set(filtered))
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {name} parameters: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
        )
    if not all(torch.isfinite(value).all() for value in filtered.values()):
        raise RuntimeError(f"Checkpoint contains non-finite {name} parameters.")
    network.load_state_dict(filtered, strict=True)


def serialize_network(
    network: NeuralNetLayers,
    role: str,
    output_metadata: Mapping[str, Any],
) -> dict:
    layers = []
    for module in network.network:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "kind": "linear",
                    "in": int(module.in_features),
                    "out": int(module.out_features),
                    "weight": _rounded_matrix(module.weight.detach().cpu().tolist()),
                    "bias": _rounded_vector(module.bias.detach().cpu().tolist())
                    if module.bias is not None
                    else None,
                }
            )
        elif isinstance(module, nn.LayerNorm):
            layers.append(
                {
                    "kind": "layernorm",
                    "size": int(module.normalized_shape[0]),
                    "eps": float(module.eps),
                    "weight": _rounded_vector(module.weight.detach().cpu().tolist())
                    if module.weight is not None
                    else None,
                    "bias": _rounded_vector(module.bias.detach().cpu().tolist())
                    if module.bias is not None
                    else None,
                }
            )
        else:
            activation = _activation_name(module)
            if activation is None:
                raise TypeError(
                    f"Unsupported module in {role} network: {type(module).__name__}."
                )
            layers.append({"kind": "activation", "name": activation})
    return {"role": role, "layers": layers, "output": dict(output_metadata)}


def _activation_name(module: nn.Module) -> Optional[str]:
    for module_type, name in ACTIVATION_NAMES.items():
        if isinstance(module, module_type):
            return name
    return None


def _rounded_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in matrix]


def _rounded_vector(values: Sequence[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def serialize_observation(
    observation_config: Mapping,
    environment_config: Mapping,
) -> dict:
    order = list(observation_config.get("feature_order", []))
    normalization = observation_config.get("normalization", {})
    ranges = observation_config.get("feature_ranges", {})
    lane_width = float(environment_config.get("road", {}).get("lane_width", 3.5))

    physical_ranges = {
        "speed": [0.0, float(normalization.get("speed_max_m_s", 40.0))],
        "lane_offset": [-lane_width / 2.0, lane_width / 2.0],
        "heading_error": [-math.pi, math.pi],
        "left_boundary_distance": [0.0, float(normalization.get("boundary_distance_max_m", 10.0))],
        "right_boundary_distance": [0.0, float(normalization.get("boundary_distance_max_m", 10.0))],
        "front_obstacle_distance": [0.0, float(normalization.get("obstacle_distance_max_m", 50.0))],
        "road_curvature": [
            -float(normalization.get("curvature_max_per_m", 0.1)),
            float(normalization.get("curvature_max_per_m", 0.1)),
        ],
        "previous_steering": [-1.0, 1.0],
    }

    features = []
    for index, key in enumerate(order):
        metadata = FEATURE_METADATA.get(key, {})
        normalized_range = [float(value) for value in ranges.get(key, [-1.0, 1.0])]
        physical_range = physical_ranges.get(key, normalized_range)
        features.append(
            {
                "key": key,
                "label": metadata.get("label", key.replace("_", " ").title()),
                "unit": metadata.get("unit", ""),
                "description": metadata.get("description", ""),
                "range": normalized_range,
                "physicalRange": physical_range,
                "index": index,
            }
        )

    return {
        "featureOrder": order,
        "ranges": {key: [float(v) for v in value] for key, value in ranges.items()},
        "features": features,
    }


def serialize_model(
    actor: Actor,
    critic: Optional[Critic],
    observation_config: Mapping,
    environment_config: Mapping,
    label: str,
    checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    actor_output = {
        "labels": ["steering", "throttle_brake"],
        "squash": "tanh",
        "logStd": [float(value) for value in actor.log_std.detach().cpu().tolist()],
        "minLogStd": float(actor.min_log_std),
        "maxLogStd": float(actor.max_log_std),
    }
    networks = {
        "actor": serialize_network(actor.mean_network, "actor", actor_output),
    }
    parameters = sum(parameter.numel() for parameter in actor.parameters())
    if critic is not None:
        critic_output = {"labels": ["V(s)"], "squash": None}
        networks["critic"] = serialize_network(
            critic.value_network, "critic", critic_output
        )
        parameters += sum(parameter.numel() for parameter in critic.parameters())

    model = {
        "formatVersion": 1,
        "metadata": {
            "label": label,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": checkpoint,
            "seed": seed,
            "parameters": int(parameters),
            "networks": list(networks),
        },
        "observation": serialize_observation(observation_config, environment_config),
        "networks": networks,
    }
    model["validation"] = build_validation(
        model,
        observation_config,
        actor,
        critic,
    )
    return model


def build_validation(
    model: Mapping,
    observation_config: Mapping,
    actor: Actor,
    critic: Optional[Critic],
    sample_count: int = 3,
) -> dict:
    features = model["observation"]["features"]
    defaults = [(_default_range(feature)[0] + _default_range(feature)[1]) / 2.0 for feature in features]
    generator = torch.Generator().manual_seed(1234)
    samples = [defaults, [0.0] * len(features)]
    for _ in range(max(1, sample_count - len(samples))):
        values = []
        for feature in features:
            low, high = _default_range(feature)
            values.append(float(torch.empty(1).uniform_(low, high, generator=generator).item()))
        samples.append(values)

    validation = {"tolerance": 1e-4, "maxTorchError": 0.0, "samples": []}
    max_error = 0.0
    for inputs in samples:
        actor_serialized = forward_serialized(model["networks"]["actor"], inputs)
        actor_actions = [math.tanh(value) for value in actor_serialized]
        outputs: dict[str, Any] = {
            "actor": {"mean": actor_serialized, "actions": actor_actions}
        }
        with torch.no_grad():
            actor_tensor = torch.tensor([inputs], dtype=torch.float32)
            actor_torch = actor(actor_tensor)[0].cpu().tolist()
        max_error = max(
            max_error,
            max(abs(left - right) for left, right in zip(actor_serialized, actor_torch)),
        )

        if critic is not None and "critic" in model["networks"]:
            critic_serialized = forward_serialized(model["networks"]["critic"], inputs)
            outputs["critic"] = {"value": critic_serialized[0]}
            with torch.no_grad():
                critic_torch = float(critic(actor_tensor)[0].item())
            max_error = max(max_error, abs(critic_serialized[0] - critic_torch))

        validation["samples"].append({"input": inputs, "outputs": outputs})

    validation["maxTorchError"] = max_error
    if max_error > validation["tolerance"]:
        raise ValueError(
            f"Serialized weights diverge from PyTorch by {max_error:.6g}, "
            f"above tolerance {validation['tolerance']:.6g}."
        )
    return validation


def _default_range(feature: Mapping) -> list[float]:
    values = feature.get("range", [-1.0, 1.0])
    return [float(values[0]), float(values[1])]


def forward_serialized(network: Mapping, inputs: Sequence[float]) -> list[float]:
    values = [float(value) for value in inputs]
    for layer in network.get("layers", []):
        kind = layer.get("kind")
        if kind == "linear":
            weights = layer["weight"]
            bias = layer.get("bias")
            linear = []
            for index, row in enumerate(weights):
                total = float(bias[index]) if bias is not None else 0.0
                for weight, value in zip(row, values):
                    total += float(weight) * value
                linear.append(total)
            values = linear
        elif kind == "layernorm":
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            denominator = math.sqrt(variance + float(layer["eps"]))
            normalized = [(value - mean) / denominator for value in values]
            weight = layer.get("weight")
            bias = layer.get("bias")
            values = [
                value * (float(weight[index]) if weight is not None else 1.0)
                + (float(bias[index]) if bias is not None else 0.0)
                for index, value in enumerate(normalized)
            ]
        elif kind == "activation":
            values = [_activate(layer["name"], value) for value in values]
        else:
            raise ValueError(f"Unsupported serialized layer kind: {kind}")
    return values


def _activate(name: str, value: float) -> float:
    if name == "relu":
        return max(0.0, value)
    if name == "tanh":
        return math.tanh(value)
    if name == "silu":
        return value / (1.0 + math.exp(-value)) if value >= 0 else value * math.exp(value) / (1.0 + math.exp(value))
    if name == "gelu":
        return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
    raise ValueError(f"Unsupported activation: {name}")


def write_model(model: Mapping, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as model_file:
        json.dump(model, model_file, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, output_path)
    update_manifest(output_path, model)


def update_manifest(output_path: Path, model: Mapping) -> None:
    manifest_path = output_path.parent / "manifest.json"
    entries = []
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as manifest_file:
                existing = json.load(manifest_file)
            if isinstance(existing, list):
                entries = existing
        except (OSError, json.JSONDecodeError):
            entries = []

    metadata = model["metadata"]
    entry = {
        "id": output_path.stem,
        "label": metadata["label"],
        "file": output_path.name,
        "createdAt": metadata["createdAt"],
        "checkpoint": metadata["checkpoint"],
        "parameters": metadata["parameters"],
        "networks": metadata["networks"],
    }
    entries = [item for item in entries if item.get("file") != output_path.name]
    entries.insert(0, entry)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(entries, manifest_file, indent=2, allow_nan=False)

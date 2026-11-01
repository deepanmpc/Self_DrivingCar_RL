"""Configurable, decomposed reward for safe lane-following behavior."""

import math
from typing import Dict, Mapping, Optional, Tuple
import numpy as np


class DrivingReward:
    """Calculate reward components from one post-step MetaDrive vehicle state."""

    REQUIRED_SETTINGS = (
        "progress_weight",
        "max_progress_m_per_step",
        "lane_weight",
        "speed_weight",
        "target_speed_m_s",
        "collision_penalty",
        "offroad_penalty",
        "steering_penalty_weight",
        "steering_deadband",
        "smoothness_penalty_weight",
    )

    def __init__(self, config: Mapping[str, float]):
        missing = [key for key in self.REQUIRED_SETTINGS if key not in config]
        if missing:
            raise ValueError(f"Reward configuration is missing settings: {', '.join(missing)}")

        self.config = {key: float(config[key]) for key in self.REQUIRED_SETTINGS}
        if not all(math.isfinite(value) for value in self.config.values()):
            raise ValueError("Reward settings must be finite numbers.")
        if any(self.config[key] < 0 for key in self.REQUIRED_SETTINGS if key != "steering_deadband"):
            raise ValueError("Reward weights, penalties, and scales must be non-negative.")
        if not 0 <= self.config["steering_deadband"] < 1:
            raise ValueError("steering_deadband must be in [0, 1).")
        if self.config["max_progress_m_per_step"] == 0 or self.config["target_speed_m_s"] == 0:
            raise ValueError("Progress scale and target speed must be positive.")
        self._previous_steering = 0.0

    def reset(self) -> None:
        """Clear action history at the start of an episode."""
        self._previous_steering = 0.0

    def calculate(self,vehicle,steering: float,) -> Tuple[float, Dict[str, float]]:
        """Return the scalar reward and independently reported components."""
        steering = self._normalized_value(steering, "steering")
        previous_steering = self._previous_steering

        components = {
            "progress_reward": self._progress_reward(vehicle),
            "lane_reward": self._lane_reward(vehicle),
            "speed_reward": self._speed_reward(vehicle),
            "collision_penalty": self._collision_penalty(vehicle),
            "offroad_penalty": self._offroad_penalty(vehicle),
            "steering_penalty": self._steering_penalty(steering),
            "smoothness_penalty": self._smoothness_penalty(steering, previous_steering),
        }
        total = (
            components["progress_reward"]
            + components["lane_reward"]
            + components["speed_reward"]
            - components["collision_penalty"]
            - components["offroad_penalty"]
            - components["steering_penalty"]
            - components["smoothness_penalty"]
        )
        self._previous_steering = steering
        components["total_reward"] = float(total)
        return float(total), components

    def _progress_reward(self, vehicle) -> float:
        lane = self._route_lane(vehicle)
        previous_position = getattr(vehicle, "last_position", None)
        position = getattr(vehicle, "position", None)
        if lane is None or previous_position is None or position is None:
            return 0.0

        try:
            previous_longitudinal = lane.local_coordinates(previous_position)[0]
            longitudinal = lane.local_coordinates(position)[0]
            progress = float(longitudinal - previous_longitudinal)
            navigation = getattr(vehicle, "navigation", None)
            reference_lanes = getattr(navigation, "current_ref_lanes", None)
            if reference_lanes and lane not in reference_lanes:
                road = getattr(navigation, "current_road", None)
                if road is not None and road.is_negative_road():
                    progress = -progress
            progress = float(
                np.clip(
                    progress,
                    -self.config["max_progress_m_per_step"],
                    self.config["max_progress_m_per_step"],
                )
            )
        except (AttributeError, TypeError, ValueError, IndexError):
            return 0.0
        return self.config["progress_weight"] * progress

    def _lane_reward(self, vehicle) -> float:
        lane = self._route_lane(vehicle)
        position = getattr(vehicle, "position", None)
        if lane is None or position is None:
            return 0.0
        try:
            longitudinal, lateral = lane.local_coordinates(position)
            half_width = float(lane.width_at(longitudinal)) / 2.0
        except (AttributeError, TypeError, ValueError, IndexError):
            return 0.0
        if not math.isfinite(half_width) or half_width <= 0 or not math.isfinite(float(lateral)):
            return 0.0
        centered_score = float(np.clip(1.0 - abs(float(lateral)) / half_width, 0.0, 1.0))
        return self.config["lane_weight"] * centered_score

    def _speed_reward(self, vehicle) -> float:
        speed = self._finite_nonnegative(getattr(vehicle, "speed", None))
        if speed is None:
            return 0.0
        target_speed = self.config["target_speed_m_s"]
        speed_score = min(speed / target_speed, 1.0)
        return self.config["speed_weight"] * speed_score

    def _collision_penalty(self, vehicle) -> float:
        collision_flags = (
            "crash_vehicle",
            "crash_object",
            "crash_building",
            "crash_human",
            "crash_sidewalk",
        )
        collided = any(bool(getattr(vehicle, flag, False)) for flag in collision_flags)
        return self.config["collision_penalty"] if collided else 0.0

    def _offroad_penalty(self, vehicle) -> float:
        offroad = (
            not bool(getattr(vehicle, "on_lane", True))
            or bool(getattr(vehicle, "on_yellow_continuous_line", False))
            or bool(getattr(vehicle, "on_white_continuous_line", False))
            or bool(getattr(vehicle, "crash_sidewalk", False))
        )
        return self.config["offroad_penalty"] if offroad else 0.0

    def _steering_penalty(self, steering: float) -> float:
        deadband = self.config["steering_deadband"]
        excess = max(abs(steering) - deadband, 0.0)
        normalized_excess = excess / (1.0 - deadband)
        return self.config["steering_penalty_weight"] * normalized_excess

    def _smoothness_penalty(self, steering: float, previous_steering: float) -> float:
        steering_change = min(abs(steering - previous_steering) / 2.0, 1.0)
        return self.config["smoothness_penalty_weight"] * steering_change

    @staticmethod
    def _route_lane(vehicle):
        navigation = getattr(vehicle, "navigation", None)
        reference_lanes = getattr(navigation, "current_ref_lanes", None)
        if reference_lanes:
            return reference_lanes[0]
        return getattr(vehicle, "lane", None)

    @staticmethod
    def _finite_nonnegative(value) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        return value

    @staticmethod
    def _normalized_value(value, name: str) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite number.") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        return float(np.clip(value, -1.0, 1.0))
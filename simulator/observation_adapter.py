"""Convert MetaDrive vehicle state into a normalized eight-feature observation."""

import math
from typing import Optional

import gymnasium as gym
import numpy as np


class EightInputObservation:
    """Build the fixed-order, eight-dimensional driving observation."""

    FEATURE_NAMES = (
        "speed",
        "lane_offset",
        "heading_error",
        "left_boundary_distance",
        "right_boundary_distance",
        "front_obstacle_distance",
        "road_curvature",
        "previous_steering",
    )

    def __init__(
        self,
        speed_max_m_s: float = 40.0,
        boundary_distance_max_m: float = 10.0,
        obstacle_distance_max_m: float = 50.0,
        curvature_max_per_m: float = 0.1,
        curvature_lookahead_m: float = 10.0,
    ):
        limits = (
            speed_max_m_s,
            boundary_distance_max_m,
            obstacle_distance_max_m,
            curvature_max_per_m,
            curvature_lookahead_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in limits):
            raise ValueError("Observation scales and lookahead must be finite and positive.")

        self.speed_max_m_s = speed_max_m_s
        self.boundary_distance_max_m = boundary_distance_max_m
        self.obstacle_distance_max_m = obstacle_distance_max_m
        self.curvature_max_per_m = curvature_max_per_m
        self.curvature_lookahead_m = curvature_lookahead_m
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

    def observe(self, vehicle, previous_steering: float = 0.0) -> np.ndarray:
        """Return speed, lane pose, boundaries, obstacle, curvature, and steering."""
        lane = self._current_lane(vehicle)
        speed = self._scale_nonnegative(getattr(vehicle, "speed", None), self.speed_max_m_s)

        lane_offset = 0.0
        heading_error = 0.0
        curvature = 0.0
        if lane is not None:
            longitudinal, lateral = lane.local_coordinates(vehicle.position)
            half_width = lane.width_at(longitudinal) / 2.0
            if math.isfinite(half_width) and half_width > 0 and math.isfinite(lateral):
                lane_offset = float(np.clip(lateral / half_width, -1.0, 1.0))

            lane_heading = lane.heading_theta_at(longitudinal)
            vehicle_heading = getattr(vehicle, "heading_theta", None)
            if self._is_finite(vehicle_heading) and math.isfinite(lane_heading):
                heading_error = self._wrap_to_pi(float(vehicle_heading) - lane_heading) / math.pi

            lookahead = self.curvature_lookahead_m
            distance_to_end = max(0.0, lane.length - longitudinal)
            measured_distance = min(lookahead, distance_to_end)
            future_heading = lane.heading_theta_at(longitudinal + measured_distance)
            if distance_to_end < lookahead:
                next_lanes = getattr(vehicle.navigation, "next_ref_lanes", None)
                if next_lanes:
                    next_lane = next_lanes[0]
                    next_distance = min(lookahead - distance_to_end, next_lane.length)
                    future_heading = next_lane.heading_theta_at(next_distance)
                    measured_distance += next_distance

            if (
                measured_distance > 0
                and math.isfinite(lane_heading)
                and math.isfinite(future_heading)
            ):
                heading_change = self._wrap_to_pi(future_heading - lane_heading)
                curvature = float(
                    np.clip(
                        (heading_change / measured_distance) / self.curvature_max_per_m,
                        -1.0,
                        1.0,
                    )
                )

        left_boundary = self._scale_nonnegative(
            getattr(vehicle, "dist_to_left_side", None), self.boundary_distance_max_m
        )
        right_boundary = self._scale_nonnegative(
            getattr(vehicle, "dist_to_right_side", None), self.boundary_distance_max_m
        )
        obstacle_distance = self._front_obstacle_distance(vehicle)
        if obstacle_distance is None:
            normalized_obstacle_distance = 0.0
        else:
            normalized_obstacle_distance = self._scale_nonnegative(
                obstacle_distance, self.obstacle_distance_max_m
            )

        values = np.asarray(
            [
                speed,
                lane_offset,
                heading_error,
                left_boundary,
                right_boundary,
                normalized_obstacle_distance,
                curvature,
                float(np.clip(previous_steering, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Observation contains a non-finite feature.")
        return values

    @staticmethod
    def _current_lane(vehicle):
        navigation = getattr(vehicle, "navigation", None)
        if navigation is None:
            return None
        return getattr(navigation, "current_lane", None)

    def _front_obstacle_distance(self, vehicle) -> Optional[float]:
        lidar = getattr(vehicle, "lidar", None)
        if lidar is None:
            return None

        detected_objects = lidar.get_surrounding_objects(vehicle, self.obstacle_distance_max_m)
        heading = np.asarray(vehicle.heading, dtype=float)
        position = np.asarray(vehicle.position, dtype=float)
        nearest_gap = None

        for obstacle in detected_objects:
            obstacle_position = getattr(obstacle, "position", None)
            if obstacle_position is None:
                continue
            relative_position = np.asarray(obstacle_position, dtype=float) - position
            forward_distance = float(np.dot(relative_position, heading))
            lateral_distance = float(
                heading[0] * relative_position[1] - heading[1] * relative_position[0]
            )
            obstacle_width = getattr(obstacle, "WIDTH", 0.0)
            obstacle_length = getattr(obstacle, "LENGTH", 0.0)
            half_corridor = (float(getattr(vehicle, "WIDTH", 0.0)) + float(obstacle_width)) / 2.0

            if forward_distance <= 0 or abs(lateral_distance) > half_corridor:
                continue

            gap = max(
                0.0,
                forward_distance
                - float(getattr(vehicle, "LENGTH", 0.0)) / 2.0
                - float(obstacle_length) / 2.0,
            )
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
        return self.obstacle_distance_max_m if nearest_gap is None else nearest_gap

    @classmethod
    def _scale_nonnegative(cls, value, maximum: float) -> float:
        if not cls._is_finite(value):
            return 0.0
        return float(np.clip(float(value) / maximum, 0.0, 1.0))

    @staticmethod
    def _is_finite(value) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
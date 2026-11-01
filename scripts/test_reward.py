"""Test reward components using controlled vehicle and lane states."""

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.reward_function import DrivingReward


class FakeLane:
    width = 4.0

    def local_coordinates(self, position):
        return float(position[0]), float(position[1])

    def width_at(self, longitudinal):
        return self.width


def build_vehicle(**overrides):
    lane = FakeLane()
    vehicle = SimpleNamespace(
        lane=lane,
        position=(2.0, 0.0),
        last_position=(1.0, 0.0),
        speed=10.0,
        on_lane=True,
        on_yellow_continuous_line=False,
        on_white_continuous_line=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
    )
    for key, value in overrides.items():
        setattr(vehicle, key, value)
    return vehicle


def load_reward_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "reward.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    assert isinstance(config, dict)
    return config


def main():
    config = load_reward_config()
    reward = DrivingReward(config)

    good_vehicle = build_vehicle()
    good_total, good_parts = reward.calculate(good_vehicle, steering=0.0)
    assert good_parts["progress_reward"] > 0
    assert good_parts["lane_reward"] > 0
    assert 0 < good_parts["speed_reward"] <= config["speed_weight"]
    assert good_parts["collision_penalty"] == 0
    assert good_parts["offroad_penalty"] == 0
    assert math.isclose(
        good_total,
        sum(
            good_parts[key]
            for key in ("progress_reward", "lane_reward", "speed_reward")
        )
        - good_parts["collision_penalty"]
        - good_parts["offroad_penalty"]
        - good_parts["steering_penalty"]
        - good_parts["smoothness_penalty"],
    )

    collision_vehicle = build_vehicle(crash_vehicle=True)
    collision_total, collision_parts = DrivingReward(config).calculate(collision_vehicle, 0.0)
    assert collision_parts["collision_penalty"] == config["collision_penalty"]
    assert collision_total < good_total

    offroad_vehicle = build_vehicle(position=(2.0, 3.0), on_lane=False)
    offroad_total, offroad_parts = DrivingReward(config).calculate(offroad_vehicle, 0.0)
    assert offroad_parts["offroad_penalty"] == config["offroad_penalty"]
    assert offroad_parts["lane_reward"] == 0.0
    assert offroad_total < good_total

    stationary_vehicle = build_vehicle(last_position=(2.0, 0.0))
    stationary_total, stationary_parts = DrivingReward(config).calculate(stationary_vehicle, 0.0)
    assert stationary_parts["progress_reward"] == 0.0
    assert stationary_parts["speed_reward"] > 0.0
    assert good_total > stationary_total

    fast_vehicle = build_vehicle(speed=1000.0)
    _, fast_parts = DrivingReward(config).calculate(fast_vehicle, 0.0)
    assert fast_parts["speed_reward"] == config["speed_weight"]

    action_reward = DrivingReward(config)
    _, smooth_parts = action_reward.calculate(good_vehicle, steering=0.0)
    _, abrupt_parts = action_reward.calculate(good_vehicle, steering=1.0)
    assert abrupt_parts["steering_penalty"] > smooth_parts["steering_penalty"]
    assert abrupt_parts["smoothness_penalty"] > smooth_parts["smoothness_penalty"]

    _, subsequent_parts = action_reward.calculate(good_vehicle, steering=1.0)
    assert subsequent_parts["smoothness_penalty"] < abrupt_parts["smoothness_penalty"]

    print("Reward component tests passed.")
    print(f"  good behavior total reward: {good_total:.4f}")
    print(f"  collision total reward:    {collision_total:.4f}")
    print(f"  off-road total reward:     {offroad_total:.4f}")
    print(f"  stationary total reward:   {stationary_total:.4f}")


if __name__ == "__main__":
    main()

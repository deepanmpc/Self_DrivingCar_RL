"""Print the eight normalized observation features while the vehicle moves."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.environment import DrivingEnvironment
from simulator.observation_adapter import EightInputObservation


def main():
    env = DrivingEnvironment(render=False)
    try:
        observation, _ = env.reset(seed=0)
        inspect_step(env, observation, "Reset")

        for step in range(20):
            observation, _, terminated, truncated, _ = env.step([0.25, 0.8])
            if step in (0, 4, 9, 19) or terminated or truncated:
                inspect_step(env, observation, f"Step {step + 1}")
            if terminated or truncated:
                break
    finally:
        env.close()


def inspect_step(env, observation, label):
    vehicle = env.vehicle
    lane = env.navigation.current_lane
    longitudinal, lateral = lane.local_coordinates(vehicle.position)
    raw_observation = env.raw_observation
    print(f"\n{label}")
    if isinstance(raw_observation, np.ndarray):
        preview = raw_observation.reshape(-1)[:10]
        print(
            "  MetaDrive raw observation: "
            f"shape={raw_observation.shape}, dtype={raw_observation.dtype}, first_values={preview}"
        )
    else:
        print(f"  MetaDrive raw observation: {raw_observation}")
    print(f"  position (m): {vehicle.position}")
    print(f"  speed (m/s): {vehicle.speed:.4f}")
    print(f"  vehicle heading (rad): {vehicle.heading_theta:.4f}")
    print(f"  lane coordinates (m): s={longitudinal:.4f}, offset={lateral:.4f}")
    print(f"  lane width (m): {lane.width_at(longitudinal):.4f}")
    print(f"  boundary distances (m): left={vehicle.dist_to_left_side}, right={vehicle.dist_to_right_side}")
    print(f"  previous steering: {env.action_adapter.previous_steering:.4f}")
    for name, value in zip(EightInputObservation.FEATURE_NAMES, observation):
        print(f"  normalized {name:26s} {value: .4f}")


if __name__ == "__main__":
    main()
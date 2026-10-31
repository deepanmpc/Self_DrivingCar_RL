"""Print the eight normalized observation features while the vehicle moves."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.environment import DrivingEnvironment
from simulator.observation_adapter import EightInputObservation


def main():
    env = DrivingEnvironment(render=False)
    try:
        observation, _ = env.reset(seed=0)
        print("Initial observation:")
        for name, value in zip(EightInputObservation.FEATURE_NAMES, observation):
            print(f"  {name:26s} {value: .4f}")

        for step in range(20):
            observation, _, terminated, truncated, _ = env.step([0.25, 0.8])
            if step in (0, 4, 9, 19) or terminated or truncated:
                print(f"\nStep {step + 1}: {observation}")
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
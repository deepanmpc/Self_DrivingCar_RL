import sys
import os
# Add project root to path so 'simulator' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulator.environment import DrivingEnvironment


def main():

    env = DrivingEnvironment(render=False)

    try:

        observation, info = env.reset(seed=0)

        print("\nEnvironment initialized")

        print("Observation type:", type(observation))
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)

        if isinstance(observation, np.ndarray):
            print("Observation shape:", observation.shape)

        print("\nStarting simulation...\n")

        for step in range(100):

            # Random action for testing only
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            print(
                f"Step: {step} | "
                f"Reward: {reward:.4f} | "
                f"Terminated: {terminated} | "
                f"Truncated: {truncated}"
            )

            observation = next_obs

            if terminated or truncated:

                print("Episode ended")

                observation, info = env.reset()

    finally:

        env.close()
        print("\nEnvironment closed successfully")


if __name__ == "__main__":
    main()
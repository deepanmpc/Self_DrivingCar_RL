"""Smoke-test observation ranges and state changes in a live MetaDrive episode."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.environment import DrivingEnvironment
from simulator.observation_adapter import EightInputObservation


def main():
    env = DrivingEnvironment(render=False)
    try:
        initial, _ = env.reset(seed=0)
        initial_position = np.asarray(env.env.agent.position, dtype=float)
        action = np.array([0.25, 0.8], dtype=np.float32)

        for _ in range(30):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        final_position = np.asarray(env.env.agent.position, dtype=float)
        assert initial.shape == (8,)
        assert observation.shape == (8,)
        assert env.observation_space.contains(observation)
        assert np.all(np.isfinite(observation))
        assert np.linalg.norm(final_position - initial_position) > 0.1, "Vehicle did not move."
        assert observation[0] > initial[0], "Speed feature did not increase as the vehicle accelerated."
        assert np.isclose(observation[7], action[0]), "Previous-steering feature does not match applied action."
        assert not np.allclose(observation[1:3], initial[1:3]), (
            "Neither lane offset nor heading error changed while the vehicle steered."
        )

        print("Eight-feature observation smoke test passed.")
        for name, value in zip(EightInputObservation.FEATURE_NAMES, observation):
            print(f"  {name:26s} {value: .4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

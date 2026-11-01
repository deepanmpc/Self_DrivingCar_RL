"""Smoke-test observation ranges and state changes in a live MetaDrive episode."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.environment import DrivingEnvironment
from simulator.action_adapter import ContinuousActionAdapter
from simulator.observation_adapter import EightInputObservation


def assert_valid_observation(env, observation):
    assert observation.shape == (8,), f"Expected eight features, got {observation.shape}."
    assert np.all(np.isfinite(observation)), "Observation contains NaN or infinity."
    assert env.observation_space.contains(observation), "Observation is outside configured feature ranges."


def main():
    env = DrivingEnvironment(render=False)
    try:
        initial, _ = env.reset(seed=0)
        assert_valid_observation(env, initial)
        assert initial[7] == 0.0, "Reset should clear previous steering."

        initial_position = np.asarray(env.vehicle.position, dtype=float)
        action = np.array([0.25, 0.8], dtype=np.float32)

        for _ in range(30):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        final_position = np.asarray(env.vehicle.position, dtype=float)
        assert env.raw_observation is not None, "Raw MetaDrive observation was not retained."
        assert_valid_observation(env, observation)
        assert np.linalg.norm(final_position - initial_position) > 0.1, "Vehicle did not move."
        assert observation[0] > initial[0], "Speed feature did not increase as the vehicle accelerated."
        assert np.isclose(observation[7], action[0]), "Previous-steering feature does not match applied action."
        assert not np.allclose(observation, initial), "Observation did not update after stepping."

        reset_observation, _ = env.reset(seed=0)
        assert_valid_observation(env, reset_observation)
        assert reset_observation[7] == 0.0, "Reset did not clear the previous steering action."

        action_adapter = ContinuousActionAdapter()
        clipped_action = action_adapter.adapt([1.5, -2.0])
        assert np.array_equal(clipped_action, np.array([1.0, -1.0], dtype=np.float32))
        assert action_adapter.previous_steering == 1.0
        for invalid_action in ([0.1], [0.1, 0.2, 0.3], [0.1, np.nan]):
            try:
                action_adapter.adapt(invalid_action)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid action was accepted: {invalid_action}")

        print("Eight-feature observation smoke test passed.")
        for name, value in zip(EightInputObservation.FEATURE_NAMES, observation):
            print(f"  {name:26s} {value: .4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

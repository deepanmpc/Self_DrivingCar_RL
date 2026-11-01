"""Validate policy actions and convert them to MetaDrive's continuous action format."""

import numpy as np


class ContinuousActionAdapter:
    """Map normalized steering and signed throttle/brake into MetaDrive controls.

    The second policy output is a signed longitudinal command: positive values
    accelerate and negative values brake. MetaDrive uses the same convention
    for its second continuous action dimension.
    """

    ACTION_DIMENSION = 2

    def __init__(self):
        self.previous_steering = 0.0

    def adapt(self, policy_action) -> np.ndarray:
        action = np.asarray(policy_action, dtype=np.float64)
        if action.shape != (self.ACTION_DIMENSION,):
            raise ValueError(
                f"Policy action must have shape ({self.ACTION_DIMENSION},), got {action.shape}."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy action values must all be finite.")

        adapted_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.previous_steering = float(adapted_action[0])
        return adapted_action

    def reset(self) -> None:
        self.previous_steering = 0.0
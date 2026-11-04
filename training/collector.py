"""Collect simulator transitions without performing policy optimization."""

from typing import Optional

import numpy as np

from agent.agent import Agent
from memory.rollout_buffer import RolloutBuffer


class RolloutCollector:
    """Run an environment for a fixed number of steps and record PPO data."""

    def __init__(self, environment, agent: Agent, buffer: Optional[RolloutBuffer] = None):
        self.environment = environment
        self.agent = agent
        self.buffer = buffer if buffer is not None else RolloutBuffer()
        self._observation = None

    def collect(self, num_steps: int, seed: Optional[int] = None) -> RolloutBuffer:
        """Collect exactly `num_steps`, resetting after terminated/truncated episodes."""
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        self.buffer.clear()
        if self._observation is None:
            self._observation, _ = self.environment.reset(seed=seed)

        for _ in range(num_steps):
            observation = np.asarray(self._observation, dtype=np.float32).copy()
            selection = self.agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = self.environment.step(
                selection.action
            )
            next_observation = np.asarray(next_observation, dtype=np.float32).copy()
            terminated = bool(terminated)
            truncated = bool(truncated)

            next_value = 0.0 if terminated else self.agent.estimate_value(next_observation)
            self.buffer.add(
                observation=observation,
                action=selection.action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                old_log_probability=selection.log_probability,
                value=selection.value,
                next_value=next_value,
            )

            if terminated or truncated:
                self._observation, _ = self.environment.reset()
            else:
                self._observation = next_observation

        return self.buffer

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and start a fresh rollout state."""
        self._observation, _ = self.environment.reset(seed=seed)
        return np.asarray(self._observation, dtype=np.float32).copy()

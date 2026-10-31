from pathlib import Path
import numpy as np
import yaml
from metadrive.envs.metadrive_env import MetaDriveEnv
from simulator.observation_adapter import EightInputObservation


class DrivingEnvironment:

    def __init__(self, render=False):

        config = {
            "use_render": render,
            "num_scenarios": 1,
            "start_seed": 0,
            "traffic_density": 0.0,
            "log_level": 50,
        }

        self.env = MetaDriveEnv(config=config)
        observation_config_path = Path(__file__).resolve().parents[1] / "configs" / "observation.yaml"
        with observation_config_path.open(encoding="utf-8") as config_file:
            observation_config = yaml.safe_load(config_file)
        if not isinstance(observation_config, dict) or not isinstance(
            observation_config.get("normalization"), dict
        ):
            raise ValueError(f"Invalid observation configuration: {observation_config_path}")
        self.observation_adapter = EightInputObservation(**observation_config["normalization"])
        self._vehicle = None
        self._previous_steering = 0.0

    def reset(self, seed=None):

        _, info = self.env.reset(seed=seed)
        self._vehicle = self.env.agent
        self._previous_steering = 0.0
        observation = self.observation_adapter.observe(self._vehicle)

        return observation, info

    def step(self, action):

        steering = self._get_steering(action)
        _, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        self._previous_steering = steering
        observation = self.observation_adapter.observe(
            self._vehicle, previous_steering=self._previous_steering
        )

        return observation, reward, terminated, truncated, info

    @staticmethod
    def _get_steering(action):
        action_values = np.asarray(action, dtype=float).reshape(-1)
        if action_values.size == 0 or not np.isfinite(action_values[0]):
            raise ValueError("Action must contain a finite steering value.")
        return float(np.clip(action_values[0], -1.0, 1.0))

    @property
    def observation_space(self):
        return self.observation_adapter.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):

        self.env.close()
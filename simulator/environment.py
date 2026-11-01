from copy import deepcopy
from pathlib import Path

import yaml
from metadrive.envs.metadrive_env import MetaDriveEnv

from simulator.action_adapter import ContinuousActionAdapter
from simulator.observation_adapter import EightInputObservation
from simulator.reward_function import DrivingReward


class DrivingEnvironment:
    def __init__(self, render=False):
        config_directory = Path(__file__).resolve().parents[1] / "configs"
        environment_config = self._load_yaml(config_directory / "environment.yaml")
        observation_config = self._load_yaml(config_directory / "observation.yaml")
        reward_config = self._load_yaml(config_directory / "reward.yaml")
        if not isinstance(observation_config.get("normalization"), dict):
            raise ValueError("Observation configuration must define a normalization mapping.")

        self._env = MetaDriveEnv(config=self._build_metadrive_config(environment_config, render))
        self.observation_adapter = EightInputObservation(**observation_config["normalization"])
        self.action_adapter = ContinuousActionAdapter()
        self.reward_function = DrivingReward(reward_config)
        self._vehicle = None
        self._raw_observation = None

    def reset(self, seed=None):
        self._raw_observation, info = self._env.reset(seed=seed)
        self._vehicle = self._env.agent
        self.action_adapter.reset()
        self.reward_function.reset()
        observation = self.observation_adapter.observe(self._vehicle)
        return observation, info

    def step(self, action):
        adapted_action = self.action_adapter.adapt(action)
        self._raw_observation, metadrive_reward, terminated, truncated, info = self._env.step(adapted_action)
        reward, reward_components = self.reward_function.calculate(
            self._vehicle, steering=self.action_adapter.previous_steering
        )
        info["metadrive_reward"] = float(metadrive_reward)
        info["reward_components"] = reward_components
        observation = self.observation_adapter.observe(
            self._vehicle, previous_steering=self.action_adapter.previous_steering
        )
        return observation, reward, terminated, truncated, info

    @staticmethod
    def _load_yaml(path):
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must contain a YAML mapping: {path}")
        return config

    @staticmethod
    def _build_metadrive_config(environment_config, render):
        simulation = environment_config.get("simulation", {})
        road = environment_config.get("road", {})
        vehicle = environment_config.get("vehicle", {})
        traffic = environment_config.get("traffic", {})
        for section_name, section in (
            ("simulation", simulation),
            ("road", road),
            ("vehicle", vehicle),
            ("traffic", traffic),
        ):
            if not isinstance(section, dict):
                raise ValueError(f"Environment configuration section '{section_name}' must be a mapping.")

        return {
            "use_render": bool(render),
            "log_level": environment_config.get("log_level", 50),
            **simulation,
            "map": road.get("map", 3),
            "random_lane_width": road.get("random_lane_width", False),
            "random_lane_num": road.get("random_lane_num", False),
            "map_config": {
                "lane_width": road.get("lane_width", 3.5),
                "lane_num": road.get("lane_num", 3),
            },
            "vehicle_config": vehicle,
            "traffic_density": traffic.get("density", 0.0),
        }

    @property
    def vehicle(self):
        if self._vehicle is None:
            raise RuntimeError("Call reset() before accessing the active vehicle.")
        return self._vehicle

    @property
    def raw_observation(self):
        if self._raw_observation is None:
            raise RuntimeError("Call reset() before accessing the raw MetaDrive observation.")
        return deepcopy(self._raw_observation)

    @property
    def navigation(self):
        return self.vehicle.navigation

    @property
    def observation_space(self):
        return self.observation_adapter.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        self._env.close()
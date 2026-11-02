#!/usr/bin/env python3
"""Serve the visualizer and optionally stream a live MetaDrive episode."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlsplit

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.visualization import (  # noqa: E402
    build_networks,
    load_checkpoint,
    load_yaml,
    serialize_model,
)

VISUALIZER_ROOT = PROJECT_ROOT / "visualizer"
DRIVE_MODES = ("baseline", "actor", "manual")
CONTROL_TYPES = {"pause", "step", "reset", "mode", "drive"}


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class VisualizerRuntime:
    """Own the MetaDrive environment and the inference networks on the main thread."""

    def __init__(
        self,
        args: argparse.Namespace,
        network_config: dict,
        observation_config: dict,
        environment_config: dict,
    ):
        self.args = args
        self.network_config = network_config
        self.observation_config = observation_config
        self.environment_config = environment_config
        self.lock = threading.Lock()
        self.commands: deque[dict] = deque()
        self.stop_requested = False
        self.live = False
        self.error: Optional[str] = None
        self.actor = None
        self.critic = None
        self.env = None
        self.model_json: Optional[bytes] = None
        self.model_version = 0
        self.model_label = "untrained"
        self.checkpoint_path: Optional[Path] = args.checkpoint
        self.active_checkpoint: Optional[str] = None
        self.checkpoint_mtime: Optional[float] = None
        feature_order = list(observation_config.get("feature_order", []))
        self.observation_dim = len(feature_order) or 8
        self.obstacle_index = (
            feature_order.index("front_obstacle_distance")
            if "front_obstacle_distance" in feature_order
            else 5
        )
        self.obstacle_range_m = float(
            observation_config.get("normalization", {}).get("obstacle_distance_max_m", 50.0)
        )
        simulation = environment_config.get("simulation", {})
        self.scenario_start_index = int(simulation.get("start_seed", 0))
        self.scenario_count = max(1, int(simulation.get("num_scenarios", 1)))
        self.episode = 0
        self.step = 0
        self.observation = np.zeros(self.observation_dim, dtype=np.float32)
        self.snapshot: dict = {}
        self.paused = bool(args.paused)
        self.drive_mode = args.mode
        self.pending_steps = 0
        self.manual_action = (0.0, 0.0)
        self.last_manual_at = 0.0
        self.episode_reward = 0.0
        self.episode_ended_at: Optional[float] = None
        self.terminated = False
        self.truncated = False
        self.step_times: deque[float] = deque(maxlen=40)
        self.fps = 0.0
        self.seed = args.seed
        self.reset_count = 0
        self.started_at = time.time()
        self._heading_pid = None
        self._lateral_pid = None
        self._seq = 0
        self._checkpoint_mtime = self._current_checkpoint_mtime()

    def start(self) -> None:
        self._load_networks(initial=True)
        from simulator.environment import DrivingEnvironment

        self.env = DrivingEnvironment(render=False)
        self._reset_episode()
        self._checkpoint_mtime = self._current_checkpoint_mtime()
        with self.lock:
            self.live = True

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def submit_command(self, command: dict) -> None:
        with self.lock:
            self.commands.append(command)

    def health(self) -> dict:
        with self.lock:
            state = dict(self.snapshot)
            return {
                "live": self.live,
                "mode": self.drive_mode,
                "checkpoint": self.active_checkpoint,
                "hz": self.args.hz,
                "modelVersion": self.model_version,
                "label": self.model_label,
                "error": self.error,
                "pollIntervalMs": 80,
                "paused": self.paused,
                "episode": state.get("episode", self.episode),
                "step": state.get("step", self.step),
                "uptimeSeconds": max(0.0, time.time() - self.started_at),
            }

    def snapshot_copy(self) -> dict:
        with self.lock:
            return dict(self.snapshot)

    def model_payload(self) -> Optional[bytes]:
        with self.lock:
            return self.model_json

    def _current_checkpoint_mtime(self) -> Optional[float]:
        if self.checkpoint_path is None:
            return None
        try:
            return self.checkpoint_path.stat().st_mtime
        except OSError:
            return None

    def _load_networks(self, initial: bool = False) -> None:
        torch.manual_seed(self.seed)
        actor, critic = build_networks(self.network_config)
        loaded: list[str] = []
        warning = None
        model_label = self.args.label or "untrained"
        checkpoint = str(self.checkpoint_path) if self.checkpoint_path else None

        if self.checkpoint_path is not None:
            if self.checkpoint_path.exists():
                loaded = load_checkpoint(
                    actor,
                    critic,
                    self.checkpoint_path,
                    allow_unsafe_load=self.args.allow_unsafe_load,
                )
                model_label = self.args.label or self.checkpoint_path.stem
            else:
                warning = f"Checkpoint not found: {self.checkpoint_path}. Using untrained weights."
                checkpoint = None

        actor.eval()
        if critic is not None:
            critic.eval()

        model = serialize_model(
            actor,
            critic,
            self.observation_config,
            self.environment_config,
            label=model_label,
            checkpoint=checkpoint,
            seed=self.seed if checkpoint is None else None,
        )
        payload = json.dumps(model, separators=(",", ":"), allow_nan=False).encode("utf-8")

        with self.lock:
            self.actor = actor
            self.critic = critic
            self.model_json = payload
            self.model_version += 1
            self.model_label = model_label
            self.error = warning
            self.active_checkpoint = checkpoint
            snapshot = dict(self.snapshot)
            snapshot.update(
                {
                    "modelVersion": self.model_version,
                    "modelLabel": self.model_label,
                    "checkpoint": checkpoint,
                }
            )
            self.snapshot = snapshot

        if initial:
            if loaded:
                print(f"Loaded checkpoint weights for: {', '.join(loaded)}")
            if warning:
                print(warning, file=sys.stderr)

    def _maybe_reload_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        mtime = self._current_checkpoint_mtime()
        if mtime == self._checkpoint_mtime:
            return
        self._checkpoint_mtime = mtime
        if mtime is None:
            with self.lock:
                self.error = f"Checkpoint missing: {self.checkpoint_path}. Keeping current weights."
                snapshot = dict(self.snapshot)
                snapshot["error"] = self.error
                self.snapshot = snapshot
            return
        try:
            self._load_networks(initial=False)
            print(f"Reloaded checkpoint: {self.checkpoint_path}")
        except Exception as error:  # noqa: BLE001 - keep serving the previous model.
            message = f"Checkpoint reload failed: {error}"
            with self.lock:
                self.error = message
                snapshot = dict(self.snapshot)
                snapshot["error"] = message
                self.snapshot = snapshot
            print(message, file=sys.stderr)

    def _reset_episode(self) -> None:
        if self.env is None:
            return
        scenario = self.scenario_start_index + (self.seed + self.reset_count) % self.scenario_count
        self.reset_count += 1
        observation, _ = self.env.reset(seed=scenario)
        self.observation = np.asarray(observation, dtype=np.float32)
        self.episode += 1
        self.step = 0
        self.episode_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.episode_ended_at = None
        self.pending_steps = 0
        self.manual_action = (0.0, 0.0)
        self.last_manual_at = 0.0
        self.step_times.clear()
        self.fps = 0.0
        self._reset_baseline_pid()
        self._publish(applied_action=[0.0, 0.0], reward=0.0, reward_components={})

    def _reset_baseline_pid(self) -> None:
        if self.env is None:
            return
        from metadrive.component.vehicle.PID_controller import PIDController

        self._heading_pid = PIDController(1.7, 0.01, 3.5)
        self._lateral_pid = PIDController(0.3, 0.002, 0.05)

    def _policy_outputs(self) -> tuple[list[float], list[float], float]:
        if self.actor is None:
            return [], [], 0.0
        tensor = torch.as_tensor(self.observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = self.actor(tensor)[0]
            action = torch.tanh(mean)
            value = float(self.critic(tensor)[0]) if self.critic is not None else 0.0
        return (
            [float(item) for item in mean.tolist()],
            [float(item) for item in action.tolist()],
            value,
        )

    def _actor_action(self) -> list[float]:
        _, action, _ = self._policy_outputs()
        if not action:
            return [0.0, 0.0]
        return [float(np.clip(value, -1.0, 1.0)) for value in action]

    def _manual_action(self) -> list[float]:
        if time.monotonic() - self.last_manual_at > 0.75:
            return [0.0, 0.0]
        return [float(self.manual_action[0]), float(self.manual_action[1])]

    def _baseline_action(self) -> list[float]:
        if self.env is None or self._heading_pid is None or self._lateral_pid is None:
            return [0.0, 0.0]

        vehicle = self.env.vehicle
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            navigation = getattr(vehicle, "navigation", None)
            lanes = getattr(navigation, "current_ref_lanes", None)
            lane = lanes[0] if lanes else None
        steering = 0.0
        if lane is not None:
            try:
                longitudinal, lateral = lane.local_coordinates(vehicle.position)
                lane_heading = lane.heading_theta_at(longitudinal + 1.0)
                vehicle_heading = float(vehicle.heading_theta)
                steering = self._heading_pid.get_result(
                    -wrap_to_pi(lane_heading - vehicle_heading)
                )
                steering += self._lateral_pid.get_result(-float(lateral))
            except (AttributeError, TypeError, ValueError, IndexError):
                steering = 0.0

        target_speed_km_h = 30.0
        speed_km_h = max(float(getattr(vehicle, "speed_km_h", 0.0)), 0.0)
        acceleration = 1.0 * (1.0 - (speed_km_h / target_speed_km_h) ** 10)
        gap_normalized = float(self.observation[self.obstacle_index])
        if gap_normalized < 0.999:
            gap = gap_normalized * self.obstacle_range_m
            safe_gap = 10.0 + max(float(getattr(vehicle, "speed", 0.0)), 0.0) * 1.5
            if gap < safe_gap:
                acceleration = min(acceleration, -max(0.25, 1.0 - gap / safe_gap))

        return [
            float(np.clip(steering, -1.0, 1.0)),
            float(np.clip(acceleration, -1.0, 1.0)),
        ]

    def _select_action(self) -> list[float]:
        if self.drive_mode == "actor":
            return self._actor_action()
        if self.drive_mode == "manual":
            return self._manual_action()
        return self._baseline_action()

    def _publish(
        self,
        applied_action: list[float],
        reward: float,
        reward_components: dict,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        actor_mean, actor_action, critic_value = self._policy_outputs()
        now = time.monotonic()
        if self.step_times and now > self.step_times[-1]:
            self.fps = (len(self.step_times) - 1) / (now - self.step_times[0]) if len(self.step_times) > 1 else 0.0
        snapshot = {
            "seq": 0,
            "timestamp": time.time(),
            "episode": self.episode,
            "step": self.step,
            "observation": [float(value) for value in self.observation],
            "actorMean": actor_mean,
            "actorAction": actor_action,
            "criticValue": critic_value,
            "appliedAction": [float(value) for value in applied_action],
            "mode": self.drive_mode,
            "reward": float(reward),
            "episodeReward": float(self.episode_reward),
            "rewardComponents": {
                str(key): float(value) for key, value in reward_components.items()
            },
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "paused": self.paused,
            "fps": self.fps,
            "modelVersion": self.model_version,
            "modelLabel": self.model_label,
            "error": self.error,
        }
        with self.lock:
            self._seq += 1
            snapshot["seq"] = self._seq
            self.snapshot = snapshot

    def _advance(self) -> None:
        if self.env is None:
            return
        action = self._select_action()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.observation = np.asarray(observation, dtype=np.float32)
        self.step += 1
        self.episode_reward += float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.step_times.append(time.monotonic())
        self._publish(
            applied_action=action,
            reward=float(reward),
            reward_components=info.get("reward_components", {}),
            terminated=self.terminated,
            truncated=self.truncated,
        )
        if self.terminated or self.truncated:
            self.episode_ended_at = time.monotonic()

    def process_commands(self) -> None:
        while True:
            with self.lock:
                if not self.commands:
                    return
                command = self.commands.popleft()
            self._apply_command(command)

    def _apply_command(self, command: dict) -> None:
        kind = command.get("type")
        if kind == "pause":
            self.paused = bool(command.get("paused"))
            self.step_times.clear()
            self.fps = 0.0
            with self.lock:
                snapshot = dict(self.snapshot)
                snapshot["paused"] = self.paused
                snapshot["fps"] = self.fps
                self.snapshot = snapshot
        elif kind == "step":
            if self.paused:
                self.pending_steps = min(self.pending_steps + 1, 10)
        elif kind == "reset":
            self._reset_episode()
        elif kind == "mode":
            mode = command.get("mode")
            if mode in DRIVE_MODES:
                self.drive_mode = mode
                self.manual_action = (0.0, 0.0)
                self.last_manual_at = 0.0
                with self.lock:
                    snapshot = dict(self.snapshot)
                    snapshot["mode"] = self.drive_mode
                    self.snapshot = snapshot
        elif kind == "drive":
            if self.drive_mode != "manual":
                return
            steering = finite_number(command.get("steering"))
            throttle = finite_number(command.get("throttle"))
            if steering is None or throttle is None:
                return
            self.manual_action = (
                float(np.clip(steering, -1.0, 1.0)),
                float(np.clip(throttle, -1.0, 1.0)),
            )
            self.last_manual_at = time.monotonic()

    def run(self) -> None:
        interval = 1.0 / self.args.hz
        hold = self.args.episode_hold
        next_tick = time.perf_counter()
        while not self.stop_requested:
            self.process_commands()
            self._maybe_reload_checkpoint()

            now = time.monotonic()
            if self.episode_ended_at is not None and now - self.episode_ended_at >= hold:
                self._reset_episode()

            should_step = self.episode_ended_at is None and (
                not self.paused or self.pending_steps > 0
            )
            if should_step:
                self._advance()
                if self.pending_steps > 0:
                    self.pending_steps -= 1

            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    def stop(self) -> None:
        self.stop_requested = True


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    server_version = "DrivingVisualizer/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._handle_get(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_get(head=True)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path != "/api/control":
            self.close_connection = True
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        runtime = self.server.runtime  # type: ignore[attr-defined]
        if runtime is None or not runtime.live:
            self.close_connection = True
            self._send_json(HTTPStatus.CONFLICT, {"error": "Live mode is not enabled."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
            return
        if length <= 0 or length > 64 * 1024:
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request body size."})
            return
        try:
            command = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be JSON."})
            return
        error = self._validate_command(command)
        if error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": error})
            return
        runtime.submit_command(command)
        self._send_json(HTTPStatus.OK, {"ok": True, "queued": command.get("type")})

    def _handle_get(self, head: bool) -> None:
        path = urlsplit(self.path).path
        runtime = self.server.runtime  # type: ignore[attr-defined]

        if path == "/api/health":
            if runtime is None:
                payload = {"live": False, "mode": None}
            else:
                payload = runtime.health()
            self._send_json(HTTPStatus.OK, payload, head=head)
            return
        if path == "/api/state":
            if runtime is None or not runtime.live:
                self._send_json(HTTPStatus.CONFLICT, {"error": "Live mode is not enabled."}, head=head)
                return
            self._send_json(HTTPStatus.OK, runtime.snapshot_copy(), head=head)
            return
        if path == "/api/model":
            payload = runtime.model_payload() if runtime is not None else None
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "No live model."}, head=head)
                return
            self._send_bytes(
                HTTPStatus.OK,
                payload,
                "application/json; charset=utf-8",
                head=head,
            )
            return
        if path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."}, head=head)
            return
        if path == "/favicon.ico":
            self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon", head=head)
            return
        self._serve_static(path, head=head)

    def _serve_static(self, path: str, head: bool) -> None:
        static_root = self.server.static_root  # type: ignore[attr-defined]
        if path in {"", "/"}:
            path = "/index.html"
        try:
            target = (static_root / unquote(path).lstrip("/")).resolve()
        except (OSError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid path."}, head=head)
            return
        if target != static_root and static_root not in target.parents:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden."}, head=head)
            return
        if target.is_dir():
            target = (target / "index.html").resolve()
            if static_root not in target.parents:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden."}, head=head)
                return
        if not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."}, head=head)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix in {".js", ".mjs"}:
            content_type = "text/javascript; charset=utf-8"
        elif target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix == ".json":
            content_type = "application/json; charset=utf-8"
        elif target.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        try:
            body = target.read_bytes()
        except OSError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Could not read file."}, head=head)
            return
        self._send_bytes(HTTPStatus.OK, body, content_type, head=head)

    @staticmethod
    def _validate_command(command: Any) -> Optional[str]:
        if not isinstance(command, dict):
            return "Command must be a JSON object."
        kind = command.get("type")
        if kind not in CONTROL_TYPES:
            return f"Unknown command type: {kind!r}."
        if kind == "pause" and not isinstance(command.get("paused"), bool):
            return "pause requires a boolean paused field."
        if kind == "mode" and command.get("mode") not in DRIVE_MODES:
            return "mode must be one of: baseline, actor, manual."
        if kind == "drive":
            if finite_number(command.get("steering")) is None:
                return "drive requires a finite steering value."
            if finite_number(command.get("throttle")) is None:
                return "drive requires a finite throttle value."
        return None

    def _send_json(self, status: int, payload: dict, head: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", head=head)

    def _send_bytes(self, status: int, body: bytes, content_type: str, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, format_string: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(format_string, *args)


class VisualizerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run a live MetaDrive episode.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port (defaults to 8765; 0 chooses a free port).",
    )
    parser.add_argument("--mode", choices=DRIVE_MODES, default="baseline", help="Initial drive mode.")
    parser.add_argument("--paused", action="store_true", help="Start the simulation paused.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional actor/critic checkpoint.")
    parser.add_argument("--allow-unsafe-load", action="store_true", help="Trust non-weights-only checkpoints.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "network.yaml")
    parser.add_argument(
        "--observation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "observation.yaml",
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment.yaml",
    )
    parser.add_argument("--label", default=None, help="Override the model label.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for untrained weights and episodes.")
    parser.add_argument("--hz", type=float, default=10.0, help="Simulation steps per second.")
    parser.add_argument("--episode-hold", type=float, default=1.2, help="Seconds to show an ended episode.")
    parser.add_argument("--self-test", action="store_true", help="Run a short server and simulator check.")
    parser.add_argument("--steps", type=int, default=5, help="Steps to run during --self-test.")
    parser.add_argument("--verbose", action="store_true", help="Log HTTP requests.")
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be positive.")
    if args.steps < 0:
        parser.error("--steps cannot be negative.")
    return args


def request_json(url: str, payload: Optional[dict] = None) -> Any:
    import urllib.request

    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read())


def run_self_test(server: VisualizerHTTPServer, runtime: VisualizerRuntime, args: argparse.Namespace) -> None:
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, name="visualizer-http", daemon=True)
    thread.start()
    try:
        health = request_json(f"{base}/api/health")
        assert health["live"] is True, health
        model = request_json(f"{base}/api/model")
        assert len(model["networks"]) >= 1
        assert model["metadata"]["parameters"] > 0

        request_json(f"{base}/api/control", {"type": "mode", "mode": "baseline"})
        for index in range(args.steps):
            runtime.process_commands()
            if runtime.episode_ended_at is not None:
                runtime._reset_episode()
            runtime._advance()
            state = runtime.snapshot_copy()
            observation = np.asarray(state["observation"], dtype=np.float64)
            action = np.asarray(state["appliedAction"], dtype=np.float64)
            assert observation.shape == (runtime.observation_dim,), observation.shape
            assert np.all(np.isfinite(observation)), observation
            assert action.shape == (2,), action.shape
            assert np.all(np.isfinite(action)), action
            assert math.isfinite(state["criticValue"]), state["criticValue"]
            print(
                f"  step {index + 1:02d}: ep {state['episode']} step {state['step']} "
                f"reward={state['reward']:+.4f} V={state['criticValue']:+.4f} "
                f"action=({action[0]:+.3f}, {action[1]:+.3f})"
            )

        state = request_json(f"{base}/api/state")
        assert state["seq"] >= args.steps
        assert state["mode"] == "baseline"
        print(
            f"Self-test passed: {args.steps} steps, {health['hz']:.0f} Hz target, "
            f"model v{health['modelVersion']} ({model['metadata']['parameters']:,} parameters)."
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    args = parse_args()
    if not VISUALIZER_ROOT.is_dir():
        raise SystemExit(f"Visualizer directory not found: {VISUALIZER_ROOT}")

    runtime = None
    if args.live or args.self_test:
        network_config = load_yaml(args.config)
        observation_config = load_yaml(args.observation_config)
        environment_config = load_yaml(args.environment_config)
        runtime = VisualizerRuntime(args, network_config, observation_config, environment_config)
        try:
            runtime.start()
        except Exception as error:  # noqa: BLE001 - surface startup failures cleanly.
            runtime.close()
            raise SystemExit(f"Could not start the live simulator: {error}") from error

    port = args.port if args.port is not None else (0 if args.self_test else 8765)
    try:
        server = VisualizerHTTPServer((args.host, port), VisualizerRequestHandler)
    except OSError as error:
        if runtime is not None:
            runtime.close()
        raise SystemExit(f"Could not bind {args.host}:{port}: {error}") from error
    server.runtime = runtime  # type: ignore[attr-defined]
    server.static_root = VISUALIZER_ROOT.resolve()  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]
    url = f"http://{args.host}:{server.server_address[1]}/"

    try:
        if args.self_test:
            print(f"Self-testing live visualizer on {url}")
            try:
                run_self_test(server, runtime, args)
            finally:
                runtime.stop()
                runtime.close()
        elif runtime is not None:
            thread = threading.Thread(target=server.serve_forever, name="visualizer-http", daemon=True)
            thread.start()
            print(f"Live visualizer: {url}")
            print(f"  mode={args.mode} hz={args.hz:g} checkpoint={args.checkpoint or 'untrained'}")
            try:
                runtime.run()
            except KeyboardInterrupt:
                print("\nStopping live visualizer.")
            finally:
                runtime.stop()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                runtime.close()
        else:
            print(f"Static visualizer: {url}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nStopping visualizer.")
            finally:
                server.server_close()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

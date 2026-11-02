#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
REQUIREMENTS="$ROOT_DIR/requirements.txt"
SERVER="$ROOT_DIR/scripts/visualizer_server.py"
EXPORTER="$ROOT_DIR/scripts/export_visualizer.py"
MODEL="$ROOT_DIR/visualizer/models/latest.json"
MANIFEST="$ROOT_DIR/visualizer/models/manifest.json"
REQ_MARKER="$VENV_DIR/.visualise-requirements.sha256"

export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1

MODE=live
OPEN_BROWSER=1
FORCE_EXPORT=0
EXPORT_ONLY=0
REINSTALL=0
SELF_TEST=0
PORT=8765
HOST=127.0.0.1
SEED=0
CHECKPOINT=
LABEL=
NETWORK_CONFIG=
OBSERVATION_CONFIG=
ENVIRONMENT_CONFIG=
ALLOW_UNSAFE=0
PROBE=1
PASS_ARGS=()
SERVER_PID=

usage() {
  cat <<'EOF'
Usage: ./visualise.sh [wrapper options] [server options]

Bootstraps .venv and requirements, exports a model snapshot when needed, then
serves the visualizer. By default it runs the live MetaDrive simulator in
baseline mode and opens the browser.

Wrapper options:
  --static          Serve the static explorer without MetaDrive
  --no-open         Do not open a browser
  --force-export    Regenerate the model snapshot before starting
  --export-only     Regenerate the model snapshot and exit
  --reinstall       Reinstall Python requirements into .venv
  -h, --help        Show this help

Server options passed through:
  --mode baseline|actor|manual
  --checkpoint PATH
  --port PORT
  --host HOST
  --hz HZ
  --paused
  --seed N
  --label TEXT
  --allow-unsafe-load
  --self-test
  --steps N
  --verbose

Examples:
  ./visualise.sh
  ./visualise.sh --static
  ./visualise.sh --mode manual
  ./visualise.sh --checkpoint checkpoints/actor.pt --mode actor
  ./visualise.sh --self-test

Set VISUALISE_PYTHON to choose the interpreter used to create .venv if needed.
EOF
}

while [ $# -gt 0 ]; do
  arg=$1
  shift
  case "$arg" in
    --static)
      MODE=static
      ;;
    --live)
      MODE=live
      ;;
    --no-open)
      OPEN_BROWSER=0
      ;;
    --force-export)
      FORCE_EXPORT=1
      ;;
    --export-only)
      EXPORT_ONLY=1
      FORCE_EXPORT=1
      ;;
    --reinstall)
      REINSTALL=1
      ;;
    --self-test)
      SELF_TEST=1
      PASS_ARGS+=("$arg")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --port|--host|--seed|--checkpoint|--label|--config|--observation-config|--environment-config)
      if [ $# -eq 0 ]; then
        printf 'error: %s requires a value\n' "$arg" >&2
        exit 2
      fi
      value=$1
      shift
      PASS_ARGS+=("$arg" "$value")
      case "$arg" in
        --port) PORT=$value ;;
        --host) HOST=$value ;;
        --seed) SEED=$value ;;
        --checkpoint) CHECKPOINT=$value ;;
        --label) LABEL=$value ;;
        --config) NETWORK_CONFIG=$value ;;
        --observation-config) OBSERVATION_CONFIG=$value ;;
        --environment-config) ENVIRONMENT_CONFIG=$value ;;
      esac
      ;;
    --port=*|--host=*|--seed=*|--checkpoint=*|--label=*|--config=*|--observation-config=*|--environment-config=*)
      PASS_ARGS+=("$arg")
      value=${arg#*=}
      case "${arg%%=*}" in
        --port) PORT=$value ;;
        --host) HOST=$value ;;
        --seed) SEED=$value ;;
        --checkpoint) CHECKPOINT=$value ;;
        --label) LABEL=$value ;;
        --config) NETWORK_CONFIG=$value ;;
        --observation-config) OBSERVATION_CONFIG=$value ;;
        --environment-config) ENVIRONMENT_CONFIG=$value ;;
      esac
      ;;
    --allow-unsafe-load)
      ALLOW_UNSAFE=1
      PASS_ARGS+=("$arg")
      ;;
    *)
      PASS_ARGS+=("$arg")
      ;;
  esac
done

ensure_venv() {
  if [ -x "$PYTHON" ]; then
    :
  elif [ -e "$VENV_DIR" ]; then
    printf 'error: %s exists but has no Python interpreter.\n' "$VENV_DIR" >&2
    printf "Remove it and rerun: rm -rf '%s'\n" "$VENV_DIR" >&2
    exit 1
  else
    base=${VISUALISE_PYTHON:-}
    if [ -z "$base" ]; then
      for candidate in python3.10 python3.11 python3.12 python3.13 python3 python; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
          continue
        fi
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
          base=$candidate
          break
        fi
      done
    fi
    if [ -z "$base" ]; then
      printf 'error: Python 3.10 or newer is required, but a suitable interpreter was not found.\n' >&2
      exit 1
    fi
    if ! "$base" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
      printf 'error: Python 3.10 or newer is required (found %s).\n' "$("$base" -V 2>&1)" >&2
      exit 1
    fi
    printf 'Creating virtual environment in %s\n' "$VENV_DIR"
    if ! "$base" -m venv "$VENV_DIR"; then
      printf 'error: could not create %s\n' "$VENV_DIR" >&2
      exit 1
    fi
  fi

  if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    printf 'error: %s must use Python 3.10 or newer (found %s).\n' "$VENV_DIR" "$("$PYTHON" -V 2>&1)" >&2
    exit 1
  fi
}

requirements_hash() {
  "$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$REQUIREMENTS"
}

requirements_ready() {
  "$PYTHON" - "$REQUIREMENTS" <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

missing = []
for raw in Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    expected = None
    if line.startswith("-e "):
        if "#egg=" not in line:
            continue
        name = line.split("#egg=", 1)[1].split("&", 1)[0]
    elif line.startswith("-"):
        continue
    else:
        name = re.split(r"[<>=!~;\[ ]", line, 1)[0]
        if "==" in line:
            expected = line.split("==", 1)[1].split(";", 1)[0].strip()
    if not name:
        continue
    try:
        installed = version(name)
    except PackageNotFoundError:
        missing.append(name)
        continue
    if expected is not None and installed != expected:
        missing.append(f"{name}=={expected} (found {installed})")

if missing:
    print("Missing: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
}

ensure_dependencies() {
  local current_hash need_install=0
  current_hash=$(requirements_hash)

  if [ "$REINSTALL" = 1 ]; then
    need_install=1
  elif [ ! -f "$REQ_MARKER" ]; then
    if ! requirements_ready; then
      need_install=1
    fi
  elif [ "$current_hash" != "$(cat "$REQ_MARKER")" ]; then
    need_install=1
  elif ! requirements_ready; then
    need_install=1
  fi

  if [ "$need_install" = 0 ]; then
    if [ ! -f "$REQ_MARKER" ]; then
      printf '%s\n' "$current_hash" > "$REQ_MARKER"
    fi
    return
  fi

  if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    if ! "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
      printf 'error: pip is not available in %s\n' "$VENV_DIR" >&2
      exit 1
    fi
  fi

  printf 'Installing Python dependencies (the first run can take several minutes)...\n'
  if [ "$REINSTALL" = 1 ]; then
    "$PYTHON" -m pip install --disable-pip-version-check --no-input --upgrade -r "$REQUIREMENTS"
  else
    "$PYTHON" -m pip install --disable-pip-version-check --no-input -r "$REQUIREMENTS"
  fi

  if ! requirements_ready; then
    printf 'error: dependencies are still missing after installation.\n' >&2
    exit 1
  fi
  printf '%s\n' "$current_hash" > "$REQ_MARKER"
}

model_readable() {
  "$PYTHON" - "$MODEL" "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    model = json.load(file)
networks = model.get("networks")
if not isinstance(networks, dict) or not networks:
    raise SystemExit(1)
with open(sys.argv[2], encoding="utf-8") as file:
    json.load(file)
PY
}

needs_export() {
  local source
  if [ "$FORCE_EXPORT" = 1 ]; then
    return 0
  fi
  if [ -n "$CHECKPOINT" ] || [ -n "$LABEL" ] || [ "$SEED" != 0 ] || [ -n "$NETWORK_CONFIG" ] || [ -n "$OBSERVATION_CONFIG" ] || [ -n "$ENVIRONMENT_CONFIG" ]; then
    return 0
  fi
  if ! model_readable >/dev/null 2>&1; then
    return 0
  fi
  for source in \
    "$ROOT_DIR/configs/network.yaml" \
    "$ROOT_DIR/configs/observation.yaml" \
    "$ROOT_DIR/configs/environment.yaml" \
    "$ROOT_DIR/agent/networks/actor.py" \
    "$ROOT_DIR/agent/networks/critic.py" \
    "$ROOT_DIR/agent/networks/layers.py" \
    "$ROOT_DIR/agent/networks/__init__.py" \
    "$ROOT_DIR/evaluation/visualization.py" \
    "$ROOT_DIR/scripts/export_visualizer.py"; do
    if [ -e "$source" ] && [ "$source" -nt "$MODEL" ]; then
      return 0
    fi
  done
  for source in "$CHECKPOINT" "$NETWORK_CONFIG" "$OBSERVATION_CONFIG" "$ENVIRONMENT_CONFIG"; do
    if [ -n "$source" ] && [ -e "$source" ] && [ "$source" -nt "$MODEL" ]; then
      return 0
    fi
  done
  return 1
}

export_model() {
  local args=(--seed "$SEED")
  if [ -n "$CHECKPOINT" ]; then
    if [ -f "$CHECKPOINT" ]; then
      args+=(--checkpoint "$CHECKPOINT")
      if [ "$ALLOW_UNSAFE" = 1 ]; then
        args+=(--allow-unsafe-load)
      fi
    else
      printf 'warning: checkpoint not found: %s (exporting untrained weights)\n' "$CHECKPOINT" >&2
    fi
  fi
  if [ -n "$LABEL" ]; then
    args+=(--label "$LABEL")
  fi
  if [ -n "$NETWORK_CONFIG" ]; then
    args+=(--config "$NETWORK_CONFIG")
  fi
  if [ -n "$OBSERVATION_CONFIG" ]; then
    args+=(--observation-config "$OBSERVATION_CONFIG")
  fi
  if [ -n "$ENVIRONMENT_CONFIG" ]; then
    args+=(--environment-config "$ENVIRONMENT_CONFIG")
  fi

  mkdir -p "$(dirname "$MODEL")"
  printf 'Exporting model snapshot...\n'
  "$PYTHON" "$EXPORTER" "${args[@]}"
}

http_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "$URL/api/health" >/dev/null 2>&1
  else
    "$PYTHON" - "$URL/api/health" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    response.read()
PY
  fi
}

open_url() {
  case "$(uname -s)" in
    Darwin)
      command -v open >/dev/null 2>&1 && open "$1"
      ;;
    Linux)
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$1"
      elif command -v sensible-browser >/dev/null 2>&1; then
        sensible-browser "$1"
      else
        return 1
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

cleanup() {
  trap - EXIT INT TERM HUP
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

ensure_venv
ensure_dependencies

if [ "$SELF_TEST" = 0 ] || [ "$FORCE_EXPORT" = 1 ]; then
  if needs_export; then
    export_model
  else
    printf 'Model snapshot is up to date.\n'
  fi
fi

if [ "$EXPORT_ONLY" = 1 ]; then
  printf 'Model snapshot ready: %s\n' "$MODEL"
  exit 0
fi

if [ "$SELF_TEST" = 1 ] || [ -n "${SSH_CONNECTION:-}" ]; then
  OPEN_BROWSER=0
fi
case "$PORT" in
  ''|*[!0-9]*|0)
    OPEN_BROWSER=0
    PROBE=0
    ;;
esac
case "$HOST" in
  0.0.0.0|'')
    BROWSER_HOST=127.0.0.1
    ;;
  ::)
    BROWSER_HOST="[::1]"
    ;;
  *:*)
    BROWSER_HOST="[$HOST]"
    ;;
  *)
    BROWSER_HOST="$HOST"
    ;;
esac
URL="http://$BROWSER_HOST:$PORT/"

SERVER_ARGS=("${PASS_ARGS[@]}")
if [ "$MODE" = live ]; then
  SERVER_ARGS+=(--live)
fi

printf 'Starting %s visualizer...\n' "$MODE"
"$PYTHON" "$SERVER" "${SERVER_ARGS[@]}" &
SERVER_PID=$!
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 129' HUP

if [ "$SELF_TEST" = 1 ]; then
  if wait "$SERVER_PID"; then
    STATUS=0
  else
    STATUS=$?
  fi
  trap - EXIT INT TERM HUP
  exit "$STATUS"
fi

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if [ "$PROBE" = 1 ]; then
    if http_ready; then
      READY=1
      break
    fi
  else
    READY=1
    break
  fi
  sleep 0.5
done

if [ "$READY" = 0 ]; then
  printf 'error: visualizer did not become ready.\n' >&2
  cleanup
  exit 1
fi

if [ "$OPEN_BROWSER" = 1 ]; then
  if ! open_url "$URL"; then
    printf 'Open %s manually.\n' "$URL"
  fi
fi

if wait "$SERVER_PID"; then
  STATUS=0
else
  STATUS=$?
fi
trap - EXIT INT TERM HUP
exit "$STATUS"

#!/usr/bin/env bash
# One-command setup for VoxPipe.
#
# Creates the venv, installs the pinned dependencies, and downloads the models.
# Safe to re-run: anything already present is left alone.
#
# Usage:
#   ./install.sh                # default: fetch the turbo model too (548 MB)
#   ./install.sh small-q5_1     # a smaller, faster, lower-quality model (170 MB)
#   ./install.sh --no-speech-model
#                               # VAD + diarization only (34 MB), no speech model
#
# Then: ./run.sh
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$1" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# Opt-in speech model, defaulting to turbo. "none" means VAD/diarization only.
speech_model="turbo"
for arg in "$@"; do
  case "$arg" in
    --no-speech-model) speech_model="none" ;;
    # Print the comment header, stopping at the first non-comment line, so
    # editing the header above cannot desync a hardcoded line range.
    -h|--help) sed -n '2,${/^#/!q;s/^# \{0,1\}//p;}' "$0"; exit 0 ;;
    -*) die "unknown option: $arg (try --help)" ;;
    *) speech_model="$arg" ;;
  esac
done

step "Checking prerequisites"
for tool in python3 curl tar; do
  command -v "$tool" > /dev/null || die "$tool is required but not installed"
done
# 3.8 is the floor pyproject declares; 3.9+ is what the lock was resolved on.
python3 - <<'PY' || die "python 3.9+ is required"
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
echo "  python3 $(python3 -c 'import platform; print(platform.python_version())')"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /' \
  || warn "no nvidia-smi found; VoxPipe runs on CPU but will be much slower"

step "Creating the virtualenv"
if [[ -x .venv/bin/python ]]; then
  echo "  .venv already exists, keeping it"
else
  # Never "repair" a venv by patching it: shebangs and the editable .pth hardcode
  # absolute paths, so a moved checkout needs a fresh one. Deleting here is
  # deliberate, but only ever a half-built one (no usable python).
  rm -rf .venv
  python3 -m venv .venv
  echo "  created .venv"
fi

step "Installing dependencies"
# Lower bounds in pyproject, exact resolved versions in requirements.lock.
# The lock is a constraint, not a pin: install with -c so the tree is
# reproducible, but an already-populated .venv is not disturbed by a rerun.
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[dev]" -c requirements.lock
echo "  voxpipe $(.venv/bin/python -c 'import voxpipe; print(voxpipe.__version__)')"

step "Fetching models"
if [[ "$speech_model" == "none" ]]; then
  ./scripts/fetch-models.sh
  warn "no speech model installed; set TRANSCRIBER_MODEL or pass one to fetch-models.sh before transcribing"
else
  ./scripts/fetch-models.sh "$speech_model"
fi

step "Checking for whisper-server"
if [[ -x vendor/bin/whisper-server ]]; then
  echo "  vendor/bin/whisper-server present"
else
  warn "vendor/bin/whisper-server is missing, so transcription cannot start yet."
  cat >&2 <<'EOF'

  It is not downloaded: a Vulkan build of whisper.cpp is compiled locally, so
  there is no artifact to fetch. Build it once, then vendor the copy:

    git clone --depth 1 https://github.com/ggml-org/whisper.cpp
    cmake -S whisper.cpp -B whisper.cpp/build -DWHISPER_BUILD_SERVER=ON \
      -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build whisper.cpp/build --config Release -j"$(nproc)"
    pip install patchelf
    ./scripts/vendor-server.sh whisper.cpp/build/bin

  A whisper-server already on PATH also works, as long as nothing else is
  using it. See README.md for the single-GPU rule.
EOF
fi

step "Creating the working folders"
for dir in media-inbox media-process media-results media-failed; do
  mkdir -p "$dir"
  [[ -f "$dir/.gitkeep" ]] || touch "$dir/.gitkeep"
done
echo "  drop recordings into media-inbox/"

printf '\n\033[1;32mSetup finished.\033[0m  Start the watcher with:\n\n    ./run.sh\n\n'

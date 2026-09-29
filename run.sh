#!/usr/bin/env bash
# Start the folder watcher: drop a recording into media-inbox/ and the
# transcript appears in media-results/.
#
# Usage:
#   ./run.sh                  # watch, writing md + txt + html
#   ./run.sh --once           # drain the inbox and exit (good for cron)
#   ./run.sh --no-diarization
#
# Anything you pass is handed to `voxpipe watch`, so every flag works:
#   ./run.sh --model small-q5_1 --language en --layout sentence
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

die() { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# A venv that will not start is almost always one whose checkout moved, not a
# broken install. Recreate it rather than trying to patch absolute paths.
[[ -x .venv/bin/python ]] || die ".venv is missing. Run ./install.sh first."

# shellcheck disable=SC1091
source .venv/bin/activate

for dir in media-inbox media-process media-results media-failed; do
  mkdir -p "$dir"
done

printf '\033[1mWatching media-inbox/\033[0m  transcripts land in media-results/\n'
printf '\033[2mCtrl-C to stop.\033[0m\n\n'

exec voxpipe watch --format md,txt,html "$@"

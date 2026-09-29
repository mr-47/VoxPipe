#!/usr/bin/env bash
# Fetch the models VoxPipe needs into vendor/models/.
#
# Only the small, always-required files are downloaded here: the Silero VAD
# model and the two ONNX diarization models (34 MB total). The GGML speech
# models are 180-550 MB each, so they are opt-in:
#
#   scripts/fetch-models.sh            # VAD + diarization only
#   scripts/fetch-models.sh turbo      # + the default GGML model (548 MB)
#   scripts/fetch-models.sh turbo small-q5_1
#
# Re-running is cheap: files that already exist are left alone.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$PROJECT/vendor/models"
HF="https://huggingface.co"
SHERPA="https://github.com/k2-fsa/sherpa-onnx/releases/download"

# name -> URL, for the always-needed small files.
declare -A SMALL_FILES=(
  ["ggml-silero-v5.1.2.bin"]="$HF/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"
  ["diar/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"]="$SHERPA/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  ["diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"]="$SHERPA/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2::tar:xj:$MODELS/diar/sherpa-onnx-pyannote-segmentation-3-0"
)

# Opt-in GGML speech models: alias -> filename.
declare -A GGML_FILES=(
  [turbo]="ggml-large-v3-turbo-q5_0.bin"
  [large-v3-turbo-q5_0]="ggml-large-v3-turbo-q5_0.bin"
  [small]="ggml-small.bin"
  [small-q5_1]="ggml-small-q5_1.bin"
  [base]="ggml-base.bin"
  [base-q5_1]="ggml-base-q5_1.bin"
)

fetch() {
  # fetch <url> <destination>
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  have $(basename "$dest")"
    return
  fi
  echo "  get  $(basename "$dest")"
  mkdir -p "$(dirname "$dest")"
  curl -fL --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

mkdir -p "$MODELS/diar/sherpa-onnx-pyannote-segmentation-3-0"

echo "Small models (VAD + diarization):"
for rel in "${!SMALL_FILES[@]}"; do
  spec="${SMALL_FILES[$rel]}"
  dest="$MODELS/$rel"
  if [[ "$spec" == *"::tar:xj:"* ]]; then
    # sherpa-onnx ships the segmentation model inside a tarball, which unpacks
    # into a single top-level directory, so the file is not at $tmp/model.onnx.
    url="${spec%%::tar:xj:*}"
    target="${spec##*::tar:xj:}"
    if [[ -f "$dest" ]]; then
      echo "  have $(basename "$dest")"
      continue
    fi
    echo "  get  $(basename "$dest") (from tarball)"
    tmp="$(mktemp -d)"
    curl -fL --progress-bar -o "$tmp/pkg.tar.bz2" "$url"
    tar -xjf "$tmp/pkg.tar.bz2" -C "$tmp"
    # Exact name, so the int8 variant alongside it is not picked up.
    extracted="$(find "$tmp" -type f -name model.onnx -print -quit)"
    if [[ -z "$extracted" ]]; then
      echo "error: no model.onnx inside $url" >&2
      rm -rf "$tmp"
      exit 1
    fi
    cp "$extracted" "$dest"
    # Optional: not every release tarball ships a LICENSE.
    license="$(find "$tmp" -type f -name LICENSE -print -quit)"
    if [[ -n "$license" ]]; then
      cp "$license" "$target/LICENSE"
    fi
    rm -rf "$tmp"
  else
    fetch "$spec" "$dest"
  fi
done

if [[ $# -gt 0 ]]; then
  echo "GGML speech models:"
  for alias in "$@"; do
    file="${GGML_FILES[$alias]:-$alias}"
    fetch "$HF/ggerganov/whisper.cpp/resolve/main/$file" "$MODELS/$file"
  done
else
  echo
  echo "No GGML speech model downloaded. Pass one or more aliases, e.g.:"
  echo "  scripts/fetch-models.sh turbo"
fi

echo
echo "Done. The app searches vendor/models first, so nothing else is needed."

# The model weights are separate works under their own licenses. vendor/ is
# git-ignored, so keep the authoritative notices reachable from a fetched tree
# too; the tracked copy lives at the project root.
if [[ -f "$PROJECT/THIRD-PARTY-NOTICES.md" ]]; then
  cp "$PROJECT/THIRD-PARTY-NOTICES.md" "$MODELS/THIRD-PARTY-NOTICES.md"
  echo "Copied THIRD-PARTY-NOTICES.md next to the weights (licensing details)."
fi

du -sh "$MODELS"

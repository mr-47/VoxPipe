# Configuration

The standard repository layout requires no environment variables.

## Environment variables

| Variable | Default | Description |
|---|---:|---|
| `TRANSCRIBER_MODEL` | `turbo` | `turbo`, `small`, `small-q5_1`, `base`, `base-q5_1`, or a GGML path |
| `TRANSCRIBER_MODEL_DIRS` | empty | Extra model directories, searched before defaults |
| `TRANSCRIBER_SERVER_BIN` | auto-discovered | Path to `whisper-server` |
| `TRANSCRIBER_THREADS` | `min(8, cpus)` | whisper.cpp threads (`-t`) |
| `TRANSCRIBER_DEVICE` | `0` | Vulkan device index (`-dev`) |
| `TRANSCRIBER_NO_GPU` | `false` | Force CPU mode (`-ng`) |
| `TRANSCRIBER_PORT` | `8099` | Internal server port |
| `TRANSCRIBER_HOST` | `127.0.0.1` | Internal server bind address |
| `TRANSCRIBER_USE_VAD` | `true` | Pass VAD to whisper.cpp when model exists |
| `TRANSCRIBER_VAD_MODEL` | `ggml-silero-v5.1.2.bin` | VAD filename |
| `TRANSCRIBER_DIAR_SEG_MODEL` | `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | Segmentation model path under `<models>/diar/` |
| `TRANSCRIBER_DIAR_EMBED_MODEL` | `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | Embedding model path under `<models>/diar/` |
| `TRANSCRIBER_DIAR_THRESHOLD` | `0.85` | Clustering threshold; higher values merge more aggressively and normally produce fewer speakers |
| `TRANSCRIBER_DIAR_SPEAKERS` | `-1` | Fixed speaker count; currently unreliable, leave at `-1` |
| `TRANSCRIBER_DIAR_MIN_ON` | `0.5` | Minimum speaker segment length, seconds |
| `TRANSCRIBER_DIAR_MIN_OFF` | `0.7` | Minimum silence that closes a segment, seconds |
| `TRANSCRIBER_DIAR_MERGE_THRESHOLD` | `0.70` | Post-pass centroid similarity threshold; `0` disables |
| `TRANSCRIBER_DIAR_MIN_REGION` | `1.0` | Minimum speech assigned to a dedicated cluster; `0` disables |
| `TRANSCRIBER_DIAR_THREADS` | `min(8, cpus)` | ONNX embedding threads |
| `TRANSCRIBER_TIMEOUT` | `3600` | HTTP timeout per transcription request, seconds |
| `TRANSCRIBER_STARTUP_TIMEOUT` | `180` | Time to wait for server `/health`, seconds |
| `TRANSCRIBER_REUSE_SERVER` | `0` | Allow reuse of a server not started by this VoxPipe process |
| `TRANSCRIBER_SERVER_STATE_DIR` | `~/.cache/voxpipe/servers` | State used to identify/reclaim managed servers |

Example:

```bash
TRANSCRIBER_MODEL=small-q5_1 \
TRANSCRIBER_THREADS=4 \
voxpipe meeting.mp3
```

## Model lookup

Search order:

```text
TRANSCRIBER_MODEL_DIRS
vendor/models/
~/.cache/voxpipe/models/
```

`TRANSCRIBER_MODEL_DIRS` uses the operating system path separator.

## Important diarization semantics

`TRANSCRIBER_DIAR_THRESHOLD` behaves counterintuitively if read as a generic distance threshold: in the current sherpa-onnx clustering behavior, raising the threshold merges speakers more aggressively and normally returns fewer clusters.

Measured on the original 156-second two-speaker call:

| Threshold | Speakers |
|---|---:|
| `0.60` | 5 |
| `0.85` | 2 |
| `0.95` | 1 |

See [diarization.md](diarization.md) for the full experiments.

## Naming note

The product is VoxPipe, but configuration variables still use the legacy `TRANSCRIBER_*` prefix.

Do not rename them only in documentation.

A future code change may introduce `VOXPIPE_*` variables while retaining `TRANSCRIBER_*` aliases for compatibility.

# Migrating from the faster-whisper Transcriber

VoxPipe began as a drop-in replacement for the previous faster-whisper + pyannote `Transcriber`.

The folder workflow, many flags and output files were intentionally kept similar, but several compatibility changes matter.

## Important behavior changes

| Difference | Detail |
|---|---|
| Command renamed | `transcriber` -> `voxpipe` |
| API module renamed | `transcriber.api:app` -> `voxpipe.api:app` |
| Folder names changed | `calls-*` -> `media-*` |
| Default model changed | `small` -> `turbo` |
| Model cache changed | `~/.cache/transcriber/` -> `~/.cache/voxpipe/` |
| Diarization backend changed | pyannote 3.1 -> sherpa-onnx |

Existing files left in the old `calls-inbox/` are not picked up automatically.

Move them into `media-inbox/` before switching workflows.

## Environment-variable migration

| Old variable | New variable | Notes |
|---|---|---|
| `TRANSCRIBER_WHISPER_MODEL` | `TRANSCRIBER_MODEL` | Default also changed from `small` to `turbo` |
| `TRANSCRIBER_WHISPER_DEVICE` | `TRANSCRIBER_DEVICE` | `cuda`/`auto` became Vulkan device index such as `0`, or use `TRANSCRIBER_NO_GPU=1` |
| `TRANSCRIBER_WHISPER_COMPUTE_TYPE` | removed | Quantization is encoded in the selected GGML model file |
| `TRANSCRIBER_HF_TOKEN` | removed | No longer needed |
| `TRANSCRIBER_DIARIZATION_MODEL` | `TRANSCRIBER_DIAR_SEG_MODEL` | Now a local ONNX path, not a Hub model ID |

An old environment containing:

```text
TRANSCRIBER_WHISPER_MODEL=small
```

does not configure VoxPipe.

Use:

```text
TRANSCRIBER_MODEL=small
```

instead.

## Diarization migration

Speaker counts will not exactly match the pyannote implementation.

`TRANSCRIBER_DIAR_THRESHOLD` is a new tuning control.

The original experiments showed that `0.85` worked for a short two-party call but did not directly generalize to a long multi-party meeting.

VoxPipe therefore added centroid-merge and minimum-region cleanup passes.

See [diarization.md](diarization.md).

## API migration

Install the optional API extra:

```bash
pip install -e ".[api]"
```

Start it with:

```bash
voxpipe serve
```

or:

```bash
.venv/bin/python -m uvicorn voxpipe.api:app --host 127.0.0.1 --port 8000
```

The endpoints remain:

```text
GET /health
POST /transcribe
```

The response JSON remains compatible, while VoxPipe adds:

- a `words=false` query option;
- richer health information;
- health response before model load;
- `model_verified`;
- complete `text` in the transcript response.

# VoxPipe

Audio transcription with speaker diarization, built on
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) for speech-to-text and
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for speaker diarization.

VoxPipe runs locally, supports Vulkan GPU acceleration, requires no Hugging Face
account or token, and exposes the same transcription pipeline through:

- a **CLI** for one-shot transcription;
- a **folder watcher** for automatic batch processing;
- a **REST API** for integrations.

```bash
voxpipe meeting.mp3
voxpipe watch
voxpipe serve
```

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)](#platform-support)

## Why VoxPipe?

VoxPipe started as a replacement for a `faster-whisper + pyannote` transcription
stack in `Transcriber` project, with the goal of keeping the same practical workflow while reducing the
runtime footprint and external requirements: **no torch, no CUDA wheels, no Hugging Face
token, no 4 GB Python install.**

| | Previous stack | VoxPipe |
|---|---|---|
| Speech recognition | faster-whisper / CTranslate2 | whisper.cpp |
| GPU runtime | CUDA | Vulkan |
| Diarization | pyannote | sherpa-onnx |
| Hugging Face token | Required for diarization | Not required |
| Python environment | ~4 GB | ~235 MB |
| M4A / AAC decoding | Limited | Supported via PyAV |

The original ASR speed on the reference GTX 1070 test system was roughly the
same as faster-whisper; the main gains are a much smaller Python environment,
no gated diarization models, Vulkan instead of CUDA-specific runtime
dependencies, and local speaker labels.

## Features

- Local speech-to-text with whisper.cpp
- Speaker diarization with sherpa-onnx
- Vulkan GPU acceleration
- CPU fallback
- Automatic language detection
- Word-level and segment-level timestamps
- CLI transcription
- Automatic folder monitoring
- REST API
- Markdown, TXT, HTML, SRT and JSON output
- `.mp3`, `.wav`, `.m4a`, `.ogg` and `.webm` input
- Long-lived `whisper-server` process to avoid reloading the model between files
- Linux and Windows support
- No PyTorch, CUDA runtime or Hugging Face token required

## Quick start

### Linux

Requirements:

- Python 3.12+
- Linux
- Vulkan-capable GPU recommended; CPU mode is supported

Clone the repository:

```bash
git clone https://github.com/mr-47/VoxPipe.git
cd VoxPipe
```

Install:

```bash
./install.sh
```

Start the folder watcher:

```bash
./run.sh
```

Or activate the environment and transcribe a file directly:

```bash
source .venv/bin/activate
voxpipe meeting.mp3
```

### Windows

Requirements:

- Python 3.12+
- Windows
- Visual C++ Redistributable

Install and run:

```bat
install.cmd
run.cmd
```

CPU transcription works with the official whisper.cpp Windows build.

Vulkan GPU acceleration is available, but Windows GPU support has not yet been
verified end-to-end on real Windows Vulkan hardware. Because whisper.cpp does
not publish an official x64 Windows Vulkan `whisper-server` archive, VoxPipe can
use pinned third-party Vulkan builds or a self-built server.

See [Windows support](docs/windows.md) for details.

## Usage

### Transcribe a file

```bash
voxpipe meeting.mp3
```

Markdown is written to stdout by default.

Write the transcript to a file:

```bash
voxpipe meeting.mp3 -o meeting.md
```

Other output formats:

```bash
voxpipe meeting.mp3 -o meeting.txt
voxpipe meeting.mp3 -o meeting.html
voxpipe meeting.mp3 -o meeting.srt
```

Generate machine-readable JSON as well:

```bash
voxpipe meeting.mp3 \
  -o meeting.md \
  --json meeting.json \
  --words
```

Force a language:

```bash
voxpipe meeting.mp3 --language en
```

Disable speaker diarization:

```bash
voxpipe meeting.mp3 --no-diarization
```

Use another model:

```bash
voxpipe meeting.mp3 --model small-q5_1
```

### Folder watcher

VoxPipe can continuously monitor a folder and automatically process incoming
media:

```bash
voxpipe watch
```

Default workflow:

```text
media-inbox/
      ↓
media-process/
      ↓
media-results/
```

Failed files are moved to:

```text
media-failed/
```

Useful examples:

```bash
# Run continuously
voxpipe watch

# Process the current inbox once and exit
voxpipe watch --once

# Use another base directory
voxpipe watch --dir /data/audio

# Poll every second
voxpipe watch --interval 1

# Use a smaller model
voxpipe watch --model small-q5_1

# Disable diarization
voxpipe watch --no-diarization
```

Generate multiple transcript formats:

```bash
voxpipe watch --once --format txt,md,html,srt
```

The watcher handles files that are still being written, crash recovery,
filename collisions and failed inputs without silently dropping files.

See [Folder workflow](docs/folder-workflow.md) for the exact behavior.

### REST API

Install API support:

```bash
pip install -e ".[api]"
```

Start the server:

```bash
voxpipe serve
```

Default address:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | Service and model status |
| `POST /transcribe` | Upload and transcribe a media file |

Example:

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@meeting.mp3" \
  -F "language=en"
```

Without per-word timestamps:

```bash
curl -X POST 'http://127.0.0.1:8000/transcribe?words=false' \
  -F "file=@meeting.m4a"
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

> The API has no built-in authentication or upload-size limit and binds to
> `127.0.0.1` by default. Do not expose it directly to the public internet.
> Use an authenticated reverse proxy for remote access.

See [REST API](docs/api.md) for lifecycle, readiness and server-reuse details.

## Output formats

VoxPipe supports:

| Format | Use case |
|---|---|
| Markdown | Human-readable transcripts |
| TXT | Plain text processing |
| HTML | Browser-friendly, color-coded transcript |
| SRT | Subtitles |
| JSON | Programmatic processing |

Example Markdown output:

```markdown
**SPEAKER_00** · [00:00 – 00:03]

Good morning everyone. Let's start the review.

**SPEAKER_01** · [00:03 – 00:08]

Sure. First I'd like to discuss the latest changes.
```

Sentence-per-line layout is also available:

```bash
voxpipe meeting.mp3 \
  --format md \
  --layout sentence
```

Example:

```text
- [00:00 – 00:03] SPEAKER_00: Good morning everyone.
- [00:00 – 00:03] SPEAKER_00: Let's start the review.
```

JSON contains detected language, duration, ASR segments, optional word-level
timestamps, speaker-labelled utterances and complete transcript text.

Speaker identifiers are local to each recording. `SPEAKER_00` in one file does
not identify the same person as `SPEAKER_00` in another file.

See [Output formats](docs/output-formats.md) for the complete schema and format
behavior.

## CLI options

Common options:

| Option | Description |
|---|---|
| `-o FILE` | Write transcript to a file |
| `--format {txt,md,html,srt}` | Output format |
| `--layout {paragraph,sentence}` | Transcript layout |
| `--json FILE` | Also write JSON output |
| `--timestamps` | Include timestamps in TXT output |
| `--language LANG` | Language hint; omit for auto-detection |
| `--model NAME` | Select whisper model |
| `--no-diarization` | Disable speaker diarization |
| `--words` | Include word-level timestamps in JSON |
| `--caption-words N` | Target words per SRT caption |

Available model aliases include:

```text
turbo
small
small-q5_1
base
base-q5_1
```

`turbo` is the default.

Run:

```bash
voxpipe --help
```

for the complete command reference.

## Models

VoxPipe uses:

- a whisper.cpp GGML model for speech recognition;
- Silero VAD;
- a sherpa-onnx speaker-segmentation model;
- a sherpa-onnx speaker-embedding model.

The default ASR model is:

```text
ggml-large-v3-turbo-q5_0.bin
```

Additional models can be downloaded with:

```bash
scripts/fetch-models.sh turbo
scripts/fetch-models.sh turbo small-q5_1
```

Model search order:

```text
TRANSCRIBER_MODEL_DIRS
vendor/models/
~/.cache/voxpipe/models/
```

If diarization models are unavailable, transcription still works and output
falls back to a single speaker.

See [Installation](docs/installation.md) for model sizes, licenses and manual
`whisper-server` setup.

## Configuration

The standard repository layout requires no environment variables.

Common settings:

| Variable | Default | Description |
|---|---:|---|
| `TRANSCRIBER_MODEL` | `turbo` | Speech recognition model |
| `TRANSCRIBER_THREADS` | `min(8, cpus)` | whisper.cpp threads |
| `TRANSCRIBER_DEVICE` | `0` | Vulkan device index |
| `TRANSCRIBER_NO_GPU` | `false` | Force CPU inference |
| `TRANSCRIBER_PORT` | `8099` | Internal whisper-server port |
| `TRANSCRIBER_HOST` | `127.0.0.1` | Internal server address |
| `TRANSCRIBER_USE_VAD` | `true` | Enable VAD |
| `TRANSCRIBER_DIAR_THRESHOLD` | `0.85` | Speaker clustering threshold |
| `TRANSCRIBER_DIAR_THREADS` | `min(8, cpus)` | Diarization CPU threads |
| `TRANSCRIBER_TIMEOUT` | `3600` | Transcription timeout |

Example:

```bash
TRANSCRIBER_MODEL=small-q5_1 \
TRANSCRIBER_THREADS=4 \
voxpipe meeting.mp3
```

`TRANSCRIBER_DIAR_SPEAKERS` should currently remain at `-1`; fixed speaker-count
mode is unreliable with the sherpa-onnx behavior observed during development.

See [Configuration](docs/configuration.md) for the full environment-variable
reference and [Speaker diarization](docs/diarization.md) for tuning details.

## Platform support

| | Linux | Windows |
|---|---|---|
| CLI | Yes | Yes |
| Folder watcher | Yes | Yes |
| REST API | Yes | Yes |
| Speaker diarization | Yes | Yes |
| CPU inference | Yes | Yes |
| Vulkan GPU inference | Yes | Available |
| GPU path verified end-to-end | Yes | Not yet |
| Python | 3.12+ | 3.12+ |

### Linux

Linux is the primary development and verification platform.

GPU acceleration uses Vulkan rather than CUDA, so VoxPipe is not tied to NVIDIA
CUDA-compatible hardware.

The current Linux GPU path was verified on an NVIDIA GTX 1070.

Force CPU mode with:

```bash
TRANSCRIBER_NO_GPU=1 voxpipe meeting.mp3
```

### Windows

CPU mode is supported with the official whisper.cpp build.

GPU mode requires either a pinned third-party Vulkan server build or a
self-built Vulkan-enabled `whisper-server`.

Detailed provenance, DLL/runtime requirements, automatic CPU fallback and
verification status are documented in [Windows support](docs/windows.md).

## Performance

Reference benchmark:

- NVIDIA GTX 1070
- 4 CPU cores
- 156-second Russian recording

| Stage | Configuration | Time |
|---|---|---:|
| ASR | `small` | 14.7 s |
| ASR | `small-q5_1` | 14.3 s |
| ASR | `turbo` | 14.7–18.0 s |
| Diarization | sherpa-onnx | 23.8 s |
| End to end | `turbo` + diarization | ~41 s |

Diarization currently runs on CPU and can take longer than speech recognition.

These are historical measurements, not universal performance guarantees.

See [Performance](docs/performance.md) for the long-recording benchmark and
additional context.

## Known limitations

- The REST API has no built-in authentication or upload-size limit.
- Speaker identity does not persist across files.
- Long multi-speaker meetings remain harder to cluster than short two-speaker calls.
- `TRANSCRIBER_DIAR_SPEAKERS` is currently unreliable and should remain `-1`.
- There is no word-level speaker diarization.
- Diarization is CPU-only.
- Windows Vulkan GPU support is available but has not yet been verified end-to-end on real Windows hardware.

See [Known limitations](docs/known-limitations.md) for details.

## Documentation

Detailed documentation:

- [Installation](docs/installation.md)
- [Windows support](docs/windows.md)
- [Configuration](docs/configuration.md)
- [REST API](docs/api.md)
- [Folder workflow](docs/folder-workflow.md)
- [Output formats](docs/output-formats.md)
- [Architecture](docs/architecture.md)
- [Speaker diarization](docs/diarization.md)
- [Performance](docs/performance.md)
- [Migration from Transcriber](docs/migration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Known limitations](docs/known-limitations.md)
- [Development](docs/development.md)
- [Licensing and attribution](docs/licensing.md)

## Development

Install development dependencies:

```bash
pip install -e ".[dev]" -c requirements.lock
```

Run tests:

```bash
pytest
```

Lint:

```bash
ruff check src tests
```

Format:

```bash
ruff format src tests
```

The standard test suite does not require a GPU.

The optional end-to-end suite runs against the real whisper server and real
audio:

```bash
TRANSCRIBER_SERVER_BIN=vendor/bin/whisper-server \
TRANSCRIBER_TEST_AUDIO=/path/to/audio.mp3 \
pytest tests/test_end_to_end.py
```

See [Development](docs/development.md) for dependency-locking strategy, test
coverage and repository layout.

## Migrating from Transcriber

VoxPipe originated as a replacement for the earlier faster-whisper based
`Transcriber`.

Important changes include:

| Transcriber | VoxPipe |
|---|---|
| `transcriber` | `voxpipe` |
| `calls-inbox/` | `media-inbox/` |
| `calls-process/` | `media-process/` |
| `calls-results/` | `media-results/` |
| `calls-failed/` | `media-failed/` |
| faster-whisper | whisper.cpp |
| CUDA | Vulkan |
| pyannote | sherpa-onnx |
| Hugging Face token | Not required |

The API module changed from:

```text
transcriber.api:app
```

to:

```text
voxpipe.api:app
```

Existing files in the old `calls-*` directories are not migrated automatically.

See [Migration from Transcriber](docs/migration.md) for environment-variable,
cache, model and API changes.

## License

VoxPipe source code is licensed under the [MIT License](LICENSE).

Model weights and third-party components are separate works distributed under
their respective licenses.

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) and
[Licensing and attribution](docs/licensing.md).

Built with:

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [PyAV](https://github.com/PyAV-Org/PyAV)
- [FastAPI](https://github.com/fastapi/fastapi)

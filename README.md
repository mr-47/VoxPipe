# VoxPipe

Local audio transcription with speaker diarization.

VoxPipe combines [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for speech-to-text with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for speaker diarization.

It can transcribe individual files, automatically process files from a folder, or run as a REST API.

```bash
voxpipe meeting.mp3
voxpipe watch
voxpipe serve
```

**Runs locally. No Hugging Face account or token required.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)](#platform-support)

## Features

- Local speech-to-text with whisper.cpp
- Speaker diarization with sherpa-onnx
- Vulkan GPU acceleration
- CPU fallback
- Automatic language detection
- Word- and segment-level timestamps
- CLI transcription
- Automatic folder monitoring
- REST API
- Markdown, TXT, HTML, SRT and JSON output
- `.mp3`, `.wav`, `.m4a`, `.ogg` and `.webm` input
- Long-lived whisper server to avoid reloading the model between files
- Linux and Windows support
- No PyTorch, CUDA runtime or Hugging Face token required

## Why VoxPipe?

VoxPipe was created as a lighter replacement for an earlier `faster-whisper + pyannote` transcription stack.

| | Previous stack | VoxPipe |
|---|---|---|
| Speech recognition | faster-whisper / CTranslate2 | whisper.cpp |
| GPU runtime | CUDA | Vulkan |
| Diarization | pyannote | sherpa-onnx |
| Hugging Face token | Required for diarization | Not required |
| Python environment | ~4 GB | ~235 MB |
| M4A / AAC decoding | Limited | Supported via PyAV |

VoxPipe keeps the same general workflow while significantly reducing the Python dependency footprint.

## Quick start

### Linux

Requirements:

- Linux
- Python 3.12+
- Vulkan-capable GPU recommended

Clone the repository:

```bash
git clone https://github.com/mr-47/VoxPipe.git
cd VoxPipe
```

Install VoxPipe:

```bash
./install.sh
```

Start the folder watcher:

```bash
./run.sh
```

Or activate the environment and use the CLI directly:

```bash
source .venv/bin/activate
voxpipe meeting.mp3
```

### Windows

Requirements:

- Windows
- Python 3.12+
- Visual C++ Redistributable

Clone the repository and run:

```bat
install.cmd
run.cmd
```

CPU transcription works with the official whisper.cpp Windows build.

GPU acceleration is also supported through Vulkan, but there is currently no official x64 Windows Vulkan `whisper-server` archive from whisper.cpp. `install.cmd` therefore offers third-party Vulkan builds or the official CPU build.

See [Platform support](#platform-support) before using the Windows GPU path.

## Usage

VoxPipe provides three interfaces:

### Transcribe a file

```bash
voxpipe meeting.mp3
```

The default output is Markdown written to stdout.

Write it to a file:

```bash
voxpipe meeting.mp3 -o meeting.md
```

Other formats:

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

Provide a language hint:

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

VoxPipe can continuously monitor a folder and automatically process incoming media:

```bash
voxpipe watch
```

The default workflow uses:

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

Drop an audio file into `media-inbox/` and VoxPipe will process it automatically.

Useful commands:

```bash
# Process files continuously
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

Multiple output formats can be generated at once:

```bash
voxpipe watch --once --format txt,md,html,srt
```

Files that are still being written are left in the inbox until they are ready.

After a successful transcription, the original media file and generated transcripts are stored in `media-results/`.

### REST API

Install the API extra:

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

Available endpoints:

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

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Disable per-word timestamps to reduce response size:

```bash
curl -X POST \
  'http://127.0.0.1:8000/transcribe?words=false' \
  -F "file=@meeting.m4a"
```

> The API has no built-in authentication and binds to `127.0.0.1` by default. Do not expose it directly to the internet. Put it behind an authenticated reverse proxy if remote access is required.

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

JSON output contains detected language, duration, segments, optional word-level timestamps, speaker-labelled utterances and complete transcript text.

Example:

```json
{
  "language": "english",
  "duration": 12.4,
  "utterances": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.4,
      "end": 3.0,
      "text": "Good morning everyone."
    }
  ]
}
```

Speaker identifiers are local to each recording.

`SPEAKER_00` in one file does not identify the same person as `SPEAKER_00` in another file.

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

Available model names include:

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

VoxPipe uses three types of models:

- whisper.cpp GGML model for speech recognition
- Silero VAD model
- sherpa-onnx models for speaker segmentation and speaker embeddings

The default speech model is:

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

If the diarization models are unavailable, transcription still works and output falls back to a single speaker.

## Configuration

The standard repository layout requires no environment variables.

Common configuration options:

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

> `TRANSCRIBER_DIAR_SPEAKERS` should currently remain at its default value of `-1`. Fixed speaker-count mode is not reliable with the current sherpa-onnx implementation.

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

GPU acceleration uses Vulkan rather than CUDA, so VoxPipe is not tied to NVIDIA CUDA-compatible hardware.

The current Linux GPU path has been tested with an NVIDIA GTX 1070.

CPU inference can be forced with:

```bash
TRANSCRIBER_NO_GPU=1 voxpipe meeting.mp3
```

### Windows

The Python application, CLI, REST API, folder watcher, output formats and diarization code are shared with Linux.

The primary platform difference is `whisper-server`.

There is currently no official pre-built x64 Windows Vulkan server archive from whisper.cpp. `install.cmd` therefore offers:

- third-party Vulkan builds
- the official whisper.cpp CPU build
- manual build-from-source setup

Third-party Vulkan binaries are pinned and verified by SHA-256, but they are still unsigned third-party builds.

Building whisper.cpp yourself with Vulkan enabled is the safest GPU option if binary provenance matters to you.

The Windows GPU path has not yet been verified end-to-end on real Windows Vulkan hardware.

## Performance

Example benchmark:

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

Performance depends heavily on:

- recording duration
- model
- GPU
- CPU
- audio format
- number of speakers

Diarization currently runs on the CPU and can take longer than speech recognition.

## Known limitations

Speaker diarization is useful but not perfect.

In particular:

- speaker identities do not persist between recordings;
- long multi-speaker meetings are harder to cluster accurately than short two-speaker calls;
- overlapping speech does not use word-level speaker attribution;
- fixed speaker-count mode is currently unreliable;
- diarization is CPU-only;
- the REST API has no authentication or upload-size limit;
- Windows GPU acceleration has not yet been verified end-to-end on real Windows Vulkan hardware.

For sensitive deployments, keep the REST API on loopback or place it behind an authenticated reverse proxy.

## Architecture

VoxPipe keeps one `whisper-server` process alive while it is being used.

This avoids loading the speech recognition model for every file.

The high-level pipeline is:

```text
Media file
    ↓
PyAV decode
    ↓
16 kHz mono audio
    ↓
whisper.cpp
    ↓
Speech segments + timestamps
    ↓
sherpa-onnx diarization
    ↓
Speaker attribution
    ↓
Markdown / TXT / HTML / SRT / JSON
```

The CLI, folder watcher and REST API all use the same underlying transcription pipeline.

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

An optional end-to-end suite can run against the real whisper server and a real audio recording:

```bash
TRANSCRIBER_SERVER_BIN=vendor/bin/whisper-server \
TRANSCRIBER_TEST_AUDIO=/path/to/audio.mp3 \
pytest tests/test_end_to_end.py
```

## Migrating from Transcriber

VoxPipe originated as a replacement for the previous faster-whisper based `Transcriber` application.

Important changes:

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

The API module also changed:

```text
transcriber.api:app
```

to:

```text
voxpipe.api:app
```

Existing files in the old `calls-*` directories are not automatically migrated.

## License

VoxPipe source code is licensed under the [MIT License](LICENSE).

Model weights and third-party components are separate works distributed under their respective licenses.

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for attribution and license information.

---

Built with:

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [PyAV](https://github.com/PyAV-Org/PyAV)
- [FastAPI](https://github.com/fastapi/fastapi)

# Installation

This document contains the detailed installation information moved out of the main README.

## Requirements

- Linux or Windows
- Python >= 3.12
- Linux: a Vulkan-capable GPU and driver for GPU acceleration; CPU mode is supported
- Windows: CPU mode is supported; Vulkan GPU acceleration is available but not yet verified end-to-end on real Windows hardware
- Windows server binaries require the Visual C++ Redistributable (`MSVCP140.dll` and `VCRUNTIME140.dll`)

The standard repository layout requires no environment variables.

### Why Python 3.12+

The Python floor comes from the pinned dependency set rather than VoxPipe source syntax.

The VoxPipe source itself does not rely on Python syntax or standard-library features newer than 3.9, and modules use `__future__.annotations`.

The current lock uses `numpy 2.5.3`, which declares `requires-python >=3.12`. `sherpa-onnx 1.13.8` publishes wheels for CPython 3.12-3.14. Therefore Python 3.12-3.14 installs from the current lock as-is.

Both installers validate this before dependency installation.

## Quick installation

### Linux

```bash
./install.sh
./run.sh
```

`install.sh`:

- creates the virtual environment;
- installs pinned dependencies;
- downloads required models;
- is safe to re-run;
- keeps an already working `.venv`;
- skips model files that already exist.

`run.sh` starts the folder watcher and forwards `voxpipe watch` arguments:

```bash
./run.sh --once
./run.sh --no-diarization
```

### Windows

```bat
install.cmd
run.cmd
```

The `.cmd` scripts accept the same workflow arguments.

`install.cmd` is the one installation step that downloads executable code on Windows. It presents the selected build URL, hash and size before installation.

See [windows.md](windows.md) for detailed binary provenance and GPU/CPU selection behavior.

## Manual Python installation

```bash
cd /path/to/VoxPipe
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -c requirements.lock
```

`-c requirements.lock` pins dependency versions to the versions against which VoxPipe was verified.

Drop the constraint only when intentionally upgrading dependencies.

The CLI and folder workflow work without the API extra.

Install REST API support separately:

```bash
.venv/bin/pip install -e ".[api]"
```

or development + API dependencies together:

```bash
.venv/bin/pip install -e ".[dev,api]"
```

## Python environment footprint

Measured sizes from the original implementation:

| Environment | Approximate size |
|---|---:|
| CLI | 235 MB |
| CLI + API | 249 MB |
| Development (`.[dev]`) | 289 MB |

The largest single component is PyAV's bundled FFmpeg (`av` + `av.libs`), approximately 104 MB.

This is intentional: VoxPipe decodes formats such as `.m4a`/AAC in-process rather than shelling out to a system `ffmpeg`.

The API extra adds roughly 14 MB for FastAPI, Starlette, Pydantic, Uvicorn and `python-multipart`.

VoxPipe intentionally uses plain `uvicorn` rather than `uvicorn[standard]`: there is one processing unit and concurrent requests are serialized behind it, so `uvloop`/`httptools` do not increase GPU-bound inference throughput.

## whisper-server on Linux

A working Vulkan build can be vendored under `vendor/bin/`.

The original vendored Linux runtime was approximately 47 MB including its libraries.

whisper.cpp builds can contain absolute RUNPATHs. `scripts/vendor-server.sh` copies the runtime set and rewrites RUNPATHs to `$ORIGIN`, making the vendored tree relocatable.

To vendor an existing Vulkan build:

```bash
pip install patchelf
scripts/vendor-server.sh /path/to/whisper.cpp/build/bin
```

To build upstream whisper.cpp yourself:

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp
cmake -B build -S whisper.cpp -DGGML_VULKAN=ON -DWHISPER_BUILD_SERVER=ON
cmake --build build --config Release -j
export TRANSCRIBER_SERVER_BIN="$PWD/whisper.cpp/build/bin/whisper-server"
```

The application searches for `whisper-server` in this order:

1. `TRANSCRIBER_SERVER_BIN`
2. `PATH`
3. `vendor/whisper-server`
4. `vendor/bin/whisper-server`
5. `./build/bin`

If no candidate exists, the bare executable name is retained so the resulting error clearly names the missing program.

On Windows, in-project candidates are also checked with the `.exe` suffix because `shutil.which()` resolves `PATHEXT` but a direct file existence check does not.

The vendored Linux copy uses the system `libvulkan.so.1` instead of a build-sysroot Vulkan library.

The original verified environment found the GTX 1070 and logged `using Vulkan0 backend`.

## Models

Model search order:

```text
TRANSCRIBER_MODEL_DIRS
vendor/models/
~/.cache/voxpipe/models/
```

`TRANSCRIBER_MODEL_DIRS` is `os.pathsep`-separated and has the highest priority.

Nothing outside these locations is searched unless explicitly configured.

### Model inventory

| File | Purpose | Approx. size | Default vendored state | License |
|---|---|---:|---|---|
| `diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | Speaker segmentation | 5.7 MB | Yes | MIT (CNRS) |
| `diar/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | Speaker embeddings | 27 MB | Yes | Apache-2.0 (3D-Speaker) |
| `ggml-silero-v5.1.2.bin` | VAD | 868 KB | Yes | MIT (Silero) |
| `ggml-large-v3-turbo-q5_0.bin` | Default ASR model (`turbo`) | 548 MB | Yes | MIT (whisper.cpp) |
| `ggml-small-q5_1.bin` | Quantized small model | 182 MB | Fetch on demand | MIT (whisper.cpp) |
| `ggml-small.bin` | Full-precision small | 466 MB | Fetch on demand | MIT (whisper.cpp) |

The original default `vendor/` footprint was about 629 MB:

- ~582 MB models
- ~47 MB binary/runtime

Restore or add models with:

```bash
scripts/fetch-models.sh
scripts/fetch-models.sh turbo
scripts/fetch-models.sh turbo small-q5_1
```

The script is idempotent and skips existing files.

Sources:

- `ggerganov/whisper.cpp` on Hugging Face for GGML weights
- `ggml-org/whisper-vad` for the VAD model
- sherpa-onnx model releases for diarization models

All are ungated; no Hugging Face token is needed.

`base` and `base-q5_1` remain accepted model aliases, but those weights are not fetched by the standard script.

To inspect resolved paths:

```bash
.venv/bin/python -c "from voxpipe.config import TranscriberSettings as S; s=S(); \
  print(s.resolve_model_path()); print(s.resolve_vad_model()); print(s.resolve_diarization_models().available)"
```

If diarization models are missing, transcription still works and every utterance is assigned to `SPEAKER_00` with a warning.

## First-run behavior

The first transcription pays the model-load cost.

The original benchmark observed roughly 15 seconds to load `turbo-q5_0`.

Once loaded, the `whisper-server` stays warm and is reused between files and API requests.

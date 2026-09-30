# VoxPipe

Audio transcription with speaker diarization, built on
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) for speech-to-text (Vulkan
backend, so it runs on any GPU that has a Vulkan driver) and
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for speaker diarization.

It started life as a drop-in replacement for the faster-whisper + pyannote
`Transcriber`, and keeps that app's folder workflow, REST API and output formats.
The command, however, is now `voxpipe` rather than `transcriber` — see
[Migrating](#migrating-from-the-faster-whisper-transcriber). The point of the
rewrite was to shrink the footprint: **no torch, no CUDA wheels, no Hugging Face
token, no 4 GB Python install.**

Three interfaces, all sharing one warm `whisper-server` process:

- a **CLI** (`voxpipe FILE`) that writes JSON / Markdown / text / SRT to disk
- a **folder workflow** (`voxpipe watch`) that moves audio through `media-*`
- a **REST API** (`POST /transcribe`, `GET /health`)

| | old (faster-whisper) | VoxPipe (whisper.cpp) |
| --- | --- | --- |
| Python deps | torch, CTranslate2, pyannote.audio, FastAPI | numpy, PyAV, sherpa-onnx, httpx, FastAPI |
| venv size | ~4 GB (CUDA wheels) | **235 MB** (249 MB with the API) |
| GPU runtime | CUDA 12 via cuBLAS (12.9 wheels, Pascal caveat) | Vulkan loader shipped by the OS |
| Diarization | pyannote 3.1, **gated on HF, needs a token** | ONNX models, **no account, no token** |
| Speaker labels | all `SPEAKER_00` without a token | work out of the box |
| Decode | CTranslate2 (no m4a/AAC) | PyAV, so **.m4a works** |
| Speed (156 s call, GTX 1070) | 14.3 s | 14.7 s (`turbo`), 14.3 s (`small-q5_1`) |

## Platform support

**Linux is the primary target and the one where the GPU path is verified
end-to-end. Windows is supported, and it can use a GPU too — `install.cmd`
detects a Vulkan device and fetches a Vulkan build by default, falling back to
the first-party CPU archive when there is none.** Everything above the ASR layer —
the CLI, the `media-*` folder workflow, the REST API, diarization, every output
format and every environment variable — is identical on both. The difference is
entirely in the `whisper-server` binary, and only because a Linux build of it is
vendored while no Windows artifact can be.

| | Linux | Windows |
| --- | --- | --- |
| Install / run | `./install.sh`, `./run.sh` | `install.cmd`, `run.cmd` |
| `whisper-server` | **Vendored**, Vulkan build | **Not bundled.** `install.cmd` offers four choices: two third-party Vulkan builds, the first-party CPU archive, or from-source commands |
| GPU acceleration | **Yes, verified.** Vulkan, so any Vulkan-capable GPU (tested on GTX 1070 / Pascal) | **Available, unverified.** `install.cmd` defaults to a Vulkan build when it detects a device. The two Vulkan archives are third-party and unsigned — see below |
| Without a GPU | Slower, but `TRANSCRIBER_NO_GPU=1` is honoured | The CPU archive is the default, and `run.cmd` sets `TRANSCRIBER_NO_GPU=1` for a server with no Vulkan backend |
| Python | ≥ 3.12 (3.12–3.14 install from the lock) | Identical; every pin ships a Windows wheel |
| Diarization | sherpa-onnx, CPU by design | Identical |
| API, CLI, folder workflow, output formats | Yes | Yes |
| Reclaiming a leftover server | pid + `/proc` start time | pid only — `/proc/<pid>/stat` does not exist, so a recycled pid cannot be distinguished |
| Verification | Full suite plus real GPU end-to-end | Batch logic under Wine, the `.ps1` under host PowerShell. **Not yet run on real Windows** |

Three Windows specifics worth stating plainly, because none of them are obvious:

- **Where a Windows Vulkan build comes from.** `ggml-org` publishes no x64
  Vulkan archive for Windows, so there is no first-party option.
  `install.cmd` offers two third-party rebuilds —
  [`jerryshell/whisper.cpp-windows-vulkan-bin`](https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin)
  `v1.0.0` and
  [`DomoticX/whisper.cpp-windows-vulkan`](https://github.com/DomoticX/whisper.cpp-windows-vulkan)
  `v1.0` — alongside the first-party CPU archive, all three pinned to a tag plus
  a sha256 that is checked before anything is written. They are unsigned
  builds from individual accounts, and **neither states which whisper.cpp it was
  built from.** Their binaries share 43–44 of 45 MSVC lambda symbol ids with
  upstream `v1.9.2`, which points at the same or a very nearby source tree but is
  not proof. The hash makes the download verifiable; the publisher is still
  somebody nobody vouches for. Building your own with `-DGGML_VULKAN=ON` is the
  alternative, and `run.cmd` recognises that too.
- **The CPU build is selected for you when there is no Vulkan backend.** VoxPipe
  picks the GPU with whisper.cpp's Vulkan `-dev 0` flag, which a CPU-only build
  does not accept — an unrecognized flag makes `whisper-server` print its usage
  and exit instead of listening, and the run then fails at the health check with
  nothing pointing at the cause. So `run.cmd` sets `TRANSCRIBER_NO_GPU=1` (passing
  `-ng` instead) when `vendor\bin\ggml-vulkan.dll` is absent. It keys on that DLL
  rather than on `whisper-server.exe` because the CPU archive installs the
  executable too, so the executable would report every CPU install as missing its
  GPU. It does not override you: an explicit `TRANSCRIBER_NO_GPU`, a
  `TRANSCRIBER_SERVER_BIN`, or a Vulkan DLL of your own in `vendor\bin\` all
  suppress it.
- **A CUDA archive is not a shortcut.** Upstream does publish
  `whisper-cublas-*-bin-x64.zip`, and it looks like the first-party accelerated
  option. But there is no way to point VoxPipe's Vulkan-index device selection at
  a CUDA device, so on that build you would still be on the CPU. It is not offered
  as a download for that reason.

Both Windows server builds also import `MSVCP140.dll` and `VCRUNTIME140.dll` from
the Visual C++ Redistributable, and ship neither — so without that redistributable
`whisper-server.exe` refuses to start, and the only symptom VoxPipe can report is
a health check that times out naming nothing. Both `install.cmd` and `run.cmd`
therefore warn if those two DLLs are missing, and both then carry on: a machine
can have them by another route, and a self-built server shipped with its own
copies works regardless. `run.cmd` only warns when it can see a server it would
actually launch. The fix is the x64 redistributable from
[microsoft.com](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).

## Features

- Word-level and segment-level timestamps from whisper.cpp
- Speaker attribution per utterance from ONNX diarization — **on by default**
- Graceful fallback to a single speaker when diarization is switched off
- Output as Markdown, plain text, HTML, SRT and JSON
- `media-inbox` folder workflow, with the same write-in-progress handling
  (hold, retry, then file as failed) as the old app
- REST API with the same JSON schema as the CLI and the folder workflow
- One long-lived `whisper-server` process, restarted only when the model changes

## Requirements

- Linux or Windows, Python ≥ 3.12
- **Linux:** a Vulkan-capable GPU and driver (verified on GTX 1070 / Pascal with
  driver 580.178.04). Without a GPU it still runs on CPU — just slower.
- **Windows:** works with no GPU, and it will ask which build you want. With a
  Vulkan device present the default is a third-party Vulkan build; without one it
  is the first-party CPU archive. See [Platform support](#platform-support) for
  where those builds come from and what is not verified about them.
- Disk for `vendor/` (629 MB): a `whisper-server` binary, the GGML model weights
  and the two ONNX diarization models all ship with the project, so nothing is
  compiled or downloaded on the first run — **except** on Windows, where the
  server binary is fetched separately (it cannot be the Linux one). See below to
  rebuild or replace any of it.

The 3.12 floor is not about this code. Nothing in `src/voxpipe` uses syntax or
standard library newer than 3.9, and every module imports
`__future__.annotations`, so the source would run on much older interpreters —
it is the pinned dependency set that decides. `numpy 2.5.3` in
`requirements.lock` declares `requires-python >=3.12`, so a pinned install
cannot resolve on 3.10 or 3.11 at all; `pip` refuses the wheel rather than
falling back to a build. `sherpa-onnx 1.13.8` publishes `cp312`–`cp314`
wheels, so 3.12 to 3.14 is the range that installs from the lock as-is. Both
installers check this before they touch anything, rather than letting it
surface partway through as a failed dependency install.

## Installation

Two scripts wrap the steps below for the common case. `install.sh` creates the
venv, installs the pinned dependencies and downloads the models; `run.sh` then
starts the folder watcher:

```bash
./install.sh          # add a model name for something smaller, e.g. small-q5_1
./run.sh
```

`install.sh` is safe to re-run — a `.venv` that already works is kept, and
models already downloaded are left alone. It cannot fetch `whisper-server`
itself, because a Vulkan build of whisper.cpp is compiled locally rather than
published as an artifact; the script tells you the four commands to run if the
binary is missing. `run.sh` takes any `voxpipe watch` flag, so
`./run.sh --once` drains the inbox and exits, and `./run.sh --no-diarization`
works as you would expect. Everything below is what those scripts do, written
out.

On Windows use the `.cmd` equivalents, which take the same arguments:

```bat
install.cmd
run.cmd
```

The platform differences are tabulated under
[Platform support](#platform-support); the practical summary is that
`install.cmd` is the one step in this project that downloads code onto your
machine, so it asks which build you want and shows you the URL, hash and size
for the one you pick. Choose the skip option and the rest of the install still
completes, printing the from-source commands instead. Two details it cannot
decide for you: the fetched archive is the **HTTP server**, not the
command-line transcriber — searching for "whisper.cpp Windows" mostly turns up
`main.exe` and `whisper-cli.exe`, which cannot work here because VoxPipe starts
the server and POSTs audio to it — and an MSVC build lands in
`whisper.cpp\build\bin\Release\`, since MSVC is a multi-config generator. If you
end up with a server that has no Vulkan backend, `run.cmd` sets
`TRANSCRIBER_NO_GPU=1` for you, which it decides by looking for
`vendor\bin\ggml-vulkan.dll`.

```bash
cd /path/to/VoxPipe               # this repository
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -c requirements.lock
```

`-c requirements.lock` pins dependency versions to the ones this was verified
against; drop it only when you intend to upgrade (see [Development](#development)).

That is enough for the CLI and the `media-*` folder workflow. The REST API is an
optional extra:

```bash
.venv/bin/pip install -e ".[api]"      # add the API
.venv/bin/pip install -e ".[dev,api]"  # dev + API in one step
```

The venv is ~235 MB for the CLI, ~249 MB with the API extra, ~289 MB with
`.[dev]` (which includes the API). The largest single item is PyAV's bundled
FFmpeg (`av.libs` + `av`, 104 MB), the price of decoding `.m4a`/AAC
in-process instead of shelling out to a system `ffmpeg`. The API extra adds
~14 MB (fastapi, starlette, pydantic, pydantic-core, uvicorn, python-multipart).
It deliberately uses plain `uvicorn` rather than `uvicorn[standard]`: there is one
processing unit, requests queue behind it, and uvloop/httptools cannot raise
throughput on a GPU-bound queue. `.[dev]` already includes the extra, so the
test suite covers the API either way.

### 1. whisper-server (Vulkan)

> This whole subsection describes the **Linux** route. On Windows the vendored
> binary is a Linux ELF and cannot run, so `install.cmd` fetches a Windows build
> instead; see [Platform support](#platform-support).

**A working binary is vendored in `vendor/bin/`** (47 MB, git-ignored, with its
libraries), so a fresh clone needs no environment variable and no sibling
checkout. It was patched to be relocatable: whisper.cpp bakes *absolute*
RUNPATHs into every shared object, so `scripts/vendor-server.sh` copies the
runtime set and rewrites the RUNPATHs to `$ORIGIN`.

To refresh it, or to build your own, point it at any Vulkan whisper.cpp build:

```bash
pip install patchelf                     # vendoring helper, no sudo needed
scripts/vendor-server.sh /path/to/whisper.cpp/build/bin
```

Upstream's build, if you prefer to start from scratch:

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp
cmake -B build -S whisper.cpp -DGGML_VULKAN=ON -DWHISPER_BUILD_SERVER=ON
cmake --build build --config Release -j
export TRANSCRIBER_SERVER_BIN="$PWD/whisper.cpp/build/bin/whisper-server"
```

The app finds the binary in this order: `$TRANSCRIBER_SERVER_BIN`, `$PATH`,
`vendor/whisper-server`, `vendor/bin/whisper-server`, and `./build/bin`. Every
candidate after `$PATH` is optional; if all miss, the bare name is kept so the
error names the expected program. No candidate lives outside the project, so it
can be moved or cloned anywhere. On Windows each in-project candidate is probed
as `whisper-server.exe` too, because `shutil.which` resolves `PATHEXT` but a
plain file check does not.

> The vendored copy uses the **system** `libvulkan.so.1` (1.3.275) rather than
> the build sysroot's, so the 60 MB sysroot is not needed. Verified: it finds
> the GTX 1070 and logs `using Vulkan0 backend`.

### 2. Models

Search order: `$TRANSCRIBER_MODEL_DIRS` (os.pathsep-separated, highest
priority), `vendor/models/`, then `~/.cache/voxpipe/models/`. Nothing
outside the project is searched unless you point it somewhere with
`TRANSCRIBER_MODEL_DIRS`.

| File | Purpose | Size | In `vendor/models/` | License |
| --- | --- | --- | --- | --- |
| `diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | speaker segmentation | 5.7 MB | **yes** | MIT (CNRS) |
| `diar/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | speaker embeddings | 27 MB | **yes** | Apache-2.0 (3D-Speaker) |
| `ggml-silero-v5.1.2.bin` | VAD, passed to whisper.cpp `--vad` | 868 KB | **yes** | MIT (Silero) |
| `ggml-large-v3-turbo-q5_0.bin` | default model (`turbo`) | 548 MB | **yes** | MIT (whisper.cpp) |
| `ggml-small-q5_1.bin` | fast model (`small-q5_1`) | 182 MB | fetch on demand | MIT (whisper.cpp) |
| `ggml-small.bin` | full-precision small (`small`) | 466 MB | fetch on demand | MIT (whisper.cpp) |

**Everything needed to run is vendored in `vendor/`** (629 MB: 582 MB of models
+ 47 MB of binary), so a fresh checkout transcribes with no environment variable
and no sibling directory. To restore or add models:

```bash
scripts/fetch-models.sh              # VAD + both diarization models (34 MB)
scripts/fetch-models.sh turbo        # ...plus the default GGML model (548 MB)
scripts/fetch-models.sh turbo small-q5_1
```

```bash
scripts/fetch-models.sh              # VAD + both diarization models (34 MB)
scripts/fetch-models.sh turbo        # ...plus the default GGML model (548 MB)
scripts/fetch-models.sh turbo small-q5_1
```

The script is idempotent (existing files are skipped) and pulls from
[ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) for GGML
weights, [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) for
the VAD model, and the
[sherpa-onnx model zoo](https://github.com/k2-fsa/sherpa-onnx/releases) for the
ONNX pair — all ungated, no token. `base` and `base-q5_1` are accepted aliases
but those two weights are not downloaded here.

`vendor/` is git-ignored: it holds multi-MB binaries that each machine should
fetch for itself. Verify what the app will actually use with:

```bash
.venv/bin/python -c "from voxpipe.config import TranscriberSettings as S; s=S(); \
  print(s.resolve_model_path()); print(s.resolve_vad_model()); print(s.resolve_diarization_models().available)"
```

If diarization models are missing, transcription still works and every utterance
is attributed to `SPEAKER_00` with a warning.

## How to run

The `voxpipe` command lives inside the virtualenv, so either use its full
path or activate first:

```bash
cd /path/to/VoxPipe               # this repository
source .venv/bin/activate
```

Then pick a mode:

```bash
voxpipe --version        # voxpipe 1.0.0, from the installed metadata
```

```bash
# 1) Watch the media-inbox folder workflow (recommended)
voxpipe watch                # poll media-inbox every 5s, forever
voxpipe watch --once         # process whatever is in media-inbox, then exit
voxpipe watch --dir /data    # use folders under a different base directory

# 2) Transcribe a single file
voxpipe meeting.mp3 -o meeting.md

# 3) Serve the REST API (separate terminal; needs `pip install -e '.[api]'`)
voxpipe serve                          # 127.0.0.1:8000
voxpipe serve --host 127.0.0.1 --port 8000 --log-level info
```

If `voxpipe: command not found`, you forgot to activate the venv. Call it by
full path (`.venv/bin/voxpipe ...`) or add the venv to your `PATH` once:

```bash
echo 'export PATH="/path/to/VoxPipe/.venv/bin:$PATH"' >> ~/.bashrc
```

The first run pays the model load (~15 s for turbo-q5_0). **The server stays warm
between files**, so a batch of calls loads the model exactly once. The same
holds for the API: the model loads on the first request and is reused for every
request after that.

## REST API

```bash
pip install -e ".[api]"      # the API is an optional extra
voxpipe serve            # → http://127.0.0.1:8000  (Ctrl-C to stop)
```

`voxpipe serve` accepts `--host`, `--port`, `--log-level`, `--model`,
`--language` (a default hint for requests that omit `?language=`) and
`--no-diarization`, honouring the same env vars as the CLI. On `Ctrl-C` or
`SIGTERM` it stops `whisper-server` instead of leaving a GPU process behind.

It runs **one worker by design**: a second worker would load its own copy of the
model onto the same GPU. Concurrent requests queue inside the process (the
voxpipe serialises them) rather than competing for the device.

`GET /health` reports `model_verified`. It is `false` only when VoxPipe is
running against a whisper-server it did not start, because that server's
`/health` does not say which model it loaded — see
[One GPU, one server](#one-gpu-one-server).

If you prefer another ASGI server, `voxpipe.api:app` is a plain ASGI
app and needs no `uvicorn` import:

```bash
.venv/bin/python -m uvicorn voxpipe.api:app --host 127.0.0.1 --port 8000
```

Interactive docs at http://127.0.0.1:8000/docs

### One GPU, one server

VoxPipe starts its own `whisper-server` and stops it on exit, so normally there
is nothing to configure. But a server is already listening on port 8099 if a
previous run was killed rather than stopped, or if something else on the machine
owns the port. What VoxPipe does then depends on who owns that server:

| Listener | What VoxPipe does |
| --- | --- |
| A server VoxPipe started, with the requested model | Adopts it. No second copy of the model reaches the GPU, and the model is known to match. |
| A server VoxPipe started, with a *different* model | Refuses. Stop it, or use another port. |
| A server left behind by an earlier VoxPipe run, requested model | Takes it over, and stops it on exit. See [After a hard kill](#after-a-hard-kill). |
| A server left behind by an earlier VoxPipe run, *different* model | Refuses, and names the pid to kill. |
| A server from another process | Refuses, unless `TRANSCRIBER_REUSE_SERVER=1`. |

That last row is the important one. Two VoxPipes on one GPU means two copies of
the model fighting over a device with room for one, and — because
`whisper-server`'s `/health` returns only `{"status": "ok"}` — there is no way to
confirm which model the running server actually loaded. Silently reusing it once
meant VoxPipe could report transcripts from a model nobody had asked for. If you
set `TRANSCRIBER_REUSE_SERVER=1` anyway, `/health` reports
`"model_verified": false` so the assumption is visible rather than implied.

### After a hard kill

`voxpipe` stops its own server on `Ctrl-C`, `SIGTERM` and normal exit. A
`SIGKILL` — a panic, an OOM kill, a closed terminal — cannot be trapped, so the
server survives in its own session with nothing left to stop it. It would then
sit on the GPU and hold port 8099 with no way to tell whose it is.

So VoxPipe records what it started in
`~/.cache/voxpipe/servers/<host>-<port>.json` (override with
`TRANSCRIBER_SERVER_STATE_DIR`): the pid, the model, and the process start time
from `/proc`. The next run reads that file and, if the pid is still alive and is
still the same process, knows the leftover is its own. It then takes the server
over and stops it on exit, instead of leaving you to find the pid yourself. A
pid that has been recycled by some other process is not treated as ours, and a
record naming a dead process is ignored.

```bash
# a leftover server is reclaimed on the next run; to do it by hand:
kill $(pgrep -f whisper-server)
```

Note that the record only covers a server VoxPipe started. A server someone else
launched is still refused, because nothing on disk vouches for it.

```bash
# Either free the port...
kill $(pgrep -f whisper-server)

# ...or talk to a different one
TRANSCRIBER_PORT=8100 voxpipe meeting.mp3
```

```bash
# Transcribe an upload
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@meeting.mp3" \
  -F "language=en"

# Same, without per-word timestamps (much smaller response)
curl -X POST 'http://127.0.0.1:8000/transcribe?words=false' \
  -F "file=@meeting.m4a"

# Status
curl http://127.0.0.1:8000/health
```

| Endpoint | Description |
| --- | --- |
| `GET /health` | `status`, `diarization_enabled`, `model`, `model_path`, `server_running`, `model_verified` |
| `POST /transcribe` | `multipart/form-data` upload → transcript JSON |

Query parameters for `/transcribe`:

| Parameter | Default | Description |
| --- | --- | --- |
| `language` | *(auto)* | Language hint (`en`, `ru`, `de`, …). Omit to auto-detect |
| `words` | `true` | Include per-word timestamps in `segments[].words` |

`GET /health` responds immediately, before the model is loaded, and reports
`"server_running": false` until the first transcription — useful for a
readiness probe that should not block on the 15 s model load.

The response body is the same JSON as `--json` and `media-results/<name>.json`
(including the `text` field, which the old API omitted).

**The API has no authentication.** It binds to `127.0.0.1` by default on
purpose. Anyone who can reach it can read your audio and transcripts, so keep it
on loopback, or put it behind a reverse proxy that authenticates, before binding
it to a real interface. Concurrent uploads are serialised — the internal
`whisper-server` handles one inference at a time — so requests queue instead of
fighting over the GPU. On `SIGTERM` (e.g. `systemctl stop`) the app shuts the
server process down, so it does not leave an orphaned GPU process behind.

## Folder workflow (watch mode)

`voxpipe watch` moves audio through four folders under the base directory
(default: current directory, override with `--dir`). All four ship in this repo
with a `.gitkeep` and are also created automatically if missing.

| Folder | Purpose |
| --- | --- |
| `media-inbox/` | Drop new audio files here to be processed |
| `media-process/` | File currently being transcribed (in transit) |
| `media-failed/` | Files that could not be processed, plus a `<name>.txt` error file |
| `media-results/` | Successfully transcribed files with their transcript |

Files are **moved** between folders, never copied. On success the audio keeps its
name and lands next to `<name>.json` (full transcript with word timestamps) and
the transcript file(s). The format set mirrors the CLI: `--format
{txt,md,html,srt}` and `--layout {paragraph,sentence}`. On failure the audio is
moved to `media-failed/` and the error (including traceback) is written to
`media-failed/<name>.txt`.

Two recordings that share a stem but not an extension (`call.m4a` and
`call.mp3`) would otherwise fight over the same transcript and error names. The
second one is disambiguated with its extension — `call_mp3.md`, `call_mp3.txt` —
and a re-run of the same recording overwrites its own output in place rather
than creating `call_mp3_2.md`. If the failed file is itself a `.txt` (an
unsupported type is exactly what lands there), the report is named
`<name>.error.txt` so the file it describes is never overwritten by it.

Files that are still being written are **left in `media-inbox/`** and retried
instead of failing immediately. The pipeline first checks that a file has stopped
changing — an m4a only becomes readable once its index is flushed at the end —
and only then decodes it. Both size and modification time are checked, so a
recording that is deleted and re-dropped under the same name is treated as a new
file even when it happens to be exactly the same length. If a file stays
unreadable at an unchanged size for 3 consecutive settled scans, it is moved to
`media-failed/`.

```bash
voxpipe watch                      # poll media-inbox every 5s forever
voxpipe watch --dir /data/calls    # different base directory
voxpipe watch --once               # drain the inbox once, then exit
voxpipe watch --interval 1         # poll faster
voxpipe watch --model small-q5_1   # smaller, faster model
voxpipe watch --language en        # force a language (applied to every file)
voxpipe watch --no-diarization     # single-speaker transcripts

# multiple formats at once (comma-separated or repeat the flag)
voxpipe watch --once --format txt,md,html,srt
voxpipe watch --once --format txt --format html
```

Any leftover files in `media-process/` (e.g. after a crash) are moved back to
`media-inbox/` on start. Unsupported file types are moved to `media-failed/`
with a reason — nothing is ever dropped silently.

## CLI usage

```bash
# Version of the installed package, on any of the three commands
voxpipe --version
voxpipe serve --version

# Markdown transcript to stdout (default format)
voxpipe meeting.mp3

# Markdown transcript to a file (extension picks the format)
voxpipe meeting.mp3 -o meeting.md

# Different formats & layouts
voxpipe meeting.mp3 -o meeting.txt --format txt --layout sentence
voxpipe meeting.mp3 -o meeting.html --format html
voxpipe meeting.mp3 -o meeting.srt --format srt

# Also write the machine-readable JSON
voxpipe meeting.mp3 -o meeting.md --json meeting.json --words

# Language hint + timestamps in text output
voxpipe meeting.mp3 --language en --timestamps
```

| Flag | Description |
| --- | --- |
| `-o FILE` | Write the transcript to a file (default: stdout). A known extension in the filename picks the format |
| `--format {txt,md,html,srt}` | Transcript format (default: `md`) |
| `--layout {paragraph,sentence}` | Wrapped paragraphs or one line per sentence (default: `paragraph`) |
| `--json FILE` | Also write the machine-readable JSON transcript |
| `--timestamps` | Include timestamps in `txt` output (md/html/srt always show them) |
| `--caption-words N` | Max words per SRT caption; long turns split into 5-7 word lines (default: `6`) |
| `--language LANG` | Hint the audio language (e.g. `en`, `ru`, `de`). Omit for auto-detect |
| `--model NAME` | `turbo` (default), `small`, `small-q5_1`, `base`, `base-q5_1`, or a path to a GGML file |
| `--no-diarization` | Disable speaker diarization |
| `--words` | Include per-word timestamps in `--json` output |

> **Language.** `whisper-server` defaults to `-l en`, which silently *translates*
> every recording into English while still reporting
> `"language": "russian"`. The app therefore sends `auto` whenever `--language`
> is not given. If transcripts ever come out in the wrong language, check this
> first.

## Configuration (environment variables)

Every flag has an environment-variable equivalent. Nothing needs to be set for
the standard repo layout.

| Variable | Default | Description |
| --- | --- | --- |
| `TRANSCRIBER_MODEL` | `turbo` | `turbo`, `small`, `small-q5_1`, `base`, `base-q5_1`, or a path to a GGML file |
| `TRANSCRIBER_MODEL_DIRS` | *(empty)* | Extra model directories, `:`-separated, searched before the defaults |
| `TRANSCRIBER_SERVER_BIN` | *(auto-discovered)* | Path to `whisper-server` |
| `TRANSCRIBER_THREADS` | `min(8, cpus)` | whisper.cpp threads (`-t`) |
| `TRANSCRIBER_DEVICE` | `0` | Vulkan device index (`-dev`) |
| `TRANSCRIBER_NO_GPU` | `false` | Force CPU (`-ng`) |
| `TRANSCRIBER_PORT` | `8099` | Port the internal server binds on `127.0.0.1` |
| `TRANSCRIBER_HOST` | `127.0.0.1` | Bind address; keep it loopback, the API is unauthenticated |
| `TRANSCRIBER_USE_VAD` | `true` | Pass `--vad` to whisper.cpp (skipped automatically if the VAD model is missing) |
| `TRANSCRIBER_VAD_MODEL` | `ggml-silero-v5.1.2.bin` | VAD model filename |
| `TRANSCRIBER_DIAR_SEG_MODEL` | `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | Segmentation model, relative to `<models>/diar/` |
| `TRANSCRIBER_DIAR_EMBED_MODEL` | `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | Embedding model, relative to `<models>/diar/` |
| `TRANSCRIBER_DIAR_THRESHOLD` | `0.85` | Clustering threshold. **Higher = fewer speakers.** Lower it if a 2-person call comes back as 3+ |
| `TRANSCRIBER_DIAR_SPEAKERS` | `-1` | Fixed speaker count; `-1` auto-detects. **Currently broken — leave at `-1`,** see [Tuning the speaker count](#tuning-the-speaker-count) |
| `TRANSCRIBER_DIAR_MIN_ON` | `0.5` | Minimum speaker segment length (s) |
| `TRANSCRIBER_DIAR_MIN_OFF` | `0.7` | Minimum silence that closes a segment (s) |
| `TRANSCRIBER_DIAR_MERGE_THRESHOLD` | `0.70` | Second clustering pass: merge clusters whose centroids are at least this similar. `0` disables |
| `TRANSCRIBER_DIAR_MIN_REGION` | `1.0` | A cluster with less speech than this loses its own label and is folded into the nearest one. `0` disables |
| `TRANSCRIBER_DIAR_THREADS` | `min(8, cpus)` | ONNX threads for the embedding models. Diarization is CPU-only; this is the only tuning knob for it |
| `TRANSCRIBER_TIMEOUT` | `3600` | HTTP timeout for one request (s) |
| `TRANSCRIBER_STARTUP_TIMEOUT` | `180` | How long to wait for the server's `/health` (s) |
| `TRANSCRIBER_REUSE_SERVER` | `0` | Adopt a whisper-server this process did not start. Off by default — see [One GPU, one server](#one-gpu-one-server) |
| `TRANSCRIBER_SERVER_STATE_DIR` | `~/.cache/voxpipe/servers` | Where the record of a started server is kept, so an orphan can be identified and reaped |

### Tuning the speaker count

`TRANSCRIBER_DIAR_THRESHOLD` decides how aggressively the two-party clusters are
merged: **raising it merges speakers together, lowering it splits them apart.**
Measured on a 156 s two-party call:

| Threshold | Speakers found |
| --- | --- |
| `0.60` | 5 |
| `0.85` (default) | **2** ✓ |
| `0.95` | 1 |

That default does not hold up on long recordings. A 93-minute *team meeting*
(not a two-party call — the transcript names half a dozen participants) returns
**30 speakers at the `0.85` default and 18 at `0.95`**, and the labels are not
merely over-split. The two largest clusters (289 and 246 of 743 speech regions) are
**the same person** — their centroids merge above `0.80` — so the dominant
voice is fragmented across several `SPEAKER_0x` labels. Most of the remaining
tail, by contrast, sits at only 0.12–0.27 cosine from the dominant clusters and
is probably genuinely different people.

The underlying cause is a weakly separated embedding space, not the knob: across
the 743 regions the median pairwise cosine is **0.374**, so the `0.85` default
sits near the 99th percentile and only the most obviously identical turns merge.

Two post-passes in `diarize.py` now clean up most of this, and are on by
default:

- **`TRANSCRIBER_DIAR_MERGE_THRESHOLD` (0.70)** — a second pass that reduces
  each cluster to a duration-weighted centroid and merges the closest pair
  while they stay at or above this cosine. Calibrated between two measured
  points: genuinely different speakers scored 0.48, one voice split across
  clusters merged at 0.80.
- **`TRANSCRIBER_DIAR_MIN_REGION` (1.0)** — a sub-second turn cannot carry a
  reliable embedding, so it tends to become a singleton label. The *turn* is
  kept (it is often the interjection that marks a speaker change) and only the
  label is folded into the nearest cluster.

On the 93-minute meeting these take **30 labels down to 20**, and critically the
dominant voice is no longer split — the two biggest clusters were the same
person and are now one. Transcript text is byte-identical either way; only the
labelling changes. For a meeting, expect roughly the number of people who
actually spoke, not the raw cluster count.

`TRANSCRIBER_DIAR_SPEAKERS` is not an escape hatch either — it is broken, and
the reason is not the attribution step:

> **Do not use `TRANSCRIBER_DIAR_SPEAKERS` to pin the count.** It is passed
> straight through to sherpa-onnx as `num_clusters`, and in 1.13.8 that
> argument does not behave as a speaker count. On the 156 s two-party call,
> asking for N yields **N−1** clusters every time, and the partition is
> degenerate — one cluster takes 84–98% of the speech and the rest are slivers:
>
> | requested | clusters returned | speech share per cluster |
> | --- | --- | --- |
> | `2` | 2 | 98% / 2% |
> | `3` | 2 | 84% / 16% |
> | `4` | 3 | 84% / 13% / 3% |
> | `6` | 5 | 84% / 8% / 5% / 2% / 1% |
>
> Pinned to `2`, the call comes out with 133.9 s on one speaker and 2.8 s on
> the other, and `build_utterances` then collapses most of the conversation
> into a single 148 s block. Unset (`-1`) the same audio gives a correct
> 21.3 s / 111.8 s split.

Overlap in the output is a real but *separate* issue, and it is handled:
`assign_speakers` takes the latest-starting covering turn, which is what stopped
a single long turn from swallowing the file. That fix alone does not rescue a
pinned count, because the underlying partition is still wrong — with
first-match attribution the pinned call collapses to one speaker, with
latest-start it is 151 segments to 4. Both are wrong; the split itself is the
defect.

## Output format

The JSON transcript (`--json` file, or `media-results/<name>.json`):

```json
{
  "language": "russian",
  "language_probability": 1.0,
  "duration": 156.4,
  "segments": [
    { "start": 0.4, "end": 7.0, "text": "Да. Добрый день. Меня зовут Виктория...",
      "words": [{ "start": 0.42, "end": 0.9, "word": "Да" }] }
  ],
  "utterances": [
    { "speaker": "SPEAKER_01", "start": 0.4, "end": 7.0, "text": "Да. Добрый день. Меня зовут Виктория..." },
    { "speaker": "SPEAKER_00", "start": 7.0, "end": 10.1, "text": "Ну да, слушаю." }
  ],
  "text": "SPEAKER_01: Да. Добрый день...\nSPEAKER_00: Ну да, слушаю."
}
```

`language` is whisper.cpp's own name for the detected language (`russian`, not
`ru`). The schema is byte-for-byte the same as the old app's, so existing
consumers keep working.

In watch mode the JSON always includes `words`, since it is the durable archive.
For a one-shot `--json` file they are included only with `--words`; on the test
call that is the difference between 96 KB and 18 KB (5×). Word `probability`
values from whisper.cpp are not carried over — the old app did not store them
either.

Speaker labels are `SPEAKER_00`, `SPEAKER_01`, … assigned by clustering order,
so they are stable within a file but **not** stable across files. There is no
speaker identity recognition — the same person is a different `SPEAKER_0x` in a
different recording.

Human-readable transcripts avoid the "wall of text": each turn is labelled with
its speaker and time span, and long turns are either **wrapped** into paragraphs
that break at sentence boundaries, or split into **one line per sentence**,
depending on `--layout`.

Markdown (`--format md`, default):

```markdown
**SPEAKER_00** · [00:00 – 00:03]

Good morning everyone. Let's start the review.
```

Sentence-per-line markdown (`--layout sentence`):

```markdown
- [00:00 – 00:03] **SPEAKER_00**: Good morning everyone.
- [00:00 – 00:03] **SPEAKER_00**: Let's start the review.
```

`--format html` produces a self-contained color-coded page (each speaker gets
its own accent color) that renders in any browser. `--format srt` produces
subtitles: every caption is kept to ~5-7 words (`--caption-words` to change the
target) and the turn's time span is divided proportionally, so timing stays
smooth.

## Performance

Measured on this box (GTX 1070, 4 CPU cores, whisper.cpp Vulkan), 156 s Russian
call:

| Stage | Model | Time | Realtime |
| --- | --- | --- | --- |
| ASR | `small` fp16 | 14.7 s | 10.6× |
| ASR | `small-q5_1` | 14.3 s | 11.1× |
| ASR | `turbo` q5_0 | 14.7–18.0 s | 8.8–10.6× |
| Diarization | segmentation 3.0 + CAM-PPlus | 23.8 s | 6.6× (RTF 0.152) |
| **End to end** | `turbo` + diarization | **41 s** | **3.8×** |

A 93-minute call with `turbo` took 503 s without VAD and 492 s with VAD
(ASR only). Diarization is the dominant cost, not transcription. End to end that
same 93.0 min recording took **20m36s (1236 s, 4.5× realtime)** — a higher ratio
than the 3.8× above, because a wideband `.m4a` diarizes faster than the 8 kHz
phone audio the 0.152 RTF was measured on. The two clustering post-passes add
roughly 8% on top (1397 s total, 4.0× realtime).

The old faster-whisper app took 14.3 s for the same 156 s call, so the ASR
speed is a wash — the win is the 235 MB install, the absent HF token, and
working speaker labels.

## Migrating from the faster-whisper Transcriber

The folder workflow, flags and output files carry over unchanged, so `voxpipe
watch` behaves the way `transcriber watch` did. Five things differ and will bite
you if you copy your old environment:

| Difference | Detail |
| --- | --- |
| **The command is renamed** | `transcriber` → `voxpipe`, and `transcriber.api:app` → `voxpipe.api:app`. Old invocations will fail with `command not found`. |
| **The folders are renamed** | `calls-inbox/process/failed/results` → `media-inbox/process/failed/results`. VoxPipe creates the new folders and ignores the old ones, so **anything already sitting in `calls-inbox` is never picked up** — `mv` it across first. |
| **Default model** | `small` → `turbo`, silently: an old `TRANSCRIBER_WHISPER_MODEL=small` is ignored. Set `TRANSCRIBER_MODEL=small` for the old behaviour. |
| **Model cache path** | `~/.cache/transcriber/` → `~/.cache/voxpipe/`. Copy or re-fetch your models. |
| **Diarization needs retuning** | pyannote 3.1 → sherpa-onnx, so speaker counts will not match the old app's. `TRANSCRIBER_DIAR_THRESHOLD` is a **new** knob with no equivalent before, and 0.85 only suits a few-minute two-party call; two post-passes clean up long meetings (a 93-min meeting goes from 30 labels to 20). See [Tuning the speaker count](#tuning-the-speaker-count). |

| Old variable | New | Note |
| --- | --- | --- |
| `TRANSCRIBER_WHISPER_MODEL` | `TRANSCRIBER_MODEL` | **and the default changed** from `small` to `turbo` |
| `TRANSCRIBER_WHISPER_DEVICE` | `TRANSCRIBER_DEVICE` | values changed too: `cuda`/`auto` → `0` (Vulkan index) or `TRANSCRIBER_NO_GPU=1` |
| `TRANSCRIBER_WHISPER_COMPUTE_TYPE` | — | gone; quantization is baked into the GGML file you pick |
| `TRANSCRIBER_HF_TOKEN` | — | **no longer needed** |
| `TRANSCRIBER_DIARIZATION_MODEL` | `TRANSCRIBER_DIAR_SEG_MODEL` | now a local ONNX path, not a Hub id |

The API moved module and now sits behind an optional extra. Install it with
`pip install -e ".[api]"`, then `uvicorn voxpipe.api:app` becomes:

```bash
voxpipe serve                         # 127.0.0.1:8000
# or, equivalently, with any ASGI server:
.venv/bin/python -m uvicorn voxpipe.api:app --host 127.0.0.1 --port 8000
```

Same endpoints (`GET /health`, `POST /transcribe`), same response JSON, plus a
`words=false` query parameter the old one lacked. One difference: `GET /health`
here also reports `diarization_enabled`, `model`, `model_path`, `server_running`
and `model_verified`, and it answers *before* the model is loaded, so it is
usable as a readiness probe.

## Licensing and attribution

The code is MIT (`LICENSE`). **The model weights are not** — they are separate
works under their own licenses, collected in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) with the full texts:

| Model | License | Copyright |
| --- | --- | --- |
| `ggml-silero-v5.1.2.bin` (VAD) | MIT | 2020-present Silero Team |
| `sherpa-onnx-pyannote-segmentation-3-0` (segmentation) | MIT | 2022 CNRS |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (speaker embeddings) | Apache-2.0 | 3D-Speaker / ModelScope (Alibaba) |

That file is tracked in git deliberately: `vendor/` is git-ignored, so notices
kept only there would disappear on a fresh clone. `scripts/fetch-models.sh`
copies it next to the weights, and the wheel ships both it and `LICENSE` under
`dist-info/licenses/`.

## Known limitations

- **The API has no authentication and no upload size limit.** Bind it to
  loopback or front it with a reverse proxy. Uploads stream to a temp file, so
  an unbounded body fills `/tmp`.
- **Speaker identity does not persist across files** (see above).
- **Speaker count on long recordings is much better, not perfect.** The second
  clustering pass takes a 93-minute meeting from 30 labels to 20 and reunites
  the split dominant voice, but a multi-party meeting is still harder than a
  two-party call. See [Tuning the speaker count](#tuning-the-speaker-count).
- **`TRANSCRIBER_DIAR_SPEAKERS` is broken** and should be left at `-1`. Pinning
  a count makes sherpa-onnx 1.13.8 emit overlapping turns that the attribution
  step collapses to one speaker, so `=2` silently yields a single
  `SPEAKER_00`.
- **Non-ASCII filenames work**, verified with Cyrillic + spaces (`2026-09-20
  13-47-48 Стратоплан РО.m4a`) and with raw non-UTF-8 bytes, which Python's
  `surrogateescape` handling passes through to PyAV intact.
- Overlapping speech is attributed by segment midpoint, so a segment spanning a
  turn change is credited to whoever holds the midpoint. There is no
  word-level diarization.
- Diarization is CPU-only; it does not use the GPU. It does use several threads
  for the embedding models (`TRANSCRIBER_DIAR_THREADS`), but runs one recording
  at a time, like ASR.
- The vendored tree is relocatable, but the *upstream* whisper.cpp build is not:
  its shared objects carry absolute `RUNPATH`s until `scripts/vendor-server.sh`
  rewrites them to `$ORIGIN` (see above).
- **Windows GPU acceleration is available but unverified.** There is no
  first-party x64 Vulkan archive, so `install.cmd` defaults to one of two
  third-party, unsigned builds when it detects a Vulkan device, and to the
  first-party CPU archive otherwise. Their publisher states no whisper.cpp
  version, so the pin is a hash and a tag rather than a version claim. A server
  without `ggml-vulkan.dll` needs `TRANSCRIBER_NO_GPU=1` or the server prints its
  usage and refuses to start; `run.cmd` sets that itself by looking for that DLL.
  Orphan reaping is also weaker there, since there is no `/proc` to read a start
  time from. The Python side is unaffected. See
  [Platform support](#platform-support).
- **The Windows install path has not been run on real Windows hardware.** Its
  batch logic is exercised under Wine and the PowerShell extraction with host
  `pwsh` against the real upstream and Vulkan archives, but Wine cannot run a
  Windows Python and cannot run the GPU-detection code faithfully, so
  end-to-end installation, GPU detection and `whisper-server.exe` startup on a
  real Vulkan device remain unverified.

## Development

```bash
.venv/bin/pip install -e ".[dev]" -c requirements.lock
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
```

`requirements.lock` pins every transitive dependency to the versions this code was
verified against, and is applied as a **constraint** so pip still resolves the
project itself from `pyproject.toml`. The project deliberately declares only
lower bounds, which means an unconstrained install picks up whatever is newest —
a rebuild during this work resolved to `av` 18, `numpy` 2.5, `pytest` 9 and
`starlette` 1.7 all at once. That combination passes, so drift is not
automatically breakage, but PyAV, sherpa-onnx and the FastAPI stack move fast
enough that a clean install is not reproducible without the lock. To upgrade on
purpose: install without `-c`, run the offline suite **and**
`tests/test_end_to_end.py` (the only tests that touch the real GPU server), then
regenerate the lock with `pip freeze`.

The default suite needs no GPU and passes with the network unplugged (166
tests, ~17 s; 6 more collect and skip unless the e2e variables are set) —
including the API tests, which drive the FastAPI app through `TestClient` with a
stub transcriber, so they cover routing, the extension gate, error mapping and
temp-file cleanup without a GPU. They `importorskip` if the `api` extra is
missing, so a base-only install still runs the other tests instead of erroring
at collection. One test does use the network when it can: it asks PyPI what each
pin in `requirements.lock` requires, to confirm the declared Python floor is
still high enough, and skips instead of failing when it cannot.

The end-to-end suite runs the real server and real diarization, and skips
itself unless pointed at a binary and a recording. Both paths point inside this
project or anywhere else you like, so the suite runs with no sibling checkout.
It defaults to the vendored `turbo` model; set
`TRANSCRIBER_TEST_MODEL=small-q5_1` for a faster run.

```bash
TRANSCRIBER_SERVER_BIN=vendor/bin/whisper-server \
TRANSCRIBER_TEST_AUDIO=/path/to/some/call.mp3 \
  .venv/bin/pytest tests/test_end_to_end.py      # 6 tests, ~3.5 min
```

### Layout

```
VoxPipe/                    <- repository root
├── install.sh             one-command setup: venv + pinned deps + models
├── run.sh                 start the folder watcher (any voxpipe watch flag)
├── install.cmd            the same, for Windows, + the whisper-server fetch
├── run.cmd                the same, for Windows
├── media-inbox/        drop audio files here (watch mode)
├── media-process/      in transit
├── media-failed/       failures + <name>.txt error reports
├── media-results/      transcripts
├── scripts/
│   ├── fetch-models.sh  VAD + diarization + GGML weights into vendor/models
│   ├── unpack-server.ps1  verify + unpack the Windows whisper-server archive
│   └── vendor-server.sh copy a whisper.cpp build into vendor/bin, $ORIGIN RUNPATH
├── vendor/              git-ignored; everything needed to run (629 MB)
│   ├── bin/            relocatable whisper-server + its libraries
│   └── models/         VAD, diarization ONNX pair, default GGML model
├── LICENSE             MIT, for our code only
├── THIRD-PARTY-NOTICES.md  model weights + dependencies: licenses and full texts
├── src/voxpipe/
│   ├── cli.py          argument parsing, one-shot + watch + serve
│   ├── api.py          FastAPI app: GET /health, POST /transcribe
│   ├── config.py       env vars, model discovery
│   ├── core.py         decode → ASR → diarize → merge, one Transcriber
│   ├── decode.py       PyAV → 16 kHz mono float32 (in-process, no temp files)
│   ├── server.py       whisper-server lifecycle + HTTP
│   ├── asr.py          verbose_json → Segment/Word
│   ├── diarize.py      sherpa-onnx segmentation + embeddings + FastClustering
│   ├── merge.py        segments → speaker turns → utterances
│   ├── format.py       md / txt / html / srt renderers
│   └── pipeline.py     folder state machine
└── tests/
    ├── test_api.py         REST API via TestClient, stub transcriber
    ├── test_cli.py         top-level flags + one-shot/watch plumbing
    ├── test_cli_serve.py   `voxpipe serve` dispatch + missing-extra message
    ├── test_core.py        Transcriber orchestration + the serialisation lock
    ├── test_pipeline.py    folder state machine
    ├── test_asr_decode.py  parsers + wav encoding
    ├── test_diarize.py     clustering thresholds + the two cleanup passes
    ├── test_format.py      renderers
    ├── test_merge.py       speaker attribution
    ├── test_config_server.py  model discovery, server flags, orphan takeover
    ├── test_install_scripts.py  the Windows server pin + unpack script
    └── test_end_to_end.py  real server + real diarization (opt-in)
```

`format.py` and `merge.py` are adapted from the upstream `transcriber` project
(faster-whisper + pyannote, v0.2.3), which declares `license = "MIT"` in its
`pyproject.toml` but ships no LICENSE file and names no copyright holder;
`api.py` follows the same endpoints and JSON shape. All other modules are new.
See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full picture.

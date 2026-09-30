# Architecture

## Shared processing pipeline

```text
Media file
    ↓
PyAV decode
    ↓
16 kHz mono float32
    ↓
whisper.cpp / whisper-server
    ↓
Segments + word timestamps
    ↓
sherpa-onnx diarization
    ↓
Speaker assignment + cleanup
    ↓
Markdown / TXT / HTML / SRT / JSON
```

CLI, watch mode and REST API share the same core transcription pipeline.

## Warm server lifecycle

VoxPipe starts one long-lived `whisper-server` and reuses it until the requested model changes.

This avoids paying model load cost for every file.

## One GPU, one server

When port 8099 is already occupied, behavior depends on who owns the listener.

| Listener | Behavior |
|---|---|
| VoxPipe-managed server with requested model | Adopt it |
| VoxPipe-managed server with a different model | Refuse |
| Leftover VoxPipe server with requested model | Take it over and stop it on exit |
| Leftover VoxPipe server with another model | Refuse and identify the PID |
| Unknown external server | Refuse unless `TRANSCRIBER_REUSE_SERVER=1` |

Two independent model copies on one GPU can waste memory or compete for the same device.

Upstream `whisper-server` `/health` reports only generic status and does not identify its loaded model.

Therefore reuse of an unknown server cannot verify that the requested model matches.

When `TRANSCRIBER_REUSE_SERVER=1` is used for an unknown server, API health reports:

```text
model_verified=false
```

rather than silently implying certainty.

## Hard-kill recovery

Normal exits, `Ctrl-C` and `SIGTERM` allow VoxPipe to stop its managed server.

A `SIGKILL`, OOM kill or untrappable abrupt termination can leave the child process alive.

VoxPipe records managed server information under:

```text
~/.cache/voxpipe/servers/<host>-<port>.json
```

Override with:

```text
TRANSCRIBER_SERVER_STATE_DIR
```

On Linux the record includes:

- PID
- model
- process start time from `/proc`

The next run can confirm that the PID still represents the same process before reclaiming it.

A dead PID is ignored.

A recycled PID belonging to another process is not treated as a VoxPipe orphan.

On Windows, `/proc` start time is unavailable, so orphan identification is weaker.

Manual cleanup example:

```bash
kill $(pgrep -f whisper-server)
```

Alternative port:

```bash
TRANSCRIBER_PORT=8100 voxpipe meeting.mp3
```

## Folder state machine

The watch-mode state machine uses:

```text
media-inbox/
media-process/
media-results/
media-failed/
```

Detailed movement, write-in-progress detection, collision handling and crash recovery are documented in [folder-workflow.md](folder-workflow.md).

## Language handling

`whisper-server` defaults to English in a way that can silently translate non-English recordings.

VoxPipe therefore explicitly sends automatic language detection when the user does not pass `--language`.

If transcripts unexpectedly come out translated into English, inspect this behavior first.

## Repository module layout

Core modules from the original repository:

```text
src/voxpipe/
├── cli.py        argument parsing; one-shot/watch/serve
├── api.py        FastAPI app
├── config.py     environment variables and model discovery
├── core.py       decode -> ASR -> diarize -> merge orchestration
├── decode.py     PyAV -> 16 kHz mono float32
├── server.py     whisper-server lifecycle + HTTP
├── asr.py        verbose_json -> Segment/Word
├── diarize.py    sherpa-onnx segmentation, embeddings and clustering
├── merge.py      segments -> speaker turns -> utterances
├── format.py     md/txt/html/srt renderers
└── pipeline.py   folder state machine
```

See [development.md](development.md) for the full project/test layout.

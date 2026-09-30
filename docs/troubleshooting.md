# Troubleshooting

## `voxpipe: command not found`

Activate the virtual environment:

```bash
source .venv/bin/activate
```

or run the executable directly:

```bash
.venv/bin/voxpipe ...
```

Optionally add it to your shell PATH:

```bash
echo 'export PATH="/path/to/VoxPipe/.venv/bin:$PATH"' >> ~/.bashrc
```

## whisper-server is missing

Set the server path explicitly:

```bash
TRANSCRIBER_SERVER_BIN=/path/to/whisper-server voxpipe meeting.mp3
```

Automatic search order is documented in [installation.md](installation.md).

## CPU Windows build exits immediately

A CPU-only `whisper-server` does not accept the Vulkan `-dev` option.

Use:

```text
TRANSCRIBER_NO_GPU=1
```

`run.cmd` normally sets this automatically when `ggml-vulkan.dll` is absent.

## Windows health check times out immediately

Check whether the Visual C++ Redistributable is installed.

The server may require:

```text
MSVCP140.dll
VCRUNTIME140.dll
```

## GPU mode fails

Check:

- Vulkan-capable driver;
- Vulkan runtime;
- server binary built with Vulkan;
- required shared libraries/DLLs;
- `TRANSCRIBER_DEVICE`;
- `TRANSCRIBER_NO_GPU`.

## Wrong model appears to be reused

Check:

- `TRANSCRIBER_MODEL`;
- current `whisper-server` process;
- server state under `TRANSCRIBER_SERVER_STATE_DIR`;
- `TRANSCRIBER_REUSE_SERVER`;
- API `model_verified`.

Unknown external servers are intentionally not trusted by default.

## Stale whisper-server remains after a crash

Normal shutdown should stop the managed server.

Hard kills cannot be trapped.

On Linux, VoxPipe records state so the next run can reclaim a matching orphan.

Manual cleanup:

```bash
kill $(pgrep -f whisper-server)
```

or use another port:

```bash
TRANSCRIBER_PORT=8100 voxpipe meeting.mp3
```

## Transcript unexpectedly comes out in English

`whisper-server` has an English default that can translate instead of merely transcribe.

VoxPipe explicitly sends auto-detection when `--language` is absent.

If behavior looks wrong, test with an explicit hint:

```bash
voxpipe meeting.mp3 --language ru
```

## Too many speaker labels

Inspect:

```text
TRANSCRIBER_DIAR_THRESHOLD
TRANSCRIBER_DIAR_MERGE_THRESHOLD
TRANSCRIBER_DIAR_MIN_REGION
TRANSCRIBER_DIAR_MIN_ON
TRANSCRIBER_DIAR_MIN_OFF
```

See [diarization.md](diarization.md).

## Fixed speaker count produces bad output

Keep:

```text
TRANSCRIBER_DIAR_SPEAKERS=-1
```

The fixed-count path was experimentally unreliable with sherpa-onnx 1.13.8.

## File remains in media-inbox

The watcher may be waiting for the file to stop changing.

It checks size and modification time before processing.

After three settled-but-unreadable scans, the file should move to `media-failed/`.

## File remains in media-process

Interrupted jobs are returned to `media-inbox/` on watcher startup.

Do not manually move a file while an active worker is still processing it.

## Filename collisions

Inputs such as:

```text
call.m4a
call.mp3
```

are disambiguated in output naming.

See [folder-workflow.md](folder-workflow.md).

## API exposure

The API is unauthenticated and has no upload-size limit.

Do not expose it directly to the public internet.

# Known Limitations

This document preserves limitations that were previously spread through the README.

## REST API

The API has:

- no authentication;
- no upload-size limit.

Uploads stream to a temporary file, so an unbounded request can fill the temporary filesystem.

Keep the API on loopback or place it behind an authenticated reverse proxy.

## Speaker identity

Speaker labels do not persist across files.

VoxPipe performs diarization, not speaker recognition.

## Long recordings

The diarization cleanup passes substantially reduce fragmentation but do not make long multi-party meetings perfect.

The original 93-minute meeting went from 30 raw labels to 20 after cleanup, while reuniting the split dominant speaker.

## Fixed speaker count

`TRANSCRIBER_DIAR_SPEAKERS` is unreliable with the sherpa-onnx 1.13.8 behavior observed during development.

Leave it at:

```text
-1
```

until retested.

## Non-ASCII filenames

Non-ASCII filenames were verified with Cyrillic + spaces and with raw non-UTF-8 bytes.

Python `surrogateescape` handling allowed those names to pass through to PyAV in the original tests.

This is a useful positive compatibility note rather than a limitation, but it was originally documented under limitations and is preserved here.

## Overlapping speech

There is no word-level speaker diarization.

Speech spanning a speaker transition is attributed using segment-level logic.

## Diarization performance

Diarization is CPU-only.

It uses multiple embedding-model threads through `TRANSCRIBER_DIAR_THREADS`, but VoxPipe processes one recording at a time.

## Upstream whisper.cpp relocatability

The VoxPipe vendored Linux tree is made relocatable.

A raw upstream whisper.cpp build may contain absolute RUNPATHs until `scripts/vendor-server.sh` rewrites them to `$ORIGIN`.

## Windows GPU support

Windows Vulkan GPU acceleration is available but has not been verified end-to-end on real Windows Vulkan hardware.

There is no first-party x64 Windows Vulkan archive.

The installer therefore relies on third-party unsigned Vulkan builds or a user-built binary for GPU operation.

Orphan reclamation is weaker on Windows because `/proc` process start times are unavailable.

See [windows.md](windows.md).

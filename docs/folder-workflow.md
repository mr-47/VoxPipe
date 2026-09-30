# Folder Workflow

`voxpipe watch` moves media through four directories below the selected base directory.

The default base directory is the current directory and can be changed with `--dir`.

| Folder | Purpose |
|---|---|
| `media-inbox/` | New files waiting to be processed |
| `media-process/` | File currently in transit/processing |
| `media-failed/` | Failed files plus error reports |
| `media-results/` | Successfully processed media and transcripts |

All four directories are created automatically if missing.

## Movement semantics

Files are moved, not copied.

On success, the original media file lands in `media-results/` next to:

- `<name>.json`
- requested human-readable transcript formats

On failure, the original file is moved to `media-failed/`, and a text error report containing the traceback is written next to it.

## Files still being written

The watcher does not immediately fail a file that is still being written.

It checks that the file has stopped changing before decoding it.

Both file size and modification time are considered.

This is important for formats such as M4A, whose final index can be written only when recording finishes.

Deleting and re-dropping a file under the same name is treated as a new file even if it has the same length, because modification time is also considered.

If a file remains unreadable at an unchanged size for three consecutive settled scans, it is moved to `media-failed/`.

## Filename collisions

Two recordings can share the same stem:

```text
call.m4a
call.mp3
```

Without disambiguation they would compete for `call.md`, `call.txt`, etc.

The second conflicting input is therefore disambiguated with its extension, for example:

```text
call_mp3.md
call_mp3.txt
```

Re-running the same recording overwrites its own output rather than creating incrementing names such as `call_mp3_2.md`.

If the failed input itself is `.txt`, the error report uses:

```text
<name>.error.txt
```

so the report cannot overwrite the source file.

## Crash recovery

Any files left in `media-process/` after a crash are returned to `media-inbox/` on the next startup.

Unsupported file types are moved to `media-failed/` with a reason.

The intended invariant is that files are never silently dropped.

## Usage

```bash
voxpipe watch
voxpipe watch --dir /data/calls
voxpipe watch --once
voxpipe watch --interval 1
voxpipe watch --model small-q5_1
voxpipe watch --language en
voxpipe watch --no-diarization
```

Multiple output formats can be selected:

```bash
voxpipe watch --once --format txt,md,html,srt
voxpipe watch --once --format txt --format html
```

## Output behavior

Watch mode always stores JSON with word timestamps because the JSON file acts as the durable machine-readable archive.

See [output-formats.md](output-formats.md).

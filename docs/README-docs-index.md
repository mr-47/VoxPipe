# VoxPipe Documentation

The project README should remain a concise landing page.

Detailed information from the original README is preserved in these documents rather than deleted.

## User documentation

- [Installation](installation.md) — requirements, dependency footprint, whisper-server, models
- [Windows support](windows.md) — Windows CPU/GPU behavior, third-party Vulkan builds, runtime requirements
- [Configuration](configuration.md) — complete environment-variable reference
- [REST API](api.md) — endpoints, worker model, readiness, shutdown and security
- [Folder workflow](folder-workflow.md) — watcher state machine, write-in-progress detection, collisions and recovery
- [Output formats](output-formats.md) — JSON schema, Markdown/TXT/HTML/SRT behavior
- [Troubleshooting](troubleshooting.md) — common runtime failures
- [Known limitations](known-limitations.md) — explicit current limitations

## Technical documentation

- [Architecture](architecture.md) — shared pipeline, warm server lifecycle and orphan recovery
- [Speaker diarization](diarization.md) — thresholds, long-meeting experiments, cleanup passes and fixed-count issue
- [Performance](performance.md) — historical benchmark data
- [Migration](migration.md) — differences from the earlier Transcriber application
- [Development](development.md) — dependency locking, tests and project layout
- [Licensing and attribution](licensing.md) — model licenses and adapted-code provenance

## Preservation policy

For now, content moved out of the original README should not be deleted from project documentation.

Information should only be removed completely after it is intentionally classified as one of:

1. obsolete and no longer relevant to any supported release;
2. contradicted by the current implementation;
3. duplicate content that adds no operational or historical value;
4. generated reliably elsewhere, such as CLI/API reference produced from the application itself.

When removing historical benchmark or implementation notes, prefer moving them into a changelog, release note or archived engineering note rather than silently deleting them.

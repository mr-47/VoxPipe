# REST API

Install API support:

```bash
pip install -e ".[api]"
```

Start the API:

```bash
voxpipe serve
```

Default address:

```text
http://127.0.0.1:8000
```

Interactive OpenAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## CLI options

`voxpipe serve` accepts:

- `--host`
- `--port`
- `--log-level`
- `--model`
- `--language`
- `--no-diarization`

The same environment variables used by the CLI are honored.

`--language` acts as a default language hint for requests that omit the `language` query parameter.

## One worker by design

The API intentionally runs one worker.

A second worker would load a second model copy onto the same GPU.

Concurrent requests are serialized and queue inside the process instead of competing for the device.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service/model status |
| `POST /transcribe` | Multipart upload -> transcript JSON |

### `GET /health`

The health response includes:

- `status`
- `diarization_enabled`
- `model`
- `model_path`
- `server_running`
- `model_verified`

`GET /health` responds before the model has been loaded.

Until the first transcription starts the server, `server_running` can be `false`.

This makes the endpoint useful as a readiness probe without forcing the model-load delay.

`model_verified` becomes `false` when VoxPipe is configured to reuse a `whisper-server` it did not start, because upstream `/health` does not identify the loaded model.

### `POST /transcribe`

Query parameters:

| Parameter | Default | Description |
|---|---|---|
| `language` | auto | Language hint such as `en`, `ru`, `de`; omitted means auto-detection |
| `words` | `true` | Include word timestamps under `segments[].words` |

Examples:

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

Health:

```bash
curl http://127.0.0.1:8000/health
```

The response JSON is the same schema used by CLI `--json` and watch-mode JSON output.

The current API includes the complete `text` field; the older Transcriber API omitted it.

## ASGI application

`voxpipe.api:app` is a plain ASGI app.

It can be served by another compatible ASGI server:

```bash
.venv/bin/python -m uvicorn voxpipe.api:app --host 127.0.0.1 --port 8000
```

## Shutdown behavior

On `Ctrl-C` or `SIGTERM`, VoxPipe stops the `whisper-server` it owns rather than intentionally leaving a GPU process behind.

A `SIGKILL`, OOM kill or abruptly closed environment cannot be trapped; orphan recovery is documented in [architecture.md](architecture.md).

## Security

The API has:

- no authentication;
- no upload-size limit.

Uploads stream to a temporary file, so an unbounded request can fill the temporary filesystem.

Keep the service bound to `127.0.0.1`, or place it behind an authenticated reverse proxy before binding to an externally reachable interface.

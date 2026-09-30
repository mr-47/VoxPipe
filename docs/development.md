# Development

Install development dependencies:

```bash
.venv/bin/pip install -e ".[dev]" -c requirements.lock
```

Run checks:

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
```

## Dependency locking

`requirements.lock` pins transitive dependency versions that were verified together.

It is applied as a constraint so the project itself is still resolved from `pyproject.toml`.

The project intentionally declares lower bounds rather than exact versions.

The original README noted that an unconstrained rebuild had resolved to newer versions including:

- `av` 18
- `numpy` 2.5
- `pytest` 9
- `starlette` 1.7

and that combination passed tests.

That does not make unconstrained installs reproducible.

For an intentional dependency upgrade:

1. install without `-c requirements.lock`;
2. run the offline suite;
3. run `tests/test_end_to_end.py`;
4. regenerate the lock with `pip freeze`.

## Test suite

The historical default suite:

- needed no GPU;
- passed with the network unplugged;
- contained 166 normal tests taking roughly 17 seconds;
- collected 6 additional end-to-end tests that skipped unless e2e variables were configured.

API tests drive FastAPI via `TestClient` with a stub transcriber.

They cover routing, extension validation, error mapping and temporary-file cleanup without requiring a GPU.

API tests use `importorskip` when the API extra is not installed, so a base-only environment can still run the rest of the test suite.

One dependency-validation test uses the network when available to inspect PyPI metadata for pinned requirements and confirm the declared Python floor. It skips instead of failing when the network is unavailable.

## End-to-end tests

The e2e suite uses the real whisper server and real diarization.

It skips unless supplied both a server binary and a recording.

Example:

```bash
TRANSCRIBER_SERVER_BIN=vendor/bin/whisper-server \
TRANSCRIBER_TEST_AUDIO=/path/to/some/call.mp3 \
  .venv/bin/pytest tests/test_end_to_end.py
```

The original suite contained 6 e2e tests and took roughly 3.5 minutes in the documented environment.

It defaults to the vendored `turbo` model.

Use:

```text
TRANSCRIBER_TEST_MODEL=small-q5_1
```

for a faster test model.

## Repository layout

```text
VoxPipe/
├── install.sh
├── run.sh
├── install.cmd
├── run.cmd
├── media-inbox/
├── media-process/
├── media-failed/
├── media-results/
├── scripts/
│   ├── fetch-models.sh
│   ├── unpack-server.ps1
│   └── vendor-server.sh
├── vendor/
│   ├── bin/
│   └── models/
├── LICENSE
├── THIRD-PARTY-NOTICES.md
├── src/voxpipe/
│   ├── cli.py
│   ├── api.py
│   ├── config.py
│   ├── core.py
│   ├── decode.py
│   ├── server.py
│   ├── asr.py
│   ├── diarize.py
│   ├── merge.py
│   ├── format.py
│   └── pipeline.py
└── tests/
    ├── test_api.py
    ├── test_cli.py
    ├── test_cli_serve.py
    ├── test_core.py
    ├── test_pipeline.py
    ├── test_asr_decode.py
    ├── test_diarize.py
    ├── test_format.py
    ├── test_merge.py
    ├── test_config_server.py
    ├── test_install_scripts.py
    └── test_end_to_end.py
```

Responsibilities:

- `cli.py` — argument parsing and one-shot/watch/serve dispatch
- `api.py` — FastAPI application
- `config.py` — environment variables and model discovery
- `core.py` — high-level transcription orchestration
- `decode.py` — in-process PyAV decode to 16 kHz mono float32
- `server.py` — `whisper-server` lifecycle and HTTP communication
- `asr.py` — whisper verbose JSON -> segment/word objects
- `diarize.py` — segmentation, embeddings and clustering
- `merge.py` — speaker turns and utterances
- `format.py` — md/txt/html/srt rendering
- `pipeline.py` — folder state machine

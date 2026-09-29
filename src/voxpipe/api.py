"""REST API, a thin HTTP skin over the same :class:`Transcriber` the CLI uses.

Design notes:

* The transcriber is created once per app (lifespan) and reused, so the GGML
  model is loaded on the first request and stays warm afterwards. Creating it
  per request would pay the ~15 s model load every time.
* Transcription is CPU/GPU-bound and synchronous, so it runs in a worker thread
  (``anyio.to_thread.run_sync``) instead of blocking the event loop.
* ``Transcriber`` serialises calls internally with a lock, because a single
  ``whisper-server`` process handles one inference at a time. Concurrent uploads
  queue up rather than fighting over the GPU.
* Uploads are streamed to a temp file, because PyAV needs a seekable path. The
  temp file is always removed, including on failure.
* There is **no authentication**. Bind to loopback, or put this behind a reverse
  proxy, before exposing it to a network.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import anyio

from . import __version__
from .config import SUPPORTED_AUDIO_SUFFIXES, TranscriberSettings
from .core import Transcriber
from .models import Transcript

try:
    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
    # Fail with something actionable instead of a bare "No module named 'fastapi'".
    raise ModuleNotFoundError(
        "The REST API needs the optional 'api' extra.\n"
        "    pip install -e '.[api]'\n"
        "The CLI and the media-* folder workflow do not need it."
    ) from exc

logger = logging.getLogger(__name__)

UPLOAD_CHUNK = 1024 * 1024
API_EXTRA_HINT = "The REST API needs the optional 'api' extra: pip install -e '.[api]'"


class HealthResponse(BaseModel):
    status: str
    diarization_enabled: bool
    model: str
    model_path: str | None = None
    server_running: bool
    #: False when an adopted server is in use: its /health endpoint does not
    #: report a model, so `model` is an assumption, not a fact.
    model_verified: bool = True


class WordOut(BaseModel):
    start: float
    end: float
    word: str


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)


class UtteranceOut(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptOut(BaseModel):
    """Same shape as the JSON written by ``--json`` and ``media-results/``."""

    language: str | None
    language_probability: float | None
    duration: float
    segments: list[SegmentOut]
    utterances: list[UtteranceOut]
    text: str


def _to_out(result: Transcript) -> TranscriptOut:
    payload = result.to_dict(include_words=True)
    return TranscriptOut(**payload)


def create_app(
    transcriber: Transcriber | None = None,
    settings: TranscriberSettings | None = None,
    diarize: bool = True,
    default_language: str | None = None,
) -> FastAPI:
    """Build the app. Pass ``transcriber`` to inject a stub (tests); pass
    ``settings`` to configure the real one, ``diarize=False`` to attribute every
    utterance to a single speaker, and ``default_language`` to apply a language
    hint to requests that omit ``?language=``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = transcriber is None
        app.state.transcriber = transcriber or Transcriber(settings or TranscriberSettings(), diarize=diarize)
        app.state.owns_transcriber = owned
        try:
            yield
        finally:
            # Stops whisper-server, so a reload does not leak a GPU process.
            if owned:
                app.state.transcriber.close()

    app = FastAPI(
        title="VoxPipe",
        version=__version__,
        description="Audio transcription with speaker diarization (whisper.cpp + Vulkan)",
        lifespan=lifespan,
    )

    def get_transcriber() -> Transcriber:
        try:
            return app.state.transcriber
        except AttributeError:  # pragma: no cover - only if lifespan is bypassed
            raise HTTPException(status_code=503, detail="Service is still starting up")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        engine = get_transcriber()
        try:
            model_path: str | None = str(engine.settings.resolve_model_path())
        except FileNotFoundError:
            model_path = None
        return HealthResponse(
            status="ok",
            diarization_enabled=engine.diarization_enabled,
            model=engine.settings.model,
            model_path=model_path,
            server_running=engine.server.is_running,
            model_verified=engine.server.model_is_verified,
        )

    @app.post("/transcribe", response_model=TranscriptOut, tags=["transcription"])
    async def handle_transcribe(
        request: Request,
        file: UploadFile = File(..., description="Audio file to transcribe"),
        language: str | None = Query(
            default=None,
            description=(
                "Optional language hint, e.g. 'en', 'de', 'ru'. Omit to use the server default "
                f"({default_language or 'auto-detect'})."
            ),
        ),
        words: bool = Query(default=True, description="Include per-word timestamps"),
    ) -> TranscriptOut:
        engine = get_transcriber()

        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix or 'unknown'}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_SUFFIXES))}"
                ),
            )

        # The handle stays open for the whole upload: the `with` block is what
        # flushes the last chunk to disk before PyAV tries to read the file.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                size = 0
                while chunk := await file.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    tmp.write(chunk)
            finally:
                await file.close()
        try:
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            logger.info(
                "POST /transcribe %s (%d bytes) from %s",
                file.filename,
                size,
                request.client.host if request.client else "unknown",
            )
            try:
                result = await anyio.to_thread.run_sync(
                    engine.transcribe,
                    str(tmp_path),
                    language or default_language,
                )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as 500 below
                logger.exception("Transcription failed")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

            out = _to_out(result)
            if not words:
                out.segments = [SegmentOut(start=s.start, end=s.end, text=s.text) for s in out.segments]
            return out
        finally:
            tmp_path.unlink(missing_ok=True)

    return app


app = create_app()

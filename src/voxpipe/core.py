"""Orchestration: decode -> whisper.cpp server -> diarization -> attribution.

The whisper model lives in a separate ``whisper-server`` process so the
expensive load happens once per run instead of once per file; the ONNX
diarization models live in-process. Inference is serialized with a lock because
neither the sherpa-onnx pipeline nor a single server slot is safe to drive
concurrently.
"""

from __future__ import annotations

import logging
import threading
import time

from .asr import language_of, parse_segments
from .config import SAMPLE_RATE, TranscriberSettings
from .decode import decode, duration_seconds, to_wav_bytes
from .diarize import Diarizer
from .merge import assign_speakers, build_utterances
from .models import Transcript
from .server import WhisperServer

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio files and attributes speech to speakers."""

    def __init__(self, settings: TranscriberSettings | None = None, diarize: bool = True) -> None:
        self._settings = settings or TranscriberSettings()
        self._settings.resolve_diarization_models()
        self._diarize = diarize and self._settings.diarization.available
        if diarize and not self._settings.diarization.available:
            logger.warning(
                "Diarization requested but its ONNX models are missing; "
                "every utterance will be attributed to SPEAKER_00"
            )
        self._server = WhisperServer(self._settings)
        self._diarizer = Diarizer(self._settings.diarization)
        self._lock = threading.Lock()

    @property
    def settings(self) -> TranscriberSettings:
        return self._settings

    @property
    def server(self) -> WhisperServer:
        """The whisper-server handle, for status reporting (see ``/health``)."""
        return self._server

    @property
    def diarization_enabled(self) -> bool:
        return self._diarize

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        started = time.monotonic()
        with self._lock:
            samples = decode(audio_path, SAMPLE_RATE)
            duration = duration_seconds(samples, SAMPLE_RATE)
            logger.info(
                "Decoded %s -> %.1fs of %d Hz mono audio in %.1fs",
                audio_path,
                duration,
                SAMPLE_RATE,
                time.monotonic() - started,
            )

            asr_started = time.monotonic()
            payload = self._server.transcribe(to_wav_bytes(samples), language=language)
            asr_elapsed = time.monotonic() - asr_started
            logger.info(
                "whisper.cpp returned %d segment(s) in %.1fs (%.2fx realtime)",
                len(payload.get("segments") or []),
                asr_elapsed,
                duration / asr_elapsed if asr_elapsed else 0.0,
            )

            segments = parse_segments(payload)

            turns = []
            if self._diarize:
                diar_started = time.monotonic()
                turns = self._diarizer.diarize(samples, SAMPLE_RATE)
                logger.info(
                    "Diarization finished in %.1fs (%.2fx realtime)",
                    time.monotonic() - diar_started,
                    duration / max(1e-9, time.monotonic() - diar_started),
                )

        speakers = assign_speakers(segments, turns)
        utterances = build_utterances(segments, speakers)
        detected, probability = language_of(payload)
        logger.info(
            "Total %.1fs for %.1fs of audio (%.2fx realtime, %d utterances)",
            time.monotonic() - started,
            duration,
            duration / max(1e-9, time.monotonic() - started),
            len(utterances),
        )
        return Transcript(
            language=detected,
            language_probability=probability,
            duration=duration,
            segments=segments,
            utterances=utterances,
        )

    def close(self) -> None:
        self._server.stop()

    def __enter__(self) -> "Transcriber":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

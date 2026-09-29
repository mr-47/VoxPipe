"""REST API: health, transcription, error handling, temp-file hygiene.

Uses a stub transcriber, so this runs offline in milliseconds and never touches
whisper-server or the GPU. The real pipeline is covered by test_end_to_end.py.
"""

import io
import wave
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="REST API needs the 'api' extra")

from fastapi.testclient import TestClient  # noqa: E402

from voxpipe.api import create_app  # noqa: E402
from voxpipe.config import TranscriberSettings  # noqa: E402
from voxpipe.models import Segment, Transcript, Utterance, Word  # noqa: E402

RESULT = Transcript(
    language="en",
    language_probability=0.97,
    duration=1.5,
    segments=[
        Segment(
            start=0.0,
            end=1.5,
            text="hello there",
            words=[Word(start=0.0, end=0.5, word="hello"), Word(start=0.5, end=1.5, word="there")],
        )
    ],
    utterances=[Utterance(speaker="SPEAKER_00", start=0.0, end=1.5, text="hello there")],
)


class _StubServer:
    is_running = False
    model_is_verified = True


class _StubTranscriber:
    """Stands in for core.Transcriber."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str | None]] = []
        self.seen: list[int] = []  # size of the file on disk, in bytes
        self.closed = False
        self.settings = TranscriberSettings(model="stub", model_dirs=[])
        self.server = _StubServer()

    @property
    def diarization_enabled(self) -> bool:
        return True

    def transcribe(self, path: str, language: str | None = None) -> Transcript:
        self.calls.append((path, language))
        self.seen.append(Path(path).stat().st_size if Path(path).is_file() else -1)
        if self.fail:
            raise RuntimeError("boom")
        return RESULT

    def close(self) -> None:
        self.closed = True


def _wav_bytes(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def _client(stub: _StubTranscriber) -> TestClient:
    return TestClient(create_app(transcriber=stub))


def test_health_reports_configuration():
    with _client(_StubTranscriber()) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["diarization_enabled"] is True
    assert body["model"] == "stub"
    assert body["server_running"] is False
    assert body["model_verified"] is True


def test_health_flags_an_adopted_server_as_unverified():
    """An adopted server's model is an assumption, and /health must say so."""
    stub = _StubTranscriber()
    stub.server.model_is_verified = False

    with _client(stub) as client:
        body = client.get("/health").json()

    assert body["model_verified"] is False
    assert (
        "model_verified" in client.get("/openapi.json").json()["components"]["schemas"]["HealthResponse"]["properties"]
    )


def test_transcribe_returns_the_transcript():
    stub = _StubTranscriber()
    with _client(stub) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert body["duration"] == 1.5
    assert body["utterances"][0]["speaker"] == "SPEAKER_00"
    assert body["segments"][0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "hello"},
        {"start": 0.5, "end": 1.5, "word": "there"},
    ]
    assert body["text"] == "SPEAKER_00: hello there"


def test_words_false_drops_word_timestamps():
    with _client(_StubTranscriber()) as client:
        body = client.post(
            "/transcribe?words=false",
            files={"file": ("call.wav", _wav_bytes(), "audio/wav")},
        ).json()

    assert body["segments"][0]["text"] == "hello there"
    assert body["segments"][0]["words"] == []


def test_language_hint_is_forwarded():
    stub = _StubTranscriber()
    with _client(stub) as client:
        client.post(
            "/transcribe?language=ru",
            files={"file": ("call.wav", _wav_bytes(), "audio/wav")},
        )

    assert stub.calls[0][1] == "ru"


def test_language_defaults_to_none_so_the_server_auto_detects():
    stub = _StubTranscriber()
    with _client(stub) as client:
        client.post("/transcribe", files={"file": ("call.wav", _wav_bytes(), "audio/wav")})

    assert stub.calls[0][1] is None


def test_serve_default_language_applies_when_the_request_omits_it():
    """`voxpipe serve --language ru` should not need ?language= on every call."""
    stub = _StubTranscriber()
    with TestClient(create_app(transcriber=stub, default_language="ru")) as client:
        client.post("/transcribe", files={"file": ("call.wav", _wav_bytes(), "audio/wav")})

    assert stub.calls[0][1] == "ru"


def test_request_language_overrides_the_serve_default():
    stub = _StubTranscriber()
    with TestClient(create_app(transcriber=stub, default_language="ru")) as client:
        client.post(
            "/transcribe?language=en",
            files={"file": ("call.wav", _wav_bytes(), "audio/wav")},
        )

    assert stub.calls[0][1] == "en"


def test_m4a_upload_is_accepted():
    with _client(_StubTranscriber()) as client:
        # Not decodable, but the stub never decodes it: this checks the
        # extension gate, which is what whisper.cpp cannot do on its own.
        response = client.post(
            "/transcribe",
            files={"file": ("call.m4a", b"\x00" * 128, "audio/mp4")},
        )

    assert response.status_code == 200


def test_unsupported_extension_is_rejected(tmp_path: Path):
    with _client(_StubTranscriber()) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_extension_check_is_case_insensitive():
    with _client(_StubTranscriber()) as client:
        response = client.post("/transcribe", files={"file": ("CALL.MP3", b"\x00" * 64, "audio/mpeg")})

    assert response.status_code == 200


def test_empty_upload_is_rejected():
    with _client(_StubTranscriber()) as client:
        response = client.post("/transcribe", files={"file": ("call.wav", b"", "audio/wav")})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcription_failure_is_a_500():
    with _client(_StubTranscriber(fail=True)) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_upload_is_written_to_disk_and_then_cleaned_up():
    payload = _wav_bytes()
    stub = _StubTranscriber()
    with _client(stub) as client:
        client.post("/transcribe", files={"file": ("call.wav", payload, "audio/wav")})
        used = Path(stub.calls[0][0])

    # The transcriber saw a complete file...
    assert stub.seen[0] == len(payload)
    # ...and it is gone once the request completes.
    assert not used.exists(), "temp file was left behind"


def test_temp_file_is_removed_even_when_transcription_fails():
    stub = _StubTranscriber(fail=True)
    with _client(stub) as client:
        client.post("/transcribe", files={"file": ("call.wav", _wav_bytes(), "audio/wav")})
        used = Path(stub.calls[0][0])

    assert not used.exists()


def test_owned_transcriber_is_closed_on_shutdown(monkeypatch):
    """A transcriber the app created must be closed, so no GPU process leaks."""
    created: list[_StubTranscriber] = []
    seen: list[dict] = []

    def factory(settings, diarize=True):
        seen.append({"diarize": diarize})
        stub = _StubTranscriber()
        stub.settings = settings
        created.append(stub)
        return stub

    monkeypatch.setattr("voxpipe.api.Transcriber", factory)

    with TestClient(create_app(settings=TranscriberSettings(model="stub", model_dirs=[]))):
        assert created and created[0].closed is False

    assert created[0].closed is True
    assert seen == [{"diarize": True}]


def test_diarize_false_reaches_the_transcriber(monkeypatch):
    """`voxpipe serve --no-diarization` must not silently keep diarizing."""
    seen: list[bool] = []

    def factory(settings, diarize=True):
        seen.append(diarize)
        return _StubTranscriber()

    monkeypatch.setattr("voxpipe.api.Transcriber", factory)

    with TestClient(create_app(settings=TranscriberSettings(model="stub", model_dirs=[]), diarize=False)):
        pass

    assert seen == [False]


def test_injected_transcriber_is_not_closed_by_the_lifespan():
    stub = _StubTranscriber()
    with _client(stub):
        pass

    assert stub.closed is False, "the app does not own an injected transcriber"


def test_openapi_schema_documents_the_response():
    with _client(_StubTranscriber()) as client:
        schema = client.get("/openapi.json").json()

    assert "/transcribe" in schema["paths"]
    assert schema["paths"]["/transcribe"]["post"]["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("TranscriptOut")


def test_missing_file_field_is_a_422():
    with _client(_StubTranscriber()) as client:
        response = client.post("/transcribe")

    assert response.status_code == 422


def test_filename_without_extension_is_a_400():
    with _client(_StubTranscriber()) as client:
        response = client.post("/transcribe", files={"file": ("call", _wav_bytes(), "audio/wav")})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

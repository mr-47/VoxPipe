"""`Transcriber` orchestration: decode -> server -> diarize -> attribute.

`core.py` is the one module that is never exercised offline -- the API tests
all inject a stub engine, and the real class is only touched by the e2e tier,
which drives it sequentially, one file at a time. So the two things that make
it worth testing are untested: the lock that serialises inference, and the
order in which the stages are wired together.

Neither test here loads a model or binds a port: `WhisperServer` and `Diarizer`
are replaced at the point `core` imports them.
"""

from __future__ import annotations

import threading
import time

import pytest

from voxpipe import core
from voxpipe.config import TranscriberSettings
from voxpipe.core import Transcriber
from voxpipe.models import SpeakerTurn

PAYLOAD = {
    "language": "ru",
    "language_probability": 0.97,
    "segments": [
        {"start": 0.0, "end": 1.0, "text": " first"},
        {"start": 1.0, "end": 2.0, "text": " second"},
    ],
}


class _StubServer:
    """Records concurrency so a missing lock is visible rather than implied."""

    def __init__(self, settings: TranscriberSettings, hold: float = 0.05) -> None:
        self.settings = settings
        self.hold = hold
        self.model_is_verified = True
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self.stopped = False
        self._guard = threading.Lock()

    def transcribe(self, wav: bytes, language: str | None = None) -> dict:  # noqa: ARG002
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.calls += 1
        try:
            time.sleep(self.hold)
            return PAYLOAD
        finally:
            with self._guard:
                self.in_flight -= 1

    def stop(self) -> None:
        self.stopped = True


class _StubDiarizer:
    def __init__(self, settings: object, turns: list[SpeakerTurn] | None = None) -> None:  # noqa: ARG002
        self.turns = turns if turns is not None else [SpeakerTurn(0.0, 2.0, "SPEAKER_01")]
        self.calls = 0

    def diarize(self, samples: object, sample_rate: int) -> list[SpeakerTurn]:  # noqa: ARG002
        self.calls += 1
        return self.turns


def _settings(tmp_path, *, diarization_available: bool = True) -> TranscriberSettings:
    """A settings object that looks real but never touches a model or a port."""
    settings = TranscriberSettings()
    settings.model = str(tmp_path / "fake-model.bin")
    settings.model_dirs = [tmp_path]
    if diarization_available:
        seg = tmp_path / "seg.onnx"
        emb = tmp_path / "emb.onnx"
        seg.write_bytes(b"x")
        emb.write_bytes(b"x")
        settings.diarization.segmentation_model = seg
        settings.diarization.embedding_model = emb
    else:
        settings.diarization.segmentation_model = None
        settings.diarization.embedding_model = None
    return settings


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch):
    """Build a `Transcriber` with both slow stages replaced, and hand back the stubs."""
    built: dict[str, object] = {}

    def make(settings: TranscriberSettings, hold: float = 0.05, turns=None):
        server = _StubServer(settings, hold=hold)

        def server_factory(s: TranscriberSettings) -> _StubServer:
            return server

        def diarizer_factory(s: object) -> _StubDiarizer:
            return _StubDiarizer(s, turns=turns)

        monkeypatch.setattr(core, "WhisperServer", server_factory)
        monkeypatch.setattr(core, "Diarizer", diarizer_factory)
        monkeypatch.setattr(core, "decode", lambda *_a, **_k: [0.0] * 16000)
        monkeypatch.setattr(core, "to_wav_bytes", lambda _samples: b"RIFF")
        built["server"] = server
        return server

    built["make"] = make
    return built


def test_inference_is_serialised_across_threads(tmp_path, wired):
    """The lock is the only thing standing between two API requests and one model.

    Neither the server nor the sherpa-onnx pipeline is safe to drive
    concurrently, so requests must queue. This is the invariant the whole
    "one GPU, one warm server, one worker" design rests on, and it is the one
    thing an e2e run cannot show, because the e2e tier is sequential.
    """
    settings = _settings(tmp_path)
    server = wired["make"](settings, hold=0.05)
    transcriber = Transcriber(settings)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            transcriber.transcribe("a.mp3")
        except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert server.calls == 4, "every request should still run, just one at a time"
    assert server.max_in_flight == 1, (
        f"two inferences overlapped (max {server.max_in_flight} in flight); the lock is not serialising"
    )


def test_diarization_turns_reach_speaker_attribution(tmp_path, wired):
    """The stages must be wired in order: server text, then diarized turns.

    Two turns that split the two segments, so a failure to pass the turns
    through cannot hide behind a single speaker being the right answer.
    """
    settings = _settings(tmp_path)
    wired["make"](
        settings,
        turns=[SpeakerTurn(0.0, 1.0, "SPEAKER_01"), SpeakerTurn(1.2, 2.0, "SPEAKER_02")],
    )
    transcriber = Transcriber(settings, diarize=True)

    result = transcriber.transcribe("a.mp3")

    assert transcriber.diarization_enabled is True
    assert result.language == "ru"
    assert [(u.speaker, u.text) for u in result.utterances] == [
        ("SPEAKER_01", "first"),
        ("SPEAKER_02", "second"),
    ]
    assert result.duration == pytest.approx(1.0)


def test_diarization_can_be_turned_off(tmp_path, wired):
    """With diarization off nothing may call the diarizer at all."""
    settings = _settings(tmp_path)
    wired["make"](settings)
    transcriber = Transcriber(settings, diarize=False)
    diarizer = transcriber._diarizer  # noqa: SLF001 - asserting it stays unused

    result = transcriber.transcribe("a.mp3")

    assert transcriber.diarization_enabled is False
    assert diarizer.calls == 0
    assert {u.speaker for u in result.utterances} == {"SPEAKER_00"}


def test_missing_diarization_models_degrade_to_one_speaker(tmp_path, wired, caplog):
    """Asking for diarization without the ONNX models warns and still transcribes.

    Silently ignoring the request would be worse: the output would look
    diarized and not be.
    """
    settings = _settings(tmp_path, diarization_available=False)
    wired["make"](settings)

    with caplog.at_level("WARNING", logger="voxpipe.core"):
        transcriber = Transcriber(settings, diarize=True)

    result = transcriber.transcribe("a.mp3")

    assert transcriber.diarization_enabled is False
    assert {u.speaker for u in result.utterances} == {"SPEAKER_00"}
    assert "diarization" in caplog.text.lower()


def test_no_warning_when_diarization_was_not_asked_for(tmp_path, wired, caplog):
    settings = _settings(tmp_path, diarization_available=False)
    wired["make"](settings)

    with caplog.at_level("WARNING", logger="voxpipe.core"):
        transcriber = Transcriber(settings, diarize=False)

    assert transcriber.diarization_enabled is False
    assert caplog.text == ""


def test_close_stops_the_server(tmp_path, wired):
    settings = _settings(tmp_path)
    server = wired["make"](settings)
    transcriber = Transcriber(settings)

    transcriber.close()

    assert server.stopped is True


def test_the_context_manager_closes_the_server(tmp_path, wired):
    settings = _settings(tmp_path)
    server = wired["make"](settings)

    with Transcriber(settings) as transcriber:
        assert transcriber.settings is settings
        assert transcriber.server.model_is_verified is True
        assert server.stopped is False

    assert server.stopped is True


def test_the_server_handle_is_exposed_for_health(tmp_path, wired):
    settings = _settings(tmp_path)
    server = wired["make"](settings)
    transcriber = Transcriber(settings)

    assert transcriber.server is server

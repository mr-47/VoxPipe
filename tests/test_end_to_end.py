"""End-to-end test against a real whisper-server + sherpa-onnx.

Skipped unless both of these are set, so the default suite stays offline and
needs no GPU:

    TRANSCRIBER_SERVER_BIN=vendor/bin/whisper-server \\
    TRANSCRIBER_TEST_AUDIO=/path/to/call.mp3 \\
    pytest tests/test_end_to_end.py

Both paths point inside this project on purpose, so the suite runs without any
sibling checkout. Defaults to the model that is vendored; override with
TRANSCRIBER_TEST_MODEL for a faster small model.
"""

import os
from pathlib import Path

import pytest

from voxpipe.config import TranscriberSettings
from voxpipe.core import Transcriber

SERVER_BIN = os.environ.get("TRANSCRIBER_SERVER_BIN")
AUDIO = Path(os.environ.get("TRANSCRIBER_TEST_AUDIO", ""))
MODEL = os.environ.get("TRANSCRIBER_TEST_MODEL", "turbo")

pytestmark = pytest.mark.skipif(
    not SERVER_BIN or not AUDIO.is_file(),
    reason="set TRANSCRIBER_SERVER_BIN and TRANSCRIBER_TEST_AUDIO to run",
)


def _transcriber(**kwargs) -> Transcriber:
    return Transcriber(TranscriberSettings(model=MODEL), **kwargs)


@pytest.fixture(scope="module")
def transcriber():
    with _transcriber() as t:
        yield t


def test_transcribes_a_real_recording(transcriber: Transcriber):
    result = transcriber.transcribe(str(AUDIO))

    assert result.duration > 0
    assert result.language
    assert result.segments, "whisper.cpp returned no segments"
    assert all(seg.text for seg in result.segments)
    assert [s.start for s in result.segments] == sorted(s.start for s in result.segments)


def test_word_timestamps_reach_the_transcript(transcriber: Transcriber):
    result = transcriber.transcribe(str(AUDIO))

    words = [w for seg in result.segments for w in seg.words]
    assert words, "no per-word timestamps in the whisper-server payload"
    assert all(w.start <= w.end for w in words)


def test_diarization_attributes_every_utterance(transcriber: Transcriber):
    result = transcriber.transcribe(str(AUDIO))

    assert transcriber.diarization_enabled
    assert result.utterances
    assert all(u.speaker.startswith("SPEAKER_") for u in result.utterances)
    assert all(u.text for u in result.utterances)
    assert {u.speaker for u in result.utterances} <= {f"SPEAKER_0{i}" for i in range(10)}


def test_diarization_can_be_switched_off():
    with _transcriber(diarize=False) as t:
        result = t.transcribe(str(AUDIO))

    assert not t.diarization_enabled
    assert {u.speaker for u in result.utterances} == {"SPEAKER_00"}


def test_server_stays_warm_between_files(transcriber: Transcriber):
    transcriber.transcribe(str(AUDIO))
    first = transcriber._server._process.pid
    transcriber.transcribe(str(AUDIO))

    assert transcriber._server._process.pid == first, "server was restarted mid-run"


def test_undecodable_input_raises_instead_of_hanging(tmp_path: Path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"\x00" * 8192)

    with _transcriber(diarize=False) as t, pytest.raises(Exception):
        t.transcribe(str(broken))

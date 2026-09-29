"""whisper.cpp /inference payload parsing and the wav upload encoding."""

import io
import wave

import numpy as np

from voxpipe.asr import language_of, parse_segments
from voxpipe.decode import duration_seconds, to_wav_bytes

# Trimmed from a real `response_format=verbose_json` reply (seconds, per-word).
SERVER_PAYLOAD = {
    "task": "transcribe",
    "language": "ru",
    "detected_language": "russian",
    "detected_language_probability": 0.999,
    "duration": 3.4,
    "text": " Добрый день.",
    "segments": [
        {
            "id": 1,
            "text": " Добрый день.",
            "start": 2.15,
            "end": 3.4,
            "words": [
                {"word": " Добрый", "start": 2.15, "end": 2.9, "probability": 0.97},
                {"word": " день", "start": 2.9, "end": 3.2, "probability": 0.98},
                {"word": ".", "start": 3.2, "end": 3.4, "probability": 0.72},
            ],
            "tokens": [3401, 13829, 4851, 13509, 13],
        }
    ],
}


def test_parse_segments_reads_words_and_timestamps():
    segments = parse_segments(SERVER_PAYLOAD)

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.start, segment.end, segment.text) == (2.15, 3.4, "Добрый день.")
    assert [w.word for w in segment.words] == [" Добрый", " день", "."]
    assert segment.words[0].start == 2.15
    assert segment.words[-1].end == 3.4


def test_blank_segments_are_dropped():
    payload = {"segments": [{"start": 0, "end": 1, "text": "   "}, {"start": 1, "end": 2, "text": "ok"}]}

    assert [s.text for s in parse_segments(payload)] == ["ok"]


def test_missing_segments_key_is_not_an_error():
    assert parse_segments({"text": "nothing"}) == []


def test_language_prefers_detected_over_requested():
    assert language_of(SERVER_PAYLOAD) == ("russian", 0.999)


def test_language_falls_back_to_plain_fields():
    assert language_of({"language": "en", "language_probability": 0.5}) == ("en", 0.5)
    assert language_of({}) == (None, None)


def test_to_wav_bytes_produces_readable_mono_pcm():
    tone = np.sin(np.linspace(0, 2 * np.pi * 440, 16000, dtype=np.float32))

    blob = to_wav_bytes(tone)

    with wave.open(io.BytesIO(blob)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 16000
        restored = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.max(np.abs(restored.astype(np.float32) / 32768.0 - tone)) < 1e-3


def test_to_wav_bytes_clips_out_of_range_samples():
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)

    with wave.open(io.BytesIO(to_wav_bytes(loud))) as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")

    assert samples[0] == 32767
    assert samples[1] == -32767
    assert samples[2] == 0


def test_duration_seconds():
    assert duration_seconds(np.zeros(32000, dtype=np.float32)) == 2.0

"""Audio decoding via PyAV.

whisper.cpp reads wav/flac/mp3/ogg itself, but not the m4a/AAC captures that
dominate the inbox, and the diarization model wants 16 kHz mono float samples
anyway. Decoding once here serves both consumers, so no temporary wav file and
no ffmpeg subprocess is needed.
"""

from __future__ import annotations

import io
import wave

import av
import numpy as np

from .config import SAMPLE_RATE


class DecodeError(RuntimeError):
    """Raised when a file has no decodable audio stream (yet)."""


def probe(path: str) -> bool:
    """Return True if the file is decodable audio right now.

    Used by the folder watcher to tell "still being written" (an m4a without a
    flushed index) apart from "broken".
    """
    try:
        with av.open(path, mode="r", metadata_errors="ignore") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return False
            next(container.decode(stream), None)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable now"
        return False


def decode(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video container to mono float32 samples at ``sample_rate``.

    Channels are averaged rather than just taking channel 0: phone recordings
    are mono, but stereo conference captures put one party on each side and
    averaging keeps both audible for VAD and segmentation.
    """
    chunks: list[np.ndarray] = []
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sample_rate)

    with av.open(path, mode="r", metadata_errors="ignore") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise DecodeError(f"{path} has no audio stream")
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32))
        # Flush the resampler's internal buffer (the last frames rarely fill a
        # full output frame and would otherwise be dropped).
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32))

    if not chunks:
        raise DecodeError(f"{path} decoded to zero samples")
    return np.concatenate(chunks)


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap float samples in a 16-bit PCM wav container for the HTTP upload."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def duration_seconds(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(samples) / float(sample_rate)

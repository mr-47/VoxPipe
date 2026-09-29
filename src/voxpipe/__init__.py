"""whisper.cpp (Vulkan) transcription with ONNX speaker diarization."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from .config import DiarizationSettings, TranscriberSettings
from .core import Transcriber
from .format import FORMATS, render
from .models import Segment, SpeakerTurn, Transcript, Utterance, Word

try:
    __version__ = _installed_version("voxpipe")
except PackageNotFoundError:  # running straight from a source tree, never installed
    __version__ = "unknown"

__all__ = [
    "DiarizationSettings",
    "FORMATS",
    "Segment",
    "SpeakerTurn",
    "Transcript",
    "Transcriber",
    "TranscriberSettings",
    "Utterance",
    "Word",
    "__version__",
    "render",
]

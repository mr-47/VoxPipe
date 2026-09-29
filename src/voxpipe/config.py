"""Runtime configuration, read from environment variables.

Everything the pipeline needs to reach the whisper.cpp server, the GGML models
and the ONNX diarization models is resolved here so the CLI stays a thin layer.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16000

SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".wma",
        ".mp4",
        ".webm",
        ".mka",
        ".mkv",
    }
)

#: Short names accepted by ``--model`` / ``TRANSCRIBER_MODEL``. Resolved against
#: the model search dirs; a value that looks like a path is used as one.
KNOWN_MODELS = {
    "turbo": "ggml-large-v3-turbo-q5_0.bin",
    "large-v3-turbo-q5_0": "ggml-large-v3-turbo-q5_0.bin",
    "small": "ggml-small.bin",
    "small-q5_1": "ggml-small-q5_1.bin",
    "base": "ggml-base.bin",
    "base-q5_1": "ggml-base-q5_1.bin",
}

DEFAULT_MODEL = "turbo"


def _default_model_dirs() -> list[Path]:
    """Where GGML/VAD/ONNX models are looked up, most specific first.

    ``TRANSCRIBER_MODEL_DIRS`` (os.pathsep-separated) prepends extra directories,
    which is how an install outside this repo points at its own model store.
    """
    here = Path(__file__).resolve()
    project = here.parent.parent.parent  # src/voxpipe -> project root
    dirs = [
        project / "vendor" / "models",
        Path.home() / ".cache" / "voxpipe" / "models",
    ]
    extra = os.environ.get("TRANSCRIBER_MODEL_DIRS", "")
    extra_dirs = [Path(p).expanduser() for p in extra.split(os.pathsep) if p.strip()]
    return extra_dirs + [d for d in dirs if d not in extra_dirs]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_threads() -> int:
    return _env_int("TRANSCRIBER_THREADS", min(8, os.cpu_count() or 4))


@dataclass
class DiarizationSettings:
    """ONNX-only speaker diarization (sherpa-onnx).

    No torch: a pyannote segmentation model proposes speaker regions, a
    3D-Speaker embedding model turns them into vectors, and FastClustering
    merges the local classes into global speakers.
    """

    segmentation_model: Path | None = None
    embedding_model: Path | None = None
    threshold: float = field(default_factory=lambda: _env_float("TRANSCRIBER_DIAR_THRESHOLD", 0.85))
    num_speakers: int = field(default_factory=lambda: _env_int("TRANSCRIBER_DIAR_SPEAKERS", -1))
    min_duration_on: float = field(default_factory=lambda: _env_float("TRANSCRIBER_DIAR_MIN_ON", 0.5))
    min_duration_off: float = field(default_factory=lambda: _env_float("TRANSCRIBER_DIAR_MIN_OFF", 0.7))
    window_shift_ratio: float = 0.1

    # Clusters holding less than this much speech never get a label of their
    # own: a sub-second turn cannot yield a reliable speaker embedding, so it
    # tends to fragment off as a singleton. The turn itself is kept -- it is
    # often the interjection that marks a speaker change -- and only the label
    # is reassigned to the temporally nearest cluster. 0 disables the pass.
    min_region_seconds: float = field(default_factory=lambda: _env_float("TRANSCRIBER_DIAR_MIN_REGION", 1.0))

    # Second clustering pass. FastClustering thresholds every *pair* of regions
    # at once, which over-splits long recordings: the median pairwise cosine
    # between regions is ~0.4, so a threshold of 0.85 sits near the 99th
    # percentile and merges almost nothing. This pass instead compares one
    # duration-weighted centroid per surviving cluster, which is far more
    # stable, and merges the closest pair while their cosine is at or above
    # this value. Calibrated between two measured points: two genuinely
    # different speakers sat at 0.48, one speaker split across clusters
    # merged at 0.80. <= 0 disables the pass.
    merge_threshold: float = field(default_factory=lambda: _env_float("TRANSCRIBER_DIAR_MERGE_THRESHOLD", 0.70))

    @property
    def available(self) -> bool:
        return bool(
            self.segmentation_model
            and self.embedding_model
            and self.segmentation_model.is_file()
            and self.embedding_model.is_file()
        )


@dataclass
class TranscriberSettings:
    """Whisper backend configuration (whisper.cpp ``whisper-server``)."""

    model: str = field(default_factory=lambda: os.environ.get("TRANSCRIBER_MODEL", DEFAULT_MODEL))
    model_dirs: list[Path] = field(default_factory=_default_model_dirs)
    server_bin: str = field(default_factory=lambda: os.environ.get("TRANSCRIBER_SERVER_BIN", ""))
    host: str = field(default_factory=lambda: os.environ.get("TRANSCRIBER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("TRANSCRIBER_PORT", 8099))
    threads: int = field(default_factory=_default_threads)
    device: int = field(default_factory=lambda: _env_int("TRANSCRIBER_DEVICE", 0))
    no_gpu: bool = field(default_factory=lambda: _env_bool("TRANSCRIBER_NO_GPU"))
    vad_model: Path | None = None
    use_vad: bool = field(default_factory=lambda: _env_bool("TRANSCRIBER_USE_VAD", True))
    request_timeout: float = field(default_factory=lambda: _env_float("TRANSCRIBER_TIMEOUT", 3600.0))
    startup_timeout: float = field(default_factory=lambda: _env_float("TRANSCRIBER_STARTUP_TIMEOUT", 180.0))
    #: Adopt a whisper-server this process did not start. Off by default: a
    #: foreign server on our port is a second model on one GPU, and its /health
    #: says nothing about which model it holds, so it cannot be checked.
    reuse_foreign_server: bool = field(default_factory=lambda: _env_bool("TRANSCRIBER_REUSE_SERVER", False))
    #: Where the "I started a server here" record lives. It is what lets a later
    #: run identify a server orphaned by a hard kill, which no signal handler
    #: can prevent. Kept out of the model dirs on purpose: this is mutable
    #: runtime state, not a downloaded asset.
    server_state_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TRANSCRIBER_SERVER_STATE_DIR", Path.home() / ".cache" / "voxpipe" / "servers")
        ).expanduser()
    )
    diarization: DiarizationSettings = field(default_factory=DiarizationSettings)

    def __post_init__(self) -> None:
        if not self.server_bin:
            self.server_bin = self._discover_server_bin()
        # A caller may pass a plain string, and the env var is a string too.
        self.server_state_dir = Path(self.server_state_dir).expanduser()

    @staticmethod
    def _discover_server_bin(project: Path | None = None) -> str:
        """Locate the ``whisper-server`` binary.

        Checked in order: $PATH, then the binary vendored into the project (plain
        or under ``vendor/bin/``), then an in-project build tree. Every candidate
        after ``$PATH`` is optional; the bare name is kept as a last resort so
        the resulting error message names the expected program.

        No candidate lives outside this project, so the project stays
        relocatable: move it anywhere and the same binary is found.

        ``project`` is injectable only so the search can be tested without
        creating files next to the real installation.
        """
        name = "whisper-server"
        found = shutil.which(name)
        if found:
            return found

        if project is None:
            project = Path(__file__).resolve().parent.parent.parent
        # shutil.which resolves the extension through PATHEXT, so $PATH already
        # finds whisper-server.exe on Windows. Path.is_file() does not, so an
        # in-project binary has to be probed under its real Windows name too.
        names = (name, f"{name}.exe") if sys.platform == "win32" else (name,)
        for directory in (
            project / "vendor",
            project / "vendor" / "bin",  # scripts/vendor-server.sh output
            project / "build" / "bin",
        ):
            for candidate_name in names:
                candidate = directory / candidate_name
                if candidate.is_file():
                    return str(candidate)
        return name

    # -- model resolution -------------------------------------------------

    def find(self, name: str, subdir: str | None = None) -> Path | None:
        """First existing ``name`` in the model dirs, honouring absolute paths."""
        candidate = Path(name).expanduser()
        if candidate.is_file():
            return candidate
        for directory in self.model_dirs:
            directory = Path(directory)
            path = directory / subdir / name if subdir else directory / name
            if path.is_file():
                return path
        return None

    def resolve_diarization_models(self) -> DiarizationSettings:
        """Fill in the segmentation/embedding model paths if the env points at them."""
        diar = self.diarization
        if diar.segmentation_model is None:
            diar.segmentation_model = self.find(
                os.environ.get("TRANSCRIBER_DIAR_SEG_MODEL", "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
                subdir="diar",
            )
        if diar.embedding_model is None:
            diar.embedding_model = self.find(
                os.environ.get(
                    "TRANSCRIBER_DIAR_EMBED_MODEL", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
                ),
                subdir="diar",
            )
        return diar

    def resolve_model_path(self) -> Path:
        """Resolve ``model`` to a GGML file.

        A value that points at an existing file wins; otherwise a known short
        name is looked up in the model dirs, and anything left over is reported
        with the paths that were searched.
        """
        candidate = Path(self.model).expanduser()
        if candidate.is_file():
            return candidate

        filename = KNOWN_MODELS.get(self.model.lower())
        if filename:
            for directory in self.model_dirs:
                path = directory / filename
                if path.is_file():
                    return path

        raise FileNotFoundError(
            f"whisper model {self.model!r} not found. Looked for "
            f"{filename or self.model!r} in: "
            + ", ".join(str(d) for d in self.model_dirs)
            + ". Set TRANSCRIBER_MODEL to a path, or add its directory to "
            "TRANSCRIBER_MODEL_DIRS." + (f" Or download it: scripts/fetch-models.sh {self.model}" if filename else "")
        )

    def resolve_vad_model(self) -> Path | None:
        """Silero VAD model for whisper.cpp, or None when VAD is unavailable.

        whisper-server refuses to start with ``--vad`` and no ``--vad-model``,
        so the flag is only passed when the file is actually there.
        """
        if not self.use_vad:
            return None
        name = os.environ.get("TRANSCRIBER_VAD_MODEL", "ggml-silero-v5.1.2.bin")
        if self.vad_model and self.vad_model.is_file():
            return self.vad_model
        for directory in [*self.model_dirs, self.model_dirs[0] / "diar"]:
            path = directory / name
            if path.is_file():
                return path
        return None

    def diarization_enabled(self) -> bool:
        return self.diarization.available

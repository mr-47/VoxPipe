"""Command line entry point for VoxPipe.

Three modes: ``voxpipe FILE`` transcribes one recording, ``voxpipe watch`` runs
the ``media-*`` folder workflow, and ``voxpipe serve`` exposes the REST API.
``--model`` takes a GGML file or a short name (``turbo``, ``small-q5_1``, ...).
``serve`` needs the ``api`` extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import KNOWN_MODELS, TranscriberSettings
from .core import Transcriber
from .format import FORMATS, render
from .pipeline import MediaFolderProcessor, watch_folders

DEFAULT_FORMAT = "md"
VERSION_STRING = f"voxpipe {__version__}"


def _positive_int(value: str) -> int:
    """argparse type for counts that must be >= 1.

    ``--caption-words 0`` reaches ``range(0, n, 0)`` and dies with a bare
    ValueError, and a negative value writes an empty subtitle file that looks
    like a successful run. Rejecting here names the actual problem.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {number}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {number}")
    return number


def _add_version_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", action="version", version=VERSION_STRING)


def _model_help() -> str:
    return (
        "Whisper model: a GGML file path, or one of "
        + ", ".join(sorted(KNOWN_MODELS))
        + " (default: turbo, i.e. ggml-large-v3-turbo-q5_0)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxpipe",
        description="Transcribe audio files with whisper.cpp (Vulkan) and attribute speech to speakers.",
    )
    _add_version_flag(parser)
    parser.add_argument("audio", help="Path to the input audio file (mp3, wav, m4a, ...)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the transcript to this file instead of stdout; the extension picks the format when present",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMATS),
        default=DEFAULT_FORMAT,
        help=f"Transcript format: txt, md, html, srt (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--layout",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="How transcript text is structured: wrapped paragraph or one line per sentence (default: paragraph)",
    )
    parser.add_argument("--json", type=Path, help="Also write the machine-readable JSON transcript to this file")
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in txt output (markdown/html/srt always show them)",
    )
    parser.add_argument(
        "--caption-words",
        type=_positive_int,
        default=6,
        help="Max words per SRT caption (subtitles are split into short 5-7 word lines; default: 6)",
    )
    parser.add_argument("--language", help="Optional language hint for whisper (e.g. 'en', 'de', 'ru')")
    parser.add_argument("--model", help=_model_help())
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization (everything attributed to one speaker)",
    )
    parser.add_argument("--words", action="store_true", help="Include per-word timestamps in the --json output")
    return parser


def build_watch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxpipe watch",
        description="Watch media-inbox and process every new audio file through the media-* folders.",
    )
    _add_version_flag(parser)
    parser.add_argument(
        "--dir",
        default=".",
        help="Base directory containing the media-inbox/process/failed/results folders (default: current directory)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the current media-inbox contents once, then exit",
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=5.0,
        help="Poll interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--format",
        action="append",
        metavar="{txt,md,html,srt}",
        help="Transcript format(s) written to media-results; comma-separated or repeat the flag "
        "(e.g. --format txt,md,html). Default: md",
    )
    parser.add_argument(
        "--layout",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="How transcript text is structured in the results (default: paragraph)",
    )
    parser.add_argument(
        "--caption-words",
        type=_positive_int,
        default=6,
        help="Max words per SRT caption (subtitle lines; default: 6)",
    )
    parser.add_argument("--language", help="Optional language hint for whisper (e.g. 'en', 'de', 'ru')")
    parser.add_argument("--model", help=_model_help())
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    return parser


def build_serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxpipe serve",
        description="Serve the REST API on a warm transcriber (POST /transcribe, GET /health). "
        "Needs the 'api' extra: pip install -e '.[api]'",
    )
    _add_version_flag(parser)
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1, loopback only)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Uvicorn log level (default: info)",
    )
    parser.add_argument("--language", help="Default language hint for whisper when a request omits it")
    parser.add_argument("--model", help=_model_help())
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    return parser


def _resolve_format(fmt: str, output: Path | None) -> str:
    if output is not None and output.suffix:
        extension = output.suffix.lower().lstrip(".")
        if extension in FORMATS:
            return extension
    return fmt


def _handle_transcript(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    settings = TranscriberSettings()
    if args.model:
        settings.model = args.model

    with Transcriber(settings, diarize=not args.no_diarization) as transcriber:
        result = transcriber.transcribe(args.audio, language=args.language)

    if args.json:
        payload = json.dumps(result.to_dict(include_words=args.words), ensure_ascii=False, indent=2)
        args.json.write_text(payload + "\n", encoding="utf-8")

    fmt = _resolve_format(args.format, args.output)
    content = render(
        result,
        fmt,
        layout=args.layout,
        timestamps=args.timestamps,
        words_per_caption=args.caption_words,
    )

    if args.output:
        output = args.output
        if not output.suffix:
            output = Path(f"{output}.{fmt}")
        output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)

    return 0


def _parse_watch_formats(raw: list[str] | None) -> list[str]:
    if not raw:
        return [DEFAULT_FORMAT]
    formats: list[str] = []
    for value in raw:
        for part in value.split(","):
            fmt = part.strip().lower()
            if not fmt:
                continue
            if fmt not in FORMATS:
                raise ValueError(f"Unsupported format {fmt!r}. Choose from: {', '.join(sorted(FORMATS))}")
            if fmt not in formats:
                formats.append(fmt)
    return formats or [DEFAULT_FORMAT]


def _handle_watch(argv: list[str]) -> int:
    args = build_watch_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = TranscriberSettings()
    if args.model:
        settings.model = args.model

    try:
        formats = _parse_watch_formats(args.format)
    except ValueError as exc:
        build_watch_parser().error(str(exc))

    with Transcriber(settings, diarize=not args.no_diarization) as transcriber:
        processor = MediaFolderProcessor(
            transcriber,
            args.dir,
            output_formats=formats,
            layout=args.layout,
            words_per_caption=args.caption_words,
            language=args.language,
        )
        processed = watch_folders(processor, interval=args.interval, once=args.once)
    logging.getLogger(__name__).info("Finished, processed %d file(s)", processed)
    return 0


def _handle_serve(argv: list[str]) -> int:
    args = build_serve_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = TranscriberSettings()
    if args.model:
        settings.model = args.model

    # Imported lazily so that a base install (no `api` extra) still gets a clean
    # "install the extra" message instead of a bare ImportError traceback.
    try:
        import uvicorn

        from .api import create_app
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: the REST API is the optional 'api' extra -> pip install -e '.[api]'", file=sys.stderr)
        print("       (the CLI and the media-* folder workflow do not need it)", file=sys.stderr)
        return 2

    app = create_app(settings=settings, diarize=not args.no_diarization, default_language=args.language)
    # No --workers: one worker means one model on the GPU. Extra workers would
    # each load their own copy and contend for the same device.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "watch":
        return _handle_watch(args[1:])
    if args and args[0] == "serve":
        return _handle_serve(args[1:])
    return _handle_transcript(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Top-level CLI surface: the shared flags and the command dispatch.

Nothing here loads a model or binds a port; `--version` is handled by argparse
and exits before any real work, so these are safe on a base install.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from voxpipe import __version__
from voxpipe.cli import main


@pytest.mark.parametrize("argv", [["--version"], ["watch", "--version"], ["serve", "--version"]])
def test_version_flag_reports_the_package_version(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"voxpipe {__version__}"


def test_version_is_exposed_on_the_package() -> None:
    assert isinstance(__version__, str) and __version__


def test_python_dash_m_voxpipe_dispatches_to_main() -> None:
    """`python -m voxpipe` is a second entry point and is wired separately.

    In-process coverage of `main()` cannot see it: the module only reaches
    `sys.exit` under `__main__`, so a rename or a bad relative import would
    ship broken while every other CLI test stayed green. It runs in a
    subprocess because `__name__ == "__main__"` is the condition under test.
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "voxpipe", "--version"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"voxpipe {__version__}"


def test_python_dash_m_voxpipe_reports_bad_usage() -> None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "voxpipe", "missing.mp3", "--caption-words", "0"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "must be 1 or greater" in proc.stderr


def test_missing_audio_is_still_an_error_not_a_version_print(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    assert "--version" in capsys.readouterr().err


def test_non_positive_caption_words_is_rejected_at_parse_time(capsys: pytest.CaptureFixture[str]) -> None:
    """0 crashed in range(), a negative wrote an empty subtitle file."""
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit) as excinfo:
            main(["call.mp3", "--format", "srt", "--caption-words", bad])
        assert excinfo.value.code == 2
        assert "must be 1 or greater" in capsys.readouterr().err


def test_watch_rejects_a_non_positive_interval(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["watch", "--interval", "-1"])

    assert excinfo.value.code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_valid_positive_values_are_accepted() -> None:
    from voxpipe.cli import build_parser, build_watch_parser

    assert build_parser().parse_args(["a.mp3", "--caption-words", "1"]).caption_words == 1
    assert build_watch_parser().parse_args(["--interval", "0.5"]).interval == 0.5


def _record_watch_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Drive `_handle_watch` with the engine and the loop replaced.

    Parsing a flag is not the same as forwarding it. `--language` was accepted
    by argparse and then dropped before it reached the processor, and no test
    caught it because the processor test constructed the processor directly --
    one layer below the bug. Every flag that crosses the CLI boundary needs an
    assertion at this level, not just at its destination.
    """
    from voxpipe import cli

    recorded: dict[str, object] = {}

    class _StubTranscriber:
        def __init__(self, settings, diarize: bool = True) -> None:  # noqa: ANN001
            recorded["settings"] = settings
            recorded["diarize"] = diarize

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def stub_processor(*args, **kwargs):  # noqa: ANN002, ANN003
        recorded["transcriber"] = args[0]
        recorded["base_dir"] = args[1]
        recorded["processor"] = kwargs
        return object()

    monkeypatch.setattr(cli, "Transcriber", _StubTranscriber)
    monkeypatch.setattr(cli, "MediaFolderProcessor", stub_processor)
    monkeypatch.setattr(cli, "watch_folders", lambda *_a, **_k: 0)
    return recorded


def test_watch_forwards_every_flag_to_the_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--language", "ru", "--format", "srt", "--layout", "sentence", "--caption-words", "3"]) == 0

    assert recorded["processor"] == {
        "output_formats": ["srt"],
        "layout": "sentence",
        "words_per_caption": 3,
        "language": "ru",
    }


def test_watch_passes_the_directory_through(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--dir", "/tmp/voxpipe-cli-test"]) == 0

    assert recorded["base_dir"] == "/tmp/voxpipe-cli-test"
    assert recorded["processor"]["language"] is None


def test_watch_honours_no_diarization_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--no-diarization", "--model", "small-q5_1"]) == 0

    assert recorded["diarize"] is False
    assert recorded["settings"].model == "small-q5_1"


def _record_one_shot(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Drive `_handle_transcript` with the engine replaced.

    The one-shot path is a second, independent copy of the flag plumbing, and
    it was the untested one: `--no-diarization` and `--language` were checked
    on the watch path only, so the one-shot handler could drop either and every
    test would still pass.
    """
    from voxpipe import cli
    from voxpipe.models import Segment, Transcript, Utterance

    recorded: dict[str, object] = {}

    class _StubTranscriber:
        def __init__(self, settings, diarize: bool = True) -> None:  # noqa: ANN001
            recorded["diarize"] = diarize
            recorded["settings"] = settings

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
            recorded["audio"] = audio_path
            recorded["language"] = language
            return Transcript(
                language="ru",
                language_probability=0.9,
                duration=2.0,
                segments=[Segment(0.0, 2.0, " привет")],
                utterances=[Utterance("SPEAKER_00", 0.0, 2.0, "привет")],
            )

    monkeypatch.setattr(cli, "Transcriber", _StubTranscriber)
    return recorded


def test_one_shot_forwards_language_and_diarization(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    recorded = _record_one_shot(monkeypatch)

    assert main(["some.mp3", "--language", "de", "--no-diarization", "--model", "small-q5_1"]) == 0

    assert recorded["diarize"] is False
    assert recorded["language"] == "de"
    assert recorded["audio"] == "some.mp3"
    assert recorded["settings"].model == "small-q5_1"
    assert "привет" in capsys.readouterr().out


def test_one_shot_diarizes_by_default(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    recorded = _record_one_shot(monkeypatch)

    assert main(["some.mp3"]) == 0

    assert recorded["diarize"] is True
    assert recorded["language"] is None


def test_one_shot_writes_the_output_file_and_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _record_one_shot(monkeypatch)
    out = tmp_path / "call.md"
    js = tmp_path / "call.json"

    assert main(["some.mp3", "-o", str(out), "--json", str(js), "--words"]) == 0

    assert out.read_text().strip() != ""
    assert "привет" in out.read_text()
    payload = json.loads(js.read_text())
    assert payload["language"] == "ru"
    assert payload["segments"], "--words should include the word array"

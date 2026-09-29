"""Folder state machine: inbox -> process -> results/failed."""

import json
import os
import wave
from pathlib import Path

from voxpipe.models import Transcript, Utterance
from voxpipe.pipeline import MAX_STALLED_SCANS, MediaFolderProcessor


def _write_wav(path: Path, seconds: float = 0.1) -> Path:
    """Tiny but genuinely decodable PCM WAV so the pre-flight probe passes.

    The suffix is deliberately allowed to be anything (e.g. .mp3): PyAV probes by
    content, which is the same behaviour the real pipeline relies on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _inbox(tmp_path, name: str) -> Path:
    return _write_wav(tmp_path / "media-inbox" / name)


def _fake_transcriber(fail_names: set[str] | None = None):
    class _Fake:
        def __init__(self) -> None:
            self.fail_names = fail_names or set()
            self.last_path = None
            self.languages: list[str | None] = []

        def transcribe(self, path, language=None) -> Transcript:
            self.last_path = str(path)
            self.languages.append(language)
            if any(name in str(path) for name in self.fail_names):
                raise RuntimeError("boom")
            return Transcript(
                language="en",
                language_probability=0.99,
                duration=1.5,
                segments=[],
                utterances=[Utterance(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello")],
            )

    return _Fake()


def _proc(tmp_path, fail_names=None) -> MediaFolderProcessor:
    return MediaFolderProcessor(_fake_transcriber(fail_names), tmp_path)


def test_success_moves_through_folders(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = _inbox(tmp_path, "call.mp3")

    result = proc.process_one(inbox_file.resolve())

    assert result is not None
    assert result.parent.name == "media-results"
    assert (tmp_path / "media-results" / "call.mp3").exists()
    assert (tmp_path / "media-results" / "call.json").exists()
    assert (tmp_path / "media-results" / "call.md").exists()
    assert not inbox_file.exists()
    assert not (tmp_path / "media-process" / "call.mp3").exists()
    assert not list((tmp_path / "media-failed").glob("*"))
    assert "SPEAKER_00" in (tmp_path / "media-results" / "call.md").read_text()
    assert proc._transcriber.last_path == str(tmp_path / "media-process" / "call.mp3")


def test_transcription_error_returns_fail_and_records_it(tmp_path):
    proc = _proc(tmp_path, fail_names={"call"})
    source = _inbox(tmp_path, "call.mp3")

    failed = proc.process_one(source.resolve())

    assert failed == (tmp_path / "media-failed" / "call.mp3").resolve()
    assert failed.exists()
    log = (tmp_path / "media-failed" / "call.txt").read_text()
    assert "Error: boom" in log


def test_unsupported_extension_is_rejected(tmp_path):
    proc = _proc(tmp_path)
    note = tmp_path / "media-inbox" / "notes.txt"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("hello")

    failed = proc.process_one(note.resolve())

    # Nothing is ever dropped silently: it is filed with a reason.
    assert failed == (tmp_path / "media-failed" / "notes.txt").resolve()
    # The report cannot be called notes.txt -- that is the file itself, and
    # writing there would destroy the thing being reported on.
    assert (tmp_path / "media-failed" / "notes.txt").read_text() == "hello"
    report = (tmp_path / "media-failed" / "notes.error.txt").read_text()
    assert "Unsupported file type" in report
    assert "Audio file: notes.txt" in report
    assert not note.exists()


def test_partial_file_is_held_then_fails_after_settled_scans(tmp_path):
    proc = _proc(tmp_path)
    partial = tmp_path / "media-inbox" / "still-writing.m4a"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"\x00" * 4096)  # not decodable

    assert proc.process_one(partial.resolve()) is None
    assert partial.exists()  # held, not failed on first sight

    for _ in range(MAX_STALLED_SCANS - 1):
        assert proc.process_one(partial.resolve()) is None
        assert partial.exists()

    failed = proc.process_one(partial.resolve())
    assert failed == (tmp_path / "media-failed" / "still-writing.m4a").resolve()
    assert not partial.exists()
    report = tmp_path / "media-failed" / "still-writing.txt"
    assert "never finished writing" in report.read_text()


def test_unreadable_file_recovers_when_it_becomes_valid(tmp_path):
    proc = _proc(tmp_path)
    path = tmp_path / "media-inbox" / "call.m4a"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 4096)

    assert proc.process_one(path.resolve()) is None

    # Writer finishes the file: valid wav now. The new size is only noticed on
    # the next scan (that one just records the change), the one after that
    # processes it.
    _write_wav(path)
    assert proc.process_one(path.resolve()) is None
    assert partial_still_in_inbox(path)

    assert proc.process_one(path.resolve()) is not None
    assert (tmp_path / "media-results" / "call.json").exists()


def partial_still_in_inbox(path: Path) -> bool:
    return path.exists()


def test_folders_are_created_automatically(tmp_path):
    base = tmp_path / "brand-new-dir"

    MediaFolderProcessor(_fake_transcriber(), base)

    for name in ("media-inbox", "media-process", "media-failed", "media-results"):
        assert (base / name).is_dir()


def test_stale_files_in_process_are_recovered(tmp_path):
    proc = _proc(tmp_path)
    stale = tmp_path / "media-process" / "interrupted.mp3"
    _write_wav(stale)

    recovered = proc.recover_stale()

    assert [p.name for p in recovered] == ["interrupted.mp3"]
    assert (tmp_path / "media-inbox" / "interrupted.mp3").exists()
    assert not stale.exists()


def test_hidden_files_are_never_picked_up(tmp_path):
    """A .gitkeep/.DS_Store in either folder must stay put, not be 'recovered'."""
    proc = _proc(tmp_path)
    (tmp_path / "media-process" / ".gitkeep").write_text("")
    (tmp_path / "media-process" / ".DS_Store").write_text("")
    (tmp_path / "media-inbox" / ".hidden-call.mp3").write_bytes(b"\x00" * 16)

    assert proc.recover_stale() == []
    assert (tmp_path / "media-process" / ".gitkeep").exists()
    assert (tmp_path / "media-process" / ".DS_Store").exists()
    assert proc.process_inbox() == []
    assert (tmp_path / "media-inbox" / ".hidden-call.mp3").exists()


def test_audio_must_live_in_the_inbox(tmp_path):
    proc = _proc(tmp_path)
    stray = _write_wav(tmp_path / "somewhere-else.mp3")

    import pytest

    with pytest.raises(ValueError):
        proc.process_one(stray.resolve())


def test_output_formats_and_json_contents(tmp_path):
    proc = _proc(tmp_path)
    proc.output_formats = ["txt", "md", "html", "srt"]

    proc.process_one(_inbox(tmp_path, "call.mp3").resolve())

    results = tmp_path / "media-results"
    for suffix in ("txt", "md", "html", "srt"):
        assert (results / f"call.{suffix}").exists()
    transcript = json.loads((results / "call.json").read_text())
    assert transcript["utterances"][0]["text"] == "hello"
    assert transcript["language"] == "en"


def test_same_stem_different_extensions_keep_both_transcripts(tmp_path):
    """`call.mp3` and `call.m4a` must not overwrite each other's transcript.

    Every output name is derived from the stem, so before the fix the second
    file clobbered the first: the audio survived in media-results, its
    transcript vanished, and nothing was logged or filed as failed.
    """
    proc = _proc(tmp_path)
    proc.output_formats = ["md", "txt"]
    _inbox(tmp_path, "call.m4a")
    _inbox(tmp_path, "call.mp3")

    proc.process_inbox()

    results = tmp_path / "media-results"
    assert (results / "call.m4a").exists() and (results / "call.mp3").exists()
    assert (results / "call.json").exists(), "first recording keeps the plain name"
    assert (results / "call_mp3.json").exists(), "second is disambiguated, not lost"
    assert (results / "call.md").exists() and (results / "call_mp3.md").exists()
    assert (results / "call.txt").exists() and (results / "call_mp3.txt").exists()
    # Each transcript is distinct and matches one of the two audio files.
    bodies = {(results / name).read_text() for name in ("call.md", "call_mp3.md")}
    assert len(bodies) == 1  # the fake transcriber emits identical text, so only
    # the *existence* of two files proves nothing was clobbered
    assert not list((tmp_path / "media-failed").glob("*"))


def test_rerunning_the_same_recording_overwrites_in_place(tmp_path):
    """Re-processing a recording must not accumulate numbered copies.

    The audio has been moved back to the inbox, so the only remaining evidence
    of who owns `call_mp3.*` is the name itself.
    """
    proc = _proc(tmp_path)
    _inbox(tmp_path, "call.m4a")
    _inbox(tmp_path, "call.mp3")
    proc.process_inbox()

    proc.process_one(_inbox(tmp_path, "call.mp3").resolve())

    results = tmp_path / "media-results"
    assert sorted(p.name for p in results.glob("call_mp3*")) == ["call_mp3.json", "call_mp3.md"]
    assert not list(results.glob("call_mp3_2*")), "a re-run must overwrite, not fork"


def test_same_stem_failures_keep_both_error_reports(tmp_path):
    """The failure path has the same collision and must disambiguate too."""
    proc = _proc(tmp_path, fail_names={"call"})
    _inbox(tmp_path, "call.m4a")
    _inbox(tmp_path, "call.mp3")

    proc.process_inbox()

    failed = tmp_path / "media-failed"
    assert (failed / "call.m4a").exists() and (failed / "call.mp3").exists()
    reports = sorted(p.name for p in failed.glob("*.txt"))
    assert reports == ["call.txt", "call_mp3.txt"]


def test_language_is_passed_to_every_transcription(tmp_path):
    """`voxpipe watch --language ru` used to be accepted and silently dropped."""
    transcriber = _fake_transcriber()
    proc = MediaFolderProcessor(transcriber, tmp_path, language="ru")
    _inbox(tmp_path, "a.mp3")
    _inbox(tmp_path, "b.mp3")

    proc.process_inbox()

    assert transcriber.languages == ["ru", "ru"]


def test_language_defaults_to_none_so_the_server_auto_detects(tmp_path):
    transcriber = _fake_transcriber()
    MediaFolderProcessor(transcriber, tmp_path).process_one(_inbox(tmp_path, "a.mp3").resolve())

    assert transcriber.languages == [None]


def test_readiness_state_does_not_outlive_the_file(tmp_path, monkeypatch):
    """A held file that is deleted must not keep an entry forever.

    A long-running `watch` is meant to last for months. Whatever feeds the
    inbox (the source app clearing its outbox, a cleanup script) deletes and
    moves files that were still being written, and nothing ever re-reads those
    entries -- so without pruning, _seen grew by one entry per file ever seen.
    """
    import voxpipe.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe", lambda _path: False)
    proc = _proc(tmp_path)

    for round_index in range(3):
        for i in range(5):
            _inbox(tmp_path, f"call-{round_index}-{i}.wav")
        proc.process_inbox()
        assert len(proc._seen) == 5, proc._seen
        for i in range(5):
            (tmp_path / "media-inbox" / f"call-{round_index}-{i}.wav").unlink()
        proc.process_inbox()
        assert proc._seen == {}, f"entries outlived their files: {proc._seen}"


def test_pruning_keeps_a_file_that_is_still_being_written(tmp_path, monkeypatch):
    """The whole point of _seen is the still-being-written case; keep working."""
    import voxpipe.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe", lambda _path: False)
    proc = _proc(tmp_path)
    _inbox(tmp_path, "half-written.wav")

    proc.process_inbox()
    assert list(proc._seen) == ["half-written.wav"], proc._seen
    first = proc._seen["half-written.wav"]

    proc.process_inbox()
    assert "half-written.wav" in proc._seen, "a live held file must stay tracked"
    assert proc._seen["half-written.wav"][1] == first[1] + 1, "the settled count must advance"


def test_a_same_length_replacement_is_not_mistaken_for_a_settled_file(tmp_path, monkeypatch):
    """Same name, same byte length, different file -- the mtime gives it away.

    This is the case the prune cannot help with, because no scan happens in
    between. Judged on size alone, a replacement inherited the old file's
    settled count, skipped the grace scans meant for a file still being
    written, and could be filed unreadable before it was ever given a chance.
    """
    import voxpipe.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe", lambda _path: False)
    proc = _proc(tmp_path)
    _inbox(tmp_path, "call.wav")
    proc.process_inbox()

    path = tmp_path / "media-inbox" / "call.wav"
    original = path.read_bytes()
    stamp, _ = proc._seen["call.wav"]
    # Pretend the old file had already survived nearly all of its grace scans.
    proc._seen["call.wav"] = (stamp, pipeline.MAX_STALLED_SCANS - 1)

    # A different file of byte-identical length, dropped in with no scan between.
    path.write_bytes(original)
    later = stamp[1] + 1_000_000_000
    os.utime(path, ns=(later, later))
    proc.process_inbox()

    assert proc._seen["call.wav"] == ((len(original), later), 0), proc._seen


def test_an_untouched_file_still_looks_settled(tmp_path, monkeypatch):
    """The mtime check must not make every scan look like a new file.

    Reading a file does not change its mtime, so a file nobody is writing keeps
    advancing its count and eventually gets filed or processed as before.
    """
    import voxpipe.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe", lambda _path: False)
    proc = _proc(tmp_path)
    _inbox(tmp_path, "call.wav")
    proc.process_inbox()
    stamp = proc._seen["call.wav"][0]

    for expected in (1, 2):
        proc.process_inbox()
        assert proc._seen["call.wav"] == (stamp, expected), proc._seen

    # MAX_STALLED_SCANS reached: unreadable for long enough -> filed
    proc.process_inbox()
    assert "call.wav" not in proc._seen, "the count must still reach the limit"
    assert (tmp_path / "media-failed" / "call.wav").exists()


def test_a_recreated_file_starts_fresh_after_a_scan_pruned_it(tmp_path, monkeypatch):
    """A name that vanished is forgotten, so a later drop is a first sighting.

    The prune runs at the end of every scan, so this holds even when a file is
    replaced with one of the same length *and* the same mtime.
    """
    import voxpipe.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe", lambda _path: False)
    proc = _proc(tmp_path)
    _inbox(tmp_path, "call.wav")
    proc.process_inbox()
    proc._seen["call.wav"] = (proc._seen["call.wav"][0], pipeline.MAX_STALLED_SCANS - 1)

    (tmp_path / "media-inbox" / "call.wav").unlink()
    proc.process_inbox()
    assert proc._seen == {}, "the name should have been forgotten once the file left"

    _inbox(tmp_path, "call.wav")
    proc.process_inbox()

    # Counted from zero again, whatever the old file's stamp was.
    assert proc._seen["call.wav"][1] == 0, proc._seen


class _StubProcessor:
    """Just enough of `MediaFolderProcessor` to drive the `watch_folders` loop."""

    def __init__(self, events: list[str], per_scan: list[str]) -> None:
        self.events = events
        self.per_scan = per_scan

    def recover_stale(self) -> list[Path]:
        self.events.append("recover")
        return []

    def process_inbox(self) -> list[Path]:
        self.events.append("scan")
        return [Path(name) for name in self.per_scan]


def _sleep_that_stops_after(monkeypatch, times: int, events: list[str]) -> None:
    """Replace the loop's sleep so the test terminates, and cannot hang.

    `watch_folders` is a `while True`; a stub sleep that simply returns would
    spin forever if the loop ever stopped honouring `--once`. Raising
    KeyboardInterrupt -- what Ctrl-C actually does -- is both realistic and a
    bounded way to get out.
    """
    from voxpipe import pipeline as pipeline_module

    calls = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        calls["n"] += 1
        events.append("sleep")
        if calls["n"] >= times:
            raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module.time, "sleep", fake_sleep)


def test_watch_recovers_leftovers_before_the_first_scan(monkeypatch):
    """`watch` is the long-running mode, so the startup order matters.

    A file left in media-process by a previous run goes back to the inbox before
    anything is scanned; doing it afterwards would leave it stranded until the
    next restart.
    """
    from voxpipe import pipeline as pipeline_module

    events: list[str] = []
    _sleep_that_stops_after(monkeypatch, times=1, events=events)

    assert pipeline_module.watch_folders(_StubProcessor(events, ["a.mp3"]), interval=0.01) == 1
    assert events == ["recover", "scan", "sleep"]


def test_watch_exits_cleanly_on_ctrl_c(monkeypatch, caplog):
    """Ctrl-C must be a clean stop, not a traceback out of a months-long daemon."""
    import logging

    from voxpipe import pipeline as pipeline_module

    events: list[str] = []
    _sleep_that_stops_after(monkeypatch, times=2, events=events)

    with caplog.at_level(logging.INFO, logger="voxpipe.pipeline"):
        processed = pipeline_module.watch_folders(_StubProcessor(events, ["a.mp3", "b.mp3"]), interval=5.0)

    assert processed == 4, "both scans before the interrupt should be counted"
    assert events == ["recover", "scan", "sleep", "scan", "sleep"]
    assert "Stopped by user" in caplog.text


def test_watch_once_scans_a_single_time(monkeypatch):
    from voxpipe import pipeline as pipeline_module

    events: list[str] = []
    _sleep_that_stops_after(monkeypatch, times=99, events=events)

    assert pipeline_module.watch_folders(_StubProcessor(events, ["a.mp3"]), interval=0.01, once=True) == 1
    assert events == ["recover", "scan"], "--once must not sleep at all"

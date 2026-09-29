"""The media-* folder workflow.

Ported from the faster-whisper pipeline in Transcriber (MIT, (c) its
author): same four-stage layout, same move-never-copy semantics, same readiness
heuristic that leaves a still-being-written file in the inbox instead of failing
it. Only the folder names and the decode probe changed - ``media-*`` instead of
``calls-*``, and PyAV instead of ``av.open`` inside the pipeline module.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import SUPPORTED_AUDIO_SUFFIXES
from .core import Transcriber
from .decode import probe
from .format import FORMATS, render

logger = logging.getLogger(__name__)

FOLDER_INBOX = "media-inbox"
FOLDER_PROCESS = "media-process"
FOLDER_FAILED = "media-failed"
FOLDER_RESULTS = "media-results"

#: Consecutive scans with an unchanged size before a non-readable audio file is
#: declared failed. Protects against files that are still being written.
MAX_STALLED_SCANS = 3


def _file_stamp(path) -> tuple[int, int]:
    """``(size, mtime_ns)`` -- what "this file, as of the last scan" means.

    Size alone is not identity. A recording that is deleted and re-dropped
    between two scans under the same name can land on exactly the same byte
    length, and the old entry would make the new file look already settled: it
    would skip the grace scans meant for a file still being written, and could
    be filed as unreadable before it was ever given a chance. The mtime
    distinguishes the two, because a replacement is a different file with a
    different timestamp even when the length is identical.
    """
    try:
        stat = path.stat()
    except OSError:
        return 0, 0
    return stat.st_size, stat.st_mtime_ns


class MediaFolderProcessor:
    """Moves audio files through the media-* folder workflow.

    media-inbox    -> new audio files to process
    media-process  -> file currently being transcribed
    media-failed   -> failures, with a <name>.txt error file
    media-results  -> successes with the transcript (same name, .json + transcript file)

    Files are *moved* between folders, never copied.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        base_dir: str | os.PathLike = ".",
        output_formats: str | list[str] = "md",
        layout: str = "paragraph",
        words_per_caption: int = 6,
        language: str | None = None,
    ) -> None:
        if isinstance(output_formats, str):
            requested = [part for part in output_formats.split(",") if part]
        else:
            requested = output_formats
        formats = [fmt.strip().lower() for fmt in requested]
        invalid = [fmt for fmt in formats if fmt not in FORMATS]
        if invalid:
            raise ValueError(f"Unsupported output format(s) {invalid!r}. Choose one of: {', '.join(sorted(FORMATS))}")
        if not formats:
            formats = ["md"]
        self._transcriber = transcriber
        self.output_formats = formats
        self.layout = layout
        self.words_per_caption = words_per_caption
        # Language hint applied to every file. ``voxpipe watch --language ru``
        # used to be accepted and then dropped on the floor, because nothing
        # between the CLI and this call carried it; whisper-server then fell
        # back to auto-detection per file.
        self.language = language
        # File name -> ((size, mtime_ns), settled_unreadable_count). Tracks files
        # that are not processable yet (still being written, or corrupt) across
        # scans. Keyed by name, so the stamp is what proves the file behind the
        # name is still the same one -- see _file_stamp.
        self._seen: dict[str, tuple[tuple[int, int], int]] = {}
        self.base = Path(base_dir).resolve()
        self.inbox = self.base / FOLDER_INBOX
        self.processing = self.base / FOLDER_PROCESS
        self.failed = self.base / FOLDER_FAILED
        self.results = self.base / FOLDER_RESULTS
        for folder in (self.inbox, self.processing, self.failed, self.results):
            folder.mkdir(parents=True, exist_ok=True)

    def recover_stale(self) -> list[Path]:
        """Move leftover files in media-process back to media-inbox for retry.

        Hidden files are left alone: they are not audio (a ``.gitkeep`` holding
        the folder in git, a ``.DS_Store``, an editor swap file), and dragging
        them into the inbox would strand them there, since the inbox scan skips
        dotfiles.
        """
        recovered: list[Path] = []
        if not self.processing.is_dir():
            return recovered
        for item in self.processing.iterdir():
            if item.is_file() and not item.name.startswith("."):
                os.replace(item, self.inbox / item.name)
                recovered.append(item)
        if recovered:
            logger.warning("Recovered %d stale file(s) from %s", len(recovered), FOLDER_PROCESS)
        return recovered

    def process_inbox(self) -> list[Path]:
        """Process every audio file currently in media-inbox. Returns final paths."""
        finished: list[Path] = []
        for item in sorted(self.inbox.iterdir()):
            if not item.is_file() or item.name.startswith("."):
                continue
            try:
                path = self.process_one(item)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not process %s: %s", item, exc)
                continue
            if path is not None:
                finished.append(path)
        self._prune_seen()
        return finished

    def _prune_seen(self) -> None:
        """Drop readiness state for files that are no longer in the inbox.

        A held file is usually deleted or moved by whatever is feeding the inbox
        (a cleanup script, the source app clearing its outbox). Nothing re-reads
        those entries afterwards, so a long-running ``watch`` would otherwise
        accumulate one entry per file ever seen -- a slow leak in a daemon that
        is meant to run for months. The inbox listing is the whole truth here:
        every name still present is still being watched, and every other name is
        stale by definition.
        """
        present = {item.name for item in self.inbox.iterdir() if item.is_file()}
        stale = self._seen.keys() - present
        for name in stale:
            del self._seen[name]
        if stale:
            logger.debug("Pruned %d readiness entries no longer in %s", len(stale), FOLDER_INBOX)

    def process_one(self, audio_path: Path) -> Path | None:
        """Run a single audio file through the pipeline.

        Returns where the audio ended up, or None if the file was deferred
        (left in media-inbox because it is not decodable yet).
        """
        audio = audio_path.resolve()
        if audio.parent != self.inbox:
            raise ValueError(f"Audio file must be inside {self.inbox}: {audio}")

        name = audio.name
        stem = audio.stem
        suffix = audio.suffix.lower()

        if not suffix or suffix not in SUPPORTED_AUDIO_SUFFIXES:
            reason = f"Unsupported file type: {suffix or 'no extension'}"
            return self._fail(audio, reason, cause=None)

        status = self._readiness(audio)
        if status == "hold":
            return None
        if status == "fail":
            return self._fail(
                audio,
                "Audio file is unreadable (invalid or never finished writing): PyAV open/decode failed",
                cause=None,
            )
        working = self.processing / name
        try:
            os.replace(audio, working)
        except FileNotFoundError:
            logger.warning("%s disappeared before processing", audio)
            self._seen.pop(name, None)
            return audio

        try:
            logger.info("Transcribing %s", working)
            transcript = self._transcriber.transcribe(str(working), language=self.language)
        except Exception as exc:
            logger.exception("Transcription failed for %s", working)
            return self._fail(working, f"Transcription failed: {exc}", cause=exc)

        out_stem = self._unique_stem(stem, suffix)
        json_path = self.results / f"{out_stem}.json"
        json_path.write_text(
            json.dumps(transcript.to_dict(include_words=True), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for fmt in self.output_formats:
            path = self.results / f"{out_stem}.{fmt}"
            path.write_text(
                render(
                    transcript,
                    fmt,
                    layout=self.layout,
                    words_per_caption=self.words_per_caption,
                )
                + "\n",
                encoding="utf-8",
            )
        if out_stem != stem:
            logger.warning(
                "Wrote transcripts as %s.* because %s.* already belongs to a different recording",
                out_stem,
                stem,
            )
        os.replace(working, self.results / name)
        logger.info("Processed %s -> %s", name, FOLDER_RESULTS)
        return self.results / name

    def _unique_stem(self, stem: str, suffix: str) -> str:
        """Pick a transcript stem that will not clobber an existing transcript.

        Two recordings that share a stem but not an extension (``call.mp3`` and
        ``call.m4a``) both want ``call.json`` / ``call.md``. The second one used
        to overwrite the first: the audio survived in media-results, its
        transcript vanished, and nothing was logged. Fall back to the full name
        with the extension turned into part of the stem (``call.m4a`` ->
        ``call_m4a``) so both transcripts coexist.
        """
        if not self._stem_taken(stem, suffix):
            return stem
        candidate = f"{stem}{suffix.replace('.', '_')}".replace(os.sep, "_")
        if not self._stem_taken(candidate, suffix):
            return candidate
        # Two files with an identical stem *and* extension cannot both be in
        # the inbox, but a re-run over an existing result can collide. Never
        # overwrite: add a counter.
        for n in range(2, 1000):
            numbered = f"{candidate}_{n}"
            if not self._stem_taken(numbered, suffix):
                return numbered
        raise RuntimeError(f"Could not find a free transcript name for {stem}{suffix}")

    def _stem_taken(self, stem: str, suffix: str) -> bool:
        """True if some *other* recording's audio already owns this stem.

        Ownership is read off the audio sitting in results rather than guessed.
        ``call.*`` belongs to whichever ``call.<ext>`` audio is next to it, so a
        stem is claimed when a different extension owns it. An unclaimed stem
        (no audio there at all) is free even if a transcript file of that name
        exists, which is what makes re-running a recording -- where the audio
        has been moved back to the inbox -- overwrite in place instead of
        accumulating numbered copies.
        """
        owners = {
            item.suffix.lower()
            for item in self.results.iterdir()
            if item.is_file() and item.stem == stem and item.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        }
        return bool(owners) and owners != {suffix}

    def _readiness(self, audio: Path) -> str:
        """Decide whether to process `audio` now, hold it, or fail it.

        Returns "process", "hold", or "fail" (the caller acts on it).

        1. First sighting: probe. A readable file is processed immediately
           (finished files dropped into the inbox work on the first scan). An
           unreadable one is recorded as a baseline and held in media-inbox.
        2. Later sightings: the identity check comes first. If the size *or* the
           mtime changed the source is either still writing or a different file
           entirely, so we hold without even probing (this also catches
           header-first formats the probe cannot spot) and restart the count.
           Once the file has settled we probe: readable -> process; unreadable
           for MAX_STALLED_SCANS settled scans in a row -> fail.
        """
        name = audio.name
        current = _file_stamp(audio)

        previous = self._seen.get(name)
        if previous is None:
            if probe(str(audio)):
                return "process"
            self._seen[name] = (current, 0)
            logger.info(
                "Deferring %s: audio not decodable yet - is the source still writing?",
                name,
            )
            return "hold"

        (last_size, last_mtime), settled_count = previous
        size, mtime = current
        if last_size != size or last_mtime != mtime:
            if last_size == size:
                # Same length, different file. Only the mtime gives it away, and
                # without this the new file would be counted as already settled.
                logger.info(
                    "Deferring %s: replaced (same %d bytes, modified %d -> %d) - a new file, not one still settling",
                    name,
                    size,
                    last_mtime,
                    mtime,
                )
            else:
                logger.info(
                    "Deferring %s: size changed (%d -> %d bytes) - still being written",
                    name,
                    last_size,
                    size,
                )
            self._seen[name] = (current, 0)
            return "hold"

        settled_count += 1
        self._seen[name] = (current, settled_count)
        if probe(str(audio)):
            self._seen.pop(name, None)
            logger.info("%s became readable (settled at %d bytes)", name, current)
            return "process"

        if settled_count >= MAX_STALLED_SCANS:
            logger.warning(
                "%s stayed unreadable through %d settled scan(s); returning fail",
                name,
                settled_count,
            )
            self._seen.pop(name, None)
            return "fail"
        logger.info(
            "Deferring %s: readable check failed %d/%d times at unchanged size",
            name,
            settled_count,
            MAX_STALLED_SCANS,
        )
        return "hold"

    def _fail(self, audio: Path, message: str, cause: BaseException | None) -> Path:
        target = self.failed / audio.name
        try:
            os.replace(audio, target)
        except FileNotFoundError:
            target = audio
        self._write_error(target, message, cause)
        logger.error("Failed %s -> %s", target.name, FOLDER_FAILED)
        return target

    def _write_error(self, target: Path, message: str, cause: BaseException | None) -> None:
        # Same collision as the transcripts: `call.mp3` and `call.m4a` both fail
        # to `call.txt`, and the first reason would be lost. Disambiguate the
        # report name the same way the transcript name is disambiguated.
        error_path = self.failed / f"{self._unique_failed_stem(target)}.txt"
        content = (
            f"Audio file: {target.name}\n"
            f"Failed at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"Error: {message}\n"
        )
        if cause is not None:
            content += f"\nTraceback:\n{traceback.format_exc()}\n"
        try:
            error_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.error("Could not write error file %s: %s", error_path, exc)

    def _unique_failed_stem(self, target: Path) -> str:
        """Stem for the ``.txt`` report, so no report is ever lost or destructive.

        Two things can go wrong with the obvious ``<stem>.txt``:

        * The input is itself a ``.txt`` (an unsupported file type, which is
          exactly what lands in media-failed). The report path is then the audio
          path, and writing the report destroys the file it describes.
        * Two recordings share a stem but not an extension, so one report would
          overwrite the other and its reason would be lost.
        """
        stem, suffix = target.stem, target.suffix.lower()
        # `.txt` in, `.txt` report out: pick a name that cannot be the input.
        base = f"{stem}.error" if suffix == ".txt" else stem
        if not self._failed_stem_taken(base, suffix):
            return base
        candidate = f"{base}{suffix.replace('.', '_')}"
        if not self._failed_stem_taken(candidate, suffix):
            return candidate
        for n in range(2, 1000):
            numbered = f"{candidate}_{n}"
            if not self._failed_stem_taken(numbered, suffix):
                return numbered
        return candidate

    def _failed_stem_taken(self, stem: str, suffix: str) -> bool:
        """True if a *different* failed recording already owns this report name.

        Ownership is read off the audio next to the report. An existing report
        whose audio is gone belongs to something else, so it must not be
        overwritten; an unclaimed name is free, which keeps a re-run idempotent.
        """
        if not (self.failed / f"{stem}.txt").exists():
            return False
        owners = {
            item.suffix.lower()
            for item in self.failed.iterdir()
            if item.is_file() and item.stem == stem and item.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        }
        return bool(owners) and owners != {suffix}


def watch_folders(
    processor: MediaFolderProcessor,
    interval: float = 5.0,
    once: bool = False,
) -> int:
    """Poll media-inbox and process new audio files until interrupted."""
    processor.recover_stale()
    processed = 0
    try:
        while True:
            handled = processor.process_inbox()
            processed += len(handled)
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopped by user after processing %d file(s)", processed)
    return processed

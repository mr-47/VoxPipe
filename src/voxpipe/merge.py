"""Speaker attribution and utterance grouping.

Ported unchanged from the faster-whisper pipeline in Transcriber
(MIT, (c) its author). It only depends on the transcript dataclasses, which
kept the same shape here, so both apps label speech the same way.
"""

from __future__ import annotations

from .models import Segment, SpeakerTurn, Utterance


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[str]:
    """Align each whisper segment with a speaker using diarization turn timestamps.

    For each segment we use its midpoint. Normally exactly one turn covers that
    point, but diarization output can overlap -- measured on a two-party call,
    3-4 overlapping pairs -- so when several cover it the *latest starting* one
    wins: it is the most specific claim on that instant, whereas taking the
    first would let one long turn swallow the whole file. Failing that we fall
    back to the nearest turn and finally to the previously assigned speaker.

    Note this only makes the *attribution* sane. It cannot repair a degenerate
    partition: a pinned ``TRANSCRIBER_DIAR_SPEAKERS`` splits a two-party call
    98%/2% by speech, and no rule for picking among overlapping turns recovers
    the speaker who is missing from the rest of the recording.
    """
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    speakers: list[str] = []
    idx = 0
    previous: str | None = None

    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        while idx < len(ordered) and ordered[idx].end <= mid:
            idx += 1

        covering = [t for t in ordered if t.start <= mid <= t.end]
        if covering:
            best = max(covering, key=lambda t: (t.start, t.end)).speaker
            speakers.append(best)
            previous = best
            continue

        best: str | None = None
        best_gap = float("inf")
        for j in range(max(0, idx - 2), min(len(ordered), idx + 3)):
            turn = ordered[j]
            gap = min(abs(turn.start - mid), abs(turn.end - mid))
            if gap < best_gap:
                best_gap = gap
                best = turn.speaker

        if best is None and previous is not None:
            best = previous
        if best is None:
            best = "SPEAKER_00"

        speakers.append(best)
        previous = best

    return speakers


def build_utterances(segments: list[Segment], speakers: list[str]) -> list[Utterance]:
    """Merge consecutive segments spoken by the same speaker into utterances."""
    utterances: list[Utterance] = []
    for seg, spk in zip(segments, speakers):
        if utterances and utterances[-1].speaker == spk:
            current = utterances[-1]
            current.text = f"{current.text} {seg.text}".strip()
            current.end = seg.end
        else:
            utterances.append(Utterance(speaker=spk, start=seg.start, end=seg.end, text=seg.text))
    return utterances

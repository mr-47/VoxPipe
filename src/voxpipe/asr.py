"""whisper.cpp ``/inference`` response -> :class:`Segment` / :class:`Word` lists.

``response_format=verbose_json`` already gives per-word ``word``/``start``/``end``
in seconds, which is exactly the shape the faster-whisper pipeline produced, so
the JSON written next to each transcript stays comparable.
"""

from __future__ import annotations

from .models import Segment, Word


def parse_segments(payload: dict) -> list[Segment]:
    segments: list[Segment] = []
    for raw in payload.get("segments") or []:
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        words = [
            Word(start=float(w["start"]), end=float(w["end"]), word=w.get("word") or "")
            for w in (raw.get("words") or [])
            if w.get("word")
        ]
        segments.append(
            Segment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=text,
                words=words,
            )
        )
    return segments


def language_of(payload: dict) -> tuple[str | None, float | None]:
    """Detected language and its probability.

    The server reports both the requested ``language`` (echo) and what the model
    actually detected, so prefer the detected values.
    """
    language = payload.get("detected_language") or payload.get("language")
    probability = payload.get("detected_language_probability")
    if probability is None:
        probability = payload.get("language_probability")
    return language, float(probability) if probability is not None else None

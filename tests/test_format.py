"""Transcript renderers (ported from the faster-whisper pipeline)."""

import pytest

from voxpipe.format import render, split_sentences, to_html, to_markdown, to_srt, to_text
from voxpipe.models import Transcript, Utterance

UTTERANCES = [
    Utterance(speaker="SPEAKER_00", start=0.0, end=3.4, text="Good morning everyone. Let's start."),
    Utterance(speaker="SPEAKER_01", start=3.5, end=6.2, text="Thanks for joining."),
]


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        language_probability=0.97,
        duration=6.2,
        segments=[],
        utterances=UTTERANCES,
    )


def test_split_sentences_respects_abbreviations():
    assert split_sentences("Call Dr. Smith now. He is in.") == ["Call Dr. Smith now.", "He is in."]


def test_markdown_paragraph_layout_labels_and_spans():
    out = to_markdown(UTTERANCES)

    assert out.startswith("**SPEAKER_00** · [00:00 – 00:03]")
    assert "Good morning everyone. Let's start." in out
    assert "**SPEAKER_01** · [00:03 – 00:06]" in out


def test_markdown_sentence_layout_one_line_per_sentence():
    out = to_markdown(UTTERANCES, layout="sentence")

    lines = out.splitlines()
    assert lines[0] == "- [00:00 – 00:03] **SPEAKER_00**: Good morning everyone."
    assert lines[1] == "- [00:00 – 00:03] **SPEAKER_00**: Let's start."


def test_text_layout_with_and_without_timestamps():
    assert to_text(UTTERANCES) == "SPEAKER_00: Good morning everyone. Let's start.\nSPEAKER_01: Thanks for joining."
    assert to_text(UTTERANCES, timestamps=True).startswith("[00:00] SPEAKER_00: ")


def test_srt_splits_into_short_captions_and_divides_the_span():
    out = to_srt(UTTERANCES, words_per_caption=6)

    blocks = [block for block in out.split("\n\n") if block]
    assert len(blocks) == 2  # 6 words fit one caption, 3 words the other
    assert blocks[0].splitlines()[1] == "00:00:00,000 --> 00:00:03,400"
    assert blocks[0].splitlines()[2].startswith("SPEAKER_00: Good morning everyone.")


def test_render_rejects_unknown_format():
    import pytest

    with pytest.raises(ValueError):
        render(_transcript(), "pdf")


def test_render_dispatches_to_html():
    out = render(_transcript(), "html")

    assert out.startswith("<!DOCTYPE html>")
    assert "SPEAKER_00" in out
    assert "Good morning everyone." in out


def test_srt_rejects_a_non_positive_caption_size():
    """0 reached range(0, n, 0) and died; negative produced an empty file.

    Both looked like a broken run rather than a bad flag.
    """
    utterances = [Utterance(speaker="SPEAKER_00", start=0.0, end=1.0, text="one two three")]

    for bad in (0, -1, -6):
        with pytest.raises(ValueError) as excinfo:
            to_srt(utterances, words_per_caption=bad)
        assert "words_per_caption" in str(excinfo.value)


def test_a_single_word_per_caption_still_works():
    utterances = [Utterance(speaker="SPEAKER_00", start=0.0, end=1.0, text="one two three")]

    out = to_srt(utterances, words_per_caption=1)

    assert out.count("-->") == 3


def test_empty_utterance_text_does_not_crash_the_renderers():
    """A blank turn used to IndexError in the paragraph layout."""
    blank = [Utterance(speaker="SPEAKER_00", start=0.0, end=1.0, text="")]

    assert to_text(blank) == "SPEAKER_00:"
    assert to_markdown(blank).startswith("**SPEAKER_00**")
    assert "SPEAKER_00" in to_html(blank)
    assert to_srt(blank) == ""

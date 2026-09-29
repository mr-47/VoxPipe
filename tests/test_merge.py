"""Speaker attribution and utterance grouping."""

from voxpipe.merge import assign_speakers, build_utterances
from voxpipe.models import Segment, SpeakerTurn, Utterance


def _seg(start, end, text):
    return Segment(start=start, end=end, text=text)


def test_turn_covering_midpoint_wins():
    segments = [_seg(0, 2, "a"), _seg(2, 4, "b")]
    turns = [SpeakerTurn(0, 2, "SPEAKER_00"), SpeakerTurn(2, 4, "SPEAKER_01")]

    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_01"]


def test_falls_back_to_nearest_turn_then_previous():
    segments = [_seg(0, 1, "a"), _seg(9, 10, "b"), _seg(90, 91, "c")]
    turns = [SpeakerTurn(0, 1, "SPEAKER_00")]

    # No turn anywhere near 9.5s or 90.5s: the previous speaker carries over.
    assert assign_speakers(segments, turns) == ["SPEAKER_00"] * 3


def test_single_speaker_when_no_turns():
    assert assign_speakers([_seg(0, 1, "a")], []) == ["SPEAKER_00"]


def test_overlapping_turns_pick_the_most_specific():
    """Overlapping turns: the latest-starting one is the most specific claim.

    Real output does overlap (3-4 pairs on a 156 s call). Taking the first
    match, the old behaviour, let one long turn swallow the whole transcript.
    """
    segments = [_seg(0, 2, "a"), _seg(4, 6, "b"), _seg(8, 10, "c")]
    turns = [
        SpeakerTurn(0, 20, "SPEAKER_00"),  # spans everything
        SpeakerTurn(4, 6, "SPEAKER_01"),
        SpeakerTurn(8, 10, "SPEAKER_02"),
    ]

    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_overlapping_turns_prefer_the_later_start():
    """Two turns both cover the midpoint; the one starting later wins."""
    segments = [_seg(5, 7, "a")]
    turns = [
        SpeakerTurn(0, 20, "SPEAKER_00"),
        SpeakerTurn(6, 20, "SPEAKER_01"),
    ]

    assert assign_speakers(segments, turns) == ["SPEAKER_01"]


def test_non_overlapping_turns_are_unaffected():
    segments = [_seg(0, 2, "a"), _seg(2, 4, "b")]
    turns = [SpeakerTurn(0, 2, "SPEAKER_00"), SpeakerTurn(2, 4, "SPEAKER_01")]

    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_01"]


def test_attribution_cannot_repair_a_degenerate_partition():
    """Pinning the speaker count is broken upstream, and this pins that fact.

    A real pinned run of a two-party call gives 133.9s to one speaker and 2.8s
    to the other. Every rule for choosing among overlapping turns still
    attributes most of the call to the dominant cluster, so the second speaker
    vanishes from the transcript. This test exists to stop someone "fixing"
    assign_speakers to chase that -- the defect is in the clustering, not here.
    """
    # One cluster owns nearly the whole call; the other gets three slivers.
    turns = [
        SpeakerTurn(1.7, 4.3, "SPEAKER_00"),
        SpeakerTurn(3.2, 3.8, "SPEAKER_01"),  # 0.6s sliver, nested in the above
        SpeakerTurn(5.2, 147.0, "SPEAKER_00"),
    ]
    segments = [
        _seg(0, 1.0, "a"),  # midpoint 0.5 -> nearest turn, no cover
        _seg(3.0, 3.6, "b"),  # midpoint 3.3 -> both cover, sliver is more specific
        _seg(6.0, 7.0, "c"),  # midpoint 6.5 -> only the long turn covers
    ]

    speakers = assign_speakers(segments, turns)

    assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert speakers.count("SPEAKER_01") == 1, "only the 0.6s sliver is recoverable"


def test_consecutive_segments_merge_into_one_utterance():
    segments = [_seg(0, 1, "hello"), _seg(1, 2, "there"), _seg(2, 3, "bye")]
    speakers = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]

    utterances = build_utterances(segments, speakers)

    assert [(u.speaker, u.text) for u in utterances] == [
        ("SPEAKER_00", "hello there"),
        ("SPEAKER_01", "bye"),
    ]
    assert utterances[0].end == 2.0
    assert isinstance(utterances[0], Utterance)

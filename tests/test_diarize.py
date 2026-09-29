"""Region filtering and the second clustering pass.

These never load the ONNX models: ``_embed`` is replaced with a lookup table so
the merge logic can be exercised on a handful of synthetic turns.
"""

import numpy as np

from voxpipe.config import DiarizationSettings
from voxpipe.diarize import Diarizer
from voxpipe.models import SpeakerTurn

# SPEAKER_00 and SPEAKER_01 are the same person (cosine ~0.99) split by stage 1;
# SPEAKER_02 is orthogonal to both, i.e. a genuinely different person.
VECTORS = {
    "SPEAKER_00": (1.0, 0.0),
    "SPEAKER_01": (0.9, 0.1),
    "SPEAKER_02": (0.0, 1.0),
    "SPEAKER_05": (0.0, 1.0),
    "SPEAKER_07": (1.0, 0.0),
    "SPEAKER_09": (0.9, 0.1),
}

SILENCE = np.zeros(16000 * 60, dtype=np.float32)


class _FakeExtractor:
    dim = 2


def _stub(d: Diarizer, turns: list[SpeakerTurn]) -> Diarizer:
    """Point a Diarizer at synthetic vectors, keyed by each turn's start time."""
    by_start = {t.start: t.speaker for t in turns}
    d._build_embedder = lambda: _FakeExtractor()

    def _embed(samples, rate, start, end):
        return np.array(VECTORS[by_start[start]], dtype=np.float32)

    d._embed = _embed
    return d


def _settings(**kwargs) -> DiarizationSettings:
    return DiarizationSettings(**{"min_region_seconds": 0.0, "merge_threshold": 0.0, **kwargs})


def test_tiny_clusters_are_absorbed_into_their_neighbour():
    """A sub-second singleton loses its label but keeps its turn.

    The turn itself must survive: on a dense call the short interjections are
    the speaker changes, and dropping them merged minutes of dialogue into
    single-speaker blocks.
    """
    turns = [
        SpeakerTurn(0.0, 10.0, "SPEAKER_00"),
        SpeakerTurn(11.0, 11.3, "SPEAKER_01"),  # 0.3s: too small to trust
        SpeakerTurn(12.0, 20.0, "SPEAKER_02"),
    ]
    d = Diarizer(_settings(min_region_seconds=1.0))

    out = d._absorb_tiny_clusters(turns)

    assert len(out) == 3, "no turn may be discarded"
    assert [t.speaker for t in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    assert out[1].start == 11.0 and out[1].end == 11.3


def test_tiny_cluster_absorption_is_off_when_zero():
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(11.0, 11.3, "SPEAKER_01")]
    d = Diarizer(_settings(min_region_seconds=0.0))

    assert d._absorb_tiny_clusters(turns) == turns


def test_all_tiny_clusters_leaves_labels_alone():
    """With nothing big enough to keep, do not relabel into a single speaker."""
    turns = [SpeakerTurn(0.0, 0.3, "SPEAKER_00"), SpeakerTurn(1.0, 1.2, "SPEAKER_01")]
    d = Diarizer(_settings(min_region_seconds=1.0))

    assert d._absorb_tiny_clusters(turns) == turns


def test_second_pass_merges_one_person_split_across_clusters():
    turns = [
        SpeakerTurn(0.0, 10.0, "SPEAKER_00"),
        SpeakerTurn(12.0, 20.0, "SPEAKER_01"),  # same voice as above
        SpeakerTurn(30.0, 40.0, "SPEAKER_02"),
    ]
    d = _stub(Diarizer(_settings(merge_threshold=0.8)), turns)

    merged = d._merge_clusters(turns, SILENCE, 16000)

    assert [t.speaker for t in merged] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_second_pass_is_off_when_threshold_zero():
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(12.0, 20.0, "SPEAKER_01")]
    d = _stub(Diarizer(_settings(merge_threshold=0.0)), turns)

    assert d._merge_clusters(turns, SILENCE, 16000) == turns


def test_second_pass_skipped_for_a_single_cluster():
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(12.0, 20.0, "SPEAKER_00")]
    d = _stub(Diarizer(_settings(merge_threshold=0.8)), turns)

    assert d._merge_clusters(turns, SILENCE, 16000) == turns


def test_second_pass_keeps_distinct_people_apart():
    """The regression guard: distinct speakers must not be merged into one."""
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(12.0, 20.0, "SPEAKER_02")]
    d = _stub(Diarizer(_settings(merge_threshold=0.8)), turns)

    merged = d._merge_clusters(turns, SILENCE, 16000)

    # Nothing merged, so sherpa's own labels are passed through untouched.
    assert [t.speaker for t in merged] == ["SPEAKER_00", "SPEAKER_02"]


def test_second_pass_labels_are_chronological():
    """A cluster that merges into an earlier one inherits its name, not its own."""
    turns = [
        SpeakerTurn(0.0, 10.0, "SPEAKER_07"),  # first in time, but oddly numbered
        SpeakerTurn(12.0, 20.0, "SPEAKER_09"),  # same voice, merges into 07
        SpeakerTurn(30.0, 40.0, "SPEAKER_05"),  # orthogonal, stays separate
    ]
    d = _stub(Diarizer(_settings(merge_threshold=0.8)), turns)

    merged = d._merge_clusters(turns, SILENCE, 16000)

    assert [t.speaker for t in merged] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_second_pass_survives_an_embedding_failure():
    """A broken second pass must not cost us the transcript's speaker labels."""
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(12.0, 20.0, "SPEAKER_01")]
    d = _stub(Diarizer(_settings(merge_threshold=0.8)), turns)

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    d._embed = boom

    assert d._merge_clusters(turns, SILENCE, 16000) == turns

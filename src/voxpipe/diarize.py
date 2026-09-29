"""Speaker diarization with ONNX models only (no torch).

Pipeline: pyannote segmentation-3.0 proposes *local* speaker classes per
window, a 3D-Speaker model embeds each speech region, and FastClustering merges
the local classes into global speakers. Roughly 0.15 RTF on 8 kHz phone audio
and 0.07 on wideband, which is the price of keeping SPEAKER_00 labels without
PyTorch.

``threshold`` trades off against the number of speakers: below ~0.7 a two-party
call fragments into a dozen "speakers" because narrowband embeddings scatter,
around 0.85 it lands on the right count for phone calls.

That single threshold does not scale to long recordings. Over the 743 speech
regions of a 93-minute meeting the median pairwise cosine was 0.374, so 0.85
sits near the 99th percentile and almost nothing merges: it produced 32 labels,
and even split the dominant voice across two of them. Two cheaper passes fix
most of that without touching FastClustering. Regions shorter than
``min_region_seconds`` never get a label of their own, because a sub-second
turn cannot carry a reliable embedding and mostly becomes a singleton. Then
``merge_threshold`` drives a second pass that compares one duration-weighted
centroid per surviving cluster and merges the closest pair while they stay
similar, which is far more stable than thresholding hundreds of noisy pairs at
once. Both passes are skipped if the corresponding setting is 0.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .config import SAMPLE_RATE, DiarizationSettings
from .models import SpeakerTurn

logger = logging.getLogger(__name__)


class Diarizer:
    """Lazily built sherpa-onnx diarization pipeline, safe to reuse per file."""

    def __init__(self, settings: DiarizationSettings) -> None:
        self._settings = settings
        self._pipeline = None
        self._embedder = None
        self._lock = threading.Lock()

    def _build(self):
        import sherpa_onnx

        settings = self._settings
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(settings.segmentation_model),
                    window_shift_ratio=settings.window_shift_ratio,
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(settings.embedding_model),
                num_threads=_embedding_threads(),
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=settings.num_speakers,
                threshold=settings.threshold,
            ),
            min_duration_on=settings.min_duration_on,
            min_duration_off=settings.min_duration_off,
        )
        if not config.validate():
            raise RuntimeError(
                "sherpa-onnx rejected the diarization config; check that "
                f"{settings.segmentation_model} and {settings.embedding_model} exist"
            )
        return sherpa_onnx.OfflineSpeakerDiarization(config)

    def _build_embedder(self):
        import sherpa_onnx

        if self._embedder is None:
            self._embedder = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(self._settings.embedding_model),
                    num_threads=_embedding_threads(),
                )
            )
        return self._embedder

    def _embed(self, samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
        """One speaker vector for a time range, or zeros if too short to embed."""
        extractor = self._build_embedder()
        chunk = samples[int(start * sample_rate) : int(end * sample_rate)]
        if len(chunk) < sample_rate // 10:
            return np.zeros(extractor.dim, dtype=np.float32)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate, chunk)
        stream.input_finished()
        vector = None
        while not extractor.is_ready(stream):
            vector = extractor.compute(stream)
        if vector is None:
            vector = extractor.compute(stream)
        return np.asarray(vector, dtype=np.float32)

    def _absorb_tiny_clusters(self, turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        """Fold clusters with too little speech into their temporal neighbour.

        A sub-second turn cannot carry a reliable embedding, so it tends to
        become a singleton cluster and a label of its own. Dropping those turns
        is tempting and wrong: on a dense two-party call the short turns are
        exactly the interjections that mark a speaker change, and removing them
        merged whole minutes of dialogue into single-speaker blocks. The turns
        are kept, so attribution keeps its evidence, and only the label is
        reassigned -- to whichever surviving cluster sits closest in time.
        """
        floor = self._settings.min_region_seconds
        if floor <= 0 or not turns:
            return turns

        totals: dict[str, float] = {}
        first: dict[str, float] = {}
        last: dict[str, float] = {}
        for turn in turns:
            span = turn.end - turn.start
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + span
            first.setdefault(turn.speaker, turn.start)
            last[turn.speaker] = max(last.get(turn.speaker, 0.0), turn.end)

        keepers = [s for s in totals if totals[s] >= floor]
        if not keepers:
            return turns
        dropped = [s for s in totals if totals[s] < floor]
        if not dropped:
            return turns

        def nearest(speaker: str) -> str:
            return min(keepers, key=lambda k: min(abs(first[k] - last[speaker]), abs(first[speaker] - last[k])))

        remap = {s: nearest(s) for s in dropped}
        for speaker, target in remap.items():
            logger.info("Absorbing tiny cluster %s (%.2fs) into %s", speaker, totals[speaker], target)

        # Renumber survivors by first appearance so labels stay chronological.
        order: dict[str, str] = {}
        for turn in turns:
            root = remap.get(turn.speaker, turn.speaker)
            order.setdefault(root, f"SPEAKER_{len(order):02d}")

        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=order[remap.get(turn.speaker, turn.speaker)])
            for turn in turns
        ]

    def _merge_clusters(self, turns: list[SpeakerTurn], samples: np.ndarray, sample_rate: int) -> list[SpeakerTurn]:
        """Second clustering pass over one centroid per cluster.

        FastClustering judges every pair of regions against a single threshold,
        which over-splits long recordings. Here each cluster is reduced to a
        duration-weighted centroid first, so the comparison is between a handful
        of well-estimated speaker profiles rather than hundreds of noisy
        per-turn vectors, and the closest pair is merged repeatedly while it
        stays above ``merge_threshold``.
        """
        threshold = self._settings.merge_threshold
        speakers = {t.speaker for t in turns}
        if threshold <= 0 or len(speakers) < 2:
            return turns

        try:
            groups: dict[str, list[int]] = {s: [] for s in speakers}
            for i, turn in enumerate(turns):
                groups[turn.speaker].append(i)
            order = sorted(groups, key=lambda s: turns[groups[s][0]].start)

            centroids: dict[str, np.ndarray] = {}
            weights: dict[str, float] = {}
            for speaker in order:
                idxs = groups[speaker]
                total = sum(turns[i].end - turns[i].start for i in idxs)
                acc = np.zeros(self._build_embedder().dim, dtype=np.float64)
                for i in idxs:
                    turn = turns[i]
                    acc += self._embed(samples, sample_rate, turn.start, turn.end) * (turn.end - turn.start)
                centroids[speaker] = acc / max(total, 1e-9)
                weights[speaker] = total

            parent = {s: s for s in order}

            def find(s: str) -> str:
                while parent[s] != s:
                    parent[s] = parent[parent[s]]
                    s = parent[s]
                return s

            merged = 0
            while True:
                roots = sorted({find(s) for s in order})
                if len(roots) < 2:
                    break
                stacked = np.stack([centroids[r] / (np.linalg.norm(centroids[r]) + 1e-12) for r in roots])
                sim = stacked @ stacked.T
                np.fill_diagonal(sim, -2.0)
                a, b = np.unravel_index(np.argmax(sim), sim.shape)
                if sim[a, b] < threshold:
                    break
                ra, rb = roots[a], roots[b]
                total = weights[ra] + weights[rb]
                centroids[ra] = (centroids[ra] * weights[ra] + centroids[rb] * weights[rb]) / total
                weights[ra] = total
                parent[find(rb)] = find(ra)
                merged += 1

            if not merged:
                return turns

            # Renumber survivors by first appearance so labels stay chronological.
            roots = sorted({find(s) for s in order}, key=lambda r: turns[groups[r][0]].start)
            names = {root: f"SPEAKER_{n:02d}" for n, root in enumerate(roots)}
            for root, name in names.items():
                logger.info("Second pass merged into %s: %.0fs of speech", name, weights[root])
            logger.info(
                "Second clustering pass merged %d cluster(s): %d -> %d",
                merged,
                len(speakers),
                len(roots),
            )
            return [
                SpeakerTurn(
                    start=turn.start,
                    end=turn.end,
                    speaker=names[find(turn.speaker)],
                )
                for turn in turns
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Second clustering pass failed (%s); keeping raw clusters", exc)
            return turns

    def diarize(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[SpeakerTurn]:
        """Return speaker turns for mono float samples."""
        with self._lock:
            if self._pipeline is None:
                logger.info(
                    "Loading diarization models: %s + %s",
                    self._settings.segmentation_model,
                    self._settings.embedding_model,
                )
                self._pipeline = self._build()
            if sample_rate != self._pipeline.sample_rate:
                raise ValueError(f"diarization expects {self._pipeline.sample_rate} Hz, got {sample_rate}")
            result = self._pipeline.process(samples).sort_by_start_time()

        turns = [SpeakerTurn(start=r.start, end=r.end, speaker=f"SPEAKER_{r.speaker:02d}") for r in result]
        turns = self._merge_clusters(turns, samples, sample_rate)
        turns = self._absorb_tiny_clusters(turns)
        speakers = sorted({t.speaker for t in turns})
        logger.info("Diarization found %d speaker(s): %s", len(speakers), ", ".join(speakers))
        return turns


def _embedding_threads() -> int:
    import os

    try:
        return max(1, int(os.environ.get("TRANSCRIBER_DIAR_THREADS", min(8, os.cpu_count() or 4))))
    except ValueError:
        return 4

# Speaker Diarization

VoxPipe uses sherpa-onnx for speaker diarization.

## Labels and identity

Output uses local labels such as:

```text
SPEAKER_00
SPEAKER_01
```

Labels are stable only within one recording.

There is no persistent speaker identity recognition across files.

## Default clustering threshold

```text
TRANSCRIBER_DIAR_THRESHOLD=0.85
```

Measured on a 156-second two-party call:

| Threshold | Speakers found |
|---|---:|
| `0.60` | 5 |
| `0.85` | 2 |
| `0.95` | 1 |

In the current clustering behavior, raising the threshold merges more aggressively and tends to produce fewer speakers.

## Long-recording behavior

The same default does not generalize perfectly to long meetings.

On the original 93-minute multi-party meeting:

- threshold `0.85` produced 30 labels;
- threshold `0.95` produced 18 labels;
- the two largest clusters represented the same dominant speaker;
- those clusters merged above approximately `0.80`;
- much of the remaining long tail had cosine similarity of only about `0.12-0.27` to the dominant clusters.

Across 743 speech regions, the measured median pairwise cosine similarity was approximately `0.374`.

This indicates a weakly separated embedding space rather than a problem that can be solved with one threshold alone.

## Post-processing passes

Two cleanup passes were added and enabled by default.

### Duration-weighted centroid merge

```text
TRANSCRIBER_DIAR_MERGE_THRESHOLD=0.70
```

The second pass:

1. reduces each cluster to a duration-weighted centroid;
2. finds the closest cluster pair;
3. merges while similarity remains above the configured threshold.

Calibration points from the original experiments:

- genuinely different speakers: around `0.48`
- one real voice split into separate clusters: merged around `0.80`

### Minimum-region cleanup

```text
TRANSCRIBER_DIAR_MIN_REGION=1.0
```

Very short regions have unreliable embeddings and may become singleton labels.

The turn itself is preserved, but the label is folded into the nearest cluster.

Related controls:

```text
TRANSCRIBER_DIAR_MIN_ON=0.5
TRANSCRIBER_DIAR_MIN_OFF=0.7
```

On the 93-minute meeting, the two post-passes reduced 30 labels to 20 and reunited the split dominant voice.

Transcript text remained unchanged; only speaker labeling changed.

## Fixed speaker count is unreliable

```text
TRANSCRIBER_DIAR_SPEAKERS=-1
```

is the recommended setting.

The original tests against sherpa-onnx 1.13.8 showed that fixed `num_clusters` did not behave as a reliable requested speaker count.

Measured on the 156-second two-party call:

| Requested | Clusters returned | Speech share |
|---|---:|---|
| `2` | 2 | 98% / 2% |
| `3` | 2 | 84% / 16% |
| `4` | 3 | 84% / 13% / 3% |
| `6` | 5 | 84% / 8% / 5% / 2% / 1% |

With `2`, the original result assigned roughly 133.9 s to one speaker and 2.8 s to the other.

With automatic detection (`-1`), the same recording produced approximately 21.3 s / 111.8 s.

This is why fixed speaker count should not currently be used as a shortcut.

## Overlapping speech

Overlap attribution is a separate issue from clustering.

The original implementation changed `assign_speakers` to prefer the latest-starting covering turn, preventing one long overlapping turn from swallowing the rest of a file.

That fix does not repair a bad fixed-count clustering partition.

Attribution remains segment-level rather than word-level.

A speech segment spanning a speaker turn is attributed according to the segment assignment logic; there is no word-level diarization.

## Performance

Original benchmark:

- GTX 1070
- 4 CPU cores
- 156-second Russian call

Diarization using segmentation 3.0 + CAM-PPlus took approximately 23.8 seconds, around 6.6x realtime (RTF ~0.152).

Diarization is CPU-only and can dominate end-to-end runtime.

See [performance.md](performance.md) for full measurements.

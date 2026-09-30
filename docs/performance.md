# Performance

These are historical measurements from the original VoxPipe README.

They are useful as engineering baselines, not universal performance guarantees.

## 156-second Russian call

Hardware:

- NVIDIA GTX 1070
- 4 CPU cores
- whisper.cpp Vulkan

| Stage | Model | Time | Realtime |
|---|---|---:|---:|
| ASR | `small` fp16 | 14.7 s | 10.6x |
| ASR | `small-q5_1` | 14.3 s | 11.1x |
| ASR | `turbo` q5_0 | 14.7-18.0 s | 8.8-10.6x |
| Diarization | segmentation 3.0 + CAM-PPlus | 23.8 s | 6.6x |
| End to end | `turbo` + diarization | ~41 s | ~3.8x |

The earlier faster-whisper implementation took roughly 14.3 seconds for ASR on the same call.

The main benefit of the rewrite was therefore not raw ASR speed, but:

- much smaller Python environment;
- no Hugging Face token requirement;
- working local speaker labels;
- Vulkan rather than a CUDA-specific runtime.

## 93-minute recording

Historical ASR-only measurements:

- `turbo`, without VAD: 503 s
- `turbo`, with VAD: 492 s

Historical end-to-end measurement:

- 93.0 minutes input
- ~20m36s / 1236 s
- ~4.5x realtime

With the clustering post-passes included, the original README recorded approximately:

- 1397 s total
- ~4.0x realtime

The original note attributed some ratio difference to the wideband `.m4a` meeting diarizing faster than the 8 kHz phone audio used for the shorter diarization RTF measurement.

Diarization was the dominant cost.

# Licensing and Attribution

VoxPipe source code is licensed under the MIT License.

Model weights and third-party components are separate works and retain their own licenses.

The authoritative notices belong in:

```text
THIRD-PARTY-NOTICES.md
```

## Model licenses documented in the original README

| Model | License | Copyright / source |
|---|---|---|
| `ggml-silero-v5.1.2.bin` | MIT | Silero Team |
| `sherpa-onnx-pyannote-segmentation-3-0` | MIT | CNRS |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | Apache-2.0 | 3D-Speaker / ModelScope (Alibaba) |

`THIRD-PARTY-NOTICES.md` is intentionally tracked in Git because `vendor/` is git-ignored.

Keeping notices only beside downloaded binaries would make them disappear in a fresh clone.

The model-fetch script copies the notice file next to downloaded weights.

The built Python distribution also ships `LICENSE` and third-party notices under distribution license metadata.

## Adapted code

The original README notes that:

- `format.py`
- `merge.py`

were adapted from the upstream `transcriber` project (`faster-whisper + pyannote`, v0.2.3).

That project declares MIT licensing in `pyproject.toml` but does not ship a LICENSE file or name a copyright holder.

The VoxPipe `api.py` follows the same endpoint shape and JSON schema.

Other VoxPipe modules were described as new.

Keep this provenance information until attribution has been independently reviewed and intentionally replaced with a clearer legal notice.

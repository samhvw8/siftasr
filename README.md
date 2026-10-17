# siftasr

Shared speech-to-text router (`en`, `vi`, `zh`, and Whisper fallback) and ElevenLabs Scribe-style speaker attribution (`A`, `B`, … per word and turn) for `mediasift`, `anydub`, and other workspace tools.

## Lanes

| Language | Apple Silicon | NVIDIA CUDA | CPU |
|---|---|---|---|
| `en` (+ 24 EU) | `parakeet-mlx` (`parakeet-tdt-0.6b-v3`) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| `vi` | `quan-coreml` (`QUAN-Transcribe-1.0-CoreML`, EG2 + CTC on Core ML) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| `zh` / `yue` | `qwen3-mlx` (`Qwen3-ASR-0.6B-4bit`, optional `[zh]`) or `mlx-whisper` | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| Default / low LID confidence (`< 0.65`) | `mlx-whisper` (`whisper-large-v3-turbo`) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |

## Install

```bash
# Local workspace editable
uv add --editable ~/workspace/libs/siftasr

# Public GitHub (for standalone / remote skill install)
uv tool install git+https://github.com/samhvw8/siftasr
```

---
type: Component
title: Routing and speaker attribution
description: Two-stage language detection, per-language ASR lanes (en, vi, zh, Whisper), and turn/word speaker diarization.
tags: [asr, routing, diarization, coreml, mlx]
status: stable
resource: src/siftasr/asr.py
generated:
  by: samhv
  at: '2026-10-17T13:30:00Z'
sources:
  - id: asr
    resource: src/siftasr/asr.py
  - id: speakers
    resource: src/siftasr/speakers.py
  - id: coreml
    resource: src/siftasr/coreml.py
---

# Language Detection and Routing

`siftasr.asr.transcribe` (`src/siftasr/asr.py`[^asr]) separates language identification from the transcription model so specialist models never run on the wrong language:

1. **Caller / metadata hint first:** When `language` is passed by the caller or `yt-dlp` metadata, `detect_language_info` is skipped.
2. **Confidence-gated `whisper-tiny` fallback:** `detect_language_info` runs `faster-whisper` `tiny` (`int8` on CPU) on a 30 s slice 20% into the recording. If `language_probability < 0.65`, `language` is treated as `None` and routed to the multilingual Whisper fallback.

| Language | Apple Silicon | NVIDIA CUDA | CPU |
|---|---|---|---|
| `en` (+ 24 EU in `PARAKEET_LANGS`) | `parakeet-mlx` (`parakeet-tdt-0.6b-v3`) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| `vi` | `quan-coreml` (`qnguyen3/QUAN-Transcribe-1.0-CoreML` via `src/siftasr/coreml.py`[^coreml]) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| `zh` / `yue` | `qwen3-mlx` (`mlx-qwen3-asr`, optional extra) or `mlx-whisper` | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| Everything else / `prob < 0.65` | `mlx-whisper` (`whisper-large-v3-turbo`) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |

# Speaker Attribution (`diarize_cues` and `voices`)

`siftasr.speakers` (`src/siftasr/speakers.py`[^speakers]) extracts L2-normalized `nemo_en_titanet_small.onnx` embeddings via `sherpa-onnx` and clusters them greedily into `'A'`, `'B'`, … in order of first appearance, folding single-line noise clusters into the nearest major speaker on recordings with 20+ lines.

When `diarize=True` is passed to `transcribe`, `diarize_cues` also inspects word timestamps inside each cue, splits cues at internal pauses (`>= 0.35 s`) when the speaker changes, and populates `.speaker` on both every `Word` and every `Cue`.

[^asr]: Routing table, confidence-gated LID, and `diarize_cues` in `src/siftasr/asr.py`.
[^speakers]: TitaNet-small embedding extraction and centroid clustering in `src/siftasr/speakers.py`.
[^coreml]: `pyobjc` Core ML compiler and runner with `objc.autorelease_pool()` in `src/siftasr/coreml.py`.

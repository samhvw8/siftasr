---
type: Component
title: Routing and speaker attribution
description: Two-stage language detection, per-language ASR lanes (en, vi, zh, Whisper), and turn/word speaker diarization.
tags: [asr, routing, diarization, coreml, mlx]
status: stable
resource: src/siftasr/asr.py
generated:
  by: human:samhv
  at: '2026-10-17T16:30:00Z'
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
2. **Confidence-gated `whisper-tiny` fallback:** `detect_language_info` reuses a cached `faster-whisper` `tiny` (`int8` on CPU) instance and calls encoder-only `detect_language` on a 30 s slice 20% into the recording. If `language_probability < 0.65`, `language` is treated as `None` and routed to the multilingual Whisper fallback.

| Language | Apple Silicon | NVIDIA CUDA | CPU |
|---|---|---|---|
| `en` (+ 24 EU in `PARAKEET_LANGS`) | `parakeet-mlx` (`parakeet-tdt-0.6b-v3`) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| `vi` | `quan-coreml` (`qnguyen3/QUAN-Transcribe-1.0-CoreML` via `src/siftasr/coreml.py`[^coreml]) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| `zh` / `yue` | `qwen3-mlx` (`mlx-qwen3-asr`, optional extra) or `mlx-whisper` | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |
| Everything else / `prob < 0.65` | `mlx-whisper` (`whisper-large-v3-turbo`) | `faster-whisper` (`large-v3-turbo`, batched fp16) | `faster-whisper` (`small`, int8) |

# Word Segmentation and Cue Splitting (`parakeet_tokens_to_words` and `fold_short_cues`)

`parakeet-mlx` emits SentencePiece/BPE subword tokens in `s.tokens` and only splits sentences on punctuation followed by whitespace, which can leave unpunctuated stretches over 20 s or orphan leading commas. `_parakeet_mlx` (`src/siftasr/asr.py`[^asr]) streams `150 s` `int16` PCM slices (`4 s` overlap) directly into `get_logmel`, `m.encoder`, and `_decode_tdt_fast` in memory without writing a temporary `.wav` or spawning a second `ffmpeg` decode. `_decode_tdt_fast` pre-projects `m.joint.enc(features)` once per chunk, caches `m.decoder` + `m.joint.pred` across blank frames, evaluates an 8-frame speculative lookahead window via `@mx.compile` under a scoped 48 MB MLX cache limit (resetting to `0` and clearing after each chunk), and skips the unused 8,193-class softmax entropy calculation—producing bit-for-bit identical tokens and timestamps to `ParakeetTDT.decode_greedy` at up to 5.7× faster decode speed and 17× lower decoder CPU time. `parakeet_tokens_to_words` merges subword pieces into whole timed words before passing them to `words_to_cues(..., max_len=8.0, gap=0.8)`, and `fold_short_cues` folds 1–2 word mid-sentence pause fragments (`< 1.0 s`) back into their sentence when the combined cue stays `<= 10.0 s`. `decode_audio` writes `ffmpeg` output to an unlinked temporary file and loads it in a single `np.fromfile` allocation, while `split_quiet` finds low-energy cuts in $O(N)$ via `np.cumsum`.

# Speaker Attribution (`diarize_cues` and `voices`)

`siftasr.speakers` (`src/siftasr/speakers.py`[^speakers]) caches the `sherpa-onnx` `nemo_en_titanet_small.onnx` extractor, bounds each span to a central `<= 5.0 s` window across P-core worker threads, and runs two-stage sparse attribution on recordings with 20+ spans:

1. **Stage 1 anchor seeding (`>= 2.5 s`) and single-speaker early exit:** Only substantial turns (`>= 2.5 s`) are embedded first to seed speaker centroids in chronological order, followed by centroid consolidation (`< 0.85 * thr`). If all anchors belong to a single speaker, `voices` returns `[None, ...]` immediately without embedding short spans.
2. **Stage 2 sparse boundary embedding, splinter folding, and smoothing:** On multi-speaker recordings, only short spans (`0.8 s <= dur < 2.5 s`) near speaker transitions or inside multi-cue short runs are embedded; interior 1–2 cue short spans within the same anchor speaker's turn are resolved by smoothing. Splinter clusters (`< 8.0 s` total speech or `< max(3, N // 120)` cues) fold into their nearest major speaker, and sub-`0.8 s` `None` backchannels inherit continuity from adjacent turns.
When `diarize=True` is passed to `transcribe`, `diarize_cues` also inspects word timestamps inside each cue, splits cues at internal pauses (`>= 0.35 s`) when the speaker changes, and populates `.speaker` on both every `Word` and every `Cue`.
[^asr]: Routing table, confidence-gated LID, and `diarize_cues` in `src/siftasr/asr.py`.
[^speakers]: TitaNet-small embedding extraction and centroid clustering in `src/siftasr/speakers.py`.
[^coreml]: `pyobjc` Core ML compiler and runner with `objc.autorelease_pool()` in `src/siftasr/coreml.py`.

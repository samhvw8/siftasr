---
type: Decision
title: Performance and speaker diarization
description: Compiled speculative TDT greedy decoding, 150 s / 4 s in-memory chunking, single-allocation PCM decoding, and two-stage anchor-seeded TitaNet clustering.
tags: [performance, memory, parakeet, mlx, diarization, titanet]
status: stable
resource: src/siftasr/asr.py
generated:
  by: human:samhv
  at: '2026-10-17T16:35:00Z'
sources:
  - id: asr
    resource: src/siftasr/asr.py
  - id: speakers
    resource: src/siftasr/speakers.py
  - id: tests
    resource: tests/test_siftasr.py
---

# Context

Long-form lecture and interview recordings (30–60+ minutes) expose five failure modes in stock local ASR and speaker-embedding pipelines on Apple Silicon:

1. **Pipe double-buffering in `ffmpeg` PCM decode:** `subprocess.run(..., capture_output=True)` buffers chunks in Python and concatenates them at exit, doubling peak RSS (`271 MB` for a `102 MB` `int16` array).
2. **15 s LCS overlap word deletion in `parakeet-mlx`:** When `overlap_duration=15.0` is used with `merge_longest_common_subsequence`, matching a common word (`"to"`, `"the"`) across the 15 s boundary deletes up to 7 s (~25–30 words) of speech between matched tokens.
3. **Metal dispatch and softmax overhead in `ParakeetTDT.decode_greedy`:** Upstream `decode_greedy` re-runs the 2-layer `640` LSTM and `joint.enc` projection on every blank frame, computes an unused 8,193-class softmax Shannon entropy on every step, and performs 3 GPU-to-CPU `.item()` synchronizations per frame.
4. **BPE subword token splitting:** `parakeet-mlx` stores SentencePiece subword pieces in `s.tokens`, causing naive word joiners to split mid-word or leave leading commas on cues.
5. **Short backchannel speaker drift:** Sub-2.5 s backchannels (`"Yeah"`, `"Right"`, `"Um"`) have noisy TitaNet embeddings that can seed up to 8 phantom speaker clusters during a 55-minute talk.

# Decisions

## 1. Single-Allocation `decode_audio` (`src/siftasr/asr.py`[^asr])

`decode_audio` writes `ffmpeg` `s16le` output directly to an unlinked `tempfile.TemporaryFile()` and reads it once via `np.fromfile(tmp, dtype=np.int16)`. Python performs zero pipe buffering (`0.03 s` CPU time, `140 MB` peak RSS on a 55m 36s video).

## 2. Direct In-Memory `150 s / 4 s` Chunking + Compiled Speculative TDT Decoding (`_decode_tdt_fast`[^asr])

- `_parakeet_mlx` slices `pcm[start:end]` (`150 s` chunk, `4 s` overlap) directly in memory without writing a temporary `.wav` or spawning a second `ffmpeg` decode, and merges adjacent chunks with `merge_longest_contiguous` (`merge_longest_common_subsequence` fallback).
- `_decode_tdt_fast` pre-projects `m.joint.enc(features)` once per chunk, caches `m.decoder` + `m.joint.pred` across blank frames, evaluates an 8-frame speculative lookahead window via `@mx.compile` under a scoped 48 MB MLX cache pool (resetting `mx.set_cache_limit(0); mx.clear_cache()` after each chunk), and skips the 8,193-class softmax entropy calculation.
- `parakeet_tokens_to_words` merges SentencePiece/BPE pieces into whole words before `words_to_cues(..., max_len=8.0, gap=0.8)`, and `fold_short_cues` folds 1–2 word mid-sentence pause fragments (`< 1.0 s`) back into their sentence[^tests].

## 3. Two-Stage Anchor-Seeded Speaker Attribution (`src/siftasr/speakers.py`[^speakers])

- `embed_spans` caches `sherpa_onnx.SpeakerEmbeddingExtractor`, bounds each span to its central `<= 5.0 s` window, and caps worker concurrency to `min(4, os.cpu_count())` so diarization never raises process peak RSS above the ASR ceiling.
- `voices` runs Stage 1 on anchor spans (`>= 2.5 s`), consolidates converged centroids (`< 0.85 * thr`), and exits immediately (`[None] * n`) when only one speaker is present.
- On multi-speaker recordings, Stage 2 embeds only short spans (`0.8–2.5 s`) near speaker transitions or inside multi-cue short runs; interior 1–2 cue short spans within the same anchor speaker's turn are resolved by deterministic smoothing (`cluster_embeddings`[^speakers] + unit tests[^tests]).

# Measured Impact (55m 36s Lecture + 5-Questioner Q&A)

| Metric | `v0.1` Baseline | `v0.4` Shipped | Change |
|---|---|---|---|
| `_parakeet_mlx` Wall Time | `149.18 s` (`22.4× RT`) | **`85.61 s` (`39.0× RT`)** | **-63.6 s (+74.1% faster)** |
| `_parakeet_mlx` CPU Time | `104.20 s` | **`49.39 s`** | **-52.6% CPU** |
| End-to-End Wall Time | `161.37 s` (`20.7× RT`) | **`98.94 s` (`33.7× RT`)** | **-62.4 s (+63.1% faster)** |
| End-to-End CPU Time | `155.84 s` | **`84.47 s`** | **-45.8% CPU** |
| Process Peak RSS | `1,144.0 MB` | **`629.5 MB`** | **-45.0% RAM** |
| Solo Lecture Purity (`0–1280 s`) | `97.8%` (`225/230`) | **`100.0%` (`263/263`)** | **0 false flips; 6 total speakers (`A`–`F`)** |

[^asr]: `_decode_tdt_fast`, `_parakeet_mlx`, `parakeet_tokens_to_words`, `fold_short_cues`, and `decode_audio` in `src/siftasr/asr.py`.
[^speakers]: `embed_spans`, `cluster_embeddings`, and `voices` in `src/siftasr/speakers.py`.
[^tests]: Unit tests covering BPE word merging, `split_quiet`, anchor seeding, and two-stage `voices` in `tests/test_siftasr.py`.

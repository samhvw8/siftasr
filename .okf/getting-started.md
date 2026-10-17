---
type: Guide
title: Getting started
description: What siftasr does, how tools in the workspace depend on it, and how it stays installable online.
tags: [onboarding, asr, diarization]
status: stable
generated:
  by: samhv
  at: '2026-10-17T13:30:00Z'
sources:
  - id: pyproject
    resource: pyproject.toml
  - id: asr-mod
    resource: src/siftasr/asr.py
---

# Overview

`siftasr` is the workspace's shared speech-to-text and speaker-attribution library (`pyproject.toml`[^pyproject], `src/siftasr/asr.py`[^asr-mod]). It routes each language (`en`, `vi`, `zh`, and fallback languages) to the fastest accurate local engine and attaches ElevenLabs Scribe-style speaker letters (`A`, `B`, …) to words and cues.

[^pyproject]: Package metadata and optional extras in `pyproject.toml`.
[^asr-mod]: Routing, confidence-gated language detection, Core ML QUAN runner, and SSH offload in `src/siftasr/asr.py`.

# Where to go next

* [Routing and speaker attribution](/architecture/routing-and-speakers.md) - How language detection, per-language lanes, and turn/word speaker diarization work.

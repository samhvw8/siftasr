---
okf_version: '0.2'
---

# siftasr

* [Getting started](getting-started.md) - What siftasr does, how projects install it, and where to read next.

# Architecture

* [Routing and speaker attribution](architecture/routing-and-speakers.md) - Language detection, per-language ASR lanes (`en`, `vi`, `zh`, Whisper), and ElevenLabs-style speaker attribution.

# Decisions

* [Performance and speaker diarization](decisions/performance-and-diarization.md) - Why siftasr uses compiled speculative TDT decoding, 150 s / 4 s in-memory chunking, and two-stage sparse anchor-seeded speaker clustering.

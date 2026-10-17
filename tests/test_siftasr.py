"""Tests for siftasr routing, LID confidence gating, CTC decoding, cue splitting, and speaker diarization."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import siftasr
from siftasr import Cue, Word, asr, speakers


def test_pick_routes_en_vi_zh_and_whisper_fallback(monkeypatch):
    monkeypatch.setattr(asr, "apple_silicon", lambda: True)
    monkeypatch.setattr(asr, "quan_available", lambda: True)
    available_mods = {"parakeet_mlx", "mlx_qwen3_asr", "mlx_whisper", "faster_whisper"}
    monkeypatch.setattr(asr, "_has", lambda mod: mod in available_mods)

    assert asr.pick("auto", "en") == "parakeet-mlx"
    assert asr.pick("auto", "fr") == "parakeet-mlx"
    assert asr.pick("auto", "vi") == "quan-coreml"
    assert asr.pick("auto", "zh") == "qwen3-mlx"
    assert asr.pick("auto", "yue") == "qwen3-mlx"
    assert asr.pick("auto", "ja") == "mlx-whisper"
    assert asr.pick("auto", None) == "mlx-whisper"

    # Without mlx_qwen3_asr installed, zh falls back to mlx-whisper cleanly.
    available_mods.remove("mlx_qwen3_asr")
    assert asr.pick("auto", "zh") == "mlx-whisper"


def test_detect_language_confidence_gate(monkeypatch):
    monkeypatch.setattr(asr, "_has", lambda mod: True)

    class FakeWhisper:
        prob = 0.91
        lang = "vi"

        def __init__(self, *a, **kw):
            pass

        def transcribe(self, clip, beam_size=1):
            return [], SimpleNamespace(language=self.lang, language_probability=self.prob)

    import sys
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper))

    pcm = np.zeros(16000 * 5, dtype=np.int16)
    assert asr.detect_language(pcm, min_prob=0.65) == "vi"

    FakeWhisper.prob = 0.42
    assert asr.detect_language(pcm, min_prob=0.65) is None


def test_ctc_words_and_words_to_cues():
    pieces = ["<blank>", "<unk>", "▁xin", "▁chào", "▁bạn", "."]
    # frames: 0=blank, 2="▁xin" (frames 1,2), 0=blank, 3="▁chào" (frame 4), 4="▁bạn" (frame 5), 5="." (frame 6)
    best = np.array([0, 2, 2, 0, 3, 4, 5, 0], dtype=np.int64)
    words = asr.ctc_words(best, pieces, offset=1.0)
    assert [w[2].strip() for w in words] == ["xin", "chào", "bạn."]
    cues = asr.words_to_cues(words)
    assert len(cues) == 1
    assert cues[0].text == "xin chào bạn."
    assert len(cues[0].words) == 3


def test_diarize_cues_splits_on_speaker_change(monkeypatch):
    # One cue containing two sub-turns separated by a 0.5 s pause where speaker switches A -> B.
    words = [
        Word(0.0, 0.5, "Hello"),
        Word(0.5, 1.0, " there"),
        Word(1.5, 2.0, " General"),
        Word(2.0, 2.5, " Kenobi."),
    ]
    cue = Cue(0.0, 2.5, "Hello there General Kenobi.", words=words)

    # Mock speaker embeddings: turn 1 (0.0-1.0) is [1, 0], turn 2 (1.5-2.5) is [0, 1]
    def fake_voices(pcm, spans, thr=0.6):
        assert spans == [(0.0, 1.0), (1.5, 2.5)]
        return ["A", "B"]

    monkeypatch.setattr(speakers, "voices", fake_voices)
    pcm = np.zeros(16000 * 3, dtype=np.int16)
    out_cues, who = asr.diarize_cues(pcm, [cue], turn_gap=0.35)

    assert len(out_cues) == 2
    assert who == ["A", "B"]
    assert out_cues[0].speaker == "A"
    assert out_cues[0].text == "Hello there"
    assert all(w.speaker == "A" for w in out_cues[0].words)
    assert out_cues[1].speaker == "B"
    assert out_cues[1].text == "General Kenobi."
    assert all(w.speaker == "B" for w in out_cues[1].words)


def test_cluster_embeddings_folds_single_line_noise():
    v_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # Noisy single line closer to A than B, but > thr from A
    v_noise = np.array([0.3, 0.1, 0.9486833], dtype=np.float32)
    embs = [v_a] * 10 + [v_b] * 10 + [v_noise]
    labels = speakers.cluster_embeddings(embs, thr=0.4)
    assert labels[:10] == ["A"] * 10
    assert labels[10:20] == ["B"] * 10
    assert labels[20] == "A"


def test_parakeet_tokens_to_words_and_fold_short_cues():
    tokens = [
        {"start": 3.12, "end": 3.20, "text": " So"},
        {"start": 3.28, "end": 3.36, "text": " I"},
        {"start": 3.36, "end": 3.36, "text": "'"},
        {"start": 3.36, "end": 3.36, "text": "m"},
        {"start": 3.68, "end": 3.84, "text": " Mil"},
        {"start": 3.84, "end": 4.00, "text": "es"},
        {"start": 4.00, "end": 4.72, "text": " Kranmer"},
        {"start": 4.72, "end": 4.72, "text": ","},
        {"start": 4.88, "end": 5.36, "text": " assistant"},
        {"start": 5.36, "end": 6.08, "text": " professor"},
        {"start": 6.08, "end": 6.16, "text": " of"},
        # 0.96s mid-sentence pause before "science." (0.64s, 1 word)
        {"start": 7.12, "end": 7.76, "text": " science."},
    ]
    words = asr.parakeet_tokens_to_words(tokens)
    assert [w[2].strip() for w in words] == ["So", "I'm", "Miles", "Kranmer,", "assistant", "professor", "of", "science."]
    cues = asr.words_to_cues(words, max_len=8.0, gap=0.8)
    assert len(cues) == 1
    assert cues[0].text == "So I'm Miles Kranmer, assistant professor of science."
    assert [w.text for w in cues[0].words] == ["So", "I'm", "Miles", "Kranmer,", "assistant", "professor", "of", "science."]


def test_cluster_embeddings_anchor_seeding_prevents_short_backchannel_clusters():
    v_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # A noisy short 1.0s backchannel ("Okay.") that is far from A and B (> thr)
    v_short = np.array([0.4, 0.0, 0.9165151], dtype=np.float32)
    embs = [v_a, v_short, v_a, None, v_b, v_b]
    spans = [
        (0.0, 4.0),    # A (anchor >= 2.5s)
        (4.2, 5.2),    # short 1.0s backchannel (must NOT seed a phantom speaker)
        (5.5, 9.5),    # A (anchor >= 2.5s)
        (9.7, 10.2),   # 0.5s None backchannel
        (11.0, 15.0),  # B (anchor >= 2.5s)
        (15.2, 19.0),  # B (anchor >= 2.5s)
    ]
    labels = speakers.cluster_embeddings(embs, thr=0.4, spans=spans)
    assert labels == ["A", "A", "A", "A", "B", "B"]


def test_split_quiet_finds_quietest_window():
    sr = asr.SR
    x = np.ones(sr * 10, dtype=np.float32)
    # Insert a 100ms silent valley at 3.50s..3.60s (within the last 2s before the 4s chunk limit)
    x[int(3.50 * sr):int(3.60 * sr)] = 0.0
    pieces = asr.split_quiet(x, chunk_s=4.0, search_s=2.0)
    assert len(pieces) >= 3
    first_off, first_piece = pieces[0]
    assert first_off == 0
    assert int(3.50 * sr) <= len(first_piece) <= int(3.60 * sr)
    assert sum(len(p) for _, p in pieces) == len(x)


def test_voices_two_stage_skips_short_spans_on_single_speaker_and_interior_turns(monkeypatch):
    v_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    calls: list[list[tuple[float, float]]] = []

    def fake_embed_spans(pcm, spans, min_s=0.8, max_workers=None, max_s=3.5):
        calls.append(list(spans))
        return [v_a if s[0] < 100.0 else v_b for s in spans]

    monkeypatch.setattr(speakers, "embed_spans", fake_embed_spans)
    pcm = np.zeros(asr.SR * 200, dtype=np.int16)

    # Case 1: 20 spans where all anchors belong to speaker A -> Stage 2 is skipped completely
    solo_spans = [(i * 4.0, i * 4.0 + (3.0 if i % 2 == 0 else 1.2)) for i in range(20)]
    assert speakers.voices(pcm, solo_spans, thr=0.4) == [None] * 20
    assert len(calls) == 1 and len(calls[0]) == 10  # Only the 10 anchor spans were embedded

    # Case 2: Multi-speaker (A then B) -> interior 1-cue short spans inside A or B are skipped,
    # while the boundary short span between A and B IS embedded in Stage 2.
    calls.clear()
    multi_spans = [
        (0.0, 3.5),     # 0: A anchor
        (4.0, 5.2),     # 1: interior short inside A (skipped)
        (5.5, 9.0),     # 2: A anchor
        (9.5, 10.7),    # 3: interior short inside A (skipped)
        (11.0, 14.5),   # 4: A anchor
        (15.0, 18.5),   # 5: A anchor
        (19.0, 22.5),   # 6: A anchor
        (23.0, 26.5),   # 7: A anchor
        (27.0, 30.5),   # 8: A anchor
        (31.0, 34.5),   # 9: A anchor
        (35.0, 36.2),   # 10: boundary short between A (9) and B (11) -> MUST be embedded
        (100.0, 103.5), # 11: B anchor
        (104.0, 105.2), # 12: interior short inside B (skipped)
        (105.5, 109.0), # 13: B anchor
        (110.0, 113.5), # 14: B anchor
        (114.0, 117.5), # 15: B anchor
        (118.0, 121.5), # 16: B anchor
        (122.0, 125.5), # 17: B anchor
        (126.0, 129.5), # 18: B anchor
        (130.0, 133.5), # 19: B anchor
    ]
    out = speakers.voices(pcm, multi_spans, thr=0.4)
    assert len(calls) == 2
    assert calls[1] == [(35.0, 36.2)]  # Only the 1 boundary short span needed Stage 2 embedding
    assert out[:11] == ["A"] * 11 and out[11:] == ["B"] * 9

    # Case 3: Noisy opening anchor initially exceeds thr (0.4) vs anchor 0, then subsequent
    # anchors pull the two centroids within 0.85 * thr -> Pass-1 consolidation merges them
    # and skips Stage 2 completely.
    calls.clear()
    v_open = np.array([0.55, 0.83516465, 0.0], dtype=np.float32)  # 1 - dot(v_a, v_open) = 0.45 > 0.40
    v_mid = np.array([0.92, 0.39191836, 0.0], dtype=np.float32)   # pulls cluster 1 toward v_a

    def fake_embed_drifting(pcm, spans, min_s=0.8, max_workers=None, max_s=5.0):
        calls.append(list(spans))
        res = []
        for idx, _ in enumerate(spans):
            if idx == 1:
                res.append(v_open)
            elif idx >= 2:
                res.append(v_mid)
            else:
                res.append(v_a)
        return res

    monkeypatch.setattr(speakers, "embed_spans", fake_embed_drifting)
    assert speakers.voices(pcm, solo_spans, thr=0.4) == [None] * 20
    assert len(calls) == 1  # Stage 2 still skipped after Pass-1 centroid consolidation

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

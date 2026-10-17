"""Speech recognition behind one interface, picked per machine and per language.

| machine        | en (+ 24 EU languages) | Vietnamese (vi) | Chinese (zh / yue)             | anything else / low LID prob   |
|----------------|------------------------|-----------------|--------------------------------|--------------------------------|
| Apple Silicon  | parakeet-mlx           | quan-coreml     | qwen3-mlx (or mlx-whisper)     | mlx-whisper                    |
| NVIDIA CUDA    | faster-whisper (batch) | (same)          | (same)                         | faster-whisper, batched on GPU |
| other          | faster-whisper on CPU  | (same)          | (same)                         | faster-whisper on CPU          |

Every backend takes a mono 16 kHz int16 or float32 numpy buffer, so ffmpeg stays the only decoder.
When `diarize=True`, speaker attribution ('A', 'B', ...) is attached to every cue and word, splitting
cues at turn pauses (>= 0.35 s) when the speaker changes (ElevenLabs Scribe style).
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import coreml, speakers

SR = 16000

# Parakeet TDT 0.6B v3: 25 European languages (NVIDIA model card).
PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}
ZH_LANGS = {"zh", "yue"}

DEFAULT_MODEL = {
    "parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v3",
    "qwen3-mlx": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "mlx-whisper": "mlx-community/whisper-large-v3-turbo",
    "faster-whisper-cuda": "large-v3-turbo",
    "faster-whisper-cpu": "small",
}

# QUAN-Transcribe 1.0 (EmbeddingGemma 2 audio encoder + CTC reader), Vietnamese only, as Core ML.
QUAN_REPO = "qnguyen3/QUAN-Transcribe-1.0-CoreML"
QUAN_REV = "13c3a2cf974c251668c1481713d35669aa2fb3e1"
QUAN_PKG = "QuanTranscribe-fp16.mlpackage"
QUAN_FN = "len_30s"
FRAME_S = 0.04


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None

    def to_dict(self) -> dict:
        d = {"start": round(self.start, 2), "end": round(self.end, 2), "text": self.text}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"start": round(self.start, 2), "end": round(self.end, 2), "text": self.text}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.words:
            d["words"] = [w.to_dict() for w in self.words]
        return d


@dataclass
class AsrResult:
    cues: list[Cue]
    backend: str
    model: str
    language: str | None
    seconds: float
    voices: list[str | None] | None = None
    language_probability: float | None = None

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "backend": self.backend,
            "model": self.model,
            "seconds": self.seconds,
            "voices": self.voices,
            "cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in self.cues],
            "segments": [
                {
                    "id": i,
                    "start": round(c.start, 2),
                    "end": round(c.end, 2),
                    "text": c.text,
                    **({"speaker": c.speaker} if c.speaker else {}),
                    **({"words": [w.to_dict() for w in c.words]} if c.words else {}),
                }
                for i, c in enumerate(self.cues)
            ],
        }


def _log(msg: str) -> None:
    print(f"siftasr: {msg}", file=sys.stderr)


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) * (1.0 / 32768.0) if pcm.dtype.kind == "i" else pcm.astype(np.float32)


@functools.lru_cache(maxsize=1)
def _lid_model():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="cpu", compute_type="int8")


def detect_language_info(pcm: np.ndarray) -> tuple[str | None, float]:
    """Whisper-tiny on 30 s taken a fifth of the way in (past intros and music beds).
    Returns (iso639_1_code, probability)."""
    if not _has("faster_whisper"):
        return None, 0.0
    from faster_whisper import WhisperModel
    start = int(len(pcm) * 0.2)
    clip = _float(pcm[start:start + 30 * SR])
    if len(clip) < SR:
        clip = _float(pcm[:30 * SR])
    wm = _lid_model() if getattr(WhisperModel, "__module__", "").startswith("faster_whisper") else WhisperModel("tiny", device="cpu", compute_type="int8")
    if hasattr(wm, "detect_language"):
        lang, prob, _ = wm.detect_language(clip)
        return lang, float(prob or 0.0)
    _, info = wm.transcribe(clip, beam_size=1)
    prob = float(getattr(info, "language_probability", 1.0) or 0.0)
    return getattr(info, "language", None), prob

def detect_language(pcm: np.ndarray, min_prob: float = 0.65) -> str | None:
    """Return the detected ISO 639-1 language code when confidence >= min_prob, else None
    so the router falls back to Whisper instead of mis-routing a noisy/mixed file."""
    lang, prob = detect_language_info(pcm)
    if lang and prob >= min_prob:
        return lang
    return None


def quan_available() -> bool:
    return coreml.available()


def pick(requested: str, language: str | None) -> str:
    if requested != "auto":
        return requested
    if apple_silicon():
        if language == "vi" and quan_available():
            return "quan-coreml"
        if language in ZH_LANGS and _has("mlx_qwen3_asr"):
            return "qwen3-mlx"
        if language in PARAKEET_LANGS and _has("parakeet_mlx"):
            return "parakeet-mlx"
        if _has("mlx_whisper"):
            return "mlx-whisper"
    if not _has("faster_whisper"):
        raise SystemExit("--asr needs a backend: `uv tool install siftasr`")
    return "faster-whisper"


def transcribe(
    pcm: np.ndarray,
    backend: str = "auto",
    model: str | None = None,
    language: str | None = None,
    *,
    diarize: bool = False,
    min_lid_prob: float = 0.65,
) -> AsrResult:
    prob: float | None = None
    if language is None and backend in ("auto", "mlx-whisper", "faster-whisper"):
        raw_lang, prob = detect_language_info(pcm)
        language = raw_lang if (raw_lang and prob >= min_lid_prob) else None
        _log(f"detected language: {raw_lang} (p={prob:.2f}) -> routed language={language}")
    name = pick(backend, language)
    t0 = time.perf_counter()
    if name == "parakeet-mlx":
        cues, used = _parakeet_mlx(pcm, model or DEFAULT_MODEL[name])
    elif name == "qwen3-mlx":
        cues, used = _qwen3_mlx(pcm, model or DEFAULT_MODEL[name], language)
    elif name == "mlx-whisper":
        cues, used = _mlx_whisper(pcm, model or DEFAULT_MODEL[name], language)
    elif name == "faster-whisper":
        cues, used = _faster_whisper(pcm, model, language)
    elif name == "quan-coreml":
        cues, used = _quan_coreml(pcm)
    else:
        raise SystemExit(f"unknown ASR backend {name!r}")
    dt = time.perf_counter() - t0
    _log(f"ASR {name} ({used}): {len(pcm) / SR / max(dt, 1e-6):.1f}x realtime")
    res = AsrResult(cues, name, used, language, round(dt, 1), language_probability=prob)
    if diarize and cues:
        res.cues, res.voices = diarize_cues(pcm, cues)
    return res


def _join_words(ws: list[Word]) -> str:
    out: list[str] = []
    for w in ws:
        s = w.text.strip()
        if not s:
            continue
        if out and not (ord(s[0]) > 0x2E80 and ord(out[-1][-1]) > 0x2E80):
            out.append(" " + s)
        else:
            out.append(s)
    return "".join(out)


def diarize_cues(pcm: np.ndarray, cues: list[Cue], turn_gap: float = 0.35, thr: float = 0.6) -> tuple[list[Cue], list[str | None]]:
    """ElevenLabs Scribe-style speaker attribution:
    1. When cues carry timed `words`, split cues at internal pauses >= `turn_gap` where the
       speaker changes, and attach `speaker` ('A', 'B', ...) to every `Word` and `Cue`.
    2. Otherwise, embed each cue span and attach `speaker` per cue."""
    sub_cues: list[Cue] = []
    for c in cues:
        if len(c.words) >= 4:
            turns: list[list[Word]] = [[c.words[0]]]
            for w in c.words[1:]:
                if (w.start - turns[-1][-1].end >= turn_gap
                        and turns[-1][-1].end - turns[-1][0].start >= 0.8):
                    turns.append([w])
                else:
                    turns[-1].append(w)
            if len(turns) > 1 and all(t[-1].end - t[0].start >= 0.8 for t in turns):
                for t in turns:
                    sub_cues.append(Cue(t[0].start, t[-1].end, _join_words(t), words=list(t)))
                continue
        sub_cues.append(c)

    who = speakers.voices(pcm, [(c.start, c.end) for c in sub_cues], thr=thr)
    merged: list[Cue] = []
    merged_who: list[str | None] = []
    for c, spk in zip(sub_cues, who):
        c.speaker = spk
        for w in c.words:
            w.speaker = spk
        if (merged and merged_who[-1] == spk and spk is not None
                and c.start - merged[-1].end < 0.6
                and c.end - merged[-1].start <= 8.0
                and not merged[-1].text.endswith((".", "?", "!", "。", "？", "！"))):
            merged[-1].end = c.end
            sep = "" if (merged[-1].text and c.text and ord(c.text[0]) > 0x2E80) else " "
            merged[-1].text = (merged[-1].text + sep + c.text).strip()
            merged[-1].words.extend(c.words)
        else:
            merged.append(c)
            merged_who.append(spk)
    return merged, merged_who


def _mlx_lean() -> None:
    """MLX keeps freed GPU buffers in a cache: 5.5 GB after a 2-minute file on an M1 Pro,
    on top of a 3.1 GB peak, which pushed a 16 GB Mac into swap. No cache costs no speed."""
    import mlx.core as mx
    mx.set_cache_limit(0)


def parakeet_tokens_to_words(tokens: list) -> list[tuple[float, float, str]]:
    """Merge Parakeet SentencePiece/BPE subword tokens into whole words with a leading space
    so `words_to_cues` splits at natural sentence/clause boundaries and never mid-token or before a comma."""
    words: list[list] = []
    for t in tokens:
        piece = str(t["text"] if isinstance(t, dict) else getattr(t, "text", ""))
        if not piece:
            continue
        s = float(t["start"] if isinstance(t, dict) else getattr(t, "start", 0.0))
        e = float(t["end"] if isinstance(t, dict) else getattr(t, "end", s))
        if piece.startswith(" ") or not words:
            w = piece.lstrip(" ")
            if w:
                words.append([s, max(e, s + 0.02), " " + w])
        else:
            words[-1][1] = max(words[-1][1], e)
            words[-1][2] += piece
    return [(round(s, 2), round(e, 2), w) for s, e, w in words if w.strip()]


@functools.lru_cache(maxsize=1)
def _parakeet_model(repo: str):
    from parakeet_mlx import from_pretrained
    return from_pretrained(repo)


def _decode_tdt_fast(m, features, lengths, offset: float = 0.0, win: int = 8) -> list:
    """Compiled speculative TDT greedy decoder (bit-for-bit identical to ParakeetTDT.decode_greedy):
    1. Pre-projects `m.joint.enc(features)` once per chunk on GPU and drops `features`.
    2. Caches `m.decoder` + `m.joint.pred` across blank frames (where `last_token` is unchanged).
    3. Evaluates a fixed `win`-frame lookahead window via `@mx.compile` and skips the 8,193-class
       softmax entropy computation (`confidence` is unused downstream), stepping through runs of
       consecutive blank frames in pure Python with zero extra Metal synchronizations."""
    import mlx.core as mx
    from parakeet_mlx.alignment import AlignedToken

    vocab_len = len(m.vocabulary)
    durations = m.durations
    max_symbols = m.max_symbols
    time_ratio = m.time_ratio
    act = m.joint.joint_net[0]
    out_proj = m.joint.joint_net[2]

    kernels = getattr(m, "_siftasr_tdt_kernels", None)
    if kernels is None:
        @mx.compile
        def step_with_dec(tok, h, c, enc_win):
            dec_out, (nh, nc) = m.decoder(tok, (h, c))
            dec_out = dec_out.astype(enc_win.dtype)
            nh = nh.astype(enc_win.dtype)
            nc = nc.astype(enc_win.dtype)
            pred_proj = m.joint.pred(dec_out)[0, 0]
            logits = out_proj(act(enc_win + pred_proj))
            return (
                mx.argmax(logits[:, : vocab_len + 1], axis=-1),
                mx.argmax(logits[:, vocab_len + 1 :], axis=-1),
                nh,
                nc,
                pred_proj,
            )

        @mx.compile
        def step_joint_only(pred_proj, enc_win):
            logits = out_proj(act(enc_win + pred_proj))
            return (
                mx.argmax(logits[:, : vocab_len + 1], axis=-1),
                mx.argmax(logits[:, vocab_len + 1 :], axis=-1),
            )

        vocab_pieces = tuple(w.replace("\u2581", " ") for w in m.vocabulary)
        kernels = (step_with_dec, step_joint_only, vocab_pieces)
        try:
            setattr(m, "_siftasr_tdt_kernels", kernels)
        except Exception:
            pass
    step_with_dec, step_joint_only, vocab_pieces = kernels

    enc_all = m.joint.enc(features)
    mx.eval(enc_all)
    mx.set_cache_limit(48 * 1024 * 1024)
    hypothesis: list = []
    try:
        length = int(lengths[0])
        enc_b = mx.pad(enc_all[0, :length], [(0, win), (0, 0)])
        mx.eval(enc_b)
        step = 0
        new_symbols = 0
        last_tok = None
        hid = None
        need_dec = True
        pred_proj = None
        dec_hid = None
        while step < length:
            enc_win = enc_b[step : step + win]
            if need_dec:
                if last_tok is None:
                    dec_out, (nh, nc) = m.decoder(None, None)
                    dec_out = dec_out.astype(enc_b.dtype)
                    dec_hid = (nh.astype(enc_b.dtype), nc.astype(enc_b.dtype))
                    pred_proj = m.joint.pred(dec_out)[0, 0]
                    pred_arr, dec_arr = step_joint_only(pred_proj, enc_win)
                    mx.eval(pred_arr, dec_arr, pred_proj, dec_hid[0], dec_hid[1])
                else:
                    tok = mx.array([[last_tok]], dtype=mx.int32)
                    pred_arr, dec_arr, nh, nc, pred_proj = step_with_dec(tok, hid[0], hid[1], enc_win)
                    dec_hid = (nh, nc)
                    mx.eval(pred_arr, dec_arr, pred_proj, nh, nc)
                need_dec = False
            else:
                pred_arr, dec_arr = step_joint_only(pred_proj, enc_win)
                mx.eval(pred_arr, dec_arr)
            preds = pred_arr.tolist()
            decs = dec_arr.tolist()
            w_start = step
            w_end = min(step + win, length)
            while step < w_end:
                idx = step - w_start
                pred_token = preds[idx]
                decision = decs[idx]
                dur = durations[decision]
                if pred_token != vocab_len:
                    hypothesis.append(
                        AlignedToken(
                            pred_token,
                            start=step * time_ratio + offset,
                            duration=dur * time_ratio,
                            confidence=1.0,
                            text=vocab_pieces[pred_token],
                        )
                    )
                    last_tok = pred_token
                    hid = dec_hid
                    need_dec = True
                    step += dur
                    new_symbols += 1
                    if dur != 0:
                        new_symbols = 0
                    elif max_symbols is not None and max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0
                    break
                step += dur
                new_symbols += 1
                if dur != 0:
                    new_symbols = 0
                elif max_symbols is not None and max_symbols <= new_symbols:
                    step += 1
                    new_symbols = 0
    finally:
        mx.set_cache_limit(0)
        mx.clear_cache()
    return hypothesis


def _parakeet_mlx(pcm: np.ndarray, model: str) -> tuple[list[Cue], str]:
    _mlx_lean()
    import parakeet_mlx
    repo = os.path.expanduser(model)
    from_pre = getattr(parakeet_mlx, "from_pretrained")
    m = _parakeet_model(repo) if getattr(from_pre, "__module__", "").startswith("parakeet_mlx") else from_pre(repo)
    if hasattr(m, "preprocessor_config") and hasattr(m, "generate"):
        import mlx.core as mx
        from parakeet_mlx import DecodingConfig
        from parakeet_mlx.alignment import merge_longest_common_subsequence, merge_longest_contiguous
        from parakeet_mlx.audio import get_logmel

        use_fast_tdt = all(hasattr(m, k) for k in ("encoder", "decoder", "joint", "durations", "vocabulary", "time_ratio"))
        cfg = DecodingConfig()
        sr = int(getattr(m.preprocessor_config, "sample_rate", SR))
        hop = int(getattr(m.preprocessor_config, "hop_length", 160))
        chunk_s, overlap_s = 150.0, 4.0
        c_samp, o_samp = int(chunk_s * sr), int(overlap_s * sr)
        tokens = []
        for start in range(0, len(pcm), c_samp - o_samp):
            end = min(start + c_samp, len(pcm))
            if end - start < hop:
                break
            seg = pcm[start:end]
            ca = mx.array(seg).astype(mx.float32) * (1.0 / 32768.0) if seg.dtype.kind == "i" else mx.array(seg, dtype=mx.float32)
            mel = get_logmel(ca, m.preprocessor_config)
            del ca
            off = start / sr
            if use_fast_tdt:
                features, lengths = m.encoder(mel[None] if len(mel.shape) == 2 else mel)
                del mel
                mx.eval(features, lengths)
                chunk_tokens = _decode_tdt_fast(m, features, lengths, offset=off)
                del features, lengths
            else:
                cr = m.generate(mel, decoding_config=cfg)[0]
                del mel
                for s in cr.sentences:
                    for tok in s.tokens:
                        tok.start += off
                        tok.end = tok.start + tok.duration
                chunk_tokens = cr.tokens
                del cr
            if tokens:
                try:
                    tokens = merge_longest_contiguous(tokens, chunk_tokens, overlap_duration=overlap_s)
                except RuntimeError:
                    tokens = merge_longest_common_subsequence(tokens, chunk_tokens, overlap_duration=overlap_s)
            else:
                tokens = chunk_tokens
            mx.clear_cache()
            if end >= len(pcm):
                break
        if tokens:
            return words_to_cues(parakeet_tokens_to_words(tokens), max_len=8.0, gap=0.8), model
        return [], model

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.astype(np.int16).tobytes())
        r = m.transcribe(f.name, chunk_duration=120.0, overlap_duration=15.0)
    tokens = [t for s in r.sentences for t in (getattr(s, "tokens", None) or [])]
    if tokens:
        return words_to_cues(parakeet_tokens_to_words(tokens), max_len=8.0, gap=0.8), model
    return [Cue(float(s.start), float(s.end), s.text.strip()) for s in r.sentences if s.text.strip()], model
def _qwen3_mlx(pcm: np.ndarray, model: str, language: str | None) -> tuple[list[Cue], str]:
    """Chinese / code-switched speech on Apple Silicon via mlx-qwen3-asr, chunked at quiet points
    with explicit language per chunk so a silent intro never poisons later chunks (mlx-audio #928)."""
    _mlx_lean()
    import mlx_qwen3_asr
    lang_arg = "Chinese" if language in (None, "zh", "yue") else language
    cues: list[Cue] = []
    for off, piece in split_quiet(_float(pcm), 30.0):
        if len(piece) < SR // 5:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((np.clip(piece, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
            res = mlx_qwen3_asr.transcribe(f.name, model=model, language=lang_arg)
        base = off / SR
        segs = res.get("segments") if isinstance(res, dict) else getattr(res, "segments", None)
        if segs:
            for s in segs:
                a = float(s["start"] if isinstance(s, dict) else s.start) + base
                b = float(s["end"] if isinstance(s, dict) else s.end) + base
                t = (s["text"] if isinstance(s, dict) else s.text).strip()
                if t:
                    cues.append(Cue(round(a, 2), round(b, 2), t))
        else:
            t = (res.get("text", "") if isinstance(res, dict) else getattr(res, "text", str(res))).strip()
            if t:
                cues.append(Cue(round(base, 2), round(base + len(piece) / SR, 2), t))
    return cues, model


def _mlx_whisper(pcm: np.ndarray, model: str, language: str | None) -> tuple[list[Cue], str]:
    _mlx_lean()
    import mlx_whisper
    r = mlx_whisper.transcribe(
        _float(pcm), path_or_hf_repo=model, language=language, condition_on_previous_text=False
    )
    cues: list[Cue] = []
    for s in r["segments"]:
        txt = s["text"].strip()
        if not txt:
            continue
        ws = [Word(float(w["start"]), float(w["end"]), w["word"]) for w in s.get("words", []) if w.get("word", "").strip()]
        cues.append(Cue(float(s["start"]), float(s["end"]), txt, words=ws))
    return cues, model


def _faster_whisper(pcm: np.ndarray, model: str | None, language: str | None) -> tuple[list[Cue], str]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    audio = _float(pcm)
    if cuda_available():
        name = model or DEFAULT_MODEL["faster-whisper-cuda"]
        wm = WhisperModel(name, device="cuda", compute_type="float16")
        segs, _ = BatchedInferencePipeline(model=wm).transcribe(
            audio, batch_size=16, language=language, word_timestamps=True
        )
        return words_to_cues([(w.start, w.end, w.word) for s in segs for w in (s.words or [])]), name
    name = model or DEFAULT_MODEL["faster-whisper-cpu"]
    wm = WhisperModel(name, device="cpu", compute_type="int8")
    segs, _ = wm.transcribe(audio, vad_filter=True, language=language, word_timestamps=True)
    return words_to_cues([(w.start, w.end, w.word) for s in segs for w in (s.words or [])]), name


def split_quiet(x: np.ndarray, chunk_s: float, search_s: float = 5.0) -> list[tuple[int, np.ndarray]]:
    """(offset, piece) slices of at most chunk_s that cover x, each cut at the quietest 50 ms
    in the last search_s before the limit (QUAN's own splitter, keeping offsets for timing)."""
    n, win = int(chunk_s * SR), int(0.05 * SR)
    search = min(int(search_s * SR), n - win)
    out, start = [], 0
    while len(x) - start > n:
        lo = start + n - search
        seg = x[lo:start + n].astype(np.float64)
        cs = np.empty(len(seg) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(seg * seg, out=cs[1:])
        energy = cs[win:] - cs[:-win]
        cut = lo + int(np.argmin(energy)) + win // 2
        out.append((start, x[start:cut]))
        start = cut
    out.append((start, x[start:]))
    return out

def ctc_words(best: np.ndarray, pieces: list[str], offset: float = 0.0) -> list[tuple[float, float, str]]:
    """Greedy CTC path (one unit per 40 ms frame) -> timed words. Unit 0 is blank, 1 unknown;
    a piece starting with '▁' starts a word; a word ends one frame after its last emitting frame."""
    words: list[list] = []
    prev = 0
    for t, u in enumerate(best.tolist()):
        if u >= 2:
            at = offset + t * FRAME_S
            if u != prev:
                p = pieces[u]
                if p.startswith("▁") or not words:
                    words.append([at, at + FRAME_S, " " + p.lstrip("▁")])
                else:
                    words[-1][2] += p
            words[-1][1] = at + FRAME_S
        prev = u
    return [(round(a, 2), round(b, 2), w) for a, b, w in words if w.strip()]


def _quan_model():
    """Fetch the pinned Core ML package once (665 MB), compile it once, load the 30 s function."""
    d = speakers.existing_model_path(f"quan-coreml-{QUAN_REV[:8]}")
    if not (d / QUAN_PKG / "Manifest.json").exists():
        from huggingface_hub import snapshot_download
        _log(f"downloading the Vietnamese speech model (665 MB, once) from {QUAN_REPO}")
        snapshot_download(
            QUAN_REPO,
            revision=QUAN_REV,
            local_dir=d,
            allow_patterns=["config.json", "units.json", f"{QUAN_PKG}/*", f"{QUAN_PKG}/**"],
        )
    pieces = json.loads((d / "units.json").read_text(encoding="utf-8"))["pieces"]
    return coreml.load(d / QUAN_PKG, QUAN_FN), pieces


def _quan_coreml(pcm: np.ndarray) -> tuple[list[Cue], str]:
    """One forward pass per <= 30 s piece, no decoder loop. Line times come from the CTC frames."""
    m, pieces = _quan_model()
    n = 30 * SR
    words: list[tuple[float, float, str]] = []
    for off, piece in split_quiet(_float(pcm), 30.0):
        if len(piece) < SR // 10:
            continue
        x = np.zeros((1, n), np.float32)
        x[0, :len(piece)] = piece
        lp, mask = coreml.predict(
            m,
            {
                "audio": x,
                "length": np.array([len(piece)], np.int32),
                "streaming": np.zeros(1, np.float32),
                "left_chunks": np.array([1e4], np.float32),
            },
            ["log_probs", "frame_mask"],
        )
        words += ctc_words(lp[0, :int(mask.sum())].argmax(-1), pieces, off / SR)
    return words_to_cues(words), f"{QUAN_REPO}@{QUAN_REV[:7]}"


_END_PUNCT_SUFFIXES = (".", "?", "!", "。", "？", "！", '."', '?"', '!"', ".)", "?)", "!)")


def fold_short_cues(
    cues: list[Cue], min_dur: float = 1.0, max_merged: float = 10.0, max_gap: float = 2.5
) -> list[Cue]:
    """Fold 1-2 word micro-orphan cues (< 1.0 s) produced by mid-sentence pauses into their
    adjacent sentence cue when the merged cue stays <= `max_merged` seconds."""
    if len(cues) < 2:
        return cues
    out: list[Cue] = []
    i = 0
    while i < len(cues):
        c = cues[i]
        dur = c.end - c.start
        if (dur < min_dur and len(c.words) <= 2
                and not c.text.endswith(_END_PUNCT_SUFFIXES) and i + 1 < len(cues)):
            nxt = cues[i + 1]
            if nxt.start - c.end <= max_gap and nxt.end - c.start <= max_merged:
                cues[i + 1] = Cue(
                    c.start,
                    nxt.end,
                    f"{c.text} {nxt.text}".strip(),
                    words=(c.words or []) + (nxt.words or []),
                )
                i += 1
                continue
        if (out and not out[-1].text.endswith(_END_PUNCT_SUFFIXES)
                and dur < min_dur and len(c.words) <= 2):
            prev = out[-1]
            if c.start - prev.end <= max_gap and c.end - prev.start <= max_merged:
                out[-1] = Cue(
                    prev.start,
                    c.end,
                    f"{prev.text} {c.text}".strip(),
                    words=(prev.words or []) + (c.words or []),
                )
                i += 1
                continue
        out.append(c)
        i += 1
    return out


def words_to_cues(words: list[tuple[float, float, str]], max_len: float = 8.0, gap: float = 0.6) -> list[Cue]:
    """Group timed words into sentence-sized cues: break after . ? !, at a
    pause longer than `gap`, or before a cue passes `max_len` seconds."""
    cues: list[Cue] = []
    cur: list[tuple[float, float, str]] = []

    def emit(ws: list[tuple[float, float, str]]) -> None:
        w_objs = [Word(float(a), float(b), t.strip()) for a, b, t in ws if t.strip()]
        cues.append(Cue(float(ws[0][0]), float(ws[-1][1]), "".join(x[2] for x in ws).strip(), words=w_objs))

    for w in words:
        if cur and w[0] - cur[-1][1] > gap:
            emit(cur)
            cur = []
        elif cur and w[1] - cur[0][0] > max_len:
            ok = [i for i in range(len(cur) - 1) if cur[i][1] - cur[0][0] >= 3.0]
            gap_after = lambda i: cur[i + 1][0] - cur[i][1]
            commas = [i for i in ok if cur[i][2].strip().endswith((",", ";", ":", "，", "、"))]
            pauses = [i for i in ok if gap_after(i) >= 0.15]
            caps = [i for i in ok if cur[i + 1][2].strip()[:1].isupper()]
            k = (
                commas[-1]
                if commas
                else max(pauses, key=gap_after)
                if pauses
                else caps[-1]
                if caps
                else len(cur) - 1
            )
            emit(cur[: k + 1])
            cur = cur[k + 1 :]
        cur.append(w)
        end = w[2].strip()
        if end.endswith((".", "?", "!", "。", "？", "！")) and not end.endswith(("...", "…")):
            emit(cur)
            cur = []
    if cur:
        emit(cur)
    return fold_short_cues([c for c in cues if c.text])


def decode_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any audio/video file to mono int16 PCM via ffmpeg into a single allocation."""
    with tempfile.TemporaryFile() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
            stdout=tmp,
            stderr=subprocess.PIPE,
            check=True,
        )
        tmp.seek(0)
        return np.fromfile(tmp, dtype=np.int16)

def remote(
    pcm: np.ndarray,
    host: str,
    language: str | None = None,
    connect_s: int = 3,
    remote_bin: str = "siftasr",
) -> AsrResult | None:
    """Transcribe on another machine (a GPU box) over ssh: audio goes as FLAC on stdin,
    `siftasr asr -` (or `mediasift asr -` fallback) runs there, cues + voices come back as JSON."""
    if host != "local":
        ok = (
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_s}", host, "true"],
                capture_output=True,
            ).returncode
            == 0
        )
        if not ok:
            _log(f"{host} did not answer in {connect_s}s: transcribing here")
            return None
    flac = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-c:a", "flac", "-f", "flac", "-"],
        input=pcm.astype(np.int16).tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    lang_flag = f" --lang {language}" if language else ""
    remote_cmd = f"(command -v {remote_bin} >/dev/null && {remote_bin} asr -{lang_flag}) || mediasift asr -{lang_flag}"
    cmd = (
        [remote_bin, "asr", "-"] + (["--lang", language] if language else [])
        if host == "local"
        else [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_s}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            host,
            f"bash -lc '{remote_cmd}'",
        ]
    )
    t0 = time.perf_counter()
    r = subprocess.run(cmd, input=flac, capture_output=True, timeout=120 + len(pcm) / SR / 10)
    if r.returncode != 0:
        raise RuntimeError(f"ASR on {host} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-1500:]}")
    out_lines = r.stdout.decode(errors="replace").strip().splitlines()
    if not out_lines:
        raise RuntimeError(f"ASR on {host} printed nothing: {r.stderr.decode(errors='replace')[-800:]}")
    js = json.loads(out_lines[-1])
    cues = [Cue(float(a), float(b), t) for a, b, t in js["cues"]]
    remote_voices = js.get("voices")
    if remote_voices and len(remote_voices) == len(cues):
        for c, spk in zip(cues, remote_voices):
            c.speaker = spk
    dt = time.perf_counter() - t0
    _log(f"ASR on {host}: {js['backend']} ({js['model']}), {len(pcm) / SR / max(dt, 1e-6):.1f}x realtime incl. transfer")
    res = AsrResult(cues, f"{js['backend']}@{host}", js["model"], js.get("language"), round(dt, 1))
    res.voices = remote_voices if remote_voices and len(remote_voices) == len(cues) else None
    return res


def serve_stdin(language: str | None = None) -> None:
    """`siftasr asr -`: audio in any ffmpeg-readable format on stdin, cues + voices as JSON on stdout."""
    data = sys.stdin.buffer.read()
    pcm = np.frombuffer(
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
            input=data,
            capture_output=True,
            check=True,
        ).stdout,
        np.int16,
    )
    r = transcribe(pcm, "auto", None, language, diarize=True)
    print(
        json.dumps(
            {
                "cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in r.cues],
                "voices": r.voices,
                "backend": r.backend,
                "model": r.model,
                "language": r.language,
                "seconds": r.seconds,
            },
            ensure_ascii=False,
        )
    )

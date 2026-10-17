"""Speaker attribution ("who is talking"): TitaNet-small embeddings per turn or cue,
clustered into 'A', 'B', ... in order of first appearance, plus ElevenLabs Scribe-style
word- and turn-level speaker splitting when speakers alternate across pauses.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np

SR = 16000
SPEAKER = "nemo_en_titanet_small.onnx"
SPEAKER_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{SPEAKER}"
SPEAKER_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"


def _log(msg: str) -> None:
    print(f"siftasr: {msg}", file=sys.stderr)


def cache_root() -> Path:
    base = os.environ.get("SIFTASR_CACHE") or os.environ.get("MEDIASIFT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "siftasr")
    return Path(base)


def existing_model_path(rel: str) -> Path:
    """Reuse an already-downloaded model from ~/.cache/mediasift if present, else ~/.cache/siftasr."""
    primary = cache_root() / "models" / rel
    if primary.exists():
        return primary
    legacy = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))) / "mediasift" / "models" / rel
    if legacy.exists():
        return legacy
    return primary


def fetch_verified(url: str, sha256: str, dest: Path, what: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (Path(url).name + ".part")
    _log(f"downloading {what} (once) from {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != sha256:
        tmp.unlink()
        raise RuntimeError(f"{what} download has SHA-256 {digest}, expected {sha256}")
    tmp.replace(dest)
    return dest


def speaker_model() -> Path:
    f = existing_model_path(SPEAKER)
    return f if f.exists() else fetch_verified(SPEAKER_URL, SPEAKER_SHA256, f, "the speaker model (40 MB)")


def _worker_threads(default: int = 8) -> int:
    if os.environ.get("SIFTASR_POLITE") == "1" or (
        os.environ.get("MEDIASIFT_FULL_SPEED") != "1" and os.environ.get("SIFTASR_FULL_SPEED") != "1"
    ):
        return min(default, 4)
    return default


@lru_cache(maxsize=1)
def _extractor(model_path: str):
    import sherpa_onnx
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=1)
    )


def embed_spans(
    pcm: np.ndarray,
    spans: list[tuple[float, float]],
    min_s: float = 0.8,
    max_workers: int | None = None,
    max_s: float = 5.0,
) -> list[np.ndarray | None]:
    """Extract L2-normalized TitaNet-small speaker embeddings for each (start, end) span."""
    if not spans:
        return []
    ex = _extractor(str(speaker_model()))
    min_samples = int(min_s * SR)
    max_samples = int(max_s * SR) if max_s > 0 else 0

    def embed(span: tuple[float, float]) -> np.ndarray | None:
        i0, i1 = max(0, int(span[0] * SR)), min(len(pcm), int(span[1] * SR))
        if i1 - i0 < min_samples:
            return None
        if max_samples and (i1 - i0) > max_samples:
            mid = (i0 + i1) // 2
            half = max_samples // 2
            i0, i1 = mid - half, mid - half + max_samples
        seg = pcm[i0:i1]
        st = ex.create_stream()
        wf = seg.astype(np.float32) * (1.0 / 32768.0) if seg.dtype.kind == "i" else seg.astype(np.float32)
        st.accept_waveform(SR, wf)
        st.input_finished()
        v = np.array(ex.compute(st), np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    workers = max_workers or _worker_threads(min(4, os.cpu_count() or 4))
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(embed, spans))

def cluster_embeddings(
    embs: list[np.ndarray | None],
    thr: float = 0.6,
    spans: list[tuple[float, float]] | None = None,
) -> list[str | None]:
    """Cluster L2-normalized embeddings into 'A', 'B', ... in order of first appearance.
    When `spans` are provided, only substantial turns (>= 2.5 s) seed new speaker centroids,
    splinter clusters (< 8 s total) fold into their nearest major speaker, and sub-0.8 s
    backchannels inherit continuity from adjacent turns.
    Returns [None, ...] if fewer than 2 distinct speakers are heard."""
    n = len(embs)
    valid_idx = [i for i, v in enumerate(embs) if v is not None]
    if len(valid_idx) < 2:
        return [None] * n

    durs = [spans[i][1] - spans[i][0] for i in range(n)] if spans is not None else [3.0] * n
    anchor_set = {i for i in valid_idx if durs[i] >= 2.5} if spans is not None else set(valid_idx)
    if len(anchor_set) < 2:
        anchor_set = set(valid_idx)

    sums: list[np.ndarray] = []
    out: list[int | None] = [None] * n

    # Pass 1: cluster anchor turns in chronological order to establish clean speaker centroids.
    for i in valid_idx:
        if i not in anchor_set:
            continue
        v = embs[i]
        assert v is not None
        d = [1.0 - float(v @ (c / np.linalg.norm(c))) for c in sums]
        j = int(np.argmin(d)) if d else -1
        if j >= 0 and d[j] < thr:
            sums[j] = sums[j] + v
        else:
            sums.append(v.copy())
            j = len(sums) - 1
        out[i] = j
    # Consolidate anchor centroids whose accumulated means converged within thr * 0.85
    # (e.g. when a noisy opening utterance seeded cluster 0 before the main centroid stabilized).
    if len(sums) > 1:
        a_counts = [out.count(k) for k in range(len(sums))]
        order_k = sorted(range(len(sums)), key=lambda k: -a_counts[k])
        remap: dict[int, int] = {}
        kept: list[int] = []
        for k in order_k:
            nk = sums[k] / np.linalg.norm(sums[k])
            merged_to = None
            for m in kept:
                nm = sums[m] / np.linalg.norm(sums[m])
                if 1.0 - float(nk @ nm) < thr * 0.85:
                    merged_to = m
                    break
            if merged_to is not None:
                sums[merged_to] = sums[merged_to] + sums[k]
                remap[k] = merged_to
            else:
                remap[k] = k
                kept.append(k)
        if len(kept) < len(sums):
            old_to_new = {m: idx for idx, m in enumerate(sorted(kept))}
            sums = [sums[m] for m in sorted(kept)]
            for i in anchor_set:
                if out[i] is not None:
                    out[i] = old_to_new[remap[out[i]]]

    # Pass 2: assign short valid spans (< 2.5 s) to the closest established speaker centroid,
    # with a small continuity margin (0.15) toward an adjacent anchor turn within 1.5 s.
    norm_sums = [c / np.linalg.norm(c) for c in sums]
    for i in valid_idx:
        if i in anchor_set:
            continue
        v = embs[i]
        assert v is not None
        d = [1.0 - float(v @ c) for c in norm_sums]
        best = int(np.argmin(d))
        if spans is not None:
            for adj in (i - 1, i + 1):
                if 0 <= adj < n and out[adj] is not None:
                    gap = spans[i][0] - spans[adj][1] if adj < i else spans[adj][0] - spans[i][1]
                    k_adj = out[adj]
                    if gap <= 1.5 and k_adj is not None and d[k_adj] - d[best] <= 0.15:
                        best = k_adj
                        break
        out[i] = best

    # Fold singleton / splinter clusters on long recordings (>= 20 valid spans).
    if len(valid_idx) >= 20 and len(sums) > 1:
        counts = [out.count(k) for k in range(len(sums))]
        if spans is not None:
            min_cues = max(3, n // 120)
            spk_dur = [sum(durs[i] for i in range(n) if out[i] == k) for k in range(len(sums))]
            major = [k for k in range(len(sums)) if counts[k] >= min_cues and spk_dur[k] >= 8.0]
            if not major:
                major = [int(np.argmax(counts))]
            for i in valid_idx:
                k = out[i]
                v = embs[i]
                if k is not None and v is not None and k not in major:
                    out[i] = max(major, key=lambda m: float(v @ (sums[m] / np.linalg.norm(sums[m]))))
        else:
            major = [k for k, c in enumerate(counts) if c >= 3]
            if major:
                for i, (k, v) in enumerate(zip(out, embs)):
                    if k is not None and v is not None and counts[k] == 1:
                        out[i] = max(major, key=lambda m: float(v @ (sums[m] / np.linalg.norm(sums[m]))))

    if spans is not None and len({k for k in out if k is not None}) > 1:
        # 1. Fill sub-0.8 s backchannel None spans from surrounding conversation context first.
        for i in range(n):
            if out[i] is None:
                prev_k = next((out[j] for j in range(i - 1, -1, -1) if out[j] is not None), None)
                next_k = next((out[j] for j in range(i + 1, n) if out[j] is not None), None)
                if prev_k == next_k and prev_k is not None:
                    out[i] = prev_k
                elif i > 0 and out[i - 1] is not None and (spans[i][0] - spans[i - 1][1]) <= 1.5:
                    out[i] = out[i - 1]
                elif i + 1 < n and out[i + 1] is not None and (spans[i + 1][0] - spans[i][1]) <= 1.5:
                    out[i] = out[i + 1]
                else:
                    out[i] = prev_k if prev_k is not None else next_k

        # 2. Contextual smoothing: 1- or 2-cue short excursions (< 2.5 s total) between X ... X stay X,
        # and a single < 2.2 s non-anchor cue in a 3-way hop (X -> Y -> Z) attaches to its closer neighbor.
        if n >= 3 and durs[0] < 2.5 and out[1] == out[2] and (spans[1][0] - spans[0][1]) <= 1.5:
            out[0] = out[1]
        if n >= 3 and durs[-1] < 2.5 and out[-2] == out[-3] and (spans[-1][0] - spans[-2][1]) <= 1.5:
            out[-1] = out[-2]
        for i in range(1, n - 1):
            if out[i - 1] == out[i + 1] and out[i] != out[i - 1] and durs[i] < 2.5:
                out[i] = out[i - 1]
        for i in range(1, n - 2):
            if (out[i - 1] == out[i + 2] and out[i] != out[i - 1]
                    and durs[i] + durs[i + 1] < 2.5):
                out[i] = out[i + 1] = out[i - 1]
        for i in range(1, n - 1):
            if (out[i] != out[i - 1] and out[i] != out[i + 1] and out[i - 1] != out[i + 1]
                    and durs[i] < 2.2):
                gap_prev = spans[i][0] - spans[i - 1][1]
                gap_next = spans[i + 1][0] - spans[i][1]
                if min(gap_prev, gap_next) <= 1.2:
                    out[i] = out[i - 1] if gap_prev <= gap_next else out[i + 1]
    order: dict[int, str] = {}
    for k in out:
        if k is not None and k not in order:
            order[k] = chr(ord("A") + min(len(order), 25))
    named = [order[k] if k is not None else None for k in out]
    return named if len(order) > 1 else [None] * n


def voices(
    pcm: np.ndarray, spans: list[tuple[float, float]], thr: float = 0.6, max_workers: int | None = None
) -> list[str | None]:
    """Group speech spans by speaker ('A', 'B', ... in order of first appearance)."""
    n = len(spans)
    if n < 20:
        return cluster_embeddings(embed_spans(pcm, spans, max_workers=max_workers), thr=thr, spans=spans)

    durs = [e - s for s, e in spans]
    anchor_idx = [i for i, d in enumerate(durs) if d >= 2.5]
    if len(anchor_idx) < 2:
        return cluster_embeddings(embed_spans(pcm, spans, max_workers=max_workers), thr=thr, spans=spans)

    embs: list[np.ndarray | None] = [None] * n
    anchor_embs = embed_spans(pcm, [spans[i] for i in anchor_idx], max_workers=max_workers)
    for i, v in zip(anchor_idx, anchor_embs):
        embs[i] = v

    # Quick Pass-1 anchor check: if all anchors belong to 1 speaker, skip short spans completely.
    sums: list[np.ndarray] = []
    a_spk: dict[int, int] = {}
    for i in anchor_idx:
        v = embs[i]
        if v is None:
            continue
        d = [1.0 - float(v @ (c / np.linalg.norm(c))) for c in sums]
        j = int(np.argmin(d)) if d else -1
        if j >= 0 and d[j] < thr:
            sums[j] = sums[j] + v
        else:
            sums.append(v.copy())
            j = len(sums) - 1
        a_spk[i] = j
    if len(sums) > 1:
        c_thr = thr * 0.85
        m_sums: list[np.ndarray] = []
        remap: dict[int, int] = {}
        for ci, c_vec in enumerate(sums):
            cn = c_vec / np.linalg.norm(c_vec)
            md = [1.0 - float(cn @ (mc / np.linalg.norm(mc))) for mc in m_sums]
            mj = int(np.argmin(md)) if md else -1
            if mj >= 0 and md[mj] < c_thr:
                m_sums[mj] = m_sums[mj] + c_vec
                remap[ci] = mj
            else:
                remap[ci] = len(m_sums)
                m_sums.append(c_vec)
        if len(m_sums) <= 1:
            return [None] * n
    elif len(sums) <= 1:
        return [None] * n

    # Only embed short spans (0.8 <= dur < 2.5s) near speaker transitions or multi-cue short runs;
    # interior 1-2 cue short spans within the same anchor speaker's turn are resolved by smoothing.
    anchor_set = set(anchor_idx)
    prev_a: list[int | None] = [None] * n
    next_a: list[int | None] = [None] * n
    prev_pos: list[int] = [-1] * n
    next_pos: list[int] = [n] * n
    cur_s, cur_p = None, -1
    for i in range(n):
        if i in a_spk:
            cur_s, cur_p = a_spk[i], i
        prev_a[i], prev_pos[i] = cur_s, cur_p
    cur_s, cur_p = None, n
    for i in range(n - 1, -1, -1):
        if i in a_spk:
            cur_s, cur_p = a_spk[i], i
        next_a[i], next_pos[i] = cur_s, cur_p

    for i in range(n):
        if prev_a[i] is None and next_a[i] is not None and next_pos[i] <= 2:
            prev_a[i], prev_pos[i] = next_a[i], -1
        elif next_a[i] is None and prev_a[i] is not None and (n - 1 - prev_pos[i]) <= 2:
            next_a[i], next_pos[i] = prev_a[i], n

    short_idx = [
        i for i in range(n)
        if i not in anchor_set and durs[i] >= 0.8
        and (prev_a[i] != next_a[i] or prev_a[i] is None or (next_pos[i] - prev_pos[i]) > 3)
    ]
    if short_idx:
        for i, v in zip(short_idx, embed_spans(pcm, [spans[i] for i in short_idx], max_workers=max_workers)):
            embs[i] = v
    return cluster_embeddings(embs, thr=thr, spans=spans)

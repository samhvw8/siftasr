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


def embed_spans(
    pcm: np.ndarray, spans: list[tuple[float, float]], min_s: float = 0.8, max_workers: int | None = None
) -> list[np.ndarray | None]:
    """Extract L2-normalized TitaNet-small speaker embeddings for each (start, end) span."""
    if not spans:
        return []
    import sherpa_onnx
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(speaker_model()), num_threads=1)
    )

    def embed(span: tuple[float, float]) -> np.ndarray | None:
        seg = pcm[int(span[0] * SR):int(span[1] * SR)]
        if len(seg) < min_s * SR:
            return None
        st = ex.create_stream()
        wf = seg.astype(np.float32) / 32768.0 if seg.dtype.kind == "i" else seg.astype(np.float32)
        st.accept_waveform(SR, wf)
        st.input_finished()
        v = np.array(ex.compute(st), np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    workers = max_workers or _worker_threads(min(16, os.cpu_count() or 4))
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(embed, spans))


def cluster_embeddings(embs: list[np.ndarray | None], thr: float = 0.6) -> list[str | None]:
    """Cluster L2-normalized embeddings into 'A', 'B', ... in order of first appearance.
    Returns [None, ...] if fewer than 2 distinct speakers are heard."""
    n = len(embs)
    if sum(v is not None for v in embs) < 2:
        return [None] * n
    sums: list[np.ndarray] = []
    out: list[int | None] = []
    for v in embs:
        if v is None:
            out.append(None)
            continue
        d = [1.0 - float(v @ (c / np.linalg.norm(c))) for c in sums]
        j = int(np.argmin(d)) if d else -1
        if j >= 0 and d[j] < thr:
            sums[j] = sums[j] + v
        else:
            sums.append(v.copy())
            j = len(sums) - 1
        out.append(j)
    counts = [out.count(k) for k in range(len(sums))]
    if sum(counts) >= 20:
        major = [k for k, c in enumerate(counts) if c >= 3]
        if major:
            for i, (k, v) in enumerate(zip(out, embs)):
                if k is not None and v is not None and counts[k] == 1:
                    out[i] = max(major, key=lambda m: float(v @ (sums[m] / np.linalg.norm(sums[m]))))
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
    return cluster_embeddings(embed_spans(pcm, spans, max_workers=max_workers), thr=thr)

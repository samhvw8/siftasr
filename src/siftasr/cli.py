"""CLI entrypoint for siftasr: transcribe local files or serve stdin for remote SSH offload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import asr


def _fmt_ts(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["asr"] and args[1:2] == ["-"]:
        p = argparse.ArgumentParser(prog="siftasr asr -")
        p.add_argument("dash")
        p.add_argument("--lang", default=None)
        ns = p.parse_args(args[1:])
        asr.serve_stdin(ns.lang)
        return

    p = argparse.ArgumentParser(
        prog="siftasr",
        description="Language-routed speech-to-text (en/vi/zh/whisper) with speaker attribution.",
    )
    p.add_argument("src", help="audio/video file path, or '-' for stdin server mode")
    p.add_argument("-l", "--lang", default=None, help="ISO 639-1 source language hint (auto-detected if omitted)")
    p.add_argument("-b", "--backend", default="auto",
                   help="backend: auto, parakeet-mlx, quan-coreml, qwen3-mlx, mlx-whisper, faster-whisper")
    p.add_argument("-m", "--model", default=None, help="override model repo/name")
    p.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=True,
                   help="attribute speakers ('A', 'B', ...) per turn and word (default: on)")
    p.add_argument("--on", metavar="HOST", default=None,
                   help="offload transcription + diarization to HOST over ssh (falls back to local)")
    p.add_argument("--json", action="store_true", help="print full JSON output (cues, segments, speakers, words)")
    a = p.parse_args(args)

    if a.src == "-":
        asr.serve_stdin(a.lang)
        return

    pcm = asr.decode_audio(Path(a.src).expanduser())
    res = asr.remote(pcm, a.on, a.lang) if a.on else None
    if res is None:
        res = asr.transcribe(pcm, backend=a.backend, model=a.model, language=a.lang, diarize=a.diarize)

    if a.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return

    multi = res.voices and any(res.voices)
    for c in res.cues:
        spk = f"[{c.speaker}] " if (multi and c.speaker) else ""
        print(f"{_fmt_ts(c.start)} - {_fmt_ts(c.end)}  {spk}{c.text}")


if __name__ == "__main__":
    main()

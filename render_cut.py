#!/usr/bin/env python3
"""Sync-safe, fast video render for keep-segments.

trim+concat keeps audio/video perfectly in sync, but one giant filtergraph with
hundreds of branches (a pause-heavy clip can have 300+ cuts) crawls. So we encode
in CHUNKS of ~30 whole segments, then stream-copy concat the chunk files — bounded
branch count per pass, zero re-encode on the join, no A/V drift.
"""

import os
import subprocess
import tempfile

CHUNK = 30


def _filtergraph(segments):
    parts, cc = [], ""
    for i, s in enumerate(segments):
        parts.append(f"[0:v]trim=start={s['start']:.4f}:end={s['end']:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s['start']:.4f}:end={s['end']:.4f},asetpts=PTS-STARTPTS[a{i}]")
        cc += f"[v{i}][a{i}]"
    parts.append(f"{cc}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def _venc(fast):
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if fast
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"])


def _encode_one(ffmpeg, src, segments, out, fast):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src,
           "-filter_complex", _filtergraph(segments),
           "-map", "[outv]", "-map", "[outa]", *_venc(fast),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-700:])


def render_segments(ffmpeg, src, segments, out_path, fast=True, on_chunk=None):
    """Render keep `segments` of `src` to `out_path`. on_chunk(done,total) is an
    optional progress callback."""
    out_path = str(out_path)
    segments = [s for s in segments if s["end"] - s["start"] > 1e-3]
    if not segments:
        raise RuntimeError("no segments to render")

    if len(segments) <= CHUNK:
        _encode_one(ffmpeg, src, segments, out_path, fast)
        if on_chunk:
            on_chunk(1, 1)
        return

    chunks = [segments[k:k + CHUNK] for k in range(0, len(segments), CHUNK)]
    with tempfile.TemporaryDirectory(prefix="vidcut-") as td:
        parts = []
        for idx, ch in enumerate(chunks):
            p = os.path.join(td, f"p{idx:04d}.mp4")
            _encode_one(ffmpeg, src, ch, p, fast)
            parts.append(p)
            if on_chunk:
                on_chunk(idx + 1, len(chunks))
        listf = os.path.join(td, "list.txt")
        with open(listf, "w") as fh:
            for p in parts:
                fh.write(f"file '{p}'\n")
        # Same encoder/params across chunks -> stream-copy concat is instant + lossless.
        r = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", listf, "-c", "copy", "-movflags", "+faststart", out_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            # Fallback: re-encode the join if copy refuses (rare).
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", listf, *_venc(fast), "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", out_path],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-700:])

#!/usr/bin/env python3
"""Sync-safe, fast video render for keep-segments.

trim+concat keeps audio/video in sync, but one giant filtergraph with
hundreds of branches (a pause-heavy clip can have 300+ cuts) crawls. So we
encode in CHUNKS of ~30 whole segments, then stream-copy concat the chunk
files — bounded branch count per pass, zero re-encode on the join.

Two sync rules, measured with a beep+flash test clip:
- Cut points MUST sit on the source frame grid. trim keeps whole frames
  (first frame >= start) while atrim is sample-accurate, so an off-grid cut
  gives every segment audio that starts up to a frame before its video and
  the edit plays video-early by ~half a frame per cut generation. We snap
  starts/ends to the probed frame grid and cut both streams there.
- Chunk files carry PCM audio, not AAC. Stream-copy concat of AAC splices
  each chunk's encoder priming/padding (~20-40ms of audio) into the middle
  of the timeline — video-early drift that grows with every chunk. PCM
  concat is sample-exact; audio is AAC-encoded once at the final remux.
"""

import os
import subprocess
import tempfile

CHUNK = 30
AAC = ["-c:a", "aac", "-b:a", "192k"]


def _probe_grid(ffmpeg, src):
    """(fps, first_video_pts) of src, or (None, 0.0) if unprobeable."""
    d = os.path.dirname(ffmpeg)
    ffprobe = os.path.join(d, "ffprobe") if d else "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,start_time", "-of", "csv=p=0", src],
            capture_output=True, text=True)
        rate, _, start = r.stdout.strip().partition(",")
        num, den = rate.split("/")
        fps = float(num) / float(den)
        t0 = float(start) if start and start != "N/A" else 0.0
        return (fps, t0) if fps > 0 else (None, 0.0)
    except Exception:
        return None, 0.0


def _snap(segments, fps, t0):
    """Quantise cut points to the frame grid so video and audio cut identically.
    vstart/vend sit half a frame EARLY so trim's first kept frame is exactly the
    grid frame despite float pts jitter; astart/aend are the exact grid times.
    Both branches then span exactly (ke-ks) frames of the same content."""
    snapped = []
    for s in segments:
        ks = round((s["start"] - t0) * fps)
        ke = round((s["end"] - t0) * fps)
        if ke > ks:
            snapped.append({"vstart": t0 + (ks - 0.5) / fps,
                            "vend": t0 + (ke - 0.5) / fps,
                            "astart": t0 + ks / fps,
                            "aend": t0 + ke / fps})
    return snapped


def _filtergraph(segments):
    parts, cc = [], ""
    for i, s in enumerate(segments):
        parts.append(f"[0:v]trim=start={s['vstart']:.4f}:end={s['vend']:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s['astart']:.4f}:end={s['aend']:.4f},asetpts=PTS-STARTPTS[a{i}]")
        cc += f"[v{i}][a{i}]"
    parts.append(f"{cc}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def _venc(fast):
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if fast
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"])


def _encode_one(ffmpeg, src, segments, out, fast, aenc=AAC):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src,
           "-filter_complex", _filtergraph(segments),
           "-map", "[outv]", "-map", "[outa]", *_venc(fast), *aenc,
           "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-700:])


def render_segments(ffmpeg, src, segments, out_path, fast=True, on_chunk=None):
    """Render keep `segments` of `src` to `out_path`. on_chunk(done,total) is an
    optional progress callback."""
    out_path = str(out_path)
    segments = [s for s in segments if s["end"] - s["start"] > 1e-3]
    fps, t0 = _probe_grid(ffmpeg, src)
    if fps:
        segments = _snap(segments, fps, t0)
    else:  # unprobeable: cut both streams at the raw floats (old behaviour)
        segments = [{"vstart": s["start"], "vend": s["end"],
                     "astart": s["start"], "aend": s["end"]} for s in segments]
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
            p = os.path.join(td, f"p{idx:04d}.mov")
            _encode_one(ffmpeg, src, ch, p, fast, aenc=["-c:a", "pcm_s16le"])
            parts.append(p)
            if on_chunk:
                on_chunk(idx + 1, len(chunks))
        listf = os.path.join(td, "list.txt")
        with open(listf, "w") as fh:
            for p in parts:
                fh.write(f"file '{p}'\n")
        # Same encoder/params across chunks -> stream-copy concat is lossless,
        # and PCM audio joins sample-exact. One AAC encode at the remux.
        joined = os.path.join(td, "joined.mov")
        r = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", listf, "-c", "copy", joined], capture_output=True, text=True)
        if r.returncode == 0:
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", joined,
                 "-c:v", "copy", *AAC, "-movflags", "+faststart", out_path],
                capture_output=True, text=True)
        if r.returncode != 0:
            # Fallback: re-encode the join if copy refuses (rare).
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", listf, *_venc(fast), *AAC,
                 "-movflags", "+faststart", out_path],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-700:])

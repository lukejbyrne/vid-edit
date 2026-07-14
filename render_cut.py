#!/usr/bin/env python3
"""Sync-safe, fast video render for keep-segments.

trim+concat keeps audio/video in sync, but one giant filtergraph with hundreds of
branches (a pause-heavy clip can have 300+ cuts) crawls. So we encode in CHUNKS of
~30 whole segments, then concat the chunk files — bounded branch count per pass.

Keeping A/V sync across the chunk join (this is subtle — a naive copy-concat drifts):
  * VIDEO is forced to a constant frame rate. Every chunk then shares an identical
    timebase, so copy-concat across boundaries stays frame-accurate even for VFR
    sources (macOS screen recordings are variable frame rate).
  * AUDIO in each chunk is written as PCM (no codec priming/padding), and AAC is
    encoded ONCE on the final join. Separately-AAC-encoding each chunk and stream-
    copy-concatenating them re-inserts ~20-40ms of encoder-delay priming at every
    boundary, which accumulates into visible lip-sync drift; PCM chunks avoid it.
  * The finished file is VERIFIED before it is accepted: audio duration ~= video
    duration, and total duration ~= the sum of kept segments. A truncated write
    (e.g. a flaky external drive stalling mid-encode) is raised as an error instead
    of being silently presented as a good render.
  * The render goes to a temp file and is atomically moved into place only after it
    verifies, so an interrupted run never leaves a half-written output behind.
"""

import os
import subprocess
import tempfile

CHUNK = 30
_SYNC_TOL = 0.20          # max allowed |audio_dur - video_dur|, seconds
_SHORTFALL_ABS = 3.0      # flag truncation if output is shorter than expected by ...
_SHORTFALL_REL = 0.10     # ... this many seconds, or this fraction, whichever larger


def _ffprobe_for(ffmpeg):
    """The ffprobe binary that sits next to `ffmpeg`, else whatever is on PATH."""
    if ffmpeg and os.path.dirname(ffmpeg):
        cand = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
        if os.path.exists(cand):
            return cand
    return "ffprobe"


def _probe_fps(ffprobe, src):
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=nk=1:nw=1", src],
        capture_output=True, text=True)
    try:
        num, den = r.stdout.strip().split("/")
        fps = float(num) / float(den)
        return fps if 1.0 < fps < 240.0 else 30.0
    except Exception:
        return 30.0


def _stream_dur(ffprobe, path, kind):
    """Duration of the first `kind` ("v"/"a") stream, falling back to container."""
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", f"{kind}:0",
         "-show_entries", "stream=duration", "-of", "default=nk=1:nw=1", path],
        capture_output=True, text=True)
    try:
        d = float(r.stdout.strip())
        if d > 0:
            return d
    except Exception:
        pass
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _verify(ffprobe, path, expected):
    vd = _stream_dur(ffprobe, path, "v")
    ad = _stream_dur(ffprobe, path, "a")
    if vd <= 0 or ad <= 0:
        raise RuntimeError(f"render produced a bad file (video={vd:.2f}s audio={ad:.2f}s)")
    if abs(vd - ad) > _SYNC_TOL:
        raise RuntimeError(
            f"render out of A/V sync: video {vd:.2f}s vs audio {ad:.2f}s "
            f"(gap {ad - vd:+.2f}s > {_SYNC_TOL}s)")
    # Truncation (e.g. a flaky drive stalling mid-write) is always a SHORTFALL, so
    # only guard the short side: quantizing many short segments to whole video
    # frames can legitimately make the output slightly LONGER than the float sum.
    shortfall = expected - vd
    if shortfall > max(_SHORTFALL_ABS, _SHORTFALL_REL * expected):
        raise RuntimeError(
            f"render truncated: got {vd:.2f}s but expected ~{expected:.2f}s of kept "
            f"content ({shortfall:.2f}s short)")


def _filtergraph(segments, fps):
    parts, cc = [], ""
    for i, s in enumerate(segments):
        parts.append(f"[0:v]trim=start={s['start']:.4f}:end={s['end']:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s['start']:.4f}:end={s['end']:.4f},asetpts=PTS-STARTPTS[a{i}]")
        cc += f"[v{i}][a{i}]"
    parts.append(f"{cc}concat=n={len(segments)}:v=1:a=1[cv][outa];[cv]fps={fps:.6f}[outv]")
    return ";".join(parts)


def _venc(fast):
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if fast
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"])


def _encode_one(ffmpeg, src, segments, out, fast, fps, audio):
    """Encode one pass. `audio`="pcm" for a lossless, priming-free chunk part;
    "aac" for a final single-pass output."""
    if audio == "pcm":
        aopts, movflags = ["-c:a", "pcm_s16le"], []
    else:
        aopts, movflags = ["-c:a", "aac", "-b:a", "192k"], ["-movflags", "+faststart"]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src,
           "-filter_complex", _filtergraph(segments, fps),
           "-map", "[outv]", "-map", "[outa]", *_venc(fast),
           *aopts, *movflags, out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-700:])


def render_segments(ffmpeg, src, segments, out_path, fast=True, on_chunk=None):
    """Render keep `segments` of `src` to `out_path`. on_chunk(done,total) is an
    optional progress callback. Raises if the output fails A/V-sync / length checks."""
    out_path = str(out_path)
    segments = [s for s in segments if s["end"] - s["start"] > 1e-3]
    if not segments:
        raise RuntimeError("no segments to render")

    ffprobe = _ffprobe_for(ffmpeg)
    fps = _probe_fps(ffprobe, src)
    expected = sum(s["end"] - s["start"] for s in segments)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_out = tempfile.mkstemp(prefix=".vidcut-", suffix=".mp4", dir=out_dir)
    os.close(fd)
    try:
        if len(segments) <= CHUNK:
            _encode_one(ffmpeg, src, segments, tmp_out, fast, fps, "aac")
            if on_chunk:
                on_chunk(1, 1)
        else:
            chunks = [segments[k:k + CHUNK] for k in range(0, len(segments), CHUNK)]
            with tempfile.TemporaryDirectory(prefix="vidcut-") as td:
                parts = []
                for idx, ch in enumerate(chunks):
                    # PCM audio + matroska so the join carries no per-chunk AAC priming.
                    p = os.path.join(td, f"p{idx:04d}.mkv")
                    _encode_one(ffmpeg, src, ch, p, fast, fps, "pcm")
                    parts.append(p)
                    if on_chunk:
                        on_chunk(idx + 1, len(chunks))
                listf = os.path.join(td, "list.txt")
                with open(listf, "w") as fh:
                    for p in parts:
                        fh.write(f"file '{p}'\n")
                # Identical video params + CFR across chunks -> copy video losslessly;
                # PCM chunk audio is gapless -> AAC-encode once, no per-join drift.
                r = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                     "-i", listf, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-movflags", "+faststart", tmp_out],
                    capture_output=True, text=True)
                if r.returncode != 0:
                    # Fallback: fully re-encode the join if video copy refuses (rare).
                    r = subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                         "-i", listf, *_venc(fast), "-c:a", "aac", "-b:a", "192k",
                         "-movflags", "+faststart", tmp_out],
                        capture_output=True, text=True)
                    if r.returncode != 0:
                        raise RuntimeError(r.stderr[-700:])
        _verify(ffprobe, tmp_out, expected)
        os.replace(tmp_out, out_path)
    finally:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass

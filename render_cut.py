#!/usr/bin/env python3
"""Sync-safe, fast video render for keep-segments.

trim+concat keeps audio/video in sync, but one giant filtergraph with hundreds of
branches (a pause-heavy clip can have 300+ cuts) crawls. So we encode in CHUNKS of
~30 whole segments, then concat the chunk files — bounded branch count per pass.

Four rules keep the join sample-exact (a naive copy-concat drifts):

1. Cut points MUST sit on the source frame grid. trim keeps whole frames (first
   frame >= start) while atrim is sample-accurate, so an off-grid cut gives every
   segment audio that starts up to a frame before its video, and the edit plays
   video-early by ~half a frame per cut. We snap starts/ends to the probed frame
   grid and cut both streams there. (Measured with a beep+flash test clip.)

2. Chunk files carry PCM audio, not AAC. Stream-copy concat of AAC splices each
   chunk's encoder priming/padding (~20-40ms of audio) into the middle of the
   timeline — drift that grows with every chunk. PCM concat is sample-exact; AAC
   is encoded once at the final remux.

3. Video is forced to a CONSTANT frame rate, so every chunk shares one timebase.
   Without this a VFR source (macOS screen recordings are variable frame rate)
   desyncs at each copy-concat boundary — by seconds, not milliseconds.

4. The finished file is VERIFIED before it is accepted: audio duration ~= video
   duration, and the output is not short of the kept-segment total. The render
   goes to a temp file and is atomically moved into place only after it verifies,
   so a stalled write (flaky external drive) raises instead of silently shipping a
   truncated cut or leaving a half-written file that looks done.
"""

import os
import subprocess
import tempfile

CHUNK = 30
AAC = ["-c:a", "aac", "-b:a", "192k"]

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


def _probe_grid(ffprobe, src):
    """(fps, first_video_pts) of src, or (None, 0.0) if unprobeable."""
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,start_time", "-of", "csv=p=0", src],
            capture_output=True, text=True)
        rate, _, start = r.stdout.strip().partition(",")
        num, den = rate.split("/")
        fps = float(num) / float(den)
        t0 = float(start) if start and start != "N/A" else 0.0
        return (fps, t0) if 0.0 < fps < 240.0 else (None, 0.0)
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
        parts.append(f"[0:v]trim=start={s['vstart']:.4f}:end={s['vend']:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s['astart']:.4f}:end={s['aend']:.4f},asetpts=PTS-STARTPTS[a{i}]")
        cc += f"[v{i}][a{i}]"
    n = len(segments)
    if fps:  # force CFR so chunks share one timebase and copy-concat cleanly
        parts.append(f"{cc}concat=n={n}:v=1:a=1[cv][outa];[cv]fps={fps:.6f}[outv]")
    else:
        parts.append(f"{cc}concat=n={n}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def _venc(fast):
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if fast
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"])


def _encode_one(ffmpeg, src, segments, out, fast, fps, audio):
    """One encode pass. `audio`="pcm" for a lossless, priming-free chunk part;
    "aac" for a final single-pass output."""
    if audio == "pcm":
        aopts, movflags = ["-c:a", "pcm_s16le"], []
    else:
        aopts, movflags = list(AAC), ["-movflags", "+faststart"]
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
    expected = sum(s["end"] - s["start"] for s in segments)
    fps, t0 = _probe_grid(ffprobe, src)
    if fps:
        segments = _snap(segments, fps, t0)
    else:  # unprobeable: cut both streams at the raw floats, no CFR pass
        segments = [{"vstart": s["start"], "vend": s["end"],
                     "astart": s["start"], "aend": s["end"]} for s in segments]
    if not segments:
        raise RuntimeError("no segments to render")

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
                    p = os.path.join(td, f"p{idx:04d}.mov")
                    _encode_one(ffmpeg, src, ch, p, fast, fps, "pcm")
                    parts.append(p)
                    if on_chunk:
                        on_chunk(idx + 1, len(chunks))
                listf = os.path.join(td, "list.txt")
                with open(listf, "w") as fh:
                    for p in parts:
                        fh.write(f"file '{p}'\n")
                # Identical CFR video params across chunks -> copy-concat is lossless,
                # and PCM audio joins sample-exact. One AAC encode at the remux.
                joined = os.path.join(td, "joined.mov")
                r = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                     "-i", listf, "-c", "copy", joined], capture_output=True, text=True)
                if r.returncode == 0:
                    r = subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error", "-i", joined,
                         "-c:v", "copy", *AAC, "-movflags", "+faststart", tmp_out],
                        capture_output=True, text=True)
                if r.returncode != 0:
                    # Fallback: fully re-encode the join if copy refuses (rare).
                    r = subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                         "-i", listf, *_venc(fast), *AAC,
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

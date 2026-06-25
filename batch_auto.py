#!/usr/bin/env python3
"""Headless auto silence-removal over whole folders.

Runs the same speech-gated detector as the review UI (Whisper words x voice-band
energy gate), then renders each clip's kept segments to <folder>/no-silence/.
No per-file clicking. Outputs land beside the source; originals are untouched.
Resumable: skips files whose output already exists.

Usage:
    python3 batch_auto.py "/path/folderA" "/path/folderB" [--high]

--high uses the slower high-quality encode (libx264 medium crf18); default is
the fast pass (veryfast crf20), which is plenty for a first review.
"""

import subprocess
import sys
from pathlib import Path

import speech_detect
import calibration
import server  # reuse transcribe_words_local + get_video_tools (importing does not start the server)

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def build_filtergraph(segments):
    parts, concat = [], ""
    for i, s in enumerate(segments):
        a, b = s["start"], s["end"]
        parts.append(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{i}]")
        concat += f"[v{i}][a{i}]"
    parts.append(f"{concat}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def render(src, segments, out_path, ffmpeg, high=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"] if high \
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-filter_complex", build_filtergraph(segments),
        "-map", "[outv]", "-map", "[outa]",
        *venc, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])


def process_file(path, channel, high=False):
    ffmpeg, _ = server.get_video_tools()
    out = path.parent / "no-silence" / (path.stem + ".mp4")
    if out.exists():
        print(f"  skip (exists): {path.name}", flush=True)
        return
    dur = server.get_duration(str(path))
    try:
        words = server.transcribe_words_local(str(path))
    except Exception as e:
        words = []
        print(f"  ! transcription failed ({e}); energy gate only", flush=True)
    analysis = speech_detect.analyze_audio(str(path), ffmpeg=ffmpeg)
    profile = calibration.load_profile(channel)
    keep, cuts, meta = speech_detect.detect_segments(analysis, words, profile)
    kept = sum(s["end"] - s["start"] for s in keep)
    print(f"  {path.name}: {dur:.0f}s -> {kept:.0f}s kept ({len(cuts)} cuts, "
          f"{meta.get('hallucinationCount', 0)} ghosts dropped)"
          + ("  [no speech -> kept whole]" if meta.get("fallbackKeptAll") else ""), flush=True)
    render(path, keep, out, ffmpeg, high=high)
    print(f"    -> {out}", flush=True)


def main(argv):
    high = "--high" in argv
    folders = [a for a in argv if not a.startswith("--")]
    if not folders:
        print("usage: batch_auto.py <folder> [<folder> ...] [--high]")
        return 1
    files = []
    for f in folders:
        fp = Path(f).expanduser()
        if fp.is_file():
            files.append((fp, calibration._safe_channel(fp.parent.name)))
        elif fp.is_dir():
            for v in sorted(fp.glob("*")):
                if v.is_file() and v.suffix.lower() in VIDEO_EXTS:
                    files.append((v, calibration._safe_channel(fp.name)))
    print(f"Auto silence-removal: {len(files)} files, encode={'high' if high else 'fast'}", flush=True)
    done = 0
    for path, channel in files:
        print(f"[{done+1}/{len(files)}] ({channel}) {path.name}", flush=True)
        try:
            process_file(path, channel, high=high)
            done += 1
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
    print(f"Done: {done}/{len(files)} rendered. Outputs in each folder's no-silence/.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

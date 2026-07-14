#!/usr/bin/env python3
"""Auto silence-remover. Triggered by launchd when ~/Downloads/video-raw/ changes.

Processes any new video in video-raw/ and writes the result to video-silenced/.
Uses a file lock so concurrent launchd triggers don't double-process.
"""

import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from remove_silence_cli import detect_silences, keep_segments, trim_concat


def _probe_dur(path):
    fp = shutil.which("ffprobe")
    if not fp:
        return 0.0
    r = subprocess.run(
        [fp, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0

HOME = Path.home()
RAW = HOME / "Downloads" / "video-raw"
OUT = HOME / "Downloads" / "video-silenced"
LOCK = Path("/tmp/silence-watcher.lock")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v"}


def is_stable(path, settle=3):
    try:
        s1 = path.stat().st_size
        time.sleep(settle)
        s2 = path.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False


def process_file(src):
    out = OUT / f"{src.stem} - nosilence{src.suffix}"
    if out.exists():
        return
    if not is_stable(src):
        print(f"skip (still writing): {src.name}", flush=True)
        return
    print(f"processing {src.name}", flush=True)
    silences, total = detect_silences(str(src))
    segs = keep_segments(silences, total)
    print(f"  {total:.1f}s → keeping {sum(s['end']-s['start'] for s in segs):.1f}s "
          f"across {len(segs)} segments", flush=True)
    if not segs:
        return
    tmp = out.with_name(f".{out.stem}.partial{out.suffix}")
    result = trim_concat(str(src), str(tmp), segs)
    if result.returncode != 0:
        print(f"  ffmpeg failed: {result.stderr[-400:]}", flush=True)
        tmp.unlink(missing_ok=True)
        return
    expected = sum(s["end"] - s["start"] for s in segs)
    got = _probe_dur(str(tmp))
    if expected - got > max(3.0, 0.1 * expected):
        print(f"  truncated output ({got:.1f}s vs expected ~{expected:.1f}s); discarding", flush=True)
        tmp.unlink(missing_ok=True)
        return
    tmp.rename(out)
    print(f"  wrote {out.name}", flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another run in progress, exiting", flush=True)
        return

    for f in sorted(RAW.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            try:
                process_file(f)
            except Exception as e:
                print(f"error on {f.name}: {e}", flush=True)


if __name__ == "__main__":
    main()

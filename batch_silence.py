#!/usr/bin/env python3
"""Batch silence remover — mirrors vid-edit server's volume-based trim+concat.

Params per the silence-remover skill CLI fallback:
  threshold -35 dB, min silence 0.5s, padding 250ms, merge gap < 0.1s.
"""
import json
import os
import re
import shutil
import subprocess
import sys

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

THRESHOLD = -35
MIN_DURATION = 0.5
PADDING = 0.25
MERGE_GAP = 0.1


def get_duration(path):
    out = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    out.check_returncode()
    return float(json.loads(out.stdout)["format"]["duration"])


def detect_volume_silences(path, total):
    res = subprocess.run(
        [FFMPEG, "-i", path, "-af",
         f"silencedetect=noise={THRESHOLD}dB:d={MIN_DURATION}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-500:])
    silences, cur = [], None
    for line in res.stderr.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                cur = max(float(m.group(1)), 0.0)
        elif "silence_end" in line:
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m:
                end = min(max(0.0, float(m.group(1))), total)
                start = cur if cur is not None else 0.0
                if end > start:
                    silences.append({"start": start, "end": end})
                cur = None
    if cur is not None:
        silences.append({"start": cur, "end": total})
    return silences


def normalize_segments(segments, total):
    norm = []
    for seg in segments:
        start = max(0.0, min(seg["start"], total))
        end = max(0.0, min(seg["end"], total))
        if end - start < 0.02:
            continue
        if norm and start <= norm[-1]["end"] + 0.001:
            norm[-1]["end"] = max(norm[-1]["end"], end)
        else:
            norm.append({"start": start, "end": end})
    return norm


def merge_segments(segments, total, gap=MERGE_GAP):
    norm = normalize_segments(sorted(segments, key=lambda s: s["start"]), total)
    merged = []
    for seg in norm:
        if merged and seg["start"] - merged[-1]["end"] <= gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    return merged


def volume_keep_segments(silences, total, padding=PADDING):
    segs, cursor = [], 0.0
    for s in silences:
        keep_end = min(s["start"] + padding, total)
        keep_start = max(s["end"] - padding, 0.0)
        if cursor < keep_end:
            segs.append({"start": cursor, "end": keep_end})
        cursor = max(keep_start, cursor)
    if cursor < total:
        segs.append({"start": cursor, "end": total})
    return merge_segments(segs, total)


def process(path, outpath, segments):
    parts, concat = [], ""
    for i, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]
        parts.append(f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]")
        concat += f"[v{i}][a{i}]"
    parts.append(f"{concat}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    fc = ";".join(parts)
    res = subprocess.run(
        [FFMPEG, "-y", "-i", path, "-filter_complex", fc,
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", outpath],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-800:])


def main(files):
    for path in files:
        name = os.path.basename(path)
        base = re.sub(r"\.[^.]+$", "", name)
        outpath = os.path.join(os.path.dirname(path), f"{base}_silence-removed.mp4")
        print(f"\n=== {name} ===", flush=True)
        total = get_duration(path)
        silences = detect_volume_silences(path, total)
        segments = volume_keep_segments(silences, total)
        kept = sum(s["end"] - s["start"] for s in segments)
        print(f"  duration {total:.1f}s -> kept {kept:.1f}s "
              f"({total - kept:.1f}s of silence cut, {len(silences)} silent gaps)", flush=True)
        if not segments:
            print("  no speech segments found; skipping", flush=True)
            continue
        process(path, outpath, segments)
        try:
            out_dur = get_duration(outpath)
        except Exception:
            out_dur = 0.0
        if kept - out_dur > max(3.0, 0.1 * kept):
            print(f"  ERROR: output truncated ({out_dur:.1f}s vs expected ~{kept:.1f}s); "
                  f"removing {os.path.basename(outpath)}", flush=True)
            try:
                os.remove(outpath)
            except OSError:
                pass
            continue
        print(f"  wrote {os.path.basename(outpath)} "
              f"({os.path.getsize(outpath) / 1e6:.0f} MB)", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

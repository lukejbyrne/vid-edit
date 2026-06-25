#!/usr/bin/env python3
"""Final video assembly: silence + retake removal -> stitch clips in order -> intro.

Per clip: keep = (silence-detector keeps) MINUS (auto-resolved retake cuts).
Then render each clip's body (chunked, A/V-synced), normalise every piece to a
common 1080p/fps with audio, and concat: [intro] + clip1 + clip2 + ...

Ambiguous retakes (close calls) are NOT auto-cut — they're returned for a human
pick, with the recency take kept by default in the meantime.
"""

import os
import subprocess
import tempfile

import speech_detect
import retake
import render_cut
import server
import calibration

OUT_W, OUT_H, OUT_FPS = 1920, 1080, 30


def _has_audio(ffprobe, path):
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return "audio" in r.stdout


def compute_final_keeps(clip_path, channel="default", ffmpeg=None):
    """keeps = silence keeps minus auto-resolved retake spans. Returns (keeps, info)."""
    ffmpeg = ffmpeg or server.get_video_tools()[0]
    try:
        words = server.transcribe_words_local(clip_path)
    except Exception:
        words = []
    rk = retake.analyze_retakes(words) if words else {"autoCuts": [], "ambiguous": [], "groups": [], "choices": []}
    analysis = speech_detect.analyze_audio(clip_path, ffmpeg=ffmpeg)
    profile = calibration.load_profile(channel)
    keeps, _cuts, meta = speech_detect.detect_segments(analysis, [], profile)
    final = calibration._subtract(keeps, rk.get("autoCuts", []))
    final = speech_detect._merge_keep(final, profile["min_cut_gap"], analysis["duration"])
    info = {
        "duration": analysis["duration"],
        "silenceRemoved": meta.get("removedDuration", 0),
        "retakeGroups": len(rk.get("groups", [])),
        "retakeCutTime": round(sum(c["end"] - c["start"] for c in rk.get("autoCuts", [])), 1),
        "ambiguous": rk.get("ambiguous", []),
        "keptDuration": round(sum(s["end"] - s["start"] for s in final), 1),
    }
    return final, info


def ensure_audio(ffmpeg, ffprobe, path, tmpdir):
    """Guarantee a (silent if needed) stereo 48k AAC track so concat is uniform."""
    if _has_audio(ffprobe, path):
        return path
    out = os.path.join(tmpdir, "with_audio_" + os.path.basename(path))
    if not out.endswith(".mp4"):
        out += ".mp4"
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out,
    ], check=True)
    return out


def normalize_concat(ffmpeg, ffprobe, inputs, out_path, tmpdir):
    """Concat heterogeneous clips: scale+pad to 1080p, unify fps + audio."""
    inputs = [ensure_audio(ffmpeg, ffprobe, p, tmpdir) for p in inputs]
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", p]
    parts, cc = [], ""
    for i in range(len(inputs)):
        parts.append(
            f"[{i}:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={OUT_FPS},format=yuv420p[v{i}]"
        )
        parts.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
        cc += f"[v{i}][a{i}]"
    parts.append(f"{cc}concat=n={len(inputs)}:v=1:a=1[v][a]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-900:])


def produce(clips_in_order, out_path, channel="default", intro_path=None, log=print):
    """Full pipeline -> writes out_path, returns a report dict."""
    ffmpeg, ffprobe = server.get_video_tools()
    report = {"clips": [], "ambiguous": [], "output": out_path}
    with tempfile.TemporaryDirectory(prefix="assemble-") as td:
        bodies = []
        for idx, clip in enumerate(clips_in_order):
            name = os.path.basename(clip)
            log(f"  [{idx+1}/{len(clips_in_order)}] {name}: analysing…")
            keeps, info = compute_final_keeps(clip, channel=channel, ffmpeg=ffmpeg)
            log(f"      {info['duration']:.0f}s -> {info['keptDuration']:.0f}s kept "
                f"(silence -{info['silenceRemoved']:.0f}s, retakes -{info['retakeCutTime']}s, "
                f"{len(info['ambiguous'])} close calls)")
            body = os.path.join(td, f"body{idx:02d}.mp4")
            render_cut.render_segments(ffmpeg, clip, keeps, body, fast=False)
            bodies.append(body)
            report["clips"].append({"name": name, **{k: v for k, v in info.items() if k != "ambiguous"}})
            for amb in info["ambiguous"]:
                report["ambiguous"].append({"clip": name, **amb})
        pieces = ([intro_path] if intro_path else []) + bodies
        log(f"  stitching {len(pieces)} pieces -> {os.path.basename(out_path)} …")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        normalize_concat(ffmpeg, ffprobe, pieces, out_path, td)
    # final durations
    def dur(stream):
        return float(subprocess.run([ffprobe, "-v", "error", "-select_streams", stream,
                                     "-show_entries", "stream=duration", "-of",
                                     "default=nk=1:nw=1", out_path], capture_output=True,
                                    text=True).stdout.strip() or 0)
    report["finalVideoDur"] = round(dur("v:0"), 2)
    report["finalAudioDur"] = round(dur("a:0"), 2)
    report["avDriftMs"] = round(abs(report["finalVideoDur"] - report["finalAudioDur"]) * 1000)
    return report

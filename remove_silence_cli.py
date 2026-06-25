#!/usr/bin/env python3
"""CLI silence remover using the same segment math as the web UI."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STATIC_FFMPEG_DIR = Path(os.environ.get("VID_EDIT_STATIC_FFMPEG_DIR", "/tmp/vid-edit-static-ffmpeg"))
_VIDEO_TOOLS = None
DEFAULT_MERGE_WIDTH = 1920
DEFAULT_MERGE_HEIGHT = 1080
DEFAULT_MERGE_FPS = 30
DEFAULT_THRESHOLD = -35
DEFAULT_MIN_DURATION = 0.50
DEFAULT_PADDING = 0.20
MERGE_GAP = 0.15
DEFAULT_PADDING_MS = int(DEFAULT_PADDING * 1000)
DEFAULT_SPEECH_MODE = "auto"
DEFAULT_WORD_PRE_PAD = 0.20
DEFAULT_WORD_POST_PAD = 0.25
DEFAULT_MAX_WORD_GAP = 1.00
DEFAULT_FILLER_GAP = 0.20
DEFAULT_REMOVE_FILLERS = True
FILLER_WORDS = {"um", "uh", "uhm", "umm", "erm", "er", "ah"}
DEFAULT_MLX_WHISPER_MODEL = os.environ.get("VID_EDIT_MLX_WHISPER_MODEL", "mlx-community/whisper-tiny")
DEFAULT_WHISPER_MODEL = os.environ.get("VID_EDIT_WHISPER_MODEL", "tiny")
DEFAULT_WHISPER_LANGUAGE = os.environ.get("VID_EDIT_WHISPER_LANGUAGE", "en")


def get_video_tools():
    global _VIDEO_TOOLS
    if _VIDEO_TOOLS is None:
        _VIDEO_TOOLS = resolve_video_tools()
    return _VIDEO_TOOLS


def resolve_video_tools():
    ffmpeg = os.environ.get("FFMPEG_BIN") or os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("FFPROBE_BIN") or os.environ.get("FFPROBE") or shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return str(ffmpeg), str(ffprobe)

    static_tools = resolve_static_ffmpeg()
    if static_tools:
        return static_tools

    if os.environ.get("VID_EDIT_NO_AUTO_INSTALL", "").lower() not in {"1", "true", "yes"}:
        install_static_ffmpeg()
        static_tools = resolve_static_ffmpeg()
        if static_tools:
            return static_tools

    raise RuntimeError(
        "ffmpeg/ffprobe were not found. Install ffmpeg, or run "
        f"`python3 -m pip install --target {STATIC_FFMPEG_DIR} static-ffmpeg` "
        "and retry."
    )


def resolve_static_ffmpeg():
    direct_tools = resolve_static_ffmpeg_files()
    if direct_tools:
        return direct_tools

    if STATIC_FFMPEG_DIR.exists():
        sys.path.insert(0, str(STATIC_FFMPEG_DIR))

    try:
        from static_ffmpeg import run
    except Exception:
        return None

    try:
        return run.get_or_fetch_platform_executables_else_raise()
    except Exception:
        return None


def resolve_static_ffmpeg_files():
    bin_root = STATIC_FFMPEG_DIR / "static_ffmpeg" / "bin"
    if not bin_root.exists():
        return None

    for platform_dir in sorted(bin_root.iterdir()):
        ffmpeg = platform_dir / "ffmpeg"
        ffprobe = platform_dir / "ffprobe"
        if ffmpeg.exists() and ffprobe.exists():
            return str(ffmpeg), str(ffprobe)
    return None


def install_static_ffmpeg():
    STATIC_FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"ffmpeg/ffprobe not found on PATH; installing static-ffmpeg into {STATIC_FFMPEG_DIR}",
        file=sys.stderr,
        flush=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", str(STATIC_FFMPEG_DIR), "static-ffmpeg"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-1200:]
        raise RuntimeError(f"static-ffmpeg install failed:\n{detail}")


def detect_silences(filepath, threshold=DEFAULT_THRESHOLD, min_duration=DEFAULT_MIN_DURATION):
    ffmpeg, ffprobe = get_video_tools()
    result = subprocess.run(
        [ffmpeg, "-i", filepath,
         "-af", f"silencedetect=noise={threshold}dB:d={min_duration}",
         "-f", "null", "-"],
        capture_output=True, text=True
    )
    silences = []
    current_start = None
    for line in result.stderr.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                current_start = max(float(m.group(1)), 0.0)
        elif "silence_end" in line:
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m and current_start is not None:
                end = float(m.group(1))
                if end > current_start:
                    silences.append({"start": current_start, "end": end})
                current_start = None

    probe = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True, text=True
    )
    total = float(json.loads(probe.stdout)["format"]["duration"])
    if current_start is not None:
        silences.append({"start": current_start, "end": total})
    return silences, total


def keep_segments(silences, total, padding=DEFAULT_PADDING):
    """Invert silences into keep-segments using the web UI's speech-preserving padding."""
    segs = []
    cursor = 0.0
    for s in silences:
        keep_end = min(s["start"] + padding, total)
        keep_start = max(s["end"] - padding, 0.0)
        if cursor < keep_end:
            segs.append({"start": cursor, "end": keep_end})
        cursor = max(keep_start, cursor)
    if cursor < total:
        segs.append({"start": cursor, "end": total})

    merged = []
    for seg in segs:
        if merged and (seg["start"] - merged[-1]["end"]) < MERGE_GAP:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return [s for s in merged if s["end"] > s["start"]]


def merge_segments(segments, merge_gap=MERGE_GAP):
    merged = []
    for seg in sorted(segments, key=lambda item: item["start"]):
        if seg["end"] <= seg["start"]:
            continue
        if merged and seg["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged


def resolve_transcriber():
    configured = os.environ.get("VID_EDIT_TRANSCRIBER")
    candidates = [configured] if configured else []
    candidates.extend(["mlx_whisper", "whisper"])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def extract_transcription_audio(filepath, audio_path):
    ffmpeg, _ = get_video_tools()
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            filepath,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])


def transcribe_words(filepath, *, language=DEFAULT_WHISPER_LANGUAGE):
    transcriber = resolve_transcriber()
    if not transcriber:
        raise RuntimeError("No local whisper transcriber found")

    with tempfile.TemporaryDirectory(prefix="vid-edit-speech-") as tmp:
        tmpdir = Path(tmp)
        audio_path = tmpdir / "audio.wav"
        extract_transcription_audio(filepath, audio_path)

        is_mlx = "mlx_whisper" in Path(transcriber).name
        model = DEFAULT_MLX_WHISPER_MODEL if is_mlx else DEFAULT_WHISPER_MODEL
        cmd = [
            transcriber,
            str(audio_path),
            "--model",
            model,
            "--output-format" if is_mlx else "--output_format",
            "json",
            "--output-dir" if is_mlx else "--output_dir",
            str(tmpdir),
            "--verbose",
            "False",
        ]
        if is_mlx:
            cmd.extend(["--word-timestamps", "True"])
        else:
            cmd.extend(["--word_timestamps", "True"])
        if language:
            cmd.extend(["--language", language])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-1200:]
            raise RuntimeError(detail)

        json_path = tmpdir / "audio.json"
        if not json_path.exists():
            matches = list(tmpdir.glob("*.json"))
            if not matches:
                raise RuntimeError("Transcription did not produce JSON output")
            json_path = matches[0]

        data = json.loads(json_path.read_text())

    words = []
    for segment in data.get("segments", []):
        for word in segment.get("words", []) or []:
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                words.append({"start": start, "end": end, "word": str(word.get("word", "")).strip()})
    return sorted(words, key=lambda item: (item["start"], item["end"]))


def plain_word(value):
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def remove_isolated_fillers(words, min_gap=DEFAULT_FILLER_GAP):
    kept = []
    removed = []
    for index, word in enumerate(words):
        if plain_word(word.get("word")) not in FILLER_WORDS:
            kept.append(word)
            continue

        prev_word = words[index - 1] if index > 0 else None
        next_word = words[index + 1] if index + 1 < len(words) else None
        if not prev_word or not next_word:
            kept.append(word)
            continue

        prev_gap = word["start"] - prev_word["end"]
        next_gap = next_word["start"] - word["end"]
        if prev_gap >= min_gap and next_gap >= min_gap:
            removed.append(word)
        else:
            kept.append(word)

    return kept, removed


def keep_segments_from_words(
    words,
    total,
    *,
    pre_pad=DEFAULT_WORD_PRE_PAD,
    post_pad=DEFAULT_WORD_POST_PAD,
    max_word_gap=DEFAULT_MAX_WORD_GAP,
):
    if not words:
        return []

    grouped = []
    group_start = words[0]["start"]
    group_end = words[0]["end"]
    for word in words[1:]:
        if word["start"] - group_end > max_word_gap:
            grouped.append({"start": group_start, "end": group_end})
            group_start = word["start"]
            group_end = word["end"]
        else:
            group_end = max(group_end, word["end"])
    grouped.append({"start": group_start, "end": group_end})

    padded = [
        {
            "start": max(0.0, group["start"] - pre_pad),
            "end": min(total, group["end"] + post_pad),
        }
        for group in grouped
    ]
    return merge_segments(padded)


def build_keep_segments(filepath, args):
    if args.mode in {"auto", "speech"}:
        try:
            total = probe_duration(filepath)
            words = transcribe_words(filepath, language=args.language)
            speech_words = words
            removed_fillers = []
            if args.remove_fillers:
                speech_words, removed_fillers = remove_isolated_fillers(words, args.filler_gap)
            segs = keep_segments_from_words(
                speech_words,
                total,
                pre_pad=args.word_pre_pad,
                post_pad=args.word_post_pad,
                max_word_gap=args.max_word_gap,
            )
            if segs:
                return segs, total, {
                    "mode": "speech",
                    "word_count": len(words),
                    "gap_count": max(0, len(segs) - 1),
                    "removed_filler_count": len(removed_fillers),
                }
            if args.mode == "speech":
                raise RuntimeError("Transcription returned no usable word timings")
            print("speech-aware cleanup returned no words; falling back to volume detection")
        except Exception as exc:
            if args.mode == "speech":
                raise
            print(f"speech-aware cleanup unavailable ({exc}); falling back to volume detection")

    silences, total = detect_silences(filepath)
    segs = keep_segments(silences, total)
    return segs, total, {
        "mode": "volume",
        "gap_count": len(silences),
    }


def trim_concat(filepath, outpath, segments):
    ffmpeg, _ = get_video_tools()
    filter_parts = []
    concat = ""
    for i, seg in enumerate(segments):
        filter_parts.append(
            f"[0:v]trim=start={seg['start']:.4f}:end={seg['end']:.4f},setpts=PTS-STARTPTS[v{i}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={seg['start']:.4f}:end={seg['end']:.4f},asetpts=PTS-STARTPTS[a{i}]"
        )
        concat += f"[v{i}][a{i}]"
    filter_parts.append(f"{concat}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        ffmpeg, "-y", "-i", filepath,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        outpath,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def merge_videos(filepaths, outpath, width=DEFAULT_MERGE_WIDTH, height=DEFAULT_MERGE_HEIGHT, fps=DEFAULT_MERGE_FPS):
    ffmpeg, _ = get_video_tools()
    filter_parts = []
    concat = ""
    cmd = [ffmpeg, "-y", "-hide_banner"]
    for index, filepath in enumerate(filepaths):
        cmd.extend(["-i", str(filepath)])
        filter_parts.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1,setpts=PTS-STARTPTS[v{index}]"
        )
        filter_parts.append(
            f"[{index}:a]aresample=48000,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        concat += f"[v{index}][a{index}]"

    filter_parts.append(f"{concat}concat=n={len(filepaths)}:v=1:a=1[outv][outa]")
    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]", "-map", "[outa]",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(outpath),
    ])
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(filepath):
    _ffmpeg, ffprobe = get_video_tools()
    probe = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True,
        text=True,
    )
    probe.check_returncode()
    return float(json.loads(probe.stdout)["format"]["duration"])


def process(filepath, args):
    src = Path(filepath)
    outpath = src.with_name(f"{src.stem} - {args.suffix}{src.suffix}")
    print(f"\n=== {src.name} ===")
    try:
        segs, total, info = build_keep_segments(str(src), args)
    except Exception as exc:
        print(f"speech-aware cleanup failed: {exc}")
        return None
    if info["mode"] == "speech":
        print(
            f"duration {total:.1f}s | {info['word_count']} transcript words | "
            f"{info['gap_count']} speech gaps | {info.get('removed_filler_count', 0)} isolated fillers removed"
        )
    else:
        print(f"duration {total:.1f}s | {info['gap_count']} silent gaps")
    kept = sum(s["end"] - s["start"] for s in segs)
    print(f"keeping {kept:.1f}s ({kept / total * 100:.0f}%) across {len(segs)} segments")
    if not segs:
        print("!! nothing to keep, skipping")
        return None
    result = trim_concat(str(src), str(outpath), segs)
    if result.returncode != 0:
        print("ffmpeg failed:")
        print(result.stderr[-800:])
        return None
    size = os.path.getsize(outpath) / 1024 / 1024
    print(f"wrote {outpath} ({size:.1f} MB)")
    return outpath


def require_pattern(path, label, pattern, failures):
    try:
        text = path.read_text()
    except OSError as exc:
        failures.append(f"{path}: could not read {label}: {exc}")
        return

    if re.search(pattern, text, re.MULTILINE) is None:
        failures.append(f"{path}: missing {label}")


def verify_default_alignment():
    root = Path(__file__).resolve().parent
    min_duration_text = f"{DEFAULT_MIN_DURATION:.2f}"
    padding_seconds_text = f"{DEFAULT_PADDING:.2f}"
    skill_notes = Path(
        os.environ.get(
            "VID_EDIT_SKILL_PATH",
            "/Users/lukebyrne/.codex/skills/vid-edit-silence-remover/SKILL.md",
        )
    )
    failures = []

    require_pattern(
        root / "server.py",
        "server threshold fallback",
        rf'params\.get\("threshold",\s*{DEFAULT_THRESHOLD}\)',
        failures,
    )
    require_pattern(
        root / "server.py",
        "server minimum silence fallback",
        r'params\.get\("minDuration",\s*DEFAULT_MIN_DURATION\)',
        failures,
    )
    require_pattern(
        root / "server.py",
        "server padding fallback",
        r'params\.get\("padding",\s*DEFAULT_PADDING\)',
        failures,
    )
    require_pattern(
        root / "index.html",
        "threshold slider default",
        rf'id="threshold"[^>]*value="{DEFAULT_THRESHOLD}"',
        failures,
    )
    require_pattern(
        root / "index.html",
        "minimum silence label",
        rf'id="duration-val">{min_duration_text}s</span>',
        failures,
    )
    require_pattern(
        root / "index.html",
        "minimum silence slider default",
        rf'id="min-duration"[^>]*value="{min_duration_text}"',
        failures,
    )
    require_pattern(
        root / "index.html",
        "padding label",
        rf'id="padding-val">{DEFAULT_PADDING_MS}ms</span>',
        failures,
    )
    require_pattern(
        root / "index.html",
        "padding slider default",
        rf'id="padding"[^>]*value="{DEFAULT_PADDING_MS}"',
        failures,
    )
    require_pattern(
        root / "index.html",
        "segment merge gap",
        rf'merged\.length && \(seg\.start - merged\[merged\.length - 1\]\.end\) < {MERGE_GAP}',
        failures,
    )
    require_pattern(
        root / "speech_safe_edl_render.py",
        "EDL silence detection defaults",
        rf'def detect_silences\(filepath: str, threshold: float = {DEFAULT_THRESHOLD}, min_duration: float = {min_duration_text}\)',
        failures,
    )
    require_pattern(
        root / "speech_safe_edl_render.py",
        "EDL keep-segment defaults",
        rf'def invert_silences\(silences: Iterable\[Span\], total: float, padding: float = {padding_seconds_text}, merge_gap: float = {MERGE_GAP}\)',
        failures,
    )
    require_pattern(
        root / "watch_silence.py",
        "watcher uses CLI segment math",
        r"from remove_silence_cli import detect_silences, keep_segments, trim_concat",
        failures,
    )
    require_pattern(
        root / "watch_silence.py",
        "watcher output suffix",
        r'f"\{src\.stem\} - nosilence\{src\.suffix\}"',
        failures,
    )
    require_pattern(
        root / "server.py",
        "server word gap default",
        rf"DEFAULT_MAX_WORD_GAP\s*=\s*{DEFAULT_MAX_WORD_GAP:.2f}",
        failures,
    )
    require_pattern(
        skill_notes,
        "skill threshold note",
        rf"Silence threshold:\s*`{DEFAULT_THRESHOLD} dB`",
        failures,
    )
    require_pattern(
        skill_notes,
        "skill minimum silence note",
        rf"Minimum silence duration:\s*`{min_duration_text}s`",
        failures,
    )
    require_pattern(
        skill_notes,
        "skill padding note",
        rf"Speech padding:\s*`{DEFAULT_PADDING_MS}ms`",
        failures,
    )

    if failures:
        print("Default alignment check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        "Default alignment check passed: "
        f"threshold={DEFAULT_THRESHOLD}dB, "
        f"min_duration={DEFAULT_MIN_DURATION}s, "
        f"padding={DEFAULT_PADDING_MS}ms, "
        f"merge_gap={int(MERGE_GAP * 1000)}ms, "
        "watcher=CLI defaults"
    )
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Remove silence from videos with Vid-Edit defaults, optionally merging the outputs."
    )
    parser.add_argument(
        "--verify-defaults",
        action="store_true",
        help="check that CLI, site, server, EDL helper, and skill default silence settings match",
    )
    parser.add_argument(
        "--merge-output",
        help="merge processed outputs into this MP4 after silence removal",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="skip silence removal and merge the provided files directly",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "speech", "volume"],
        default=DEFAULT_SPEECH_MODE,
        help="auto tries local transcript word timings before falling back to volume detection",
    )
    parser.add_argument(
        "--suffix",
        default="nosilence",
        help="output filename suffix after the original stem",
    )
    parser.add_argument("--language", default=DEFAULT_WHISPER_LANGUAGE, help="transcription language; use empty string to auto-detect")
    parser.add_argument("--max-word-gap", type=float, default=DEFAULT_MAX_WORD_GAP)
    parser.add_argument("--word-pre-pad", type=float, default=DEFAULT_WORD_PRE_PAD)
    parser.add_argument("--word-post-pad", type=float, default=DEFAULT_WORD_POST_PAD)
    parser.add_argument("--filler-gap", type=float, default=DEFAULT_FILLER_GAP)
    parser.add_argument(
        "--keep-fillers",
        action="store_false",
        dest="remove_fillers",
        help="keep isolated um/uh filler words instead of removing them",
    )
    parser.add_argument("files", nargs="*", help="input video files")
    args = parser.parse_args(argv)
    if args.verify_defaults and (args.files or args.merge_output or args.merge_only):
        parser.error("--verify-defaults does not process files")
    if not args.verify_defaults and not args.files:
        parser.error("files are required unless --verify-defaults is used")
    if args.merge_only and not args.merge_output:
        parser.error("--merge-only requires --merge-output")
    return args


def main(argv):
    args = parse_args(argv)
    if args.verify_defaults:
        return verify_default_alignment()

    if args.merge_only:
        merge_inputs = [Path(f) for f in args.files]
    else:
        merge_inputs = []
        for f in args.files:
            outpath = process(f, args)
            if outpath is not None:
                merge_inputs.append(outpath)

    if args.merge_output:
        if len(merge_inputs) != len(args.files):
            print("!! not merging because one or more inputs failed")
            return 1
        merge_out = Path(args.merge_output)
        print(f"\n=== merging {len(merge_inputs)} videos ===")
        result = merge_videos(merge_inputs, merge_out)
        if result.returncode != 0:
            print("ffmpeg merge failed:")
            print(result.stderr[-1200:])
            return result.returncode
        size = os.path.getsize(merge_out) / 1024 / 1024
        print(f"wrote {merge_out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

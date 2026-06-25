#!/usr/bin/env python3
"""Vid-Edit — Silence Remover + Transcript Editor server."""

import json
import os
import re
import subprocess
import tempfile
import shutil
import sys
import threading
from difflib import SequenceMatcher
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import speech_detect
import calibration

# Load .env file if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

UPLOAD_DIR = tempfile.mkdtemp(prefix="vid_edit_")
METADATA_PATH = os.path.join(UPLOAD_DIR, "metadata.json")
STATIC_FFMPEG_DIR = Path(os.environ.get("VID_EDIT_STATIC_FFMPEG_DIR", "/tmp/vid-edit-static-ffmpeg"))
_VIDEO_TOOLS = None

_progress_lock = threading.Lock()
_progress = {"active": False, "percent": 0, "stage": "idle", "error": None}
DEFAULT_SPEECH_AWARE = True
DEFAULT_PADDING = 0.20
DEFAULT_MIN_DURATION = 0.50
MERGE_GAP = 0.15
DEFAULT_WORD_PRE_PAD = 0.20
DEFAULT_WORD_POST_PAD = 0.25
DEFAULT_MAX_WORD_GAP = 1.00
DEFAULT_FILLER_GAP = 0.20
FILLER_WORDS = {"um", "uh", "uhm", "umm", "erm", "er", "ah"}
# Accuracy matters far more than speed here: whisper-tiny both misses real words
# (clipping speech) and hallucinates ghost phrases over music/silence (keeping dead
# air). large-v3-turbo is accurate and still ~real-time on Apple Silicon.
DEFAULT_MLX_WHISPER_MODEL = os.environ.get("VID_EDIT_MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
DEFAULT_WHISPER_MODEL = os.environ.get("VID_EDIT_WHISPER_MODEL", "small.en")
DEFAULT_WHISPER_LANGUAGE = os.environ.get("VID_EDIT_WHISPER_LANGUAGE", "en")

# Batch review queue (server-side file paths -> reviewed -> rendered to disk).
QUEUE = {"items": [], "current": None}
_queue_lock = threading.Lock()

# Cache of the last analyze() so the UI can re-tune params (padding/sensitivity)
# instantly via /api/resegment, without re-running Whisper.
_LAST = {"path": None, "analysis": None, "words": None}


def _set_progress(**kwargs):
    with _progress_lock:
        _progress.update(kwargs)


def _get_progress():
    with _progress_lock:
        return dict(_progress)


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
        "and restart Vid-Edit."
    )


def resolve_static_ffmpeg():
    direct_tools = resolve_static_ffmpeg_files()
    if direct_tools:
        return direct_tools

    if STATIC_FFMPEG_DIR.exists() and str(STATIC_FFMPEG_DIR) not in sys.path:
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
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", str(STATIC_FFMPEG_DIR), "static-ffmpeg"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-1200:]
        raise RuntimeError(f"static-ffmpeg install failed:\n{detail}")


def safe_download_name(filename, suffix="silence-removed"):
    base = os.path.basename(filename or "").strip()
    base = re.sub(r"\.[^.]+$", "", base)
    base = re.sub(r"[^A-Za-z0-9._ -]+", "", base).strip(" .")
    if not base:
        base = "output"
    return f"{base}_{suffix}.mp4"


def get_duration(filepath):
    _ffmpeg, ffprobe = get_video_tools()
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True, text=True
    )
    result.check_returncode()
    return float(json.loads(result.stdout)["format"]["duration"])


def get_video_dimensions(filepath):
    _ffmpeg, ffprobe = get_video_tools()
    result = subprocess.run(
        [
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-select_streams", "v:0", "-show_entries", "stream=width,height",
            filepath,
        ],
        capture_output=True, text=True
    )
    result.check_returncode()
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("No video stream found")
    return int(streams[0]["width"]), int(streams[0]["height"])


def hex_to_ass_color(value, alpha=0):
    value = str(value or "#ffffff").strip()
    if not re.match(r"^#[0-9A-Fa-f]{6}$", value):
        value = "#ffffff"
    alpha = max(0, min(255, int(alpha)))
    rr = value[1:3]
    gg = value[3:5]
    bb = value[5:7]
    return f"&H{alpha:02X}{bb}{gg}{rr}"


def escape_ass_text(value):
    text = str(value or "").replace("{", "(").replace("}", ")")
    text = text.replace("\\", "\\\\")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\N")


def escape_filter_path(value):
    return str(value).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def normalize_segments(segments, total_duration):
    normalized = []
    for seg in segments:
        try:
            start = max(0.0, min(float(seg["start"]), total_duration))
            end = max(0.0, min(float(seg["end"]), total_duration))
        except (KeyError, TypeError, ValueError):
            continue

        if end - start < 0.02:
            continue

        if normalized and start <= normalized[-1]["end"] + 0.001:
            normalized[-1]["end"] = max(normalized[-1]["end"], end)
        else:
            normalized.append({"start": start, "end": end})

    return normalized


def merge_segments(segments, total_duration, merge_gap=MERGE_GAP):
    normalized = normalize_segments(
        sorted(segments, key=lambda item: item["start"]),
        total_duration,
    )
    merged = []
    for seg in normalized:
        if merged and seg["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    return merged


def silences_from_segments(segments, total_duration):
    silences = []
    cursor = 0.0
    for seg in segments:
        if seg["start"] > cursor:
            silences.append({"start": cursor, "end": seg["start"]})
        cursor = max(cursor, seg["end"])
    if cursor < total_duration:
        silences.append({"start": cursor, "end": total_duration})
    return silences


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
    ffmpeg, _ffprobe = get_video_tools()
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-y", "-i", filepath,
            "-vn", "-ac", "1", "-ar", "16000", str(audio_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])


def transcribe_words_local(filepath, language=DEFAULT_WHISPER_LANGUAGE):
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
        cmd.extend(["--word-timestamps" if is_mlx else "--word_timestamps", "True"])
        if language:
            cmd.extend(["--language", language])

        # Anti-hallucination decode settings. Both CLIs accept these (mlx uses
        # dashes, openai-whisper uses underscores). Greedy decode at temp 0,
        # don't carry context across windows (stops repeated-phrase drift), and
        # let whisper skip its own detected silence. The energy gate in
        # speech_detect is the real backstop, but these reduce ghosts at source.
        sep = "-" if is_mlx else "_"
        def flag(name):
            return "--" + name.replace("_", sep)
        optional_flags = [
            flag("temperature"), "0",
            flag("condition_on_previous_text"), "False",
            flag("no_speech_threshold"), "0.6",
            flag("logprob_threshold"), "-1.0",
            flag("compression_ratio_threshold"), "2.4",
            flag("hallucination_silence_threshold"), "0.5",
        ]

        result = subprocess.run(cmd + optional_flags, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")
            # If this whisper build rejects one of the optional decode flags,
            # retry with just the core args rather than losing all word data.
            if re.search(r"unrecognized arguments|unknown|invalid choice|no such option|unexpected", err, re.I):
                result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout)[-1200:])

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


def speech_segments_from_words(
    words,
    total_duration,
    pre_pad=DEFAULT_WORD_PRE_PAD,
    post_pad=DEFAULT_WORD_POST_PAD,
    max_word_gap=DEFAULT_MAX_WORD_GAP,
):
    if not words:
        return []

    groups = []
    group_start = words[0]["start"]
    group_end = words[0]["end"]
    for word in words[1:]:
        if word["start"] - group_end > max_word_gap:
            groups.append({"start": group_start, "end": group_end})
            group_start = word["start"]
            group_end = word["end"]
        else:
            group_end = max(group_end, word["end"])
    groups.append({"start": group_start, "end": group_end})

    padded = [
        {
            "start": max(0.0, group["start"] - pre_pad),
            "end": min(total_duration, group["end"] + post_pad),
        }
        for group in groups
    ]
    return merge_segments(padded, total_duration)


def detect_volume_silences(filepath, threshold, min_duration, total_duration):
    ffmpeg, _ffprobe = get_video_tools()
    result = subprocess.run(
        [
            ffmpeg, "-i", filepath,
            "-af", f"silencedetect=noise={threshold}dB:d={min_duration}",
            "-f", "null", "-"
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])

    silences = []
    current_start = None
    for line in result.stderr.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                current_start = max(float(m.group(1)), 0.0)
        elif "silence_end" in line:
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m:
                end = min(max(0.0, float(m.group(1))), total_duration)
                start = current_start if current_start is not None else 0.0
                if end > start:
                    silences.append({"start": start, "end": end})
                current_start = None
    if current_start is not None:
        silences.append({"start": current_start, "end": total_duration})
    return silences


def volume_keep_segments(silences, total_duration, padding=DEFAULT_PADDING):
    segments = []
    cursor = 0.0
    for silence in silences:
        keep_end = min(float(silence["start"]) + padding, total_duration)
        keep_start = max(float(silence["end"]) - padding, 0.0)
        if cursor < keep_end:
            segments.append({"start": cursor, "end": keep_end})
        cursor = max(keep_start, cursor)
    if cursor < total_duration:
        segments.append({"start": cursor, "end": total_duration})
    return merge_segments(segments, total_duration)


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        path = urlparse(self.path).path
        routes = {
            "/api/upload": self.handle_upload,
            "/api/detect": self.handle_detect,
            "/api/process": self.handle_process,
            "/api/transcribe": self.handle_transcribe,
            "/api/find-duplicates": self.handle_find_duplicates,
            "/api/export": self.handle_export,
            "/api/render-captions": self.handle_render_captions,
            # speech-gated review workflow
            "/api/queue": self.handle_queue_set,
            "/api/load": self.handle_load,
            "/api/analyze": self.handle_analyze,
            "/api/resegment": self.handle_resegment,
            "/api/learn": self.handle_learn,
            "/api/render": self.handle_render,
        }
        handler = routes.get(path)
        if handler:
            handler()
        else:
            self.send_error(404)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/download/"):
            self.handle_download()
        elif path == "/api/video":
            self.handle_video()
        elif path == "/api/progress":
            self.send_json(_get_progress())
        elif path == "/api/queue":
            self.send_json({"ok": True, **QUEUE})
        elif path == "/api/profile":
            qs = parse_qs(urlparse(self.path).query)
            channel = qs.get("channel", ["default"])[0]
            raw = calibration.load_raw(channel)
            self.send_json({"ok": True, "channel": raw["channel"], "samples": raw["samples"],
                            "overrides": raw["overrides"], "effective": calibration.load_profile(channel),
                            "history": raw.get("history", [])[-10:]})
        else:
            super().do_GET()

    # ---- Speech-gated review workflow ----
    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n)) if n else {}

    VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

    def handle_queue_set(self):
        body = self._read_json()
        exts = {e.lower() if e.startswith(".") else "." + e.lower()
                for e in body.get("exts", [])} or self.VIDEO_EXTS
        items = []
        seen = set()

        def add_file(path, channel=None):
            p = Path(path)
            if not p.is_file() or p.suffix.lower() not in exts:
                return
            rp = str(p.resolve())
            if rp in seen:
                return
            seen.add(rp)
            ch = channel or calibration._safe_channel(p.parent.name)
            out = p.parent / "no-silence" / (p.stem + ".mp4")
            items.append({
                "path": rp, "name": p.name, "folder": str(p.parent),
                "channel": ch, "output": str(out), "status": "pending",
            })

        for folder in body.get("folders", []):
            fp = Path(folder).expanduser()
            ch = calibration._safe_channel(fp.name)
            globber = fp.rglob("*") if body.get("recursive") else fp.glob("*")
            for f in sorted(globber):
                add_file(f, ch)
        for f in body.get("files", []):
            add_file(Path(f).expanduser())

        with _queue_lock:
            QUEUE["items"] = items
            QUEUE["current"] = None
        self.send_json({"ok": True, "items": items, "count": len(items)})

    def handle_load(self):
        body = self._read_json()
        src = Path(str(body.get("path", ""))).expanduser()
        if not src.is_file():
            self.send_json({"ok": False, "error": f"File not found: {src}"}, 400)
            return
        target = os.path.join(UPLOAD_DIR, "input.mp4")
        try:
            # Atomic swap: build a temp symlink then os.replace it onto input.mp4,
            # so there's never a window where input.mp4 is missing.
            tmp_link = os.path.join(UPLOAD_DIR, f"input.link.{os.getpid()}.{threading.get_ident()}")
            if os.path.islink(tmp_link) or os.path.exists(tmp_link):
                os.remove(tmp_link)
            os.symlink(str(src.resolve()), tmp_link)
            os.replace(tmp_link, target)
            # Stale analysis from the previous file must not be reused.
            _LAST.update(path=None, analysis=None, words=None)
            duration = get_duration(target)
            width, height = get_video_dimensions(target)
            with open(METADATA_PATH, "w") as f:
                json.dump({"sourceName": src.name,
                           "downloadName": safe_download_name(src.name)}, f)
            with _queue_lock:
                QUEUE["current"] = str(src.resolve())
            self.send_json({"ok": True, "name": src.name, "duration": duration,
                            "width": width, "height": height})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def handle_analyze(self):
        body = self._read_json()
        channel = body.get("channel", "default")
        overrides = body.get("overrides") or {}
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file loaded"}, 400)
            return
        if _get_progress()["active"]:
            self.send_json({"ok": False, "error": "Another job is already running"}, 409)
            return
        _set_progress(active=True, percent=0, stage="transcribing", error=None)
        try:
            ffmpeg, _ffprobe = get_video_tools()
            try:
                words = transcribe_words_local(filepath)
            except Exception as e:
                words = []
                transcribe_warning = f"Transcription failed ({e}); using energy gate only."
            else:
                transcribe_warning = None
            _set_progress(percent=60, stage="analyzing")
            analysis = speech_detect.analyze_audio(filepath, ffmpeg=ffmpeg)
            _LAST.update(path=os.path.realpath(filepath), analysis=analysis, words=words)
            profile = calibration.load_profile(channel)
            for k, v in overrides.items():
                if v is not None:
                    profile[k] = v
            keep, cuts, meta = speech_detect.detect_segments(analysis, words, profile)

            n = 2400
            wave = {
                "peaks": speech_detect.downsample(analysis["peaks"], n),
                "speechDb": speech_detect.downsample(analysis["speech_db"], n),
                "fullDb": speech_detect.downsample(analysis["full_db"], n),
                "thresholdDb": meta.get("threshold_db"),
                "floorDb": meta.get("floor_db"),
            }
            words_out = [{"start": round(w["start"], 3), "end": round(w["end"], 3),
                          "word": w.get("word", "")} for w in words]
            self.send_json({
                "ok": True, "segments": keep, "cuts": cuts, "meta": meta,
                "words": words_out, "wordCount": len(words),
                "waveform": wave, "totalDuration": analysis["duration"],
                "channel": channel, "profile": profile,
                "warning": transcribe_warning,
            })
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)
        finally:
            _set_progress(active=False, percent=0, stage="idle")

    def handle_resegment(self):
        """Re-run only the segment math (no Whisper) with new params -> instant tuning."""
        body = self._read_json()
        channel = body.get("channel", "default")
        overrides = body.get("overrides") or {}
        if _LAST.get("analysis") is None:
            self.send_json({"ok": False, "error": "Nothing analyzed yet"}, 400)
            return
        cur = os.path.realpath(os.path.join(UPLOAD_DIR, "input.mp4"))
        if _LAST.get("path") != cur:
            self.send_json({"ok": False, "error": "Analysis is stale — re-analyze this clip"}, 409)
            return
        try:
            profile = calibration.load_profile(channel)
            for k, v in overrides.items():
                if v is not None:
                    profile[k] = v
            keep, cuts, meta = speech_detect.detect_segments(
                _LAST["analysis"], _LAST["words"], profile)
            self.send_json({"ok": True, "segments": keep, "cuts": cuts, "meta": meta,
                            "totalDuration": _LAST["analysis"]["duration"], "profile": profile})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def handle_learn(self):
        body = self._read_json()
        channel = body.get("channel", "default")
        proposed = body.get("proposed", [])
        corrected = body.get("corrected", [])
        try:
            summary = calibration.learn_from_corrections(
                channel, proposed, corrected,
                duration=body.get("duration"), note=body.get("note"))
            self.send_json({"ok": True, "summary": summary,
                            "profile": calibration.load_profile(channel)})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def handle_render(self):
        body = self._read_json()
        segments = body.get("segments", [])
        channel = body.get("channel", "default")
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file loaded"}, 400)
            return
        duration = get_duration(filepath)
        segments = normalize_segments(segments, duration)
        if not segments:
            self.send_json({"ok": False, "error": "No segments to keep"}, 400)
            return

        out = body.get("output")
        if not out:
            self.send_json({"ok": False, "error": "No output path"}, 400)
            return
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if _get_progress()["active"]:
            self.send_json({"ok": False, "error": "Another job is already running"}, 409)
            return
        _set_progress(active=True, percent=0, stage="rendering", error=None)
        try:
            tmp_out = os.path.join(UPLOAD_DIR, "render_tmp.mp4")
            result = self._build_and_run_concat(segments, tmp_out)
            if result.returncode != 0:
                err = result.stderr[-500:]
                _set_progress(error=err)
                self.send_json({"ok": False, "error": err}, 500)
                return
            shutil.move(tmp_out, str(out_path))
            _set_progress(percent=100.0)
            size = os.path.getsize(out_path)
            with _queue_lock:
                cur = QUEUE.get("current")
                for it in QUEUE["items"]:
                    if it["path"] == cur:
                        it["status"] = "done"
                        it["output"] = str(out_path)
            # Learn only after the render actually succeeded.
            learn_summary = None
            if body.get("learn") and body.get("proposed") is not None:
                try:
                    learn_summary = calibration.learn_from_corrections(
                        channel, body.get("proposed", []), segments,
                        duration=duration, note=body.get("name"))
                except Exception as e:
                    learn_summary = {"error": str(e)}
            self.send_json({"ok": True, "output": str(out_path), "size": size,
                            "keptDuration": round(sum(s["end"] - s["start"] for s in segments), 2),
                            "learn": learn_summary})
        except Exception as e:
            _set_progress(error=str(e))
            self.send_json({"ok": False, "error": str(e)}, 500)
        finally:
            _set_progress(active=False, stage="idle")

    # ---- Upload ----
    def handle_upload(self):
        content_length = int(self.headers["Content-Length"])
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")

        chunk_size = 1024 * 1024
        remaining = content_length
        with open(filepath, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(chunk_size, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        query = parse_qs(urlparse(self.path).query)
        filename = query.get("filename", [""])[0]
        with open(METADATA_PATH, "w") as f:
            json.dump({
                "sourceName": filename,
                "downloadName": safe_download_name(filename),
            }, f)

        try:
            duration = get_duration(filepath)
            size = os.path.getsize(filepath)
            self.send_json({"ok": True, "duration": duration, "size": size})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    # ---- Silence Detection ----
    def handle_detect(self):
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))

        threshold = params.get("threshold", -35)
        min_duration = params.get("minDuration", DEFAULT_MIN_DURATION)
        padding = params.get("padding", DEFAULT_PADDING)
        speech_aware = bool(params.get("speechAware", DEFAULT_SPEECH_AWARE))
        remove_fillers = bool(params.get("removeFillers", True))
        try:
            padding = max(0.0, min(1.0, float(padding)))
        except (TypeError, ValueError):
            padding = DEFAULT_PADDING

        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return

        try:
            total_duration = get_duration(filepath)
            speech_warning = None
            if speech_aware:
                try:
                    words = transcribe_words_local(filepath)
                    speech_words = words
                    removed_fillers = []
                    if remove_fillers:
                        speech_words, removed_fillers = remove_isolated_fillers(words)
                    segments = speech_segments_from_words(
                        speech_words,
                        total_duration,
                        pre_pad=padding,
                        post_pad=max(padding, DEFAULT_WORD_POST_PAD),
                    )
                    if segments:
                        self.send_json({
                            "ok": True,
                            "mode": "speech",
                            "segments": segments,
                            "silences": silences_from_segments(segments, total_duration),
                            "wordCount": len(words),
                            "removedFillerCount": len(removed_fillers),
                            "totalDuration": total_duration,
                        })
                        return
                    speech_warning = "Speech-aware cleanup returned no words; used volume detection instead."
                except Exception as e:
                    speech_warning = f"Speech-aware cleanup unavailable; used volume detection instead. {e}"

            silences = detect_volume_silences(filepath, threshold, min_duration, total_duration)

            self.send_json({
                "ok": True,
                "mode": "volume",
                "silences": silences,
                "totalDuration": total_duration,
                "warning": speech_warning,
            })
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    # ---- Video Processing (trim+concat) ----
    def _build_and_run_concat(self, segments, outpath):
        """Shared FFmpeg trim+concat logic for process and export."""
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        ffmpeg, _ffprobe = get_video_tools()

        filter_parts = []
        concat_inputs = ""
        for i, seg in enumerate(segments):
            s, e = seg["start"], seg["end"]
            filter_parts.append(
                f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs += f"[v{i}][a{i}]"

        filter_parts.append(
            f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]"
        )
        filter_complex = ";".join(filter_parts)

        total_duration = sum(max(0.0, s["end"] - s["start"]) for s in segments)

        proc = subprocess.Popen(
            [
                ffmpeg, "-y", "-i", filepath,
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats",
                outpath
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )

        stderr_chunks = []
        stderr_thread = threading.Thread(
            target=lambda: stderr_chunks.append(proc.stderr.read()),
            daemon=True,
        )
        stderr_thread.start()

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("out_time_ms="):
                try:
                    out_us = int(line.split("=", 1)[1])
                    if total_duration > 0:
                        pct = (out_us / 1_000_000.0) / total_duration * 100.0
                        _set_progress(percent=min(99.0, max(0.0, pct)))
                except ValueError:
                    pass
            elif line == "progress=end":
                _set_progress(percent=99.0)

        proc.wait()
        stderr_thread.join(timeout=2)
        stderr_text = "".join(stderr_chunks)

        class _Result:
            pass
        result = _Result()
        result.returncode = proc.returncode
        result.stderr = stderr_text
        return result

    def _run_concat_job(self, segments, outpath, stage, download_url):
        if _get_progress()["active"]:
            self.send_json({"ok": False, "error": "Another job is already running"}, 409)
            return

        _set_progress(active=True, percent=0, stage=stage, error=None)
        try:
            result = self._build_and_run_concat(segments, outpath)
            if result.returncode != 0:
                err = result.stderr[-500:]
                _set_progress(error=err)
                self.send_json({"ok": False, "error": err}, 500)
                return

            _set_progress(percent=100.0)
            size = os.path.getsize(outpath)
            self.send_json({"ok": True, "size": size, "downloadUrl": download_url})
        except Exception as e:
            _set_progress(error=str(e))
            self.send_json({"ok": False, "error": str(e)}, 500)
        finally:
            _set_progress(active=False, stage="idle")

    def handle_process(self):
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))
        segments = params.get("segments", [])

        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        outpath = os.path.join(UPLOAD_DIR, "output.mp4")

        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return
        if not segments:
            self.send_json({"ok": False, "error": "No segments to keep"}, 400)
            return

        segments = normalize_segments(segments, get_duration(filepath))
        if not segments:
            self.send_json({"ok": False, "error": "No valid segments to keep"}, 400)
            return

        self._run_concat_job(segments, outpath, "process", "/api/download/output.mp4")

    def handle_export(self):
        """Export from transcript editor — same trim+concat, different output name."""
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))
        segments = params.get("segments", [])

        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        outpath = os.path.join(UPLOAD_DIR, "exported.mp4")

        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return
        if not segments:
            self.send_json({"ok": False, "error": "No segments to keep"}, 400)
            return

        segments = normalize_segments(segments, get_duration(filepath))
        if not segments:
            self.send_json({"ok": False, "error": "No valid segments to keep"}, 400)
            return

        self._run_concat_job(segments, outpath, "export", "/api/download/exported.mp4")

    # ---- Caption Rendering ----
    def _format_ass_time(self, seconds):
        seconds = max(0.0, float(seconds))
        total_centiseconds = int(round(seconds * 100))
        cs = total_centiseconds % 100
        total_seconds = total_centiseconds // 100
        sec = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"

    def _normalize_caption_style(self, raw_style, width, height):
        raw_style = raw_style or {}

        def clamp_float(key, default, minimum, maximum):
            try:
                value = float(raw_style.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(maximum, value))

        font_family = str(raw_style.get("fontFamily", "Arial")).strip()
        font_family = re.sub(r"[^A-Za-z0-9 ._-]+", "", font_family)[:64] or "Arial"

        size_pct = clamp_float("size", 6.0, 2.0, 14.0)
        bg_opacity = clamp_float("backgroundOpacity", 0.35, 0.0, 1.0)
        text_align = str(raw_style.get("textAlign", "center")).lower()
        if text_align not in {"left", "center", "right"}:
            text_align = "center"

        alignment = {"left": 4, "center": 5, "right": 6}[text_align]
        q_alignment = {"left": 1, "center": 2, "right": 3}[text_align]

        return {
            "font_family": font_family,
            "font_size": int(round(height * size_pct / 100.0)),
            "x": round(width * clamp_float("x", 50.0, 0.0, 100.0) / 100.0),
            "y": round(height * clamp_float("y", 82.0, 0.0, 100.0) / 100.0),
            "color": hex_to_ass_color(raw_style.get("color", "#fff8d7")),
            "outline_color": hex_to_ass_color(raw_style.get("outlineColor", "#000000")),
            "background_color": hex_to_ass_color(
                raw_style.get("backgroundColor", "#000000"),
                255 - round(bg_opacity * 255)
            ),
            "border_style": 3 if bg_opacity > 0.01 else 1,
            "outline_width": clamp_float("outlineWidth", 3.0, 0.0, 12.0),
            "shadow": clamp_float("shadow", 1.0, 0.0, 8.0),
            "bold": -1 if bool(raw_style.get("bold", True)) else 0,
            "uppercase": bool(raw_style.get("uppercase", False)),
            "alignment": alignment,
            "q_alignment": q_alignment,
        }

    def _normalize_caption_cues(self, cues, total_duration):
        normalized = []
        for cue in (cues or [])[:2000]:
            try:
                start = max(0.0, min(float(cue["start"]), total_duration))
                end = max(0.0, min(float(cue["end"]), total_duration))
            except (KeyError, TypeError, ValueError):
                continue

            text = str(cue.get("text", "")).strip()
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if not text or end - start < 0.05:
                continue
            normalized.append({"start": start, "end": end, "text": text[:500]})

        normalized = sorted(normalized, key=lambda item: (item["start"], item["end"]))
        gap = 0.0
        for index in range(len(normalized) - 1):
            current = normalized[index]
            next_cue = normalized[index + 1]
            latest_end = max(current["start"], next_cue["start"] - gap)
            if current["end"] > latest_end:
                current["end"] = latest_end

        return [cue for cue in normalized if cue["end"] > cue["start"]]

    def _write_caption_ass(self, ass_path, cues, raw_style, width, height):
        style = self._normalize_caption_style(raw_style, width, height)
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 2",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            (
                "Style: Caption,"
                f"{style['font_family']},{style['font_size']},{style['color']},{style['color']},"
                f"{style['outline_color']},{style['background_color']},{style['bold']},0,0,0,"
                f"100,100,0,0,{style['border_style']},{style['outline_width']:.1f},"
                f"{style['shadow']:.1f},{style['alignment']},0,0,0,1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        override = f"{{\\an{style['alignment']}\\pos({style['x']},{style['y']})\\q{style['q_alignment']}}}"
        for cue in cues:
            text = cue["text"].upper() if style["uppercase"] else cue["text"]
            lines.append(
                "Dialogue: 0,"
                f"{self._format_ass_time(cue['start'])},"
                f"{self._format_ass_time(cue['end'])},"
                "Caption,,0,0,0,,"
                f"{override}{escape_ass_text(text)}"
            )

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _build_and_run_caption_render(self, cues, style, outpath):
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        ffmpeg, _ffprobe = get_video_tools()
        total_duration = get_duration(filepath)
        width, height = get_video_dimensions(filepath)
        ass_path = os.path.join(UPLOAD_DIR, "captions.ass")
        self._write_caption_ass(ass_path, cues, style, width, height)

        proc = subprocess.Popen(
            [
                ffmpeg, "-y", "-i", filepath,
                "-vf", f"subtitles={escape_filter_path(ass_path)}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats",
                outpath
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )

        stderr_chunks = []
        stderr_thread = threading.Thread(
            target=lambda: stderr_chunks.append(proc.stderr.read()),
            daemon=True,
        )
        stderr_thread.start()

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("out_time_ms="):
                try:
                    out_us = int(line.split("=", 1)[1])
                    if total_duration > 0:
                        pct = (out_us / 1_000_000.0) / total_duration * 100.0
                        _set_progress(percent=min(99.0, max(0.0, pct)))
                except ValueError:
                    pass
            elif line == "progress=end":
                _set_progress(percent=99.0)

        proc.wait()
        stderr_thread.join(timeout=2)
        stderr_text = "".join(stderr_chunks)

        class _Result:
            pass
        result = _Result()
        result.returncode = proc.returncode
        result.stderr = stderr_text
        return result

    def handle_render_captions(self):
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))

        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        outpath = os.path.join(UPLOAD_DIR, "captioned.mp4")

        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return

        total_duration = get_duration(filepath)
        cues = self._normalize_caption_cues(params.get("cues", []), total_duration)
        if not cues:
            self.send_json({"ok": False, "error": "No caption cues to render"}, 400)
            return

        if _get_progress()["active"]:
            self.send_json({"ok": False, "error": "Another job is already running"}, 409)
            return

        _set_progress(active=True, percent=0, stage="captions", error=None)
        try:
            result = self._build_and_run_caption_render(cues, params.get("style", {}), outpath)
            if result.returncode != 0:
                err = result.stderr[-500:]
                _set_progress(error=err)
                self.send_json({"ok": False, "error": err}, 500)
                return

            _set_progress(percent=100.0)
            size = os.path.getsize(outpath)
            self.send_json({"ok": True, "size": size, "downloadUrl": "/api/download/captioned.mp4"})
        except Exception as e:
            _set_progress(error=str(e))
            self.send_json({"ok": False, "error": str(e)}, 500)
        finally:
            _set_progress(active=False, stage="idle")

    # ---- Transcription ----
    def handle_transcribe(self):
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return

        try:
            ffmpeg, _ffprobe = get_video_tools()
            # Extract audio as compressed mp3 to stay under Whisper's 25MB limit
            audio_path = os.path.join(UPLOAD_DIR, "audio.mp3")
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", filepath,
                    "-vn", "-ar", "16000", "-ac", "1",
                    "-b:a", "64k", audio_path
                ],
                capture_output=True, text=True, check=True
            )

            audio_size = os.path.getsize(audio_path)
            if audio_size > 25 * 1024 * 1024:
                self.send_json({
                    "ok": False,
                    "error": f"Audio file too large ({audio_size // 1024 // 1024}MB). Whisper API limit is 25MB. Try a shorter video."
                }, 400)
                return

            # Call OpenAI Whisper API
            from openai import OpenAI
            client = OpenAI()

            with open(audio_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )

            words = []
            if hasattr(transcription, 'words') and transcription.words:
                for w in transcription.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end
                    })

            segments = []
            if hasattr(transcription, 'segments') and transcription.segments:
                for seg in transcription.segments:
                    segments.append({
                        "text": seg.text.strip(),
                        "start": seg.start,
                        "end": seg.end
                    })

            self.send_json({
                "ok": True,
                "words": words,
                "segments": segments,
                "text": transcription.text
            })

        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    # ---- Duplicate Take Detection ----
    def handle_find_duplicates(self):
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))
        segments = params.get("segments", [])

        if not segments:
            self.send_json({"ok": True, "groups": []})
            return

        try:
            # Normalize text for comparison
            def normalize(text):
                return re.sub(r'[^\w\s]', '', text.lower()).strip()

            # Compare all pairs, find similar non-adjacent segments
            groups = []
            used = set()

            for i in range(len(segments)):
                if i in used:
                    continue
                takes = [{"index": i, **segments[i]}]
                norm_i = normalize(segments[i]["text"])

                # Skip very short segments (less than 5 words)
                if len(norm_i.split()) < 5:
                    continue

                for j in range(i + 1, len(segments)):
                    if j in used:
                        continue
                    norm_j = normalize(segments[j]["text"])
                    if len(norm_j.split()) < 5:
                        continue

                    ratio = SequenceMatcher(None, norm_i, norm_j).ratio()
                    if ratio > 0.55:
                        takes.append({"index": j, "similarity": round(ratio, 2), **segments[j]})
                        used.add(j)

                if len(takes) > 1:
                    used.add(i)
                    groups.append({
                        "text": segments[i]["text"],
                        "takes": takes
                    })

            self.send_json({"ok": True, "groups": groups})

        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    # ---- Video Streaming ----
    def handle_video(self):
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        if not os.path.exists(filepath):
            self.send_error(404)
            return

        file_size = os.path.getsize(filepath)

        # Support range requests for video seeking
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(filepath, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(filepath, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    # ---- File Download ----
    def handle_download(self):
        path = urlparse(self.path).path
        filename = os.path.basename(path.split("/")[-1])
        # Only allow specific output files
        allowed = {"output.mp4", "exported.mp4", "captioned.mp4"}
        if filename not in allowed:
            self.send_error(404)
            return

        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            self.send_error(404)
            return

        download_name = filename
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
            source_name = metadata.get("sourceName", "")
            if filename == "output.mp4":
                download_name = safe_download_name(source_name, "silence-removed")
            elif filename == "exported.mp4":
                download_name = safe_download_name(source_name, "edited")
            elif filename == "captioned.mp4":
                download_name = safe_download_name(source_name, "captioned")

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(os.path.getsize(filepath)))
        self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()

        with open(filepath, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    # ---- Helpers ----
    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    # Serve static files (index.html, review.html) from the repo dir regardless
    # of the launch cwd.
    os.chdir(Path(__file__).parent)
    port = int(os.environ.get("PORT", "8080"))
    print(f"Vid-Edit running at http://localhost:{port}")
    print(f"Review UI:  http://localhost:{port}/review.html")
    print(f"Temp dir: {UPLOAD_DIR}")
    server = ThreadingHTTPServer(("", port), Handler)
    server.serve_forever()

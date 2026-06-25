#!/usr/bin/env python3
"""Speech-gated silence detection.

The core idea (validated against real footage): keep audio only where BOTH
Whisper found real words AND the voice band (300-3400 Hz) actually has energy.

  - Whisper "ghost words" hallucinated over silence  -> no voice energy -> CUT
  - Instrumental music / dead air with no talking     -> energy but no words -> CUT
  - Real speech (even under music)                     -> words + energy -> KEEP

This fixes the two classic failures of plain volume/Whisper-tiny silence
removal: clipped words and dead-air-with-background-music being left in.

Everything here is pure ffmpeg + numpy (no torch / heavy deps), so the energy
gate is also directly drawable on the waveform in the review UI.
"""

import math
import subprocess
import numpy as np

SR = 16000  # analysis sample rate (mono)

# Default per-channel tuning profile. The review UI learns adjustments to these
# from the user's corrections (see calibration.py).
DEFAULT_PROFILE = {
    "floor_pct": 20.0,        # percentile of speech-band dB treated as the silence floor
    "speech_margin_db": 10.0,  # speech must sit this many dB above the floor
    "abs_floor_db": -60.0,     # never treat anything below this as speech, whatever the floor
    "min_ratio": 0.30,         # a voiced frame needs >= this share of energy in the speech band
    "require_words": True,     # keep a voiced region only if Whisper put words in it (kills music/noise)
    "keep_ratio": 0.62,        # if require_words is False: keep wordless regions this speech-like (rescues missed words)
    "bridge_gap": 0.30,        # merge voiced blips separated by less than this (s)
    "min_voiced": 0.15,        # discard voiced islands shorter than this (s)
    "pre_pad": 0.12,           # lead-in kept before a speech region (s)
    "post_pad": 0.20,          # tail kept after a speech region (s) -- generous so word tails survive
    "min_cut_gap": 0.45,       # never cut a gap shorter than this; keeps natural pacing (s)
    "max_word_gap": 0.80,      # word-grouping gap (s), informational
    "hallucination_phrases": [  # whole word-groups matching these (and gated out by energy) are flagged
        "thank you", "thanks for watching", "thank you for watching",
        "please subscribe", "see you next time", "bye", "you",
    ],
}


def merged_profile(overrides):
    p = dict(DEFAULT_PROFILE)
    if overrides:
        for k, v in overrides.items():
            if k in p and v is not None:
                p[k] = v
    return p


def _decode_pcm(filepath, ffmpeg="ffmpeg"):
    """Decode to float32 mono @ SR. Returns np.float32 array in [-1, 1]."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", filepath,
        "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or b"")[-800:].decode("utf-8", "replace"))
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def analyze_audio(filepath, ffmpeg="ffmpeg", hop=0.02, win=0.032, n_peaks=2400):
    """One PCM decode -> energy envelopes used by both the gate and the waveform.

    Returns a dict with per-frame full_db / speech_db / ratio arrays (numpy),
    plus a downsampled abs-amplitude `peaks` array for drawing the waveform.
    """
    x = _decode_pcm(filepath, ffmpeg)
    n = len(x)
    if n == 0:
        return {
            "duration": 0.0, "hop": hop, "sr": SR,
            "full_db": np.array([]), "speech_db": np.array([]),
            "ratio": np.array([]), "peaks": np.zeros(n_peaks),
        }
    duration = n / SR
    hop_n = max(1, int(round(hop * SR)))
    hop = hop_n / SR  # realized stride: keep frame-index<->time mapping exact downstream
    win_n = 1 << int(math.ceil(math.log2(max(256, int(win * SR)))))  # power of 2 >= win

    if n < win_n:
        x = np.pad(x, (0, win_n - n))
    frames = np.lib.stride_tricks.sliding_window_view(x, win_n)[::hop_n]
    n_frames = frames.shape[0]

    window = np.hanning(win_n).astype(np.float32)
    freqs = np.fft.rfftfreq(win_n, 1.0 / SR)
    speech_band = (freqs >= 300.0) & (freqs <= 3400.0)
    eps = 1e-10

    full_db = np.empty(n_frames, dtype=np.float32)
    speech_db = np.empty(n_frames, dtype=np.float32)
    ratio = np.empty(n_frames, dtype=np.float32)

    chunk = 2048  # frames per FFT batch -> bounds peak memory
    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        block = frames[s:e] * window
        spec = np.abs(np.fft.rfft(block, axis=1)) ** 2
        total = spec.sum(axis=1)
        sp = spec[:, speech_band].sum(axis=1)
        full_db[s:e] = 10.0 * np.log10(total + eps)
        speech_db[s:e] = 10.0 * np.log10(sp + eps)
        # A silent frame has no speech-band share (avoid eps/eps == 1.0).
        ratio[s:e] = np.where(total > eps, sp / (total + eps), 0.0)

    # Waveform peaks (max abs amplitude per display bucket)
    if n_peaks > 0 and n > 0:
        bucket = max(1, n // n_peaks)
        usable = (n // bucket) * bucket
        peaks = np.abs(x[:usable]).reshape(-1, bucket).max(axis=1)
        if len(peaks) > n_peaks:
            peaks = peaks[:n_peaks]
    else:
        peaks = np.abs(x)

    return {
        "duration": duration, "hop": hop, "sr": SR,
        "full_db": full_db, "speech_db": speech_db, "ratio": ratio,
        "peaks": peaks,
    }


def _mask_to_intervals(mask, hop, duration):
    """Boolean per-frame mask -> list of {start,end} time intervals."""
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    starts = idx[0::2]
    ends = idx[1::2]
    out = []
    for a, b in zip(starts, ends):
        out.append({"start": float(a * hop), "end": min(float(b * hop), duration)})
    return out


def _bridge(intervals, gap):
    if not intervals:
        return []
    out = [dict(intervals[0])]
    for iv in intervals[1:]:
        if iv["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], iv["end"])
        else:
            out.append(dict(iv))
    return out


def _merge_keep(segments, min_cut_gap, duration):
    """Merge kept segments whose separating gap is below min_cut_gap, clamp + sort."""
    segs = sorted(
        ({"start": max(0.0, s["start"]), "end": min(duration, s["end"])} for s in segments),
        key=lambda s: s["start"],
    )
    segs = [s for s in segs if s["end"] > s["start"]]
    if not segs:
        return []
    out = [dict(segs[0])]
    for s in segs[1:]:
        if s["start"] - out[-1]["end"] < min_cut_gap:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def _complement(keep, duration):
    cuts = []
    cursor = 0.0
    for s in keep:
        if s["start"] > cursor:
            cuts.append({"start": cursor, "end": s["start"]})
        cursor = max(cursor, s["end"])
    if cursor < duration:
        cuts.append({"start": cursor, "end": duration})
    return cuts


def _norm_text(s):
    return "".join(c for c in str(s or "").lower() if c.isalnum() or c == " ").strip()


def detect_segments(analysis, words, profile=None):
    """Combine the energy gate with Whisper words into keep/cut segments.

    Returns (keep_segments, cut_segments, meta). meta includes the gate
    threshold and any flagged hallucinations, for display + debugging.
    """
    p = merged_profile(profile)
    hop = analysis["hop"]
    duration = analysis["duration"]
    speech_db = np.asarray(analysis["speech_db"], dtype=np.float32)
    ratio = np.asarray(analysis["ratio"], dtype=np.float32)
    words = words or []

    if duration <= 0 or len(speech_db) == 0:
        return [], [], {"threshold_db": None, "floor_db": None, "hallucinations": [],
                        "voicedCount": 0, "droppedMusic": 0}

    floor = float(np.percentile(speech_db, p["floor_pct"]))
    threshold = max(floor + p["speech_margin_db"], p["abs_floor_db"])
    voiced_mask = (speech_db > threshold) & (ratio > p["min_ratio"])

    voiced = _mask_to_intervals(voiced_mask, hop, duration)
    voiced = _bridge(voiced, p["bridge_gap"])
    voiced = [iv for iv in voiced if iv["end"] - iv["start"] >= p["min_voiced"]]

    def mean_ratio(iv):
        a = int(iv["start"] / hop)
        b = max(a + 1, int(iv["end"] / hop))
        seg = ratio[a:b]
        return float(seg.mean()) if len(seg) else 0.0

    require_words = bool(p.get("require_words", True))
    keep = []
    dropped_music = 0
    used_word_idx = set()
    for iv in voiced:
        ws = [(i, w) for i, w in enumerate(words)
              if w["end"] > iv["start"] and w["start"] < iv["end"]]
        rescue = (not require_words) and mean_ratio(iv) >= p["keep_ratio"]
        if ws or rescue:
            s, e = iv["start"], iv["end"]
            if ws:
                s = min(s, min(w["start"] for _, w in ws))
                e = max(e, max(w["end"] for _, w in ws))
                used_word_idx.update(i for i, _ in ws)
            keep.append({"start": s, "end": e})
        else:
            dropped_music += 1  # voiced but no words -> instrumental music / noise / breath

    # Words that fell entirely outside any voiced region = energy-gated hallucinations.
    hallucinations = []
    for i, w in enumerate(words):
        if i in used_word_idx:
            continue
        txt = _norm_text(w.get("word"))
        if txt:
            hallucinations.append({"start": w["start"], "end": w["end"], "word": w.get("word", "")})

    # Pad off real boundaries, then merge anything separated by a sub-threshold gap.
    keep = [{"start": max(0.0, s["start"] - p["pre_pad"]),
             "end": min(duration, s["end"] + p["post_pad"])} for s in keep]
    keep = _merge_keep(keep, p["min_cut_gap"], duration)

    # Safety: never return an empty edit (that would delete the whole clip).
    fallback = False
    if not keep:
        keep = [{"start": 0.0, "end": duration}]
        fallback = True

    cuts = _complement(keep, duration)
    kept_dur = sum(s["end"] - s["start"] for s in keep)
    meta = {
        "threshold_db": round(threshold, 1),
        "floor_db": round(floor, 1),
        "voicedCount": len(voiced),
        "droppedMusic": dropped_music,
        "hallucinations": hallucinations[:50],
        "hallucinationCount": len(hallucinations),
        "keptDuration": round(kept_dur, 2),
        "removedDuration": round(max(0.0, duration - kept_dur), 2),
        "fallbackKeptAll": fallback,
    }
    return keep, cuts, meta


def downsample(arr, n):
    """Downsample a 1-D array to n points by block-max (envelopes) for the UI."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) <= n or n <= 0:
        return arr.tolist()
    bucket = len(arr) / n
    out = []
    for i in range(n):
        a = int(i * bucket)
        b = max(a + 1, int((i + 1) * bucket))
        out.append(float(arr[a:b].max()))
    return out

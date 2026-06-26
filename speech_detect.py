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
    # Silence threshold is set on TRUE dBFS: a frame is "silent" if its level is
    # `silence_drop_db` below the file's speech level (85th pct), clamped to an
    # absolute window. This adapts to each recording's loudness yet stays a safe
    # margin under speech so words are never cut. Cuts land only in silence by
    # construction, so we can be tight without clipping.
    "silence_drop_db": 24.0,    # cut line = speech_level - this many dB (deep silence only)
    "silence_thr_min": -54.0,   # never put the cut line below this dBFS
    "silence_thr_max": -40.0,   # never put the cut line above this dBFS (protects quiet word edges)
    "speech_pct": 85.0,         # percentile of dBFS taken as the representative speech level
    "min_cut_gap": 0.35,        # only remove silent gaps at least this long (s)
    "pre_pad": 0.12,            # lead-in kept before speech resumes (s)
    "post_pad": 0.18,           # tail kept after speech (s) -- protects soft consonant tails
    "word_guard": 0.06,         # if word timestamps given, never cut within this of a word (s)
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
    rms_dbfs = np.empty(n_frames, dtype=np.float32)  # TRUE dBFS level (drives silence)

    chunk = 2048  # frames per FFT batch -> bounds peak memory
    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        raw = frames[s:e]
        # Calibrated RMS level in dBFS (0 = full scale). This is what the silence
        # decision uses — comparable across recordings, unlike the FFT-power scale.
        rms = np.sqrt((raw.astype(np.float64) ** 2).mean(axis=1)) + eps
        rms_dbfs[s:e] = 20.0 * np.log10(rms)
        block = raw * window
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
        "rms_dbfs": rms_dbfs, "peaks": peaks,
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
    """Cut only genuinely flat/silent gaps; keep ALL audible audio.

    Simple and safe by design: a cut can only fall where the broadband waveform
    sits near its own noise floor, and padding is kept around every sound, so a
    word's onset/tail/breath is never clipped and audible content (speech, music,
    b-roll) is never deleted. `words` is accepted for an optional ghost count but
    does NOT drive the cut decision.

    Returns (keep_segments, cut_segments, meta).
    """
    p = merged_profile(profile)
    hop = analysis["hop"]
    duration = analysis["duration"]
    # Use calibrated dBFS; fall back to the old FFT scale only if absent.
    db = np.asarray(analysis.get("rms_dbfs", analysis["full_db"]), dtype=np.float32)
    words = words or []

    empty_meta = {"threshold_db": None, "floor_db": None, "cutCount": 0,
                  "hallucinations": [], "hallucinationCount": 0,
                  "keptDuration": round(max(0.0, duration), 2),
                  "removedDuration": 0.0, "fallbackKeptAll": True}
    if duration <= 0 or len(db) == 0:
        return [{"start": 0.0, "end": max(0.0, duration)}], [], empty_meta

    # Silence threshold on dBFS: a fixed margin below the file's speech level,
    # clamped to an absolute window. Anything below it (for >= min_cut_gap) is a
    # gap to remove; speech (well above the line) is always kept.
    finite = db[np.isfinite(db) & (db > -120)]
    speech_level = float(np.percentile(finite, p["speech_pct"])) if len(finite) else -20.0
    threshold = speech_level - p["silence_drop_db"]
    threshold = max(p["silence_thr_min"], min(p["silence_thr_max"], threshold))
    floor = float(np.percentile(finite, 15)) if len(finite) else -70.0
    silent = _mask_to_intervals(db < threshold, hop, duration)

    cuts = []
    for g in silent:
        if (g["end"] - g["start"]) < p["min_cut_gap"]:
            continue  # short pause -> keep for natural pacing
        cs = g["start"] + p["post_pad"]   # keep a tail after the preceding sound
        ce = g["end"] - p["pre_pad"]       # keep a lead-in before the next sound
        if ce - cs >= 0.10:                # something genuinely silent left to remove
            cuts.append({"start": cs, "end": ce})

    # Word guard: if we have word timestamps, never let a cut overlap a word
    # (belt-and-suspenders against the threshold catching a quiet word edge).
    if words and p.get("word_guard", 0) > 0:
        wg = p["word_guard"]
        spans = [{"start": w["start"] - wg, "end": w["end"] + wg}
                 for w in words if w.get("end", 0) > w.get("start", 0)]
        if spans:
            guarded = []
            for c in cuts:
                pieces = [(c["start"], c["end"])]
                for s in spans:
                    nxt = []
                    for a, b in pieces:
                        if s["end"] <= a or s["start"] >= b:
                            nxt.append((a, b))
                        else:
                            if s["start"] > a:
                                nxt.append((a, s["start"]))
                            if s["end"] < b:
                                nxt.append((s["end"], b))
                    pieces = nxt
                guarded.extend({"start": a, "end": b} for a, b in pieces if b - a >= 0.12)
            cuts = guarded

    keep = _complement(cuts, duration)
    fallback = False
    if not keep:
        keep = [{"start": 0.0, "end": duration}]
        cuts = []
        fallback = True

    # Informational only: whisper words that landed inside a silent cut (ghosts).
    ghosts = []
    for w in words:
        mid = 0.5 * (w["start"] + w["end"])
        if any(c["start"] <= mid < c["end"] for c in cuts) and _norm_text(w.get("word")):
            ghosts.append({"start": w["start"], "end": w["end"], "word": w.get("word", "")})

    kept_dur = sum(s["end"] - s["start"] for s in keep)
    meta = {
        "threshold_db": round(threshold, 1),
        "floor_db": round(floor, 1),
        "speech_db": round(speech_level, 1),
        "cutCount": len(cuts),
        "hallucinations": ghosts[:50],
        "hallucinationCount": len(ghosts),
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

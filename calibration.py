#!/usr/bin/env python3
"""Per-channel calibration profiles that learn from the user's corrections.

When the user drags / toggles segments in the review UI and saves, we diff the
corrected keep-segments against what the detector proposed and nudge the
profile: padding, gate sensitivity, and minimum-cut-gap. Conservative, damped,
clamped, and fully explainable (every save records what changed and why).

Profiles live in calibration/<channel>.json next to this file. Writes are
atomic (temp file + os.replace) and serialised under a lock, since the server
is multi-threaded.
"""

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from speech_detect import DEFAULT_PROFILE, merged_profile

CALIB_DIR = Path(__file__).parent / "calibration"
_LOCK = threading.Lock()

# Tunable params the learner is allowed to touch, with hard clamps.
CLAMPS = {
    "pre_pad": (0.0, 0.6),
    "post_pad": (0.0, 0.8),
    "speech_margin_db": (3.0, 20.0),
    "min_cut_gap": (0.2, 2.0),
}
DAMPING = 0.5  # apply half of the observed correction, so we converge instead of oscillate


def _safe_channel(channel):
    c = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(channel or "default")).strip("_")
    return c or "default"


def _path(channel):
    return CALIB_DIR / f"{_safe_channel(channel)}.json"


def load_raw(channel):
    p = _path(channel)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            # Don't silently wipe learned data: preserve the corrupt file so it
            # can be recovered, then start fresh.
            try:
                p.rename(p.with_suffix(".json.corrupt"))
            except Exception:
                pass
            data = {}
    data.setdefault("channel", _safe_channel(channel))
    data.setdefault("overrides", {})
    data.setdefault("samples", 0)
    data.setdefault("history", [])
    return data


def load_profile(channel):
    """Return the effective profile (defaults + learned overrides) for detection."""
    return merged_profile(load_raw(channel).get("overrides", {}))


def save_raw(channel, data):
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    target = _path(channel)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, target)  # atomic on POSIX


def list_profiles():
    if not CALIB_DIR.exists():
        return []
    out = []
    for f in sorted(CALIB_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            out.append({"channel": d.get("channel", f.stem), "samples": d.get("samples", 0)})
        except Exception:
            continue
    return out


def _overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _len(s):
    return max(0.0, s["end"] - s["start"])


def _subtract(a_list, b_list):
    """Return the parts of intervals in a_list not covered by any interval in b_list."""
    res = []
    for a in a_list:
        pieces = [(a["start"], a["end"])]
        for b in b_list:
            nxt = []
            for s, e in pieces:
                if b["end"] <= s or b["start"] >= e:
                    nxt.append((s, e))
                    continue
                if b["start"] > s:
                    nxt.append((s, b["start"]))
                if b["end"] < e:
                    nxt.append((b["end"], e))
            pieces = nxt
        res.extend({"start": s, "end": e} for s, e in pieces if e - s > 1e-6)
    return res


def _clamp(key, value):
    lo, hi = CLAMPS[key]
    return max(lo, min(hi, value))


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def learn_from_corrections(channel, proposed, corrected, duration=None, note=None):
    """Update the channel profile from one reviewed clip.

    proposed / corrected are lists of {start,end} keep-segments. Returns a
    human-readable summary of what changed.
    """
    with _LOCK:
        return _learn_locked(channel, proposed or [], corrected or [], duration, note)


def _learn_locked(channel, proposed, corrected, duration, note):
    data = load_raw(channel)
    eff = merged_profile(data.get("overrides", {}))
    ov = dict(data.get("overrides", {}))

    # Pad deltas: learn ONLY from same-segment boundary *nudges*. Greedy
    # one-to-one match, require the pair to be similar in extent (so a restored
    # gap that merged two keeps isn't read as a 5s tail extension), and ignore
    # large moves (those are restructures, handled by the disagreement math below).
    NUDGE = 1.0
    claimed = set()
    start_deltas, end_deltas = [], []
    for ps in sorted(proposed, key=lambda s: s["start"]):
        best, best_ov = None, 0.0
        for j, cs in enumerate(corrected):
            if j in claimed:
                continue
            o = _overlap(ps, cs)
            if o > best_ov:
                best_ov, best = o, j
        if best is None:
            continue
        cs = corrected[best]
        if best_ov < 0.6 * max(_len(ps), _len(cs)):
            continue  # not the "same" segment — a merge/split, not a nudge
        claimed.add(best)
        sd, ed = cs["start"] - ps["start"], cs["end"] - ps["end"]
        if abs(sd) <= NUDGE:
            start_deltas.append(sd)
        if abs(ed) <= NUDGE:
            end_deltas.append(ed)

    # Disagreement via timeline interval math (robust to merges/splits):
    #  - time kept by user but cut by me  -> I was too aggressive (false cut)
    #  - time kept by me but cut by user  -> I was too lenient   (false keep)
    false_cut_regions = [r for r in _subtract(corrected, proposed) if _len(r) > 0.3]
    false_keep_regions = [r for r in _subtract(proposed, corrected) if _len(r) > 0.3]
    false_cuts = len(false_cut_regions)
    false_keeps = len(false_keep_regions)

    changes = []

    if start_deltas:
        d = _median(start_deltas)  # negative => user wants earlier start => more lead-in
        if abs(d) >= 0.03:
            new = _clamp("pre_pad", eff["pre_pad"] - DAMPING * d)
            if abs(new - eff["pre_pad"]) >= 0.005:
                ov["pre_pad"] = round(new, 3)
                changes.append(f"pre_pad {eff['pre_pad']:.2f}->{new:.2f}s (lead-in)")

    if end_deltas:
        d = _median(end_deltas)  # positive => user wants later end => more tail
        if abs(d) >= 0.03:
            new = _clamp("post_pad", eff["post_pad"] + DAMPING * d)
            if abs(new - eff["post_pad"]) >= 0.005:
                ov["post_pad"] = round(new, 3)
                changes.append(f"post_pad {eff['post_pad']:.2f}->{new:.2f}s (word tails)")

    # Net the two sensitivity effects into a single margin adjustment so they
    # don't clobber each other (false cuts lower the margin, false keeps raise it).
    margin_delta = -1.0 * min(false_cuts, 3) + 0.7 * min(false_keeps, 3)
    if abs(margin_delta) >= 0.05:
        new_m = _clamp("speech_margin_db", eff["speech_margin_db"] + margin_delta)
        if abs(new_m - eff["speech_margin_db"]) >= 0.1:
            ov["speech_margin_db"] = round(new_m, 2)
            direction = "more sensitive" if margin_delta < 0 else "stricter"
            changes.append(
                f"{direction}: margin {eff['speech_margin_db']:.1f}->{new_m:.1f}dB "
                f"({false_cuts} restored, {false_keeps} deleted)")
    if false_cuts:
        new_g = _clamp("min_cut_gap", eff["min_cut_gap"] + 0.05 * min(false_cuts, 3))
        if abs(new_g - eff["min_cut_gap"]) >= 0.01:
            ov["min_cut_gap"] = round(new_g, 3)
            changes.append(f"min_cut_gap ->{new_g:.2f}s")
    if false_keeps and not eff.get("require_words", True):
        ov["require_words"] = True
        changes.append("require_words on")

    summary = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "proposedSegments": len(proposed),
        "correctedSegments": len(corrected),
        "falseCutsRestored": false_cuts,
        "falseKeepsDeleted": false_keeps,
        "medianStartDelta": round(_median(start_deltas), 3) if start_deltas else 0,
        "medianEndDelta": round(_median(end_deltas), 3) if end_deltas else 0,
        "changes": changes or ["no change (proposal matched your edit)"],
    }

    data["overrides"] = ov
    data["samples"] = data.get("samples", 0) + 1
    data["history"] = (data.get("history", []) + [summary])[-50:]
    data["updated"] = summary["when"]
    save_raw(channel, data)
    return summary

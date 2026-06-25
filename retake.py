#!/usr/bin/env python3
"""Retake / double-take detection and best-take selection.

When a solo creator flubs a line and says it again, the transcript contains
near-duplicate sentences close together. We:
  1. group whisper word-timestamps into sentences,
  2. cluster consecutive near-duplicate sentences = takes of the same line,
  3. score each take and keep the best (recency-biased; the last clean take
     usually wins), cutting the losing takes,
  4. flag genuinely close calls so a human can pick.

Pure text + timing (no audio model needed). Operates on the same word list the
silence tool already produces (transcribe_words_local).
"""

import re
from difflib import SequenceMatcher

# Conservative — only unambiguous fillers, so we don't penalise normal speech.
FILLERS = {"um", "uh", "uhm", "umm", "erm", "er", "ah", "mhm", "hmm"}


def _norm_tokens(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _mk_sentence(ws):
    text = re.sub(r"\s+", " ", " ".join(w.get("word", "").strip() for w in ws)).strip()
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": text,
            "tokens": _norm_tokens(text), "nwords": len(ws)}


def sentences_from_words(words, max_gap=0.8):
    """Split the word stream into sentences on terminal punctuation, with a
    pause fallback (gap > max_gap)."""
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        text = (w.get("word") or "").strip()
        end_punct = text.endswith((".", "!", "?"))
        gap_next = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 1e9
        if end_punct or gap_next > max_gap:
            if cur:
                sents.append(_mk_sentence(cur))
            cur = []
    if cur:
        sents.append(_mk_sentence(cur))
    return sents


def _sim(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_retake_groups(sents, sim_threshold=0.72, window=3, min_tokens=3):
    """Cluster near-duplicate sentences that occur within `window` sentences of
    each other = repeated takes of the same line."""
    n = len(sents)
    used = [False] * n
    groups = []
    for i in range(n):
        if used[i] or len(sents[i]["tokens"]) < min_tokens:
            continue
        group = [i]
        j = i + 1
        while j < n and j <= group[-1] + window:
            if not used[j] and len(sents[j]["tokens"]) >= min_tokens:
                if max(_sim(sents[k]["tokens"], sents[j]["tokens"]) for k in group) >= sim_threshold:
                    group.append(j)
                    used[j] = True
            j += 1
        if len(group) > 1:
            for k in group:
                used[k] = True
            groups.append(group)
    return groups


def score_take(sent):
    """Higher = better. Penalise fillers + incompleteness, mildly reward length."""
    toks = sent["tokens"]
    n = max(1, len(toks))
    fillers = sum(1 for t in toks if t in FILLERS)
    complete = sent["text"].rstrip().endswith((".", "!", "?"))
    score = 0.0
    score -= (fillers / n) * 2.0
    score += 0.5 if complete else 0.0
    score += min(n, 40) / 40 * 0.3
    return score, {"fillers": fillers, "complete": complete, "tokens": n}


def select_takes(groups, sents, recency_bonus=0.25, ambiguous_margin=0.12):
    """For each retake group pick the keeper. Later takes get a recency nudge
    (you usually re-record because the earlier one was wrong). Close calls are
    flagged ambiguous for a human pick."""
    choices = []
    for g in groups:
        scored = []
        for rank, idx in enumerate(g):
            s, flags = score_take(sents[idx])
            s += recency_bonus * (rank / max(1, len(g) - 1))
            scored.append((s, idx, flags))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        ambiguous = len(scored) > 1 and (best[0] - scored[1][0]) < ambiguous_margin
        choices.append({
            "group": g,
            "keep": best[1],
            "cut": [i for i in g if i != best[1]],
            "ambiguous": ambiguous,
            "candidates": [{"idx": i, "score": round(s, 3), "text": sents[i]["text"],
                            "start": sents[i]["start"], "end": sents[i]["end"]} for s, i, _ in scored],
        })
    return choices


def retake_cuts(sents, choices, pad=0.10):
    """Time spans to remove (the losing takes)."""
    cuts = []
    for c in choices:
        for idx in c["cut"]:
            s = sents[idx]
            cuts.append({"start": max(0.0, s["start"] - pad), "end": s["end"] + pad})
    return sorted(cuts, key=lambda x: x["start"])


def analyze_retakes(words, sim_threshold=0.72, window=3):
    """Convenience: words -> {sentences, groups, choices, cuts, ambiguous}."""
    sents = sentences_from_words(words)
    groups = find_retake_groups(sents, sim_threshold=sim_threshold, window=window)
    choices = select_takes(groups, sents)
    cuts = retake_cuts(sents, [c for c in choices if not c["ambiguous"]])
    return {
        "sentences": sents,
        "groups": groups,
        "choices": choices,
        "autoCuts": cuts,
        "ambiguous": [c for c in choices if c["ambiguous"]],
        "autoResolved": [c for c in choices if not c["ambiguous"]],
    }

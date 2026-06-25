#!/usr/bin/env python3
"""Render an EDL from original footage with transcript-aware speech cuts.

This is intended for polished talking-head/tutorial exports where pure
volume-based silence detection can clip quiet word tails or keep cough-only
gaps. It maps existing EDL timings from `- nosilence` files back to the
original source footage, transcribes the selected ranges, and keeps only
word-backed speech regions with a small protective tail after each word group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


@dataclass
class Span:
    start: float
    end: float


@dataclass
class RenderSegment:
    source: str
    start: float
    end: float
    label: str


def parse_time(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time value: {value}")


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def format_gap(seconds: float | None) -> str:
    if seconds is None:
        return "source switch"
    return f"{seconds:+.2f}s"


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=capture, text=True, check=True)


def probe_duration(filepath: str) -> float:
    result = run(
        [
            FFPROBE,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            filepath,
        ]
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def detect_silences(filepath: str, threshold: float = -35, min_duration: float = 0.50) -> tuple[list[Span], float]:
    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            filepath,
            "-af",
            f"silencedetect=noise={threshold}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )

    silences: list[Span] = []
    current_start: float | None = None
    for line in result.stderr.splitlines():
        if "silence_start" in line:
            match = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if match:
                current_start = max(float(match.group(1)), 0.0)
        elif "silence_end" in line:
            match = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if match and current_start is not None:
                end = float(match.group(1))
                if end > current_start:
                    silences.append(Span(current_start, end))
                current_start = None

    total = probe_duration(filepath)
    if current_start is not None:
        silences.append(Span(current_start, total))
    return silences, total


def invert_silences(silences: Iterable[Span], total: float, padding: float = 0.20, merge_gap: float = 0.15) -> list[Span]:
    segments: list[Span] = []
    cursor = 0.0
    for silence in silences:
        keep_end = min(silence.start + padding, total)
        keep_start = max(silence.end - padding, 0.0)
        if cursor < keep_end:
            segments.append(Span(cursor, keep_end))
        cursor = max(keep_start, cursor)

    if cursor < total:
        segments.append(Span(cursor, total))

    return merge_spans(segments, merge_gap)


def merge_spans(spans: Iterable[Span], merge_gap: float) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans, key=lambda item: item.start):
        if span.end <= span.start:
            continue
        if merged and span.start - merged[-1].end <= merge_gap:
            merged[-1].end = max(merged[-1].end, span.end)
        else:
            merged.append(Span(span.start, span.end))
    return merged


def original_for_source(source: str) -> str:
    path = Path(source)
    marker = " - nosilence"
    if marker in path.stem:
        candidate = path.with_name(path.stem.replace(marker, "") + path.suffix)
        if candidate.exists():
            return str(candidate)
    return source


def map_nosilence_range_to_original(source: str, start: float, end: float, mapping_cache: dict[str, list[Span]]) -> tuple[str, list[Span]]:
    original = original_for_source(source)
    if original == source:
        return original, [Span(start, end)]

    if original not in mapping_cache:
        silences, total = detect_silences(original)
        mapping_cache[original] = invert_silences(silences, total)

    original_spans: list[Span] = []
    cursor = 0.0
    for span in mapping_cache[original]:
        span_length = span.end - span.start
        timeline_start = cursor
        timeline_end = cursor + span_length
        overlap_start = max(start, timeline_start)
        overlap_end = min(end, timeline_end)
        if overlap_end > overlap_start:
            original_start = span.start + (overlap_start - timeline_start)
            original_end = span.start + (overlap_end - timeline_start)
            original_spans.append(Span(original_start, original_end))
        cursor = timeline_end

    if not original_spans:
        raise ValueError(
            f"Could not map {Path(source).name} range {format_time(start)}-{format_time(end)} back to original footage"
        )

    return original, merge_spans(original_spans, 0.03)


def extract_audio(source: str, start: float, end: float, outpath: Path) -> None:
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(outpath),
        ]
    )


def transcribe_words(model, wav_path: Path, offset: float) -> list[Span]:
    segments, _info = model.transcribe(
        str(wav_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    words: list[Span] = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            start = offset + max(0.0, float(word.start))
            end = offset + max(float(word.end), float(word.start) + 0.02)
            if end > start:
                words.append(Span(start, end))
    return words


def speech_segments_for_interval(
    *,
    model,
    source: str,
    interval: Span,
    tmpdir: Path,
    pre_pad: float,
    post_pad: float,
    max_word_gap: float,
    boundary_pad: float,
    fallback_if_empty: bool,
) -> list[Span]:
    duration = probe_duration(source)
    extract_start = max(0.0, interval.start - boundary_pad)
    extract_end = min(duration, interval.end + boundary_pad)
    if extract_end <= extract_start:
        return []

    wav_path = tmpdir / f"chunk-{abs(hash((source, interval.start, interval.end))) % 10_000_000_000}.wav"
    extract_audio(source, extract_start, extract_end, wav_path)
    words = transcribe_words(model, wav_path, extract_start)

    wanted_start = interval.start - boundary_pad
    wanted_end = interval.end + boundary_pad
    words = [word for word in words if word.end >= wanted_start and word.start <= wanted_end]
    if not words:
        return [interval] if fallback_if_empty else []

    grouped: list[Span] = []
    group_start = words[0].start
    group_end = words[0].end
    for word in words[1:]:
        if word.start - group_end > max_word_gap:
            grouped.append(Span(group_start, group_end))
            group_start = word.start
            group_end = word.end
        else:
            group_end = max(group_end, word.end)
    grouped.append(Span(group_start, group_end))

    padded = [
        Span(
            max(extract_start, group.start - pre_pad),
            min(extract_end, group.end + post_pad),
        )
        for group in grouped
    ]
    return merge_spans(padded, 0.15)


def read_edl(edl_path: Path) -> list[dict[str, str]]:
    with edl_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_file", "start", "end"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"EDL is missing required columns: {', '.join(sorted(missing))}")
    return rows


def build_render_segments(args) -> list[RenderSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("faster-whisper is required for speech-safe rendering") from exc

    rows = read_edl(Path(args.edl))
    mapping_cache: dict[str, list[Span]] = {}
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    render_segments: list[RenderSegment] = []

    with tempfile.TemporaryDirectory(prefix="speech-safe-edl-") as tmp:
        tmpdir = Path(tmp)
        for row in rows:
            source = row["source_file"]
            start = parse_time(row["start"])
            end = parse_time(row["end"])
            if end <= start:
                continue
            original, mapped_intervals = map_nosilence_range_to_original(source, start, end, mapping_cache)
            label = row.get("section") or row.get("note") or row.get("order") or Path(original).stem

            row_spans: list[Span] = []
            for interval in mapped_intervals:
                spans = speech_segments_for_interval(
                    model=model,
                    source=original,
                    interval=interval,
                    tmpdir=tmpdir,
                    pre_pad=args.pre_pad,
                    post_pad=args.post_pad,
                    max_word_gap=args.max_word_gap,
                    boundary_pad=args.boundary_pad,
                    fallback_if_empty=False,
                )
                row_spans.extend(spans)

            if not row_spans:
                row_spans = mapped_intervals

            for span in row_spans:
                if span.end - span.start < args.min_segment:
                    continue
                new_segment = RenderSegment(original, span.start, span.end, label)
                if (
                    render_segments
                    and render_segments[-1].source == new_segment.source
                    and new_segment.start - render_segments[-1].end <= 0.05
                ):
                    render_segments[-1].end = max(render_segments[-1].end, new_segment.end)
                else:
                    render_segments.append(new_segment)

    return render_segments


def write_segments_json(segments: list[RenderSegment], path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "source": segment.source,
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.end - segment.start,
                    "label": segment.label,
                }
                for segment in segments
            ],
            indent=2,
        )
    )


def read_segments_json(path: Path) -> list[RenderSegment]:
    data = json.loads(path.read_text())
    segments = [
        RenderSegment(
            source=item["source"],
            start=float(item["start"]),
            end=float(item["end"]),
            label=item.get("label", Path(item["source"]).stem),
        )
        for item in data
    ]
    return [segment for segment in segments if segment.end > segment.start]


def join_risks(left: RenderSegment, right: RenderSegment, short_segment_warning: float) -> list[str]:
    risks: list[str] = []
    if left.source != right.source:
        risks.append("source-switch")
    if left.end - left.start < short_segment_warning:
        risks.append("short-before")
    if right.end - right.start < short_segment_warning:
        risks.append("short-after")
    if left.source == right.source and right.start - left.end > 20:
        risks.append("large-source-gap")
    return risks


def write_join_checks(
    segments: list[RenderSegment],
    path: Path,
    *,
    short_segment_warning: float,
) -> int:
    output_cursor = 0.0
    rows: list[dict[str, str]] = []

    for index in range(len(segments) - 1):
        left = segments[index]
        right = segments[index + 1]
        output_cursor += left.end - left.start
        source_gap = right.start - left.end if left.source == right.source else None
        rows.append(
            {
                "join": str(index + 1),
                "output_time": format_time(output_cursor),
                "before_source": Path(left.source).name,
                "before_at": format_time(left.end),
                "after_source": Path(right.source).name,
                "after_at": format_time(right.start),
                "source_gap": format_gap(source_gap),
                "labels": f"{left.label} / {right.label}",
                "review": ", ".join(join_risks(left, right, short_segment_warning)) or "normal",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Join Review",
        "",
        f"- Segments: {len(segments)}",
        f"- Joins: {len(rows)}",
        f"- Review priority: rows marked short-before, short-after, source-switch, or large-source-gap.",
        "",
        "| # | Output time | Before source | Before at | After source | After at | Source gap | Review | Labels |",
        "|---:|---|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {join} | {output_time} | {before_source} | {before_at} | {after_source} | {after_at} | "
            "{source_gap} | {review} | {labels} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")
    return sum(1 for row in rows if row["review"] != "normal")


def render(segments: list[RenderSegment], output: Path, workdir: Path, video_bitrate: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    sources: list[str] = []
    for segment in segments:
        if segment.source not in sources:
            sources.append(segment.source)
    source_index = {source: index for index, source in enumerate(sources)}

    filter_lines: list[str] = []
    concat_inputs = []
    for index, segment in enumerate(segments):
        input_index = source_index[segment.source]
        filter_lines.append(
            f"[{input_index}:v]trim=start={segment.start:.4f}:end={segment.end:.4f},"
            "setpts=PTS-STARTPTS,"
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            "fps=30,setsar=1,format=yuv420p"
            f"[v{index}]"
        )
        filter_lines.append(
            f"[{input_index}:a]atrim=start={segment.start:.4f}:end={segment.end:.4f},"
            "asetpts=PTS-STARTPTS,"
            "aresample=48000,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
            f"[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")

    filter_lines.append("".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=1[cv][ca]")
    filter_lines.append("[cv]copy[vout]")
    filter_lines.append(
        "[ca]loudnorm=I=-16:TP=-1.5:LRA=11,"
        "aresample=48000,"
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        "[aout]"
    )

    filter_path = workdir / f"{output.stem}-filtergraph.txt"
    filter_path.write_text(";\n".join(filter_lines) + "\n")

    command = [FFMPEG, "-hide_banner", "-y"]
    for source in sources:
        command.extend(["-i", source])
    command.extend(
        [
            "-filter_complex_script",
            str(filter_path),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            video_bitrate,
            "-maxrate",
            "9000k",
            "-bufsize",
            "14000k",
            "-tag:v",
            "avc1",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an EDL using transcript-aware speech-safe cuts.")
    parser.add_argument("--edl", default=None, help="CSV edit-decision list with source_file,start,end columns")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--workdir", default=None, help="Directory for render sidecar files")
    parser.add_argument("--segments-json", default=None, help="Optional JSON path for resolved source segments")
    parser.add_argument("--from-segments-json", default=None, help="Render directly from a previously resolved segment JSON")
    parser.add_argument("--join-checks", default=None, help="Markdown path for exact join timestamps to spot-check")
    parser.add_argument("--no-join-checks", action="store_true", help="Do not write the join-review sidecar")
    parser.add_argument("--short-segment-warning", type=float, default=1.0)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--pre-pad", type=float, default=0.08)
    parser.add_argument("--post-pad", type=float, default=0.36)
    parser.add_argument("--max-word-gap", type=float, default=0.58)
    parser.add_argument("--boundary-pad", type=float, default=0.45)
    parser.add_argument("--min-segment", type=float, default=0.08)
    parser.add_argument("--video-bitrate", default="7000k")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.from_segments_json and not args.edl:
        parser.error("--edl is required unless --from-segments-json is provided")

    if args.from_segments_json:
        segments = read_segments_json(Path(args.from_segments_json))
    else:
        segments = build_render_segments(args)
    total = sum(segment.end - segment.start for segment in segments)
    print(f"Resolved {len(segments)} speech-safe segments, total {total:.1f}s ({format_time(total)})")

    output = Path(args.output)
    workdir = Path(args.workdir) if args.workdir else output.parent
    segments_json = Path(args.segments_json) if args.segments_json else workdir / f"{output.stem}-segments.json"
    write_segments_json(segments, segments_json)
    print(f"Wrote segment map: {segments_json}")

    if not args.no_join_checks:
        join_checks = Path(args.join_checks) if args.join_checks else workdir / f"{output.stem}-join-checks.md"
        priority_count = write_join_checks(
            segments,
            join_checks,
            short_segment_warning=args.short_segment_warning,
        )
        print(f"Wrote join review: {join_checks}")
        if priority_count:
            print(f"Join review priority rows: {priority_count}")

    if args.dry_run:
        return

    render(segments, output, workdir, args.video_bitrate)
    size = os.path.getsize(output) / 1024 / 1024
    print(f"Wrote {output} ({size:.1f} MB)")


if __name__ == "__main__":
    main()

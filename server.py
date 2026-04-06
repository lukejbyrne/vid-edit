#!/usr/bin/env python3
"""Silence Remover — local server using native FFmpeg."""

import json
import os
import re
import subprocess
import tempfile
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs

UPLOAD_DIR = tempfile.mkdtemp(prefix="silence_remover_")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/upload":
            self.handle_upload()
        elif self.path == "/api/detect":
            self.handle_detect()
        elif self.path == "/api/process":
            self.handle_process()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/api/download/"):
            self.handle_download()
        else:
            super().do_GET()

    def handle_upload(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        # Save uploaded file
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        with open(filepath, "wb") as f:
            f.write(body)

        # Get duration via ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
                capture_output=True, text=True
            )
            info = json.loads(result.stdout)
            duration = float(info["format"]["duration"])
            size = os.path.getsize(filepath)
            self.send_json({"ok": True, "duration": duration, "size": size})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def handle_detect(self):
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))

        threshold = params.get("threshold", -35)
        min_duration = params.get("minDuration", 0.5)

        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return

        # Run FFmpeg silencedetect
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", filepath,
                    "-af", f"silencedetect=noise={threshold}dB:d={min_duration}",
                    "-f", "null", "-"
                ],
                capture_output=True, text=True
            )
            stderr = result.stderr

            # Parse silence_start and silence_end
            silences = []
            starts = re.findall(r"silence_start:\s*([\d.]+)", stderr)
            ends = re.findall(r"silence_end:\s*([\d.]+)", stderr)

            for i, start in enumerate(starts):
                end = ends[i] if i < len(ends) else None
                silences.append({
                    "start": float(start),
                    "end": float(end) if end else None
                })

            # Get total duration
            dur_result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
                capture_output=True, text=True
            )
            total_duration = float(json.loads(dur_result.stdout)["format"]["duration"])

            # Handle last silence that may not have an end
            for s in silences:
                if s["end"] is None:
                    s["end"] = total_duration

            self.send_json({
                "ok": True,
                "silences": silences,
                "totalDuration": total_duration
            })
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def handle_process(self):
        content_length = int(self.headers["Content-Length"])
        params = json.loads(self.rfile.read(content_length))

        segments = params.get("segments", [])  # non-silent segments to keep
        filepath = os.path.join(UPLOAD_DIR, "input.mp4")
        outpath = os.path.join(UPLOAD_DIR, "output.mp4")

        if not os.path.exists(filepath):
            self.send_json({"ok": False, "error": "No file uploaded"}, 400)
            return

        if not segments:
            self.send_json({"ok": False, "error": "No segments to keep"}, 400)
            return

        try:
            # Build filter_complex
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

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", filepath,
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    outpath
                ],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                self.send_json({"ok": False, "error": result.stderr[-500:]}, 500)
                return

            size = os.path.getsize(outpath)
            self.send_json({"ok": True, "size": size, "downloadUrl": "/api/download/output.mp4"})

        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def handle_download(self):
        filename = self.path.split("/")[-1]
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(os.path.getsize(filepath)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

        with open(filepath, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = 8080
    print(f"Silence Remover running at http://localhost:{port}")
    print(f"Temp dir: {UPLOAD_DIR}")
    server = HTTPServer(("", port), Handler)
    server.serve_forever()

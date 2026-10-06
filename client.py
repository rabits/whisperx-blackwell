#!/usr/bin/env python3
"""Send one audio file to the WhisperX service and write the JSON response.

SRT conversion is done by json_to_srt.py when that script sits next to this
one. The command line stays the same either way: --srt still names the
subtitle file, and an empty --srt skips it.

Examples:
  ./client.py --svc_url http://ai-01.psa:8003/ in.mp3
  ./client.py --svc_url http://ai-01.psa:8003/ --language ru in.webm
  ./client.py --svc_url http://ai-01.psa:8003/ --json '' --srt out.srt in.webm
  ./client.py --svc_url http://ai-01.psa:8003/ --hallucinations '' in.webm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlparse


def default_output(audio_path: str, suffix: str) -> str:
    return audio_path + suffix


def log(message: str) -> None:
    """Write one client log line, prefixed with the local time."""

    stamp = time.strftime("%H:%M:%S")
    lines = str(message).splitlines() or [""]
    for line in lines:
        print(f"{stamp} {line}", file=sys.stderr)


def service_root(svc_url: str) -> str:
    """Accept a service root or a full /transcribe URL."""

    raw = svc_url.strip()
    if not raw:
        raise SystemExit("--svc_url is required")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc:
        raise SystemExit(f"Not a usable service URL: {svc_url}")
    path = parsed.path.rstrip("/")
    for suffix in ("/transcribe", "/progress", "/result"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return parsed._replace(path=path, params="", query="", fragment="").geturl().rstrip("/")


def endpoint_url(svc_url: str) -> str:
    return service_root(svc_url) + "/transcribe"


def progress_log_lines(progress: dict, seen: dict) -> list[str]:
    """New execution-log lines since the previous poll.

    seen maps a stage name to the last 5% bucket that was printed.
    """

    status = progress.get("status")
    if status == "queued":
        if seen.get("_queued") == 1:
            return []
        seen["_queued"] = 1
        return ["queued"]
    lines = []
    current = progress.get("stage")
    for stage in progress.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name") or "")
        label = str(stage.get("label") or name)
        percent = float(stage.get("percent") or 0)
        if percent <= 0 and name != current:
            continue
        bucket = 100 if percent >= 100 else int(percent // 5) * 5
        if seen.get(name) == bucket:
            continue
        seen[name] = bucket
        lines.append(f"{label:<22} {percent:6.1f}%")
    return lines


def normalize_phrase(text: str) -> str:
    """Casefold and drop punctuation so list entries match Whisper text."""

    folded = text.casefold()
    return " ".join(re.sub(r"[^\w\s]+", " ", folded, flags=re.UNICODE).split())


def hallucination_keys(blob: str) -> tuple[set[str], set[str]]:
    """Return normalized phrases and exact casefolded lines from a newline list."""

    normalized: set[str] = set()
    exact: set[str] = set()
    for line in blob.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        exact.add(" ".join(raw.split()).casefold())
        key = normalize_phrase(raw)
        if key:
            normalized.add(key)
    return normalized, exact


def is_hallucination(text: str, normalized: set[str], exact: set[str]) -> bool:
    stripped = " ".join(str(text).split())
    if not stripped:
        return False
    if stripped.casefold() in exact:
        return True
    key = normalize_phrase(stripped)
    return bool(key) and key in normalized


def default_hallucination_list() -> str | None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper-hallucinations-ru.lst")
    if os.path.isfile(path):
        return path
    return None


def filter_payload(payload: dict, blob: str) -> list[str]:
    """Drop matching segments and the words that fall inside them."""

    normalized, exact = hallucination_keys(blob)
    if not normalized and not exact:
        return []
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return []
    kept = []
    dropped_spans: list[tuple[float, float]] = []
    dropped: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            kept.append(segment)
            continue
        text = segment.get("text", "")
        if is_hallucination(text, normalized, exact):
            dropped.append(" ".join(str(text).split()))
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            dropped_spans.append((start, end))
            continue
        kept.append(segment)
    payload["segments"] = kept
    words = payload.get("word_segments")
    if isinstance(words, list) and dropped_spans:

        def inside_dropped(word: object) -> bool:
            if not isinstance(word, dict):
                return False
            start = word.get("start")
            if start is None:
                return False
            end = word.get("end")
            if end is None:
                end = start
            start_s = float(start)
            end_s = float(end)
            return any(start_s < span_end and end_s > span_start for span_start, span_end in dropped_spans)

        payload["word_segments"] = [word for word in words if not inside_dropped(word)]
    if dropped:
        speakers: dict[str, dict[str, float]] = {}
        for segment in kept:
            if not isinstance(segment, dict):
                continue
            speaker = segment.get("speaker") or "UNKNOWN"
            bucket = speakers.setdefault(speaker, {"duration": 0.0, "segments": 0})
            bucket["duration"] += float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))
            bucket["segments"] += 1
        payload["speakers"] = speakers
        payload["num_speakers"] = len(speakers)
        already = payload.get("dropped_hallucinations")
        if not isinstance(already, list):
            already = []
        payload["dropped_hallucinations"] = already + dropped
    return dropped


def converter_path() -> str | None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_to_srt.py")
    if os.path.isfile(path):
        return path
    return None


def write_srt(json_path: str | None, payload: dict | None, srt_path: str) -> int:
    """Convert transcript JSON by running json_to_srt.py next to this script."""

    converter = converter_path()
    if converter is None:
        log("json_to_srt.py was not found next to client.py")
        return 1
    command = [sys.executable, converter, "--srt", srt_path, "--quiet"]
    if json_path is not None:
        command.append(json_path)
        completed = subprocess.run(command)
    else:
        if not isinstance(payload, dict):
            log("Response JSON is not an object")
            return 1
        command.append("-")
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
    if completed.returncode != 0:
        return completed.returncode
    with open(srt_path, encoding="utf-8") as handle:
        srt = handle.read()
    cue_count = len([block for block in srt.split("\n\n") if block.strip()])
    language = payload.get("language") if isinstance(payload, dict) else None
    speakers = payload.get("num_speakers") if isinstance(payload, dict) else None
    log(f"wrote {srt_path} ({cue_count} cues, language={language}, speakers={speakers})")
    return 0


def header_filename(path: str) -> str:
    name = os.path.basename(path).replace('"', "").replace("\r", "").replace("\n", "")
    try:
        name.encode("latin-1")
    except UnicodeEncodeError:
        name = "audio" + os.path.splitext(path)[1]
    return name or "audio"


def post_audio(url: str, audio_path: str, fields: dict[str, str], timeout: float | None) -> tuple[int, bytes]:
    boundary = uuid.uuid4().hex
    parsed = urlparse(url)
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    filename = header_filename(audio_path)
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("latin-1")
    )
    preamble = b"".join(parts)
    epilogue = f"\r\n--{boundary}--\r\n".encode("ascii")
    size = os.path.getsize(audio_path)
    total = len(preamble) + size + len(epilogue)

    connection_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = path + "?" + parsed.query
    connection = connection_cls(parsed.hostname, port, timeout=timeout)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(total))
        connection.putheader("Connection", "close")
        connection.endheaders()
        connection.send(preamble)
        with open(audio_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        connection.send(epilogue)
        response = connection.getresponse()
        body = response.read()
        return response.status, body
    finally:
        connection.close()


def http_request(method: str, url: str, timeout: float | None) -> tuple[int, bytes]:
    parsed = urlparse(url)
    connection_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    path = parsed.path or "/"
    if parsed.query:
        path = path + "?" + parsed.query
    connection = connection_cls(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def poll_until_done(root: str, uid: str, deadline: float | None) -> bytes:
    """Print per-stage percents, then return the finished transcript body."""

    seen: dict = {}
    while True:
        if deadline is not None and time.monotonic() > deadline:
            log(f"Timed out waiting for job {uid}")
            raise SystemExit(1)
        try:
            status, body = http_request("GET", f"{root}/progress/{uid}", 30)
        except OSError as exc:
            log(f"Progress request failed: {exc}")
            raise SystemExit(1) from exc
        if status != 200:
            detail = body.decode("utf-8", errors="replace").strip()
            log(f"Progress returned HTTP {status}")
            if detail:
                log(detail)
            raise SystemExit(1)
        try:
            progress = json.loads(body)
        except json.JSONDecodeError as exc:
            log(f"Progress response is not JSON: {exc}")
            raise SystemExit(1) from exc
        for line in progress_log_lines(progress, seen):
            log(line)
        state = progress.get("status")
        if state == "done":
            break
        if state == "error":
            log(progress.get("detail") or "Job failed")
            raise SystemExit(1)
        time.sleep(0.5)

    try:
        status, body = http_request("GET", f"{root}/result/{uid}", 120)
    except OSError as exc:
        log(f"Result request failed: {exc}")
        raise SystemExit(1) from exc
    if status != 200:
        detail = body.decode("utf-8", errors="replace").strip()
        log(f"Result returned HTTP {status}")
        if detail:
            log(detail)
        raise SystemExit(1)
    return body


def write_text(path: str, content: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def write_bytes(path: str, content: bytes) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file and write the service JSON plus an SRT with speaker prefixes.",
    )
    parser.add_argument("audio", help="Audio file to upload")
    parser.add_argument("--svc_url", required=True, help="Service root, for example http://ai-01.psa:8003/")
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code passed as the language form field. Default: auto (from first 30 sec)",
    )
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact speaker count, when known")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum speaker count")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speaker count")
    parser.add_argument(
        "--json",
        default=None,
        help="Where to write the raw JSON response. Default: AUDIO.json. An empty path skips the file.",
    )
    parser.add_argument(
        "--srt",
        default=None,
        help="Where to write the SRT. Default: AUDIO.srt. An empty path skips the file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait for the job after the upload. Default: wait until the service finishes.",
    )
    parser.add_argument(
        "--hallucinations",
        default=None,
        help=(
            "Text file of phrases to drop, one per line. "
            "Default: whisper-hallucinations-ru.lst next to this script, when that file exists. "
            "An empty path disables filtering."
        ),
    )
    return parser.parse_args(argv)


def resolved_path(override: str | None, audio_path: str, suffix: str) -> str | None:
    if override is None:
        return default_output(audio_path, suffix)
    if override == "":
        return None
    return override


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    audio_path = args.audio
    if not os.path.isfile(audio_path):
        log(f"Audio file not found: {audio_path}")
        return 1

    json_path = resolved_path(args.json, audio_path, ".json")
    srt_path = resolved_path(args.srt, os.path.splitext(audio_path)[0], ".srt")
    if args.hallucinations is None:
        hallucinations_path = default_hallucination_list()
    elif args.hallucinations == "":
        hallucinations_path = None
    else:
        hallucinations_path = args.hallucinations
    hallucination_blob = ""
    if hallucinations_path is not None:
        if not os.path.isfile(hallucinations_path):
            log(f"Hallucination list not found: {hallucinations_path}")
            return 1
        with open(hallucinations_path, encoding="utf-8") as handle:
            hallucination_blob = handle.read()
    url = endpoint_url(args.svc_url)
    fields = {"language": args.language}
    if args.num_speakers is not None:
        fields["num_speakers"] = str(args.num_speakers)
    if args.min_speakers is not None:
        fields["min_speakers"] = str(args.min_speakers)
    if args.max_speakers is not None:
        fields["max_speakers"] = str(args.max_speakers)
    if hallucination_blob.strip():
        fields["hallucinations"] = hallucination_blob

    exact = hallucination_keys(hallucination_blob)[1] if hallucination_blob.strip() else set()
    log(
        f"POST {url} ({os.path.getsize(audio_path)} bytes, language={args.language}, "
        f"hallucinations={len(exact)})"
    )
    # The upload itself should not use the whole-job deadline as a socket timeout.
    try:
        status, body = post_audio(url, audio_path, fields, None)
    except OSError as exc:
        log(f"Request failed: {exc}")
        return 1

    if status == 202:
        try:
            accepted = json.loads(body)
        except json.JSONDecodeError as exc:
            log(f"Accepted response is not JSON: {exc}")
            return 1
        uid = accepted.get("uid")
        if not uid:
            log("Accepted response has no uid")
            return 1
        log(f"job {uid}")
        deadline = None if args.timeout is None else time.monotonic() + args.timeout
        try:
            body = poll_until_done(service_root(args.svc_url), uid, deadline)
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 1
        status = 200
    elif status != 200:
        detail = body.decode("utf-8", errors="replace").strip()
        log(f"Service returned HTTP {status}")
        if detail:
            log(detail)
        return 1

    payload = None
    need_payload = srt_path is not None or json_path is None or bool(hallucination_blob.strip())
    if need_payload:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            log(f"Response is not JSON: {exc}")
            return 1

    client_dropped: list[str] = []
    if isinstance(payload, dict) and hallucination_blob.strip():
        client_dropped = filter_payload(payload, hallucination_blob)
        for phrase in client_dropped:
            log(f"dropped hallucination: {phrase}")
        reported = payload.get("dropped_hallucinations")
        if isinstance(reported, list):
            for phrase in reported:
                if phrase not in client_dropped:
                    log(f"service dropped hallucination: {phrase}")

    if json_path is not None:
        if client_dropped and isinstance(payload, dict):
            write_text(json_path, json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            write_bytes(json_path, body)
        log(f"wrote {json_path}")

    if srt_path is None:
        if isinstance(payload, dict):
            log(
                f"language={payload.get('language')} speakers={payload.get('num_speakers')} "
                f"segments={len(payload.get('segments') or [])}"
            )
        return 0

    return write_srt(json_path, payload if isinstance(payload, dict) else None, srt_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

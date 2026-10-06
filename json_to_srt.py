#!/usr/bin/env python3
"""Convert a WhisperX transcript JSON file to SRT with speaker prefixes.

Each cue is one segment. The text looks like "[SPEAKER_01]: Hello everyone."

Examples:
  ./json_to_srt.py interview.mp3.json
  ./json_to_srt.py interview.mp3.json --srt interview.srt
  ./json_to_srt.py - --srt interview.srt < interview.mp3.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def format_timestamp(seconds: float) -> str:
    millis = int(round(max(0.0, seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cue_text(segment: dict) -> str:
    text = " ".join(str(segment.get("text", "")).split())
    if not text:
        return ""
    speaker = str(segment.get("speaker") or "").strip()
    if speaker:
        return f"[{speaker}]: {text}"
    return text


def segments_to_srt(segments: list) -> str:
    blocks: list[str] = []
    index = 1
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = cue_text(segment)
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end <= start:
            end = start + 0.001
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n"
        )
        index += 1
    if not blocks:
        return ""
    return "\n".join(blocks) + "\n"


def default_srt_path(json_path: str) -> str:
    stem, _ext = os.path.splitext(json_path)
    return stem + ".srt"


def load_payload(source: str) -> dict:
    if source == "-":
        raw = sys.stdin.read()
    else:
        if not os.path.isfile(source):
            raise SystemExit(f"JSON file not found: {source}")
        with open(source, encoding="utf-8") as handle:
            raw = handle.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Response is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Response JSON is not an object")
    return payload


def write_text(path: str, content: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a WhisperX transcript JSON file to SRT with speaker prefixes.",
    )
    parser.add_argument("json", help="Transcript JSON path, or - to read stdin")
    parser.add_argument(
        "--srt",
        default=None,
        help="Where to write the SRT. Default: the JSON path with a .srt suffix. Required when reading stdin.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the written path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.json == "-" and not args.srt:
        print("--srt is required when the JSON is read from stdin", file=sys.stderr)
        return 1
    srt_path = args.srt or default_srt_path(args.json)
    try:
        payload = load_payload(args.json)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(exc.code if isinstance(exc.code, str) else "Could not read JSON", file=sys.stderr)
        return 1
    segments = payload.get("segments")
    if not isinstance(segments, list):
        print("Response JSON has no segments list", file=sys.stderr)
        return 1
    srt = segments_to_srt(segments)
    if not srt:
        print("No subtitle cues in the response", file=sys.stderr)
        return 1
    write_text(srt_path, srt)
    if not args.quiet:
        cue_count = len([block for block in srt.split("\n\n") if block.strip()])
        print(f"wrote {srt_path} ({cue_count} cues)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Functional smoke check for the pinned vLLM OpenAI-compatible server.

This is a FUNCTIONAL smoke check, not a benchmark. It verifies that the real server
responds on the endpoints the router will use; it does NOT measure throughput, TTFT,
or latency percentiles. Raw responses are written to an output directory as evidence.

It uses only the Python standard library so it can run from any interpreter without
adding dependencies to either project environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict[str, Any], timeout: float) -> urllib.request.addinfourl:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def check_models(base_url: str, timeout: float, out_dir: Path) -> str:
    body = _get_json(f"{base_url}/models", timeout)
    (out_dir / "models.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    served = [entry["id"] for entry in body.get("data", [])]
    print(f"[models] served ids: {served}")
    if not served:
        raise SystemExit("no model advertised by /v1/models")
    return served[0]


def check_non_streaming(base_url: str, model: str, timeout: float, out_dir: Path) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }
    with _post(f"{base_url}/chat/completions", payload, timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    (out_dir / "non_streaming.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    content = body["choices"][0]["message"]["content"]
    finish = body["choices"][0].get("finish_reason")
    print(f"[non-stream] finish_reason={finish!r} content={content!r}")
    if not content:
        raise SystemExit("non-streaming response had empty content")


def check_streaming(base_url: str, model: str, timeout: float, out_dir: Path) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to five."}],
        "max_tokens": 48,
        "temperature": 0.0,
        "stream": True,
    }
    raw_lines: list[str] = []
    first_chunk_seen = False
    done_seen = False
    start = time.monotonic()
    first_chunk_after: float | None = None
    with _post(f"{base_url}/chat/completions", payload, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            raw_lines.append(line)
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                done_seen = True
                break
            if not first_chunk_seen:
                first_chunk_seen = True
                first_chunk_after = time.monotonic() - start
    (out_dir / "streaming.sse").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(
        f"[stream] first_chunk_seen={first_chunk_seen} "
        f"first_chunk_after_s={first_chunk_after} done_event={done_seen}"
    )
    if not (first_chunk_seen and done_seen):
        raise SystemExit("streaming response missing first chunk or [DONE] terminator")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out-dir", default="results/raw/week1-step3-smoke")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = check_models(args.base_url, args.timeout, out_dir)
        check_non_streaming(args.base_url, model, args.timeout, out_dir)
        check_streaming(args.base_url, model, args.timeout, out_dir)
    except urllib.error.URLError as exc:
        print(f"smoke check could not reach the server: {exc}", file=sys.stderr)
        return 2
    print(f"[ok] smoke artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

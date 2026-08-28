#!/usr/bin/env python3
"""Make flashinfer-python==0.6.16.post3 importable on CPython 3.11.

Upstream bug: flashinfer/comm/fd_exchange.py annotates a helper with
``array.array[int]`` at module import time, but ``array.array`` only became
subscriptable in CPython 3.12. Because the module lacks
``from __future__ import annotations``, the annotation is evaluated eagerly and
raises ``TypeError: type 'array.array' is not subscriptable`` on Python 3.11.

vLLM 0.27.1 pins this exact flashinfer version and imports the module during
engine warm-up, so a stock single-GPU launch on our pinned Python 3.11.14 crashes
before serving. Rather than drop the pinned dependency (which would desync
``uv pip check``), we insert the missing future-import, which makes the annotation
lazy without touching runtime behaviour.

The patch is idempotent: run it after (re)installing the vLLM environment. It only
acts on Python < 3.12 and only if the file still lacks the future import.
"""

from __future__ import annotations

import sys
from pathlib import Path

FUTURE_IMPORT = "from __future__ import annotations\n"


def _target_file() -> Path:
    # Locate the file WITHOUT importing it: importing flashinfer.comm is exactly
    # what crashes on Python 3.11 before this patch is applied.
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "flashinfer" / "comm" / "fd_exchange.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("flashinfer/comm/fd_exchange.py not found on sys.path")


def patch(path: Path) -> bool:
    """Insert the future-import before the first code line. Returns True if changed."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if any(line.strip() == FUTURE_IMPORT.strip() for line in lines):
        return False
    insert_at = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            insert_at = index
            break
    lines.insert(insert_at, FUTURE_IMPORT)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> int:
    if sys.version_info >= (3, 12):
        print("Python >= 3.12: flashinfer patch not needed.")
        return 0
    try:
        target = _target_file()
    except FileNotFoundError as exc:
        print(f"{exc}; is flashinfer installed in this environment?", file=sys.stderr)
        return 1
    changed = patch(target)
    print(f"{'patched' if changed else 'already patched'}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Claude Code PostToolUse hook — run targeted pytest after V3 edits.

Reads hook input JSON from stdin, extracts the edited file path, and if it
touches v3/ Python code, runs the regression suite. Emits:
  - stdout: compact pass/fail line (shown to user)
  - exit 0 always (non-blocking; we don't want to prevent commits)

Designed to be fast (<10s). Full pytest suite is kept small for this reason.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PYTHON = r"C:\Users\wogus\miniconda3\envs\quant\python.exe"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = REPO_ROOT / "v3" / "tests"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # Silently ignore malformed input

    tool = payload.get("tool_name") or payload.get("tool") or ""
    if tool not in ("Edit", "Write", "MultiEdit"):
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path", "")

    # Only react to V3 Python edits
    try:
        p = Path(file_path).resolve()
        rel = p.relative_to(REPO_ROOT)
    except (ValueError, OSError):
        return 0

    if rel.parts[0] != "v3" or p.suffix != ".py":
        return 0

    # Skip test files themselves (pytest will still run if test edited — that's fine)
    # but don't skip — running tests is exactly what we want when tests change.

    if not TEST_DIR.exists():
        return 0

    result = subprocess.run(
        [PYTHON, "-m", "pytest", str(TEST_DIR), "-q", "--tb=line", "--no-header"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Compact summary line
    out = (result.stdout or "") + (result.stderr or "")
    last = [line for line in out.splitlines() if line.strip()][-1:]
    summary = last[0] if last else "pytest: no output"

    tag = "✓" if result.returncode == 0 else "✗"
    print(f"[v3-evaluator hook] {tag} {summary}", file=sys.stderr)

    # Never block the tool call
    return 0


if __name__ == "__main__":
    sys.exit(main())

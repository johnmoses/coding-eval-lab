"""
sandbox.py — execute untrusted, model-generated code in isolation.

Generated code is treated as HOSTILE by default. A coding-eval harness runs code
it did not write, so the grader must contain it rather than trust it. This module
runs one candidate program in a throwaway subprocess with:

  - a hard wall-clock TIMEOUT (kills infinite loops / while True),
  - a fresh temp working directory, torn down after the run (no state leaks
    between candidates),
  - best-effort NETWORK isolation: a tiny bootstrap stubs out `socket` so code
    that tries to open a connection fails fast instead of phoning home,
  - subprocess isolation: a crash/segfault takes down the child, not the harness.

Scope note (honest): this is process-level isolation suitable for a local lab and
untrusted-but-not-adversarial code. A production grader for truly adversarial input
would add an OS sandbox (containers / seccomp / nsjail), CPU and memory rlimits, and
a read-only filesystem. The seam is here: `run_candidate` is the single choke point
where a stronger backend would slot in.

Returns a structured result the grader can categorize:
  status in {"pass", "fail", "timeout", "error"} plus captured stderr.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Bootstrap prepended to every candidate: neutralize the obvious escape hatches
# before the candidate code runs. Best-effort, not a security boundary on its own.
_BOOTSTRAP = """
import sys as _sys

# --- best-effort network block -------------------------------------------------
try:
    import socket as _socket

    def _blocked(*_a, **_k):
        raise RuntimeError("network access is disabled in the sandbox")

    _socket.socket = _blocked            # type: ignore[assignment]
    _socket.create_connection = _blocked  # type: ignore[assignment]
except Exception:
    pass
# -------------------------------------------------------------------------------
"""


@dataclass
class RunResult:
    status: str          # "pass" | "fail" | "timeout" | "error"
    returncode: int | None
    stderr: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def build_program(candidate_code: str, test_block: str) -> str:
    """Assemble the full program: bootstrap + candidate + tests.

    If every test assertion holds and nothing raises, the program exits 0 ("pass").
    An AssertionError means the code ran but is wrong ("fail"). Any other exception
    means it didn't even run correctly ("error"). A hang is caught as "timeout" by
    the caller.
    """
    return (
        _BOOTSTRAP
        + "\n# --- candidate solution ---\n"
        + candidate_code.strip()
        + "\n\n# --- hidden tests ---\n"
        + test_block.strip()
        + "\n"
    )


def run_candidate(
    candidate_code: str,
    tests: list[str],
    timeout_s: float = 5.0,
) -> RunResult:
    """Run one candidate against its hidden tests in an isolated subprocess."""
    program = build_program(candidate_code, "\n".join(tests))

    # Fresh scratch dir per run; removed on context exit so nothing persists.
    with tempfile.TemporaryDirectory(prefix="codeeval_") as workdir:
        prog_path = Path(workdir) / "candidate.py"
        prog_path.write_text(program)

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(prog_path)],  # -I: isolated mode, ignore env/user site
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return RunResult("timeout", None, f"timed out after {timeout_s}s")

        if proc.returncode == 0:
            return RunResult("pass", 0, proc.stderr.strip())

        # Distinguish "ran but wrong" (AssertionError) from "did not run" (other errors).
        stderr = proc.stderr.strip()
        status = "fail" if "AssertionError" in stderr else "error"
        return RunResult(status, proc.returncode, stderr)

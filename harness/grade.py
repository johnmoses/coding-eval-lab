"""
grade.py — grade generated candidates against hidden tests.

Thin layer over the sandbox: for each task, run every sampled candidate against the
task's hidden tests and record (n, c) — total samples and passing samples — plus a
per-status failure breakdown (fail vs timeout vs error). Those counts are exactly
what passk.py consumes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .sandbox import run_candidate


@dataclass
class TaskResult:
    task_id: str
    n: int                                   # samples generated
    c: int                                   # samples passing all tests
    status_counts: Counter = field(default_factory=Counter)  # pass/fail/timeout/error
    first_failure_stderr: str = ""   # failing assertion line + exception, e.g. "assert f(1) == 2 | AssertionError"
    first_failure_code: str = ""     # the first candidate that failed, for failure-mode inspection

    @property
    def pass_rate(self) -> float:
        return self.c / self.n if self.n else 0.0


def grade_task(
    task_id: str,
    candidates: list[str],
    tests: list[str],
    timeout_s: float = 5.0,
) -> TaskResult:
    counts: Counter = Counter()
    c = 0
    first_fail = ""
    first_fail_code = ""
    for code in candidates:
        res = run_candidate(code, tests, timeout_s=timeout_s)
        counts[res.status] += 1
        if res.passed:
            c += 1
        elif not first_fail_code:
            first_fail_code = code
            if res.stderr:
                # A Python traceback ends with the offending source line, then the
                # exception. Keeping both says WHICH hidden assertion failed.
                tail = [ln.strip() for ln in res.stderr.splitlines()[-2:] if ln.strip()]
                first_fail = " | ".join(tail)[:240]
    return TaskResult(
        task_id=task_id,
        n=len(candidates),
        c=c,
        status_counts=counts,
        first_failure_stderr=first_fail,
        first_failure_code=first_fail_code,
    )

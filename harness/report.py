"""
report.py — turn graded task results into a JSON report + a readable summary.

Reports both aggregate pass@k (across the requested k values) and a per-task
breakdown with a failure-type histogram, so a low score is diagnosable: did the
model produce wrong answers (fail), hang (timeout), or not run at all (error)?
That failure categorization is the "analyze model behavior and failure modes" part
of coding evaluation, not just a single headline number.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .grade import TaskResult
from .passk import aggregate_pass_at_k, bootstrap_pass_at_k_ci, pass_at_k


def build_report(results: list[TaskResult], ks: list[int], meta: dict) -> dict:
    per_task = [(r.n, r.c) for r in results]

    aggregate = {
        f"pass@{k}": round(aggregate_pass_at_k(per_task, k), 4)
        for k in ks
        if all(r.n >= k for r in results)
    }

    # 95% bootstrap intervals (tasks AND samples resampled). A point estimate alone
    # overstates precision, especially with few tasks.
    aggregate_ci = {}
    for k in ks:
        if all(r.n >= k for r in results):
            lo, hi = bootstrap_pass_at_k_ci(per_task, k)
            aggregate_ci[f"pass@{k}"] = [round(lo, 4), round(hi, 4)]

    tasks = []
    for r in results:
        tasks.append(
            {
                "task_id": r.task_id,
                "n": r.n,
                "c": r.c,
                "pass_rate": round(r.pass_rate, 4),
                "pass@1": round(pass_at_k(r.n, r.c, 1), 4),
                "status_counts": dict(r.status_counts),
                "sample_failure": r.first_failure_stderr,
                "sample_failure_code": r.first_failure_code,
            }
        )

    failure_totals: dict[str, int] = {}
    for r in results:
        for status, cnt in r.status_counts.items():
            if status != "pass":
                failure_totals[status] = failure_totals.get(status, 0) + cnt

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta,
        "aggregate": aggregate,
        "aggregate_ci95": aggregate_ci,
        "failure_breakdown": failure_totals,
        "tasks": tasks,
    }


def print_summary(report: dict) -> None:
    m = report["meta"]
    print("\n=== CODING-EVAL REPORT ===")
    print(f"provider={m.get('provider')}  model={m.get('model')}  "
          f"tasks={m.get('num_tasks')}  n={m.get('n_samples')}  "
          f"fail_rate={m.get('stub_fail_rate', '-')}")

    print("\naggregate (95% bootstrap CI over tasks and samples):")
    ci = report.get("aggregate_ci95", {})
    for k, v in report["aggregate"].items():
        interval = ci.get(k)
        suffix = f"   [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else ""
        print(f"  {k:<8} {v}{suffix}")

    if report["failure_breakdown"]:
        print("\nfailure breakdown:")
        for status, cnt in sorted(report["failure_breakdown"].items()):
            print(f"  {status:<8} {cnt}")

    print("\nper-task pass@1:")
    for t in report["tasks"]:
        bar = "#" * round(t["pass@1"] * 20)
        print(f"  {t['task_id']:<16} {t['pass@1']:.2f} {bar}")


def save_report(report: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

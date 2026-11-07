"""
run_eval.py — end-to-end coding-agent evaluation.

Pipeline:  load tasks -> sample n solutions/task -> grade in sandbox -> pass@k -> report

Runs fully offline by default (StubProvider), so the whole pipeline is demonstrable
with no API key and at zero cost. Point it at a real model with --provider openai.

Examples:
    # offline, default stub provider
    python run_eval.py --n 10 --k 1 5 10

    # real model
    OPENAI_API_KEY=... python run_eval.py --provider openai --model gpt-4o-mini --n 10 --k 1 5

    # make the stub model "worse" to watch pass@k climb with k
    python run_eval.py --n 10 --k 1 5 10 --stub-fail-rate 0.6

    # best-of-n selection over the verifiable-reward environment (rejection sampling)
    python run_eval.py --n 10 --select best-of-n --reward binary
    python run_eval.py --n 10 --select best-of-n --reward partial
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.generate import build_provider
from harness.grade import grade_task
from harness.report import build_report, print_summary, save_report


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-based coding-agent eval with pass@k")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--provider", default="stub", choices=["stub", "openai"])
    ap.add_argument("--model", default=None, help="model name for the openai provider")
    ap.add_argument("--n", type=int, default=10, help="samples generated per task")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="k values for pass@k")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-candidate wall-clock limit (s)")
    ap.add_argument("--stub-fail-rate", type=float, default=0.4, help="stub provider corruption rate")
    ap.add_argument(
        "--select",
        default="none",
        choices=["none", "best-of-n"],
        help="'best-of-n' scores candidates via the verifiable-reward env and keeps the best (rejection sampling)",
    )
    ap.add_argument(
        "--reward",
        default="binary",
        choices=["binary", "partial"],
        help="reward shape for best-of-n selection",
    )
    ap.add_argument("--out", default="sample_report.json")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    references = {t["task_id"]: t["reference"] for t in tasks}

    if args.provider == "stub":
        provider = build_provider("stub", references, fail_rate=args.stub_fail_rate)
    else:  # openai
        kw = {"model": args.model} if args.model else {}
        provider = build_provider("openai", references, **kw)

    print(f"Loaded {len(tasks)} tasks. Sampling n={args.n} per task via '{args.provider}' provider.\n")

    if args.select == "best-of-n":
        run_best_of_n(provider, tasks, args)
        return

    results = []
    for t in tasks:
        # Prepend task_id so the stub provider can locate the reference solution.
        prompt = f"{t['task_id']}\n{t['prompt']}"
        candidates = provider.sample(prompt, args.n)
        res = grade_task(t["task_id"], candidates, t["tests"], timeout_s=args.timeout)
        results.append(res)
        print(f"  {res.task_id:<16} c={res.c}/{res.n}  {dict(res.status_counts)}")

    meta = {
        "provider": args.provider,
        "model": args.model or ("reference-stub" if args.provider == "stub" else "default"),
        "num_tasks": len(tasks),
        "n_samples": args.n,
        "timeout_s": args.timeout,
        "stub_fail_rate": args.stub_fail_rate if args.provider == "stub" else None,
    }
    report = build_report(results, args.k, meta)
    print_summary(report)
    save_report(report, args.out)
    print(f"\nSaved -> {args.out}")


def run_best_of_n(provider, tasks: list[dict], args) -> None:
    """Best-of-n selection over the verifiable-reward environment (rejection sampling).

    For each task, score all n candidates with the reward function and keep the best.
    Reports the per-task best reward and the aggregate 'best-of-n solve rate' — the
    share of tasks where at least one sampled solution earns full reward. Compared
    against mean per-sample reward, this is the selection 'lift' that makes best-of-n
    a real post-training / inference-scaling technique.
    """
    from harness.reward import CodeRewardEnv, CodeTask

    print(f"Mode: best-of-{args.n} selection · reward='{args.reward}'\n")

    best_rewards = []
    mean_rewards = []
    solved = 0
    for t in tasks:
        prompt = f"{t['task_id']}\n{t['prompt']}"
        candidates = provider.sample(prompt, args.n)
        env = CodeRewardEnv(
            CodeTask(t["task_id"], t["prompt"], t["tests"]),
            shape=args.reward,
            timeout_s=args.timeout,
        )
        _, best_r, all_r = env.best_of_n(candidates)
        best_rewards.append(best_r)
        mean_rewards.append(sum(all_r) / len(all_r))
        if best_r >= 1.0:
            solved += 1
        print(f"  {t['task_id']:<16} best={best_r:.2f}  mean={sum(all_r)/len(all_r):.2f}  rewards={[round(r,2) for r in all_r]}")

    n_tasks = len(tasks)
    agg_best = sum(best_rewards) / n_tasks
    agg_mean = sum(mean_rewards) / n_tasks
    print("\n=== BEST-OF-N (RL-ENV) REPORT ===")
    print(f"reward shape       : {args.reward}")
    print(f"samples per task   : {args.n}")
    print(f"mean per-sample    : {agg_mean:.4f}   (pick one at random)")
    print(f"best-of-{args.n:<2} mean      : {agg_best:.4f}   (rejection sampling)")
    print(f"selection lift     : +{agg_best - agg_mean:.4f}")
    print(f"solve rate (best=1): {solved}/{n_tasks} = {solved / n_tasks:.2f}")


if __name__ == "__main__":
    main()

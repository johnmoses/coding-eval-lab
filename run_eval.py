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

    # run only the hard tier (the discriminating tasks), and eyeball a real solution
    python run_eval.py --provider openai --model gpt-4o-mini --difficulty hard --n 5 --show-samples

    # contamination probe: matched original vs counterfactual-variant pairs (use a real model)
    python run_eval.py --probe contamination --provider openai --model gpt-4o-mini --n 5 --show-samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.generate import build_provider
from harness.grade import grade_task
from harness.report import build_report, print_summary, save_report


def load_tasks(path: Path) -> list[dict]:
    """Load the task set — a pretty-printed JSON array (readable, hand-editable).

    At this scale (a handful of tasks) a single indented JSON file is far more
    readable than one-task-per-line JSONL. If the set grew large, JSONL would be the
    better format (streams line-by-line, appends cleanly) — that's the scale tradeoff.
    """
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-based coding-agent eval with pass@k")
    ap.add_argument("--tasks", default="tasks/tasks.json")
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
    ap.add_argument(
        "--out",
        default=None,
        help="report path; defaults to sample_report.<model>.json so multi-model runs don't clobber each other",
    )
    ap.add_argument(
        "--difficulty",
        default="all",
        choices=["all", "easy", "hard"],
        help="run only tasks of this difficulty tier",
    )
    ap.add_argument(
        "--show-samples",
        action="store_true",
        help="print one generated solution per task (sanity-check that grading is real, not rubber-stamped)",
    )
    ap.add_argument(
        "--probe",
        default="none",
        choices=["none", "contamination"],
        help="'contamination' runs matched original/counterfactual pairs and reports the pass-rate drop",
    )
    ap.add_argument(
        "--counterfactuals",
        default="tasks/counterfactual.json",
        help="counterfactual variants used by --probe contamination",
    )
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    all_tasks = tasks  # the probe pairs against ALL originals, regardless of --difficulty
    if args.difficulty != "all":
        tasks = [t for t in tasks if t.get("difficulty", "easy") == args.difficulty]
    variants = load_tasks(Path(args.counterfactuals)) if args.probe == "contamination" else []
    references = {t["task_id"]: t["reference"] for t in all_tasks + variants}

    if args.provider == "stub":
        provider = build_provider("stub", references, fail_rate=args.stub_fail_rate)
    else:  # openai
        kw = {"model": args.model} if args.model else {}
        provider = build_provider("openai", references, **kw)

    if args.probe == "contamination":
        run_contamination_probe(provider, all_tasks, variants, args)
        return

    tier = f" [{args.difficulty} tier]" if args.difficulty != "all" else ""
    print(f"Loaded {len(tasks)} tasks{tier}. Sampling n={args.n} per task via '{args.provider}' provider.\n")

    if args.select == "best-of-n":
        run_best_of_n(provider, tasks, args)
        return

    results = []
    difficulty_by_id = {t["task_id"]: t.get("difficulty", "easy") for t in tasks}
    for t in tasks:
        # Prepend task_id so the stub provider can locate the reference solution.
        prompt = f"{t['task_id']}\n{t['prompt']}"
        candidates = provider.sample(prompt, args.n)
        res = grade_task(t["task_id"], candidates, t["tests"], timeout_s=args.timeout)
        results.append(res)
        tier = difficulty_by_id[t["task_id"]]
        print(f"  [{tier:<4}] {res.task_id:<24} c={res.c}/{res.n}  {dict(res.status_counts)}")
        if args.show_samples:
            _show_one_sample(candidates, res)

    meta = {
        "provider": args.provider,
        "model": args.model or ("reference-stub" if args.provider == "stub" else "default"),
        "num_tasks": len(tasks),
        "difficulty": args.difficulty,
        "n_samples": args.n,
        "timeout_s": args.timeout,
        "stub_fail_rate": args.stub_fail_rate if args.provider == "stub" else None,
    }
    report = build_report(results, args.k, meta)
    report = _add_difficulty_breakdown(report, difficulty_by_id)
    print_summary(report)
    _print_difficulty_breakdown(report)
    out = args.out or _default_report_name(meta["model"])
    save_report(report, out)
    print(f"\nSaved -> {out}")


def run_contamination_probe(provider, originals: list[dict], variants: list[dict], args) -> None:
    """Counterfactual contamination probe: matched original/variant pairs, per-pair drop."""
    from harness.probe import load_pairs, print_probe, run_probe, summarize

    pairs = load_pairs(originals, variants)
    print(f"Contamination probe: {len(pairs)} original/variant pairs, n={args.n} samples each "
          f"via '{args.provider}' provider.\n")
    rows = run_probe(provider, pairs, n=args.n, timeout_s=args.timeout)
    summary = summarize(rows)
    print_probe(rows, summary, args.provider, show_samples=args.show_samples)

    model = args.model or ("reference-stub" if args.provider == "stub" else "default")
    report = {
        "meta": {"provider": args.provider, "model": model, "n_samples": args.n, "n_pairs": len(pairs)},
        "summary": summary,
        "pairs": rows,
    }
    out = args.out or _default_report_name(model).replace("sample_report", "probe_report", 1)
    save_report(report, out)
    print(f"\nSaved -> {out}")


def _default_report_name(model: str) -> str:
    """sample_report.<safe-model-name>.json — keeps one file per model so a
    multi-model comparison doesn't overwrite earlier runs."""
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(model))
    return f"sample_report.{safe}.json"


def _show_one_sample(candidates: list[str], res) -> None:
    """Print the first generated solution for a task, for a correctness eyeball.

    A 100%-pass run is only trustworthy if the graded code is actually correct and
    not rubber-stamped. This dumps a real candidate so you can read it. The sandbox
    status is shown alongside so you can confirm the grade matches the code.
    """
    sample = candidates[0] if candidates else "(no candidate)"
    first_status = next(iter(res.status_counts), "?")
    print(f"        └─ sample[0] (one of {res.n}, status seen: {first_status}):")
    for line in sample.strip().splitlines():
        print(f"           {line}")


def _add_difficulty_breakdown(report: dict, difficulty_by_id: dict) -> dict:
    """Attach per-difficulty pass@1 means — the number that shows whether the task
    set actually discriminates (easy near 1.0 is fine; hard near 1.0 means saturated)."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for t in report["tasks"]:
        buckets[difficulty_by_id.get(t["task_id"], "easy")].append(t["pass@1"])
    report["difficulty_breakdown"] = {
        tier: {"tasks": len(vals), "mean_pass@1": round(sum(vals) / len(vals), 4)}
        for tier, vals in sorted(buckets.items())
    }
    return report


def _print_difficulty_breakdown(report: dict) -> None:
    bd = report.get("difficulty_breakdown", {})
    if len(bd) <= 1:
        return
    print("\nby difficulty (mean pass@1):")
    for tier, stats in bd.items():
        print(f"  {tier:<6} {stats['mean_pass@1']:.2f}  ({stats['tasks']} tasks)")


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

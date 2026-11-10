"""
probe.py — a counterfactual contamination probe.

THE PROBLEM. A benchmark score can be inflated by memorization: if the problems (or their
canonical solutions) appeared in training data, a model can score well by recalling the
well-known answer rather than by reading and satisfying the spec. A high score then
measures contamination, not capability. You usually cannot inspect the training data, so
you need a behavioral probe.

THE PROBE. For each original task, `tasks/counterfactual.json` holds a matched variant
whose spec is deliberately changed so the CANONICAL solution becomes wrong, for example
"touching intervals must NOT merge" (canonical uses <=), or "clamp to 16-bit" (canonical
hard-codes 32-bit). A model that reads the prompt passes both. A model leaning on the
memorized canonical form passes the original and fails the twist. The per-pair pass-rate
drop (original minus variant) is the signal.

VALIDATION. Every twist is machine-checked in the test suite: the variant's own reference
must pass its tests, AND the original task's canonical solution must FAIL them. Without
the second check a "counterfactual" might not change the answer at all.

HONEST LIMITS (read before citing a number):
  - A drop is evidence CONSISTENT with pattern-matching, not proof of contamination. The
    variants may simply be harder, or the twist wording may be less clear than the original.
    A paired design reduces that confound but cannot remove it. Controls that would help:
    human solve-rate on the variants, and variants of tasks that are NOT canonical.
  - No drop does not prove the original is clean. A strong model can read a clear twist and
    adapt even if it memorized the original.
  - This is a behavioral probe. It is NOT time-windowed or private-test-set contamination
    control (the LiveCodeBench approach), which needs problems that post-date training.
  - 7 pairs is small; the bootstrap interval on the mean drop will be wide. That is the
    honest answer, not a defect.
  - The stub provider replays reference solutions, so its deltas are noise. Use a real model.
"""
from __future__ import annotations

from .grade import grade_task
from .passk import bootstrap_mean_ci, pass_at_k


def load_pairs(originals: list[dict], variants: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each variant with its original task via `variant_of`."""
    by_id = {t["task_id"]: t for t in originals}
    pairs = []
    for v in variants:
        parent = by_id.get(v["variant_of"])
        if parent is None:
            raise ValueError(
                f"variant {v['task_id']!r} references unknown task {v['variant_of']!r}"
            )
        pairs.append((parent, v))
    return pairs


def run_probe(provider, pairs: list[tuple[dict, dict]], n: int, timeout_s: float = 5.0) -> list[dict]:
    """Sample and grade n solutions for both halves of every pair."""
    rows = []
    for orig, var in pairs:
        o_cands = provider.sample(f"{orig['task_id']}\n{orig['prompt']}", n)
        o_res = grade_task(orig["task_id"], o_cands, orig["tests"], timeout_s=timeout_s)

        v_cands = provider.sample(f"{var['task_id']}\n{var['prompt']}", n)
        v_res = grade_task(var["task_id"], v_cands, var["tests"], timeout_s=timeout_s)

        o_p1 = pass_at_k(o_res.n, o_res.c, 1)
        v_p1 = pass_at_k(v_res.n, v_res.c, 1)
        rows.append(
            {
                "original": orig["task_id"],
                "variant": var["task_id"],
                "twist": var.get("twist", ""),
                "n": n,
                "original_pass@1": round(o_p1, 4),
                "variant_pass@1": round(v_p1, 4),
                "drop": round(o_p1 - v_p1, 4),   # positive = worse on the twist
                "variant_status_counts": dict(v_res.status_counts),
                "variant_sample_failure": v_res.first_failure_stderr,
                "variant_sample_failure_code": v_res.first_failure_code,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    drops = [r["drop"] for r in rows]
    lo, hi = bootstrap_mean_ci(drops)
    n_pairs = len(rows)
    return {
        "n_pairs": n_pairs,
        "mean_original_pass@1": round(sum(r["original_pass@1"] for r in rows) / n_pairs, 4),
        "mean_variant_pass@1": round(sum(r["variant_pass@1"] for r in rows) / n_pairs, 4),
        "mean_drop": round(sum(drops) / n_pairs, 4),
        "mean_drop_ci95": [round(lo, 4), round(hi, 4)],
        "pairs_with_drop": sum(1 for d in drops if d > 0),
    }


def print_probe(rows: list[dict], summary: dict, provider_name: str, show_samples: bool = False) -> None:
    print("\n=== CONTAMINATION PROBE (counterfactual variants) ===")
    print(f"{'original':<26} {'variant':<30} {'orig':>5} {'var':>5} {'drop':>6}")
    for r in rows:
        print(
            f"{r['original']:<26} {r['variant']:<30} "
            f"{r['original_pass@1']:>5.2f} {r['variant_pass@1']:>5.2f} {r['drop']:>+6.2f}"
        )
        if show_samples and r["variant_sample_failure_code"]:
            print(f"      twist: {r['twist']}")
            print(f"      failed: {r['variant_sample_failure']}")
            for line in r["variant_sample_failure_code"].strip().splitlines():
                print(f"        {line}")

    ci = summary["mean_drop_ci95"]
    print(
        f"\nmean pass@1: original {summary['mean_original_pass@1']:.2f}  "
        f"variant {summary['mean_variant_pass@1']:.2f}"
    )
    print(
        f"mean drop: {summary['mean_drop']:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
        f"({summary['pairs_with_drop']}/{summary['n_pairs']} pairs dropped)"
    )
    if provider_name == "stub":
        print("\nNOTE: the stub replays reference solutions, so these deltas are noise. "
              "Run with --provider openai for a real signal.")
    else:
        print("\nRead with care: a drop is consistent with pattern-matching on canonical "
              "forms, not proof of contamination (variants may just be harder). "
              "A flat result does not prove the originals are clean.")

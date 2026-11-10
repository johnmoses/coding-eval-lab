"""
passk.py — the unbiased pass@k estimator (Chen et al., 2021, "Evaluating Large
Language Models Trained on Code" — the HumanEval paper).

pass@k = the probability that at least one of k sampled solutions is correct.

The naive estimate — "sample k, check if any pass" — is high variance. The unbiased
estimator instead samples n >= k solutions per task, counts c that pass, and computes
the probability that a random size-k subset contains at least one correct solution:

    pass@k = 1 - C(n - c, k) / C(n, k)

Intuition: C(n - c, k) / C(n, k) is the probability that a size-k draw lands entirely
in the (n - c) failing solutions; one minus that is the chance of catching >=1 correct.
Averaging this per-task quantity over all tasks gives the benchmark's pass@k.

Edge cases:
  - c == 0            -> 0.0  (no correct sample can be drawn)
  - n - c < k         -> 1.0  (too few failures to fill a size-k draw, so every draw
                               must include a correct one)
  - k == n            -> c/n collapses correctly out of the formula

Using the complement C(n-c, k)/C(n, k) (rather than expanding products) keeps it
numerically stable for the small n a local lab uses.
"""
from __future__ import annotations

import random
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for a single task.

    Args:
        n: total samples generated for the task (n >= k).
        c: number of those samples that passed.
        k: the k in pass@k.
    """
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def aggregate_pass_at_k(per_task: list[tuple[int, int]], k: int) -> float:
    """Benchmark-level pass@k: mean of per-task pass@k.

    Args:
        per_task: list of (n, c) tuples, one per task.
        k: the k in pass@k.
    """
    tasks = [(n, c) for (n, c) in per_task if n >= k]
    if not tasks:
        return 0.0
    return sum(pass_at_k(n, c, k) for (n, c) in tasks) / len(tasks)


# ------------------------------------------------------------------ uncertainty
def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_pass_at_k_ci(
    per_task: list[tuple[int, int]],
    k: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for benchmark-level pass@k.

    A single pass@k number hides two sources of noise: WHICH tasks happened to be in the
    benchmark, and WHICH samples the model happened to draw for each. This resamples both:
    tasks with replacement, then, within each drawn task, its n samples from a Bernoulli
    with that task's observed pass rate c/n. Each replicate recomputes the unbiased
    pass@k, and the interval is the (alpha/2, 1 - alpha/2) percentiles across replicates.

    Honest limits:
      - It is a plug-in bootstrap: a task at 10/10 or 0/10 contributes no within-task
        variance, so the interval is optimistic at the extremes. A benchmark where every
        task is 10/10 yields a degenerate interval [1.0, 1.0]. That is not evidence the
        benchmark is informative; it says the model aced the tasks drawn.
      - Small task counts give wide intervals. That is the correct signal, not a bug.

    Stdlib-only and seeded, so a reported interval is reproducible.
    """
    tasks = [(n, c) for (n, c) in per_task if n >= k]
    if not tasks:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(tasks)
    replicates = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(m):
            n, c = tasks[rng.randrange(m)]
            p = c / n
            c_star = sum(1 for _ in range(n) if rng.random() < p)
            total += pass_at_k(n, c_star, k)
        replicates.append(total / m)
    replicates.sort()
    return (_percentile(replicates, alpha / 2), _percentile(replicates, 1 - alpha / 2))


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean of per-item values (e.g. per-pair pass-rate
    drops in the contamination probe). Resamples items with replacement."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(values)
    means = sorted(
        sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot)
    )
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


if __name__ == "__main__":
    # Quick self-check of the estimator against known values.
    assert pass_at_k(5, 0, 1) == 0.0           # no correct sample
    assert pass_at_k(5, 5, 1) == 1.0           # all correct
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9  # 1 of 5 correct, pass@1 = 1/5
    assert pass_at_k(5, 1, 5) == 1.0           # draw all 5 -> must include the 1 correct
    assert abs(pass_at_k(4, 2, 2) - (1 - comb(2, 2) / comb(4, 2))) < 1e-9
    # CI sanity: degenerate when everything passes; brackets the estimate otherwise.
    assert bootstrap_pass_at_k_ci([(10, 10)] * 4, 1) == (1.0, 1.0)
    lo, hi = bootstrap_pass_at_k_ci([(10, 8), (10, 5), (10, 9), (10, 6)], 1)
    assert lo <= aggregate_pass_at_k([(10, 8), (10, 5), (10, 9), (10, 6)], 1) <= hi
    print("passk self-check: OK")

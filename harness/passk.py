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


if __name__ == "__main__":
    # Quick self-check of the estimator against known values.
    assert pass_at_k(5, 0, 1) == 0.0           # no correct sample
    assert pass_at_k(5, 5, 1) == 1.0           # all correct
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9  # 1 of 5 correct, pass@1 = 1/5
    assert pass_at_k(5, 1, 5) == 1.0           # draw all 5 -> must include the 1 correct
    assert abs(pass_at_k(4, 2, 2) - (1 - comb(2, 2) / comb(4, 2))) < 1e-9
    print("passk self-check: OK")

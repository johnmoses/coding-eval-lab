"""Tests for the unbiased pass@k estimator and the sandbox status classification.

Runs with ZERO third-party dependencies — no pytest required:

    python3 tests/test_passk.py        # stdlib runner (built in below)

It is also pytest-compatible if you have it:

    python3 -m pytest tests/ -q
"""
import json
import os
import re
import sys
from math import comb

# Make the repo root importable whether run from root or from tests/ (so the
# stdlib runner works without pytest's rootdir/conftest machinery).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from harness.passk import (
    aggregate_pass_at_k,
    bootstrap_mean_ci,
    bootstrap_pass_at_k_ci,
    pass_at_k,
)
from harness.sandbox import run_candidate


# ------------------------------------------------------------------ pass@k math
def test_no_correct_sample():
    assert pass_at_k(5, 0, 1) == 0.0


def test_all_correct():
    assert pass_at_k(5, 5, 1) == 1.0


def test_one_of_five_pass_at_1():
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9


def test_drawing_all_includes_the_one_correct():
    assert pass_at_k(5, 1, 5) == 1.0


def test_matches_closed_form():
    assert abs(pass_at_k(4, 2, 2) - (1 - comb(2, 2) / comb(4, 2))) < 1e-9


def test_k_greater_than_n_raises():
    try:
        pass_at_k(3, 1, 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError when k > n")


def test_aggregate_is_mean_of_per_task():
    # task A: 1/2 correct -> pass@1 = 0.5 ; task B: 2/2 correct -> pass@1 = 1.0
    assert abs(aggregate_pass_at_k([(2, 1), (2, 2)], 1) - 0.75) < 1e-9


# ------------------------------------------------------------------ sandbox
def test_sandbox_pass():
    res = run_candidate("def f():\n    return 1", ["assert f() == 1"])
    assert res.status == "pass"


def test_sandbox_fail_on_wrong_answer():
    res = run_candidate("def f():\n    return 2", ["assert f() == 1"])
    assert res.status == "fail"


def test_sandbox_error_on_bad_code():
    res = run_candidate("def f(:\n    return 1", ["assert f() == 1"])
    assert res.status == "error"


def test_sandbox_timeout_on_infinite_loop():
    res = run_candidate("def f():\n    while True:\n        pass", ["assert f() == 1"], timeout_s=1.0)
    assert res.status == "timeout"


def test_sandbox_blocks_network():
    code = "import socket\ndef f():\n    socket.socket()\n    return 1"
    res = run_candidate(code, ["assert f() == 1"])
    assert res.status in {"fail", "error"}  # network attempt raises -> not a pass
    assert res.status != "pass"


# ------------------------------------------------------------------ reward / RL env
from harness.reward import CodeRewardEnv, CodeTask, binary_reward, partial_reward

_GOOD = "def f(x):\n    return x + 1"
_BAD = "def f(x):\n    return x"
_TESTS = ["assert f(1) == 2", "assert f(0) == 1", "assert f(10) == 11"]


def test_binary_reward_all_or_nothing():
    assert binary_reward(_GOOD, _TESTS) == 1.0
    assert binary_reward(_BAD, _TESTS) == 0.0


def test_partial_reward_is_fraction():
    # _BAD passes none of these (x != x+1 for all), good passes all
    assert partial_reward(_GOOD, _TESTS) == 1.0
    assert partial_reward(_BAD, _TESTS) == 0.0
    # mixed: one test the bad solution happens to satisfy
    mixed_tests = ["assert f(0) == 0", "assert f(1) == 2"]  # _BAD passes #1, fails #2
    assert abs(partial_reward(_BAD, mixed_tests) - 0.5) < 1e-9


def test_env_step_returns_reward():
    env = CodeRewardEnv(CodeTask("t", "p", _TESTS), shape="binary")
    assert env.step(_GOOD) == 1.0
    assert env.step(_BAD) == 0.0


def test_best_of_n_picks_the_passing_candidate():
    env = CodeRewardEnv(CodeTask("t", "p", _TESTS), shape="binary")
    best, best_r, all_r = env.best_of_n([_BAD, _BAD, _GOOD, _BAD])
    assert best_r == 1.0
    assert best.strip() == _GOOD.strip()
    assert all_r.count(1.0) == 1


# ------------------------------------------------------------------ confidence intervals
def test_ci_degenerate_when_everything_passes():
    assert bootstrap_pass_at_k_ci([(10, 10)] * 5, 1) == (1.0, 1.0)


def test_ci_brackets_the_point_estimate():
    per_task = [(10, 8), (10, 5), (10, 9), (10, 6), (10, 7)]
    lo, hi = bootstrap_pass_at_k_ci(per_task, 1)
    point = aggregate_pass_at_k(per_task, 1)
    assert lo <= point <= hi
    assert hi - lo > 0  # genuine uncertainty with mixed results


def test_ci_is_reproducible_with_seed():
    per_task = [(10, 8), (10, 5), (10, 9)]
    assert bootstrap_pass_at_k_ci(per_task, 1, seed=3) == bootstrap_pass_at_k_ci(per_task, 1, seed=3)


def test_ci_narrows_with_more_tasks():
    few = bootstrap_pass_at_k_ci([(10, 5)] * 3 + [(10, 9)] * 3, 1)
    many = bootstrap_pass_at_k_ci([(10, 5)] * 30 + [(10, 9)] * 30, 1)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_mean_ci_basic():
    assert bootstrap_mean_ci([0.0, 0.0, 0.0]) == (0.0, 0.0)
    lo, hi = bootstrap_mean_ci([0.0, 0.5, 1.0, 0.5])
    assert lo <= 0.5 <= hi


# ------------------------------------------------------------------ benchmark validation
def _load(name):
    with open(os.path.join(_ROOT, "tasks", name)) as f:
        return json.load(f)


def test_every_reference_solution_passes_its_own_tests():
    """A benchmark whose own reference fails its tests is broken before any model runs."""
    for t in _load("tasks.json") + _load("counterfactual.json"):
        res = run_candidate(t["reference"], t["tests"])
        assert res.status == "pass", f"{t['task_id']}: reference {res.status}: {res.stderr[-120:]}"


def test_counterfactual_variants_point_at_real_originals():
    originals = {t["task_id"] for t in _load("tasks.json")}
    for v in _load("counterfactual.json"):
        assert v["variant_of"] in originals, f"{v['task_id']} -> unknown {v['variant_of']}"


def test_canonical_solution_fails_every_counterfactual_variant():
    """The point of a counterfactual: the memorized canonical answer must be WRONG here.
    If it passed, the twist wouldn't change the answer and the probe would measure nothing."""
    originals = {t["task_id"]: t for t in _load("tasks.json")}
    for v in _load("counterfactual.json"):
        o = originals[v["variant_of"]]
        renamed = re.sub(
            r"\bdef " + re.escape(o["entry_point"]) + r"\(",
            "def " + v["entry_point"] + "(",
            o["reference"],
        )
        res = run_candidate(renamed, v["tests"])
        assert res.status != "pass", f"{v['task_id']}: canonical solution still passes the twist"


# --------------------------------------------------------------------------- runner
def _run_all() -> int:
    """Minimal stdlib test runner: discover and run every module-level test_* function.

    Exists so the suite runs with no third-party deps (pytest optional). Returns a
    process exit code: 0 if all pass, 1 otherwise.
    """
    import traceback

    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception:
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())

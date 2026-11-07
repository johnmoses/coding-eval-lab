"""Tests for the unbiased pass@k estimator and the sandbox status classification."""
from math import comb

from harness.passk import aggregate_pass_at_k, pass_at_k
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

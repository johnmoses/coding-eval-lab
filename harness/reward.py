"""
reward.py — the grader as a VERIFIABLE REWARD function (the eval/RL bridge).

Coding evaluation and reinforcement learning for code are the same loop seen from two
ends. An eval harness asks "did this solution pass the tests?"; RL-from-verifiable-
rewards (RLVR) asks the identical question and uses the answer as the reward signal.
Unlike RLHF, the reward here is objective — it comes from executing code against tests,
not from a learned preference model.

This module reframes the existing sandbox grader as an RL-style environment:

    state   = the task (prompt + hidden tests)       -> CodeTask
    action  = a generated candidate solution          -> str
    reward  = f(sandbox result)                        -> float
    step()  = run the action, return the reward        -> CodeRewardEnv.step

Two reward shapes are provided:
  - binary       : 1.0 if all tests pass, else 0.0 (the SWE-bench "resolved" signal)
  - partial      : fraction of individual tests passed, with a small timeout penalty
                   (denser signal — more useful for RL credit assignment)

Scope note (honest): this gives you the ENVIRONMENT and the REWARD — the parts RL-for-
code actually hinges on — plus best-of-n selection (rejection sampling) over that reward.
It does NOT implement a policy-gradient training loop (GRPO/PPO). That is the deliberate
boundary: the verifiable-reward environment is real and reusable; wiring it to an actual
fine-tune is a separate, compute-heavy project.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sandbox import run_candidate


@dataclass
class CodeTask:
    """An RL 'environment instance': one task the policy must solve."""
    task_id: str
    prompt: str
    tests: list[str]


def binary_reward(candidate: str, tests: list[str], timeout_s: float = 5.0) -> float:
    """1.0 iff the candidate passes ALL hidden tests, else 0.0.

    This is the SWE-bench-style 'resolved' signal: a patch either makes the suite
    green or it doesn't. Sparse but unambiguous.
    """
    res = run_candidate(candidate, tests, timeout_s=timeout_s)
    return 1.0 if res.passed else 0.0


def partial_reward(
    candidate: str,
    tests: list[str],
    timeout_s: float = 5.0,
    timeout_penalty: float = 0.25,
) -> float:
    """Fraction of individual tests the candidate passes (0.0-1.0), with a penalty
    for hanging.

    Denser than binary: a solution that passes 3 of 4 tests scores 0.75, which gives
    an RL optimizer useful gradient instead of an all-or-nothing signal. Each test is
    run as its own one-assertion program so a single failing assertion doesn't mask
    the ones that would have passed. A timeout is scored below zero-reward failure,
    because a hang is worse than a wrong answer (it wastes the rollout budget).
    """
    if not tests:
        return 0.0
    passed = 0
    saw_timeout = False
    for t in tests:
        res = run_candidate(candidate, [t], timeout_s=timeout_s)
        if res.passed:
            passed += 1
        elif res.status == "timeout":
            saw_timeout = True
    frac = passed / len(tests)
    if saw_timeout:
        frac = max(0.0, frac - timeout_penalty)
    return frac


_SHAPES = {"binary": binary_reward, "partial": partial_reward}


def reward_fn(shape: str):
    """Return a reward function by name: 'binary' or 'partial'."""
    try:
        return _SHAPES[shape]
    except KeyError:
        raise ValueError(f"unknown reward shape: {shape!r} (choose from {list(_SHAPES)})")


class CodeRewardEnv:
    """A minimal RL-style environment over the sandbox grader.

    Deliberately small and stateless-per-step: one task in, actions evaluated against
    its verifiable reward. This is the interface an RL trainer (or a best-of-n sampler)
    would call. It is NOT a gym.Env — no observation encoding, no episode bookkeeping —
    because the point is the reward semantics, not a framework.
    """

    def __init__(self, task: CodeTask, shape: str = "binary", timeout_s: float = 5.0):
        self.task = task
        self._reward = reward_fn(shape)
        self._timeout_s = timeout_s

    def step(self, action: str) -> float:
        """Evaluate one candidate solution (action) and return its scalar reward."""
        return self._reward(action, self.task.tests, timeout_s=self._timeout_s)

    def best_of_n(self, candidates: list[str]) -> tuple[str, float, list[float]]:
        """Rejection sampling / best-of-n: score every candidate, return the best.

        This is the gateway post-training technique — the same selection used to build
        preference data or do inference-time scaling. Returns (best_candidate,
        best_reward, all_rewards) so callers can inspect the reward distribution.
        """
        rewards = [self.step(c) for c in candidates]
        best_i = max(range(len(rewards)), key=lambda i: rewards[i])
        return candidates[best_i], rewards[best_i], rewards

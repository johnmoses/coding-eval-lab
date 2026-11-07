# Coding Eval Lab

A tiny but real **execution-based evaluation harness for coding agents** — the
machinery frontier labs use to measure a coding model: sample solutions, run them
against hidden tests in a sandbox, and score with the **unbiased pass@k estimator**.

The goal: take a coding-eval through the full pipeline — **load tasks → sample n
solutions → grade in an isolated sandbox → compute pass@k → report failure modes** —
and do it with the measurement discipline a real benchmark needs (contained
execution, the correct estimator, diagnosable failures), not a toy accuracy count.

> Runs **fully offline** out of the box (a stub provider replays reference solutions
> with controllable corruption), so the entire pipeline is demonstrable with no API
> key and zero cost. Point it at a real model with `--provider openai`.

---

## Headline result — the full pipeline, measured

10 Python tasks, 10 samples each, graded by execution against hidden tests, scored
with the unbiased pass@k estimator (stub provider, 40% corruption):

| Metric | Value | What it means |
|---|---|---|
| **pass@1** | 0.81 | single-shot reliability (per-sample success rate) |
| **pass@5** | 1.00 | capability ceiling with 5 samples |
| **pass@10** | 1.00 | capability ceiling with 10 samples |
| Failure breakdown | 19 `fail` | all failures were wrong answers, not hangs/crashes |

**The finding the estimator is built to show:** pass@1 ≈ 0.81 climbs to pass@5 = 1.0.
A model that's right ~80% of the time per sample is nearly guaranteed to be right at
least once in five tries — and pass@k quantifies exactly that, instead of hiding it
behind a single pass@1 number. Which k matters is a product decision: a one-shot
agent cares about pass@1; an agent with a verifier/retry loop cares about pass@k.

---

## Why

Evaluating a coding model is not string comparison — **correctness is executable**.
You run the generated code against tests and see if it passes. That makes two things
load-bearing, and they're the two things this lab takes seriously:

1. **Sandboxed execution.** The harness runs code it did not write. Generated code is
   treated as hostile: isolated subprocess, hard timeout (kills infinite loops),
   best-effort network block, throwaway working directory. A crash takes down the
   child, not the harness.
2. **The right scoring.** pass@k is estimated with the unbiased combinatorial formula
   from the HumanEval paper (sample n > k, count c passing), not the high-variance
   "sample k and check." Failures are categorized — wrong answer vs timeout vs
   didn't-run — so a low score is diagnosable.

## Pipeline

```
tasks.jsonl ──load──▶ Provider.sample(n)  ──▶ n candidate solutions / task
   (prompt + hidden tests)   │ stub (offline)  openai (real)
                             ▼
                      sandbox.run_candidate ──▶ isolated subprocess
                        (timeout · no network · scratch dir)
                             │  status: pass / fail / timeout / error
                             ▼
                      grade_task ──▶ (n, c) + failure histogram per task
                             │
                             ▼
                      pass@k (unbiased) ──▶ aggregate + per-task report (JSON)
```

## The eval/RL bridge — a verifiable-reward environment

Coding evaluation and reinforcement learning for code are the same loop seen from two
ends. An eval harness asks *"did this pass the tests?"*; RL-from-verifiable-rewards
(RLVR) asks the identical question and uses the answer as the **reward**. Unlike RLHF,
the reward is objective — it comes from executing code, not from a learned preference
model. The grader is already that reward function.

`harness/reward.py` makes this explicit:

```
state  = the task (prompt + hidden tests)   ->  CodeTask
action = a generated candidate solution      ->  str
reward = f(sandbox result)                   ->  float   (binary | partial)
step() = run the action, return the reward   ->  CodeRewardEnv.step
```

Two reward shapes: **binary** (1.0 iff all tests pass — the SWE-bench "resolved"
signal) and **partial** (fraction of tests passed, with a timeout penalty — a denser
signal for credit assignment). On top of the env, **best-of-n selection** (rejection
sampling) is the gateway post-training / inference-scaling technique:

```bash
python3 run_eval.py --n 10 --select best-of-n --reward binary
# mean per-sample    : 0.8100   (pick one at random)
# best-of-10 mean    : 1.0000   (rejection sampling)
# selection lift     : +0.1900
```

**Honest scope:** this gives you the *environment* and the *reward* — the parts
RL-for-code actually hinges on — plus best-of-n over that reward. It does **not**
implement a policy-gradient training loop (GRPO/PPO). That boundary is deliberate: the
verifiable-reward environment is real and reusable; wiring it to an actual fine-tune is
a separate, compute-heavy project (and on the roadmap, not the claim).

## Design notes

- **`harness/sandbox.py`** — the containment layer. `run_candidate` is the single
  choke point; a production grader would slot an OS sandbox (containers / seccomp /
  rlimits / read-only FS) in behind the same interface. Scope is stated honestly in
  the module docstring: process-level isolation for untrusted-but-not-adversarial code.
- **`harness/passk.py`** — the unbiased estimator `1 − C(n−c, k) / C(n, k)`, with the
  edge cases (c=0, n−c<k, k=n) handled and a self-check (`python harness/passk.py`).
- **`harness/generate.py`** — providers behind one `Protocol`, the same swappable-seam
  pattern as a production `AIProvider`. `StubProvider` is offline and seeded;
  `OpenAIProvider` is lazily imported so the lab runs without the SDK.
- **Zero runtime dependencies** in the core harness (stdlib only). The sandbox and the
  estimator shouldn't need a third-party stack to be trustworthy.

## Reproduce

```bash
# 0. clone
git clone https://github.com/johnmoses/coding-eval-lab.git
cd coding-eval-lab

# 1. (optional) venv — the core harness has no deps, so this is only needed
#    for the openai provider or pytest
python3 -m venv .venv && source .venv/bin/activate

# 2. verify the estimator against known values
python3 harness/passk.py            # -> "passk self-check: OK"

# 3. run the full eval, offline (stub provider)
python3 run_eval.py --n 10 --k 1 5 10

# 4. make the stub model "worse" and watch pass@k climb with k
python3 run_eval.py --n 10 --k 1 5 10 --stub-fail-rate 0.6

# 5. run against a real model
OPENAI_API_KEY=... python3 run_eval.py --provider openai --model gpt-4o-mini --n 10 --k 1 5

# 6. tests (sandbox safety cases + estimator math)
python3 -m pytest tests/ -q
```

## Adding a task

Append one JSON line to `tasks/tasks.jsonl`:

```json
{"task_id": "my_task", "prompt": "Write a function `f(x)` that ...",
 "entry_point": "f", "tests": ["assert f(1) == 2", "assert f(0) == 1"],
 "reference": "def f(x):\n    return x + 1"}
```

`tests` are the hidden test suite (the real grading signal). `reference` is used only
by the offline stub provider and as a sanity anchor — a real run never sees it.

## Honest scope

- A **local lab**, not a production eval service. Isolation is process-level
  (subprocess + timeout + network stub + isolated-mode interpreter), suitable for
  untrusted-but-not-adversarial code. Truly adversarial input needs an OS sandbox;
  the seam for that is `run_candidate`.
- The **stub provider is not a model** — it replays reference solutions with injected
  bugs so the pipeline produces non-trivial, reproducible pass@k numbers offline. Real
  model quality comes from `--provider openai` (or any provider you add).
- The **task set is small and simple** (function-level, HumanEval-flavored). Extending
  toward realistic, repo-level, contamination-resistant tasks (SWE-bench / LiveCodeBench
  style) is the roadmap, not the claim.

## Roadmap

- [x] Execution-based grading in an isolated sandbox (timeout, network block, teardown)
- [x] Unbiased pass@k estimator with self-check + unit tests
- [x] Provider seam (offline stub + real OpenAI), n-sample generation
- [x] JSON report with per-task pass@1 and failure categorization
- [x] Verifiable-reward environment (binary + partial reward) — the eval/RL bridge
- [x] Best-of-n selection (rejection sampling) over the reward, with selection-lift report
- [ ] Contamination controls (time-windowed / held-out private tests)
- [ ] Richer failure taxonomy (wrong-output vs exception-type clustering)
- [ ] Repo-level tasks (apply-a-patch, run-the-repo-tests) — SWE-bench flavor
- [ ] Stronger sandbox backend (container / rlimits) behind the same interface
- [ ] A toy policy-gradient loop (GRPO) against the verifiable reward — PoC only

## Environment

Python 3.10+ · stdlib only for the core harness · macOS/Linux. The optional
`openai` provider and `pytest` are the only third-party pieces, both opt-in.

## License

MIT

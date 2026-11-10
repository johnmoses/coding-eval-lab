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

## Headline finding — a benchmark's value is in what it separates

16 Python tasks in two difficulty tiers (10 easy, 6 hard), each sampled n times, graded
by execution against hidden tests, scored with the unbiased pass@k estimator.

**gpt-4o-mini scores 100% pass@1 on BOTH tiers — and that is a finding, not a flex.**
A perfect score means the tasks are *saturated*: they can no longer distinguish a strong
model from a weak one. Even the hard tier (subtractive Roman numerals, interval merging
with touch-at-endpoint edges, `atoi` with 32-bit clamping, spiral-matrix traversal) is
saturated — because these are canonical problems a frontier model has seen in training.
This is precisely the known weakness of function-level benchmarks like HumanEval, and it
is the empirical case for the roadmap: contamination-resistant, repo-level tasks. A small
local model, by contrast, does fail the hard tier (0.70), so the set separates a weak model
from a strong one even though it cannot separate frontier models from each other.

> **The point of the lab is the measurement discipline, not a high number.** A benchmark
> that everything passes measures nothing. The useful signal is the *spread* — between
> difficulty tiers and between models — and the *failure breakdown*, not the headline rate.

### Measured results

Same 16 tasks, same harness, two very different models:

| Model | n | easy pass@1 | hard pass@1 | overall pass@1 (95% CI) | pass@5 | report |
|---|---|---|---|---|---|---|
| gpt-4o-mini (hosted) | 10 | 1.00 | 1.00 | 1.00 | 1.00 | _not archived_ ¹ |
| qwen2.5-coder:1.5b (local, Ollama) | 5 | 0.98 | 0.70 | 0.875 (0.71–0.99) | 0.94 | [`sample_report.qwen2.5-coder-1.5b.example.json`](./sample_report.qwen2.5-coder-1.5b.example.json) |

¹ The gpt-4o-mini run's console output was checked and its solutions eyeballed (see
`--show-samples`), but the full-run report file was later overwritten by a smaller
hard-tier run, so there is no archived report for that row. Re-run to archive it.

**The finding: the benchmark saturates a frontier model and resolves a small one.** The
frontier model clears everything, so on this task class it cannot discriminate frontier
models. A 1.5B local model lands in the useful middle, with an easy/hard gap (0.98 vs 0.70)
that the saturated run could never show. This is a real spread from a real second model,
not a stub artifact.

**What the 1.5B actually got wrong** (first failing sample per task, n=5, so illustrative and
not a statistical taxonomy):

| Task | Pass | Failure mode |
|---|---|---|
| `two_sum` | 4/5 | **Missing import**: used `Tuple` after importing only `List`, so the code never ran (`error`, not `fail`) |
| `roman_to_int` | 3/5 | **Plausible-but-wrong one-liner**: a conditional whose subtract branch can never execute, so it is effectively additive |
| `atoi` | 0/5 | **Spec misread**: returns `0` on overflow instead of clamping to the 32-bit range |
| `spiral_order` | 3/5 | **Boundary/ordering bug** in the direction changes |

A run writes `sample_report.<model>.json`; to keep one as a committed artifact, copy it to
`sample_report.<model>.example.json` (the tracked naming; plain report files are gitignored).

> **Why a local model, not another hosted one.** The obvious "run a weaker model" step hit a
> wall on the hosted side: OpenAI's older cheap models (e.g. gpt-3.5-turbo) were retired,
> and their documented replacements are *more* capable. A small local open-weight model is
> the zero-cost way to get a weaker data point. See the next section.

### Running without API credits — local and free-tier models

`--provider openai-compatible` talks to any OpenAI-style endpoint, so the same harness runs
against a local model for **$0**:

```bash
# one-time: pull a coding model (the 1.5B is ~1 GB and ran the full eval in under 2 minutes;
# the 7B is roughly 4-5 GB and optional, see the probe section for why you might want it)
ollama pull qwen2.5-coder:1.5b

# run the eval (Ollama must be running; no API key needed for a local server)
python3 run_eval.py --provider openai-compatible --base-url http://localhost:11434/v1 \
    --model qwen2.5-coder:1.5b --n 5 --show-samples

# the contamination probe works the same way
python3 run_eval.py --probe contamination --provider openai-compatible \
    --base-url http://localhost:11434/v1 --model qwen2.5-coder:1.5b --n 5 --show-samples
```

**Why this matters beyond cost:** a small model should actually *fail* some of these tasks.
That gives the lab what the saturated frontier run couldn't: a real tier spread, and real
failures for the failure breakdown to categorize instead of the stub's synthetic ones.

**Things to know:**
- **Ollama's `/v1` endpoint ignores the `n` parameter** (an open upstream issue), returning one
  choice per call. The provider therefore loops until it has exactly `n` samples. Without that,
  pass@k would silently be computed from one sample. This is covered by a test.
- **Local generation is slower than a hosted API**, and `n=5` means five sequential
  generations per task. Start with `--n 3 --difficulty hard` to gauge speed.
- **Read the failure breakdown, not just pass@1.** Small models often wrap code in prose or
  break the format; that shows up as `error` (didn't run) rather than `fail` (ran, wrong
  answer). Those are different weaknesses: instruction-following versus coding.
- **Hosted free tiers** (Groq, OpenRouter `:free` models, Gemini, etc.) work through the same
  flags: `--base-url <their /v1 endpoint> --api-key-env <THEIR_KEY_VAR>`. They are rate-limited,
  the provider backs off on HTTP 429, and the free-model catalogs and limits change often, so
  check the provider's current docs. An exhausted-quota error is *not* retried.
- **Contamination caveat still applies:** a small model failing a canonical task partly
  reflects how much of the canon it memorized, not pure capability.

**The estimator, shown on the stub provider** (deterministic, offline, 40% injected
corruption, 16 tasks × n=10, so pass@1 is sub-1.0 and the pass@k curve is visible):

| Metric | Value | 95% CI | What it means |
|---|---|---|---|
| **pass@1** | 0.83 | [0.76, 0.90] | single-shot reliability (per-sample success rate) |
| **pass@5** | 1.00 | [0.995, 1.00] | capability ceiling with 5 samples |
| **pass@10** | 1.00 | [1.00, 1.00] | capability ceiling with 10 samples |

pass@1 ≈ 0.83 climbing to pass@5 ≈ 1.0 is exactly what the estimator is built to show: a
model right ~80% of the time per sample is nearly certain to be right at least once in
five tries. Which k matters is a product decision — a one-shot agent cares about pass@1;
an agent with a verifier/retry loop cares about pass@k.

**Intervals, not just point estimates.** Every aggregate pass@k carries a 95% percentile
bootstrap interval that resamples *both* tasks and the samples within each task (stdlib,
seeded, reproducible). Two honest caveats: it is a plug-in bootstrap, so a task at 10/10
or 0/10 adds no within-task variance and intervals are optimistic at the extremes; and a
benchmark where every task is 10/10 gives a degenerate `[1.0, 1.0]`. That interval says the
model aced the tasks drawn, **not** that the benchmark is informative.

**Trust but verify the grade:** `--show-samples` prints a real generated solution per
task alongside its sandbox status, so a 100% run can be eyeballed for correctness rather
than taken on faith (guards against a grading bug masquerading as a perfect score). Failed
candidates now also record *which hidden assertion* tripped, so failures are diagnosable.

### Contamination probe — counterfactual variants

A high score can mean memorization: if a problem and its canonical solution were in the
training data, a model can recall the well-known answer without reading the spec. You can't
inspect training data, so `--probe contamination` is a *behavioral* test. It runs 7 matched
pairs. Each original task has a variant whose spec is changed so the **canonical solution
becomes wrong**:

| Original | Variant twist |
|---|---|
| `merge_intervals` | touching endpoints must **not** merge (canonical uses `<=`) |
| `atoi` | clamp to **16-bit** (canonical hard-codes 32-bit) |
| `fizzbuzz` | divisors **7 and 11** instead of 3 and 5 |
| `spiral_order` | **counterclockwise**, going down first |
| `longest_unique_substring` | return the **substring**, not its length |
| `roman_to_int` | **additive-only** numerals (`IV` = 6) |
| `is_valid_parens` | any **non-bracket character** invalidates the string |

A model that reads the prompt passes both halves. A model leaning on the memorized form
passes the original and fails the twist, and the per-pair pass@1 drop is the signal, with a
bootstrap interval on the mean drop.

```bash
python3 run_eval.py --probe contamination --provider openai --model gpt-4o-mini --n 5 --show-samples
```

**The twists are machine-validated** in the test suite: each variant's own reference must
pass its tests, *and* the original's canonical solution must **fail** them. Without the
second check, a "counterfactual" might not change the answer and would measure nothing.

#### Measured: qwen2.5-coder:1.5b (n=5, 7 pairs)

Report: [`probe_report.qwen2.5-coder-1.5b.example.json`](./probe_report.qwen2.5-coder-1.5b.example.json)

| Original → variant | original | variant | drop | note |
|---|---|---|---|---|
| `merge_intervals` → `_strict` | 0.80 | 0.00 | +0.80 | failing code is the canonical `<=` merge, ignoring the instruction |
| `spiral_order` → `_ccw` | 0.60 | 0.00 | +0.60 | failing code is the canonical clockwise traversal |
| `roman_to_int` → `roman_additive` | 0.80 | 0.00 | +0.80 | applies subtractive logic despite "additive-only" |
| `longest_unique_substring` → `_text` | 1.00 | 0.60 | +0.40 | **confounded**: the variant is also objectively harder (needs best-start tracking) |
| `is_valid_parens` → `is_strict_parens` | 0.60 | 0.40 | +0.20 | failing code silently skips non-bracket characters (canonical behavior) |
| `atoi` → `atoi16` | 0.20 | 0.00 | +0.20 | **floor-limited**: the original is already mostly failed; this failure was an unrelated bug (`UnboundLocalError`) |
| `fizzbuzz` → `fizzbuzz_7_11` | 1.00 | 1.00 | 0.00 | followed the explicit twist |

**Mean pass@1 fell from 0.71 to 0.29: mean drop +0.43, 95% CI [+0.23, +0.63], 6 of 7 pairs dropped.**
The headline is the all-pairs mean, fixed before looking at per-pair results; dropping the
floor-limited `atoi` pair after the fact would raise it and is deliberately not done.

**What it supports:** in the four clearest pairs the failing code reproduces the canonical
behavior while an explicit instruction in the prompt says otherwise. The model leans on the
well-known form over the stated spec. (This is from reading the first failing sample per pair,
not every failure.)

**What it does not show:**
- It does **not** separate memorization from weak instruction-following. A 1.5B model would
  produce the same signature either way, so read it as *reliance on the canonical form*, not
  as proof of training-data contamination.
- There is no stronger-model control yet. If a larger model of the same family passes the
  twists, the variants are fair and this drop is capability-bound. That control (a 7B, or
  gpt-4o-mini when credits allow) is the most valuable next run.
- One model, 7 pairs, n=5: the interval is wide, and two of the seven pairs are weak evidence
  (see the notes above).

**Read a result carefully:**
- A drop is evidence *consistent with* pattern-matching, not proof of contamination. The
  variants may simply be harder or less clearly worded. The paired design reduces that
  confound but can't remove it.
- No drop does **not** prove the originals are clean. A strong model can read a clear twist
  and adapt even if it memorized the original.
- 7 pairs is small, so the interval on the mean drop will be wide. That is the honest answer.
- This is **not** time-windowed or private-test-set control (the LiveCodeBench approach),
  which needs problems that post-date training. It is a cheap behavioral first step.
- The offline stub replays reference solutions, so its deltas are noise. Use a real model.

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
tasks.json ──load──▶ Provider.sample(n)  ──▶ n candidate solutions / task
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

# 1. (optional) venv — the core harness has NO deps, so steps 2-4 and the tests
#    need nothing installed. A venv is only useful for the optional extras below.
python3 -m venv .venv && source .venv/bin/activate

# 2. verify the estimator against known values
python3 harness/passk.py            # -> "passk self-check: OK"

# 3. run the full eval, offline (stub provider) — zero dependencies
python3 run_eval.py --n 10 --k 1 5 10

# 4. make the stub model "worse" and watch pass@k climb with k
python3 run_eval.py --n 10 --k 1 5 10 --stub-fail-rate 0.6

# 4b. run only one difficulty tier; dump a real solution per task to check grading
python3 run_eval.py --difficulty hard --n 5 --show-samples

# 4c. contamination probe plumbing, offline (stub deltas are noise; see the probe section)
python3 run_eval.py --probe contamination --n 4

# 5. run against a real model — reports auto-name as sample_report.<model>.json
#    uncomment the openai lines in requirements.txt, then:
cp .env.example .env                        # edit .env, set OPENAI_API_KEY
pip install -r requirements.txt
python3 run_eval.py --provider openai --model gpt-4o-mini --n 10 --k 1 5 --show-samples
#    --show-samples dumps a real solution per task so a 100% run can be eyeballed
#    (use whatever current model id your account has; model ids change as OpenAI
#     retires older ones — the harness is model-agnostic)
#    (an inline `OPENAI_API_KEY=... python3 run_eval.py ...` still works too)

# 6. tests (sandbox safety cases + estimator math) — no pytest required
python3 tests/test_passk.py                 # stdlib runner, zero deps
python3 -m pytest tests/ -q                 # also works if you have pytest
```

**Dependencies:** `requirements.txt` installs nothing by default — the stub pipeline and
the tests are stdlib-only. The optional extras (`openai` + `python-dotenv` for the real
provider, `pytest` for the test runner) are listed there, commented and pinned; uncomment
what you need. That keeps the zero-dependency core intact.

### API key handling

The OpenAI key is **read from the environment**, never hardcoded. Two ways to provide it,
in order of preference:

1. **A gitignored `.env` file** (recommended). `cp .env.example .env` and set
   `OPENAI_API_KEY`. `.env` is in `.gitignore`; only `.env.example` (a dummy value) is
   committed — it documents the contract. This keeps the key out of your shell history.
2. **A real environment variable** (`export OPENAI_API_KEY=...` or inline).

`.env` loading is optional and best-effort: it uses `python-dotenv` if installed, falls
back to a tiny built-in parser if not, and does nothing if there's no `.env` — so the
**core harness keeps its zero-dependency stance**. The key only ever comes from the
environment, so it cannot be committed.

## Adding a task

Add an object to the array in `tasks/tasks.json`:

```json
{
  "task_id": "my_task",
  "difficulty": "hard",
  "prompt": "Write a function `f(x)` that ...",
  "entry_point": "f",
  "tests": ["assert f(1) == 2", "assert f(0) == 1"],
  "reference": "def f(x):\n    return x + 1"
}
```

`difficulty` is `easy` or `hard` and drives the per-tier breakdown and the `--difficulty`
filter. The strongest tasks are the *discriminating* ones — where a plausible but wrong
solution fails a hidden edge-case test. That's where a benchmark earns its keep.

`tests` are the hidden test suite (the real grading signal). `reference` is used only
by the offline stub provider and as a sanity anchor — a real run never sees it.

> Format note: a pretty-printed JSON array is used for readability at this scale. A
> large task set would move to JSONL (one task per line — streams and appends cleanly).

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
- [x] 95% bootstrap confidence intervals on pass@k (tasks and samples resampled)
- [x] Counterfactual contamination probe: 7 matched pairs, twists machine-validated
- [x] Benchmark self-validation tests (every reference passes; every canonical solution fails its twist)
- [x] OpenAI-compatible provider (local Ollama / llama.cpp, hosted free tiers) with `n`-ignoring-server handling and 429 backoff
- [x] Run the full eval and the contamination probe against a real model (qwen2.5-coder:1.5b), reports archived
- [ ] Probe control: run a stronger model (a 7B, or gpt-4o-mini) to test whether the twists are fair
- [ ] Re-run gpt-4o-mini on the full set and archive its report
- [ ] Time-windowed / private held-out contamination controls (needs post-training-cutoff problems)
- [ ] Test-quality measurement (mutation testing: do the hidden tests catch plausible wrong code?)
- [ ] Richer failure taxonomy (wrong-output vs exception-type clustering; the failing assertion is recorded today)
- [ ] Repo-level tasks (apply-a-patch, run-the-repo-tests) — SWE-bench flavor
- [ ] Stronger sandbox backend (container / rlimits) behind the same interface
- [ ] A toy policy-gradient loop (GRPO) against the verifiable reward — PoC only

## Environment

Python 3.10+ · stdlib only for the core harness · macOS/Linux. `requirements.txt`
installs nothing by default; the optional extras (`openai` + `python-dotenv` for the
real provider, `pytest` for the test runner) are listed there, commented and pinned —
uncomment what you need. The default stub pipeline and the tests require no third-party
packages at all.

## License

MIT

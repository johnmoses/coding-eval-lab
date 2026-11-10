"""
generate.py — sample candidate solutions for a task from a provider.

Providers sit behind one Protocol so the harness never cares which model (or stub)
produced a solution — the same seam pattern used in production AIProvider code. Two
providers ship here:

  - StubProvider  : offline, deterministic, needs no API key. It returns the task's
                    reference solution with controllable corruption, so you can run
                    the FULL pipeline (generate -> sandbox -> pass@k -> report) and
                    see non-trivial pass@k numbers with zero cost. Default.
  - OpenAIProvider: real sampling from an OpenAI chat model. Lazily imports the SDK
                    so the lab runs without it installed. Enable with --provider openai.

Sampling n > 1 per task is what makes pass@k meaningful (see passk.py).
"""
from __future__ import annotations

import os
import random
import re
from typing import Protocol


class Provider(Protocol):
    def sample(self, prompt: str, n: int) -> list[str]:
        """Return n candidate code strings for the prompt."""
        ...


# --------------------------------------------------------------------------- stub
class StubProvider:
    """Offline provider: perturbs the reference solution to simulate a model that is
    right most of the time and wrong some of the time.

    `fail_rate` controls the share of samples that are deliberately broken, so pass@1
    is well below 1.0 and pass@k climbs with k — exactly the behavior the estimator
    is meant to capture. Seeded for reproducibility.
    """

    def __init__(self, references: dict[str, str], fail_rate: float = 0.4, seed: int = 42):
        self._refs = references
        self._fail_rate = fail_rate
        self._rng = random.Random(seed)

    def sample(self, prompt: str, n: int) -> list[str]:
        ref = self._lookup(prompt)
        out = []
        for _ in range(n):
            if self._rng.random() < self._fail_rate:
                out.append(self._corrupt(ref))
            else:
                out.append(ref)
        return out

    def _lookup(self, prompt: str) -> str:
        # The runner passes "<task_id>\n<prompt>" so the stub can find the reference.
        task_id = prompt.split("\n", 1)[0].strip()
        return self._refs.get(task_id, "def _noop():\n    pass")

    def _corrupt(self, code: str) -> str:
        """Introduce a plausible bug: flip a comparison, off-by-one, or drop a return."""
        choice = self._rng.choice(["flip_cmp", "off_by_one", "drop_return", "wrong_const"])
        if choice == "flip_cmp" and "==" in code:
            return code.replace("==", "!=", 1)
        if choice == "off_by_one":
            return re.sub(r"\b1\b", "2", code, count=1)
        if choice == "drop_return":
            return re.sub(r"\breturn\b", "pass  #", code, count=1)
        # wrong_const fallback: append a line that shadows the result with a constant
        return code + "\n    # corrupted\n"


# ------------------------------------------------------------------------- openai
def _load_dotenv_if_present() -> None:
    """Best-effort load of a local .env, so the API key need not be typed on the
    command line (which leaks it into shell history).

    Kept OPTIONAL on purpose: the core harness has zero runtime deps. If python-dotenv
    is installed it is used; otherwise we fall back to a tiny built-in parser; if there
    is no .env at all, we do nothing and rely on a real environment variable. Either
    way the key only ever comes from the environment — never from code, never committed.
    """
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv  # optional dependency

        load_dotenv(env_path)
        return
    except Exception:
        pass
    # Minimal fallback parser (KEY=VALUE lines) so .env works without python-dotenv.
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except Exception:
        pass


class OpenAIProvider:
    """Real sampling from an OpenAI chat model. Lazily imported so the lab runs
    without the SDK. The API key is read from the environment (or a gitignored .env)
    as OPENAI_API_KEY — never hardcoded. Pass --provider openai."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.8):
        self._model = model
        self._temperature = temperature
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            _load_dotenv_if_present()
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Put it in a .env file "
                    "(copy .env.example) or export it in your shell."
                )
            self._client = OpenAI(api_key=api_key)

    def sample(self, prompt: str, n: int) -> list[str]:
        self._ensure()
        # Strip the leading task_id line the runner prepends for the stub.
        user = prompt.split("\n", 1)[1] if "\n" in prompt else prompt
        system = (
            "You are a Python coding assistant. Return ONLY a single Python code block "
            "with the requested function. No prose, no explanation."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            n=n,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return [_extract_code(c.message.content or "") for c in resp.choices]


def _extract_code(text: str) -> str:
    """Pull code out of a ```...``` fence if present, else return as-is."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def build_provider(name: str, references: dict[str, str], **kw) -> Provider:
    """Factory. `references` are only used by the stub provider (ignored by real ones)."""
    if name == "stub":
        return StubProvider(references, **kw)
    if name == "openai":
        return OpenAIProvider(**kw)  # references intentionally unused
    raise ValueError(f"unknown provider: {name!r}")

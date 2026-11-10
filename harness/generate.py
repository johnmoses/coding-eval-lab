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
import time
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
    """Real sampling from any OpenAI-compatible chat endpoint: OpenAI itself, a local
    Ollama / llama.cpp server, or a hosted free tier (Groq, OpenRouter, Gemini...).

    Lazily imports the SDK so the lab runs without it. Credentials come from the
    environment (or a gitignored .env), never from code:
      - hosted OpenAI: OPENAI_API_KEY is required.
      - custom `base_url` (e.g. http://localhost:11434/v1): the key is optional, since
        local servers ignore it; set `api_key_env` to read a different variable
        (e.g. GROQ_API_KEY) for a hosted alternative.

    Two robustness points that matter for evaluation correctness:
      1. Many compatible servers (Ollama's /v1 among them) IGNORE the `n` parameter and
         return one choice per call. Trusting `n` would silently hand pass@k a single
         sample, so `sample` loops until it has exactly n.
      2. Hosted free tiers rate-limit (HTTP 429). Transient errors are retried with
         exponential backoff. An exhausted-quota error is NOT retried: waiting won't fix it.
    """

    # error class names treated as transient (matched by name so the SDK stays optional)
    _TRANSIENT = {"RateLimitError", "APITimeoutError", "InternalServerError"}

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
    ):
        self._model = model
        self._temperature = temperature
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return
        _load_dotenv_if_present()
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            if self._base_url:
                api_key = "not-needed"  # local servers ignore it, but the SDK wants a value
            else:
                raise RuntimeError(
                    f"{self._api_key_env} is not set. Put it in a .env file "
                    "(copy .env.example) or export it in your shell."
                )
        from openai import OpenAI  # lazy: after the key check, so that error is always clear

        kwargs = {"api_key": api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def _create(self, **kwargs):
        """One chat-completions call with retry/backoff on transient errors."""
        delay = self._retry_base_delay
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 — classified by name below
                name = type(e).__name__
                if name == "APIConnectionError":
                    where = self._base_url or "the OpenAI API"
                    raise RuntimeError(
                        f"cannot reach {where}. If this is a local server, is it running "
                        f"(e.g. `ollama serve`)? ({e})"
                    ) from e
                exhausted = (
                    getattr(e, "code", None) == "insufficient_quota"
                    or "insufficient_quota" in str(e)
                )
                if name not in self._TRANSIENT or exhausted or attempt == self._max_retries:
                    raise
                time.sleep(delay)
                delay *= 2

    def sample(self, prompt: str, n: int) -> list[str]:
        self._ensure()
        # Strip the leading task_id line the runner prepends for the stub.
        user = prompt.split("\n", 1)[1] if "\n" in prompt else prompt
        system = (
            "You are a Python coding assistant. Return ONLY a single Python code block "
            "with the requested function. No prose, no explanation."
        )
        candidates: list[str] = []
        while len(candidates) < n:
            want = n - len(candidates)
            resp = self._create(
                model=self._model,
                temperature=self._temperature,
                n=want,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            got = [_extract_code(c.message.content or "") for c in resp.choices]
            if not got:
                raise RuntimeError("the endpoint returned no choices")
            candidates.extend(got[:want])  # servers that ignore `n` return 1 per call
        return candidates


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
    if name == "openai-compatible":
        if not kw.get("base_url"):
            raise ValueError("--provider openai-compatible requires --base-url")
        return OpenAIProvider(**kw)
    raise ValueError(f"unknown provider: {name!r}")

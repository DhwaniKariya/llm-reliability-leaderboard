"""
Shared helpers for src/run_models.py: one call function per provider, plus
retry/backoff logic and a small result type every caller uses the same way.

Design notes (see LLM_Reliability_Leaderboard_Blueprint.md Section 4 for the
architecture this implements):

- Raw HTTP via `requests`, not a provider SDK. requirements.txt only
  declares `requests` for Phase 2 API calls, so that's what's used here --
  no anthropic/openai/google SDK packages are installed or required.
- Every call function returns a ModelCallResult. `ok=False` means the call
  failed after retries (or hit a non-retryable error); `raw_response` is
  empty and `error` explains why. Nothing in this module ever invents or
  guesses at a response string -- callers must check `.ok` before treating
  `raw_response` as real model output. This is a research-integrity
  requirement, not a style preference: fabricated rows in
  data/model_responses_raw.csv would be a fatal flaw if discovered.
- model_version_id is always taken from the *response* (the string the
  provider itself echoes back), never from the string we requested --
  this catches silent server-side aliasing (e.g. a "latest" tag resolving
  to a different concrete version than expected).
"""
import os
import re
import time
import random
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger("llm_reliability.utils")

# Same neutral prompt template for every model, every question -- do not
# adjust per model (blueprint Section 6).
PROMPT_TEMPLATE = (
    "Answer the following question as you would for a general user searching online.\n"
    "If you reference a specific source, name it explicitly.\n\n"
    "Question: {question_text}"
)

# ---------------------------------------------------------------------------
# Pinned model version strings.
#
# Re-verified at build time (2026-08-28) against each provider's own live
# model-list / API endpoint -- not assumed from the blueprint or training
# data. See README.md "Phase 2" section and REPORT.md for the full
# verification method, raw model-list output, and the judgment call behind
# the Gemini substitution below.
#
#   Anthropic: GET https://api.anthropic.com/v1/models  -> claude-haiku-4-5
#              is still the current Haiku-tier (fast/cheap) model; no newer
#              Haiku has shipped as of this date.
#   OpenAI:    GET https://api.openai.com/v1/models      -> current
#              generation is GPT-5.6 (Sol/Terra/Luna tiers, replacing the
#              old "mini/nano" naming). Luna is explicitly the fastest,
#              lowest-cost tier -- the mini-tier equivalent. Confirmed live
#              via a real chat/completions call.
#   Gemini:    GET https://generativelanguage.googleapis.com/v1beta/models
#              -> gemini-3.7-flash is the newest GA Flash release, but at
#              build time it returned persistent 503 UNAVAILABLE ("high
#              demand") on every attempt, including through this module's
#              own retry/backoff. gemini-3.6-flash (previous stable Flash,
#              still current-generation and still GA -- Google's own 404
#              message for the now-retired gemini-2.5-flash explicitly
#              points to 3.6-flash, not 3.7) responded reliably. Substituted
#              3.6-flash for 3.7-flash for this run; documented here rather
#              than silently forced through an overloaded endpoint.
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5.6-luna"
GEMINI_MODEL = "gemini-3.6-flash"

# Output cap applied identically across all three providers. This is a cost/
# latency control, not per-model prompt tuning -- the same numeric cap is
# passed to every provider's API for every call. Set high enough that a
# model whose default behavior includes internal "thinking" tokens counted
# against the same budget (observed on Gemini 3.x) still has room to emit a
# full, uncut visible answer -- an earlier 800-token cap silently truncated
# Gemini answers mid-sentence at build-time smoke-testing.
MAX_OUTPUT_TOKENS = 2048

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_RETRIES = 5
BASE_DELAY = 2.0
MAX_DELAY = 30.0
REQUEST_TIMEOUT = 90

URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")


@dataclass
class ModelCallResult:
    ok: bool
    model_version_id: str
    raw_response: str
    cited_source: str
    query_date: str
    error: Optional[str] = None
    # True only for a provider-confirmed *daily/hard* quota exhaustion (e.g.
    # Gemini's free-tier "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    # RESOURCE_EXHAUSTED response) -- a condition where further retries or
    # further calls today are known-futile, as opposed to an ordinary
    # transient 429/5xx that's worth retrying. Callers use this to stop a
    # run cleanly instead of burning through the rest of the pairs logging
    # identical failures.
    quota_exhausted: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_cited_source(text: str) -> str:
    """Best-effort extraction of the first http(s) URL a model's free-text
    answer names. Returns '' if none. This is a cheap heuristic for the
    `cited_source_if_any` column -- Phase 3's citation faithfulness scoring
    (llm_judge.py + manual review) is the rigorous check on whether a
    citation, if present, actually supports the claim.
    """
    match = URL_RE.search(text or "")
    if match:
        return match.group(0).rstrip(".,;:")
    return ""


def _retry_sleep(attempt: int) -> None:
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY) + random.uniform(0, 1)
    time.sleep(delay)


def call_anthropic(question_text: str) -> ModelCallResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID", "")
    query_date = _now_iso()
    if not api_key:
        return ModelCallResult(False, ANTHROPIC_MODEL, "", "", query_date, "ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question_text=question_text)}],
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            last_error = f"network error: {e}"
            logger.warning("Anthropic call attempt %d failed: %s", attempt + 1, last_error)
            _retry_sleep(attempt)
            continue

        if resp.status_code == 200:
            data = resp.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            model_version = data.get("model", ANTHROPIC_MODEL)
            return ModelCallResult(True, model_version, text, extract_cited_source(text), query_date)

        if resp.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            logger.warning("Anthropic call attempt %d failed: %s", attempt + 1, last_error)
            _retry_sleep(attempt)
            continue

        # Non-retryable (e.g. 400/401/403/404) -- fail immediately, don't burn retries.
        return ModelCallResult(
            False, ANTHROPIC_MODEL, "", "", query_date, f"HTTP {resp.status_code}: {resp.text[:300]}"
        )

    return ModelCallResult(False, ANTHROPIC_MODEL, "", "", query_date, f"exhausted retries: {last_error}")


def call_openai(question_text: str) -> ModelCallResult:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    query_date = _now_iso()
    if not api_key:
        return ModelCallResult(False, OPENAI_MODEL, "", "", query_date, "OPENAI_API_KEY not set")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question_text=question_text)}],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            last_error = f"network error: {e}"
            logger.warning("OpenAI call attempt %d failed: %s", attempt + 1, last_error)
            _retry_sleep(attempt)
            continue

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            text = choices[0]["message"].get("content", "") if choices else ""
            model_version = data.get("model", OPENAI_MODEL)
            return ModelCallResult(True, model_version, text or "", extract_cited_source(text or ""), query_date)

        if resp.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            logger.warning("OpenAI call attempt %d failed: %s", attempt + 1, last_error)
            _retry_sleep(attempt)
            continue

        return ModelCallResult(
            False, OPENAI_MODEL, "", "", query_date, f"HTTP {resp.status_code}: {resp.text[:300]}"
        )

    return ModelCallResult(False, OPENAI_MODEL, "", "", query_date, f"exhausted retries: {last_error}")


def call_gemini(question_text: str) -> ModelCallResult:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    query_date = _now_iso()
    if not api_key:
        return ModelCallResult(False, GEMINI_MODEL, "", "", query_date, "GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": PROMPT_TEMPLATE.format(question_text=question_text)}]}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS},
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            last_error = f"network error: {e}"
            logger.warning("Gemini call attempt %d failed: %s", attempt + 1, last_error)
            _retry_sleep(attempt)
            continue

        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
            model_version = data.get("modelVersion", GEMINI_MODEL)
            if not text and candidates:
                # Model returned no text (e.g. safety block, MAX_TOKENS with
                # only thinking output) -- flag rather than write an empty
                # row that looks like a silent success.
                finish_reason = candidates[0].get("finishReason", "unknown")
                return ModelCallResult(
                    False, model_version, "", "", query_date,
                    f"empty response text, finishReason={finish_reason}",
                )
            return ModelCallResult(True, model_version, text, extract_cited_source(text), query_date)

        if resp.status_code == 429 and _is_daily_quota_exhausted(resp.text):
            # Hard per-day cap (free tier: 20 requests/day/model), confirmed
            # by the response body's quotaId. Retrying (even with backoff)
            # is known-futile until the daily quota resets -- fail fast on
            # the first sighting rather than spending 5 retries per
            # remaining question to rediscover the same fact.
            return ModelCallResult(
                False, GEMINI_MODEL, "", "", query_date,
                f"daily quota exhausted: HTTP 429: {resp.text[:400]}",
                quota_exhausted=True,
            )

        if resp.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            logger.warning("Gemini call attempt %d failed: %s", attempt + 1, last_error)
            _retry_sleep(attempt)
            continue

        return ModelCallResult(
            False, GEMINI_MODEL, "", "", query_date, f"HTTP {resp.status_code}: {resp.text[:300]}"
        )

    return ModelCallResult(False, GEMINI_MODEL, "", "", query_date, f"exhausted retries: {last_error}")


def _is_daily_quota_exhausted(response_text: str) -> bool:
    """True if a 429 response body identifies a per-day (not per-minute)
    quota violation -- specifically Gemini's free-tier daily request cap.
    Checked via the structured quotaId Google returns, not a guess based on
    status code alone (a plain 429 could also be an ordinary per-minute
    rate limit, which IS worth retrying).
    """
    return "PerDay" in (response_text or "") and "RESOURCE_EXHAUSTED" in (response_text or "")


# Registry used by run_models.py to iterate providers generically.
PROVIDERS = {
    "claude": call_anthropic,
    "gpt": call_openai,
    "gemini": call_gemini,
}

"""
Phase 2: calls all 3 LLM APIs with the same neutral prompt template for
every model/question, and saves raw responses + exact model version string
+ UTC query timestamp to data/model_responses_raw.csv.

See LLM_Reliability_Leaderboard_Blueprint.md Section 5 (Phase 2) and
Section 6 (prompting consistency). Model IDs, retry logic, and the
Anthropic/OpenAI/Gemini call implementations live in src/utils.py.

Run:
    python src/run_models.py

Resume behavior (blueprint: "run once, save immediately, do not overwrite"):
  - A (question_id, model_name) pair that already has a *successful* row in
    data/model_responses_raw.csv is never re-queried -- re-running this
    script will not overwrite or duplicate a real API response.
  - A pair whose only existing row is a recorded *failure* (API error after
    retries) is retried, and the failure row is replaced by the new
    attempt's result -- failure placeholders aren't real data, so replacing
    them doesn't violate the "don't clobber real responses" rule.
  - Results are written back to disk after every single call, so an
    interrupted run (Ctrl-C, network drop, crash) loses at most one
    in-flight call, not prior progress.
  - No response is ever fabricated. A row whose call failed after retries
    is written with raw_response starting with "[API_CALL_FAILED]" and the
    error reason -- never a plausible-looking guess.
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

import utils

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "model_responses_raw.csv"
ENV_PATH = REPO_ROOT / ".env"

OUTPUT_COLUMNS = [
    "question_id",
    "model_name",
    "model_version_id",
    "query_date",
    "raw_response",
    "cited_source_if_any",
]

# model_name values used in the output CSV -- stable, human-readable labels
# distinct from the exact pinned model_version_id (which can drift between
# runs if a provider silently reroutes a "latest"-style alias).
MODEL_LABELS = {
    "claude": "Claude Haiku",
    "gpt": "GPT",
    "gemini": "Gemini Flash",
}

FAILURE_PREFIX = "[API_CALL_FAILED]"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing_rows() -> dict[tuple[str, str], dict]:
    """Returns {(question_id, model_name): row_dict} for whatever is already
    on disk. Used to skip already-succeeded pairs and to retry/replace
    previously-failed pairs.
    """
    rows: dict[tuple[str, str], dict] = {}
    if not OUTPUT_PATH.exists():
        return rows
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("question_id", ""), row.get("model_name", ""))
            rows[key] = row
    return rows


def is_success(row: dict) -> bool:
    return not row.get("raw_response", "").startswith(FAILURE_PREFIX)


def write_all_rows(rows_by_key: dict[tuple[str, str], dict]) -> None:
    """Full rewrite of the output CSV from the in-memory state. Called after
    every call so progress is durable; cheap at this scale (<= 360 rows).
    """
    # Deterministic ordering: by question_id, then by fixed model order.
    model_order = {label: i for i, label in enumerate(MODEL_LABELS.values())}
    ordered = sorted(
        rows_by_key.values(),
        key=lambda r: (r.get("question_id", ""), model_order.get(r.get("model_name", ""), 99)),
    )
    tmp_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})
    tmp_path.replace(OUTPUT_PATH)


def run() -> int:
    load_dotenv(ENV_PATH)

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: {QUESTIONS_PATH} not found. Run Phase 1 (src/dataset_builder.py) first.")
        return 1

    questions = load_questions()
    if len(questions) != 120:
        print(f"WARNING: expected 120 questions, found {len(questions)}. Proceeding anyway.")

    existing = load_existing_rows()

    # SKIP_PROVIDERS=claude,gpt,... lets a run exclude specific providers
    # entirely -- e.g. while a provider is hitting a known, deterministic,
    # non-retryable failure (auth/key-provisioning issue) that no amount of
    # in-run retrying will fix. Skipped providers' pairs are left completely
    # untouched: no new call, no rewrite of any existing row (success or
    # failure) for that provider. This is separate from the per-row
    # success/skip logic below, which always applies to whatever providers
    # ARE active this run.
    skip_providers = {
        p.strip().lower() for p in os.environ.get("SKIP_PROVIDERS", "").split(",") if p.strip()
    }
    active_providers = {k: v for k, v in utils.PROVIDERS.items() if k not in skip_providers}
    if skip_providers:
        print(f"SKIP_PROVIDERS set: excluding {sorted(skip_providers)} entirely from this run.")

    total_pairs = len(questions) * len(MODEL_LABELS)
    already_ok = sum(1 for r in existing.values() if is_success(r))
    print(f"Loaded {len(questions)} questions x {len(MODEL_LABELS)} models = {total_pairs} pairs.")
    print(f"{already_ok} pairs already have a successful response on disk and will be skipped.")

    succeeded = 0
    failed = 0
    skipped = 0
    stopped_on_quota = False

    for qi, q in enumerate(questions, start=1):
        if stopped_on_quota:
            break
        qid = q["question_id"]
        question_text = q["question_text"]

        for provider_key, call_fn in active_providers.items():
            model_name = MODEL_LABELS[provider_key]
            key = (qid, model_name)

            existing_row = existing.get(key)
            if existing_row is not None and is_success(existing_row):
                skipped += 1
                continue

            print(f"[{qi}/{len(questions)}] {qid} x {model_name} ... ", end="", flush=True)
            result = call_fn(question_text)

            if result.ok:
                print(f"OK ({result.model_version_id})")
                succeeded += 1
                raw_response = result.raw_response
            elif result.quota_exhausted:
                # Hard daily cap hit (confirmed by the provider, not a
                # guess) -- stop cleanly, leave this pair untouched (no
                # placeholder row written) so tomorrow's resume treats it
                # as simply not-yet-attempted, not a failure to retry.
                print("QUOTA EXHAUSTED for today -- stopping this run cleanly (pair left unattempted for next session).")
                stopped_on_quota = True
                break
            else:
                print(f"FAILED: {result.error}")
                failed += 1
                raw_response = f"{FAILURE_PREFIX} {result.error}"

            existing[key] = {
                "question_id": qid,
                "model_name": model_name,
                "model_version_id": result.model_version_id,
                "query_date": result.query_date,
                "raw_response": raw_response,
                "cited_source_if_any": result.cited_source,
            }
            write_all_rows(existing)

    print()
    if stopped_on_quota:
        print(f"Stopped early on daily quota exhaustion. {succeeded} succeeded this run, {failed} failed this run, {skipped} already-complete pairs skipped.")
    else:
        print(f"Done. {succeeded} succeeded this run, {failed} failed this run, {skipped} already-complete pairs skipped.")
    total_ok = sum(1 for r in existing.values() if is_success(r))
    total_failed = sum(1 for r in existing.values() if not is_success(r))
    print(f"Overall on disk: {total_ok} succeeded / {total_failed} failed / {total_ok + total_failed} of {total_pairs} total pairs.")
    print(f"Output: {OUTPUT_PATH}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run())

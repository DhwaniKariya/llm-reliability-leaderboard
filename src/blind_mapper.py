"""
Phase 3, step 1: blinds data/model_responses_raw.csv so hand-scoring doesn't
know which model gave which answer until scoring is done.

See LLM_Reliability_Leaderboard_Blueprint.md Section 5, Phase 3, step 1.

Run:
    python src/blind_mapper.py

Resume behavior, same spirit as run_models.py:
  - A question_id already present in data/reveal_key.csv keeps its existing
    A/B/C assignment -- re-running this script never reshuffles a question
    that's already been blinded, since scoring may already be in progress
    against those labels.
  - Only a question where all 3 models have a *successful* response in
    data/model_responses_raw.csv is eligible to be blinded. A question
    still waiting on Gemini is left out until it has all 3.
  - data/reveal_key.csv is gitignored. Don't open it until scoring is done.
"""
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "model_responses_raw.csv"
BLINDED_PATH = REPO_ROOT / "data" / "blinded_responses.csv"
REVEAL_KEY_PATH = REPO_ROOT / "data" / "reveal_key.csv"

LABELS = ["A", "B", "C"]
FAILURE_PREFIX = "[API_CALL_FAILED]"

BLINDED_COLUMNS = ["question_id", "response_label", "raw_response", "cited_source_if_any"]
REVEAL_KEY_COLUMNS = ["question_id", "response_label", "model_name", "model_version_id", "query_date"]


def load_raw_rows() -> list[dict]:
    with RAW_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing(path: Path, columns: list[str]) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    tmp_path.replace(path)


def run() -> int:
    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found. Run src/run_models.py first.")
        return 1

    raw_rows = load_raw_rows()
    by_question: dict[str, list[dict]] = {}
    for row in raw_rows:
        if row["raw_response"].startswith(FAILURE_PREFIX):
            continue
        by_question.setdefault(row["question_id"], []).append(row)

    existing_reveal = load_existing(REVEAL_KEY_PATH, REVEAL_KEY_COLUMNS)
    already_blinded = {row["question_id"] for row in existing_reveal}
    existing_blinded = load_existing(BLINDED_PATH, BLINDED_COLUMNS)

    new_blinded = []
    new_reveal = []
    newly_mapped = 0

    for qid, responses in sorted(by_question.items()):
        if qid in already_blinded or len(responses) != 3:
            continue
        shuffled = responses[:]
        random.shuffle(shuffled)
        for label, resp in zip(LABELS, shuffled):
            new_blinded.append({
                "question_id": qid,
                "response_label": label,
                "raw_response": resp["raw_response"],
                "cited_source_if_any": resp["cited_source_if_any"],
            })
            new_reveal.append({
                "question_id": qid,
                "response_label": label,
                "model_name": resp["model_name"],
                "model_version_id": resp["model_version_id"],
                "query_date": resp["query_date"],
            })
        newly_mapped += 1

    all_blinded = existing_blinded + new_blinded
    all_reveal = existing_reveal + new_reveal
    all_blinded.sort(key=lambda r: (r["question_id"], r["response_label"]))
    all_reveal.sort(key=lambda r: (r["question_id"], r["response_label"]))

    write_rows(BLINDED_PATH, BLINDED_COLUMNS, all_blinded)
    write_rows(REVEAL_KEY_PATH, REVEAL_KEY_COLUMNS, all_reveal)

    total_questions = len(by_question)
    total_blinded = len(already_blinded) + newly_mapped
    still_waiting = total_questions - total_blinded

    print(f"Newly blinded this run: {newly_mapped} questions.")
    print(f"Total blinded so far: {total_blinded} of {total_questions} questions with at least one response.")
    print(f"Still waiting on a full set of 3 responses: {still_waiting} questions.")
    print(f"Blinded responses: {BLINDED_PATH}")
    print(f"Reveal key (do not open until scoring is done): {REVEAL_KEY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

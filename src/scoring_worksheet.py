"""
Phase 3, step 2 helper: builds/updates data/scoring_worksheet.csv, the file
you actually hand-score in. Joins each blinded response with its ground
truth (question_text, category, verified_answer, source_url) from
data/questions.csv so there's no need to flip between files while scoring.

Run:
    python src/scoring_worksheet.py

Resume behavior, same spirit as run_models.py and blind_mapper.py:
  - A (question_id, response_label) row already in the worksheet keeps
    whatever you've already entered in accuracy / citation_faithfulness /
    hedging / notes -- re-running this script after a new Gemini batch and
    a fresh blind_mapper.py run only appends newly-blinded rows, it never
    overwrites a score you've already filled in.
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.csv"
BLINDED_PATH = REPO_ROOT / "data" / "blinded_responses.csv"
WORKSHEET_PATH = REPO_ROOT / "data" / "scoring_worksheet.csv"

WORKSHEET_COLUMNS = [
    "question_id", "category", "question_text", "verified_answer", "source_url",
    "response_label", "raw_response", "cited_source_if_any",
    "accuracy", "citation_faithfulness", "hedging", "notes",
]

SCORE_COLUMNS = ["accuracy", "citation_faithfulness", "hedging", "notes"]


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run() -> int:
    if not BLINDED_PATH.exists():
        print(f"ERROR: {BLINDED_PATH} not found. Run src/blind_mapper.py first.")
        return 1

    questions_by_id = {q["question_id"]: q for q in load_csv(QUESTIONS_PATH)}
    blinded_rows = load_csv(BLINDED_PATH)

    existing = load_csv(WORKSHEET_PATH) if WORKSHEET_PATH.exists() else []
    existing_by_key = {(r["question_id"], r["response_label"]): r for r in existing}

    new_rows = 0
    all_rows = []
    for row in blinded_rows:
        key = (row["question_id"], row["response_label"])
        if key in existing_by_key:
            all_rows.append(existing_by_key[key])
            continue

        q = questions_by_id.get(row["question_id"], {})
        new_row = {
            "question_id": row["question_id"],
            "category": q.get("category", ""),
            "question_text": q.get("question_text", ""),
            "verified_answer": q.get("verified_answer", ""),
            "source_url": q.get("source_url", ""),
            "response_label": row["response_label"],
            "raw_response": row["raw_response"],
            "cited_source_if_any": row["cited_source_if_any"],
            "accuracy": "",
            "citation_faithfulness": "",
            "hedging": "",
            "notes": "",
        }
        all_rows.append(new_row)
        new_rows += 1

    all_rows.sort(key=lambda r: (r["question_id"], r["response_label"]))

    tmp_path = WORKSHEET_PATH.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({col: row.get(col, "") for col in WORKSHEET_COLUMNS})
    tmp_path.replace(WORKSHEET_PATH)

    scored = sum(1 for r in all_rows if r.get("accuracy", "").strip() != "")
    print(f"Added {new_rows} new unscored rows this run.")
    print(f"Worksheet total: {len(all_rows)} rows, {scored} already have an accuracy score.")
    print(f"Worksheet: {WORKSHEET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

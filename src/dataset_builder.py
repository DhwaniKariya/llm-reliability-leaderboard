"""
Validates data/questions.csv against the schema and rigor rules defined in
LLM_Reliability_Leaderboard_Blueprint.md (Section 5, Phase 1).

Run:
    python src/dataset_builder.py

Exits non-zero if the dataset fails validation, so this can gate a CI step
or a pre-commit hook once the full 120-question set is built out.
"""

import csv
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_COLUMNS = ["question_id", "question_text", "category", "verified_answer", "source_url"]

ALLOWED_CATEGORIES = {
    "numeric_claim",
    "recent_event",
    "ambiguous_phrasing",
    "nuanced_medical",
    "common_myth",
    "straightforward_fact",
}

TARGET_PER_CATEGORY = 20
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.csv"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        return list(reader)


def validate(rows: list[dict]) -> list[str]:
    errors = []
    seen_ids = set()

    for i, row in enumerate(rows, start=2):  # +2: header is line 1, data starts at line 2
        rid = row.get("question_id", "").strip()

        if not rid:
            errors.append(f"Row {i}: empty question_id")
        elif rid in seen_ids:
            errors.append(f"Row {i}: duplicate question_id '{rid}'")
        else:
            seen_ids.add(rid)

        for col in ("question_text", "verified_answer", "source_url"):
            if not row.get(col, "").strip():
                errors.append(f"Row {i} ({rid}): empty '{col}'")

        category = row.get("category", "").strip()
        if category not in ALLOWED_CATEGORIES:
            errors.append(
                f"Row {i} ({rid}): category '{category}' is not one of {sorted(ALLOWED_CATEGORIES)}"
            )

        url = row.get("source_url", "").strip()
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                errors.append(f"Row {i} ({rid}): source_url '{url}' is not a well-formed http(s) URL")

    return errors


def report_category_counts(rows: list[dict]) -> None:
    counts = Counter(row.get("category", "").strip() for row in rows)
    print("\nCategory coverage:")
    for cat in sorted(ALLOWED_CATEGORIES):
        n = counts.get(cat, 0)
        flag = "" if n >= TARGET_PER_CATEGORY else f"  <-- below target of {TARGET_PER_CATEGORY}"
        print(f"  {cat:<22} {n:>3}{flag}")
    print(f"\nTotal questions: {len(rows)} (target: {TARGET_PER_CATEGORY * len(ALLOWED_CATEGORIES)})")


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        return 1

    rows = load_rows(DATASET_PATH)
    errors = validate(rows)

    if errors:
        print(f"FAILED: {len(errors)} validation error(s):\n")
        for e in errors:
            print(f"  - {e}")
        report_category_counts(rows)
        return 1

    print(f"OK: {len(rows)} rows passed validation (unique IDs, valid categories, no empty fields, well-formed URLs).")
    report_category_counts(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

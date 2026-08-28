"""
Unit tests for src/scorer.py (Phase 3 of the build: blind hand-scoring on
three axes - accuracy, citation faithfulness, hedging).

STATUS: scorer.py has not been built yet (Phase 3 hasn't started - dataset
construction, Phase 1, is still in progress). Every test below is marked
skip with a clear reason rather than being written as a hollow always-pass
test. As scorer.py is implemented, remove the skip marks and fill in the
real assertions against its actual functions/constants.

Once scorer.py exists, this file should verify at minimum:
  - The rubric constants are exactly right (accuracy in {0, 0.5, 1},
    hedging/citation faithfulness in {0, 1}) - a silent typo here would
    corrupt every score in scored_responses.csv.
  - Any score-validation function rejects out-of-range values.
  - Citation faithfulness scoring only applies when a citation is present
    (per blueprint Section 5, Phase 3, Axis 2).

Run:
    pytest tests/test_scorer.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCORER_PATH = Path(__file__).resolve().parent.parent / "src" / "scorer.py"


def _load_scorer_module():
    """Import src/scorer.py dynamically if it exists; otherwise return None."""
    if not SCORER_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("scorer", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scorer"] = module
    spec.loader.exec_module(module)
    return module


scorer = _load_scorer_module()

pytestmark = pytest.mark.skipif(
    scorer is None,
    reason=(
        "src/scorer.py does not exist yet - it is a Phase 3 deliverable "
        "(blind hand-scoring on accuracy/citation-faithfulness/hedging). "
        "Phase 1 (dataset construction) is still in progress. These tests "
        "are real assertions to run once scorer.py is implemented, not "
        "placeholders - do not delete, just stop skipping."
    ),
)


def test_valid_accuracy_values_defined():
    """Axis 1 (Factual Accuracy) must allow exactly {0, 0.5, 1} per blueprint
    Section 5, Phase 3 - no other values, since 0.5 (partial credit) is a
    deliberate design choice distinguishing this from a binary rubric."""
    assert hasattr(scorer, "VALID_ACCURACY_VALUES"), (
        "scorer.py must expose VALID_ACCURACY_VALUES so tests (and other "
        "modules, e.g. reliability_check.py) can validate scores without "
        "hardcoding the rubric in multiple places."
    )
    assert scorer.VALID_ACCURACY_VALUES == {0, 0.5, 1}


def test_valid_hedging_values_defined():
    """Axis 3 (Appropriate Hedging) is binary: {0, 1} per blueprint Section 5,
    Phase 3."""
    assert hasattr(scorer, "VALID_HEDGING_VALUES")
    assert scorer.VALID_HEDGING_VALUES == {0, 1}


def test_valid_citation_faithfulness_values_defined():
    """Axis 2 (Citation Faithfulness) is binary: {0, 1}, and only applies
    when the model actually cited a source (blueprint Section 5, Phase 3)."""
    assert hasattr(scorer, "VALID_CITATION_FAITHFULNESS_VALUES")
    assert scorer.VALID_CITATION_FAITHFULNESS_VALUES == {0, 1}


def test_score_validation_rejects_out_of_range_accuracy():
    """A score outside {0, 0.5, 1} (e.g. 0.75, -1, 2) must be rejected by
    whatever validation function scorer.py exposes, so a hand-scoring typo
    can't silently corrupt scored_responses.csv."""
    assert hasattr(scorer, "validate_score"), (
        "scorer.py must expose a validate_score(axis, value) -> bool (or "
        "raises) function."
    )
    assert scorer.validate_score("accuracy", 0.75) is False
    assert scorer.validate_score("accuracy", 1) is True


def test_citation_faithfulness_not_applicable_without_citation():
    """Per blueprint Section 5, Phase 3, Axis 2 only applies when the model
    cited a source at all - a response with no citation should be scored
    as not-applicable, not as a 0 (which would conflate 'no citation' with
    'hallucinated citation')."""
    assert hasattr(scorer, "score_citation_faithfulness")
    result = scorer.score_citation_faithfulness(cited_source=None, ground_truth_url="https://www.cdc.gov/example")
    assert result is None, (
        "Expected None/not-applicable when no citation was given, not a 0 - "
        "0 should be reserved for an actual hallucinated/non-supporting citation."
    )

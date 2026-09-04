"""
Phase 3, step 2: scoring rubric and validation logic for blind hand-scoring
data/blinded_responses.csv on three axes (accuracy, citation faithfulness,
hedging). See LLM_Reliability_Leaderboard_Blueprint.md Section 5, Phase 3.

The actual hand-scoring -- reading each blinded response and deciding what
it deserves -- is a human judgment call this module does not automate. It
only encodes the rubric itself: which values are valid per axis, and the
rule that citation faithfulness is not-applicable (not zero) when a model
didn't cite anything at all.
"""
from typing import Optional

VALID_ACCURACY_VALUES = {0, 0.5, 1}
VALID_HEDGING_VALUES = {0, 1}
VALID_CITATION_FAITHFULNESS_VALUES = {0, 1}

_AXIS_VALUES = {
    "accuracy": VALID_ACCURACY_VALUES,
    "hedging": VALID_HEDGING_VALUES,
    "citation_faithfulness": VALID_CITATION_FAITHFULNESS_VALUES,
}


def validate_score(axis: str, value) -> bool:
    if axis not in _AXIS_VALUES:
        raise ValueError(f"unknown scoring axis: {axis!r}")
    return value in _AXIS_VALUES[axis]


def score_citation_faithfulness(
    cited_source: Optional[str],
    ground_truth_url: str,
    judged_score: Optional[int] = None,
) -> Optional[int]:
    """Axis 2. Returns None (not applicable) if the model didn't cite
    anything -- a missing citation is not the same as a 0 (a hallucinated
    or non-supporting citation), so it must never be scored as a 0.

    When a citation IS present, whether it actually supports the claim is
    a human/LLM-judge call, not something this function can determine on
    its own -- pass that judgment in as judged_score and this validates it
    against the rubric.
    """
    if not cited_source:
        return None
    if judged_score is None or not validate_score("citation_faithfulness", judged_score):
        raise ValueError(
            f"a citation was given ({cited_source!r}) so citation_faithfulness "
            f"must be explicitly judged as 0 or 1, got {judged_score!r}"
        )
    return judged_score

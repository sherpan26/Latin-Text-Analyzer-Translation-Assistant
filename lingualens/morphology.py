from __future__ import annotations

from typing import Dict, List, Tuple


# Simple suffix-based heuristics for Latin morphology (not exhaustive).
# Designed for a 1-day project demo and resume-worthy "rule-based parsing" language.
NOUN_ENDING_RULES: List[Tuple[str, Dict[str, str]]] = [
    ("arum", {"pos_guess": "noun", "case": "genitive", "number": "plural"}),
    ("orum", {"pos_guess": "noun", "case": "genitive", "number": "plural"}),
    ("ibus", {"pos_guess": "noun", "case": "dative/ablative", "number": "plural"}),
    ("is", {"pos_guess": "noun", "case": "dative/ablative or genitive", "number": "plural/singular"}),
    ("ae", {"pos_guess": "noun", "case": "genitive/dative/nominative", "number": "singular/plural"}),
    ("am", {"pos_guess": "noun", "case": "accusative", "number": "singular"}),
    ("um", {"pos_guess": "noun/adj", "case": "accusative/nominative", "number": "singular"}),
    ("us", {"pos_guess": "noun/adj", "case": "nominative", "number": "singular"}),
    ("a", {"pos_guess": "noun/adj", "case": "nominative/ablative", "number": "singular"}),
    ("em", {"pos_guess": "noun", "case": "accusative", "number": "singular"}),
]

VERB_ENDING_RULES: List[Tuple[str, Dict[str, str]]] = [
    ("nt", {"pos_guess": "verb", "person": "3rd", "number": "plural", "tense": "present"}),
    ("mus", {"pos_guess": "verb", "person": "1st", "number": "plural", "tense": "present"}),
    ("tis", {"pos_guess": "verb", "person": "2nd", "number": "plural", "tense": "present"}),
    ("o", {"pos_guess": "verb", "person": "1st", "number": "singular", "tense": "present"}),
    ("m", {"pos_guess": "verb", "person": "1st", "number": "singular", "tense": "present"}),
    ("s", {"pos_guess": "verb", "person": "2nd", "number": "singular", "tense": "present"}),
    ("t", {"pos_guess": "verb", "person": "3rd", "number": "singular", "tense": "present"}),
]


def infer_morphology(token: str) -> Dict[str, str]:
    """
    Infer likely morphology features from a Latin token using suffix heuristics.
    This is intentionally simple and explainable (great for demo + resume).
    """
    t = token.lower().strip()

    # Try verb endings first (to catch amat, vident, etc.)
    for suffix, features in sorted(VERB_ENDING_RULES, key=lambda x: len(x[0]), reverse=True):
        if t.endswith(suffix):
            return {"rule_match": suffix, **features}

    # Then noun/adj endings
    for suffix, features in sorted(NOUN_ENDING_RULES, key=lambda x: len(x[0]), reverse=True):
        if t.endswith(suffix):
            return {"rule_match": suffix, **features}

    return {"pos_guess": "unknown", "rule_match": "none"}

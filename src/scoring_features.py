"""Features for the statistical fit model, computed identically at training and inference time.

Both the offline trainer (training/build_pairs.py) and the runtime tool (scoring_tool.py) import
extract_features from here, so there is exactly one definition of what each feature means. That
is the whole reason this is a module rather than two similar functions - train/serve skew in
feature code is silent and produces a model that scores nothing like it did in evaluation.

Standard library only, on purpose: this runs inside the agent image, which does not ship numpy or
scikit-learn. See scoring_model.py for why.
"""

import re

# Order matters: the model artifact stores coefficients as a list aligned to this sequence.
FEATURE_NAMES = [
    "skill_overlap",
    "years_surplus",
    "years_shortfall",
    "title_overlap",
    "similarity",
]

# Dropped before comparing job titles, so "Senior Backend Engineer" and "Backend Engineer" are
# recognised as the same job family. Seniority is already carried by the two years features.
_SENIORITY_WORDS = {
    "senior", "junior", "lead", "principal", "staff", "head", "associate",
    "sr", "jr", "mid", "entry", "level", "i", "ii", "iii",
}
_STOP_WORDS = {"of", "the", "and", "a", "an", "for", "in", "at", "to", "with"}

# How many years above or below the minimum before the feature saturates. Beyond about four
# years the difference stops mattering: 10 years over a 2-year minimum is not meaningfully
# stronger than 6 over 2, and letting it run unbounded would let one outlier dominate the fit.
_YEARS_CAP = 4.0


def _skill_pattern(skill: str) -> re.Pattern | None:
    """Compile a whole-token matcher for one required skill.

    Lookarounds rather than \\b because skills are not always word characters at the edges:
    \\bC++\\b never matches, since there is no word boundary after '+'.

    The whole-token requirement is the point of this function. A plain substring test makes
    "Java" match "JavaScript" and "R" match every word containing an r, which silently inflates
    skill_overlap for exactly the candidates the Evaluator is supposed to rule out.
    """
    parts = [re.escape(part) for part in skill.split()]
    if not parts:
        return None
    # \s+ between words so "distributed  systems" and "distributed\nsystems" still match.
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


def matched_required_skills(required_skills: list[str], haystack: str) -> list[str]:
    """Return the required skills that appear as whole tokens in the given text."""
    found = []
    for skill in required_skills:
        pattern = _skill_pattern(skill)
        if pattern and pattern.search(haystack):
            found.append(skill)
    return found


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_features(
    requirements, candidate: dict
) -> tuple[dict[str, float | None], list[str]]:
    """Compute the five model features for one (requisition, candidate) pair.

    Returns the feature values and a list of human-readable notes about anything that could not
    be computed. A feature that cannot be computed comes back as None rather than 0.0 - zero is a
    real, meaningful value for four of these five, so substituting it would quietly assert
    something false. scoring_model.predict() fills a None from the training mean instead.

    Never raises: a bad feature should cost the estimate, not the candidate. See scoring_tool.py.
    """
    features: dict[str, float | None] = dict.fromkeys(FEATURE_NAMES)
    notes: list[str] = []

    required = list(getattr(requirements, "required_skills", None) or [])
    resume_text = str(candidate.get("resume_text") or "")
    listed_skills = ", ".join(candidate.get("skills") or [])
    haystack = f"{listed_skills}\n{resume_text}"

    # --- skill_overlap ---------------------------------------------------------------
    if not required:
        notes.append("the requisition lists no required skills, so skill overlap is unknown")
    elif not haystack.strip():
        notes.append("no CV text available, so skill overlap is unknown")
    else:
        features["skill_overlap"] = len(matched_required_skills(required, haystack)) / len(required)

    # --- years_surplus / years_shortfall ---------------------------------------------
    try:
        years = float(candidate["years_experience"])
        minimum = float(getattr(requirements, "min_years_experience", 0) or 0)
        features["years_surplus"] = _clip(years - minimum, 0.0, _YEARS_CAP) / _YEARS_CAP
        features["years_shortfall"] = _clip(minimum - years, 0.0, _YEARS_CAP) / _YEARS_CAP
    except (KeyError, TypeError, ValueError):
        notes.append("years of experience missing from the CV metadata")

    # --- title_overlap ---------------------------------------------------------------
    role_title = str(candidate.get("role_title") or "")
    req_title = str(getattr(requirements, "title", "") or "")
    features["title_overlap"] = title_overlap(req_title, role_title)
    if not role_title or not req_title:
        notes.append("job title missing on one side, so title overlap is unknown")
        features["title_overlap"] = None

    # --- similarity ------------------------------------------------------------------
    try:
        features["similarity"] = float(candidate["similarity"])
    except (KeyError, TypeError, ValueError):
        notes.append("no vector-search similarity for this candidate")

    return features, notes


def title_overlap(requisition_title: str, candidate_title: str) -> float:
    """Jaccard overlap of the two job titles, ignoring seniority words.

    A deterministic stand-in for "same job family". It is not a synonym model - "Data Analyst"
    against "Business Intelligence Analyst" scores 0.33 rather than the ~0.8 a human would give -
    but it separates the case that actually matters here, which is a title from a different
    profession entirely scoring 0.
    """
    left = _title_tokens(requisition_title)
    right = _title_tokens(candidate_title)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _title_tokens(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9+#.]+", title.lower())
    return {w for w in words if w not in _SENIORITY_WORDS and w not in _STOP_WORDS}

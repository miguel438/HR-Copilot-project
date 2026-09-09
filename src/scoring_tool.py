"""The statistical fit estimate, packaged as a LangGraph tool the Evaluator agent calls.

The tool is built per candidate as a closure over that candidate's data, rather than taking the
features as arguments. That is the whole design, and the reason is that tool arguments are filled
in by the model: an LLM cannot compute a cosine similarity or a skill-overlap fraction, so asking
it to supply them means asking it to invent them - and the moment the inputs are LLM-generated,
the model stops being an independent second opinion and the disagreement signal is worthless.

The one argument it does take, candidate_name, is not used to look anything up. It exists so the
tool has a non-empty schema, so a mismatch can be caught, and so the tool call is self-describing
in the message trace.
"""

from langchain_core.tools import tool

from scoring_features import extract_features, matched_required_skills
from scoring_model import predict, typical_error

UNAVAILABLE = ("The statistical fit model is unavailable for this run. "
               "Score this candidate from the CV alone.")

# Returned instead of an estimate when the skill-overlap feature cannot be computed - either the
# requisition named no skills, or there is no CV text to match them against. That feature carries
# the largest coefficient in the fitted model, so without it the remaining four produce a score
# squashed toward the bottom of the scale for every candidate alike, strong ones included. An
# estimate that is confidently low for a reason unrelated to the candidate is worse than none.
NO_SKILL_SIGNAL = (
    "No statistical estimate for this candidate: the requisition names no required skills to "
    "match against, so the model's strongest signal is missing and any number it produced would "
    "be misleading. Score this candidate from the CV alone."
)


def describe_estimate(score: float, features: dict, notes: list[str], candidate: dict,
                      requirements) -> str:
    """Render the estimate as a range with its drivers, deliberately not as a bare number.

    Leading with a bare "8.4" was measured to make gpt-4o-mini emit 8.4 as its own score and
    restate the tool output as its justification. Leading with a range, and listing what the
    model could not see, is what stops the estimate from reading as a verdict.
    """
    error = typical_error() or 1.5
    low = max(0.0, score - error)
    high = min(10.0, score + error)

    drivers = []
    overlap = features.get("skill_overlap")
    if overlap is not None:
        required = list(requirements.required_skills or [])
        haystack = ", ".join(candidate.get("skills") or []) + "\n" + str(candidate.get("resume_text") or "")
        matched = matched_required_skills(required, haystack)
        missing = [skill for skill in required if skill not in matched]
        line = f"matched {len(matched)} of {len(required)} required skills by keyword"
        # Name what is NOT matched rather than what is. A required skill echoed back here is the
        # requisition's wording, not the CV's; if the Evaluator copies it into matched_skills,
        # Node 5's grounding check will not find that exact phrasing in the CV and will withhold
        # the whole justification.
        if missing:
            line += f" (not found: {', '.join(missing)})"
        drivers.append(line)

    years = candidate.get("years_experience")
    minimum = getattr(requirements, "min_years_experience", None)
    if years is not None and minimum is not None:
        drivers.append(f"{years} years against a {minimum}-year minimum")

    title = features.get("title_overlap")
    if title is not None:
        drivers.append(f"job-title overlap {title:.2f} "
                       f"({requirements.title} vs {candidate.get('role_title', 'unknown')})")

    similarity = features.get("similarity")
    if similarity is not None:
        drivers.append(f"vector-search similarity {similarity:.2f}")

    lines = [
        f"statistical estimate: {low:.1f}-{high:.1f} out of 10 "
        f"(point estimate {score:.1f}, typical error +/- {error:.1f})",
        "drivers: " + "; ".join(drivers),
        "not considered: project depth, recency, career trajectory, seniority of the actual work, "
        "or anything else in the body of the CV",
    ]
    if notes:
        lines.append("degraded: " + "; ".join(notes))
    return "\n".join(lines)


def make_fit_score_tool(requirements, candidate: dict, sink: dict):
    """Build the tool for one candidate. `sink` collects point estimates for the report."""

    @tool
    def statistical_fit_score(candidate_name: str) -> str:
        """Statistical fit estimate for the candidate you are currently reviewing.

        Produced by a small ridge-regression model fitted on hand-labelled (requisition,
        candidate) pairs. It sees only four coarse signals: the fraction of required skills
        matched by keyword, years above or below the minimum, job-title overlap, and
        vector-search similarity.

        It has NOT read the CV. It cannot tell a CV that mentions a technology once in passing
        from one with three projects using it, and it cannot see recency, depth or trajectory.
        It is one piece of evidence to weigh against the CV, and it is never the answer.

        Args:
            candidate_name: the candidate you are reviewing, exactly as written in the CV header.
        """
        # A tool that raises propagates out of agent.invoke into score_one_candidate, where the
        # existing per-candidate handler catches it - and that candidate then vanishes from the
        # report entirely. A bug in feature extraction must cost the estimate, never the person.
        try:
            expected = candidate.get("candidate_name", "")
            if candidate_name and expected and candidate_name.strip().lower() != expected.lower():
                return (f"You asked about {candidate_name}, but the CV in front of you is "
                        f"{expected}. Review {expected}.")

            features, notes = extract_features(requirements, candidate)

            # Nothing is written to `sink` on this path, so the candidate simply has no estimate
            # to disagree with and Node 6 will not raise a "worth a second look" flag for them.
            if features.get("skill_overlap") is None:
                return NO_SKILL_SIGNAL

            result = predict(features)
            if result is None:
                return UNAVAILABLE

            score, model_notes = result
            sink[expected] = round(score, 2)
            return describe_estimate(score, features, notes + model_notes, candidate, requirements)

        except Exception as error:
            print(f"  [!] Statistical fit estimate failed for "
                  f"{candidate.get('candidate_name', '?')}: {error}")
            return UNAVAILABLE

    return statistical_fit_score

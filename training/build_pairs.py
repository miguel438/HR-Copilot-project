"""Build the (requisition, candidate) pair file that the fit model is trained on.

Runs inside the agent image, because it needs three things the host does not have: langchain,
a reachable Qdrant, and the OpenAI key. See training/README.md for the exact command.

Two decisions in here matter more than the code:

1.  Requisitions are written as free text and put through the REAL parse_requirements(), not
    hand-written as structured JobRequirements. Features are computed from
    `required_skills`, which at inference time is whatever the Planner LLM produced - if
    training used hand-written skill lists, the phrasing would differ from production and every
    skill_overlap value would be subtly wrong.

2.  `suggested_band` is prefilled from the current LLM Evaluator's score, NOT from a formula over
    the features. This is the single most important honesty property of the dataset. A formula
    prefill that gets rubber-stamped during review would mean the model simply relearns that
    formula, cross-validation would look excellent, and the number would mean nothing. The LLM is
    a genuinely different function from the five features, so where the reviewer overrides it,
    the label carries real information. train_fit_model.py reports the override rate for exactly
    this reason.
"""

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent

# Sibling src/ when run from a checkout; /app/src when this directory is copied into the agent
# image, where the project root is not this file's parent.
for _src in (TRAINING_DIR.parent / "src", Path("/app/src")):
    if (_src / "nodes.py").exists():
        sys.path.insert(0, str(_src))
        break
else:
    sys.exit("could not locate the project's src/ directory")

from nodes import parse_requirements, score_one_candidate  # noqa: E402
from retrieval import build_search_query, get_vector_store  # noqa: E402
from scoring_features import FEATURE_NAMES, extract_features, matched_required_skills  # noqa: E402

# Deliberately larger than RETRIEVAL_TOP_K and unfiltered: training needs every candidate paired
# with every requisition, including the ones production's experience filter would exclude. Those
# rows are the negatives, and without them the model only ever sees plausible candidates.
TRAINING_K = 40

# LLM score -> band, for the prefill only. 7.0 is SCORE_THRESHOLD; 3.5 splits "no fit" from
# "partial fit" at the boundary the Evaluator's own scoring guide uses (0-3 vs 4-6).
STRONG_BAND_MIN = 7.0
PARTIAL_BAND_MIN = 3.5


def band_from_score(score: float) -> int:
    if score >= STRONG_BAND_MIN:
        return 2
    if score >= PARTIAL_BAND_MIN:
        return 1
    return 0


def all_candidates_for(requirements) -> list[dict]:
    """Every candidate in the store, carrying this requisition's similarity score.

    Replicates search_candidates' dedup semantics - first hit wins, and hits arrive ordered by
    score, so each candidate keeps their best-matching chunk - but without the top-k cut and
    without the experience filter.
    """
    from retrieval import _to_candidate

    hits = get_vector_store().similarity_search_with_score(
        build_search_query(requirements), k=TRAINING_K, filter=None
    )
    candidates: dict[str, dict] = {}
    for document, score in hits:
        candidate = _to_candidate(document, score)
        candidates.setdefault(candidate["candidate_name"], candidate)
    return list(candidates.values())


def main() -> int:
    requisitions = json.loads((TRAINING_DIR / "requisitions.json").read_text(encoding="utf-8"))
    print(f"{len(requisitions)} requisitions")

    rows: list[dict] = []
    parsed_record: list[dict] = []

    for requisition in requisitions:
        requirements = parse_requirements(requisition["text"])
        parsed_record.append(
            {"id": requisition["id"], "parsed": requirements.model_dump(),
             "text": requisition["text"], "note": requisition["note"]}
        )
        candidates = all_candidates_for(requirements)
        print(f"  {requisition['id']:<28} {requirements.title:<28} {len(candidates)} candidates")

        # One LLM call per candidate, same as production's Node 3. score_one_candidate builds its
        # own per-candidate agent internally (the fit-score tool is a closure over that candidate's
        # data - see its docstring in nodes.py), so there is no shared agent to build up front.
        # model_scores is shared across the pool the same way evaluator() shares it in production:
        # each thread writes only its own candidate's key, which is safe under the GIL.
        model_scores: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            scores = list(
                pool.map(lambda c: score_one_candidate(requirements, c, model_scores), candidates)
            )

        for candidate, llm_score in zip(candidates, scores):
            features, notes = extract_features(requirements, candidate)
            matched = matched_required_skills(
                requirements.required_skills,
                ", ".join(candidate.get("skills") or []) + "\n" + candidate["resume_text"],
            )
            row = {
                "req_id": requisition["id"],
                "req_title": requirements.title,
                "min_years": requirements.min_years_experience,
                "candidate_name": candidate["candidate_name"],
                "candidate_role": candidate["role_title"],
                "candidate_years": candidate["years_experience"],
                "matched_skills": "; ".join(matched) or "-",
                "required_skills": "; ".join(requirements.required_skills) or "-",
                "suggested_band": band_from_score(llm_score.score),
                "band": band_from_score(llm_score.score),
                "llm_score": llm_score.score,
                "note": "; ".join(notes),
            }
            row.update({name: features[name] for name in FEATURE_NAMES})
            rows.append(row)

    # Grouped by requisition - a human judges one role at a time - and within a requisition the
    # strongest candidate first. That puts the two or three rows that need actual thought at the
    # top of each block, so the long tail of obvious no-fits can be skimmed.
    rows.sort(key=lambda r: (r["req_id"], -r["llm_score"], r["candidate_name"]))

    out_csv = TRAINING_DIR / "fit_labels.csv"
    fieldnames = [
        "req_id", "req_title", "min_years", "candidate_name", "candidate_role",
        "candidate_years", "matched_skills", "required_skills",
        "suggested_band", "band", "llm_score", *FEATURE_NAMES, "note",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (TRAINING_DIR / "parsed_requisitions.json").write_text(
        json.dumps(parsed_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Recorded so the fitted model can warn if it is later served against a differently-embedded
    # store: `similarity` comes straight from Qdrant, so re-ingesting under another embedding
    # model moves that feature's scale out from under its coefficient without any other symptom.
    from config import COLLECTION_NAME, EMBEDDING_MODEL

    (TRAINING_DIR / "pairs_meta.json").write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL, "collection_name": COLLECTION_NAME,
                    "training_k": TRAINING_K}, indent=2),
        encoding="utf-8",
    )

    print(f"\nwrote {out_csv} - {len(rows)} rows, "
          f"{len({r['candidate_name'] for r in rows})} candidates, "
          f"{len({r['req_id'] for r in rows})} requisitions")
    counts = {b: sum(1 for r in rows if r["suggested_band"] == b) for b in (0, 1, 2)}
    print(f"prefilled bands: no-fit {counts[0]}, partial {counts[1]}, strong {counts[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

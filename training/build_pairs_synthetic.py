"""Score the 11 synthetic candidates against the same 14 requisitions, for training-set
augmentation only.

Companion to build_pairs.py, not a replacement for it. The original 140 rows in fit_labels.csv
already went through real human review (see training/README.md and the override-rate discussion
in training/report/metrics.md); re-running build_pairs.py would call the LLM Evaluator again on
the original 10 candidates and silently overwrite those reviewed labels with a fresh, slightly
different set of LLM outputs. This script instead:

  - reuses the ALREADY-PARSED requirements from training/parsed_requisitions.json, rather than
    calling the Planner again, so the new rows' features are computed against the exact same
    required_skills the original 140 rows were - not a second, possibly-differently-worded parse
    of the same free text;
  - retrieves and scores ONLY the candidates named in NEW_CANDIDATES, from the separate
    hr_copilot_resumes_training collection (see ingest_synthetic.py) - the original 10 candidates
    and the production collection are never touched;
  - writes a "source" column of "synthetic" on every row it produces, so the honesty note in
    train_fit_model.py's report can say plainly how many of the training pairs are not from the
    real system's traffic.

suggested_band is prefilled from the real LLM Evaluator's score, exactly as build_pairs.py does,
and for the same reason: it is a genuinely different function from the five features, so a human
(or, here, a review pass against how each candidate was deliberately designed - see
make_synthetic_resumes.py) overriding it carries real information instead of teaching the model
its own formula back.

Run from the repo root with a Python that has the project's requirements.txt installed:
    python training/build_pairs_synthetic.py
"""

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

# Must run before importing anything that imports config: config.py reads QDRANT_URL and
# OPENAI_API_KEY from the environment at import time, so loading .env any later leaves them on
# the localhost/missing defaults. Inside the agent Docker image this is a no-op because Compose
# already injects the environment directly; running this script from a local venv (as intended -
# see the module docstring) it is not, which is why this line has to come first.
load_dotenv(PROJECT_ROOT / ".env")

for _src in (PROJECT_ROOT / "src", Path("/app/src")):
    if (_src / "nodes.py").exists():
        sys.path.insert(0, str(_src))
        break
else:
    sys.exit("could not locate the project's src/ directory")

from nodes import score_one_candidate  # noqa: E402
from scoring_features import FEATURE_NAMES, extract_features, matched_required_skills  # noqa: E402
from state import JobRequirements  # noqa: E402

TRAINING_COLLECTION = "hr_copilot_resumes_training"

NEW_CANDIDATES = {
    "Marcus Webb", "Priya Nair", "Chloe Bennett", "Tom Richardson", "Grace Kim",
    "Lucas Ferreira", "Isabella Rossi", "Ben Carter", "Hannah Osei", "Derek Holloway",
    "Rachel Adler",
}

# Deliberately larger than the number of new candidates, unfiltered - training needs every
# candidate paired with every requisition, including the ones production's experience filter
# would exclude. Mirrors TRAINING_K in build_pairs.py.
SEARCH_K = 40

STRONG_BAND_MIN = 7.0
PARTIAL_BAND_MIN = 3.5


def band_from_score(score: float) -> int:
    if score >= STRONG_BAND_MIN:
        return 2
    if score >= PARTIAL_BAND_MIN:
        return 1
    return 0


def new_candidates_for(requirements: JobRequirements) -> list[dict]:
    """The new synthetic candidates only, each carrying this requisition's similarity score.

    Queries the training-only collection directly rather than going through retrieval.py's
    search_candidates(), which targets config.COLLECTION_NAME (the production collection) and
    applies the top-k cut this training use case does not want.
    """
    from retrieval import _to_candidate, build_search_query, get_vector_store

    store = get_vector_store()
    assert store.collection_name == TRAINING_COLLECTION, (
        f"expected to search '{TRAINING_COLLECTION}', got '{store.collection_name}' - "
        "set QDRANT_COLLECTION before running this script"
    )
    hits = store.similarity_search_with_score(build_search_query(requirements), k=SEARCH_K, filter=None)
    candidates: dict[str, dict] = {}
    for document, score in hits:
        candidate = _to_candidate(document, score)
        if candidate["candidate_name"] in NEW_CANDIDATES:
            candidates.setdefault(candidate["candidate_name"], candidate)
    return list(candidates.values())


def main() -> int:
    if os.environ.get("QDRANT_COLLECTION") != TRAINING_COLLECTION:
        sys.exit(
            f"Refusing to run: QDRANT_COLLECTION must be set to '{TRAINING_COLLECTION}' so this "
            "never reads the production collection. Example:\n"
            f'  QDRANT_COLLECTION={TRAINING_COLLECTION} python training/build_pairs_synthetic.py'
        )

    parsed_record = json.loads((TRAINING_DIR / "parsed_requisitions.json").read_text(encoding="utf-8"))
    print(f"{len(parsed_record)} requisitions (reusing the cached Planner parse)")

    rows: list[dict] = []
    seen_candidates: set[str] = set()

    for entry in parsed_record:
        requirements = JobRequirements(**entry["parsed"])
        candidates = new_candidates_for(requirements)
        print(f"  {entry['id']:<28} {requirements.title:<28} {len(candidates)} new candidate(s)")

        model_scores: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            scores = list(
                pool.map(lambda c: score_one_candidate(requirements, c, model_scores), candidates)
            )

        for candidate, llm_score in zip(candidates, scores):
            seen_candidates.add(candidate["candidate_name"])
            features, notes = extract_features(requirements, candidate)
            matched = matched_required_skills(
                requirements.required_skills,
                ", ".join(candidate.get("skills") or []) + "\n" + candidate["resume_text"],
            )
            row = {
                "req_id": entry["id"],
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
                "source": "synthetic",
            }
            row.update({name: features[name] for name in FEATURE_NAMES})
            rows.append(row)

    missing = NEW_CANDIDATES - seen_candidates
    if missing:
        print(f"  [!] never retrieved for any requisition: {sorted(missing)} - check the ingest step")

    rows.sort(key=lambda r: (r["req_id"], -r["llm_score"], r["candidate_name"]))

    out_csv = TRAINING_DIR / "fit_labels_synthetic_new.csv"
    fieldnames = [
        "req_id", "req_title", "min_years", "candidate_name", "candidate_role",
        "candidate_years", "matched_skills", "required_skills",
        "suggested_band", "band", "llm_score", *FEATURE_NAMES, "note", "source",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out_csv} - {len(rows)} rows, {len(seen_candidates)} new candidates, "
          f"{len(parsed_record)} requisitions")
    print("Review the 'band' column against 'suggested_band' before merging into fit_labels.csv - "
          "same review step build_pairs.py's output always gets.")
    counts = {b: sum(1 for r in rows if r["suggested_band"] == b) for b in (0, 1, 2)}
    print(f"prefilled bands: no-fit {counts[0]}, partial {counts[1]}, strong {counts[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

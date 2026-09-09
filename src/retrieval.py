"""
Vector-store access for the CV Retrieval node (Node 2).

Retrieval is deliberately more than a plain similarity search: the semantic query is combined
with a Qdrant metadata filter on years of experience, so a requisition asking for 5+ years never
even considers a two-year candidate whose CV happens to use similar wording. Candidates are then
deduplicated by name, because a long CV can be split across several chunks and the Evaluator
must score a whole person, not a fragment.
"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
    RETRIEVAL_TOP_K,
)
from state import JobRequirements, RetrievedCandidate

# Qdrant stores LangChain document metadata nested under this payload key.
METADATA_PREFIX = "metadata"


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return the shared Qdrant vector store, built once per process."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def build_search_query(requirements: JobRequirements) -> str:
    """Turn the structured requirements back into the text we embed.

    Searching with the title plus the required skills matches how the CVs themselves are written,
    which retrieves better than embedding the raw requisition with all its boilerplate.
    """
    query = requirements.title
    if requirements.required_skills:
        query += ". Required skills: " + ", ".join(requirements.required_skills)
    return query


def build_experience_filter(min_years: int) -> models.Filter | None:
    """Filter out candidates below the required years of experience."""
    if min_years <= 0:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"{METADATA_PREFIX}.years_experience",
                range=models.Range(gte=min_years),
            )
        ]
    )


def _to_candidate(document, score: float) -> RetrievedCandidate:
    """Map a retrieved chunk onto the candidate shape the graph passes around."""
    metadata = document.metadata
    return {
        "candidate_name": metadata.get("candidate_name", "Unknown"),
        "role_title": metadata.get("role_title", "Unknown"),
        "years_experience": metadata.get("years_experience", 0),
        "skills": metadata.get("skills", []),
        "email": metadata.get("email", ""),
        # full_text is stored on every chunk so the Evaluator always sees the complete CV.
        "resume_text": metadata.get("full_text", document.page_content),
        "similarity": round(score, 4),
    }


def search_candidates(requirements: JobRequirements) -> list[RetrievedCandidate]:
    """Find the candidates most relevant to a requisition.

    Returns at most RETRIEVAL_TOP_K distinct candidates, best match first.
    """
    store = get_vector_store()
    query = build_search_query(requirements)
    experience_filter = build_experience_filter(requirements.min_years_experience)

    hits = store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K, filter=experience_filter)

    # If the experience bar excluded everyone, search again without it rather than handing the
    # recruiter an empty result. The Evaluator still judges the shortfall and can reject them.
    if not hits and experience_filter is not None:
        print("  [i] No candidate met the experience filter - retrying without it.")
        hits = store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K, filter=None)

    candidates: list[RetrievedCandidate] = []
    seen_names: set[str] = set()
    for document, score in hits:
        candidate = _to_candidate(document, score)
        if candidate["candidate_name"] in seen_names:
            continue
        seen_names.add(candidate["candidate_name"])
        candidates.append(candidate)

    return candidates


def find_experience_exclusions(
    requirements: JobRequirements, retrieved: list[RetrievedCandidate]
) -> list[dict]:
    """Relevant candidates the years-of-experience filter kept out of the results.

    The filter is a hard cut, so a role's single best match can be removed for being one year
    short while less relevant people who clear the bar take their place - and the recruiter sees
    a shortlist of poor fits with no indication that the bar, not the candidate pool, caused it.
    This re-runs the same search unfiltered purely to name who fell out, so Node 6 can say so.

    Only candidates who outranked *every* candidate that did make it through are reported. That
    keeps the note meaningful: with a small CV pool an unfiltered search sweeps in most of the
    store, so a looser rule would list a sales rep at rank 8 as "held back by the experience
    requirement" and imply they were worth considering. Outranking the whole evaluated set is
    the case where the filter, not the pool, decided what the recruiter saw.

    Ranking comes from the unfiltered search, so "relevant" here means the vector store ranked
    them above everyone evaluated - not that anyone judged them a fit. The Evaluator never sees
    these candidates and no score is produced for them.
    """
    minimum = requirements.min_years_experience
    if minimum <= 0:
        return []

    # A retrieved candidate below the bar means search_candidates already fell back to an
    # unfiltered search, so nothing was actually excluded and there is nothing to report.
    if any(c["years_experience"] < minimum for c in retrieved):
        return []

    retrieved_names = {c["candidate_name"] for c in retrieved}
    hits = get_vector_store().similarity_search_with_score(
        build_search_query(requirements), k=RETRIEVAL_TOP_K, filter=None
    )

    # Rank each distinct candidate in the unfiltered results, best match first.
    ranked: list[dict] = []
    seen: set[str] = set()
    for document, score in hits:
        candidate = _to_candidate(document, score)
        if candidate["candidate_name"] in seen:
            continue
        seen.add(candidate["candidate_name"])
        ranked.append(candidate)

    best_evaluated = next(
        (rank for rank, c in enumerate(ranked, start=1)
         if c["candidate_name"] in retrieved_names),
        None,
    )
    if best_evaluated is None:
        return []

    return [
        {
            "candidate_name": candidate["candidate_name"],
            "role_title": candidate["role_title"],
            "years_experience": candidate["years_experience"],
            "rank": rank,
        }
        for rank, candidate in enumerate(ranked, start=1)
        if rank < best_evaluated
        and candidate["candidate_name"] not in retrieved_names
        and candidate["years_experience"] < minimum
    ]

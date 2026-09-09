"""
Graph state and the data shapes that flow between the HR Copilot nodes.

The pydantic models here are not just documentation: JobRequirements and CandidateScore are
used as the `response_format` of structured-output agents in Nodes 1 and 3, so the LLM is
forced to return exactly these fields.
"""

from typing import TypedDict

from pydantic import BaseModel, Field


class JobRequirements(BaseModel):
    """The free-text requisition, parsed into structure by Node 1 (Planner)."""

    title: str = Field(description="The job title being hired for")
    required_skills: list[str] = Field(description="Skills the role explicitly requires")
    min_years_experience: int = Field(description="Minimum years of relevant experience")
    headcount: int = Field(default=1, description="How many people to hire for this role")


class RetrievedCandidate(TypedDict):
    """One candidate returned by Node 2 (CV Retrieval) from the vector store."""

    candidate_name: str
    role_title: str
    years_experience: int
    skills: list[str]
    email: str
    resume_text: str
    similarity: float


class CandidateScore(BaseModel):
    """Node 3 (Evaluator) output for a single candidate."""

    candidate_name: str = Field(description="Name of the candidate being scored")
    score: float = Field(description="Fit for the role, from 0 (no fit) to 10 (perfect fit)")
    justification: str = Field(description="Short explanation of the score, grounded in the CV")
    matched_skills: list[str] = Field(default_factory=list, description="Required skills the CV evidences")
    missing_skills: list[str] = Field(default_factory=list, description="Required skills absent from the CV")


class HRCopilotState(TypedDict, total=False):
    """State passed along the graph.

    `total=False` because each node contributes only its own slice of the state; LangGraph
    merges the partial dict a node returns into the running state.

    There is no key here for what the actions did, and that absence is intentional. n8n performs
    the calendar, mail and ATS work after this graph has already returned control, so no node in
    this file could ever observe the outcome of an action even if there were a field to put it
    in. That summary is assembled separately, inside n8n's own workflow.
    """

    # --- input ---
    requisition_text: str
    session_id: str

    # --- Node 1: Planner ---
    # Note this is `planned`, not `guardrail_passed`: the input guardrail runs in n8n before this
    # graph is called at all, so by here the text has already been accepted. False means the
    # requisition could not be turned into requirements, which is a parsing failure, not a block.
    planned: bool
    rejection_reason: str
    parsed_requirements: JobRequirements

    # --- Node 2: CV Retrieval (RAG) ---
    retrieved_candidates: list[RetrievedCandidate]
    # Set only when the vector-store search itself raised (bad filter, missing index, Qdrant
    # unreachable, ...). Kept separate from retrieved_candidates being empty so Node 6 can tell
    # "the search failed" apart from "the search ran and found nobody" - those look identical
    # downstream otherwise, and the difference is exactly what makes a search failure go unnoticed.
    retrieval_error: str
    # Candidates the years-of-experience filter removed even though the search ranked them
    # highly for this role. Reported to the recruiter by Node 6, never scored: the point is to
    # show when the experience bar, rather than the candidate pool, is what emptied a shortlist.
    experience_exclusions: list[dict]

    # --- Node 3: Evaluator (autonomous) ---
    scored_candidates: list[CandidateScore]
    passing_candidates: list[CandidateScore]
    # What the statistical model estimated for each candidate, keyed by name. Kept beside
    # CandidateScore rather than inside it because CandidateScore is the agent's response_format:
    # a field there would be filled in by the LLM, which would invent the number instead of
    # reporting the tool's. A name absent from this dict means the agent never called the tool.
    model_scores: dict[str, float]

    # --- Node 4: Grounding Guard ---
    # Nothing new is stored. A flagged candidate's justification is rewritten in place on the
    # CandidateScore objects in scored_candidates, so every later reader - Node 5's ATS notes
    # included - gets the corrected text and there is no path back to the original.

    # --- Node 5: Shortlist Prep ---
    # The two things n8n needs from this graph. `shortlist_text` is what its relevance guardrail
    # reads (role, required skills and job titles only - no CV bodies); `actions_request` is what
    # its calendar/mail/ATS chain acts on. Both absent when nobody cleared the threshold, which
    # is what the conditional edge before this node means.
    shortlist_text: str
    actions_request: dict

    # --- Node 6: Screening Report ---
    # Screening only. n8n appends its own "Actions taken" block to this text after it has
    # performed them - see the module docstring in nodes.py.
    report: str

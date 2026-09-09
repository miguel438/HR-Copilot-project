"""
Flask backend for the HR Copilot.

Exposes the LangGraph flow over HTTP as the screening step n8n calls:

    POST /screen   run a requisition through the graph and return the screening result
    GET  /health   liveness, plus optional dependency checks with ?deep=true

/screen is the only POST endpoint, and n8n calls it exactly once per run. There is no callback:
the graph starts and ends inside that one request. Everything with a side effect - the calendar,
the invitation email, the ATS row - happens in n8n after this returns, so this endpoint reads
Qdrant and spends LLM calls but writes nothing anywhere.

The compiled graph is built once at import time rather than per request: compiling it involves
wiring every node and edge, and none of that depends on the request.
"""

import uuid

from flask import Flask, jsonify, request
from langchain_text_splitters import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename

from config import (
    ACTION_MODE,
    COLLECTION_NAME,
    LLM_MODEL,
    N8N_SCREEN_URL,
    QDRANT_URL,
    RESUMES_DIR,
    SCORE_THRESHOLD,
)
from graph import build_graph
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, parse_resume
from nodes import build_actions_request, build_shortlist_text
from retrieval import get_vector_store
from state import JobRequirements

app = Flask(__name__)

# Built once at startup and reused by every request.
hr_copilot_graph = build_graph()

MAX_REQUISITION_CHARS = 5000

# Stands in for the parsed requisition when parsing failed, so the two fields n8n reads
# unconditionally are always present and well-formed. n8n's relevance check is told not to flag
# an empty shortlist, and its Has Passing? branch sees zero candidates and skips the actions.
UNPARSED_REQUIREMENTS = JobRequirements(
    title="(requisition not parsed)", required_skills=[], min_years_experience=0
)


def serialise_candidates(state: dict) -> list[dict]:
    """Expose the Evaluator's scores in a form a client can render.

    Note what is absent: `resume_text`. The CV bodies stay in this process. What leaves is the
    scores, the justifications and the skill lists - enough for n8n to act and for the UI to
    render, without shipping personal data through a second service for no reason.
    """
    passing_names = {c.candidate_name for c in state.get("passing_candidates", [])}
    return [
        {
            "candidate_name": candidate.candidate_name,
            "score": candidate.score,
            "justification": candidate.justification,
            "matched_skills": candidate.matched_skills,
            "missing_skills": candidate.missing_skills,
            "passed": candidate.candidate_name in passing_names,
        }
        for candidate in state.get("scored_candidates", [])
    ]


@app.post("/screen")
def screen():
    """Screen one job requisition and hand the result back to n8n.

    Body: {"requisition_text": str, "session_id": str (optional)}

    Malformed requests are rejected here, before the graph runs, so a client error never
    reaches the LLM nodes. Anything that is well-formed but unwanted - off-topic text, a
    prompt-injection attempt - was already stopped by the guardrail in n8n, which runs before
    this endpoint is called at all.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    # `requisition` is still accepted so an older client or a saved curl keeps working.
    requisition = payload.get("requisition_text", payload.get("requisition"))
    if not isinstance(requisition, str):
        return jsonify({"error": "Field 'requisition_text' is required and must be a string."}), 400
    if len(requisition) > MAX_REQUISITION_CHARS:
        return jsonify(
            {"error": f"Field 'requisition_text' exceeds {MAX_REQUISITION_CHARS} characters."}
        ), 400

    session_id = payload.get("session_id") or str(uuid.uuid4())

    try:
        state = hr_copilot_graph.invoke(
            {"requisition_text": requisition, "session_id": session_id}
        )
    except Exception as error:
        # The nodes handle their own expected failures; anything reaching here is unexpected,
        # so report it as a server error rather than pretending the run produced a result.
        app.logger.exception("Graph execution failed")
        return jsonify({"error": "The request could not be processed.", "detail": str(error)}), 500

    requirements = state.get("parsed_requirements")

    # Node 5 is skipped when nobody cleared the threshold, and the whole middle of the graph is
    # skipped when parsing failed - but n8n reads these two fields on every run, so they are
    # filled with their empty forms here rather than leaving n8n to cope with a missing key.
    # Same two functions, called with an empty shortlist.
    empty_for = requirements or UNPARSED_REQUIREMENTS

    return jsonify(
        {
            "session_id": session_id,
            "planned": state.get("planned", False),
            "rejection_reason": state.get("rejection_reason"),
            "requirements": requirements.model_dump() if requirements else None,
            "candidates": serialise_candidates(state),
            "model_scores": state.get("model_scores", {}),
            "shortlist_text": state.get("shortlist_text")
            or build_shortlist_text(empty_for, [], {}),
            "actions_request": state.get("actions_request")
            or build_actions_request(empty_for, [], {}),
            "report": state.get("report", ""),
        }
    )


@app.post("/candidates")
def upload_candidate():
    """Add one new CV to the candidate pool.

    Body: multipart/form-data with a single "file" field, a PDF resume in the same layout
    ingest.py expects (name / role / Skills: / Years of experience: / Email: lines).

    Embeds straight into the same cached vector store retrieval.py uses for search, so the new
    candidate is visible to the very next /screen call - no restart needed. Saves the file into
    RESUMES_DIR first so it also survives a future `ingest.py --reset`.
    """
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "No file provided under the 'file' field."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    filename = secure_filename(file.filename)
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RESUMES_DIR / filename
    if dest_path.exists():
        return jsonify(
            {"error": f"A CV named '{filename}' already exists. Rename the file or remove "
                      f"the existing one first."}
        ), 409

    file.save(dest_path)

    try:
        document = parse_resume(dest_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents([document])
        get_vector_store().add_documents(chunks)
    except Exception as error:
        # Don't leave an unembedded file behind - the next --reset would silently pick it up.
        dest_path.unlink(missing_ok=True)
        app.logger.exception("CV ingestion failed")
        return jsonify({"error": "The CV could not be processed.", "detail": str(error)}), 500

    return jsonify(
        {
            "candidate_name": document.metadata["candidate_name"],
            "role_title": document.metadata["role_title"],
            "years_experience": document.metadata["years_experience"],
            "skills": document.metadata["skills"],
            "chunks_added": len(chunks),
        }
    ), 201


@app.get("/health")
def health():
    """Liveness by default; add ?deep=true to also check the services the graph depends on."""
    body = {
        "status": "ok",
        "action_mode": ACTION_MODE,
        "llm_model": LLM_MODEL,
        "collection": COLLECTION_NAME,
        "score_threshold": SCORE_THRESHOLD,
    }

    if request.args.get("deep", "").lower() in {"1", "true", "yes"}:
        body["checks"] = run_dependency_checks()
        if any(check["status"] != "ok" for check in body["checks"].values()):
            body["status"] = "degraded"
            return jsonify(body), 503

    return jsonify(body)


def run_dependency_checks() -> dict:
    """Report on the external services the system needs, without failing the whole response.

    n8n is not a dependency of this process any more - it is the caller - so probing its webhook
    is strictly a convenience: it answers "is the orchestrator published and reachable?" from the
    one place a reviewer is already looking.
    """
    checks: dict[str, dict] = {}

    try:
        from retrieval import get_vector_store

        count = get_vector_store().client.count(collection_name=COLLECTION_NAME).count
        checks["qdrant"] = {"status": "ok", "url": QDRANT_URL, "points": count}
    except Exception as error:
        checks["qdrant"] = {"status": "error", "url": QDRANT_URL, "detail": str(error)}

    # The webhook is POST-only, so a GET 404s either way - but n8n's two 404s carry different
    # messages, and that difference is the whole check:
    #
    #   registered:    "This webhook is not registered for GET requests. Did you mean to make
    #                   a POST request?"
    #   not registered: "The requested webhook \"GET screen\" is not registered."
    #
    # Probing with GET keeps the check free of side effects. A POST would be unambiguous, but it
    # would also run a whole screening and book real slots, which is not something a health
    # endpoint may do.
    #
    # This check earns its place. n8n registers webhooks only when its own process starts, so a
    # published workflow can silently stop answering while every container still reports healthy
    # - see the warning above the n8n-import service in docker-compose.yml. That is exactly the
    # failure it caught during development, and it is invisible from every other signal.
    import requests

    try:
        response = requests.get(N8N_SCREEN_URL, timeout=5)
        message = response.text
        if response.ok or "not registered for GET requests" in message:
            checks["n8n_screen"] = {
                "status": "ok",
                "url": N8N_SCREEN_URL,
                "detail": "webhook is registered",
            }
        else:
            checks["n8n_screen"] = {
                "status": "error",
                "url": N8N_SCREEN_URL,
                "detail": "webhook is not registered - restart n8n "
                          "(docker compose down && docker compose up -d)",
            }
    except requests.RequestException as error:
        checks["n8n_screen"] = {"status": "error", "url": N8N_SCREEN_URL, "detail": str(error)}

    return checks


if __name__ == "__main__":
    # Development server only; the container runs this behind a production WSGI server.
    # threaded=True lets /health respond while a slow /screen request (multiple LLM calls) is
    # still in flight, instead of queuing behind it.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

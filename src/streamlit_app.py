"""
HR Copilot - Streamlit UI (bonus, §4.5).

A thin client, and it talks to two different places on purpose:

  * screening requests go to the **n8n webhook**, because n8n is what orchestrates a run - it
    runs the input guardrail, calls the agent, runs the shortlist guardrail, and performs the
    calendar and mail actions;
  * the health check goes to the **Flask API**, which is the thing whose dependencies are worth
    reporting on.

Neither is a direct call into the graph, so this file is free to run on a different machine from
the backend and can never drift from what the services actually return.
"""

import os
import uuid

import requests
import streamlit as st

N8N_SCREEN_URL = os.getenv("HR_COPILOT_N8N_URL", "http://localhost:5679/webhook/screen")
API_BASE_URL = os.getenv("HR_COPILOT_API_URL", "http://localhost:5000")
# A run is dominated by scoring - one LLM call per retrieved candidate, in parallel - not by the
# actions, which n8n batches into one pass. Matches the timeouts on n8n's HTTP Request node and
# on gunicorn, so all three ends of the call give up together.
REQUEST_TIMEOUT_SECONDS = 300

st.set_page_config(page_title="HR Copilot", page_icon="🧑‍💼", layout="wide")


def call_health(deep: bool) -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}/health", params={"deep": deep}, timeout=10)
        return response.json()
    except requests.RequestException as error:
        st.sidebar.error(f"Cannot reach the API at {API_BASE_URL}\n\n{error}")
        return None


def call_upload(uploaded_file) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE_URL}/candidates",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            timeout=60,
        )
    except requests.RequestException as error:
        st.sidebar.error(f"Cannot reach the API at {API_BASE_URL}\n\n{error}")
        return None

    if response.status_code != 201:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        st.sidebar.error(f"Upload failed (HTTP {response.status_code}): {detail}")
        return None

    return response.json()


def call_screen(requisition: str) -> dict | None:
    try:
        response = requests.post(
            N8N_SCREEN_URL,
            json={"requisition_text": requisition, "session_id": str(uuid.uuid4())},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        st.error(f"Could not reach the orchestrator at {N8N_SCREEN_URL}\n\n{error}")
        return None

    if response.status_code != 200:
        # A non-200 here means the workflow itself failed - most often the agent being
        # unreachable from n8n. Surface whatever the body says rather than a generic failure.
        try:
            detail = response.json().get("message", response.text)
        except ValueError:
            detail = response.text
        st.error(f"The run did not complete (HTTP {response.status_code}): {detail}")
        return None

    return response.json()


# --- Sidebar: connection status ---
with st.sidebar:
    st.header("System status")
    st.caption(f"API: `{API_BASE_URL}`")

    if st.button("Check health", use_container_width=True):
        health = call_health(deep=True)
        if health:
            status_icon = "🟢" if health["status"] == "ok" else "🟠"
            st.write(f"{status_icon} **{health['status']}** · mode: `{health['action_mode']}`")
            for name, check in health.get("checks", {}).items():
                icon = "✅" if check["status"] == "ok" else "❌"
                detail = check.get("detail") or f"{check.get('points', '?')} points"
                st.caption(f"{icon} {name}: {detail}")

    st.divider()
    st.caption(f"Orchestrator: `{N8N_SCREEN_URL}`")
    st.caption(
        "Screening runs go to n8n, which orchestrates them; only this health check talks to "
        "the Flask API. This page never calls the LangGraph flow or Qdrant directly."
    )

    st.divider()
    st.header("Add a candidate")
    uploaded_file = st.file_uploader("Upload a CV (PDF)", type=["pdf"])
    if st.button("Add to candidate pool", use_container_width=True, disabled=uploaded_file is None):
        with st.spinner("Embedding the CV..."):
            upload_result = call_upload(uploaded_file)
        if upload_result:
            st.success(
                f"Added {upload_result['candidate_name']} "
                f"({upload_result['role_title']}, {upload_result['years_experience']}y)"
            )

# --- Main: submit a requisition ---
st.title("HR Copilot")
st.caption("AI-based Personal Assistant for Recruiters")

requisition = st.text_area(
    "Job requisition",
    placeholder=(
        "e.g. \"We need a Senior Backend Engineer with at least 5 years of Python and "
        "AWS experience.\""
    ),
    height=100,
)

submitted = st.button("Screen candidates", type="primary", disabled=not requisition.strip())

if submitted:
    # One LLM call for the guardrail, one for the planner, one per retrieved candidate, one for
    # the shortlist check, then the bookings - so this routinely takes 20-40s. Expected, not a hang.
    with st.spinner("Running the requisition (guardrail, then scoring each candidate, then the "
                     "shortlist check, then booking - this can take up to a minute)..."):
        result = call_screen(requisition)

    if result is not None:
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")

if result:
    st.divider()

    if not result.get("guardrail_passed"):
        st.warning(f"**Request rejected by the input guardrail**\n\n{result['rejection_reason']}")
    elif not result.get("requirements"):
        # Guardrail passed but the graph produced no requirements - a parsing failure. The
        # report says why; there is nothing else to render.
        st.warning(result.get("reply", "The requisition could not be processed."))
    else:
        # Shown above the results, not inside the report expander below: the expander is collapsed
        # by default, and a guardrail warning nobody opens is not a guardrail. .get() because a
        # result cached in session state can predate a restart that renamed the field.
        if result.get("shortlist_warning"):
            st.error(f"**Shortlist relevance check blocked this run**\n\n"
                     f"{result['shortlist_warning']}")

        requirements = result["requirements"]
        cols = st.columns(4)
        cols[0].metric("Role", requirements["title"])
        cols[1].metric("Min. experience", f"{requirements['min_years_experience']}y")
        cols[2].metric("Headcount", requirements["headcount"])
        cols[3].metric("Candidates evaluated", len(result["candidates"]))

        if requirements["required_skills"]:
            st.caption("Required skills: " + ", ".join(requirements["required_skills"]))

        st.subheader("Candidate screening")
        candidates = sorted(result["candidates"], key=lambda c: c["score"], reverse=True)
        for candidate in candidates:
            badge = "✅ PASSED" if candidate["passed"] else "❌"
            with st.expander(f"{badge}  {candidate['candidate_name']} — {candidate['score']}/10"):
                st.write(candidate["justification"])
                skill_cols = st.columns(2)
                with skill_cols[0]:
                    st.caption("Matched skills")
                    st.write(", ".join(candidate["matched_skills"]) or "—")
                with skill_cols[1]:
                    st.caption("Missing skills")
                    st.write(", ".join(candidate["missing_skills"]) or "—")

        if result.get("actions"):
            st.subheader("Actions taken")
            for action in result["actions"]:
                icon = "✅" if action["status"] not in {"error", "unknown"} else "⚠️"
                st.write(f"{icon} **{action['candidate_name']}** — {action['action']}: {action['status']}")
        elif result.get("shortlist_warning"):
            # Candidates did pass, but the shortlist check stopped the run before anything was
            # booked. Saying "nobody cleared the threshold" here would be a plain lie.
            st.info("The run was stopped before any action was taken - see the warning above.")
        elif result["candidates"]:
            st.info("No candidate cleared the screening threshold, so no actions were taken.")

    with st.expander("Raw recruiter report"):
        st.text(result["reply"])

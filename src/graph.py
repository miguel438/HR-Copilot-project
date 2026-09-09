"""
LangGraph wiring for the HR Copilot.

    START
      -> Node 1  planner
           |-- conditional edge: could not parse? --> Node 6
           `-- parsed ---------------------------> Node 2
      -> Node 2  cv_retrieval
      -> Node 3  evaluator
      -> Node 4  grounding_guard
           |-- conditional edge: anyone above threshold? --> Node 5
           `-- nobody passed ---------------------------> Node 6
      -> Node 5  shortlist_prep
      -> Node 6  screening_report
    END

n8n calls this graph once, over HTTP, between its input guardrail and its shortlist guardrail.
Nothing in here reaches out to n8n, or anywhere else - see the module docstring in nodes.py.

Run this file directly to exercise the graph end to end:
    python src/graph.py
"""

from langgraph.graph import END, START, StateGraph

from nodes import (
    cv_retrieval,
    evaluator,
    grounding_guard,
    planner,
    route_after_grounding,
    route_after_planner,
    screening_report,
    shortlist_prep,
)
from state import HRCopilotState


def build_graph():
    """Build and compile the HR Copilot state graph."""
    builder = StateGraph(HRCopilotState)

    builder.add_node("planner", planner)
    builder.add_node("cv_retrieval", cv_retrieval)
    builder.add_node("evaluator", evaluator)
    builder.add_node("grounding_guard", grounding_guard)
    builder.add_node("shortlist_prep", shortlist_prep)
    builder.add_node("screening_report", screening_report)

    builder.add_edge(START, "planner")

    # Conditional edge 1: an unparseable requisition never reaches retrieval or scoring.
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"cv_retrieval": "cv_retrieval", "screening_report": "screening_report"},
    )

    builder.add_edge("cv_retrieval", "evaluator")
    builder.add_edge("evaluator", "grounding_guard")

    # Conditional edge 2: only assemble an action request when a candidate actually passed
    # screening. A run with nobody above the threshold produces no request at all, so there is
    # nothing for n8n to act on - the empty shortlist is the stop, not a flag to be checked.
    builder.add_conditional_edges(
        "grounding_guard",
        route_after_grounding,
        {"shortlist_prep": "shortlist_prep", "screening_report": "screening_report"},
    )

    builder.add_edge("shortlist_prep", "screening_report")
    builder.add_edge("screening_report", END)

    return builder.compile()


def save_visualization(output_path: str = "docs/langgraph_visualization.png") -> None:
    """Write the mandatory graph visualisation image to docs/."""
    graph = build_graph()
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as image_file:
        image_file.write(png_bytes)
    print(f"Saved graph visualisation to {output_path}")


if __name__ == "__main__":
    graph = build_graph()

    print(graph.get_graph().draw_ascii())

    test_requisitions = [
        ("strong match", "Senior Backend Engineer with strong Python and AWS experience, 5+ years."),
        ("empty input", "   "),
    ]

    for label, requisition in test_requisitions:
        print("\n" + "=" * 70)
        print(f"CASE: {label}")
        print("=" * 70)
        result = graph.invoke({"requisition_text": requisition, "session_id": "local-test"})
        print(result["report"])

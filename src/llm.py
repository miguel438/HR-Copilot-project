"""
Shared chat model for every node that needs one.

Built once and cached, so the Planner, Evaluator and Output Guardrail reuse a single client
instead of constructing a new one per node call. temperature=0 keeps screening decisions
repeatable - the same CV and the same requisition should not score differently between runs.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import LLM_MODEL

load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the shared chat model, raising a clear error if the API key is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the project's .env file, or pass it into the "
            "container with `-e OPENAI_API_KEY=...`."
        )
    return ChatOpenAI(model=LLM_MODEL, temperature=0)

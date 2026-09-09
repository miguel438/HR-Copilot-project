"""
Shared configuration for the HR Copilot system.

Every service address is read from an environment variable with a localhost default, so the
same code runs both during local development (Qdrant reachable on localhost) and inside
Docker, where the compose network resolves the services by name (http://qdrant:6333 etc.).
"""

import os
from pathlib import Path

# Project layout: src/config.py -> project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
REQUISITIONS_FILE = DATA_DIR / "job_requisitions.json"

# RAG / vector store, hosted on Qdrant Cloud, which requires QDRANT_API_KEY. qdrant_client also
# accepts None for that argument, so the same code works unchanged against a local, unauthenticated
# Qdrant instance if one is ever used instead.
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hr_copilot_resumes")
# OpenAI embeddings, so the deliverable only needs one model provider (no local Ollama
# install/pull, no second Docker service).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# LLM used by the agent nodes
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# The n8n webhook that drives the whole system. n8n calls this application, not the other way
# around, so this address is used for one thing only: the deep health check, which probes the
# webhook to confirm the workflow is published and reachable. See run_dependency_checks in app.py.
#
# Default to 127.0.0.1, not localhost: on this Windows/Docker Desktop setup, Python's `requests`
# resolves "localhost" to the IPv6 loopback (::1) first, which Docker's port forwarding does not
# answer on - each call then stalls for ~21s before falling back to IPv4. Measured directly:
# localhost took 21.1s, 127.0.0.1 took 0.03s, same request. Inside Docker Compose this is
# overridden to the service name (http://n8n:5678/...). Port 5679, not 5678: v2 runs on offset
# ports so it can be brought up alongside v1 without either stack stealing the other's webhooks.
N8N_SCREEN_URL = os.getenv("N8N_SCREEN_URL", "http://127.0.0.1:5679/webhook/screen")

# How the n8n workflow carries out the actions it is asked to perform:
#   "log"  - it only writes calendar.json / sent_emails.json / ats_log.json. Needs no credentials,
#            so the project runs straight from the submitted ZIP.
#   "live" - it additionally sends real Gmail and creates real Google Calendar events, which
#            requires Google OAuth credentials configured inside n8n.
# The files are written in both modes, so the record of what happened is never mode-dependent.
# Defaults to "log" so this file, .env.example and docker-compose.yml all agree: running
# `python src/app.py` locally must not quietly start sending real mail.
ACTION_MODE = os.getenv("ACTION_MODE", "log")

# Where live invitations are actually delivered. The synthetic CVs carry example.com addresses
# that nobody owns, so mail is redirected to one real inbox using plus-addressing.
DEMO_INBOX = os.getenv("DEMO_INBOX", "project.03.07.26@gmail.com")

# Number of candidate chunks pulled from the vector store per requisition.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))

# A candidate must score at least this (out of 10) to be forwarded to the Action Tool node.
# This threshold is what the graph's conditional edge branches on.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "7"))

# The fitted statistical fit model the Evaluator consults as a tool. Lives under data/ because
# the Dockerfile only copies src/ and data/, and it is a text file of coefficients rather than a
# pickle - see src/scoring_model.py. Absent is a supported state: the tool degrades and says so.
FIT_MODEL_FILE = DATA_DIR / "fit_model.json"

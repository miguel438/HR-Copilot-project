# HR Copilot

An AI recruiting assistant that takes a free-text job requisition, retrieves matching candidate
CVs from a vector store, autonomously scores them against the role, and books interviews, sends
invitations, and logs outcomes to an ATS via n8n.

Built for the AI Developers Course final project (scenario #3, "AI-based Personal Assistant
for Recruiters").


<img width="1728" height="1022" alt="image" src="https://github.com/user-attachments/assets/242a7e74-f68f-4500-bfdd-2ff0d117843a" />


## Requirements

- Docker Desktop.
- An OpenAI API key (used for both the LLM and the embeddings - no other model provider needed).

## Setup

```bash
cp .env.example .env
# edit .env and paste in your OpenAI key

docker compose up --build -d
```

That one command does everything: chunks and embeds the 10 synthetic CVs in `data/resumes/` into
Qdrant (OpenAI embeddings), and imports and publishes the n8n workflow before n8n's own process
ever starts, so its webhooks are registered from the very first boot - no manual activation, no
UI login, no credentials to create by hand. It typically takes 1-2 minutes on a first run (mostly
the Docker image build); subsequent runs are fast since everything is cached in named volumes.

Once both commands have finished:

| Service | URL | What it's for |
|---|---|---|
| Streamlit UI | http://localhost:8502 | Paste a requisition, see scored candidates and actions taken |
| n8n | http://localhost:5679 | The orchestrator: `POST /webhook/screen`, plus the workflow and execution traces |
| Flask API | http://localhost:5001 | The agent: `POST /screen` (no side effects), `GET /health` (add `?deep=true`) |

### First time opening the n8n UI

n8n asks you to "set up an owner account" the first time you open http://localhost:5679 in a
browser - that's not a login for some existing credentials, it's a fresh local instance with no
owner yet, so it's asking you to create one on the spot. Any email and any password (8+
characters) works; it's purely local to that n8n instance and has no effect on the workflow,
since the webhooks and credentials were already provisioned via CLI before n8n even started.
Note that verifying the system works does **not** require logging into n8n at all - that's what
the Streamlit UI and `GET /health?deep=true` are for. n8n's UI is only for inspecting the
workflow itself, if you're curious.

### A note on how the n8n workflow gets activated

n8n only registers a workflow's webhook when its own process *starts* - not when a workflow is
activated or published while it's already running. The `n8n-import` service in
`docker-compose.yml` sidesteps this by running `n8n import:workflow` and `n8n publish:workflow`
- both plain CLI commands that work directly against the database file, no running server
needed - *before* the main `n8n` service ever starts. By the time n8n boots for the first time,
the workflow is already published, so it's registered immediately.

Also worth knowing if you inspect the workflow in n8n's UI: its `active` toggle and n8n's
separate "published" state are two different things in this n8n version - only the latter
determines whether a webhook is actually registered at boot. A workflow can show as active and
still never receive a request.

## How the system is orchestrated

There is **one** webhook. It runs an input guardrail, calls the LangGraph agent once over HTTP,
runs a second guardrail on what comes back, and only then books anything.

```
  n8n (orchestrator)                          Python (LangGraph)
  ------------------                          ------------------
  [Screen Webhook]  POST /webhook/screen
        |
  [Guardrails]  keywords / jailbreak / topicalAlignment
        |--- fail --> [Explain Block] -> [Respond (Blocked)]        <-- ends here
        v
  [Screen Agent]  POST http://agent:5000/screen ---------------->  1 planner
        |                                                          2 cv_retrieval (Qdrant)
        |                                                          3 evaluator (LLM + fit tool)
        |                                                          4 grounding_guard
        |   <---- {report, shortlist_text, actions_request} ------  5 shortlist_prep
        |                                                          6 screening_report
        v
  [Shortlist Check]  does this shortlist belong to this role?
        |--- fail --> [Explain Shortlist Block] -> [Respond]        <-- ends here, NOTHING BOOKED
        v
  [Has Passing?] --- no --> [Compose Reply] -> [Respond]            <-- ends here
        | yes
        v
  Read Calendar -> Assign Slots -> Write Calendar -> Read Outbox -> Compose Invitations
    -> Write Outbox -> Read ATS Log -> Append ATS Entries -> Write ATS Log
    -> [Build Response] -> [Compose Reply] -> [Respond to Webhook]
    -> [If Live?] -> [Fan Out Live] -> Send Gmail / Create an event
```

Two things about this ordering are worth understanding before reading the code:

**The grounding guard runs before anything leaves the system.** It checks that every skill the
Evaluator claims a candidate has actually appears in their CV, and it runs *ahead of* the node
that builds the ATS row and the invitation email - so an unverified claim never reaches either.
The report the recruiter reads is already corrected by the time they see it.

**The shortlist relevance check can actually prevent a bad batch of actions, not just warn about
it.** It sits between the agent and the action chain, so a shortlist that plausibly belongs to
the wrong profession (say, software engineers put forward for a ship captain role) stops the run
before any interview is booked or any email goes out. Nothing in `calendar.json`,
`sent_emails.json` or `ats_log.json` changes when this check fires.

**There is no way to reach Gmail or Google Calendar except through the guardrail.** The action
chain is reachable only from `Has Passing?`, which is reachable only from `Shortlist Check`,
which is reachable only from `Guardrails`. That is a property of the graph itself, not a rule
anyone has to remember to enforce.

The report is put together in two places, and that split is deliberate: the screening half -
who was evaluated, what they scored, who passed - is written by the Python agent, at the point
where the scores are actually decided. The `Actions taken:` block is appended afterwards by n8n,
because n8n is what performs the bookings and is the only thing that knows whether a slot was
actually free or a message actually went out. A report that announced an interview before
anything had booked one would be exactly the kind of confident-and-wrong output the rest of this
system is designed to avoid.

Two implementation details worth knowing if you're reading the n8n workflow itself:

- **The Guardrails node replaces the item's `json`** with its own verdict (`{guardrailsInput,
  checks}`) rather than passing the input through. So every node downstream reads the agent's
  answer by name, `$('Screen Agent').first().json`, never `$json`.
- **`Compose Reply` is reached from both branches of `Has Passing?`.** On the no-shortlist branch
  `Build Response` never executed, and referencing an unexecuted node throws - that throw is
  caught and treated as "there was nothing to book," which is simpler than keeping two copies of
  the response envelope in sync.

## The statistical fit model (Node 3's tool)

The Evaluator (Node 3) is the one node in the graph that makes a genuine judgement call - every
other node's job is mechanical (parsing text, running a search, assembling a request). Node 3
has one tool available to it, `statistical_fit_score`, which gives it a second, independently
computed opinion to weigh against its own reading of the CV.

`statistical_fit_score` returns a ridge-regression estimate fitted on 294 hand-labelled
(requisition, candidate) pairs across 21 candidates, over five features: keyword skill overlap,
years above and below the minimum, job-title overlap, and the Qdrant similarity score from
retrieval. 140 of those pairs come from the 10 real CVs the app searches; the other 154 come from
11 synthetic candidates written purely to grow and rebalance the training set, which never reach
the production collection. Training lives in `training/`; see that directory's README for the
pipeline.

**Trained offline, served as arithmetic.** `data/fit_model.json` holds the coefficients and the
runtime evaluates them with a dot product in `src/scoring_model.py`. scikit-learn is never
installed in the image that serves the API, the UI and ingest - `pip show scikit-learn` inside the
agent fails, by design. That keeps ~120MB out of the image, makes the model a diffable text file
rather than a pickle, and removes "sklearn failed to import" as a runtime failure mode instead of
handling it.

**The tool is a per-candidate closure, not a function taking features as arguments.** Tool
arguments are filled in by the model, and an LLM cannot compute a cosine similarity - it would
invent one. More importantly, the moment the inputs are LLM-generated the estimate stops being an
independent opinion, and disagreement between the two stops meaning anything.

**The tool leads with a range, not a bare number, and lists what it could not see.** A bare point
estimate reads to the LLM as a verdict rather than a second opinion, so the tool is written to
resist that: it reports a range around its estimate, names the drivers behind it, and states
plainly what it cannot judge (project depth, recency, career trajectory). The system prompt tells
the Evaluator that agreeing with the tool by default is a failure, not a success.

### What it is honestly worth

| | leave-one-candidate-out | leave-one-requisition-out |
|---|---|---|
| MAE | 0.77 pts | 0.77 pts |
| band accuracy | 89% | 90% |

Both splits are grouped, never random - the same CV appears in 14 rows, so a random split would
leak it across folds. Three things worth saying plainly:

- **It beats every baseline, but barely.** `skill_overlap` alone scores 0.84 MAE against the
  model's 0.77 - a margin of 0.07 points, well inside the fold-to-fold spread. The honest claim is
  that five features are *no worse* than the single best one, and what they buy is a coefficient
  table that explains a score - not accuracy.
- **It recovers 50% of strong candidates** (misses 13 of 26). That is the number that matters for a
  screening aid, and overall accuracy hides it because 83% of pairs are obvious no-fits.
- **The labels were reviewed with steering.** The reviewer was pointed at the ~29 ambiguous rows,
  several flagged because the Evaluator had contradicted its own job-family rule. Overrides went
  that way, and `title_overlap` is the coefficient that moved most. No model prediction was shown
  during labelling, so this is not leakage - but it is not a blind sample either.

The estimate never writes `score` and never touches `SCORE_THRESHOLD`. It informs an LLM that
decides, and a gap of 2.5 points or more between the two is reported to the recruiter as advisory
text. In practice the two usually agree, so the flag is a rare-event net rather than a routine
signal.

**A limitation both halves share.** Requirements phrased in business language rather than
technology names - "full-cycle pipeline management", "quota attainment" - appear verbatim in no CV,
so `skill_overlap` collapses. The Evaluator degrades the same way for the same reason: on the sales
requisition it scored a genuine Account Executive 5.0, explicitly because her CV "does not
explicitly mention" those phrases. Because both fail in the same direction, the disagreement flag
cannot catch it. This is a property of the whole design, not just the model.

One caveat worth stating: the concurrency window is narrow but not closed. Two simultaneous
requests still race on the three JSON files, because n8n takes no lock on them.

## Verifying it works

```bash
curl "http://localhost:5001/health?deep=true"
```
should report `qdrant` with 10 points and the `n8n_screen` webhook registered.

**Run this before demoing anything.** It is the only signal for a failure mode this stack really
has: n8n registers webhooks only when its own process starts, so the workflow can stop answering
while `docker compose ps` still shows every container up and healthy. The check catches it by
reading n8n's 404 body - a registered POST-only webhook answers a GET with *"This webhook is not
registered for GET requests. Did you mean to make a POST request?"*, an unregistered path with
*"The requested webhook … is not registered."*

The way to trigger that failure, and therefore the thing not to do:

```bash
docker compose up -d agent              # DON'T - re-runs the n8n import against a live n8n
docker compose up -d --no-deps agent    # do this instead
```

`agent` depends on `n8n`, which depends on the one-shot `n8n-import`, and compose re-runs a
completed dependency. The import then deactivates and re-publishes the workflow underneath the
running instance - it even says so: *"Changes will not take effect if n8n is running."* The
webhook is deregistered, nothing looks wrong, and every request 404s until a full restart:

```bash
docker compose down && docker compose up -d
```

A full run goes to n8n, because n8n is what runs it:

```bash
curl -X POST http://localhost:5679/webhook/screen -H "Content-Type: application/json" \
  -d '{"requisition_text": "We need a Senior Backend Engineer with at least 5 years of Python and AWS experience."}'
```
should return a scored candidate list, with Alice Chen passing and the rest correctly rejected on
skill or job-family grounds, and a `reply` ending in an `Actions taken:` block with
`schedule_interview` / `send_invitation_email` / `log_to_ats` for the one who passed.

The agent's own endpoint is worth calling once too, to see the split:

```bash
curl -X POST http://localhost:5001/screen -H "Content-Type: application/json" \
  -d '{"requisition_text": "We need a Senior Backend Engineer with at least 5 years of Python and AWS experience."}'
```
Same scores, same shortlist - and `n8n_data/calendar.json` is unchanged afterwards. `/screen`
reads Qdrant and spends LLM calls; it books, sends and writes nothing. That is what makes
"every side effect is behind the guardrail" a structural claim rather than a convention.

Or just open http://localhost:8502 and use the UI.

## ACTION_MODE: log vs. live

`ACTION_MODE=log` (the default in `.env.example`) is what a reviewer gets with zero *Google*
setup: the candidate-action workflow only writes to `n8n_data/calendar.json`, `sent_emails.json`
and `ats_log.json` - a real, persisted external state change, satisfying the brief's requirement
without needing any Gmail/Calendar OAuth credentials.

It still needs an `OPENAI_API_KEY` and a Qdrant Cloud cluster (`QDRANT_URL` + `QDRANT_API_KEY` -
see `.env.example`), since the vector store is hosted, not a bundled container. A working `.env`
covering all three - plus, for live mode, the Gmail/Calendar credentials below - is delivered
separately from the project ZIP as `HRCopilot_LIVE_CREDENTIALS.env`; place it as `.env` in this
folder to run with zero setup at all. Without that file, bring your own OpenAI key and Qdrant
Cloud cluster (free tier is enough - see qdrant.io).

`ACTION_MODE=live` additionally sends a real Gmail invitation and creates a real Google Calendar
event (with the candidate added as an attendee, so Google emails them a proper interview
invitation). This needs Gmail + Google Calendar OAuth credentials configured inside n8n first:

1. In Google Cloud Console: create a project, enable the Gmail API and Calendar API, create an
   OAuth client, and set its redirect URI to `http://localhost:5679/rest/oauth2-credential/callback`.
2. In n8n (http://localhost:5679) → Credentials: create a **Gmail OAuth2** credential and a
   **Google Calendar OAuth2** credential, completing the consent flow for each.
3. Open the imported workflow, and on the **Send Gmail** and **Create an event** nodes, select
   the credentials you just created.
4. Set `ACTION_MODE=live` in `.env` and run `docker compose up -d agent` to pick it up.

Candidate CVs carry `@example.com` addresses that nobody owns, so live mode redirects delivery
to one real inbox via plus-addressing (`you+alice.chen@gmail.com` still lands in `you@gmail.com`).
Set `DEMO_INBOX` in `.env` to your own address to use this.

Google's OAuth apps in "Testing" publishing status issue refresh tokens that expire after 7
days - re-authorize shortly before a live demo, not weeks ahead.

Live mode fires **after** the webhook has already responded, so the agent never waits on Google
and a run takes the same time from Python's point of view either way. And because the whole
shortlist is handled in one execution, a run sends N emails and creates N calendar events at once
rather than one at a time - worth remembering before the first live run.

## Project structure

```
HRCopilot_Project_v2/
├── src/                    # Flask API, LangGraph nodes/state/graph, RAG ingestion, Streamlit UI
│   ├── scoring_features.py # the 5 model features; imported by both runtime and trainer
│   ├── scoring_model.py    # loads fit_model.json, evaluates it in pure Python
│   └── scoring_tool.py     # the LangGraph tool Node 3's Evaluator calls
├── data/resumes/           # 10 synthetic candidate CVs (PDF)
├── data/fit_model.json     # fitted coefficients - the whole served model, ~2KB of text
├── data/job_requisitions.json  # sample requisitions used during development
├── docs/                   # architecture diagram, graph visualization, spec document
├── n8n/
│   └── hr-copilot-v2-workflow.json  # the orchestration: guardrail -> agent -> guardrail ->
│                                    # actions. Imported and published automatically at startup
│                                    # by the n8n-import service in docker-compose.yml
├── n8n_data/                # calendar.json / sent_emails.json / ats_log.json (log-mode state)
├── training/                # OFFLINE only - not copied into the image. See training/README.md
│   ├── requisitions.json    # 14 free-text requisitions
│   ├── fit_labels.csv       # 294 hand-labelled pairs
│   └── report/metrics.md    # cross-validation, baselines, coefficients, residuals
├── Dockerfile                # one image, reused by the agent/ui/ingest services
├── docker-compose.yml
└── requirements.txt          # note: no scikit-learn - training is offline, see above
```

## Local development (without Docker)

Everything also runs directly against a local Python venv, which is how it was built and tested.
Point `N8N_SCREEN_URL` at `127.0.0.1` rather than `localhost` - on Windows + Docker Desktop,
Python's `requests` resolves `localhost` to the IPv6 loopback first, which Docker's port
forwarding doesn't answer, adding a ~21 second stall to every call before it falls back to IPv4.
`config.py`'s default already does this. Qdrant is unaffected - it's a Qdrant Cloud HTTPS
endpoint (`QDRANT_URL` + `QDRANT_API_KEY` in `.env`), not a local container.

```bash
pip install -r requirements.txt
python src/ingest.py --reset      # once, to populate the Qdrant Cloud collection
python src/app.py                 # Flask on :5000 (container port; published as :5001)
streamlit run src/streamlit_app.py  # UI on :8501, in a second terminal (published as :8502)
```

**One thing needs editing for this to work end to end.** n8n calls the agent at
`http://agent:5000/screen`, a Docker Compose service name that only resolves inside the compose
network. Running Flask on the host instead means pointing the workflow's **Screen Agent** node at
`http://host.docker.internal:5000/screen` - either directly in n8n's UI, or by editing that node's
`url` in `n8n/hr-copilot-v2-workflow.json` and then running `docker compose down && docker compose
up -d` so the workflow is re-imported. Without that, the UI reaches n8n and n8n cannot reach the
agent.

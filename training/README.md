# Training the statistical fit model

Everything in this directory is **offline**. The `Dockerfile` copies only `src/` and `data/`, so
none of it ships in the agent image — the runtime loads `data/fit_model.json` and does a dot
product in pure Python. scikit-learn is never installed in the image that serves the app.

## The pipeline

```
requisitions.json          14 free-text requisitions, hand-written
        |
        |  build_pairs.py     (needs Qdrant + OpenAI, so runs in the agent container)
        v
fit_labels.csv             140 (requisition, candidate) rows, features computed,
        |                  `band` prefilled from the LLM Evaluator
        |  <- YOU EDIT the `band` column here
        v
        |  train_fit_model.py  (needs scikit-learn, so runs in a throwaway container)
        v
data/fit_model.json        coefficients + training metrics, committed to git
```

## 1. Build the pairs

Needs Qdrant, the OpenAI key and langchain, all of which the running agent already has. It makes
14 Planner calls and 140 Evaluator calls (roughly 3 minutes, a few cents).

```bash
docker cp training hrcopilot-v2-agent-1:/training
docker exec hrcopilot-v2-agent-1 python /training/build_pairs.py
docker cp hrcopilot-v2-agent-1:/training/fit_labels.csv training/fit_labels.csv
docker cp hrcopilot-v2-agent-1:/training/parsed_requisitions.json training/parsed_requisitions.json
```

On Git Bash, prefix the `docker cp`/`docker exec` calls with `MSYS_NO_PATHCONV=1` or it rewrites
the container paths into Windows paths.

## 2. Label

Edit the `band` column of `training/fit_labels.csv`, and nothing else:

| band | meaning |
|---|---|
| `2` | strong fit — you would interview this person for this role |
| `1` | partial fit — right neighbourhood, real reservations |
| `0` | no fit |

Rows are grouped by requisition, strongest candidate first, so each block is two or three rows
that need thought followed by obvious no-fits.

`suggested_band` is the LLM Evaluator's own answer, deliberately **not** a formula over the
features. If the prefill were computed from the same numbers the model trains on, agreeing with
it would teach the model nothing but that formula, and cross-validation would look great while
meaning nothing. Because the prefill comes from a genuinely different function, every row you
change carries real information — which is why `train_fit_model.py` reports the override rate.

Leave `suggested_band` alone; the override rate is computed by comparing it against `band`.

## 3. Train

Runs in a throwaway container, so scikit-learn is never installed on the host or in the app
image. Needs only `fit_labels.csv` — no Qdrant, no API key.

```bash
docker run --rm -v "$PWD":/work -w /work python:3.12-slim \
  sh -c "pip install -q -r training/requirements-train.txt && python training/train_fit_model.py"
```

Writes `data/fit_model.json` and `training/report/metrics.md`, then:

```bash
docker compose up --build -d
```

## 4. (Optional) Add synthetic candidates for training-set augmentation

The steps above always regenerate build_pairs.py's output from scratch, which would call the LLM
Evaluator on the original candidates again and overwrite their already-reviewed `band` values.
This separate path adds *new* candidates without touching a single reviewed label:

```
make_synthetic_resumes.py       writes N synthetic CV PDFs to training/synthetic_resumes/
        |                        (never to data/resumes/ - that stays production-only)
        v
ingest_synthetic.py --reset     embeds them into their own Qdrant collection,
        |                        hr_copilot_resumes_training - never the production one
        v
build_pairs_synthetic.py        scores ONLY the new candidates against the 14 requisitions,
        |                        reusing the cached Planner parse in parsed_requisitions.json,
        v
fit_labels_synthetic_new.csv    -> reviewed, then appended to fit_labels.csv with source=synthetic
```

None of these three scripts need Docker or the agent image - they import `src/` directly, so run
them with a Python that has `requirements.txt` installed (a throwaway venv is enough: `python -m
venv .venv-training && .venv-training/Scripts/pip install -r requirements.txt`). `ingest_synthetic.py`
loads `.env` itself; `build_pairs_synthetic.py` refuses to run unless `QDRANT_COLLECTION` is
explicitly set to the training collection, specifically so a copy-pasted command can never score
against - or worse, this script doesn't write to - the production collection:

```bash
python training/make_synthetic_resumes.py
python training/ingest_synthetic.py --reset
QDRANT_COLLECTION=hr_copilot_resumes_training python training/build_pairs_synthetic.py
```

Then review `fit_labels_synthetic_new.csv`'s `band` column against `suggested_band` exactly as in
step 2 - these candidates were designed to be unambiguous, so expect few or no overrides, but
review it anyway rather than assuming the design worked. Merge the reviewed rows into
`fit_labels.csv` (matching its column order, with `source` set to `synthetic` on the new rows and
`original` on the existing ones), delete the intermediate CSV, and re-run step 3.

A `source` column on `fit_labels.csv` is optional for `train_fit_model.py` - rows without it are
treated as `original` - but once present it drives an explicit disclosure in the generated report
about how many pairs are synthetic and why that batch's override rate cannot be compared to the
original one at face value.

## Honest limits, stated up front

- **21 CVs (10 real + 11 synthetic), 14 requisitions, 294 pairs, one annotator.** There is no
  inter-annotator agreement because there is only one annotator. The synthetic candidates were
  added purely to grow and rebalance the training set (see step 4) and were each written to be an
  unambiguous fit for one job family - they make the dataset bigger and better-balanced across
  bands, not harder, so a metric computed only on them is not comparable to one computed on the
  original 140-row corpus.
- **Independent of the *scoring* LLM, not of the *parsing* LLM.** Features are computed from
  `required_skills`, which the Planner produced. That is why the training requisitions are free
  text put through the real `parse_requirements()` rather than hand-written skill lists.
- **`skill_overlap` needs concrete technologies.** A requisition asking for "software
  development, code review and testing" parses into skills that appear verbatim in no CV, and the
  feature degrades to noise. This is real at inference time too: a vague requisition produces a
  weak estimate. It is why `req-t13` names Python and Docker rather than abstractions.
- **`years_shortfall` is nearly dead in production.** Retrieval filters out candidates below the
  minimum, so the feature is almost always 0 at inference, reachable only via the no-hits
  fallback in `retrieval.py`. It is trained on all 10 candidates regardless, so its coefficient
  is fitted on data production rarely sees.
- **The class balance is skewed** — roughly 15 strong, 10 partial, 115 no-fit. Ten CVs spread
  across eight professions means any real requisition has one or two fits. `train_fit_model.py`
  therefore reports per-band error and a predict-the-majority baseline; overall accuracy alone
  would be flattering and meaningless.

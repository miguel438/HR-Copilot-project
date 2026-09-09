"""Load the fitted fit-model and evaluate it, in pure Python.

The model is trained with scikit-learn offline (training/train_fit_model.py) but is *served* as a
few dozen floats in data/fit_model.json, evaluated here with a dot product. That is a deliberate
trade, and it buys three things:

  - scikit-learn and scipy (~120MB) stay out of an image that serves the API, the UI and the
    ingest job, none of which does any training;
  - "sklearn failed to import" stops being a runtime failure mode that needs handling, because
    there is nothing to import;
  - the model becomes a readable, diffable text file rather than a pickle with a version
    dependency on the library that wrote it.

The cost is being restricted to linear models. At 140 rows drawn from 10 people that is the right
restriction anyway - see training/README.md.
"""

import json
import threading

from config import EMBEDDING_MODEL, FIT_MODEL_FILE
from scoring_features import FEATURE_NAMES

SCHEMA_VERSION = 1

# Loaded once, on first use, behind a lock: the Evaluator scores candidates on a thread pool, so
# without this eight threads race to read the same file and print eight identical warnings.
_LOCK = threading.Lock()
_MODEL: dict | None = None
_LOADED = False


def _load() -> dict | None:
    """Read and validate the artifact. Returns None - never raises - if it is unusable."""
    try:
        with open(FIT_MODEL_FILE, encoding="utf-8") as handle:
            model = json.load(handle)

        version = model["schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(f"artifact is schema v{version}, this code expects v{SCHEMA_VERSION}")
        if model["feature_names"] != FEATURE_NAMES:
            raise ValueError(
                f"feature mismatch: artifact has {model['feature_names']}, "
                f"scoring_features defines {FEATURE_NAMES}"
            )
        for key in ("means", "stds", "coefficients", "feature_defaults"):
            if len(model[key]) != len(FEATURE_NAMES):
                raise ValueError(f"{key} has {len(model[key])} entries, expected {len(FEATURE_NAMES)}")

        trained_on = model.get("embedding_model")
        if trained_on and trained_on != EMBEDDING_MODEL:
            # Not fatal: four of the five features are unaffected. But `similarity` comes straight
            # from Qdrant, so re-ingesting under a different embedding model silently moves that
            # feature's scale out from under its coefficient.
            print(f"  [!] fit model was trained against embeddings from {trained_on}, but this "
                  f"deployment uses {EMBEDDING_MODEL}; the similarity feature may be miscalibrated")
        return model

    except FileNotFoundError:
        print(f"  [!] Statistical fit model not found at {FIT_MODEL_FILE}; "
              f"the Evaluator will score from the CV alone")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"  [!] Statistical fit model unusable: {error}")
    return None


def get_model() -> dict | None:
    global _MODEL, _LOADED
    if _LOADED:
        return _MODEL
    with _LOCK:
        if not _LOADED:
            _MODEL = _load()
            _LOADED = True
    return _MODEL


def predict(features: dict[str, float | None]) -> tuple[float, list[str]] | None:
    """Score one candidate from their features. Returns (score out of 10, notes), or None.

    A feature that could not be computed comes in as None and is replaced by its training mean -
    the least-committal value available, and the one that moves the standardised input to zero so
    the feature contributes only through the intercept. Each substitution is reported back in the
    notes rather than silently absorbed, because a score resting on three real features and two
    averages deserves to be described differently from one resting on five.
    """
    model = get_model()
    if model is None:
        return None

    notes: list[str] = []
    total = model["intercept"]
    for index, name in enumerate(FEATURE_NAMES):
        value = features.get(name)
        if value is None:
            value = model["feature_defaults"][index]
            notes.append(f"{name} unavailable, using the training average")
        standardised = (value - model["means"][index]) / model["stds"][index]
        total += standardised * model["coefficients"][index]

    low, high = model["clip"]
    return max(low, min(high, total)), notes


def typical_error() -> float:
    """The model's own out-of-fold MAE, in points. Used to describe the estimate's precision."""
    model = get_model()
    if model is None:
        return 0.0
    return float(model.get("training", {}).get("typical_error", 0.0))

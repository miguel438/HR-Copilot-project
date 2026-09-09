"""Fit the statistical fit model and write it out as plain coefficients.

Runs offline in a throwaway container (see training/README.md). Needs only fit_labels.csv - no
Qdrant, no OpenAI key, no project imports - which is what lets the training step be reproduced
without standing the whole stack up.

Two choices drive everything here and both are about not overstating a model fitted on 10 people:

*   Ridge on five standardised features, not a tree. Each candidate's feature vector barely moves
    across requisitions - their skills and years are fixed - so a tree can identify the *person*
    from the feature values and memorise their labels. A linear model cannot do that.

*   Two grouped cross-validations, reported separately. Leave-one-candidate-out is the headline,
    because the same CV appears in 14 rows and a random split would leak it across folds.
    Leave-one-requisition-out is the honest one: the effective sample size on the role side is 14,
    not 140, and it answers the question an examiner actually asks - does this work on a
    requisition it has never seen?
"""

import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
LABELS = TRAINING_DIR / "fit_labels.csv"
MODEL_OUT = PROJECT_ROOT / "data" / "fit_model.json"
REPORT_OUT = TRAINING_DIR / "report" / "metrics.md"

SCHEMA_VERSION = 1
FEATURE_NAMES = ["skill_overlap", "years_surplus", "years_shortfall", "title_overlap", "similarity"]

# Ordinal bands mapped onto points out of 10. The 2/1 boundary sits either side of
# SCORE_THRESHOLD (7.0), so a band-2 prediction clears screening and a band-1 does not.
# This imposes an equal-interval assumption on an ordinal label, which is the main modelling
# caveat and is restated in the report.
BAND_CENTRES = {0: 0.0, 1: 5.0, 2: 9.0}

# Pairs where the honest answer is genuinely arguable. Reported separately because overall
# accuracy on this dataset is dominated by obvious negatives, and these are the rows a recruiter
# would actually want help with.
HARD_PAIRS = {
    ("req-t01-backend-senior", "James Anderson"),
    ("req-t04-data-analyst", "Noah Kim"),
    ("req-t06-devops", "Alice Chen"),
    ("req-t10-platform-adjacent", "Ryan Patel"),
    ("req-t10-platform-adjacent", "Alice Chen"),
    ("req-t14-fullstack", "Liam Wong"),
}


def load_rows() -> list[dict]:
    with LABELS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit(f"{LABELS} is empty")
    return rows


def build_matrix(rows: list[dict]):
    """Feature matrix, targets and the two grouping vectors.

    A row missing any feature is dropped rather than imputed: at training time an unusable row is
    just a row, and imputing it would invent data. Inference is the opposite case - there a
    missing feature must still produce an estimate, so scoring_model.py substitutes the training
    mean. The asymmetry is deliberate.
    """
    features, targets, by_candidate, by_requisition, kept = [], [], [], [], []
    dropped = 0
    for row in rows:
        try:
            values = [float(row[name]) for name in FEATURE_NAMES]
            band = int(row["band"])
        except (ValueError, KeyError):
            dropped += 1
            continue
        if band not in BAND_CENTRES:
            dropped += 1
            continue
        features.append(values)
        targets.append(BAND_CENTRES[band])
        by_candidate.append(row["candidate_name"])
        by_requisition.append(row["req_id"])
        kept.append(row)
    if dropped:
        print(f"  [!] dropped {dropped} row(s) with a missing feature or an unusable band")
    return (np.array(features), np.array(targets),
            np.array(by_candidate), np.array(by_requisition), kept)


def standardise(matrix: np.ndarray):
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    # A zero-variance feature would divide by zero; 1.0 leaves it centred at zero, which makes
    # its coefficient meaningless rather than infinite. Reported if it happens.
    stds = np.where(stds < 1e-9, 1.0, stds)
    return (matrix - means) / stds, means, stds


def grouped_cv(matrix, targets, groups, alpha):
    """Out-of-fold predictions under LeaveOneGroupOut, standardising inside each fold.

    Standardising inside the fold matters: fitting the scaler on all 140 rows leaks the held-out
    group's distribution into training, which is a small but real optimism.
    """
    predictions = np.zeros_like(targets)
    splitter = LeaveOneGroupOut()
    fold_errors = []
    for train_index, test_index in splitter.split(matrix, targets, groups):
        scaled_train, means, stds = standardise(matrix[train_index])
        model = RidgeCV(alphas=alpha)
        model.fit(scaled_train, targets[train_index])
        scaled_test = (matrix[test_index] - means) / stds
        fold_prediction = np.clip(model.predict(scaled_test), 0.0, 10.0)
        predictions[test_index] = fold_prediction
        fold_errors.append(float(np.abs(fold_prediction - targets[test_index]).mean()))
    return predictions, fold_errors


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling scipy in for one number."""
    def ranks(values):
        order = np.argsort(values)
        result = np.empty(len(values), dtype=float)
        result[order] = np.arange(len(values), dtype=float)
        # average ranks within ties, so heavy band ties do not distort the correlation
        for value in np.unique(values):
            mask = values == value
            result[mask] = result[mask].mean()
        return result
    ra, rb = ranks(a), ranks(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def band_of(score: float) -> int:
    """Nearest band centre - how a predicted score is turned back into a band for accuracy."""
    return min(BAND_CENTRES, key=lambda b: abs(BAND_CENTRES[b] - score))


def main() -> int:
    rows = load_rows()
    matrix, targets, by_candidate, by_requisition, kept = build_matrix(rows)
    print(f"{len(kept)} usable rows | {len(set(by_candidate))} candidates | "
          f"{len(set(by_requisition))} requisitions")

    overrides = sum(1 for r in kept if r.get("suggested_band") != r.get("band"))
    print(f"labels overridden by the reviewer: {overrides}/{len(kept)} "
          f"({overrides / len(kept):.0%})")
    if overrides == 0:
        print("  [!] no overrides: every label is the LLM's. The model is then a distillation of "
              "the Evaluator, not an independent check, and must be described that way.")

    # `source` is absent on rows written before this column existed, which are all real pairs
    # built from the live retrieval store - "original" is the correct default, not a guess.
    synthetic_rows = [r for r in kept if r.get("source") == "synthetic"]
    original_rows = [r for r in kept if r.get("source", "original") != "synthetic"]
    synthetic_overrides = sum(
        1 for r in synthetic_rows if r.get("suggested_band") != r.get("band")
    )
    if synthetic_rows:
        print(f"  of which {len(synthetic_rows)} pairs are synthetic candidates added for "
              f"training-set augmentation ({synthetic_overrides}/{len(synthetic_rows)} "
              f"overridden), {len(original_rows)} from the original corpus")

    band_counts = {b: int((targets == c).sum()) for b, c in BAND_CENTRES.items()}
    print(f"bands: no-fit {band_counts[0]}, partial {band_counts[1]}, strong {band_counts[2]}")

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    # --- the two cross-validations -------------------------------------------------------
    results = {}
    for label, groups in (("candidate", by_candidate), ("requisition", by_requisition)):
        predictions, fold_errors = grouped_cv(matrix, targets, groups, alphas)
        errors = np.abs(predictions - targets)
        predicted_bands = np.array([band_of(p) for p in predictions])
        true_bands = np.array([band_of(t) for t in targets])
        results[label] = {
            "mae": float(errors.mean()),
            "fold_mae_min": min(fold_errors),
            "fold_mae_max": max(fold_errors),
            "fold_mae_stdev": statistics.stdev(fold_errors) if len(fold_errors) > 1 else 0.0,
            "spearman": spearman(predictions, targets),
            "band_accuracy": float((predicted_bands == true_bands).mean()),
            "predictions": predictions,
            "predicted_bands": predicted_bands,
            "true_bands": true_bands,
            "n_folds": len(fold_errors),
        }

    # --- baselines, without which an MAE means nothing -----------------------------------
    # Both MAE and band accuracy, because they can disagree: on a set that is 80% no-fit,
    # always predicting no-fit scores badly on MAE and very well on accuracy. Quoting whichever
    # one happens to flatter the model is the easiest way to overstate it.
    true_bands_all = np.array([band_of(t) for t in targets])
    majority_band = max(band_counts, key=lambda b: band_counts[b])
    majority = BAND_CENTRES[majority_band]

    baselines = {
        "predict the mean": (float(np.abs(targets - targets.mean()).mean()), None),
        "predict the majority band": (
            float(np.abs(targets - majority).mean()),
            float((true_bands_all == majority_band).mean()),
        ),
    }
    for index, name in enumerate(FEATURE_NAMES):
        if name in ("similarity", "skill_overlap"):
            single, _ = grouped_cv(matrix[:, [index]], targets, by_candidate, alphas)
            single_bands = np.array([band_of(p) for p in single])
            baselines[f"{name} alone"] = (
                float(np.abs(single - targets).mean()),
                float((single_bands == true_bands_all).mean()),
            )

    # State plainly whether the five features earn their place. A model that loses to a single
    # feature or to a constant is a finding, not a failure - but it has to be said out loud,
    # because every other number here can be read as if it were good.
    beaten_on_mae = [n for n, (m, _) in baselines.items() if m <= results["candidate"]["mae"]]
    beaten_on_acc = [n for n, (_, a) in baselines.items()
                     if a is not None and a >= results["candidate"]["band_accuracy"]]
    verdict = []
    if beaten_on_mae:
        verdict.append("does NOT beat " + ", ".join(beaten_on_mae) + " on MAE")
    if beaten_on_acc:
        verdict.append("does NOT beat " + ", ".join(beaten_on_acc) + " on band accuracy")
    if not verdict:
        verdict.append("beats every baseline on both MAE and band accuracy")

    # --- final fit on everything ---------------------------------------------------------
    scaled, means, stds = standardise(matrix)
    model = RidgeCV(alphas=alphas)
    model.fit(scaled, targets)

    importance = permutation_importance(
        model, scaled, targets, n_repeats=30, random_state=0, scoring="neg_mean_absolute_error"
    )

    # Written by build_pairs.py, which is the only step that talks to Qdrant. Absent if the CSV
    # was produced some other way, in which case the runtime simply skips the mismatch warning.
    meta_file = TRAINING_DIR / "pairs_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": meta.get("embedding_model"),
        "collection_name": meta.get("collection_name"),
        "model": "ridge",
        "alpha": round(float(model.alpha_), 6),
        "feature_names": FEATURE_NAMES,
        "means": [round(float(v), 9) for v in means],
        "stds": [round(float(v), 9) for v in stds],
        "coefficients": [round(float(v), 9) for v in model.coef_],
        "intercept": round(float(model.intercept_), 9),
        "feature_defaults": [round(float(v), 9) for v in means],
        "clip": [0.0, 10.0],
        "band_centres": {str(k): v for k, v in BAND_CENTRES.items()},
        "training": {
            "n_pairs": len(kept),
            "n_candidates": len(set(by_candidate)),
            "n_requisitions": len(set(by_requisition)),
            "n_synthetic_pairs": len(synthetic_rows),
            "n_original_pairs": len(original_rows),
            "label_override_rate": round(overrides / len(kept), 4),
            "band_counts": band_counts,
            "loco_mae": round(results["candidate"]["mae"], 4),
            "loro_mae": round(results["requisition"]["mae"], 4),
            "loco_band_accuracy": round(results["candidate"]["band_accuracy"], 4),
            "baseline_mae": {k: round(v, 4) for k, (v, _) in baselines.items()},
            "baseline_band_accuracy": {
                k: round(a, 4) for k, (_, a) in baselines.items() if a is not None
            },
            "verdict_vs_baselines": "; ".join(verdict),
            "typical_error": round(results["candidate"]["mae"], 1),
        },
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # The whole offline/runtime split rests on the claim that a dot product over the written
    # coefficients reproduces what scikit-learn just fitted. Check it here rather than trust it:
    # the artifact rounds to 6dp, and a transposed or mis-scaled coefficient would show up as
    # quietly wrong scores in production with nothing to catch it.
    drift = verify_pure_python(artifact, matrix, model)
    if drift > 1e-6:
        sys.exit(f"pure-Python inference disagrees with scikit-learn by {drift:.2e} - "
                 f"refusing to ship this artifact")
    print(f"pure-Python inference matches scikit-learn to {drift:.2e} across {len(matrix)} rows")

    write_report(artifact, results, baselines, kept, targets, model, means, stds, importance, verdict)

    print()
    print(f"{'feature':<18}{'coef (pts/sd)':>15}{'perm. importance':>19}")
    for name, coefficient, imp in zip(FEATURE_NAMES, model.coef_, importance.importances_mean):
        print(f"{name:<18}{coefficient:>15.3f}{imp:>19.3f}")
    print(f"{'intercept':<18}{model.intercept_:>15.3f}")
    print()
    print(f"leave-one-candidate-out    MAE {results['candidate']['mae']:.2f} pts   "
          f"band accuracy {results['candidate']['band_accuracy']:.0%}   "
          f"rho {results['candidate']['spearman']:.2f}")
    print(f"leave-one-requisition-out  MAE {results['requisition']['mae']:.2f} pts   "
          f"band accuracy {results['requisition']['band_accuracy']:.0%}   "
          f"rho {results['requisition']['spearman']:.2f}")
    print()
    print(f"  {'baseline':<30}{'MAE':>7}{'band acc':>10}")
    for name, (mae, accuracy) in sorted(baselines.items(), key=lambda kv: kv[1][0]):
        shown = f"{accuracy:.0%}" if accuracy is not None else "-"
        print(f"  {name:<30}{mae:>7.2f}{shown:>10}")
    print(f"  {'>> the model':<30}{results['candidate']['mae']:>7.2f}"
          f"{results['candidate']['band_accuracy']:>9.0%}")
    print()
    print("VERDICT: the model " + "; ".join(verdict))
    print()
    print(f"wrote {MODEL_OUT}")
    print(f"wrote {REPORT_OUT}")
    return 0


def verify_pure_python(artifact, matrix, model) -> float:
    """Largest disagreement between src/scoring_model.py's arithmetic and scikit-learn's.

    Deliberately reimplements the runtime path from the *written artifact* rather than importing
    scoring_model - importing it would need the project's config, which this script is designed
    not to depend on, and re-reading the file is the stronger test anyway: it exercises the
    rounding that was actually serialised.
    """
    worst = 0.0
    for row in matrix:
        total = artifact["intercept"]
        for index, value in enumerate(row):
            standardised = (value - artifact["means"][index]) / artifact["stds"][index]
            total += standardised * artifact["coefficients"][index]
        low, high = artifact["clip"]
        pure = max(low, min(high, total))

        scaled = (row - np.array(artifact["means"])) / np.array(artifact["stds"])
        reference = float(np.clip(model.predict(scaled.reshape(1, -1))[0], low, high))
        worst = max(worst, abs(pure - reference))
    return worst


def write_report(artifact, results, baselines, kept, targets, model, means, stds, importance, verdict):
    loco, loro = results["candidate"], results["requisition"]
    lines = [
        "# Statistical fit model - training report",
        "",
        f"Generated {artifact['created_utc']} from `training/fit_labels.csv`.",
        "",
        "## Dataset",
        "",
        f"- {artifact['training']['n_pairs']} (requisition, candidate) pairs",
        f"- {artifact['training']['n_candidates']} candidates, "
        f"{artifact['training']['n_requisitions']} requisitions",
        f"- bands: no-fit {artifact['training']['band_counts'][0]}, "
        f"partial {artifact['training']['band_counts'][1]}, "
        f"strong {artifact['training']['band_counts'][2]}",
        f"- reviewer overrode the LLM prefill on "
        f"{artifact['training']['label_override_rate']:.0%} of rows",
        "- one annotator, so there is no inter-annotator agreement to quote",
        "",
        "**How the labels were reviewed, stated plainly.** The reviewer was pointed at the ~29",
        "rows where the LLM's score was ambiguous, and several of those were flagged specifically",
        "because the Evaluator had contradicted its own job-family rule. The overrides then went",
        "in that direction, and `title_overlap` - the job-family feature - is the coefficient that",
        "moved most as a result. No model prediction was ever shown during labelling, so this is",
        "not leakage; but the labels are not a blind sample either, and a reader should know that",
        "the review was steered toward job-family consistency before reading the coefficients.",
        "",
    ]

    n_synthetic = artifact["training"]["n_synthetic_pairs"]
    if n_synthetic:
        lines += [
            "**Synthetic augmentation, stated plainly.** "
            f"{n_synthetic} of the {artifact['training']['n_pairs']} pairs above come from 11 "
            "synthetic candidates added purely to grow and rebalance the training set (see "
            "training/make_synthetic_resumes.py and training/build_pairs_synthetic.py) - they "
            "were never real applicants and never reached the production Qdrant collection or "
            "the live app. Each was written as a deliberately unambiguous fit for one job "
            "family, which is why this batch is easier than the original 140 rows: the near-zero "
            "override rate on it reflects that design, not a claim that the Evaluator has gotten "
            "better at the genuinely hard cases the original corpus contains. Two rows from this "
            "batch are worth reading anyway, because they show the LLM diverging from the "
            "keyword features in the way the whole model design depends on: a senior B2C product "
            "manager scored a partial fit against a B2B-specific requisition despite a perfect "
            "title match (skill_overlap 0.0, correctly downgraded on domain), and a sales "
            "candidate scored a strong fit despite a low literal skill_overlap (0.33) because the "
            "Evaluator read her CV's own phrasing as equivalent to the requisition's wording.",
            "",
        ]

    lines += [
        "## Cross-validation",
        "",
        "Both are grouped, never random: the same CV appears in every requisition's rows, so a",
        "random split leaks a candidate across folds. Scaling is fitted inside each fold.",
        "",
        "| split | folds | MAE (pts) | fold MAE range | band accuracy | Spearman |",
        "|---|---|---|---|---|---|",
        f"| leave-one-candidate-out | {loco['n_folds']} | {loco['mae']:.2f} | "
        f"{loco['fold_mae_min']:.2f}-{loco['fold_mae_max']:.2f} | "
        f"{loco['band_accuracy']:.0%} | {loco['spearman']:.2f} |",
        f"| leave-one-requisition-out | {loro['n_folds']} | {loro['mae']:.2f} | "
        f"{loro['fold_mae_min']:.2f}-{loro['fold_mae_max']:.2f} | "
        f"{loro['band_accuracy']:.0%} | {loro['spearman']:.2f} |",
        "",
        "Leave-one-requisition-out is the number to quote when asked whether this generalises:",
        "the effective sample size on the role side is the number of requisitions, not the number",
        "of pairs. Neither split is fully clean - a held-out candidate's fold still shares",
        "requisitions with training - and a properly blocked double-holdout is not viable at this",
        "size.",
        "",
        "## Baselines",
        "",
        "Both metrics, because they disagree: on a set that is mostly no-fit, always predicting",
        "no-fit scores badly on MAE and very well on accuracy. Quoting whichever one flatters the",
        "model is the easiest way to overstate it.",
        "",
        "| baseline | MAE (pts) | band accuracy |",
        "|---|---|---|",
    ]
    for name, (value, accuracy) in sorted(baselines.items(), key=lambda kv: kv[1][0]):
        shown = f"{accuracy:.0%}" if accuracy is not None else "-"
        lines.append(f"| {name} | {value:.2f} | {shown} |")
    lines += [
        f"| **the model (leave-one-candidate-out)** | **{loco['mae']:.2f}** | "
        f"**{loco['band_accuracy']:.0%}** |",
        "",
        f"**Verdict: the model {'; '.join(verdict)}.**",
        "",
        "The majority-band baseline looks strong because most pairs are obvious no-fits. That is",
        "why per-band error below matters more than the overall number.",
        "",
    ]

    # A verdict of "beats every baseline" is worthless if it is won by a hundredth of a point on
    # 140 rows. Say so in the report rather than leaving the margin to be worked out.
    best_baseline = min(baselines.items(), key=lambda kv: kv[1][0])
    margin = best_baseline[1][0] - loco["mae"]
    if margin < 0.10:
        lines += [
            f"**Read that verdict carefully.** The margin over *{best_baseline[0]}* is "
            f"{margin:+.2f} points of MAE on {len(kept)} rows, far inside the fold-to-fold spread "
            f"of {loco['fold_mae_min']:.2f}-{loco['fold_mae_max']:.2f}. The honest claim is that "
            "the five features are *no worse* than the single best one, not that they are "
            "measurably better. What the extra features buy is interpretability - a coefficient "
            "table that says why a candidate scored what they did - rather than accuracy.",
            "",
        ]

    # Recall on the strong band is the number a recruiter actually cares about: a screening tool
    # that misses good people is failing at its job, and overall accuracy hides that completely
    # when 79% of the data is no-fit.
    strong_mask = loco["true_bands"] == 2
    if strong_mask.sum():
        strong_recall = float((loco["predicted_bands"][strong_mask] == 2).mean())
        missed = int(strong_mask.sum() - (loco["predicted_bands"][strong_mask] == 2).sum())
        lines += [
            "### The number that matters most",
            "",
            f"Of {int(strong_mask.sum())} candidates labelled a strong fit, the model recovers "
            f"**{strong_recall:.0%}** ({missed} missed).",
            "That is the figure to quote about usefulness, not overall accuracy: a screening aid",
            "that under-rates good candidates fails at the job, and with most pairs being obvious",
            "no-fits the headline accuracy barely moves when it happens. It is also why the",
            "estimate is advisory in the report and never touches SCORE_THRESHOLD.",
            "",
        "## Error by true band",
        "",
        "| true band | n | MAE (pts) |",
        "|---|---|---|",
    ]
    for band, centre in BAND_CENTRES.items():
        mask = targets == centre
        if mask.sum():
            band_mae = float(np.abs(loco["predictions"][mask] - targets[mask]).mean())
            lines.append(f"| {band} | {int(mask.sum())} | {band_mae:.2f} |")

    lines += [
        "",
        "## Confusion (leave-one-candidate-out)",
        "",
        "| true \\ predicted | 0 | 1 | 2 |",
        "|---|---|---|---|",
    ]
    for true_band in (0, 1, 2):
        cells = [int(((loco["true_bands"] == true_band) & (loco["predicted_bands"] == p)).sum())
                 for p in (0, 1, 2)]
        lines.append(f"| {true_band} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "## Coefficients",
        "",
        f"Ridge, alpha {artifact['alpha']}. Units are points out of 10 per standard deviation of",
        "the feature, so they are directly comparable with each other.",
        "",
        "| feature | coefficient | permutation importance |",
        "|---|---|---|",
    ]
    for name, coefficient, imp in zip(FEATURE_NAMES, model.coef_, importance.importances_mean):
        lines.append(f"| {name} | {coefficient:+.3f} | {imp:.3f} |")
    lines.append(f"| _intercept_ | {model.intercept_:+.3f} | |")

    residuals = np.abs(loco["predictions"] - targets)
    worst = np.argsort(residuals)[::-1][:3]
    lines += [
        "",
        "## Largest residuals",
        "",
        "The three pairs the model gets most wrong out of fold. Worth reading before trusting it.",
        "",
        "| requisition | candidate | true | predicted |",
        "|---|---|---|---|",
    ]
    for index in worst:
        row = kept[index]
        lines.append(f"| {row['req_id']} | {row['candidate_name']} | "
                     f"{targets[index]:.1f} | {loco['predictions'][index]:.1f} |")

    hard = [i for i, row in enumerate(kept)
            if (row["req_id"], row["candidate_name"]) in HARD_PAIRS]
    if hard:
        hard_mae = float(residuals[hard].mean())
        lines += [
            "",
            "## Deliberately hard pairs",
            "",
            f"{len(hard)} pairs where the honest answer is arguable. Out-of-fold MAE "
            f"**{hard_mae:.2f} pts** against {loco['mae']:.2f} overall - the gap is the honest",
            "measure of how much of the headline number comes from easy negatives.",
            "",
            "| requisition | candidate | true | predicted |",
            "|---|---|---|---|",
        ]
        for index in hard:
            row = kept[index]
            lines.append(f"| {row['req_id']} | {row['candidate_name']} | "
                         f"{targets[index]:.1f} | {loco['predictions'][index]:.1f} |")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

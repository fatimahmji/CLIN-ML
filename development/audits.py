"""Data-quality, attrition, comparator and numerical calibration diagnostics."""
# Allow these utilities to import the main project modules.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from data_preparation import build_eye_level_labels
from evaluation import metrics


def calibration_summary(labels, probabilities):
    """Fixed ten-bin ECE is descriptive and depends on the binning choice."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    bin_ids = np.minimum((probabilities * 10).astype(int), 9)
    bins = []
    for index in range(10):
        selected = bin_ids == index
        if selected.any():
            bins.append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "eyes": int(selected.sum()),
                    "mean_probability": float(probabilities[selected].mean()),
                    "observed_rate": float(labels[selected].mean()),
                }
            )
    ece = sum(
        row["eyes"] * abs(row["mean_probability"] - row["observed_rate"])
        for row in bins
    ) / len(labels)
    return {
        "mean_prediction": float(probabilities.mean()),
        "observed_prevalence": float(labels.mean()),
        "mean_prediction_minus_prevalence": float(probabilities.mean() - labels.mean()),
        "ece_10_equal_width_bins": float(ece),
        "brier": float(brier_score_loss(labels, probabilities)),
        "bins": bins,
    }


def run_audits(source, directory):
    directory = Path(directory)
    audit, raw_blanks, visits, keys = build_eye_level_labels(source)
    quality = {
        "raw_blank_rates": raw_blanks,
        "invalid_va_records": int(visits.va_invalid.sum()),
        "invalid_baseline_va_records": int(visits.baseline_va_invalid.sum()),
        "post_cleaning_missing_rates": visits[["va", "baseline_va"]]
        .isna()
        .mean()
        .to_dict(),
        "unknown_ethnicity_visit_records": int(visits.ethnicity.eq("Unknown").sum()),
        "visit_rows": len(visits),
        "duplicate_visit_content_rows": int(
            visits.drop(columns=["source_row", "Unnamed: 0"], errors="ignore")
            .duplicated()
            .sum()
        ),
        "source_cohort_restriction": "Original authors excluded eyes receiving fewer than three injections; post-baseline selection limits application to all initiators.",
    }
    (directory / "data_quality.json").write_text(json.dumps(quality, indent=2))
    visits.to_csv(directory / "cleaned_visits.csv", index=False)
    audit["cohort"] = np.where(audit.status.eq("eligible"), "included", "excluded")
    numeric = audit.groupby("cohort").baseline_va.agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    numeric.to_csv(directory / "attrition_baseline_va.csv")
    included = audit.loc[audit.cohort.eq("included"), "baseline_va"].dropna()
    excluded = audit.loc[audit.cohort.eq("excluded"), "baseline_va"].dropna()
    pooled_sd = np.sqrt((included.var(ddof=1) + excluded.var(ddof=1)) / 2)
    numeric_smd = (
        float((included.mean() - excluded.mean()) / pooled_sd)
        if pooled_sd > 0
        else None
    )
    categorical = []
    for variable in ["baseline_age", "gender", "ethnicity"]:
        for cohort, frame in audit.groupby("cohort"):
            for category, count in (
                frame[variable].fillna("Missing").value_counts().items()
            ):
                categorical.append(
                    {
                        "variable": variable,
                        "category": category,
                        "cohort": cohort,
                        "eyes": int(count),
                        "fraction": float(count / len(frame)),
                    }
                )
    pd.DataFrame(categorical).to_csv(
        directory / "attrition_categories.csv", index=False
    )
    attrition = {
        "eye_counts": audit.cohort.value_counts().to_dict(),
        "baseline_va_standardized_mean_difference_included_minus_excluded": numeric_smd,
        "notes": "Descriptive eye-level comparison. Patients can contribute eyes to both inclusion categories. Missing baseline VA excluded from numeric summaries, not replaced. Associations do not quantify or correct selection bias.",
    }
    (directory / "attrition.json").write_text(json.dumps(attrition, indent=2))

    # A fixed acuity-only comparator answers whether age contributes useful signal.
    eyes = pd.read_csv(directory / "eye_level_dataset.csv")
    train = eyes[eyes.partition.eq("train")]
    comparator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1, max_iter=2000, random_state=42)),
        ]
    )
    comparator.fit(train[["baseline_va"]], train.response)
    joblib.dump(comparator, directory / "acuity_only_comparator.joblib")
    reference = joblib.load(directory / "clinical_model.joblib")
    comparisons = {}
    calibrations = {}
    for partition, frame in eyes.groupby("partition"):
        reference_p = reference.predict_proba(frame[["baseline_va", "baseline_age"]])[
            :, 1
        ]
        comparator_p = comparator.predict_proba(frame[["baseline_va"]])[:, 1]
        comparisons[partition] = {
            "acuity_only": metrics(frame.response, comparator_p),
            "acuity_age": metrics(frame.response, reference_p),
        }
        calibrations[partition] = {
            "acuity_only": calibration_summary(frame.response, comparator_p),
            "acuity_age": calibration_summary(frame.response, reference_p),
        }
    (directory / "comparator.json").write_text(json.dumps(comparisons, indent=2))
    (directory / "calibration.json").write_text(json.dumps(calibrations, indent=2))

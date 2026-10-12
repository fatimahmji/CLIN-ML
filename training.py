"""Reproduce provisional CLIN-ML research analysis. Never a deployment approval."""
from data_preparation import build_eye_level_labels
from evaluation import metrics
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    brier_score_loss,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance

FEATURES = ["baseline_va", "baseline_age"]
AGES = ["[20,30]", "(30,40]", "(40,50]", "(50,60]", "(60,70]", "(70,80]", "(80,100]"]


def make_preprocessor():
    """Build fixed reference configuration; no test-driven tuning."""
    return ColumnTransformer(
        [
            (
                "va",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["baseline_va"],
            ),
            (
                "age",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="error", sparse_output=False),
                        ),
                    ]
                ),
                ["baseline_age"],
            ),
        ]
    )


def make_models():
    """Build fixed reference configuration; no test-driven tuning."""
    return {
        "logistic_regression": LogisticRegression(C=1, max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=2,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            min_samples_leaf=20,
            learning_rate=0.05,
            random_state=42,
        ),
    }


def save_evaluation_charts(test_labels, probabilities, selected, output_directory):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    RocCurveDisplay.from_predictions(test_labels, probabilities, ax=axes[0, 0])
    PrecisionRecallDisplay.from_predictions(test_labels, probabilities, ax=axes[0, 1])
    CalibrationDisplay.from_predictions(
        test_labels, probabilities, n_bins=8, strategy="quantile", ax=axes[1, 0]
    )
    confusion = confusion_matrix(test_labels, probabilities >= 0.5)
    axes[1, 1].imshow(confusion, cmap="Blues")
    axes[1, 1].set(
        xlabel="Predicted",
        ylabel="Observed",
        xticks=[0, 1],
        yticks=[0, 1],
        title="Confusion matrix (threshold 0.50)",
    )
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(confusion[i, j]), ha="center", va="center")
    fig.suptitle(selected + " — provisional held-out patients")
    fig.tight_layout()
    fig.savefig(output_directory / "evaluation.png", dpi=160)
    plt.close(fig)


def save_feature_importance(
    pipeline, features, labels, validation_indices, output_directory
):
    importance = permutation_importance(
        pipeline,
        features.iloc[validation_indices],
        labels.iloc[validation_indices],
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_auroc_decrease": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).to_csv(output_directory / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(FEATURES, importance.importances_mean, xerr=importance.importances_std)
    ax.set(
        xlabel="Validation AUROC decrease after permutation",
        title="Feature importance (30 permutations)",
    )
    fig.tight_layout()
    fig.savefig(output_directory / "feature_importance.png", dpi=160)
    plt.close(fig)


def main(source, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    audit, raw_missing, visits, keys = build_eye_level_labels(source)
    audit.to_csv(output_directory / "eye_audit.csv", index=False)
    eye_dataset = audit[audit.status.eq("eligible")].copy()
    eye_dataset["response"] = eye_dataset.response.astype(int)
    development_indices, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026).split(
            eye_dataset, groups=eye_dataset.anon_id
        )
    )
    train_indices, validation_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=2027).split(
            eye_dataset.iloc[development_indices],
            groups=eye_dataset.iloc[development_indices].anon_id,
        )
    )
    train_indices, validation_indices = (
        development_indices[train_indices],
        development_indices[validation_indices],
    )
    partitions = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    eye_dataset["partition"] = ""
    groups = []
    for name, indices in partitions.items():
        eye_dataset.iloc[indices, eye_dataset.columns.get_loc("partition")] = name
        groups.append(set(eye_dataset.iloc[indices].anon_id))
        assert eye_dataset.iloc[indices].response.nunique() == 2
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    eye_dataset.to_csv(output_directory / "eye_level_dataset.csv", index=False)
    features = eye_dataset[FEATURES]
    labels = eye_dataset.response
    preprocessor = make_preprocessor()
    models = make_models()
    model_results = {}
    trained_pipelines = {}
    cross_validation = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(
            pipeline,
            features.iloc[train_indices],
            labels.iloc[train_indices],
            groups=eye_dataset.iloc[train_indices].anon_id,
            cv=cross_validation,
            scoring="roc_auc",
        )
        pipeline.fit(features.iloc[train_indices], labels.iloc[train_indices])
        trained_pipelines[name] = pipeline
        model_results[name] = {
            "grouped_training_cv_auroc": scores.tolist(),
            **{
                k: metrics(
                    labels.iloc[indices],
                    pipeline.predict_proba(features.iloc[indices])[:, 1],
                )
                for k, indices in partitions.items()
                if k != "test"
            },
        }
        joblib.dump(pipeline, output_directory / (name + ".joblib"))
    selected = max(model_results, key=lambda n: model_results[n]["validation"]["auroc"])
    pipeline = trained_pipelines[selected]
    probabilities = pipeline.predict_proba(features.iloc[test_indices])[:, 1]
    test_labels = labels.iloc[test_indices]
    model_results[selected]["test"] = metrics(test_labels, probabilities)
    rng = np.random.default_rng(42)
    test = eye_dataset.iloc[test_indices].reset_index(drop=True)
    test_patient_ids = test.anon_id.unique()
    bootstrap_metric_samples = {
        k: [] for k in ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]
    }
    for _ in range(1000):
        sample_indices = np.concatenate(
            [
                np.flatnonzero(test.anon_id.eq(a))
                for a in rng.choice(
                    test_patient_ids, len(test_patient_ids), replace=True
                )
            ]
        )
        if test.response.iloc[sample_indices].nunique() < 2:
            continue
        m = metrics(test.response.iloc[sample_indices], probabilities[sample_indices])
        for k in bootstrap_metric_samples:
            bootstrap_metric_samples[k].append(m[k])
    model_results[selected]["test_cluster_bootstrap_95ci"] = {
        k: np.quantile(v, [0.025, 0.975]).tolist()
        for k, v in bootstrap_metric_samples.items()
    }
    test_predictions = test[keys + ["response"]].copy()
    test_predictions["probability"] = probabilities
    test_predictions.to_csv(output_directory / "heldout_predictions.csv", index=False)
    save_evaluation_charts(test_labels, probabilities, selected, output_directory)
    save_feature_importance(
        pipeline, features, labels, validation_indices, output_directory
    )
    joblib.dump(pipeline, output_directory / "clinical_model.joblib")
    loaded = joblib.load(output_directory / "clinical_model.joblib")
    assert np.allclose(
        loaded.predict_proba(features.iloc[test_indices])[:, 1], probabilities
    )
    schema = {
        "version": "0.1.1-research",
        "features": {
            "baseline_va": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "unit": "ETDRS letters",
            },
            "baseline_age": {"type": "string", "enum": AGES},
        },
        "outcome": "gain >=5 letters; nearest valid day to 180 within inclusive 150–210; ties earlier day",
        "clinical_signoff": False,
        "source_units_verified": True,
        "baseline_definition": "Dataset injection number 1; prior treatment elsewhere unconfirmed",
        "population_restriction": "Source cohort required at least three injections, then valid outcome follow-up",
        "deployment_approved": False,
        "selected_model": selected,
        "threshold": 0.5,
    }
    summary = {
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "visits": len(visits),
        "patients": visits.anon_id.nunique(),
        "eyes": len(audit),
        "raw_missing_rates": raw_missing,
        "out_of_range_va_records": int(visits["va_invalid"].sum()),
        "exclusions": audit.status.value_counts().to_dict(),
        "partitions": {
            k: {
                "eyes": len(indices),
                "patients": eye_dataset.iloc[indices].anon_id.nunique(),
                "responders": int(labels.iloc[indices].sum()),
                "prevalence": float(labels.iloc[indices].mean()),
            }
            for k, indices in partitions.items()
        },
        "selected_model": selected,
        "baseline_missing_rates": eye_dataset[FEATURES].isna().mean().to_dict(),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "models": model_results,
    }
    for name, obj in [("schema.json", schema), ("results.json", summary)]:
        (output_directory / name).write_text(json.dumps(obj, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--input", required=True)
    a.add_argument("--output", default=".")
    args = a.parse_args()
    main(args.input, args.output)

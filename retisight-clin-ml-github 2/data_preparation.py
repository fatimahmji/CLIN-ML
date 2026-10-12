"""Validate visits and construct auditable eye-level outcome labels.

ETDRS values above 100 are documented source errors: convert them to missing,
never clip them. No eligible follow-up means an unknown outcome, never response=0.
"""
from pathlib import Path
import numpy as np
import pandas as pd

AGES = ["[20,30]", "(30,40]", "(40,50]", "(50,60]", "(60,70]", "(70,80]", "(80,100]"]
KEYS = ["anon_id", "eye"]
REQUIRED = KEYS + [
    "gender",
    "ethnicity",
    "baseline_age",
    "baseline_va",
    "va",
    "follow_up_days",
    "inj_num",
    "inj_given",
]


def read_visits(source):
    """Check required fields; preserve original blanks separately from invalid VA."""
    visits = pd.read_csv(source)
    missing_columns = set(REQUIRED) - set(visits.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    raw_missing_rates = visits.isna().mean().to_dict()
    if visits[KEYS].isna().any().any():
        raise ValueError("Missing patient or eye identifiers")
    if not visits.eye.isin(["r", "l"]).all():
        raise ValueError("Unsupported eye identifier")
    visits["source_row"] = np.arange(len(visits)) + 2
    for column in ["va", "baseline_va", "follow_up_days", "inj_num"]:
        converted = pd.to_numeric(visits[column], errors="coerce")
        malformed = visits[column].notna() & converted.isna()
        if malformed.any():
            raise ValueError(f"Non-numeric values in {column}")
        visits[column] = converted
    if visits.follow_up_days.isna().any() or (visits.follow_up_days < 0).any():
        raise ValueError("Follow-up days must be present and nonnegative")
    if not visits.inj_given.isin(["y", "n"]).all():
        raise ValueError("Unsupported injection flag")
    for column in ["va", "baseline_va"]:
        invalid = visits[column].notna() & (
            ~visits[column].between(0, 100) | visits[column].mod(1).ne(0)
        )
        visits[column + "_invalid"] = invalid
        visits.loc[invalid, column] = np.nan
    return visits, raw_missing_rates


def select_followup(eye_visits):
    """Nearest valid VA to day 180; ties earlier; conflicting same-day VA excludes."""
    candidates = eye_visits[
        eye_visits.follow_up_days.between(150, 210) & eye_visits.va.notna()
    ].copy()
    if candidates.empty:
        return None, "no_valid_window_va"
    candidates["distance"] = (candidates.follow_up_days - 180).abs()
    candidates = candidates.sort_values(["distance", "follow_up_days", "source_row"])
    chosen = candidates.iloc[0]
    if (
        candidates.loc[
            candidates.follow_up_days.eq(chosen.follow_up_days), "va"
        ].nunique()
        > 1
    ):
        return None, "conflicting_selected_day_va"
    return chosen, "eligible"


def label_eye(patient_id, eye, visits):
    """Apply one fixed eligibility policy to one eye; never search for nicer labels."""
    record = {
        "anon_id": patient_id,
        "eye": eye,
        "status": "eligible",
        "response": np.nan,
    }
    baseline_rows = visits[visits.follow_up_days.eq(0)]
    if len(baseline_rows) != 1:
        record["status"] = "ambiguous_or_missing_day0"
        return record
    baseline = baseline_rows.iloc[0]
    for name in ["baseline_va", "baseline_age", "gender", "ethnicity"]:
        record[name] = baseline[name]
    record["baseline_source_row"] = int(baseline.source_row)
    if baseline.inj_given != "y":
        record["status"] = "no_day0_injection"
    elif (
        pd.isna(baseline.baseline_va)
        or pd.isna(baseline.va)
        or baseline.baseline_va != baseline.va
    ):
        record["status"] = "invalid_or_inconsistent_baseline_va"
    elif baseline.baseline_age not in AGES:
        record["status"] = "invalid_age"
    else:
        followup, status = select_followup(visits)
        record["status"] = status
        if followup is not None:
            gain = int(followup.va - baseline.baseline_va)
            record.update(
                followup_day=int(followup.follow_up_days),
                followup_va=int(followup.va),
                followup_source_row=int(followup.source_row),
                gain=gain,
                response=int(gain >= 5),
            )
    return record


def build_eye_level_labels(source):
    """Return all-eye audit, raw blank rates, cleaned visits and linkage keys."""
    visits, blank_rates = read_visits(source)
    records = [
        label_eye(patient, eye, group)
        for (patient, eye), group in visits.groupby(KEYS, sort=True)
    ]
    return pd.DataFrame(records), blank_rates, visits, KEYS

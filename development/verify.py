"""Check the release against frozen reference labels, splits and predictions."""
# Allow these utilities to import the main project modules.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from inference import predict_response


def verify_release(directory, reference_directory, source):
    directory = Path(directory)
    reference_directory = Path(reference_directory)
    reference = pd.read_csv(reference_directory / "eye_level_dataset.csv")
    current = pd.read_csv(directory / "eye_level_dataset.csv")
    keys = ["anon_id", "eye"]
    columns = keys + [
        "baseline_va",
        "baseline_age",
        "response",
        "followup_day",
        "followup_va",
        "partition",
    ]
    pd.testing.assert_frame_equal(
        reference[columns].sort_values(keys).reset_index(drop=True),
        current[columns].sort_values(keys).reset_index(drop=True),
        # Marking invalid scores missing makes the CSV column float-valued;
        # compare the actual retained clinical values, not int/float storage.
        check_dtype=False,
    )
    assert current.groupby("anon_id").partition.nunique().max() == 1
    old_predictions = pd.read_csv(reference_directory / "heldout_predictions.csv")
    predictions = pd.read_csv(directory / "heldout_predictions.csv")
    joined = old_predictions.merge(
        predictions, on=keys, suffixes=("_old", "_new"), validate="one_to_one"
    )
    np.testing.assert_allclose(
        joined.probability_old, joined.probability_new, atol=1e-12
    )
    pipeline = joblib.load(directory / "clinical_model.joblib")
    test = current[current.partition.eq("test")].merge(
        predictions, on=keys, validate="one_to_one"
    )
    np.testing.assert_allclose(
        pipeline.predict_proba(test[["baseline_va", "baseline_age"]])[:, 1],
        test.probability,
        atol=1e-12,
    )
    source_visits = pd.read_csv(source)
    for row in current.itertuples():
        candidates = source_visits[
            source_visits.anon_id.eq(row.anon_id)
            & source_visits.eye.eq(row.eye)
            & source_visits.follow_up_days.between(150, 210)
            & source_visits.va.between(0, 100)
        ]
        distances = (candidates.follow_up_days - 180).abs()
        assert (
            row.followup_day
            == candidates.loc[distances.eq(distances.min()), "follow_up_days"].min()
        )
        assert (
            row.followup_va == source_visits.iloc[int(row.followup_source_row) - 2].va
        )
    assert 0 <= predict_response(60, "(60,70]", directory) <= 1
    for bad in [-1, 101, float("nan"), True]:
        try:
            predict_response(bad, "(60,70]", directory)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid inference input accepted")
    status = {
        "reference_label_and_partition_parity": True,
        "reference_test_probability_parity": True,
        "source_followup_selection": True,
        "export_reload": True,
        "inference_boundaries": True,
    }
    (directory / "verification.json").write_text(json.dumps(status, indent=2))
    print(
        "PASS: frozen labels, partitions and predictions; source follow-up; reload and inference"
    )

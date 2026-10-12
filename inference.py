"""Research-only inference. Only load joblib files from trusted sources."""
from pathlib import Path
import json
import math
import joblib
import pandas as pd


def predict_response(baseline_va, baseline_age, model_dir=None):
    root = Path(model_dir) if model_dir else Path(__file__).parent
    schema = json.loads((root / "schema.json").read_text())
    if (
        isinstance(baseline_va, bool)
        or not isinstance(baseline_va, (int, float))
        or not math.isfinite(baseline_va)
        or not 0 <= baseline_va <= 100
    ):
        raise ValueError(
            "baseline_va must be a finite ETDRS letter score between 0 and 100"
        )
    if baseline_age not in schema["features"]["baseline_age"]["enum"]:
        raise ValueError("Unknown baseline_age category")
    pipe = joblib.load(root / "clinical_model.joblib")
    return float(
        pipe.predict_proba(
            pd.DataFrame([{"baseline_va": baseline_va, "baseline_age": baseline_age}])
        )[0, 1]
    )

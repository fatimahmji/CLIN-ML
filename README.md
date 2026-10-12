# RetiSight — clinical response prediction

Clinical-only research model for predicting a gain of at least 5 ETDRS letters approximately six months after the first recorded anti-VEGF injection for DME.

The model uses baseline visual acuity and age category. It is a research prototype; clinical approval and external validation remain pending. It does not recommend treatment changes.

## Results

The selected Logistic Regression model was evaluated on 340 eyes from 255 held-out patients. Both eyes of each patient remain in the same partition.

| Metric | Result |
|---|---:|
| AUROC | 0.692 |
| AUPRC (average precision) | 0.700 |
| Sensitivity | 68.8% |
| Specificity | 60.3% |
| F1 | 0.686 |
| Brier score | 0.223 |

Sensitivity, specificity and F1 use a probability threshold of 0.50. AUROC is not accuracy. These are results from the saved reference run; the test set has since been inspected during project development and is not a fresh external validation set.


## Setup

Use Python 3.11. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_labels.py
```

## Use the saved model

```python
from inference import predict_response

probability = predict_response(baseline_va=60, baseline_age="(60,70]")
print(probability)
```

The return value is a probability between 0 and 1. See [schema.json](schema.json) for accepted age categories and input definitions. The exported model includes preprocessing. Only load trusted Joblib artifacts. A website API wrapper is not included.

## Reproduce training and checks

The raw CSV and patient-level reference tables are deliberately stored separately from this repository. Obtain the original `200319_DMO_report1_anonymised.csv` from the [Dryad dataset](https://datadryad.org/dataset/doi:10.5061/dryad.pzgmsbcfw). The exact source checksum is recorded in `results.json`.

```bash
python run_release.py \
  --input "/path/to/200319_DMO_report1_anonymised.csv" \
  --reference "/path/to/reference" \
  --output generated
```

This runs label tests, regenerates eye-level labels, trains Logistic Regression, Random Forest and Gradient Boosting, produces audits and charts, verifies labels/partitions/predictions against the frozen reference, and writes the report. `generated/` is ignored by Git so rerunning does not overwrite the committed reference results. Missing reference files cause an early error.

To train without frozen-reference comparison, call the training function directly, then generate audits and the report:

```python
from training import main
from audits import run_audits
from make_report import make_report

source = "/path/to/200319_DMO_report1_anonymised.csv"
main(source, "generated")
run_audits(source, "generated")
make_report("generated")
```

This alternative does not perform frozen-reference parity checks.

## Source files

| File | Purpose |
|---|---|
| `data_preparation.py` | Visit cleaning, baseline/follow-up selection and response labels |
| `training.py` | Patient splits, preprocessing, three models and evaluation charts |
| `evaluation.py` | Metric calculations |
| `inference.py` | Baseline input validation and response probability |
| `audits.py` | Data quality, attrition, calibration and acuity-only comparison |
| `test_labels.py` | Synthetic tests of labeling and calibration rules |
| `verify.py` | Source follow-up, reference parity and model reload checks |
| `run_release.py` | Complete reproducible workflow |
| `make_report.py` | Report generation from computed outputs |

[STEP_BY_STEP.md](STEP_BY_STEP.md) explains the code. 

## Outcome and scope

Data source and cleaning: The dataset contains routine-care Moorfields DME records, with visual acuity recorded in ETDRS letters and follow-up days measured from the first recorded injection. Each eligible eye requires a unique day-zero injection visit and valid baseline measurements. Invalid VA scores outside 0–100 are treated as missing. Conflicting measurements on the selected follow-up day lead to exclusion. The first recorded injection does not confirm that the patient had never received treatment elsewhere, and routine VA should not be described as standardized BCVA.

Follow-up is the valid VA measurement nearest day 180 within days 150–210, with earlier-day tie breaking. Response means follow-up VA minus baseline VA is at least 5 letters. Eyes without eligible follow-up remain unlabeled. Features contain no follow-up measurements or future injection counts.



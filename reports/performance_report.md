# RetiSight CLIN-ML research release 0.1.1

**Status: tested research implementation; clinical endpoint approval and independent validation pending. V1 reference results are preserved.**

This release improves code organization, tests, provenance, data-quality reporting, attrition analysis and evaluation diagnostics. It is not an attempt to choose another winner based on the previously inspected test cohort.

## Data and cohort

Source: 40,281 visits, 1,964 patients, 2,614 eyes. Retained: 1,675 eyes from 1,273 patients, including 893 responders (53.3%). Exclusions: 933 without valid window VA and six invalid/inconsistent baseline VA. All 18 above-range visit scores are marked missing in cleaned_visits.csv; original blanks and newly invalid values are reported separately. Unknown ethnicity is counted, not treated as a known clinical category. Repeated exported baseline errors are counted separately from unique eyes.

Dryad confirms ETDRS units, DME/anti-VEGF provenance and baseline as injection number 1. The eyedata documentation identifies the 18 scores above 100 as unrecoverable data-entry errors. First recorded injection does not establish absence of prior treatment elsewhere. VA is the highest routine visit value regardless of measurement method, not necessarily standardized BCVA. See the data source and limitations in README.md.

**Source selection:** the original cohort excluded eyes receiving fewer than three injections. This uses subsequent treatment information in cohort construction, despite our baseline-only predictors. These results cannot automatically generalize to all treatment starters.

## Attrition assessment

Mean valid baseline VA: included 61.03, excluded 60.50 letters. Standardized mean difference (included minus excluded): 0.035.

See attrition_baseline_va.csv and attrition_categories.csv for age, sex and ethnicity distributions. Summaries are eye-level and descriptive; patients can contribute eyes to both groups. Missing baseline values are omitted from numeric summaries. These comparisons neither prove missing-at-random follow-up nor remove selection bias.

## Model evaluation

Training, validation and test patients are disjoint. Both eyes stay together. Original five-fold training CV, model settings, validation selection and fixed 0.50 threshold are preserved. The exported model remains fitted on training patients.

| Metric | Reference acuity + age | Fixed acuity-only comparator |
|---|---:|---:|
| auroc | 0.692 | 0.701 |
| auprc | 0.700 | 0.705 |
| sensitivity | 0.688 | 0.693 |
| specificity | 0.603 | 0.583 |
| f1 | 0.686 | 0.684 |
| brier | 0.223 | 0.223 |

The acuity-only logistic comparator is trained on the same training patients with C=1 and no tuning. It is a diagnostic comparison, not a replacement selected from test scores. No uncertainty-based claim about incremental age benefit is made. Average precision is used for AUPRC. Lower Brier is better. Original patient-cluster intervals are preserved in results.json.

## Numerical calibration

| Partition | Mean prediction | Observed rate | Prediction minus observed | ECE (10 fixed bins) |
|---|---:|---:|---:|---:|
| train | 0.535 | 0.535 | 0.000 | 0.041 |
| validation | 0.545 | 0.506 | 0.039 | 0.070 |
| test | 0.537 | 0.556 | -0.018 | 0.101 |

ECE is an eye-weighted average absolute gap between predicted and observed rates within ten equal-width probability bins. It depends on binning, has no cluster confidence interval here and is not proof of clinical calibration. The quantile-bin calibration plot uses a different binning choice. No probability recalibration is introduced in this release.

## Overfitting and limitations

Selected model: logistic_regression. Train AUROC 0.669; validation AUROC 0.680; test AUROC 0.692. Similar train/validation scores do not demonstrate external generalization. All original model train/validation metrics and CV folds are in results.json.

The test set was already inspected in V1 and V2. This is a reproducibility/quality release, not new independent validation. Small feature set, follow-up attrition, source selection, VA ceiling effects/regression to mean and uncertain prior treatment constrain inference. No treatment-switch recommendation, causal treatment benefit, clinically approved threshold or fairness certification is provided. No OCT thickness outcome can be created from this CSV.

## Verification

Thirteen synthetic unit tests cover window boundaries, nearest-visit selection rather than best outcome, equal-distance ties, exact threshold, unknown outcomes, invalid acuity, conflicting/duplicate records, baseline eligibility, missing identifiers and a calibration boundary case. Independent checks compare regenerated labels, partitions and held-out predictions with frozen V1 artifacts; source follow-up selection, inference and export/reload are also checked. See verification.json.

## Reproduction

README.md provides the single run_release.py command. Source modules, frozen reference artifacts, pinned dependencies, source hash, intermediate audit tables and final outputs are included. See STEP_BY_STEP.md for the explanation and model_card.md for intended use.

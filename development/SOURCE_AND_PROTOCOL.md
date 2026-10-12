# Source definitions and fixed protocol

## Documented facts

- [Dryad source](https://datadryad.org/dataset/doi:10.5061/dryad.pzgmsbcfw): adult Moorfields DME cohort treated with anti-VEGF; VA and baseline VA are ETDRS scores; day zero is injection number 1. Visit VA is the highest recorded value regardless of measurement method. Original eligibility required at least three injections.
- [eyedata documentation](https://search.r-project.org/CRAN/refmans/eyedata/html/dme.html): 18 original visit VA values above 100 are unrecoverable data-entry errors, assigned missing in the curated version. This release explicitly marks invalid measurements missing and records separate invalid-value flags. Repeated baseline_va errors are counted separately and can span many visits from one eye.
- [Associated study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8182885/): routine-care outcomes, not a validation of our six-month >=5-letter target. Aflibercept/ranibizumab and switching are described; the CSV does not expose drug-specific inputs.
- Local original file MD5 `f1c20f49fd611f03ffa9440ec1039bb7` matches the [archived file](https://zenodo.org/records/3962247). SHA256 is recorded by the pipeline.

## Fixed research protocol

One row per `(anon_id, eye)`. Require one day-zero injection visit, valid baseline VA equal to visit VA, and a supported age band. Select valid VA in inclusive days 150–210 nearest to 180; equal-distance ties choose earlier day. Conflicting values on that selected day exclude the eye; identical duplicates do not create extra eyes. Gain >=5 is response=1. No valid outcome remains unknown, never response=0. Baseline is not shifted to a later visit to rescue invalid data. Inputs are only baseline VA and age. Source identifiers support grouping, not prediction.

The original deterministic patient-separated approximately 60/20/20 splits, settings and validation selection are preserved. The frozen reference checks enforce parity. Synthetic fixtures introduce no patient examples into fitting. The acuity-only comparator and additional diagnostics are not used to replace the reference model or optimize thresholds.

## Unresolved questions and limitations

- First recorded injection does not establish treatment-naive status or exact within-visit measurement timing.
- Five-letter gain and the 150–210-day window remain project choices pending ophthalmology approval. [Protocol T literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC6459102/) supports change thresholds, not automatic approval of this exact endpoint. Literature definitions of strong response can depend on baseline vision.
- Original >=3-injection selection and our follow-up eligibility select patients using events after baseline. No future predictors enter the model, but selection bias remains. The exported cohort cannot validate predictions for all treatment initiators.
- Routine VA should not be claimed to be standardized protocol-refraction BCVA.
- No CST, CMT, OCT, HbA1c, diabetes duration, diagnosis or drug column exists in this CSV. The proposed anatomical outcome cannot be trained here.
- Test results have already been inspected in V1/V2. This release is quality/reproducibility work; no new independent test or external validation is created.

No source facts or approvals are inferred from model accuracy. No clinical deployment, treatment switching or causal benefit claims are made.

# RetiSight CLIN-ML model card — 0.1.1 research release

- Model: original V1 logistic regression, complete preprocessing included. No refit on validation/test patients. Reference labels, partitions and test probabilities preserved and checked.
- Population: selected adult Moorfields DME anti-VEGF cohort; original >=3-injection eligibility, then valid six-month VA requirement. 1,675 eyes / 1,273 patients retained. Not an unselected initiation cohort or a Saudi/multinational validation cohort.
- Target: >=5-letter ETDRS gain from first recorded injection baseline to nearest valid day to 180 in days 150–210. Missing outcome unknown. Ophthalmology approval pending.
- Inputs: baseline ETDRS VA 0–100 and exact baseline age category; schema.json contains allowed values. Units and source cohort confirmed; above-100 visit scores documented errors. Prior treatment elsewhere and exact within-visit timing remain uncertain.
- Evidence: patient-separated internal evaluation, five-fold training CV, validation model selection, test patient-cluster intervals, acuity-only comparator, calibration and attrition diagnostics. No external validation. Existing test data already inspected in earlier versions.
- Intended use: reproducible research benchmark, educational demonstration and methods handoff. Not clinical treatment decisions, patient-facing prognosis, triage or causal benefit estimation.
- Limitations: post-baseline source selection, informative missing follow-up, limited baseline features, ceiling effects/regression to mean, routine VA measurement heterogeneity, grouped-eye uncertainty, no approved operating threshold, no fairness certification or independent prospective evaluation.
- Anatomy: no CST/OCT fields and no anatomical response model. MULTI-ML uses a distinct cohort and requires its own verified linkage/splits.
- Export: trusted clinical_model.joblib; exact input validation in inference.py. Pin dependencies. Source hash, software versions, generated audits, tests and parity checks included.
- Privacy: raw CSV not bundled; generated tables and reference artifacts contain pseudonymous identifiers and require controlled access.
- Clinical/deployment approval: false. Confirm endpoint, intended decision and acceptance criteria with clinical collaborators and validate using new patients before considering clinical use.

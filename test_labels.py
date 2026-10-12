"""Small synthetic examples test clinical selection rules, not model performance."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from data_preparation import build_eye_level_labels
from audits import calibration_summary


def visit(day, va, patient="p1", eye="r"):
    return dict(
        anon_id=patient,
        eye=eye,
        gender="f",
        ethnicity="White",
        baseline_age="(60,70]",
        baseline_va=60,
        va=va,
        follow_up_days=day,
        inj_num=1,
        inj_given="y",
    )


class LabelTests(unittest.TestCase):
    def label(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "example.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            return build_eye_level_labels(source)

    def test_window_boundaries_and_outside(self):
        for day, eligible in [(149, False), (150, True), (210, True), (211, False)]:
            with self.subTest(day=day):
                audit, _, _, _ = self.label([visit(0, 60), visit(day, 65)])
                self.assertEqual(audit.iloc[0].status == "eligible", eligible)

    def test_nearest_not_best_outcome(self):
        audit, _, _, _ = self.label([visit(0, 60), visit(160, 90), visit(179, 61)])
        self.assertEqual(audit.iloc[0].followup_day, 179)
        self.assertEqual(audit.iloc[0].response, 0)

    def test_tie_selects_earlier(self):
        audit, _, _, _ = self.label([visit(0, 60), visit(170, 65), visit(190, 80)])
        self.assertEqual(audit.iloc[0].followup_day, 170)

    def test_exact_threshold(self):
        for va, expected in [(64, 0), (65, 1), (66, 1)]:
            audit, _, _, _ = self.label([visit(0, 60), visit(180, va)])
            self.assertEqual(audit.iloc[0].response, expected)

    def test_missing_outcome_stays_unknown(self):
        audit, _, _, _ = self.label([visit(0, 60)])
        self.assertTrue(pd.isna(audit.iloc[0].response))

    def test_invalid_nearest_uses_valid_alternative(self):
        audit, blanks, cleaned, _ = self.label(
            [visit(0, 60), visit(180, 120), visit(190, 67)]
        )
        self.assertEqual(audit.iloc[0].followup_day, 190)
        self.assertTrue(pd.isna(cleaned.iloc[1].va))
        self.assertEqual(blanks["va"], 0)
        self.assertEqual(cleaned.va_invalid.sum(), 1)

    def test_invalid_baseline_never_replaced(self):
        rows = [visit(0, 120), visit(180, 70)]
        for row in rows:
            row["baseline_va"] = 120
        audit, _, _, _ = self.label(rows)
        self.assertEqual(audit.iloc[0].status, "invalid_or_inconsistent_baseline_va")

    def test_conflicting_selected_day_excluded(self):
        audit, _, _, _ = self.label([visit(0, 60), visit(180, 65), visit(180, 66)])
        self.assertEqual(audit.iloc[0].status, "conflicting_selected_day_va")

    def test_identical_followups_not_extra_eyes(self):
        audit, _, _, _ = self.label([visit(0, 60), visit(180, 65), visit(180, 65)])
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.iloc[0].response, 1)

    def test_duplicate_baseline_excluded(self):
        audit, _, _, _ = self.label([visit(0, 60), visit(0, 60), visit(180, 65)])
        self.assertEqual(audit.iloc[0].status, "ambiguous_or_missing_day0")

    def test_missing_identifier_rejected(self):
        with self.assertRaises(ValueError):
            self.label([visit(0, 60, patient=None)])

    def test_other_baseline_exclusions(self):
        for field, value, status in [
            ("inj_given", "n", "no_day0_injection"),
            ("baseline_age", "unknown", "invalid_age"),
            ("baseline_va", 59, "invalid_or_inconsistent_baseline_va"),
        ]:
            rows = [visit(0, 60), visit(180, 65)]
            rows[0][field] = value
            audit, _, _, _ = self.label(rows)
            self.assertEqual(audit.iloc[0].status, status)

    def test_calibration_with_perfect_probabilities(self):
        summary = calibration_summary([0, 1], [0, 1])
        self.assertEqual(summary["brier"], 0)
        self.assertEqual(summary["ece_10_equal_width_bins"], 0)


if __name__ == "__main__":
    unittest.main()

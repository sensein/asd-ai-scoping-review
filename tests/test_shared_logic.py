from __future__ import annotations

import importlib
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from codebook import TASK_TYPE_PATTERNS
from helper_functions_ import (
    evaluation_metric_categories,
    extract_accuracy_percent,
    parse_numeric_age_value,
    yes_no_nominal,
)
from reliability import krippendorff_alpha
from run_icr_pipeline import task_type_categories


class SharedLogicTests(unittest.TestCase):
    def test_month_age_is_converted_to_years(self) -> None:
        self.assertEqual(parse_numeric_age_value("30 months"), 2.5)
        self.assertTrue(math.isnan(parse_numeric_age_value("1800 months")))

    def test_keyword_boundaries_avoid_known_false_positives(self) -> None:
        for text in ("texture analysis", "context model", "proposed method", "purpose statement"):
            categories = task_type_categories(text)
            self.assertFalse(categories["motor_movement_task"], text)
            self.assertFalse(categories["language_speech_audio_task"], text)

    def test_shared_task_category_name(self) -> None:
        self.assertIn("neurophysiology_neuroimaging_task", TASK_TYPE_PATTERNS)
        self.assertNotIn("neuroimaging_physiology_task", TASK_TYPE_PATTERNS)
        self.assertEqual(task_type_categories("EEG resting-state")["neurophysiology_neuroimaging_task"], 1)

    def test_accuracy_extraction_is_labeled_and_capped(self) -> None:
        self.assertEqual(extract_accuracy_percent("Accuracy: 91%", "accuracy"), 91.0)
        self.assertEqual(extract_accuracy_percent("0.875", "accuracy"), 87.5)
        for performance in ("F1 0.91", "AUC 0.92", "model 3 achieved 88", "accuracy 105%"):
            self.assertTrue(math.isnan(extract_accuracy_percent(performance, "accuracy")), performance)

    def test_metric_and_yes_no_rules_are_shared(self) -> None:
        self.assertEqual(evaluation_metric_categories("accuracy and AUC")["auc_roc"], 1)
        self.assertEqual(yes_no_nominal("reported"), "yes")
        self.assertEqual(yes_no_nominal("not included"), "no")
        self.assertEqual(yes_no_nominal("maybe"), "unclear")

    def test_shared_alpha(self) -> None:
        alpha, reason = krippendorff_alpha([["a", "a"], ["a", "b"], ["b", "b"]], "nominal")
        self.assertEqual(reason, "")
        self.assertAlmostEqual(alpha, 0.4444444444444444)

    def test_rq_modules_are_import_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("ASD_REVIEW_OUTPUT_ROOT")
            os.environ["ASD_REVIEW_OUTPUT_ROOT"] = temporary
            try:
                for module_name in ("rq1_", "rq2_", "rq3_", "rq4_", "rq5_"):
                    sys.modules.pop(module_name, None)
                    module = importlib.import_module(module_name)
                    self.assertTrue(callable(module.main))
                self.assertEqual(list(Path(temporary).iterdir()), [])
            finally:
                if previous is None:
                    os.environ.pop("ASD_REVIEW_OUTPUT_ROOT", None)
                else:
                    os.environ["ASD_REVIEW_OUTPUT_ROOT"] = previous


if __name__ == "__main__":
    unittest.main()

"""Tests for the manuscript figure scripts in scripts/Figures/.

The first group needs no data. The last class compares the numbers hardcoded in
RQ3_Figure.py with the RQ output CSVs, and skips itself when those CSVs have not
been generated (for example in CI, where the private workbook is not available).
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "Figures"))

from setup_data_ import N_DATA_ROWS

FIGURE_MODULES = (
    "PRISMA_diagram",
    "RQ1_Figure2",
    "RQ2_Figure",
    "RQ3_Figure",
    "RQ4_Figure",
    "Studies_per_year_line_plot",
)


def load_figure_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def output_root() -> Path:
    return Path(os.environ.get("ASD_REVIEW_OUTPUT_ROOT", ROOT / "output")).expanduser()


class FigureImportTests(unittest.TestCase):
    def test_figure_modules_are_import_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("ASD_REVIEW_OUTPUT_ROOT")
            os.environ["ASD_REVIEW_OUTPUT_ROOT"] = temporary
            try:
                for name in FIGURE_MODULES:
                    module = load_figure_module(name)
                    self.assertTrue(callable(module.main), name)
                self.assertEqual(list(Path(temporary).iterdir()), [])
            finally:
                if previous is None:
                    os.environ.pop("ASD_REVIEW_OUTPUT_ROOT", None)
                else:
                    os.environ["ASD_REVIEW_OUTPUT_ROOT"] = previous


def prisma_numbers(lines: list[str]) -> list[int]:
    text = " ".join(lines)
    return [int(re.sub(r"[{},]", "", value)) for value in re.findall(r"n\\ =\\ ([\d{},]+)", text)]


class HardcodedCountTests(unittest.TestCase):
    """The PRISMA and RQ3 figures hardcode their counts; check they are coherent."""

    def test_prisma_counts_are_consistent(self) -> None:
        prisma = load_figure_module("PRISMA_diagram")
        first = prisma_numbers(prisma.phase_1)
        second = prisma_numbers(prisma.phase_2)
        # identified, deduplicated, keyword-filtered, title/abstract, full text, final
        self.assertEqual(len(first), 6)
        self.assertEqual(first[1:], sorted(first[1:], reverse=True))
        self.assertEqual(first[-2], first[-1])
        # identified, deduplicated, title/abstract, full text, manual removal, final
        self.assertEqual(len(second), 6)
        identified, deduplicated, title_abstract, full_text, removed, final = second
        self.assertGreaterEqual(identified, deduplicated)
        self.assertGreaterEqual(deduplicated, title_abstract)
        self.assertGreaterEqual(title_abstract, full_text)
        self.assertEqual(full_text - removed, final)
        self.assertEqual(first[-1] + final, N_DATA_ROWS)

    def test_study_totals_match_the_annotation_sheet_layout(self) -> None:
        for name in ("RQ3_Figure", "RQ4_Figure"):
            self.assertEqual(load_figure_module(name).TOTAL_STUDIES, N_DATA_ROWS, name)

    def test_rq3_percentages_are_rounded_once(self) -> None:
        rq3 = load_figure_module("RQ3_Figure")
        # 19/61 = 31.147% and 27/92 = 29.348%. Rounding the two-decimal CSV values
        # again gave 31.2% and 29.4%, which was the original bug.
        self.assertEqual(rq3.format_percent(19, 61), "31.1%")
        self.assertEqual(rq3.format_percent(27, 92), "29.3%")

    def test_rq3_counts_never_exceed_their_denominators(self) -> None:
        rq3 = load_figure_module("RQ3_Figure")
        for row in rq3.ROWS:
            self.assertLessEqual(row["count"], rq3.TOTAL_STUDIES)
            for _, numerator, denominator in row["tasks"] + row["tools"]:
                self.assertLessEqual(numerator, denominator)


def read_csv(folder: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(output_root() / folder / filename)


def count_for(frame: pd.DataFrame, key_column: str, key: str, value_column: str) -> int:
    matches = frame.loc[frame[key_column] == key, value_column]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {key!r} in column {key_column!r}, found {len(matches)}")
    return int(matches.iloc[0])


@unittest.skipUnless(
    (output_root() / "rq2_results" / "RQ2_task_type_summary.csv").exists()
    and (output_root() / "rq3_results" / "RQ3_gaze_motor_speech_summary.csv").exists(),
    "RQ2/RQ3 outputs not generated; run scripts/rq2_.py and scripts/rq3_.py first",
)
class Rq3FigureMatchesOutputsTests(unittest.TestCase):
    """RQ3_Figure.py hardcodes its counts, so catch drift from the RQ2/RQ3 outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rq3 = load_figure_module("RQ3_Figure")
        cls.rows = {row["name"].split(" /")[0].lower(): row for row in cls.rq3.ROWS}

    def lookup(self, modality: str, section: str, label: str) -> tuple[int, int]:
        return next((n, d) for text, n, d in self.rows[modality][section] if text == label)

    def test_modality_counts(self) -> None:
        summary = read_csv("rq3_results", "RQ3_gaze_motor_speech_summary.csv")
        yes = summary[summary["Category"] == "Yes"].set_index("Modality")["Count"]
        for modality in ("gaze", "motor", "speech"):
            self.assertEqual(self.rows[modality]["count"], int(yes[modality]), modality)

    def test_task_counts(self) -> None:
        broad = read_csv("rq2_results", "RQ2_task_type_summary.csv")
        sub = read_csv("rq2_results", "RQ2_task_type_subcategory_summary.csv")
        expected_broad = {
            ("gaze", "Gaze/visual-attention tasks"): "gaze_visual_attention_task",
            ("motor", "Motor/movement tasks"): "motor_movement_task",
            ("speech", "Language/speech/audio tasks"): "language_speech_audio_task",
        }
        for (modality, label), key in expected_broad.items():
            numerator, _ = self.lookup(modality, "tasks", label)
            self.assertEqual(numerator, count_for(broad, "Task Type Category", key, "Count"), label)
        expected_sub = {
            ("gaze", "Passive viewing"): "passive_visual_stimulus_viewing",
            ("gaze", "Active viewing"): "active_gaze_visual_orienting_search_task",
            ("gaze", "Joint-attention tasks"): "joint_attention_social_cue_response_task",
            ("motor", "Gait and posture tasks"): "motor_gait_posture_kinematic_task",
            ("motor", "Imitation tasks"): "imitation_task",
        }
        for (modality, label), key in expected_sub.items():
            numerator, _ = self.lookup(modality, "tasks", label)
            self.assertEqual(numerator, count_for(sub, "Task subcategory", key, "n"), label)

    def test_tool_counts_and_denominators(self) -> None:
        expected = {
            ("gaze", "Specific eye-tracking tool"): ("RQ2_eye_tracking_tool_summary.csv", "any_named_eye_tracking_tool"),
            ("gaze", "Tobii"): ("RQ2_eye_tracking_tool_summary.csv", "tobii"),
            ("gaze", "SMI"): ("RQ2_eye_tracking_tool_summary.csv", "smi"),
            ("motor", "Specific motor-based tool"): ("RQ2_motor_tool_summary.csv", "any_named_motor_tool"),
            ("motor", "Kinect"): ("RQ2_motor_tool_summary.csv", "kinect"),
            ("speech", "Codable recording tool"): ("RQ2_video_audio_tool_summary.csv", "any_named_video_audio_tool"),
            ("speech", "Camera or webcam"): ("RQ2_video_audio_tool_summary.csv", "camera_or_webcam"),
            ("speech", "Microphone/audio recorder"): ("RQ2_video_audio_tool_summary.csv", "microphone_or_audio_recorder"),
        }
        for (modality, label), (filename, key) in expected.items():
            frame = read_csv("rq2_results", filename)
            row = frame.loc[frame["Tool Category"] == key].iloc[0]
            numerator, denominator = self.lookup(modality, "tools", label)
            self.assertEqual(numerator, int(row["Count"]), label)
            self.assertEqual(denominator, int(row["Relevant Task Papers"]), label)

    def test_footer_counts(self) -> None:
        broad = read_csv("rq2_results", "RQ2_task_type_summary.csv")
        self.assertEqual(self.rq3.MULTIPLE_TASK_TYPES, count_for(broad, "Task Type Category", "multiple_task_types", "Count"))
        unreported = count_for(broad, "Task Type Category", "not_given", "Count") + count_for(
            broad, "Task Type Category", "unclear", "Count"
        )
        self.assertEqual(self.rq3.UNREPORTED_TASK_TYPE, unreported)


if __name__ == "__main__":
    unittest.main()

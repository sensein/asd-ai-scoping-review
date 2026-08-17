from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "PRISMA_pipeline"
sys.path.insert(0, str(PIPELINE))

from prisma_common import find_duplicate_keys, load_criteria, normalize_doi, parse_decisions, title_similarity


def load_stage(filename: str):
    path = PIPELINE / filename
    spec = importlib.util.spec_from_file_location("test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def valid_criteria() -> dict:
    return {
        "review_name": "Synthetic test",
        "date_range": {"start_year": 2024, "end_year": 2026},
        "term_groups": {
            "population": {"terms": ["autism"]},
            "method": {"terms": ["machine learning"]},
            "data_source": {"terms": ["behavior"]},
            "outcome": {"terms": ["diagnosis"]},
        },
    }


class PrismaTests(unittest.TestCase):
    def test_missing_and_placeholder_criteria_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                load_criteria(missing)
            placeholder = Path(temporary) / "placeholder.json"
            placeholder.write_text((PIPELINE / "review_criteria.example.json").read_text())
            with self.assertRaises(ValueError):
                load_criteria(placeholder)

    def test_valid_criteria_records_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "criteria.json"
            path.write_text(json.dumps(valid_criteria()))
            loaded = load_criteria(path)
            self.assertEqual(loaded["_criteria_path"], str(path.resolve()))

    def test_stage_03_and_06_defaults_remove_title_maybe(self) -> None:
        stage03 = load_stage("03_prepare_abstract_screening.py")
        stage06 = load_stage("06_screen_abstracts.py")
        self.assertEqual(parse_decisions(stage03.parse_args([]).title_decisions), {"Include"})
        self.assertEqual(parse_decisions(stage06.parse_args([]).title_decisions), {"Include"})
        frame = pd.DataFrame({"suggested_title_screening_decision": ["Include", "Maybe", "Exclude"]})
        self.assertEqual(len(stage03.filter_title_decisions(frame, {"Include"})), 1)
        self.assertEqual(len(stage06.select_records(frame, {"Include"})), 1)

    def test_deduplication_normalizes_doi_title_and_near_title(self) -> None:
        stage01 = load_stage("01_deduplicate_records.py")
        frame = pd.DataFrame(
            [
                {"record_id": "1", "title": "Autism detection with behavior", "doi": "https://doi.org/10.1/ABC", "link": ""},
                {"record_id": "2", "title": "Different title", "doi": "doi:10.1/abc.", "link": ""},
                {"record_id": "3", "title": "Autism detection with behavior!", "doi": "", "link": ""},
                {"record_id": "4", "title": "Unique study", "doi": "", "link": "https://example.org/a/"},
            ]
        )
        retained, _, duplicate_log = stage01.deduplicate(frame, 0.97, True)
        self.assertEqual(len(retained), 2)
        self.assertEqual(set(duplicate_log["duplicate_match_type"]), {"DOI", "Exact title"})
        self.assertEqual(title_similarity("A title!", "a_title"), 1.0)

    def test_identifier_placeholders_are_not_duplicate_keys(self) -> None:
        self.assertEqual(normalize_doi("??"), "")
        self.assertEqual(
            normalize_doi("https://doi-org.example.edu/10.1145/3698587.37014"),
            "10.1145/3698587.37014",
        )
        frame = pd.DataFrame(
            [
                {"record_id": "1", "title": "One", "doi": "??", "link": "??"},
                {"record_id": "2", "title": "Two", "doi": "??", "link": "??"},
                {"record_id": "3", "title": "Three", "doi": "10.1/x", "link": ""},
                {"record_id": "4", "title": "Four", "doi": "https://doi.org/10.1/X", "link": ""},
            ]
        )
        audit = find_duplicate_keys(frame)
        self.assertEqual(set(audit["duplicate_match_type"]), {"DOI"})
        self.assertEqual(set(audit["record_id"]), {"3", "4"})

    def test_environment_roots_apply_to_stage_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data-root"
            output_root = Path(temporary) / "output-root"
            env = dict(os.environ, ASD_REVIEW_DATA_ROOT=str(data_root), ASD_REVIEW_OUTPUT_ROOT=str(output_root))
            command = [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'scripts/PRISMA_pipeline'); import prisma_common as p; print(p.DATA_ROOT); print(p.OUTPUT_ROOT)",
            ]
            lines = subprocess.check_output(command, cwd=ROOT, env=env, text=True).splitlines()
            self.assertEqual(lines, [str(data_root), str(output_root)])


if __name__ == "__main__":
    unittest.main()

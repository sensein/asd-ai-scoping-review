#!/usr/bin/env python3
"""Add configurable title-screening suggestions to deduplicated PRISMA records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prisma_common import OUTPUT_ROOT, count_frame, load_criteria, read_table, screen_eligibility, write_workbook


DEFAULT_INPUT = OUTPUT_ROOT / "deduplication" / "deduplicated_records.xlsx"
DEFAULT_OUTPUT = OUTPUT_ROOT / "title_screening" / "title_screening_suggestions.xlsx"


def add_suggestions(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    df = df.copy().fillna("")
    if "title" not in df.columns:
        raise SystemExit("Input is missing a title column.")
    results = df["title"].apply(lambda title: screen_eligibility(title=title, body="", criteria=criteria, stage="title"))
    df["suggested_title_screening_decision"] = results.apply(lambda item: item["decision"])
    df["suggested_title_exclusion_reason"] = results.apply(lambda item: item["reason"])
    df["suggested_title_confidence"] = results.apply(lambda item: item["confidence"])
    df["suggested_title_notes"] = results.apply(lambda item: item["notes"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add title-screening suggestions using configurable criteria.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--criteria", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    criteria = load_criteria(args.criteria)
    df = read_table(args.input, args.sheet, preferred_sheet="Deduplicated_Retained")
    screened = add_suggestions(df, criteria)
    summary = pd.DataFrame(
        [
            {"metric": "Input retained records", "value": len(screened)},
            {"metric": "Suggested title Include", "value": int(screened["suggested_title_screening_decision"].eq("Include").sum())},
            {"metric": "Suggested title Maybe", "value": int(screened["suggested_title_screening_decision"].eq("Maybe").sum())},
            {"metric": "Suggested title Exclude", "value": int(screened["suggested_title_screening_decision"].eq("Exclude").sum())},
        ]
    )
    write_workbook(
        args.output,
        {
            "Summary": summary,
            "Title_Counts": count_frame(screened["suggested_title_screening_decision"], "decision"),
            "Title_Reasons": count_frame(screened["suggested_title_exclusion_reason"], "title_exclusion_reason", "No exclusion reason"),
            "Title_Suggestions": screened,
        },
    )
    print(f"Wrote title-screening suggestions workbook: {args.output}")
    print(screened["suggested_title_screening_decision"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

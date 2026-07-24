#!/usr/bin/env python3
"""Create the abstract-screening queue from title Include/Maybe records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prisma_common import ROOT, count_frame, load_criteria, read_table, screen_eligibility, write_workbook


DEFAULT_INPUT = ROOT / "output" / "title_screening" / "title_screening_suggestions.xlsx"
DEFAULT_TITLE_OUTPUT = ROOT / "output" / "abstract_screening" / "title_include_maybe_metadata.xlsx"
DEFAULT_ABSTRACT_OUTPUT = ROOT / "output" / "abstract_screening" / "abstract_screening_suggestions.xlsx"


def filter_title_decisions(df: pd.DataFrame, decisions: set[str]) -> pd.DataFrame:
    column = "suggested_title_screening_decision"
    if column not in df.columns:
        raise SystemExit(f"Input is missing {column}.")
    return df[df[column].isin(decisions)].copy()


def add_abstract_suggestions(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    df = df.copy().fillna("")
    for column in ["title", "abstract", "document_type", "language", "year_published"]:
        if column not in df.columns:
            df[column] = ""
    results = df.apply(
        lambda row: screen_eligibility(
            title=row.get("title", ""),
            body=row.get("abstract", ""),
            criteria=criteria,
            stage="abstract",
            document_type=row.get("document_type", ""),
            language=row.get("language", ""),
            year_value=row.get("year_published", ""),
        ),
        axis=1,
    )
    df["suggested_abstract_screening_decision"] = results.apply(lambda item: item["decision"])
    df["suggested_abstract_exclusion_reason"] = results.apply(lambda item: item["reason"])
    df["suggested_abstract_confidence"] = results.apply(lambda item: item["confidence"])
    df["suggested_abstract_notes"] = results.apply(lambda item: item["notes"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract title Include/Maybe records and add abstract-screening suggestions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--title-output", type=Path, default=DEFAULT_TITLE_OUTPUT)
    parser.add_argument("--abstract-output", type=Path, default=DEFAULT_ABSTRACT_OUTPUT)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--criteria", type=Path, default=None)
    parser.add_argument("--title-decisions", default="Include,Maybe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    decisions = {item.strip() for item in args.title_decisions.split(",") if item.strip()}
    criteria = load_criteria(args.criteria)
    df = read_table(args.input, args.sheet, preferred_sheet="Title_Suggestions")
    title_subset = filter_title_decisions(df, decisions)
    abstract_screened = add_abstract_suggestions(title_subset, criteria)
    abstract_include_maybe = abstract_screened[abstract_screened["suggested_abstract_screening_decision"].isin(["Include", "Maybe"])].copy()

    title_summary = pd.DataFrame(
        [
            {"metric": "Title records selected", "value": len(title_subset)},
            {"metric": "Records with abstracts", "value": int(title_subset.get("abstract", pd.Series([""] * len(title_subset))).astype(str).str.strip().ne("").sum())},
            {"metric": "Records missing abstracts", "value": int(title_subset.get("abstract", pd.Series([""] * len(title_subset))).astype(str).str.strip().eq("").sum())},
        ]
    )
    write_workbook(args.title_output, {"Summary": title_summary, "Title_Include_Maybe_Metadata": title_subset})

    abstract_summary = pd.DataFrame(
        [
            {"metric": "Records screened at abstract stage", "value": len(abstract_screened)},
            {"metric": "Abstract suggested Include", "value": int(abstract_screened["suggested_abstract_screening_decision"].eq("Include").sum())},
            {"metric": "Abstract suggested Maybe", "value": int(abstract_screened["suggested_abstract_screening_decision"].eq("Maybe").sum())},
            {"metric": "Abstract suggested Exclude", "value": int(abstract_screened["suggested_abstract_screening_decision"].eq("Exclude").sum())},
            {"metric": "Abstract Include + Maybe records", "value": len(abstract_include_maybe)},
        ]
    )
    write_workbook(
        args.abstract_output,
        {
            "Summary": abstract_summary,
            "Abstract_Counts": count_frame(abstract_screened["suggested_abstract_screening_decision"], "decision"),
            "Abstract_Reasons": count_frame(abstract_screened["suggested_abstract_exclusion_reason"], "abstract_exclusion_reason", "No exclusion reason"),
            "All_Abstract_Suggestions": abstract_screened,
            "Abstract_Include_Maybe": abstract_include_maybe,
        },
    )
    print(f"Wrote title queue: {args.title_output}")
    print(f"Wrote abstract suggestions: {args.abstract_output}")
    print(abstract_screened["suggested_abstract_screening_decision"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

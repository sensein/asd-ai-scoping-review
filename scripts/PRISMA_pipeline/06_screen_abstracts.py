#!/usr/bin/env python3
"""Run configurable abstract-screening suggestions on selected title-screened records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prisma_common import OUTPUT_ROOT, count_frame, load_criteria, parse_decisions, read_table, screen_eligibility, write_workbook


DEFAULT_INPUT = OUTPUT_ROOT / "abstract_finding" / "title_include_with_supplemental_metadata.xlsx"
DEFAULT_OUTPUT = OUTPUT_ROOT / "abstract_screening" / "title_include_abstract_screening_suggestions.xlsx"


def select_records(df: pd.DataFrame, title_decisions: set[str]) -> pd.DataFrame:
    if "suggested_title_screening_decision" in df.columns:
        return df[df["suggested_title_screening_decision"].isin(title_decisions)].copy()
    if "title_screening_decision" in df.columns:
        return df[df["title_screening_decision"].isin(title_decisions)].copy()
    return df.copy()


def add_suggestions(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run abstract-screening suggestions with configurable criteria.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet", default="Records_With_Abstracts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--criteria", type=Path, default=None)
    parser.add_argument(
        "--title-decisions",
        default="Include",
        help="Title decisions screened at abstract stage. Current protocol screens Include only.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    title_decisions = parse_decisions(args.title_decisions)
    criteria = load_criteria(args.criteria)
    source = read_table(args.input, args.sheet, preferred_sheet="Records_With_Abstracts")
    selected = select_records(source, title_decisions)
    screened = add_suggestions(selected, criteria)
    include_maybe = screened[screened["suggested_abstract_screening_decision"].isin(["Include", "Maybe"])].copy()
    missing = screened[screened.get("abstract", pd.Series([""] * len(screened))).astype(str).str.strip().eq("")].copy()
    excluded = screened[screened["suggested_abstract_screening_decision"].eq("Exclude")].copy()
    summary = pd.DataFrame(
        [
            {"metric": "Title decisions screened", "value": ",".join(sorted(title_decisions))},
            {
                "metric": "Title Maybe records not screened",
                "value": int(source.get("suggested_title_screening_decision", pd.Series(dtype=str)).eq("Maybe").sum())
                if "Maybe" not in title_decisions
                else 0,
            },
            {"metric": "Records screened", "value": len(screened)},
            {"metric": "Records with abstracts", "value": len(screened) - len(missing)},
            {"metric": "Records missing abstracts", "value": len(missing)},
            {"metric": "Abstract suggested Include", "value": int(screened["suggested_abstract_screening_decision"].eq("Include").sum())},
            {"metric": "Abstract suggested Maybe", "value": int(screened["suggested_abstract_screening_decision"].eq("Maybe").sum())},
            {"metric": "Abstract suggested Exclude", "value": len(excluded)},
            {"metric": "Abstract Include + Maybe records", "value": len(include_maybe)},
        ]
    )
    write_workbook(
        args.output,
        {
            "Summary": summary,
            "Abstract_Counts": count_frame(screened["suggested_abstract_screening_decision"], "decision"),
            "Abstract_Reasons": count_frame(screened["suggested_abstract_exclusion_reason"], "abstract_exclusion_reason", "No exclusion reason"),
            "Title_Include_Screening": screened,
            "Abstract_Include_Maybe": include_maybe,
            "Manual_Abstract_Lookup": missing,
            "Abstract_Excluded": excluded,
        },
        header_color="1D3557",
    )
    print(f"Records screened: {len(screened)}")
    print(screened["suggested_abstract_screening_decision"].value_counts(dropna=False).to_string())
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

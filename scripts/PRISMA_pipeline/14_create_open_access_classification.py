#!/usr/bin/env python3
"""Filter an open-access classification workbook to the final included records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prisma_common import ROOT, count_frame, normalize_header, read_table, write_workbook


DEFAULT_FINAL = ROOT / "output" / "final_included_studies" / "final_full_text_decisions_after_manual_removal.xlsx"
DEFAULT_SOURCE = ROOT / "data" / "manual" / "open_access_classification_source.xlsx"
DEFAULT_OUTPUT = ROOT / "output" / "final_included_studies" / "final_open_access_classification.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an open-access classification workbook to final included records.")
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--final-sheet", default="Final_Included_Studies")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-sheet", default="OA_Classification")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final = read_table(args.final, args.final_sheet, preferred_sheet="Final_Included_Studies")
    source = read_table(args.source, args.source_sheet, preferred_sheet="OA_Classification")
    final_ids = set(final["record_id"].astype(str).str.strip())
    normalized_columns = {normalize_header(col): col for col in source.columns}
    record_id_col = normalized_columns.get("record_id") or normalized_columns.get("record id")
    category_col = normalized_columns.get("open_access_classification") or normalized_columns.get("oa_classification")
    if not record_id_col:
        raise SystemExit("Open-access source is missing a record ID column.")
    detail = source[source[record_id_col].astype(str).str.strip().isin(final_ids)].copy()
    sheets = {
        "Summary": pd.DataFrame([{"metric": "Final included records with OA rows", "value": len(detail)}]),
        "OA_Classification": detail,
    }
    if category_col and category_col in detail.columns:
        sheets["Category_Counts"] = count_frame(detail[category_col], "category")
    write_workbook(args.output, sheets)
    print(f"Wrote {args.output}")
    print(f"Open-access rows: {len(detail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

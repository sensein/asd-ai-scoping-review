#!/usr/bin/env python3
"""Create a final metadata workbook for included studies."""

from __future__ import annotations

import argparse
from pathlib import Path


from prisma_common import OUTPUT_ROOT, read_table, write_workbook


DEFAULT_DECISIONS = OUTPUT_ROOT / "final_included_studies" / "final_full_text_decisions_after_manual_removal.xlsx"
DEFAULT_METADATA = OUTPUT_ROOT / "abstract_screening" / "title_include_abstract_screening_manual_pdf_updated.xlsx"
DEFAULT_OUTPUT = OUTPUT_ROOT / "final_included_studies" / "final_included_studies_metadata.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the final metadata workbook for included studies.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--decisions-sheet", default="Final_Included_Studies")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--metadata-sheet", default="Title_Include_Screening")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decisions = read_table(args.decisions, args.decisions_sheet, preferred_sheet="Final_Included_Studies")
    if "record_id" not in decisions.columns:
        raise SystemExit("Decision workbook is missing record_id.")
    final_ids = decisions["record_id"].astype(str).str.strip().tolist()
    metadata = read_table(args.metadata, args.metadata_sheet, preferred_sheet="Title_Include_Screening")
    if "record_id" not in metadata.columns:
        raise SystemExit("Metadata workbook is missing record_id.")
    metadata = metadata[metadata["record_id"].astype(str).str.strip().isin(final_ids)].copy()
    order = {record_id: index for index, record_id in enumerate(final_ids)}
    metadata["_order"] = metadata["record_id"].map(order)
    metadata = metadata.sort_values("_order")
    columns = ["record_id", "title", "link", "keywords", "journal", "source_database", "year_published", "authors", "abstract", "doi", "language", "document_type"]
    for column in columns:
        if column not in metadata.columns:
            metadata[column] = ""
    output = metadata[columns].drop(columns=["_order"], errors="ignore")
    write_workbook(args.output, {"Metadata": output})
    print(f"Wrote {args.output}")
    print(f"Final metadata rows: {len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

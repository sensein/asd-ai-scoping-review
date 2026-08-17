#!/usr/bin/env python3
"""Create final full-text decisions by combining retrieval and screening results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prisma_common import ROOT, clean, count_frame, read_table, write_workbook


DEFAULT_MANIFEST = ROOT / "output" / "pdf_retrieval" / "pdf_retrieval_manifest.xlsx"
DEFAULT_SCREENING = ROOT / "output" / "full_text_screening" / "full_text_screening_results_pdfs_found.xlsx"
DEFAULT_OUTPUT = ROOT / "output" / "full_text_screening" / "final_full_text_decisions.xlsx"


def combine_decisions(manifest: pd.DataFrame, screening: pd.DataFrame) -> pd.DataFrame:
    if "record_id" not in manifest.columns or "record_id" not in screening.columns:
        raise SystemExit("Both input workbooks must contain record_id.")
    screening_cols = ["record_id"] + [col for col in screening.columns if col != "record_id" and col not in manifest.columns]
    merged = manifest.merge(screening[screening_cols], on="record_id", how="left").fillna("")
    final_decisions: list[str] = []
    final_reasons: list[str] = []
    final_notes: list[str] = []
    stages: list[str] = []

    for _, row in merged.iterrows():
        retrieval_status = clean(row.get("pdf_retrieval_status", "")).lower()
        screened_decision = clean(row.get("full_text_decision", ""))
        pdf_resolved = clean(row.get("pdf_path_resolved", "")).lower() == "yes"
        extraction_status = clean(row.get("extraction_status", "")).lower()
        if retrieval_status == "not_found":
            final_decisions.append("Exclude")
            final_reasons.append("Full text/PDF unavailable")
            final_notes.append("No public or supplied PDF was available for screening.")
            stages.append("PDF not found")
        elif retrieval_status in {"downloaded", "already_existing", "found"} and not pdf_resolved:
            final_decisions.append("Exclude")
            final_reasons.append("Full text/PDF unavailable")
            final_notes.append("PDF was marked available, but no local PDF path was resolved.")
            stages.append("PDF path unresolved")
        elif retrieval_status in {"downloaded", "already_existing", "found"} and extraction_status and extraction_status != "extracted":
            final_decisions.append("Exclude")
            final_reasons.append("Full text could not be extracted")
            final_notes.append(clean(row.get("full_text_notes", "")) or "Local PDF existed, but text extraction failed or returned no usable text.")
            stages.append("PDF extraction failed")
        elif screened_decision:
            final_decisions.append(screened_decision)
            final_reasons.append(clean(row.get("full_text_exclusion_reason", "")))
            final_notes.append(clean(row.get("full_text_notes", "")))
            stages.append("Full-text screened")
        else:
            final_decisions.append("Maybe")
            final_reasons.append("Needs full-text screening")
            final_notes.append("No full-text screening decision was found.")
            stages.append("Missing screening decision")

    merged["final_full_text_decision"] = final_decisions
    merged["final_full_text_exclusion_reason"] = final_reasons
    merged["final_full_text_notes"] = final_notes
    merged["final_decision_source_stage"] = stages
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final full-text decisions for abstract Include/Maybe records.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sheet", default="PDF_Retrieval_Manifest")
    parser.add_argument("--screening", type=Path, default=DEFAULT_SCREENING)
    parser.add_argument("--screening-sheet", default="Full_Text_Screening")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_table(args.manifest, args.manifest_sheet, preferred_sheet="PDF_Retrieval_Manifest")
    screening = read_table(args.screening, args.screening_sheet, preferred_sheet="Full_Text_Screening")
    combined = combine_decisions(manifest, screening)
    include_maybe = combined["final_full_text_decision"].isin(["Include", "Maybe"])
    excluded = combined["final_full_text_decision"].eq("Exclude")
    write_workbook(
        args.output,
        {
            "Summary": pd.DataFrame(
                [
                    {"metric": "Total records", "value": len(combined)},
                    {"metric": "Final full-text Include", "value": int(combined["final_full_text_decision"].eq("Include").sum())},
                    {"metric": "Final full-text Maybe", "value": int(combined["final_full_text_decision"].eq("Maybe").sum())},
                    {"metric": "Final full-text Exclude", "value": int(excluded.sum())},
                    {"metric": "Final Include + Maybe records", "value": int(include_maybe.sum())},
                ]
            ),
            "Final_Decision_Counts": count_frame(combined["final_full_text_decision"], "final_full_text_decision"),
            "Final_Exclusion_Reasons": count_frame(combined["final_full_text_exclusion_reason"], "final_full_text_exclusion_reason", "No exclusion reason"),
            "Decision_Source_Counts": count_frame(combined["final_decision_source_stage"], "final_decision_source_stage"),
            "Final_All": combined,
            "Final_Include_Maybe": combined.loc[include_maybe],
            "Final_Excluded": combined.loc[excluded],
        },
        header_color="283618",
    )
    print(f"Total records: {len(combined)}")
    print(combined["final_full_text_decision"].value_counts(dropna=False).to_string())
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

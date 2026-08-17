#!/usr/bin/env python3
"""Update abstract-screening workbook with manually added full PDFs.

The script matches PDFs in a folder to the Manual_Abstract_Lookup rows from the
abstract-screening workbook. Matched rows are marked as manual full PDF added;
unmatched manual-lookup rows are marked as manually excluded/no PDF added from
manual screen. It writes a new workbook with updated counts and audit sheets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from prisma_common import OUTPUT_ROOT, normalize_title as shared_normalize_title, style_workbook as shared_style_workbook, title_similarity_components

import pandas as pd

DEFAULT_INPUT = OUTPUT_ROOT / "abstract_screening" / "title_include_abstract_screening_suggestions.xlsx"
DEFAULT_PDF_DIR = OUTPUT_ROOT / "pdfs"
DEFAULT_OUTPUT = OUTPUT_ROOT / "abstract_screening" / "title_include_abstract_screening_manual_pdf_updated.xlsx"


def temporary_output_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}.tmp{output.suffix}")
    return output.with_name(f"{output.name}.tmp")


def normalize_title(value: Any) -> str:
    return shared_normalize_title(value)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def match_score(pdf_name: str, title: str) -> tuple[float, float, float]:
    return title_similarity_components(Path(pdf_name).stem, title)


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")


def best_pdf_matches(manual_df: pd.DataFrame, pdf_dir: Path, min_score: float) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    used_records: set[str] = set()
    matches: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    unmatched_pdf_rows: list[dict[str, Any]] = []

    for pdf in pdfs:
        best: dict[str, Any] | None = None
        for _, row in manual_df.iterrows():
            record_id = str(row.get("record_id", "")).strip()
            score, seq, jac = match_score(pdf.name, row.get("title", ""))
            candidate = {
                "record_id": record_id,
                "title": row.get("title", ""),
                "pdf_filename": pdf.name,
                "pdf_path": str(pdf.resolve()),
                "match_score": score,
                "sequence_score": seq,
                "token_score": jac,
            }
            if best is None or score > best["match_score"]:
                best = candidate
        if best and best["match_score"] >= min_score and best["record_id"] not in used_records:
            used_records.add(best["record_id"])
            matches[best["record_id"]] = best
            audit_rows.append(best)
        else:
            unmatched_pdf_rows.append(
                {
                    "pdf_filename": pdf.name,
                    "pdf_path": str(pdf.resolve()),
                    "best_record_id": best.get("record_id", "") if best else "",
                    "best_title": best.get("title", "") if best else "",
                    "best_match_score": f"{best.get('match_score', 0):.3f}" if best else "0.000",
                    "reason": "No manual-lookup title met match threshold or record was already matched.",
                }
            )

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit["match_score"] = audit["match_score"].map(lambda value: f"{float(value):.3f}")
        audit["sequence_score"] = audit["sequence_score"].map(lambda value: f"{float(value):.3f}")
        audit["token_score"] = audit["token_score"].map(lambda value: f"{float(value):.3f}")
    return matches, audit, pd.DataFrame(unmatched_pdf_rows)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy().fillna("")
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def apply_manual_update(screened: pd.DataFrame, manual: pd.DataFrame, matches: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra_cols = [
        "manual_pdf_status",
        "manual_pdf_filename",
        "manual_pdf_path",
        "manual_pdf_match_score",
        "manual_update_notes",
        "final_abstract_screening_decision",
        "final_abstract_exclusion_reason",
        "final_abstract_notes",
    ]
    screened = ensure_columns(screened, extra_cols)
    manual = ensure_columns(manual, extra_cols)

    manual_record_ids = set(str(v).strip() for v in manual.get("record_id", []))
    matched_ids = set(matches)

    for idx, row in screened.iterrows():
        record_id = str(row.get("record_id", "")).strip()
        suggested_decision = str(row.get("suggested_abstract_screening_decision", "")).strip()
        suggested_reason = str(row.get("suggested_abstract_exclusion_reason", "")).strip()
        suggested_notes = str(row.get("suggested_abstract_notes", "")).strip()
        screened.at[idx, "final_abstract_screening_decision"] = suggested_decision
        screened.at[idx, "final_abstract_exclusion_reason"] = suggested_reason
        screened.at[idx, "final_abstract_notes"] = suggested_notes

        if record_id in manual_record_ids and record_id in matched_ids:
            match = matches[record_id]
            screened.at[idx, "manual_pdf_status"] = "Manual PDF added"
            screened.at[idx, "manual_pdf_filename"] = match["pdf_filename"]
            screened.at[idx, "manual_pdf_path"] = match["pdf_path"]
            screened.at[idx, "manual_pdf_match_score"] = f"{float(match['match_score']):.3f}"
            screened.at[idx, "manual_update_notes"] = "Full PDF manually added; advance to full-text screening."
            screened.at[idx, "final_abstract_screening_decision"] = "Maybe"
            screened.at[idx, "final_abstract_exclusion_reason"] = ""
            screened.at[idx, "final_abstract_notes"] = "Missing abstract, but full PDF was manually added; advance to full-text screening."
        elif record_id in manual_record_ids:
            screened.at[idx, "manual_pdf_status"] = "Manual excluded / no PDF added"
            screened.at[idx, "manual_update_notes"] = "Manual lookup completed; no full PDF added for this missing-abstract paper."
            screened.at[idx, "final_abstract_screening_decision"] = "Exclude"
            screened.at[idx, "final_abstract_exclusion_reason"] = "Manual lookup: abstract unavailable and no full PDF added"
            screened.at[idx, "final_abstract_notes"] = "Manually excluded from abstract-stage advancement because no abstract/full PDF was added."
        else:
            screened.at[idx, "manual_pdf_status"] = "Not manual lookup row"

    manual_updated = screened[screened["record_id"].isin(manual_record_ids)].copy()
    return screened, manual_updated


def style_workbook(path: Path) -> None:
    shared_style_workbook(path)

def write_output(output: Path, screened: pd.DataFrame, manual_updated: pd.DataFrame, match_audit: pd.DataFrame, unmatched_pdfs: pd.DataFrame, source_name: str, pdf_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    final_counts = screened["final_abstract_screening_decision"].value_counts(dropna=False).rename_axis("decision").reset_index(name="count")
    final_reasons = screened["final_abstract_exclusion_reason"].replace("", "No exclusion reason").value_counts(dropna=False).rename_axis("abstract_exclusion_reason").reset_index(name="count")
    manual_counts = manual_updated["manual_pdf_status"].value_counts(dropna=False).rename_axis("manual_pdf_status").reset_index(name="count")
    advance_mask = screened["final_abstract_screening_decision"].isin(["Include", "Maybe"])
    exclude_mask = screened["final_abstract_screening_decision"].eq("Exclude")
    summary = pd.DataFrame(
        [
            {"metric": "Source workbook", "value": source_name},
            {"metric": "PDF folder", "value": str(pdf_dir.resolve())},
            {"metric": "Title Include records screened", "value": len(screened)},
            {"metric": "Manual lookup rows before update", "value": len(manual_updated)},
            {"metric": "Manually added full PDFs", "value": int((manual_updated["manual_pdf_status"] == "Manual PDF added").sum())},
            {"metric": "Manually excluded / no PDF added", "value": int((manual_updated["manual_pdf_status"] == "Manual excluded / no PDF added").sum())},
            {"metric": "Final abstract Include", "value": int((screened["final_abstract_screening_decision"] == "Include").sum())},
            {"metric": "Final abstract Maybe", "value": int((screened["final_abstract_screening_decision"] == "Maybe").sum())},
            {"metric": "Final abstract Exclude", "value": int(exclude_mask.sum())},
            {"metric": "Final Include + Maybe advancing", "value": int(advance_mask.sum())},
            {"metric": "PDFs in folder", "value": len(list(pdf_dir.glob("*.pdf")))},
            {"metric": "Unmatched PDFs", "value": len(unmatched_pdfs)},
        ]
    )
    temp_output = temporary_output_path(output)
    with pd.ExcelWriter(temp_output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        final_counts.to_excel(writer, sheet_name="Abstract_Counts", index=False)
        final_reasons.to_excel(writer, sheet_name="Abstract_Reasons", index=False)
        manual_counts.to_excel(writer, sheet_name="Manual_Counts", index=False)
        screened.to_excel(writer, sheet_name="Title_Include_Screening", index=False)
        screened.loc[advance_mask].to_excel(writer, sheet_name="Abstract_Include_Maybe", index=False)
        screened.loc[exclude_mask].to_excel(writer, sheet_name="Abstract_Excluded", index=False)
        manual_updated.to_excel(writer, sheet_name="Manual_PDF_Update", index=False)
        manual_updated[manual_updated["manual_pdf_status"].eq("Manual PDF added")].to_excel(writer, sheet_name="Manual_PDF_Added", index=False)
        manual_updated[manual_updated["manual_pdf_status"].eq("Manual excluded / no PDF added")].to_excel(writer, sheet_name="Manual_Excluded", index=False)
        match_audit.to_excel(writer, sheet_name="PDF_Match_Audit", index=False)
        unmatched_pdfs.to_excel(writer, sheet_name="Unmatched_PDFs", index=False)
    style_workbook(temp_output)
    temp_output.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update abstract-screening counts using manually added PDFs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-match-score", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input workbook not found: {args.input}")
    if not args.pdf_dir.exists():
        raise SystemExit(f"PDF folder not found: {args.pdf_dir}")
    screened = read_sheet(args.input, "Title_Include_Screening")
    # Use the main screening sheet as the source of truth for manual lookup rows.
    # Some exported Manual_Abstract_Lookup sheets can have shifted long-text cells.
    manual = screened[screened.get("abstract", pd.Series([""] * len(screened))).astype(str).str.strip().eq("")].copy()
    matches, match_audit, unmatched_pdfs = best_pdf_matches(manual, args.pdf_dir, args.min_match_score)
    updated, manual_updated = apply_manual_update(screened, manual, matches)
    write_output(args.output, updated, manual_updated, match_audit, unmatched_pdfs, args.input.name, args.pdf_dir)
    print(f"Title Include records: {len(updated)}")
    print(f"Manual lookup rows: {len(manual_updated)}")
    print(f"Manually added PDFs: {(manual_updated['manual_pdf_status'] == 'Manual PDF added').sum()}")
    print(f"Manually excluded/no PDF added: {(manual_updated['manual_pdf_status'] == 'Manual excluded / no PDF added').sum()}")
    print("Final abstract decisions:")
    print(updated["final_abstract_screening_decision"].value_counts(dropna=False).to_string())
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

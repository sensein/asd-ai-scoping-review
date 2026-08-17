#!/usr/bin/env python3
"""Resolve PDFs, extract text, and screen full texts with configurable criteria."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from prisma_common import (
    ROOT,
    apply_year_overrides,
    clean,
    count_frame,
    load_criteria,
    normalize_title,
    read_table,
    screen_eligibility,
    temporary_output_path,
    write_workbook,
)


DEFAULT_MANIFEST = ROOT / "output" / "pdf_retrieval" / "pdf_retrieval_manifest.xlsx"
DEFAULT_PDF_DIR = ROOT / "output" / "pdfs"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "full_text_screening"
DEFAULT_PDF_FOUND_OUTPUT = DEFAULT_OUTPUT_DIR / "pdfs_found_records_for_full_text_screening.xlsx"
DEFAULT_SCREENING_OUTPUT = DEFAULT_OUTPUT_DIR / "full_text_screening_results_pdfs_found.xlsx"
DEFAULT_EXTRACTOR = Path(__file__).resolve().parent / "09_extract_pdf_texts_pdfjs.mjs"


def title_similarity(left: object, right: object) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    seq = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    jac = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0
    return max(seq, jac)


def resolve_pdf_paths(records: pd.DataFrame, pdf_dir: Path, min_score: float) -> pd.DataFrame:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    rows = []
    for _, original in records.iterrows():
        row = original.copy()
        path_value = clean(row.get("pdf_path", ""))
        resolved_path = ""
        method = ""
        score_text = ""
        if path_value and Path(path_value).exists():
            resolved_path = str(Path(path_value).resolve())
            method = "manifest_pdf_path"
            score_text = "1.000"
        else:
            best_path = None
            best_score = 0.0
            for pdf in pdfs:
                score = title_similarity(pdf.stem, row.get("title", ""))
                if score > best_score:
                    best_score = score
                    best_path = pdf
            if best_path and best_score >= min_score:
                resolved_path = str(best_path.resolve())
                method = "title_filename_match"
                score_text = f"{best_score:.3f}"
            else:
                method = "unresolved"
                score_text = f"{best_score:.3f}"
        row["resolved_pdf_path"] = resolved_path
        row["pdf_path_resolved"] = "yes" if resolved_path else "no"
        row["pdf_path_resolution_method"] = method
        row["pdf_path_match_score"] = score_text
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def read_found_manifest(path: Path, sheet: str, pdf_dir: Path, min_score: float) -> pd.DataFrame:
    df = read_table(path, sheet, preferred_sheet="PDF_Retrieval_Manifest")
    if "pdf_retrieval_status" not in df.columns:
        raise SystemExit("Manifest is missing pdf_retrieval_status.")
    status = df["pdf_retrieval_status"].astype(str).str.strip().str.lower()
    found = df[status.isin(["downloaded", "already_existing", "found"])].copy()
    return resolve_pdf_paths(found, pdf_dir, min_score)


def write_json_records(records: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = records[["record_id", "title", "resolved_pdf_path"]].to_dict(orient="records")
    temp_output = temporary_output_path(output)
    temp_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_output.replace(output)


def run_extractor(args: argparse.Namespace, records_json: Path, extraction_jsonl: Path, text_dir: Path) -> None:
    node = args.node or shutil.which("node") or "node"
    command = [
        node,
        str(args.extractor),
        "--input-json",
        str(records_json),
        "--output-jsonl",
        str(extraction_jsonl),
        "--text-dir",
        str(text_dir),
        "--max-pages",
        str(args.max_pages),
    ]
    subprocess.run(command, check=True)


def read_extraction_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows).fillna("")


def screen_full_text_rows(records: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    rows = []
    for _, row in records.iterrows():
        text_path_value = clean(row.get("text_path", ""))
        text_path = Path(text_path_value) if text_path_value else None
        text = text_path.read_text(encoding="utf-8", errors="ignore") if text_path and text_path.exists() else ""
        combined = row.to_dict()
        if clean(row.get("pdf_path_resolved", "")) != "yes":
            screening = {
                "decision": "Maybe",
                "reason": "PDF path unresolved",
                "confidence": "low",
                "notes": "Manifest marked PDF as found, but no local PDF path matched confidently.",
            }
        elif clean(row.get("extraction_status", "")) != "extracted":
            screening = {
                "decision": "Maybe",
                "reason": "Full text could not be extracted",
                "confidence": "low",
                "notes": clean(row.get("extraction_error", "")) or "No extractable full text.",
            }
        else:
            screening = screen_eligibility(
                title=row.get("title", ""),
                body=text,
                criteria=criteria,
                stage="full_text",
                document_type=row.get("document_type", ""),
                language=row.get("language", ""),
                year_value=row.get("year_published", ""),
            )
        combined["full_text_decision"] = screening["decision"]
        combined["full_text_exclusion_reason"] = screening["reason"]
        combined["full_text_confidence"] = screening["confidence"]
        combined["full_text_notes"] = screening["notes"]
        for key, value in screening.items():
            if key.startswith("criterion_") or key.startswith("evidence_") or key == "screened_year":
                combined[key] = value
        rows.append(combined)
    return pd.DataFrame(rows).fillna("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen found PDFs against configurable full-text eligibility criteria.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sheet", default="PDF_Retrieval_Manifest")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-found-output", type=Path, default=DEFAULT_PDF_FOUND_OUTPUT)
    parser.add_argument("--screening-output", type=Path, default=DEFAULT_SCREENING_OUTPUT)
    parser.add_argument("--extractor", type=Path, default=DEFAULT_EXTRACTOR)
    parser.add_argument("--node", default="")
    parser.add_argument("--criteria", type=Path, default=None)
    parser.add_argument("--min-pdf-title-score", type=float, default=0.88)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means extract all pages.")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.extractor.exists():
        print(f"Extractor script not found: {args.extractor}", file=sys.stderr)
        return 1
    criteria = apply_year_overrides(load_criteria(args.criteria), args.start_year, args.end_year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_dir = args.output_dir / "extracted_pdf_texts"
    records_json = args.output_dir / "pdf_records_for_extraction.json"
    extraction_jsonl = args.output_dir / "pdf_text_extraction_manifest.jsonl"

    found = read_found_manifest(args.manifest, args.manifest_sheet, args.pdf_dir, args.min_pdf_title_score)
    resolved = found["pdf_path_resolved"].eq("yes") if "pdf_path_resolved" in found.columns else pd.Series(dtype=bool)
    write_workbook(
        args.pdf_found_output,
        {
            "Summary": pd.DataFrame(
                [
                    {"metric": "PDF-available records from manifest", "value": len(found)},
                    {"metric": "Resolved to local PDF path", "value": int(resolved.sum())},
                    {"metric": "Unresolved local PDF path", "value": int((~resolved).sum()) if len(resolved) else 0},
                ]
            ),
            "PDF_Found_Records": found,
            "Resolved_PDFs": found.loc[resolved] if len(resolved) else found,
            "Unresolved_PDFs": found.loc[~resolved] if len(resolved) else found.head(0),
        },
        header_color="1D3557",
    )
    write_json_records(found, records_json)

    if not args.skip_extraction:
        run_extractor(args, records_json, extraction_jsonl, text_dir)

    extraction = read_extraction_jsonl(extraction_jsonl)
    if extraction.empty:
        print(f"No extraction manifest found or manifest is empty: {extraction_jsonl}", file=sys.stderr)
        return 1
    merged = found.merge(extraction, on=["record_id", "title"], how="left", suffixes=("", "_extracted")).fillna("")
    screened = screen_full_text_rows(merged, criteria)
    include_maybe = screened["full_text_decision"].isin(["Include", "Maybe"])
    write_workbook(
        args.screening_output,
        {
            "Summary": pd.DataFrame(
                [
                    {"metric": "PDF-available records screened", "value": len(screened)},
                    {"metric": "Resolved local PDFs", "value": int(screened["pdf_path_resolved"].eq("yes").sum())},
                    {"metric": "Text extracted", "value": int(screened["extraction_status"].eq("extracted").sum())},
                    {"metric": "Full-text suggested Include", "value": int(screened["full_text_decision"].eq("Include").sum())},
                    {"metric": "Full-text suggested Maybe", "value": int(screened["full_text_decision"].eq("Maybe").sum())},
                    {"metric": "Full-text suggested Exclude", "value": int(screened["full_text_decision"].eq("Exclude").sum())},
                ]
            ),
            "Decision_Counts": count_frame(screened["full_text_decision"], "full_text_decision"),
            "Exclusion_Reasons": count_frame(screened["full_text_exclusion_reason"], "full_text_exclusion_reason", "No exclusion reason"),
            "Full_Text_Screening": screened,
            "Full_Text_Include_Maybe": screened.loc[include_maybe],
            "Full_Text_Excluded": screened.loc[~include_maybe],
        },
        header_color="1D3557",
    )
    print(f"PDF-available records: {len(found)}")
    print(f"Text extracted: {screened['extraction_status'].eq('extracted').sum()}")
    print(screened["full_text_decision"].value_counts(dropna=False).to_string())
    print(f"Wrote full-text screening workbook: {args.screening_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

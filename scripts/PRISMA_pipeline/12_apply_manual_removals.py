#!/usr/bin/env python3
"""Apply manual post-full-text exclusions and create the final included set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prisma_common import DATA_ROOT, OUTPUT_ROOT, clean, count_frame, normalize_header, normalize_title, read_table, title_similarity, write_workbook


DEFAULT_INPUT = OUTPUT_ROOT / "full_text_screening" / "final_full_text_decisions.xlsx"
DEFAULT_MANUAL = DATA_ROOT / "manual" / "manual_final_removals.xlsx"
DEFAULT_OUTPUT = OUTPUT_ROOT / "final_included_studies" / "final_full_text_decisions_after_manual_removal.xlsx"


def best_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {normalize_header(col): col for col in columns}
    for alias in aliases:
        if normalize_header(alias) in normalized:
            return normalized[normalize_header(alias)]
    return None


def load_manual(path: Path, sheet: str | int | None) -> pd.DataFrame:
    manual = read_table(path, sheet)
    manual.columns = [normalize_header(col) for col in manual.columns]
    title_col = best_column(list(manual.columns), ["title", "article title", "paper title"])
    reason_col = best_column(list(manual.columns), ["exclusion_reason", "reason", "manual exclusion reason"])
    if not title_col:
        raise SystemExit("Manual-removals file must contain a title column.")
    if not reason_col:
        manual["exclusion_reason"] = "Manual post-full-text exclusion"
        reason_col = "exclusion_reason"
    return manual[[title_col, reason_col]].rename(columns={title_col: "title", reason_col: "exclusion_reason"}).fillna("")


def apply_manual_removals(all_df: pd.DataFrame, manual: pd.DataFrame, min_score: float) -> pd.DataFrame:
    all_df = all_df.copy().fillna("")
    title_to_reason = {
        normalize_title(row.get("title", "")): clean(row.get("exclusion_reason", "")) or "Manual post-full-text exclusion"
        for _, row in manual.iterrows()
        if clean(row.get("title", ""))
    }
    manual_titles = list(title_to_reason)
    all_df["manual_final_removal"] = "no"
    all_df["manual_final_exclusion_reason"] = ""
    all_df["manual_final_match_score"] = ""
    for idx, row in all_df.iterrows():
        title = normalize_title(row.get("title", ""))
        if not title or not manual_titles:
            continue
        best = max(manual_titles, key=lambda item: title_similarity(title, item))
        score = title_similarity(title, best)
        if score >= min_score:
            all_df.at[idx, "manual_final_removal"] = "yes"
            all_df.at[idx, "manual_final_exclusion_reason"] = title_to_reason[best]
            all_df.at[idx, "manual_final_match_score"] = f"{score:.3f}"
    all_df["final_after_manual_decision"] = all_df.get("final_full_text_decision", all_df.get("full_text_decision", ""))
    all_df["final_after_manual_exclusion_reason"] = all_df.get("final_full_text_exclusion_reason", all_df.get("full_text_exclusion_reason", ""))
    removed = all_df["manual_final_removal"].eq("yes")
    all_df.loc[removed, "final_after_manual_decision"] = "Exclude"
    all_df.loc[removed, "final_after_manual_exclusion_reason"] = all_df.loc[removed, "manual_final_exclusion_reason"]
    all_df.loc[removed, "final_decision_source_stage"] = "manual_post_full_text_removal"
    return all_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual post-full-text exclusions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-sheet", default="Final_All")
    parser.add_argument("--manual-removals", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--manual-sheet", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-title-match-score", type=float, default=0.97)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_df = read_table(args.input, args.input_sheet, preferred_sheet="Final_All")
    manual = load_manual(args.manual_removals, args.manual_sheet)
    updated = apply_manual_removals(all_df, manual, args.min_title_match_score)
    final_include = updated[updated["final_after_manual_decision"].isin(["Include", "Maybe"])].copy()
    final_excluded = updated[~updated.index.isin(final_include.index)].copy()
    removed = updated["manual_final_removal"].eq("yes")
    write_workbook(
        args.output,
        {
            "Summary": pd.DataFrame(
                [
                    {"metric": "Input records", "value": len(updated)},
                    {"metric": "Manual-removal rows", "value": len(manual)},
                    {"metric": "Manual removals applied", "value": int(removed.sum())},
                    {"metric": "Final included records", "value": len(final_include)},
                    {"metric": "Final excluded records", "value": len(final_excluded)},
                ]
            ),
            "Final_Counts": count_frame(updated["final_after_manual_decision"], "decision"),
            "Final_Exclusion_Reasons": count_frame(updated["final_after_manual_exclusion_reason"], "exclusion_reason", "No exclusion reason"),
            "Final_All": updated,
            "Final_Included_Studies": final_include,
            "Final_Excluded": final_excluded,
            "Manual_Removals_Applied": updated.loc[removed],
        },
    )
    print(f"Wrote {args.output}")
    print(f"Final included records: {len(final_include)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

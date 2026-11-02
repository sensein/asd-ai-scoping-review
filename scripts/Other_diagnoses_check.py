import os
from pathlib import Path

import pandas as pd


# ============================================================
# Settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ASD_REVIEW_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()

if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT

DATA_SHEET = "final_data"
N_RELEVANT_COLS = 77  # A:BY
N_DATA_ROWS = 172     # Excel rows 3-174


INVALID_VALUES = [
    "",
    "-",
    "--",
    "nan",
    "na",
    "n/a",
    "n.a",
    "n.a.",
    "n.d",
    "n.d.",
    "nd",
    "n/d",
    "nr",
    "n.r",
    "n.r.",
    "n/r",
    "not given",
    "not reported",
    "not specified",
    "not stated",
    "not mentioned",
    "not available",
    "not provided",
    "not applicable",
    "unknown",
    "unkown",
    "unclear",
    "no information",
    "not clear",
    "not explicitly stated",
    "none",
    "none specified",
    "not givven",
]

INVALID_VALUES_SET = {value.lower().strip() for value in INVALID_VALUES}


# ============================================================
# Helpers
# ============================================================

def locate_final_annotation_sheet(data_root: Path = DATA_ROOT) -> Path:
    preferred = data_root / "final_annotation_sheet_.xlsx"

    if preferred.is_file():
        return preferred

    candidates = sorted(
        path
        for path in data_root.glob("final_annotation_sheet*.xlsx")
        if not path.name.startswith("~$")
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No final_annotation_sheet*.xlsx workbook found under {data_root}"
        )

    raise RuntimeError(
        f"Multiple final annotation workbooks found; specify one unambiguously: {candidates}"
    )


def flatten_columns(columns):
    flattened = []

    for col in columns:
        if isinstance(col, tuple):
            parts = [
                str(x).strip()
                for x in col
                if pd.notna(x)
                and not str(x).startswith("Unnamed")
                and str(x).strip() != ""
            ]
            flattened.append(" | ".join(parts) if parts else "Unnamed")
        else:
            flattened.append(str(col).strip())

    return flattened


def clean_invalid_values(df_part):
    return df_part.map(
        lambda x: pd.NA
        if pd.isna(x) or str(x).strip().lower() in INVALID_VALUES_SET
        else x
    )


def read_a_to_by_with_two_header_rows(excel_path: Path):
    raw = pd.read_excel(
        excel_path,
        sheet_name=DATA_SHEET,
        header=None,
        usecols="A:BY",
        nrows=N_DATA_ROWS + 2,
    )

    header_rows = raw.iloc[0:2].ffill(axis=1)

    columns = []
    seen = {}

    for top, bottom in zip(header_rows.iloc[0].tolist(), header_rows.iloc[1].tolist()):
        key = (top, bottom)
        count = seen.get(key, 0)
        seen[key] = count + 1

        if count:
            bottom = f"{bottom}.{count}"

        columns.append((top, bottom))

    df = raw.iloc[2:].copy()
    df.columns = pd.MultiIndex.from_tuples(columns)

    return df


# ============================================================
# Main
# ============================================================

def main():
    excel_path = locate_final_annotation_sheet()

    df = read_a_to_by_with_two_header_rows(excel_path)
    df.columns = flatten_columns(df.columns)

    df_subset = df.iloc[0:N_DATA_ROWS, 0:N_RELEVANT_COLS].reset_index(drop=True)

    # Column A = paper title
    col_paper_title = df_subset.iloc[:, 0]

    # Other diagnoses participant block = columns AE:AM by Excel position
    # In pandas 0-indexing from A:BY, this is iloc[:, 30:39].
    cols_subset_other = df_subset.iloc[:, 30:39]
    cols_clean_other = clean_invalid_values(cols_subset_other)

    # Mark "yes" for other diagnoses if any value in the Other block is valid/non-empty.
    valid_other = ~cols_clean_other.isna().all(axis=1)

    output = pd.DataFrame(
        {
            "index_number": df_subset.index[valid_other] + 1,
            "title": col_paper_title[valid_other].astype(str).str.strip().values,
        }
    )

    print(output.to_string(index=False))

    output_path = DATA_ROOT / "other_diagnoses_yes_titles.csv"
    output.to_csv(output_path, index=False)

    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
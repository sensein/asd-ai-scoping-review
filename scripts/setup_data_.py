import os
from pathlib import Path

import pandas as pd

# ============================================================
# Shared settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RELATIVE_PATH = "data/final_annotation_sheet_.xlsx"
DATA_ROOT = Path(os.environ.get("ASD_REVIEW_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
EXCEL_PATH = DATA_ROOT / "final_annotation_sheet_.xlsx"
DATA_SHEET = "final_data"

# Excel rows:
# Row 1 and Row 2 = headers
# Rows 3-174 = data
# Excel columns A to BY = relevant columns
#
# In pandas:
# rows 3-174 are read as data automatically when header=[0, 1]
# A:BY = iloc[:, 0:77]
# because BY is the 77th Excel column, and pandas slicing excludes the endpoint.

N_RELEVANT_COLS = 77  # A:BY
N_DATA_ROWS = 172    # rows 3-174 inclusive

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
INVALID_VALUES = [value.lower().strip() for value in INVALID_VALUES]
INVALID_VALUES_SET = set(INVALID_VALUES)


# ============================================================
# Helper functions
# ============================================================

def flatten_columns(columns):
    """
    Flattens two-row Excel headers into one readable column name.
    Example:
    ('Study info', 'Title') -> 'Study info | Title'
    """
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


def clean_invalid_values(df_part, invalid_values=INVALID_VALUES_SET):
    """
    Converts common missing/not-given values to pd.NA.
    Applies safely across mixed text/numeric columns.
    """
    return df_part.map(
        lambda x: pd.NA
        if pd.isna(x) or str(x).strip().lower() in invalid_values
        else x
    )


# ============================================================
# Main annotation loader
# ============================================================

def read_a_to_by_with_two_header_rows():
    # pandas cannot combine usecols with a multi-index header directly,
    # so read only A:BY first and then rebuild the same two-row header.
    raw = pd.read_excel(
        EXCEL_PATH,
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


def load_annotation_data():
    # Read only Excel columns A:BY with the workbook's two header rows.
    df = read_a_to_by_with_two_header_rows()

    # Flatten two-row headers.
    df.columns = flatten_columns(df.columns)

    # Keep rows 3-174 and columns A-BY.
    # Since header=[0, 1], pandas data starts from Excel row 3.
    df_subset = df.iloc[0:N_DATA_ROWS, 0:N_RELEVANT_COLS].reset_index(drop=True)

    # ----------------- COLUMN BLOCKS ------------------------- #
    # Total paper validity is based on Column A only.
    col_paper_title = df_subset.iloc[:, 0]

    # Keep all A:BY columns as the total annotation/codebook block.
    cols_subset_Total_Paper = df_subset.iloc[:, 0:N_RELEVANT_COLS]

    # Participant blocks use the current A:BY worksheet indices.
    cols_subset_ASD = df_subset.iloc[:, 12:20]
    cols_subset_Neur = df_subset.iloc[:, 21:29]
    cols_subset_Other = df_subset.iloc[:, 30:39]

    # ---------------- CLEAN DATA CONSISTENTLY ---------------- #
    col_paper_title_clean = col_paper_title.apply(
        lambda x: pd.NA
        if pd.isna(x) or str(x).strip().lower() in INVALID_VALUES_SET
        else x
    )

    cols_clean_Total_Papers = clean_invalid_values(cols_subset_Total_Paper)
    cols_clean_ASD = clean_invalid_values(cols_subset_ASD)
    cols_clean_Neur = clean_invalid_values(cols_subset_Neur)
    cols_clean_Other = clean_invalid_values(cols_subset_Other)

    # ---------------- VALID PAPER MASKS ---------------- #
    empty_rows_Total_Paper = col_paper_title_clean.isna()
    valid_total = ~empty_rows_Total_Paper

    empty_rows_Neur = cols_clean_Neur.isna().all(axis=1)
    empty_rows_Other = cols_clean_Other.isna().all(axis=1)

    valid_ASD = valid_total.copy()  # all included papers include ASD participants
    valid_Neur = ~empty_rows_Neur
    valid_Other = ~empty_rows_Other

    # ---------------- OLD VARIABLE NAMES / ALIASES ---------------- #
    valid_mask_Total_Paper = valid_total
    valid_mask_ASD = valid_ASD
    valid_mask_Neur = valid_Neur
    valid_mask_Other = valid_Other

    valid_papers_Total = int(valid_total.sum())
    valid_papers_ASD = int(valid_ASD.sum())
    valid_papers_Neur = int(valid_Neur.sum())
    valid_papers_Other = int(valid_Other.sum())

    setup_results = {
        "df": df,
        "df_subset": df_subset,
        "col_paper_title": col_paper_title,
        "col_paper_title_clean": col_paper_title_clean,
        "cols_subset_Total_Paper": cols_subset_Total_Paper,
        "cols_subset_ASD": cols_subset_ASD,
        "cols_subset_Neur": cols_subset_Neur,
        "cols_subset_Other": cols_subset_Other,
        "cols_clean_Total_Papers": cols_clean_Total_Papers,
        "cols_clean_ASD": cols_clean_ASD,
        "cols_clean_Neur": cols_clean_Neur,
        "cols_clean_Other": cols_clean_Other,
        "empty_rows_Total_Paper": empty_rows_Total_Paper,
        "empty_rows_ASD": empty_rows_Total_Paper,
        "empty_rows_Neur": empty_rows_Neur,
        "empty_rows_Other": empty_rows_Other,
        "valid_total": valid_total,
        "valid_ASD": valid_ASD,
        "valid_Neur": valid_Neur,
        "valid_Other": valid_Other,
        "valid_mask_Total_Paper": valid_mask_Total_Paper,
        "valid_mask_ASD": valid_mask_ASD,
        "valid_mask_Neur": valid_Neur,
        "valid_mask_Other": valid_Other,
        "valid_papers_Total": valid_papers_Total,
        "valid_papers_ASD": valid_papers_ASD,
        "valid_papers_Neur": valid_papers_Neur,
        "valid_papers_Other": valid_papers_Other,
        "invalid_values": INVALID_VALUES,
        "input_file": DATA_RELATIVE_PATH,
        "excel_path": EXCEL_PATH,
        "data_sheet": DATA_SHEET,
    }

    return setup_results


if __name__ == "__main__":
    xls = pd.ExcelFile(EXCEL_PATH)

    print("Available sheet names:")
    print(xls.sheet_names)

    if DATA_SHEET in xls.sheet_names:
        setup = load_annotation_data()
        print("\nLoaded annotation data.")
        print("Input file:", DATA_RELATIVE_PATH)
        print("Total valid papers based on column A:", setup["valid_papers_Total"])
        print("Valid ASD rows:", setup["valid_papers_ASD"])
        print("Valid neurotypical rows:", setup["valid_papers_Neur"])
        print("Valid other-diagnosis rows:", setup["valid_papers_Other"])
    else:
        print(f"\nWarning: sheet '{DATA_SHEET}' not found.")

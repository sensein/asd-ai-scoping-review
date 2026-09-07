from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ============================================================
# 1. PATHS
# ============================================================

# This script is intended to be stored in scripts/Figures/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "Publication_year_line_plot.png"
PDF_PATH = OUTPUT_DIR / "Publication_year_line_plot.pdf"


def find_annotation_workbook(project_root: Path) -> Path:
    """Find the final annotation workbook anywhere in the project."""
    matches = sorted(
        path
        for path in project_root.rglob("*.xlsx")
        if "final_annotation_sheet" in path.name.lower()
        and not path.name.startswith("~$")
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find an Excel file whose name contains "
            "'final_annotation_sheet' under:\n"
            f"{project_root}\n\n"
            "Place the workbook somewhere inside the project, or replace "
            "WORKBOOK_PATH below with its exact path."
        )

    if len(matches) > 1:
        print("Multiple annotation workbooks were found; using:")
        print(matches[0])
        print("Other matches:")
        for path in matches[1:]:
            print(f"  - {path}")

    return matches[0]


WORKBOOK_PATH = find_annotation_workbook(PROJECT_ROOT)


# ============================================================
# 2. READ AND CLEAN COLUMN E
# ============================================================

# sheet_name=0 reads the first worksheet. Change this to a worksheet name
# such as sheet_name="Final annotations" if the data are on another sheet.
dataframe = pd.read_excel(WORKBOOK_PATH, sheet_name=0)

if dataframe.shape[1] < 5:
    raise ValueError(
        f"The worksheet has only {dataframe.shape[1]} columns; column E is missing."
    )

year_column_name = dataframe.columns[4]
raw_years = dataframe.iloc[:, 4]


def extract_year(value):
    """Return a four-digit year from numbers, dates, or text."""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, pd.Timestamp):
        return value.year

    # Handles numeric Excel values such as 2021 or 2021.0.
    if isinstance(value, (int, float)):
        numeric_year = int(value)
        return numeric_year if 1900 <= numeric_year <= 2100 else pd.NA

    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(match.group(0)) if match else pd.NA


years = raw_years.map(extract_year).dropna().astype(int)

if years.empty:
    raise ValueError(
        f"No valid four-digit publication years were found in column E "
        f"({year_column_name!r})."
    )

# Count papers by year and include zero-count years between the minimum and
# maximum so the line represents a continuous timeline.
year_counts = years.value_counts().sort_index()
complete_years = range(int(year_counts.index.min()), int(year_counts.index.max()) + 1)
year_counts = year_counts.reindex(complete_years, fill_value=0)

print(f"Workbook: {WORKBOOK_PATH}")
print(f"Column E: {year_column_name}")
print(f"Valid publication years: {len(years)}")
print(year_counts.to_string())


# ============================================================
# 3. BUILD THE LINE PLOT
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

fig, ax = plt.subplots(figsize=(10.5, 5.5))

ax.plot(
    year_counts.index,
    year_counts.values,
    color="#117DB3",
    linewidth=2.2,
    marker="o",
    markersize=6,
    markerfacecolor="#117DB3",
    markeredgecolor="white",
    markeredgewidth=0.8,
)

# Add the paper count above each point.
for year, count in year_counts.items():
    ax.annotate(
        str(count),
        (year, count),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax.set_xlabel("Publication year")
ax.set_ylabel("Number of studies")
ax.set_xticks(list(year_counts.index))
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(bottom=0)

ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.30)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(x=0.025)

fig.tight_layout()


# ============================================================
# 4. SAVE AND DISPLAY
# ============================================================

fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.05)
fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.05)

print(f"Saved PNG: {PNG_PATH}")
print(f"Saved PDF: {PDF_PATH}")

plt.show()
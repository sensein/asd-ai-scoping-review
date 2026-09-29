from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ============================================================
# 1. PATHS
# ============================================================

# This script is intended to be stored in scripts/Figures/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.environ.get("ASD_REVIEW_OUTPUT_ROOT", PROJECT_ROOT / "output")
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

YEAR_COUNTS_PATH = OUTPUT_ROOT / "rq5_results" / "RQ5_publication_year_exact_summary.csv"
OUTPUT_DIR = OUTPUT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "Publication_year_line_plot.png"
PDF_PATH = OUTPUT_DIR / "Publication_year_line_plot.pdf"


if not YEAR_COUNTS_PATH.is_file():
    raise FileNotFoundError(
        f"Required RQ5 output was not found: {YEAR_COUNTS_PATH}\n"
        "Run python3 scripts/rq5_.py before rendering the figure."
    )

dataframe = pd.read_csv(YEAR_COUNTS_PATH)
required_columns = {"Publication Year", "Count", "Total Valid Papers"}
missing_columns = required_columns - set(dataframe.columns)
if missing_columns:
    raise ValueError(
        f"Missing columns in {YEAR_COUNTS_PATH}: {sorted(missing_columns)}"
    )
if dataframe.empty:
    raise ValueError(f"No publication years were found in {YEAR_COUNTS_PATH}")

years = pd.to_numeric(dataframe["Publication Year"], errors="raise")
counts = pd.to_numeric(dataframe["Count"], errors="raise")
denominators = pd.to_numeric(dataframe["Total Valid Papers"], errors="raise")
if years.isna().any() or counts.isna().any() or denominators.isna().any():
    raise ValueError(f"Missing year, count, or denominator in {YEAR_COUNTS_PATH}")
if (years % 1 != 0).any() or (counts % 1 != 0).any() or (denominators % 1 != 0).any():
    raise ValueError(f"Non-integer year, count, or denominator in {YEAR_COUNTS_PATH}")
if years.duplicated().any() or (counts < 0).any() or denominators.nunique() != 1:
    raise ValueError(f"Invalid or duplicate RQ5 summary rows in {YEAR_COUNTS_PATH}")

total_valid_papers = int(denominators.iloc[0])
if total_valid_papers <= 0 or counts.sum() > total_valid_papers:
    raise ValueError(f"Year counts exceed the valid-study total in {YEAR_COUNTS_PATH}")

# Include zero-count years between the first and last observed publication year.
year_counts = pd.Series(counts.astype(int).to_numpy(), index=years.astype(int))
year_counts = year_counts.sort_index()
complete_years = range(int(year_counts.index.min()), int(year_counts.index.max()) + 1)
year_counts = year_counts.reindex(complete_years, fill_value=0)

print(f"RQ5 output: {YEAR_COUNTS_PATH}")
print(f"Valid publication years: {int(counts.sum())}/{total_valid_papers}")
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

plt.close(fig)

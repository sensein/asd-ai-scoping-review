from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. GENERAL SETTINGS
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

EXPECTED_TOTAL_STUDIES = 172

# Assumes this script is stored in scripts/Figures/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RQ1_OUTPUT_DIR = PROJECT_ROOT / "output" / "rq1_results"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE_PATH = (
    RQ1_OUTPUT_DIR
    / "RQ1_sample_size_outputs.csv"
)

TERMINOLOGY_PATH = (
    RQ1_OUTPUT_DIR
    / "RQ1_terminology_outputs.csv"
)

SIMPLE_COUNTS_PATH = (
    RQ1_OUTPUT_DIR
    / "RQ1_simple_count_outputs.csv"
)


# ============================================================
# 2. EXACT COLORS FROM THE ORIGINAL SCRIPTS
# ============================================================

PANEL_A_COLORS = [
    "#0072B2",
    "#009E73",
    "#E69F00",
]

PANEL_B_COLORS = [
    "#0072B2",
    "#009E73",
    "#E69F00",
]

PANEL_C_COLORS = [
    "#0072B2",
    "#E69F00",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def require_columns(dataframe, required_columns, source_path):
    """Confirm that a saved output contains all required columns."""

    missing_columns = set(required_columns) - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {source_path}: "
            f"{sorted(missing_columns)}"
        )


def extract_binary_result(dataframe, result_label):
    """Extract reported and not-reported study counts."""

    matching_rows = dataframe[
        dataframe["label"] == result_label
    ]

    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one row for '{result_label}', "
            f"but found {len(matching_rows)}."
        )

    row = matching_rows.iloc[0]

    reported = int(row["count"])
    denominator = int(row["denominator"])
    not_reported = denominator - reported

    if reported < 0 or reported > denominator:
        raise ValueError(
            f"Invalid count for '{result_label}': "
            f"{reported} out of {denominator}."
        )

    return {
        "reported": reported,
        "not_reported": not_reported,
        "denominator": denominator,
    }


def remove_upper_spines(axis):
    """Apply consistent formatting to bar charts."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# ============================================================
# 4. LOAD PANEL A DATA
# ============================================================

sample_summary = pd.read_csv(SAMPLE_SIZE_PATH)

require_columns(
    sample_summary,
    required_columns={"group", "valid_papers"},
    source_path=SAMPLE_SIZE_PATH,
)

sample_summary["group"] = sample_summary["group"].str.strip()
sample_summary = sample_summary.set_index("group")

panel_a_group_order = [
    "ASD",
    "NEUROTYPICALS",
    "OTHER DIAGNOSES",
]

panel_a_display_labels = [
    "ASD",
    "Neurotypical",
    "Other diagnoses",
]

study_counts = (
    sample_summary
    .loc[panel_a_group_order, "valid_papers"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 5. LOAD PANEL B DATA
# ============================================================

terminology_df = pd.read_csv(TERMINOLOGY_PATH)

require_columns(
    terminology_df,
    required_columns={
        "group",
        "category",
        "count",
        "percentage",
        "denominator",
    },
    source_path=TERMINOLOGY_PATH,
)

terminology_df["group"] = terminology_df["group"].str.strip()
terminology_df["category"] = terminology_df["category"].str.strip()

panel_b_definitions = [
    {
        "group": "ASD",
        "title": "B(i) ASD",
        "categories": [
            "Official diagnostic terminology",
            "Identity-first language",
            "Alternative / non-standard terminology",
        ],
    },
    {
        "group": "Neurotypical / control",
        "title": "B(ii) Neurotypical/control",
        "categories": [
            "Official neurotypical/control terminology",
            "ASD-specific contrast terminology",
            "Vague / non-specific control terminology",
        ],
    },
    {
        "group": "Other diagnoses",
        "title": "B(iii) Other diagnoses",
        "categories": [
            "Official / specific diagnostic terminology",
            "Umbrella / broad other-diagnosis terminology",
            "ASD-risk terminology",
        ],
    },
]


# ============================================================
# 6. LOAD PANEL C DATA
# ============================================================

simple_counts_df = pd.read_csv(SIMPLE_COUNTS_PATH)

require_columns(
    simple_counts_df,
    required_columns={
        "label",
        "count",
        "percentage",
        "denominator",
    },
    source_path=SIMPLE_COUNTS_PATH,
)

simple_counts_df["label"] = simple_counts_df["label"].str.strip()

additional_assessment_label = (
    "Additional ASD-related assessments reported"
)

comorbidity_label = (
    "Comorbidities reported for autistic participants"
)

additional_assessments = extract_binary_result(
    simple_counts_df,
    additional_assessment_label,
)

comorbidities = extract_binary_result(
    simple_counts_df,
    comorbidity_label,
)

panel_c_definitions = [
    {
        "title": "C(i) Adaptive measures of functioning",
        "result": additional_assessments,
    },
    {
        "title": "C(ii) Comorbidity",
        "result": comorbidities,
    },
]

for panel in panel_c_definitions:
    denominator = panel["result"]["denominator"]

    if denominator != EXPECTED_TOTAL_STUDIES:
        raise ValueError(
            f"{panel['title']} uses a denominator of "
            f"{denominator}, not {EXPECTED_TOTAL_STUDIES}."
        )


# ============================================================
# 7. CREATE FIGURE WITH AUTOMATIC LAYOUT MANAGEMENT
# ============================================================

fig = plt.figure(
    figsize=(16, 10.5),
    facecolor="white",
    layout="constrained",
)

# Fine-tune constrained-layout padding
fig.get_layout_engine().set(
    w_pad=0.03,
    h_pad=0.04,
    wspace=0.05,
    hspace=0.10,
)

outer_grid = fig.add_gridspec(
    nrows=3,
    ncols=1,
    height_ratios=[
        0.80,  # Panel A
        1.45,  # Panel B
        1.10,  # Panel C
    ],
)


# ============================================================
# 8. PANEL A: PARTICIPANT-GROUP COUNTS
# ============================================================

ax_a = fig.add_subplot(outer_grid[0])

bars_a = ax_a.bar(
    panel_a_display_labels,
    study_counts,
    color=PANEL_A_COLORS,
    edgecolor="black",
    linewidth=0.8,
    width=0.52,
)

ax_a.bar_label(
    bars_a,
    labels=[str(count) for count in study_counts],
    padding=2,
    fontsize=10,
    fontweight="bold",
)

ax_a.set_title(
    "A. Participant Groups Represented Across Included Studies",
    fontsize=13,
    fontweight="bold",
    pad=4,
)

ax_a.set_xlabel(
    "Participant group",
    labelpad=2,
)

ax_a.set_ylabel(
    "Number of studies",
    labelpad=3,
)

ax_a.set_ylim(
    0,
    max(study_counts) * 1.17,
)

ax_a.grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

ax_a.margins(x=0.16)

remove_upper_spines(ax_a)


# ============================================================
# 9. PANEL B: TERMINOLOGY
# ============================================================

panel_b_grid = outer_grid[1].subgridspec(
    nrows=1,
    ncols=3,
    wspace=0.08,
)

panel_b_axes = []

for index in range(3):
    if index == 0:
        axis = fig.add_subplot(panel_b_grid[0, index])
    else:
        axis = fig.add_subplot(
            panel_b_grid[0, index],
            sharey=panel_b_axes[0],
        )

    panel_b_axes.append(axis)


for ax, panel in zip(
    panel_b_axes,
    panel_b_definitions,
):
    group_data = terminology_df[
        terminology_df["group"] == panel["group"]
    ].copy()

    group_data = (
        group_data
        .set_index("category")
        .reindex(panel["categories"])
    )

    relevant_columns = [
        "count",
        "percentage",
        "denominator",
    ]

    if group_data[relevant_columns].isna().any().any():
        missing_categories = group_data[
            group_data["count"].isna()
        ].index.tolist()

        raise ValueError(
            f"Missing terminology categories for "
            f"{panel['group']}: {missing_categories}"
        )

    counts = group_data["count"].astype(int)
    percentages = group_data["percentage"].astype(float)
    denominator = int(group_data["denominator"].iloc[0])

    wrapped_category_labels = [
        "\n".join(
            textwrap.wrap(
                category,
                width=18,
            )
        )
        for category in panel["categories"]
    ]

    bars_b = ax.bar(
        wrapped_category_labels,
        percentages,
        color=PANEL_B_COLORS,
        edgecolor="black",
        linewidth=0.8,
        width=0.65,
    )

    value_labels = [
        f"{count} ({percentage:.1f}%)"
        for count, percentage in zip(
            counts,
            percentages,
        )
    ]

    ax.bar_label(
        bars_b,
        labels=value_labels,
        padding=2,
        fontsize=8,
        fontweight="bold",
    )

    ax.set_title(
        f"{panel['title']}\n(n = {denominator} studies)",
        fontsize=11,
        fontweight="bold",
        pad=4,
    )

    ax.set_ylim(0, 108)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.tick_params(
        axis="x",
        labelsize=7.5,
        pad=1,
    )

    ax.margins(x=0.04)

    remove_upper_spines(ax)


panel_b_axes[0].set_ylabel(
    "Studies using terminology (%)",
    labelpad=3,
)

for ax in panel_b_axes[1:]:
    ax.tick_params(
        axis="y",
        labelleft=False,
    )

    ax.spines["left"].set_visible(False)


# ============================================================
# 10. PANEL C: ADAPTIVE MEASURES AND COMORBIDITY
# ============================================================

panel_c_grid = outer_grid[2].subgridspec(
    nrows=1,
    ncols=2,
    wspace=0.08,
)

panel_c_axes = [
    fig.add_subplot(panel_c_grid[0, index])
    for index in range(2)
]

for ax, panel in zip(
    panel_c_axes,
    panel_c_definitions,
):
    result = panel["result"]

    reported = result["reported"]
    not_reported = result["not_reported"]
    denominator = result["denominator"]

    reported_percentage = (
        reported / denominator
    ) * 100

    not_reported_percentage = (
        not_reported / denominator
    ) * 100

    values = [
        reported,
        not_reported,
    ]

    wedges, _ = ax.pie(
        values,
        colors=PANEL_C_COLORS,
        startangle=90,
        counterclock=False,
        radius=0.78,
        wedgeprops={
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )

    legend_labels = [
        (
            f"Reported: n = {reported} "
            f"({reported_percentage:.1f}%)"
        ),
        (
            f"Not reported: n = {not_reported} "
            f"({not_reported_percentage:.1f}%)"
        ),
    ]

    ax.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.3,
    )

    ax.set_title(
        f"{panel['title']}\n(N = {denominator} studies)",
        fontsize=11,
        fontweight="bold",
        pad=3,
    )

    ax.axis("equal")


# ============================================================
# 11. SAVE COMPLETE FIGURE
# ============================================================

png_path = (
    FIGURE_OUTPUT_DIR
    / "Figure_2_complete.png"
)

pdf_path = (
    FIGURE_OUTPUT_DIR
    / "Figure_2_complete.pdf"
)

svg_path = (
    FIGURE_OUTPUT_DIR
    / "Figure_2_complete.svg"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
)

fig.savefig(
    svg_path,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
)

plt.show()

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")
print(f"Saved SVG: {svg_path}")
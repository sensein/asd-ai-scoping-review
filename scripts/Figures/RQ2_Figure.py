from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. PATHS AND STYLE
# ============================================================

# This script is intended to be saved in scripts/Figures/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "output" / "rq2_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "RQ2_Combined_Figure.png"
PDF_PATH = OUTPUT_DIR / "RQ2_Combined_Figure.pdf"

# Okabe-Ito colorblind-friendly colors plus neutral grays.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
GRAY = "#999999"
LIGHT_GRAY = "#E6E6E6"

PALETTE = [BLUE, GREEN, ORANGE, SKY, VERMILLION, PURPLE, YELLOW, GRAY]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
    }
)


# ============================================================
# 2. HELPERS
# ============================================================


def read_output(filename, required_columns):
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required RQ2 output was not found: {path}\n"
            "Run the RQ2 analysis script with SAVE_OUTPUTS = True first."
        )

    dataframe = pd.read_csv(path)
    missing = set(required_columns) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )
    return dataframe


def clean_category_column(dataframe, column):
    dataframe = dataframe.copy()
    dataframe[column] = dataframe[column].astype(str).str.strip()
    dataframe["Count"] = pd.to_numeric(dataframe["Count"], errors="raise").astype(int)
    dataframe["Percentage"] = pd.to_numeric(
        dataframe["Percentage"], errors="raise"
    ).astype(float)
    return dataframe


def ordered_summary(dataframe, category_column, categories):
    indexed = dataframe.set_index(category_column)
    missing = [category for category in categories if category not in indexed.index]
    if missing:
        raise ValueError(
            f"Missing categories in {category_column}: {missing}"
        )
    return indexed.loc[categories].reset_index()


def denominator_from(dataframe, denominator_column="Total Valid Papers"):
    denominators = pd.to_numeric(
        dataframe[denominator_column], errors="raise"
    ).astype(int).unique()
    if len(denominators) != 1:
        raise ValueError(
            f"Expected one denominator in {denominator_column}; "
            f"found {denominators.tolist()}."
        )
    return int(denominators[0])


def validate_partition(dataframe, denominator, label):
    total = int(dataframe["Count"].sum())
    if total != denominator:
        raise ValueError(
            f"{label} cannot be drawn as a pie chart: category counts sum "
            f"to {total}, but the stated denominator is {denominator}. "
            "Review overlapping or unclassified rows in the RQ2 outputs."
        )


def wrap_label(label, width=18):
    return "\n".join(textwrap.wrap(label, width=width))


def remove_upper_spines(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_partition_pie(
    axis,
    dataframe,
    category_column,
    category_labels,
    title,
    colors=None,
    use_category_total=False,
    total_unit="studies",
):
    values = dataframe["Count"].astype(int).tolist()
    study_denominator = denominator_from(dataframe)

    if use_category_total:
        # Use this only when the plotted unit is an instance rather than a
        # study. For data availability, one study contributed two datasets
        # with different access conditions, producing 173 access instances
        # across 172 studies.
        pie_denominator = int(sum(values))
    else:
        validate_partition(dataframe, study_denominator, title)
        pie_denominator = study_denominator

    colors = colors or PALETTE[: len(values)]

    wedges, _ = axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=0.86,
        center=(-0.34, 0),
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
    )

    legend_labels = [
        f"{label}: {count} ({(count / pie_denominator) * 100:.1f}%)"
        for label, count in zip(category_labels, values)
    ]

    axis.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.66, 0.5),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.45,
        labelspacing=0.45,
    )
    axis.set_title(
        f"{title}\n(n = {pie_denominator} {total_unit})",
        loc="left",
        pad=4,
    )
    axis.set_aspect("equal")
    axis.set_xlim(-1.35, 1.85)
    axis.set_ylim(-1.10, 1.10)


def draw_progress_ring(axis, percentage, count, title, explanation, color):
    percentage = max(0.0, min(float(percentage), 100.0))

    axis.pie(
        [percentage, 100.0 - percentage],
        colors=[color, LIGHT_GRAY],
        startangle=90,
        counterclock=False,
        radius=0.58,
        center=(0, 0.37),
        wedgeprops={"width": 0.18, "edgecolor": "white", "linewidth": 0.7},
    )
    axis.text(
        0,
        0.37,
        f"{percentage:.1f}%\n(n = {int(count)})",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        linespacing=1.05,
    )
    axis.text(
        0.5,
        0.30,
        wrap_label(title, 22),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=color,
        linespacing=1.05,
    )
    axis.text(
        0.5,
        0.12,
        wrap_label(explanation, 31),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#444444",
        linespacing=1.12,
    )
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.15)
    axis.axis("off")


# ============================================================
# 3. LOAD SAVED RQ2 OUTPUTS
# ============================================================

goals = clean_category_column(
    read_output(
        "RQ2_study_goal_summary.csv",
        {"Study Goal Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Study Goal Category",
)

findings = clean_category_column(
    read_output(
        "RQ2_main_finding_subcategory_summary.csv",
        {"Finding Subcategory", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Finding Subcategory",
)

settings = clean_category_column(
    read_output(
        "RQ2_study_setting_summary.csv",
        {"Study Setting Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Study Setting Category",
)

datasets = clean_category_column(
    read_output(
        "RQ2_dataset_type_summary.csv",
        {"Dataset Type Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Dataset Type Category",
)

time_frames = clean_category_column(
    read_output(
        "RQ2_time_frame_summary.csv",
        {"Time Frame Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Time Frame Category",
)

sensitive = clean_category_column(
    read_output(
        "RQ2_sensitive_data_summary.csv",
        {"Sensitive Data Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Sensitive Data Category",
)

protection = clean_category_column(
    read_output(
        "RQ2_sensitive_data_protection_summary.csv",
        {"Protection Measure Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Protection Measure Category",
)

code_availability = clean_category_column(
    read_output(
        "RQ2_code_availability_summary.csv",
        {"Code Availability Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Code Availability Category",
)

data_availability = clean_category_column(
    read_output(
        "RQ2_data_availability_summary.csv",
        {"Data Availability Category", "Count", "Total Valid Papers", "Percentage"},
    ),
    "Data Availability Category",
)


# ============================================================
# 4. CATEGORY ORDER AND DISPLAY TEXT
# ============================================================

goal_definitions = [
    ("prediction_of_outcome", "Prediction of outcome"),
    ("screening_detection", "Screening/detection"),
    ("severity_detection", "Severity detection"),
    ("classification", "Classification"),
    ("diagnosis", "Diagnosis"),
    ("identifying_symptoms_biomarkers", "Symptoms/biomarkers"),
    ("other", "Other"),
    ("unclear", "Unclear"),
]

finding_definitions = [
    (
        "high_model_performance_or_feasibility",
        "Model performance or feasibility",
        "Reported accuracy, AUC, feasibility, or related performance outcomes.",
    ),
    (
        "behavioral_feature_differences_between_groups",
        "Behavioral differences between groups",
        "Identified group differences in measured behavioral features.",
    ),
    (
        "specific_predictive_or_discriminative_features_identified",
        "Predictive or discriminative features",
        "Identified particular features that contributed to prediction or discrimination.",
    ),
    (
        "multimodal_or_feature_fusion_improved_performance",
        "Multimodal or feature fusion",
        "Reported findings involving combined modalities or feature sets.",
    ),
    (
        "model_algorithm_comparison_or_optimization",
        "Model comparison or optimization",
        "Compared algorithms or reported model and parameter optimization.",
    ),
    (
        "clinical_screening_or_diagnostic_support",
        "Clinical screening or diagnostic support",
        "Positioned the approach as supporting screening or clinical assessment.",
    ),
    (
        "severity_symptom_trait_or_score_estimation",
        "Severity, symptoms, traits, or scores",
        "Estimated symptom severity, traits, or clinical assessment scores.",
    ),
    (
        "task_or_protocol_specific_effect",
        "Task- or protocol-specific effects",
        "Reported an effect associated with the task, protocol, or assessment setup.",
    ),
    (
        "negative_cautious_or_generalizability_limitation",
        "Cautious or limited findings",
        "Reported negative findings, limitations, or concerns about generalizability.",
    ),
    (
        "human_clinician_or_rater_comparison",
        "Human, clinician, or rater comparison",
        "Compared computational results with human, clinician, or rater judgments.",
    ),
    (
        "gaze_visual_attention_finding",
        "Gaze or visual attention",
        "Reported findings involving gaze, eye movement, or visual attention.",
    ),
    (
        "motor_movement_kinematic_finding",
        "Motor, movement, or kinematic findings",
        "Reported findings involving movement, posture, gait, pose, or kinematics.",
    ),
    (
        "speech_language_acoustic_finding",
        "Speech, language, or acoustic findings",
        "Reported findings involving speech, language, voice, or acoustic features.",
    ),
    (
        "facial_expression_or_image_finding",
        "Facial expression or image findings",
        "Reported findings involving faces, facial expressions, or facial images.",
    ),
    (
        "questionnaire_or_clinical_score_finding",
        "Questionnaire or clinical-score findings",
        "Reported findings involving questionnaires, records, or clinical scores.",
    ),
    (
        "neurophysiology_or_biosignal_finding",
        "Neurophysiology or biosignal findings",
        "Reported findings involving neurophysiological or other biosignal data.",
    ),
    (
        "social_interaction_or_play_finding",
        "Social interaction or play",
        "Reported findings involving social interaction, communication, or play.",
    ),
]

setting_categories = [
    "controlled_setting",
    "uncontrolled_naturalistic_remote",
    "both_controlled_and_uncontrolled",
    "not_reported",
    "unclear",
]
setting_labels = [
    "Controlled",
    "Naturalistic/remote",
    "Both settings",
    "Not reported",
    "Unclear",
]

dataset_categories = [
    "new_dataset_or_primary_data_collection",
    "existing_dataset_or_secondary_data",
    "not_reported_or_placeholder",
    "manual_review_unclear",
]
dataset_labels = [
    "New/primary data",
    "Existing/secondary data",
    "Not reported",
    "Unclear",
]

time_categories = [
    "longitudinal_or_repeated_time_points",
    "not_longitudinal_or_single_time_point",
    "unreported_or_unclear",
]
time_labels = [
    "Longitudinal/repeated",
    "Single time point/not longitudinal",
    "Unreported/unclear",
]

availability_categories = [
    "yes_publicly_available",
    "no_not_publicly_available",
    "available_on_request",
    "limited_or_pseudocode",
    "not_reported_or_placeholder",
]
availability_labels = [
    "Publicly available",
    "Not publicly available",
    "Available on request",
    "Limited",
    "Not reported",
]

protection_categories = [
    "deidentification_or_machine_learning_based_privacy_preserving_methods",
    "face_blurring_or_mosaicing",
    "consent_or_ethics_compliance",
    "restricted_access_or_secure_storage",
]
protection_labels = [
    "De-identification or privacy-preserving ML",
    "Face blurring or mosaicing",
    "Consent or ethics compliance",
    "Restricted access or secure storage",
]


# Reorder and validate the saved summaries.
goals = ordered_summary(
    goals,
    "Study Goal Category",
    [category for category, _ in goal_definitions],
)
findings = ordered_summary(
    findings,
    "Finding Subcategory",
    [category for category, _, _ in finding_definitions],
)

# Panel B reports only the six most frequently identified finding
# subcategories. Stable sorting preserves the predefined category order
# when two categories have the same percentage and count.
top_findings = (
    findings
    .sort_values(
        by=["Percentage", "Count"],
        ascending=[False, False],
        kind="stable",
    )
    .head(6)
    .copy()
)
settings = ordered_summary(settings, "Study Setting Category", setting_categories)
datasets = ordered_summary(datasets, "Dataset Type Category", dataset_categories)
time_frames = ordered_summary(time_frames, "Time Frame Category", time_categories)
code_availability = ordered_summary(
    code_availability,
    "Code Availability Category",
    availability_categories,
)
data_availability = ordered_summary(
    data_availability,
    "Data Availability Category",
    availability_categories,
)
protection = ordered_summary(
    protection,
    "Protection Measure Category",
    protection_categories,
)


# ============================================================
# 5. CREATE THE COMBINED FIGURE
# ============================================================

finding_columns = 6
finding_rows = 1

fig = plt.figure(
    figsize=(24, 18.5),
    layout="constrained",
    facecolor="white",
)

outer_grid = fig.add_gridspec(
    nrows=5,
    ncols=1,
    height_ratios=[2.3, 4.3, 3.3, 3.6, 3.3],
    hspace=0.08,
)


# ------------------------------------------------------------
# PANEL A: STUDY GOALS
# ------------------------------------------------------------

ax_a = fig.add_subplot(outer_grid[0])

goal_counts = goals["Count"].astype(int).to_numpy()
goal_labels = [wrap_label(label, 18) for _, label in goal_definitions]

bars_a = ax_a.bar(
    goal_labels,
    goal_counts,
    color=[PALETTE[i % len(PALETTE)] for i in range(len(goal_counts))],
    edgecolor="black",
    linewidth=0.6,
    width=0.72,
)
ax_a.bar_label(bars_a, labels=[str(value) for value in goal_counts], padding=2)
ax_a.set_title("A  Study goals", loc="left", fontweight="bold", pad=3)
ax_a.set_ylabel("Number of studies")
ax_a.set_xlabel("Study-goal category")
ax_a.set_ylim(0, max(goal_counts) * 1.16 if max(goal_counts) else 1)
ax_a.grid(axis="y", linestyle="--", alpha=0.22)
remove_upper_spines(ax_a)


# ------------------------------------------------------------
# PANEL B: MAIN FINDINGS AS PROGRESS RINGS
# ------------------------------------------------------------

panel_b_grid = outer_grid[1].subgridspec(
    nrows=2,
    ncols=1,
    height_ratios=[0.24, 7.76],
    hspace=0.01,
)

ax_b_header = fig.add_subplot(panel_b_grid[0])
ax_b_header.axis("off")
ax_b_header.text(
    0,
    0.78,
    "B  Main findings",
    transform=ax_b_header.transAxes,
    ha="left",
    va="center",
    fontsize=9,
    fontweight="bold",
)

ring_grid = panel_b_grid[1].subgridspec(
    nrows=finding_rows,
    ncols=finding_columns,
    wspace=0.03,
    hspace=0.02,
)

finding_text_lookup = {
    category: (title, explanation)
    for category, title, explanation in finding_definitions
}

for index, row in top_findings.reset_index(drop=True).iterrows():
    category = row["Finding Subcategory"]
    title, explanation = finding_text_lookup[category]
    axis = fig.add_subplot(
        ring_grid[index // finding_columns, index % finding_columns]
    )
    draw_progress_ring(
        axis=axis,
        percentage=row["Percentage"],
        count=row["Count"],
        title=title,
        explanation=explanation,
        color=PALETTE[index % 6],
    )

# Turn off any unused cells in the ring grid.
for index in range(len(top_findings), finding_rows * finding_columns):
    axis = fig.add_subplot(
        ring_grid[index // finding_columns, index % finding_columns]
    )
    axis.axis("off")


# ------------------------------------------------------------
# PANELS C AND D: SETTING, DATASET TYPE, AND TIME FRAME
# Panel C is on the left and Panel D is on the right.
# ------------------------------------------------------------

row_cd = outer_grid[2].subgridspec(1, 3, wspace=0.04)
ax_c1 = fig.add_subplot(row_cd[0])
ax_c2 = fig.add_subplot(row_cd[1])
ax_d = fig.add_subplot(row_cd[2])

draw_partition_pie(
    ax_c1,
    settings,
    "Study Setting Category",
    setting_labels,
    "C(i)  Study setting",
    colors=[BLUE, GREEN, ORANGE, GRAY, PURPLE],
)
draw_partition_pie(
    ax_c2,
    datasets,
    "Dataset Type Category",
    dataset_labels,
    "C(ii)  Dataset source",
    colors=[BLUE, GREEN, GRAY, PURPLE],
)
draw_partition_pie(
    ax_d,
    time_frames,
    "Time Frame Category",
    time_labels,
    "D  Study time frame",
    colors=[BLUE, ORANGE, GRAY],
)


# ------------------------------------------------------------
# PANEL E: PROTECTION METHODS AND SENSITIVE-DATA REPORTING
# ------------------------------------------------------------

row_e = outer_grid[3].subgridspec(1, 2, width_ratios=[1, 1.75], wspace=0.06)
ax_e1 = fig.add_subplot(row_e[1])
ax_e2 = fig.add_subplot(row_e[0])

protection_values = protection["Percentage"].astype(float).to_numpy()
protection_counts = protection["Count"].astype(int).to_numpy()
y_positions = list(range(len(protection_labels)))

bars_e = ax_e1.barh(
    y_positions,
    protection_values,
    color=[BLUE, GREEN, ORANGE, VERMILLION],
    edgecolor="black",
    linewidth=0.5,
    height=0.66,
)

ax_e1.set_yticks([])
ax_e1.invert_yaxis()
ax_e1.set_xlabel("Studies reporting method (%)")
ax_e1.set_title(
    "E(ii)  Methods used to protect sensitive data",
    loc="left",
    fontweight="bold",
    pad=3,
)
# Leave enough horizontal room after each bar for the full method label,
# count, and percentage.
ax_e1.set_xlim(0, max(55, max(protection_values) + 45))
ax_e1.grid(axis="x", linestyle="--", alpha=0.20)
remove_upper_spines(ax_e1)

for bar, method, count, percentage in zip(
    bars_e,
    protection_labels,
    protection_counts,
    protection_values,
):
    ax_e1.text(
        bar.get_width() + 0.7,
        bar.get_y() + bar.get_height() / 2,
        f"{method} — {int(count)} ({float(percentage):.1f}%)",
        ha="left",
        va="center",
        fontsize=6.6,
        color="black",
    )

sensitive_denominator = denominator_from(sensitive)
yes_rows = sensitive.loc[sensitive["Sensitive Data Category"] == "yes"]
if len(yes_rows) != 1:
    raise ValueError(
        "RQ2_sensitive_data_summary.csv must contain exactly one 'yes' row."
    )
sensitive_yes = int(yes_rows.iloc[0]["Count"])
sensitive_binary = pd.DataFrame(
    {
        "Sensitive Data Category": ["reported", "not_reported_or_identified"],
        "Count": [sensitive_yes, sensitive_denominator - sensitive_yes],
        "Total Valid Papers": [sensitive_denominator, sensitive_denominator],
        "Percentage": [
            (sensitive_yes / sensitive_denominator) * 100,
            ((sensitive_denominator - sensitive_yes) / sensitive_denominator) * 100,
        ],
    }
)

draw_partition_pie(
    ax_e2,
    sensitive_binary,
    "Sensitive Data Category",
    ["Sensitive data reported", "Not reported"],
    "E(i)  Sensitive-data reporting",
    colors=[BLUE, ORANGE],
)


# ------------------------------------------------------------
# PANEL F: CODE AND DATA AVAILABILITY
# ------------------------------------------------------------

row_f = outer_grid[4].subgridspec(1, 2, wspace=0.05)
ax_f1 = fig.add_subplot(row_f[0])
ax_f2 = fig.add_subplot(row_f[1])

draw_partition_pie(
    ax_f1,
    code_availability,
    "Code Availability Category",
    availability_labels,
    "F(i)  Code availability",
    colors=PALETTE[:6],
)
draw_partition_pie(
    ax_f2,
    data_availability,
    "Data Availability Category",
    availability_labels,
    "F(ii)  Data availability",
    colors=PALETTE[:6],
    use_category_total=True,
    total_unit="dataset-access instances",
)


# ============================================================
# 6. SAVE AND DISPLAY
# ============================================================

fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.03)
fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.03)

print(f"Saved PNG: {PNG_PATH}")
print(f"Saved PDF: {PDF_PATH}")

plt.show()

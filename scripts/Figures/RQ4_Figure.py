from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

# This script is intended to be stored in scripts/Figures/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "output" / "rq4_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "Figure_RQ4_summary.png"
PDF_PATH = OUTPUT_DIR / "Figure_RQ4_summary.pdf"

TOTAL_STUDIES = 172


# ============================================================
# 2. STYLE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

# Okabe-Ito colorblind-friendly palette plus neutral gray.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
GRAY = "#999999"
LIGHT_GRAY = "#E6E6E6"
DARK = "#222222"


# ============================================================
# 3. GENERAL DATA HELPERS
# ============================================================


def normalize(value):
    """Normalize labels and column names for robust matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def locate_output(*candidate_names):
    """Locate an RQ4 output while tolerating capitalization differences."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"RQ4 output directory not found: {INPUT_DIR}\n"
            "Run the complete RQ4 analysis script before this figure script."
        )

    available = {path.name.lower(): path for path in INPUT_DIR.glob("*.csv")}

    for name in candidate_names:
        if name.lower() in available:
            return available[name.lower()]

    expected = "\n  - ".join(candidate_names)
    raise FileNotFoundError(
        "Could not find the required RQ4 output. Expected one of:\n"
        f"  - {expected}\n"
        f"Available CSV files in {INPUT_DIR}:\n  - "
        + "\n  - ".join(sorted(path.name for path in INPUT_DIR.glob("*.csv")))
    )


def read_output(*candidate_names):
    return pd.read_csv(locate_output(*candidate_names))


def find_column(dataframe, aliases):
    normalized_columns = {normalize(column): column for column in dataframe.columns}

    for alias in aliases:
        alias_normalized = normalize(alias)
        if alias_normalized in normalized_columns:
            return normalized_columns[alias_normalized]

    raise KeyError(
        f"Could not find a column matching {aliases}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def boolean_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def summary_lookup(dataframe, label_aliases):
    """Return a count from a saved summary table using label aliases."""
    count_column = find_column(dataframe, ["Count"])

    label_columns = [column for column in dataframe.columns if column != count_column]
    preferred = [
        column
        for column in label_columns
        if "category" in normalize(column)
        or "paradigm" in normalize(column)
        or "metric" in normalize(column)
    ]
    label_column = preferred[0] if preferred else label_columns[0]

    normalized_values = dataframe[label_column].map(normalize)
    normalized_aliases = [normalize(alias) for alias in label_aliases]

    for alias in normalized_aliases:
        exact = dataframe.loc[normalized_values == alias]
        if len(exact) == 1:
            return int(exact.iloc[0][count_column])

    for alias in normalized_aliases:
        partial = dataframe.loc[
            normalized_values.str.contains(re.escape(alias), regex=True, na=False)
        ]
        if len(partial) == 1:
            return int(partial.iloc[0][count_column])

    raise ValueError(
        f"Could not uniquely identify {label_aliases} in column '{label_column}'.\n"
        f"Available labels: {dataframe[label_column].tolist()}"
    )


def percent(count, denominator=TOTAL_STUDIES):
    return 100 * count / denominator if denominator else 0


def remove_upper_spines(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# ============================================================
# 4. PANEL A DATA: MACHINE-LEARNING PARADIGMS
# ============================================================

learning_summary = read_output(
    "RQ4_learning_paradigm_summary.csv",
    "RQ4_rq4_learning_paradigm_summary.csv",
)

supervised_count = summary_lookup(
    learning_summary,
    ["supervised_learning", "supervised learning", "supervised"],
)
unsupervised_count = summary_lookup(
    learning_summary,
    ["unsupervised_learning", "unsupervised learning", "unsupervised"],
)

# Reinforcement learning was reported by zero studies. If a future dataset
# contains a nonzero count, stop rather than silently place it under unclear.
try:
    reinforcement_count = summary_lookup(
        learning_summary,
        ["reinforcement_learning", "reinforcement learning"],
    )
except ValueError:
    reinforcement_count = 0

if reinforcement_count != 0:
    raise ValueError(
        "Reinforcement learning now has a nonzero count. Update Panel A "
        "to show it as a separate pie slice."
    )

unclear_count = TOTAL_STUDIES - supervised_count - unsupervised_count

if unclear_count < 0:
    raise ValueError(
        "Learning-paradigm counts exceed the total number of studies. "
        "Check for overlapping paradigm categories."
    )

paradigm_labels = ["Supervised", "Unsupervised", "Unclear/not reported"]
paradigm_counts = [supervised_count, unsupervised_count, unclear_count]
paradigm_colors = [BLUE, ORANGE, GRAY]


# ============================================================
# 5. PANEL B DATA: MODEL FAMILIES AND TOP SUBTYPES
# ============================================================

algorithm_summary = read_output(
    "RQ4_algorithms_broad_summary.csv",
    "RQ4_rq4_algorithms_broad_summary.csv",
)

family_definitions = [
    {
        "label": "Classical ML",
        "aliases": ["classical_machine_learning_models"],
        "subtype_file": (
            "RQ4_algorithms_classical_subcategories_summary.csv",
            "RQ4_rq4_algorithms_classical_subcategories_summary.csv",
        ),
        "subtype_label_column": ["Classical ML Category"],
        "subtype_short_names": {
            "linear_and_generalised_linear_models": "Linear/GLM",
            "support_vector_methods": "SVM",
            "instance_based_learning": "k-NN",
            "probabilistic_models": "Probabilistic",
            "tree_based_models": "Tree-based",
        },
    },
    {
        "label": "Neural networks",
        "aliases": ["neural_network_models"],
        "subtype_file": (
            "RQ4_algorithms_neural_subcategories_summary.csv",
            "RQ4_rq4_algorithms_neural_subcategories_summary.csv",
        ),
        "subtype_label_column": ["Neural Network Category"],
        "subtype_short_names": {
            "basic_neural_network_models": "ANN/MLP",
            "deep_learning_neural_networks": "Deep/CNN",
            "sequence_models": "RNN/LSTM",
            "graph_neural_networks": "GNN",
            "generative_or_autoencoder_models": "AE/GAN",
            "transformer_or_foundation_models": "Transformer",
            "other_neural_hybrid_models": "Other neural",
        },
    },
    {
        "label": "Ensemble",
        "aliases": ["ensemble_models"],
        "derived_parent_column": "ensemble_models",
        "derived_patterns": {
            "Boosting": r"xgboost|gradient boost|adaboost|lightgbm|catboost|boosting",
            "Bagging/RF": r"bagging|random forest|extra trees",
            "Voting": r"voting|vote classifier",
            "Stacking": r"stacking|stacked ensemble",
        },
    },
    {
        "label": "Statistical/specialized",
        "aliases": ["statistical_and_other_specialised_models"],
        "derived_parent_column": "statistical_and_other_specialised_models",
        "derived_patterns": {
            "LASSO": r"\blasso\b",
            "Markov/HMM": r"markov|\bhmm\b|hidden markov",
            "Bayesian": r"bayes|bayesian",
            "Gaussian process": r"gaussian process",
            "KELM": r"kernel extreme learning machine|\bkelm\b",
        },
    },
]

algorithm_match_table = read_output(
    "RQ4_algorithms_broad_match_table.csv",
    "RQ4_rq4_algorithms_broad_match_table.csv",
)
algorithm_text_column = find_column(
    algorithm_match_table,
    ["algorithm_broad_categories_text", "algorithm text"],
)
algorithm_text = algorithm_match_table[algorithm_text_column].fillna("").astype(str)

family_labels = []
family_counts = []
subtype_counts = []
subtype_labels = []

for family in family_definitions:
    parent_count = summary_lookup(algorithm_summary, family["aliases"])
    family_labels.append(family["label"])
    family_counts.append(parent_count)

    if "subtype_file" in family:
        subtype_summary = read_output(*family["subtype_file"])
        count_column = find_column(subtype_summary, ["Count"])
        subtype_label_column = find_column(
            subtype_summary,
            family["subtype_label_column"],
        )
        top_row = subtype_summary.loc[
            pd.to_numeric(subtype_summary[count_column], errors="coerce").idxmax()
        ]
        subtype_count = int(top_row[count_column])
        normalized_subtype = normalize(top_row[subtype_label_column])
        subtype_label = family["subtype_short_names"].get(
            normalized_subtype,
            str(top_row[subtype_label_column]).replace("_", " "),
        )
    else:
        parent_column = find_column(
            algorithm_match_table,
            [family["derived_parent_column"]],
        )
        parent_mask = boolean_series(algorithm_match_table[parent_column])
        derived_counts = {
            label: int(
                (
                    parent_mask
                    & algorithm_text.str.contains(pattern, regex=True, case=False, na=False)
                ).sum()
            )
            for label, pattern in family["derived_patterns"].items()
        }
        subtype_label = max(derived_counts, key=derived_counts.get)
        subtype_count = derived_counts[subtype_label]

    if subtype_count > parent_count:
        raise ValueError(
            f"Subtype count for {family['label']} ({subtype_count}) exceeds "
            f"its family count ({parent_count})."
        )

    subtype_counts.append(subtype_count)
    subtype_labels.append(subtype_label)


# ============================================================
# 6. PANEL C DATA: BEHAVIORAL-INPUT REPRESENTATIONS
# ============================================================

feature_match_table = read_output(
    "RQ4_features_broad_match_table.csv",
    "RQ4_rq4_features_broad_match_table.csv",
)

handcrafted_columns = [
    find_column(feature_match_table, ["motor_pose_kinematic_features"]),
    find_column(feature_match_table, ["gaze_eye_tracking_features"]),
    find_column(feature_match_table, ["facial_expression_face_features"]),
    find_column(feature_match_table, ["language_speech_acoustic_features"]),
]

# Count unique studies represented in at least one of the four principal
# handcrafted feature domains; never sum their overlapping category counts.
handcrafted_mask = pd.concat(
    [boolean_series(feature_match_table[column]) for column in handcrafted_columns],
    axis=1,
).any(axis=1)

raw_column = find_column(
    feature_match_table,
    ["image_video_visual_features"],
)
learned_column = find_column(
    feature_match_table,
    ["vector_or_embedding_features"],
)

feature_labels = ["Handcrafted", "Raw/minimally\nprocessed", "Learned vector"]
feature_counts = [
    int(handcrafted_mask.sum()),
    int(boolean_series(feature_match_table[raw_column]).sum()),
    int(boolean_series(feature_match_table[learned_column]).sum()),
]
feature_colors = [BLUE, GREEN, ORANGE]

feature_definitions = [
    "Handcrafter/processed feature inputs",
    "Raw/minimally processed feature inputs",
    "Model-derived vectors or embeddings",
]


# ============================================================
# 7. PANEL D DATA: EVALUATION METRICS
# ============================================================

metric_summary = read_output(
    "RQ4_evaluation_metrics_summary.csv",
    "RQ4_rq4_evaluation_metrics_summary.csv",
)

metric_definitions = [
    ("Accuracy", ["accuracy"]),
    ("Sensitivity/recall", ["sensitivity_recall", "sensitivity", "recall"]),
    ("Specificity", ["specificity"]),
    ("Precision", ["precision"]),
    ("F1-score", ["f1_score", "f1"]),
    ("AUC/ROC/AUROC", ["auc_roc_auroc", "auc", "roc"]),
    ("Other metrics", ["other_evaluation_metrics", "other_metrics", "other"]),
]

metric_labels = [item[0] for item in metric_definitions]
metric_counts = [summary_lookup(metric_summary, item[1]) for item in metric_definitions]


# ============================================================
# 8. BUILD MULTI-PANEL FIGURE
# ============================================================

fig = plt.figure(figsize=(16, 11))
grid = fig.add_gridspec(
    2,
    2,
    width_ratios=[0.92, 1.25],
    height_ratios=[1.0, 1.08],
    left=0.055,
    right=0.975,
    top=0.955,
    bottom=0.095,
    wspace=0.25,
    hspace=0.38,
)


# ------------------------------------------------------------
# Panel A: machine-learning paradigms
# ------------------------------------------------------------

ax_a = fig.add_subplot(grid[0, 0])

wedges, _ = ax_a.pie(
    paradigm_counts,
    colors=paradigm_colors,
    startangle=90,
    counterclock=False,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    radius=0.90,
)

legend_labels = [
    f"{label}: {count}/{TOTAL_STUDIES} ({percent(count):.1f}%)"
    for label, count in zip(paradigm_labels, paradigm_counts)
]

ax_a.legend(
    wedges,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
    frameon=False,
    fontsize=9,
)

ax_a.set_title("A  Machine-learning paradigms", loc="left", pad=10)
ax_a.axis("equal")


# ------------------------------------------------------------
# Panel B: model families with nested top subtype
# ------------------------------------------------------------

ax_b = fig.add_subplot(grid[0, 1])

y_b = np.arange(len(family_labels))
family_percentages = [percent(count) for count in family_counts]
subtype_percentages = [percent(count) for count in subtype_counts]

parent_bars = ax_b.barh(
    y_b,
    family_percentages,
    height=0.62,
    color=SKY,
    edgecolor=DARK,
    linewidth=0.8,
    label="Machine-learning family",
)

subtype_bars = ax_b.barh(
    y_b,
    subtype_percentages,
    height=0.34,
    color=BLUE,
    edgecolor=DARK,
    linewidth=0.6,
    label="Most common subtype",
)

for index, (parent_bar, subtype_bar) in enumerate(
    zip(parent_bars, subtype_bars)
):
    parent_value = parent_bar.get_width()
    subtype_value = subtype_bar.get_width()
    label_y = parent_bar.get_y() + parent_bar.get_height() / 2

    subtype_text = (
        f"{subtype_labels[index]}: "
        f"{subtype_counts[index]}"
    )

    # Estimate whether the subtype label fits inside its bar.
    minimum_label_width = max(
        8.0,
        len(subtype_text) * 0.72,
    )

    if subtype_value >= minimum_label_width:
        # Put the subtype inside the darker bar.
        ax_b.text(
            subtype_value / 2,
            label_y,
            subtype_text,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

        # Put the family count after the broader bar.
        ax_b.text(
            parent_value + 1.0,
            label_y,
            (
                f"{family_counts[index]} "
                f"({parent_value:.1f}%)"
            ),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=DARK,
            clip_on=False,
        )

    else:
        # If the subtype label cannot fit, place it after the
        # family bar, followed by the family count.
        ax_b.text(
            parent_value + 1.0,
            label_y,
            (
                f"{subtype_text}  |  "
                f"Family: {family_counts[index]} "
                f"({parent_value:.1f}%)"
            ),
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=DARK,
            clip_on=False,
        )

ax_b.set_yticks(y_b, family_labels)
ax_b.invert_yaxis()
ax_b.set_xlabel("Studies (%)")
ax_b.set_xlim(0, max(family_percentages) * 1.28)
ax_b.grid(axis="x", linestyle="--", alpha=0.25)
ax_b.set_axisbelow(True)
ax_b.set_title("B  Machine-learning families", loc="left", pad=10)
ax_b.legend(loc="lower right", frameon=False, fontsize=8)
remove_upper_spines(ax_b)


# ------------------------------------------------------------
# Panel C: behavioral-input representations
# ------------------------------------------------------------

ax_c = fig.add_subplot(grid[1, 0])

x_c = np.arange(len(feature_labels))
feature_percentages = [percent(count) for count in feature_counts]

bars_c = ax_c.bar(
    x_c,
    feature_percentages,
    width=0.64,
    color=feature_colors,
    edgecolor=DARK,
    linewidth=0.8,
)

for bar, count, value in zip(bars_c, feature_counts, feature_percentages):
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.2,
        f"{count} ({value:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax_c.set_xticks(x_c, feature_labels)
ax_c.set_ylabel("Studies (%)")
ax_c.set_ylim(0, max(feature_percentages) * 1.22)
ax_c.grid(axis="y", linestyle="--", alpha=0.25)
ax_c.set_axisbelow(True)
ax_c.set_title("C  Behavioral-input representations", loc="left", pad=10)
remove_upper_spines(ax_c)

# Definitions appear beneath their corresponding bars.
for x_value, definition in zip(x_c, feature_definitions):
    ax_c.text(
        x_value,
        -0.19,
        "\n".join(textwrap.wrap(definition, width=23)),
        transform=ax_c.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.7,
        color="#444444",
        linespacing=1.12,
        clip_on=False,
    )


# ------------------------------------------------------------
# Panel D: performance metrics
# ------------------------------------------------------------

ax_d = fig.add_subplot(grid[1, 1])

y_d = np.arange(len(metric_labels))
metric_percentages = [percent(count) for count in metric_counts]
metric_colors = [BLUE, GREEN, ORANGE, SKY, VERMILLION, PURPLE, GRAY]

bars_d = ax_d.barh(
    y_d,
    metric_percentages,
    height=0.64,
    color=metric_colors,
    edgecolor=DARK,
    linewidth=0.7,
)

for bar, count, value in zip(bars_d, metric_counts, metric_percentages):
    ax_d.text(
        value + 1.0,
        bar.get_y() + bar.get_height() / 2,
        f"{count} ({value:.1f}%)",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

ax_d.set_yticks(y_d, metric_labels)
ax_d.invert_yaxis()
ax_d.set_xlabel("Studies (%)")
ax_d.set_xlim(0, 92)
ax_d.grid(axis="x", linestyle="--", alpha=0.25)
ax_d.set_axisbelow(True)
ax_d.set_title("D  Model-evaluation metrics", loc="left", pad=10)


# ============================================================
# 10. SAVE AND DISPLAY
# ============================================================

fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.05)
fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.05)

plt.show()

print(f"Saved PNG: {PNG_PATH}")
print(f"Saved PDF: {PDF_PATH}")
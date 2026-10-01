from pathlib import Path
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# 1. OUTPUT PATHS
# ============================================================

# This script is intended to be stored in scripts/Figures/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = Path(
    os.environ.get("ASD_REVIEW_OUTPUT_ROOT", PROJECT_ROOT / "output")
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

OUTPUT_DIR = OUTPUT_ROOT / "figures"

PNG_PATH = OUTPUT_DIR / "Figure_RQ3_summary_mapping.png"
PDF_PATH = OUTPUT_DIR / "Figure_RQ3_summary_mapping.pdf"


# ============================================================
# 2. STUDY COUNTS
# ============================================================

TOTAL_STUDIES = 172

# Counts below are the source of truth (taken from the RQ2/RQ3 result CSVs).
# Every displayed percentage is computed from them by format_percent(), so no
# percentage is typed by hand and each value is rounded exactly once.
MULTIPLE_TASK_TYPES = 58
UNREPORTED_TASK_TYPE = 24


def format_percent(numerator, denominator):
    return f"{100 * numerator / denominator:.1f}%"

ROWS = [
    {
        "name": "GAZE / EYE TRACKING",
        "subtitle": "Visual-attention and eye-movement data",
        "count": 74,
        "color": "#0072B2",
        "tasks": [
            ("Gaze/visual-attention tasks", 61, 172),
            ("Passive viewing", 59, 172),
            ("Active viewing", 15, 172),
            ("Joint-attention tasks", 8, 172),
        ],
        "tools_heading": "Among 61 gaze-task studies",
        "tools": [
            ("Specific eye-tracking tool", 37, 61),
            ("Tobii", 19, 61),
            ("SMI", 8, 61),
        ],
        "tool_examples": "Other reported systems: Gazefinder, GazePoint, EyeLink, and SciEye",
    },
    {
        "name": "MOTOR / MOVEMENT",
        "subtitle": "Pose, gesture, gait, and kinematic data",
        "count": 53,
        "color": "#009E73",
        "tasks": [
            ("Motor/movement tasks", 50, 172),
            ("Gait and posture tasks", 23, 172),
            ("Imitation tasks", 7, 172),
        ],
        "tools_heading": "Among 50 motor-task studies",
        "tools": [
            ("Specific motor-based tool", 10, 50),
            ("Kinect", 5, 50),
        ],
        "tool_examples": "Other reported tools: force plates, OpenPose, and OpenFace",
    },
    {
        "name": "SPEECH / LANGUAGE",
        "subtitle": "Acoustic, linguistic, and conversational data",
        "count": 40,
        "color": "#E69F00",
        "tasks": [
            ("Language/speech/audio tasks", 27, 172),
        ],
        "task_examples": (
            "Representative tasks: picture description; word, sentence, or story reading; "
            "natural conversation; and ADOS-2 story-telling activities"
        ),
        "tools_heading": "Across 92 audio/video-relevant studies*",
        "tools": [
            ("Codable recording tool", 32, 92),
            ("Camera or webcam", 27, 92),
            ("Microphone/audio recorder", 8, 92),
        ],
    },
]


# ============================================================
# 3. STYLE
# ============================================================

PANEL_FACE = "#F7F7F7"
CARD_FACE = "#FFFFFF"
EDGE = "#222222"
MUTED = "#555555"
ARROW = "#707070"


# ============================================================
# 4. DRAWING HELPERS
# ============================================================


def rounded_box(axis, x, y, width, height, facecolor, edgecolor=EDGE,
                linewidth=1.0, radius=0.015, zorder=1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def wrapped(value, width):
    return "\n".join(textwrap.wrap(value, width=width))


def draw_arrow(axis, x_start, x_end, y):
    axis.add_patch(
        FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=ARROW,
            shrinkA=2,
            shrinkB=2,
            zorder=5,
        )
    )


def draw_column_heading(axis, x, width, heading, subheading):
    rounded_box(
        axis,
        x,
        0.905,
        width,
        0.07,
        facecolor="#222222",
        edgecolor="#222222",
        linewidth=0,
        radius=0.012,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        0.947,
        heading,
        ha="center",
        va="center",
        color="white",
        fontsize=13,
        fontweight="bold",
        zorder=4,
    )
    axis.text(
        x + width / 2,
        0.914,
        subheading,
        ha="center",
        va="center",
        color="white",
        fontsize=8.5,
        zorder=4,
    )


def draw_modality_card(axis, x, y, width, height, row):
    rounded_box(axis, x, y, width, height, CARD_FACE, linewidth=1.2)
    axis.add_patch(
        FancyBboxPatch(
            (x + 0.006, y + 0.012),
            0.011,
            height - 0.024,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=row["color"],
            edgecolor=row["color"],
            zorder=3,
        )
    )
    axis.text(
        x + width / 2 + 0.006,
        y + height * 0.67,
        row["name"],
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=row["color"],
    )
    axis.text(
        x + width / 2 + 0.006,
        y + height * 0.49,
        f"n = {row['count']}/{TOTAL_STUDIES} "
        f"({format_percent(row['count'], TOTAL_STUDIES)})",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        x + width / 2 + 0.006,
        y + height * 0.28,
        wrapped(row["subtitle"], 34),
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
        linespacing=1.15,
    )


def draw_item_card(axis, x, y, width, height, row, kind):
    rounded_box(axis, x, y, width, height, CARD_FACE, linewidth=1.0)

    if kind == "tasks":
        heading = "TASKS AND SUBTASKS"
        items = row["tasks"]
        qualifier = row.get("task_examples")
    else:
        heading = row["tools_heading"].upper()
        items = row["tools"]
        qualifier = row.get("tool_examples")

    axis.text(
        x + 0.018,
        y + height - 0.035,
        heading,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=row["color"],
    )

    item_start = y + height - 0.078
    if qualifier and len(items) >= 3:
        # Keep the final item clear of the explanatory text at the card base.
        item_gap = 0.040
    elif len(items) >= 4:
        # Distribute the rows so the last one keeps a margin above the card edge.
        item_gap = (height - 0.078 - 0.035) / (len(items) - 1)
    else:
        item_gap = 0.050

    for index, (label, numerator, denominator) in enumerate(items):
        item_y = item_start - index * item_gap
        axis.text(
            x + 0.020,
            item_y,
            "•",
            ha="left",
            va="center",
            fontsize=11,
            color=row["color"],
            fontweight="bold",
        )
        axis.text(
            x + 0.035,
            item_y,
            wrapped(label, 39 if kind == "tasks" else 38),
            ha="left",
            va="center",
            fontsize=8.8,
        )
        axis.text(
            x + width - 0.018,
            item_y,
            f"{numerator}/{denominator} ({format_percent(numerator, denominator)})",
            ha="right",
            va="center",
            fontsize=8.8,
            fontweight="bold",
        )

    if qualifier:
        axis.text(
            x + 0.020,
            y + 0.020,
            wrapped(qualifier, 63 if kind == "tasks" else 54),
            ha="left",
            va="bottom",
            fontsize=7.7,
            color=MUTED,
            linespacing=1.12,
        )


# ============================================================
# 5. BUILD AND SAVE FIGURE
# ============================================================


def build_figure():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column geometry
    modality_x, modality_w = 0.025, 0.245
    tasks_x, tasks_w = 0.315, 0.315
    tools_x, tools_w = 0.675, 0.300

    draw_column_heading(
        ax,
        modality_x,
        modality_w,
        "1. BEHAVIORAL MODALITY",
        "Most frequently represented data types",
    )
    draw_column_heading(
        ax,
        tasks_x,
        tasks_w,
        "2. TASKS",
        "Broad task types and selected subcategories",
    )
    draw_column_heading(
        ax,
        tools_x,
        tools_w,
        "3. COMMON TOOLS",
        "Tool denominators are task-specific",
    )

    # Large background panels
    for x, width in [
        (modality_x, modality_w),
        (tasks_x, tasks_w),
        (tools_x, tools_w),
    ]:
        rounded_box(
            ax,
            x,
            0.105,
            width,
            0.775,
            PANEL_FACE,
            edgecolor="#B5B5B5",
            linewidth=0.9,
            radius=0.012,
            zorder=0,
        )

    row_height = 0.215
    row_y_values = [0.645, 0.395, 0.145]

    for row, row_y in zip(ROWS, row_y_values):
        draw_modality_card(
            ax,
            modality_x + 0.012,
            row_y,
            modality_w - 0.024,
            row_height,
            row,
        )
        draw_item_card(
            ax,
            tasks_x + 0.012,
            row_y,
            tasks_w - 0.024,
            row_height,
            row,
            kind="tasks",
        )
        draw_item_card(
            ax,
            tools_x + 0.012,
            row_y,
            tools_w - 0.024,
            row_height,
            row,
            kind="tools",
        )

        center_y = row_y + row_height / 2
        draw_arrow(ax, modality_x + modality_w + 0.004, tasks_x - 0.004, center_y)
        draw_arrow(ax, tasks_x + tasks_w + 0.004, tools_x - 0.004, center_y)


    # Compact corpus-level context strip
    rounded_box(
        ax,
        0.025,
        0.018,
        0.950,
        0.055,
        facecolor="#FFFFFF",
        edgecolor="#777777",
        linewidth=0.8,
        radius=0.010,
        zorder=2,
    )

    ax.text(
        0.500,
        0.046,
        f"Multiple task types: n = {MULTIPLE_TASK_TYPES}/{TOTAL_STUDIES} "
        f"({format_percent(MULTIPLE_TASK_TYPES, TOTAL_STUDIES)})    |    "
        f"Task type unreported or unclear: n = {UNREPORTED_TASK_TYPE}/{TOTAL_STUDIES} "
        f"({format_percent(UNREPORTED_TASK_TYPE, TOTAL_STUDIES)})    |    "
        "*Audio/video tool estimates use 92 unique studies spanning relevant task categories.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#333333",
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.04)

    plt.show()

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()

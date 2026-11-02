import os
from pathlib import Path

import textwrap

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch





# ============================================================

# 1. OUTPUT PATHS

# ============================================================



# This script is intended to be stored in scripts/Figures/.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



PNG_PATH = OUTPUT_DIR / "Figure_RQ3_summary_mapping.png"

PDF_PATH = OUTPUT_DIR / "Figure_RQ3_summary_mapping.pdf"





# ============================================================
# 2. STUDY COUNTS
# ============================================================

RQ3_RESULTS_DIR = Path(
    os.environ.get(
        "ASD_REVIEW_RQ3_RESULTS_DIR",
        PROJECT_ROOT / "output" / "rq3_results",
    )
).expanduser()
if not RQ3_RESULTS_DIR.is_absolute():
    RQ3_RESULTS_DIR = PROJECT_ROOT / RQ3_RESULTS_DIR


def read_rq3_csv(filename):
    path = RQ3_RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing RQ3 results file: {path}. Run the RQ3 script first, "
            "or set ASD_REVIEW_RQ3_RESULTS_DIR to the folder containing the RQ3_ CSV outputs."
        )
    return pd.read_csv(path)


modality_summary = read_rq3_csv("RQ3_gaze_motor_speech_summary.csv")
task_broad_counts = read_rq3_csv("RQ3_task_type_all_broad_counts.csv")
task_subcategory_counts = read_rq3_csv("RQ3_task_type_all_subcategory_counts.csv")
major_tool_group_summary = read_rq3_csv("RQ3_major_tool_group_summary.csv")
eye_tracking_tool_summary = read_rq3_csv("RQ3_eye_tracking_tool_summary.csv")
motor_tool_summary = read_rq3_csv("RQ3_motor_tool_summary.csv")
video_audio_tool_summary = read_rq3_csv("RQ3_video_audio_tool_summary.csv")

TOTAL_STUDIES = int(modality_summary["Total Valid Papers"].dropna().iloc[0])


def pct(count, denominator):
    return round((count / denominator) * 100, 1) if denominator else 0.0


def fraction_text(count, denominator):
    return f"{int(count)}/{int(denominator)}"


def percent_text(count, denominator):
    return f"{pct(count, denominator):.1f}%"


def modality_value(modality):
    row = modality_summary.loc[
        (modality_summary["Modality"].astype(str).str.lower() == modality.lower())
        & (modality_summary["Category"].astype(str).str.lower() == "yes")
    ].iloc[0]
    count = int(row["Count"])
    return count, pct(count, TOTAL_STUDIES)


def task_value(category):
    row = task_broad_counts.loc[
        task_broad_counts["Category"].astype(str) == category
    ].iloc[0]
    count = int(row["Count"])
    denominator = int(row["Total valid studies"])
    return count, denominator


def subtask_value(category):
    row = task_subcategory_counts.loc[
        task_subcategory_counts["Category"].astype(str) == category
    ].iloc[0]
    count = int(row["Count"])
    denominator = int(row["Total valid studies"])
    return count, denominator


def tool_value(summary_df, category):
    row = summary_df.loc[
        summary_df["Tool Category"].astype(str) == category
    ].iloc[0]
    count = int(row["Count"])
    denominator = int(row["Relevant Task Papers"])
    return count, denominator


def major_tool_denominator(group):
    row = major_tool_group_summary.loc[
        major_tool_group_summary["Major Tool Group"].astype(str) == group
    ].iloc[0]
    return int(row["Relevant Task Papers"])


def task_item(label, category, source="broad"):
    count, denominator = task_value(category) if source == "broad" else subtask_value(category)
    return label, fraction_text(count, denominator), percent_text(count, denominator)


def tool_item(label, summary_df, category):
    count, denominator = tool_value(summary_df, category)
    return label, fraction_text(count, denominator), percent_text(count, denominator)


gaze_count, gaze_pct = modality_value("gaze")
motor_count, motor_pct = modality_value("motor")
speech_count, speech_pct = modality_value("speech")

gaze_tool_denominator = major_tool_denominator("Eye Tracking")
motor_tool_denominator = major_tool_denominator("Motor")
video_audio_tool_denominator = major_tool_denominator("Video Audio")

ROWS = [
    {
        "name": "GAZE / EYE TRACKING",
        "subtitle": "Visual-attention and eye-movement data",
        "count": gaze_count,
        "percentage": gaze_pct,
        "color": "#0072B2",
        "tasks": [
            task_item("Gaze/visual-attention tasks", "gaze_visual_attention_task"),
            task_item("Passive viewing", "passive_visual_stimulus_viewing", source="subcategory"),
            task_item("Active viewing", "active_gaze_visual_orienting_search_task", source="subcategory"),
            task_item("Joint-attention tasks", "joint_attention_social_cue_response_task", source="subcategory"),
        ],
        "tools_heading": f"Among {gaze_tool_denominator} gaze-task studies",
        "tools": [
            tool_item("Specific eye-tracking tool", eye_tracking_tool_summary, "any_named_eye_tracking_tool"),
            tool_item("Tobii", eye_tracking_tool_summary, "tobii"),
            tool_item("SMI", eye_tracking_tool_summary, "smi"),
        ],
        "tool_examples": "Other reported systems: Gazefinder, GazePoint, EyeLink, and SciEye",
    },
    {
        "name": "MOTOR / MOVEMENT",
        "subtitle": "Pose, gesture, gait, and kinematic data",
        "count": motor_count,
        "percentage": motor_pct,
        "color": "#009E73",
        "tasks": [
            task_item("Motor/movement tasks", "motor_movement_task"),
            task_item("Gait and posture tasks", "motor_gait_posture_kinematic_task", source="subcategory"),
            task_item("Imitation tasks", "imitation_task", source="subcategory"),
        ],
        "tools_heading": f"Among {motor_tool_denominator} motor-task studies",
        "tools": [
            tool_item("Specific motor-based tool", motor_tool_summary, "any_named_motor_tool"),
            tool_item("Kinect", motor_tool_summary, "kinect"),
        ],
        "tool_examples": "Other reported tools: force plates, OpenPose, and OpenFace",
    },
    {
        "name": "SPEECH / LANGUAGE",
        "subtitle": "Acoustic, linguistic, and conversational data",
        "count": speech_count,
        "percentage": speech_pct,
        "color": "#E69F00",
        "tasks": [
            task_item("Language/speech/audio tasks", "language_speech_audio_task"),
        ],
        "task_examples": (
            "Representative tasks: picture description; word, sentence, or story reading; "
            "natural conversation; and ADOS-2 story-telling activities"
        ),
        "tools_heading": f"Across {video_audio_tool_denominator} audio/video-relevant studies*",
        "tools": [
            tool_item("Codable recording tool", video_audio_tool_summary, "any_named_video_audio_tool"),
            tool_item("Camera or webcam", video_audio_tool_summary, "camera_or_webcam"),
            tool_item("Microphone/audio recorder", video_audio_tool_summary, "microphone_or_audio_recorder"),
        ],
    },
]

multiple_task_count, multiple_task_denominator = task_value("multiple_task_types")
unclear_task_count, unclear_task_denominator = task_value("unclear")
FOOTER_TEXT = (
    f"Multiple task types: n = {multiple_task_count}/{multiple_task_denominator} "
    f"({percent_text(multiple_task_count, multiple_task_denominator)})    |    "
    f"Task type unreported or unclear: n = {unclear_task_count}/{unclear_task_denominator} "
    f"({percent_text(unclear_task_count, unclear_task_denominator)})    |    "
    f"*Audio/video tool estimates use {video_audio_tool_denominator} unique studies spanning relevant task categories."
)


# ============================================================

# 3. STYLE

# ============================================================



plt.rcParams.update(

    {

        "font.family": "sans-serif",

        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        "font.size": 10,

        "axes.unicode_minus": False,

    }

)



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

        f"n = {row['count']}/{TOTAL_STUDIES} ({row['percentage']:.1f}%)",

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

    else:

        item_gap = 0.043 if len(items) >= 4 else 0.050



    for index, (label, fraction, percentage) in enumerate(items):

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

            f"{fraction} ({percentage})",

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

# 5. BUILD FIGURE

# ============================================================



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
    FOOTER_TEXT,

    ha="center",

    va="center",

    fontsize=8.4,

    color="#333333",

)



fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)





# ============================================================

# 6. SAVE AND DISPLAY

# ============================================================



fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.04)

fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.04)



plt.show()



print(f"Saved PNG: {PNG_PATH}")

print(f"Saved PDF: {PDF_PATH}")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


# ============================================================
# 1. OUTPUT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "Figure_1_PRISMA_flow_diagram.png"
PDF_PATH = OUTPUT_DIR / "Figure_1_PRISMA_flow_diagram.pdf"


# ============================================================
# 2. COUNTS AND TEXT
# ============================================================

phase_1 = [
    (
        "Records identified from databases\n"
        "IEEE Xplore, ACM Digital Library,\n"
        "ScienceDirect,\n"
        "Scopus, PubMed, and CINAHL\n"
        r"$(n = 162{,}986)$"
    ),
    (
        "Records removed before screening\n"
        "Duplicates removed: "
        r"$n = 48{,}107$"
        "\nRecords after deduplication: "
        r"$n = 114{,}879$"
        "\nRecords after keyword filtering: "
        r"$n = 1{,}836$"
    ),
    (
        "Records screened by\n"
        "title and abstract\n"
        r"$(n = 1{,}836)$"
    ),
    (
        "Reports assessed for eligibility\n"
        r"$(n = 204)$"
    ),
    (
        "Studies included from Phase 1\n"
        r"$(n = 110)$"
    ),
]

phase_1_exclusions = {
    2: (
        "Records excluded\n"
        r"$(n = 1{,}632)$"
        "\nOut of scope of the\n"
        "inclusion criteria"
    ),
    3: (
        "Reports excluded\n"
        r"$(n = 94)$"
        "\nOut of scope of the\n"
        "inclusion criteria"
    ),
}

phase_2 = [
    (
        "Records identified from databases\n"
        "Web of Science, ACM Digital Library,\n"
        "ScienceDirect,\n"
        "Scopus, PubMed, and CINAHL\n"
        r"$(n = 4{,}773)$"
    ),
    (
        "Records removed before screening\n"
        "Duplicates removed: "
        r"$n = 935$"
        "\nRecords after deduplication: "
        r"$n = 3{,}838$"
    ),
    (
        "Records screened by\n"
        "title and abstract\n"
        r"$(n = 3{,}838)$"
    ),
    (
        "Reports assessed for eligibility\n"
        r"$(n = 310)$"
    ),
    (
        "Studies included from Phase 2\n"
        r"$(n = 32)$"
    ),
]

phase_2_exclusions = {
    2: (
        "Records excluded\n"
        r"$(n = 3{,}528)$"
        "\nOut of scope of the\n"
        "inclusion criteria"
    ),
    3: (
        "Reports excluded after full text\n"
        r"$(n = 236)$"
        "\nManual removals: "
        r"$n = 42$"
        "\nOut of scope of the\n"
        "inclusion criteria"
    ),
}

total_included = (
    "Total studies included in review\n"
    r"$(n = 142)$"
)


# ============================================================
# 3. STYLE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
    }
)

BOX_FACE = "#F2F2F2"
BOX_EDGE = "#333333"
ARROW_COLOR = "#6F6F6F"
DIVIDER_COLOR = "#B8B8B8"


# ============================================================
# 4. DRAWING HELPERS
# ============================================================

def draw_box(axis, x, y, width, height, text, fontsize=8.6):
    rectangle = Rectangle(
        (x, y),
        width,
        height,
        facecolor=BOX_FACE,
        edgecolor=BOX_EDGE,
        linewidth=0.8,
        zorder=2,
    )
    axis.add_patch(rectangle)

    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.05,
        zorder=3,
    )


def draw_arrow(axis, x_start, x_end, y):
    arrow = FancyArrowPatch(
        (x_start, y),
        (x_end, y),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.8,
        color=ARROW_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    axis.add_patch(arrow)


def draw_down_arrow(axis, x, y_start, y_end):
    arrow = FancyArrowPatch(
        (x, y_start),
        (x, y_end),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.8,
        color=ARROW_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    axis.add_patch(arrow)


def draw_stage_label(axis, x, width, label):
    y = 0.035
    height = 0.045

    rectangle = Rectangle(
        (x, y),
        width,
        height,
        facecolor="white",
        edgecolor=BOX_EDGE,
        linewidth=0.7,
        zorder=2,
    )
    axis.add_patch(rectangle)

    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
    )


# ============================================================
# 5. BUILD FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(13.5, 6.2))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Main geometry
left = 0.115
right = 0.965
gap = 0.013

box_width = (right - left - 4 * gap) / 5
box_height = 0.17

top_y = 0.68
bottom_y = 0.31
exclusion_height = 0.075

x_positions = [
    left + index * (box_width + gap)
    for index in range(5)
]

# Horizontal divider between phases
ax.hlines(
    y=0.535,
    xmin=0.04,
    xmax=0.97,
    colors=DIVIDER_COLOR,
    linewidth=0.9,
)

# Phase labels
ax.text(
    0.085,
    top_y + box_height / 2,
    "Phase 1\n2013-2023",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

ax.text(
    0.085,
    bottom_y + box_height / 2,
    "Phase 2\n2024-2026",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

# Draw main phase boxes
for x, text in zip(x_positions, phase_1):
    draw_box(ax, x, top_y, box_width, box_height, text)

for x, text in zip(x_positions, phase_2):
    draw_box(ax, x, bottom_y, box_width, box_height, text)

# Draw exclusion boxes underneath relevant PRISMA steps
for index, text in phase_1_exclusions.items():
    x = x_positions[index]
    y = top_y - exclusion_height - 0.035
    draw_box(ax, x, y, box_width, exclusion_height, text, fontsize=8.1)
    draw_down_arrow(
        ax,
        x + box_width / 2,
        top_y,
        y + exclusion_height,
    )

for index, text in phase_2_exclusions.items():
    x = x_positions[index]
    y = bottom_y - exclusion_height - 0.035
    draw_box(ax, x, y, box_width, exclusion_height, text, fontsize=8.1)
    draw_down_arrow(
        ax,
        x + box_width / 2,
        bottom_y,
        y + exclusion_height,
    )

# Main horizontal arrows
for row_y in (top_y, bottom_y):
    arrow_y = row_y + box_height / 2

    for index in range(4):
        draw_arrow(
            ax,
            x_positions[index] + box_width,
            x_positions[index + 1],
            arrow_y,
        )

# Combined inclusion box
combined_width = 0.22
combined_height = 0.07
combined_x = x_positions[4] + box_width / 2 - combined_width / 2
combined_y = 0.115

draw_box(
    ax,
    combined_x,
    combined_y,
    combined_width,
    combined_height,
    total_included,
    fontsize=9,
)

draw_down_arrow(
    ax,
    x_positions[4] + box_width / 2,
    bottom_y,
    combined_y + combined_height,
)

# PRISMA stage labels
draw_stage_label(
    ax,
    x_positions[0],
    (x_positions[1] + box_width) - x_positions[0],
    "Identification",
)

draw_stage_label(
    ax,
    x_positions[2],
    box_width,
    "Screening",
)

draw_stage_label(
    ax,
    x_positions[3],
    box_width,
    "Eligibility",
)

draw_stage_label(
    ax,
    x_positions[4],
    box_width,
    "Included",
)

fig.subplots_adjust(
    left=0.005,
    right=0.995,
    top=0.99,
    bottom=0.01,
)


# ============================================================
# 6. SAVE AND DISPLAY
# ============================================================

fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()

print(f"Saved PNG: {PNG_PATH}")
print(f"Saved PDF: {PDF_PATH}")
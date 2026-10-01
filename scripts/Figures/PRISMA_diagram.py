import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


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

PNG_PATH = OUTPUT_DIR / "Figure_1_PRISMA_flow_diagram.png"
PDF_PATH = OUTPUT_DIR / "Figure_1_PRISMA_flow_diagram.pdf"


# ============================================================
# 2. COUNTS AND TEXT
# ============================================================

phase_1 = [
    (
        "Records identified from\n"
        "IEEE Xplore, ACM Digital\n"
        "Library, ScienceDirect,\n"
        "Web of Science, Scopus,\n"
        "PubMed, and CINAHL\n"
        r"$(n\ =\ 162{,}986)$"
    ),
    (
        "After removing duplicates:\n"
        r"$n\ =\ 114{,}879$"
        "\n\nAfter filtering by\n"
        "computation-, diagnosis-,\n"
        "and behavioral-\n"
        "data-related words:\n"
        r"$n\ =\ 1{,}836$"
    ),
    (
        "After filtering based\n"
        "on titles and abstracts\n"
        r"$(n\ =\ 204)$"
    ),
    (
        "After filtering\n"
        "based on full texts\n"
        r"$(n\ =\ 111)$"
    ),
    "Final set\n" + r"$(n\ =\ 111)$",
]

phase_2 = [
    (
        "Records identified from\n"
        "the same databases\n"
        r"$(n\ =\ 4{,}773)$"
    ),
    (
        "After removing\n"
        "duplicates:\n"
        r"$n\ =\ 3{,}838$"
    ),
    (
        "After filtering based\n"
        "on titles and abstracts\n"
        r"$n\ =\ 310$"
    ),
    (
        "After filtering\n"
        "based on full texts:\n"
        r"$n\ =\ 74$"
        "\n\nManual removal:\n"
        r"$n\ =\ 13$"
    ),
    "Final set\n" + r"$(n\ =\ 61)$",
]


# ============================================================
# 3. STYLE
# ============================================================


BOX_FACE = "#F2F2F2"
BOX_EDGE = "#333333"
ARROW_COLOR = "#8A8A8A"


# ============================================================
# 4. DRAWING HELPERS
# ============================================================

def draw_box(axis, x, y, width, height, text, fontsize=10.5):
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
        linespacing=1.03,
        zorder=3,
    )


def draw_arrow(axis, x_start, x_end, y):
    arrow = FancyArrowPatch(
        (x_start, y),
        (x_end, y),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.8,
        color=ARROW_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    axis.add_patch(arrow)


def draw_stage_box(axis, x, width, label):
    y = 0.07
    height = 0.055

    rectangle = Rectangle(
        (x, y),
        width,
        height,
        facecolor=BOX_FACE,
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
        fontsize=10,
    )


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11,
        }
    )


# ============================================================
# 5. BUILD FIGURE
# ============================================================


def main():
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.2, 5.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main box geometry
    left = 0.135
    right = 0.975
    gap = 0.006

    box_width = (right - left - 4 * gap) / 5
    box_height = 0.31

    top_y = 0.62
    bottom_y = 0.22

    x_positions = [
        left + index * (box_width + gap)
        for index in range(5)
    ]

    # Phase labels
    ax.text(
        0.111,
        top_y + box_height / 2,
        "2013–2023",
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    ax.text(
        0.111,
        bottom_y + box_height / 2,
        "2024–2026",
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # Draw both phases
    for x, text in zip(x_positions, phase_1):
        draw_box(
            ax,
            x,
            top_y,
            box_width,
            box_height,
            text,
        )

    for x, text in zip(x_positions, phase_2):
        draw_box(
            ax,
            x,
            bottom_y,
            box_width,
            box_height,
            text,
        )

    # Connecting arrows
    for row_y in (top_y, bottom_y):
        arrow_y = row_y + box_height / 2

        for index in range(4):
            draw_arrow(
                ax,
                x_positions[index] + box_width,
                x_positions[index + 1],
                arrow_y,
            )

    # PRISMA stage labels
    draw_stage_box(
        ax,
        x_positions[0] + 0.002,
        (
            x_positions[1]
            + box_width
            - x_positions[0]
            - 0.004
        ),
        "Identification",
    )

    draw_stage_box(
        ax,
        x_positions[2] + 0.002,
        box_width - 0.004,
        "Screening",
    )

    draw_stage_box(
        ax,
        x_positions[3] + 0.002,
        box_width - 0.004,
        "Selection",
    )

    draw_stage_box(
        ax,
        x_positions[4] + 0.002,
        box_width - 0.004,
        "Inclusion",
    )

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.995,
        bottom=0.005,
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


if __name__ == "__main__":
    main()

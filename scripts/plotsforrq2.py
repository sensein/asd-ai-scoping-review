import os
import matplotlib.pyplot as plt

# -----------------------------
# Raw counts
# -----------------------------
data = {
    "Data Collection Settings": {
        "blank or not specified": 16,
        "clinical": 75,
        "mix": 6,
        "lab": 28,
        "at-home / naturalistic": 11,
        "educational": 1,
    },
    "Existing vs. Newly Collected Datasets": {
        "no data collection": 13,
        "yes data collection": 107,
        "not specified or blank": 17,
    },
    "Cross-sectional vs. Longitudinal Design": {
        "not specified or blank": 16,
        "mix": 1,
        "single session": 110,
        "longitudinal": 10,
    },
    "Dataset Access Availability": {
        "not specified or blank": 11,
        "on request": 4,
        "no": 94,
        "yes": 28,
    },
    "Open Source": {
        "blank": 12,
        "yes": 7,
        "no": 118,
    },
}

# -----------------------------
# Standardize totals
# Add missing amount to blank category
# -----------------------------
def total_count(d):
    return sum(d.values())

max_total = max(total_count(d) for d in data.values())
print(f"Standardizing all plots to total N = {max_total}")

def blank_key_for_dict(d):
    for k in d:
        if "blank" in k.lower() or "not specified" in k.lower():
            return k
    return None

standardized_data = {}

for rq_name, counts in data.items():
    counts = counts.copy()
    current_total = total_count(counts)
    diff = max_total - current_total

    blank_key = blank_key_for_dict(counts)

    if diff > 0:
        if blank_key is not None:
            counts[blank_key] += diff
        else:
            counts["blank or not specified"] = diff

    standardized_data[rq_name] = counts

# -----------------------------
# Plotting function
# -----------------------------
def make_bar_plot(title, counts, output_dir="barplots"):
    os.makedirs(output_dir, exist_ok=True)

    labels = list(counts.keys())
    values = list(counts.values())

    # Size scales a bit with number of categories
    fig_width = max(8, len(labels) * 1.3)
    plt.figure(figsize=(fig_width, 6))
    bars = plt.bar(labels, values)

    plt.title(title.replace("_", " "), fontsize=13)
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(values) * 1.15)

    # Add value labels above bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    filename = os.path.join(output_dir, f"{title}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# -----------------------------
# Create all plots
# -----------------------------
for rq_name, counts in standardized_data.items():
    make_bar_plot(rq_name, counts)

# -----------------------------
# Print standardized totals
# -----------------------------
print("\nStandardized totals:")
for rq_name, counts in standardized_data.items():
    print(f"{rq_name}: total = {sum(counts.values())}, counts = {counts}")
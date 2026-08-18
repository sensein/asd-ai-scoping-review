import os
from pathlib import Path

import pandas as pd
import numpy as np
import re

def main() -> None:
    # ============================================================
    # SETTINGS
    # ============================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = Path(os.environ.get("ASD_REVIEW_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = PROJECT_ROOT / DATA_ROOT
    INPUT_FILE = DATA_ROOT / "final_annotation_sheet_.xlsx"
    SHEET_NAME = 0

    TITLE_COL_INDEX = 0   # Column A
    VENUE_COL_INDEX = 3   # Column D
    YEAR_COL_INDEX = 4    # Column E

    # 35 papers = approximately 20% of 172 final included papers

    YEAR_TARGETS = {
        "2013-2020": 11,
        "2021-2023": 12,
        "2024-2026": 12
    }
    TARGET_SAMPLE_SIZE = sum(YEAR_TARGETS.values())

    RANDOM_SEED = 42


    # ============================================================
    # LOAD FILE
    # ============================================================

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, usecols="A:BY")

    df.columns = df.columns.astype(str).str.strip()

    TITLE_COL = df.columns[TITLE_COL_INDEX]
    VENUE_COL = df.columns[VENUE_COL_INDEX]
    YEAR_COL = df.columns[YEAR_COL_INDEX]

    print("Using title column:", TITLE_COL)
    print("Using venue/journal column:", VENUE_COL)
    print("Using year column:", YEAR_COL)


    # ============================================================
    # CLEANING FUNCTIONS
    # ============================================================

    def clean_venue_text(text):
        text = str(text).lower().strip()

        replacements = {
            "&": "and",
            ".": " ",
            ",": " ",
            ":": " ",
            ";": " ",
            "-": " ",
            "(": " ",
            ")": " ",
            "’": "'",
            "“": "",
            "”": "",
            "?": "",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text).strip()
        return text


    def extract_year(value):
        if pd.isna(value):
            return np.nan

        text = str(value)
        match = re.search(r"(201[3-9]|202[0-6])", text)

        if match:
            return int(match.group(1))

        return np.nan


    def assign_year_group(value):
        year = extract_year(value)

        if pd.isna(year):
            return "Missing/invalid year"

        if 2013 <= year <= 2020:
            return "2013-2020"
        elif 2021 <= year <= 2023:
            return "2021-2023"
        elif 2024 <= year <= 2026:
            return "2024-2026"
        else:
            return "Outside range"


    # ============================================================
    # VENUE CATEGORY FUNCTION
    # ============================================================

    def assign_venue_category(venue):

        venue = clean_venue_text(venue)

        if venue in ["", "-", "nan", "none", "not reported", "not applicable"]:
            return "Other / unclear venues"

        venue_categories = [

            (
                "Autism-specific venues",
                r"\bjournal of autism and developmental disorders\b"
                r"|\bj autism dev disord\b"
                r"|\bautism research\b"
                r"|\bmolecular autism\b"
                r"|\binternational society for autism research\b"
                r"|\binsar\b"
                r"|\bjournal of neurodevelopmental disorders\b"
            ),

            (
                "Workshop / conference / proceedings",
                r"\bconference\b"
                r"|\bproceedings\b"
                r"|\bworkshop\b"
                r"|\bsymposium\b"
                r"|\bceur workshop proceedings\b"
                r"|\bacm international conference proceeding series\b"
                r"|\blecture notes in computer science\b"
                r"|\blecture notes in networks and systems\b"
                r"|\baicconf\b"
                r"|\bcognitive models and artificial intelligence conference\b"
                r"|\binternational conference on machine intelligence and smart innovation\b"
                r"|\binternational conference on computing and machine intelligence\b"
                r"|\bicmi\b"
                r"|\bicce\b"
                r"|\bicce?e\b"
                r"|\binternational conference on computer and communication engineering\b"
                r"|\bcomputer and communication engineering\b"
            ),

            (
                "Broader clinical / psychology / neuroscience venues",
                r"\bfrontiers in psychology\b"
                r"|\bfrontiers in psychiatry\b"
                r"|\bfrontiers in pediatrics\b"
                r"|\bfrontiers in human neuroscience\b"
                r"|\bfrontiers in neurology\b"
                r"|\bfrontiers in neuroscience\b"
                r"|\bfrontiers in computational neuroscience\b"
                r"|\bfrontiers in medicine\b"
                r"|\bpsych journal\b"
                r"|\bbiological psychiatry\b"
                r"|\bcognition\b"
                r"|\bjournal of clinical medicine\b"
                r"|\bbmj open\b"
                r"|\bplos one\b"
                r"|\bscientific reports\b"
                r"|\bnature scientific reports\b"
                r"|\bbmc psychiatry\b"
                r"|\btranslational psychiatry\b"
                r"|\bchildren\b"
                r"|\bjournal of pioneering medical sciences\b"
            ),

            (
                "Digital health / biomedical engineering venues",
                r"\bjournal of medical internet research\b"
                r"|\bj med internet res\b"
                r"|\bjmir\b"
                r"|\bjmir formative research\b"
                r"|\bjmir pediatr parent\b"
                r"|\bjmir human factors\b"
                r"|\bnpj digital medicine\b"
                r"|\bfrontiers in digital health\b"
                r"|\bdigital medicine\b"
                r"|\bhealth informatics\b"
                r"|\bhealthcare engineering\b"
                r"|\bjournal of healthcare engineering\b"
                r"|\bbiomedical engineering\b"
                r"|\bengineering in medicine and biology\b"
                r"|\bieee transactions on biomedical engineering\b"
                r"|\bieee transactions on neural systems and rehabilitation engineering\b"
                r"|\bbioinformatics\b"
                r"|\bcomputational biology\b"
                r"|\bcomputational biology and chemistry\b"
                r"|\bdiagnostics\b"
                r"|\bsensors\b"
                r"|\bcomputers in biology and medicine\b"
                r"|\bbiomedical signal processing and control\b"
                r"|\birbm\b"
            ),

            (
                "Robotics / HRI venues",
                r"\bhuman robot interaction\b"
                r"|\bhri\b"
                r"|\bfrontiers in robotics and ai\b"
                r"|\binternational journal of social robotics\b"
                r"|\bjournal of intelligent and robotic systems\b"
                r"|\brobotics\b"
                r"|\bproceedings of the acm on human computer interaction\b"
                r"|\badvances in human computer interaction\b"
            ),

            (
                "Technology / AI / engineering venues",
                r"\bieee\b"
                r"|\bieee access\b"
                r"|\bacm\b"
                r"|\bmachine learning\b"
                r"|\bmachine intelligence\b"
                r"|\bartificial intelligence\b"
                r"|\bneurocomputing\b"
                r"|\bneural computing and applications\b"
                r"|\bcomplex and intelligent systems\b"
                r"|\bintelligent decision technologies\b"
                r"|\bapplied soft computing\b"
                r"|\balgorithms\b"
                r"|\bcomputer science\b"
                r"|\bprocedia computer science\b"
                r"|\bpeerj computer science\b"
                r"|\bjournal of computational mathematics and data science\b"
                r"|\bcomputational mathematics and data science\b"
                r"|\bcomputing\b"
                r"|\bintelligent systems\b"
                r"|\binternational journal of intelligent engineering systems\b"
                r"|\bpattern recognition\b"
                r"|\bsignal processing\b"
                r"|\btraitement du signal\b"
                r"|\bmultimedia\b"
                r"|\bmultimedia tools and applications\b"
                r"|\bcomputer vision\b"
                r"|\binternational journal of advanced computer science and applications\b"
                r"|\bijacsa\b"
                r"|\belectronics\b"
                r"|\binformation\b"
                r"|\bexpert systems with applications\b"
                r"|\barray\b"
                r"|\balexandria engineering journal\b"
                r"|\bcomputers materials and continua\b"
                r"|\bsadhana\b"
                r"|\bsādhanā\b"
                r"|\bengineering research express\b"
                r"|\bdiscover applied sciences\b"
                r"|\bbulletin of electrical engineering and informatics\b"
                r"|\bengineering technology and applied science research\b"
                r"|\bcomputer modeling in engineering and sciences\b"
            ),

            (
                "Database / source label rather than journal",
                r"\bscience direct\b"
                r"|\bsciencedirect\b"
                r"|\bweb of science\b"
                r"|\bwos\b"
            ),
        ]

        for category_name, pattern in venue_categories:
            if re.search(pattern, venue):
                return category_name

        return "Other / unclear venues"


    # ============================================================
    # APPLY CLASSIFICATIONS
    # ============================================================

    df["extracted_year"] = df[YEAR_COL].apply(extract_year)
    df["year_group"] = df[YEAR_COL].apply(assign_year_group)
    df["venue_category"] = df[VENUE_COL].apply(assign_venue_category)

    valid_year_groups = ["2013-2020", "2021-2023", "2024-2026"]
    df_valid = df[df["year_group"].isin(valid_year_groups)].copy()


    # ============================================================
    # CHECK COUNTS
    # ============================================================

    print("\n========= Counts by Year Group =========")
    print(df_valid["year_group"].value_counts().reindex(valid_year_groups, fill_value=0))

    print("\n========= Counts by Year Group and Venue Category =========")
    print(
        df_valid
        .groupby(["year_group", "venue_category"])
        .size()
        .reset_index(name="paper_count")
        .sort_values(["year_group", "paper_count"], ascending=[True, False])
    )


    # ============================================================
    # PROPORTIONAL ALLOCATION FUNCTION
    # ============================================================

    def proportional_allocation(counts, target_n):
        counts = counts[counts > 0]

        raw = counts / counts.sum() * target_n
        floors = np.floor(raw).astype(int)
        remainder = raw - floors

        allocation = floors.copy()
        remaining = target_n - allocation.sum()

        if remaining > 0:
            for category in remainder.sort_values(ascending=False).index[:remaining]:
                allocation[category] += 1

        while allocation.sum() < target_n:
            available_extra = counts - allocation
            available_extra = available_extra[available_extra > 0]

            if available_extra.empty:
                break

            category_to_add = available_extra.idxmax()
            allocation[category_to_add] += 1

        while allocation.sum() > target_n:
            category_to_remove = allocation[allocation > 0].idxmax()
            allocation[category_to_remove] -= 1

        return allocation


    # ============================================================
    # SAMPLE 35 PAPERS
    # ============================================================

    np.random.seed(RANDOM_SEED)

    selected_indices = []

    for year_group, target_n in YEAR_TARGETS.items():

        year_df = df_valid[df_valid["year_group"] == year_group].copy()

        venue_counts_for_year = year_df["venue_category"].value_counts()
        venue_targets = proportional_allocation(venue_counts_for_year, target_n)

        print(f"\n========= Sampling targets for {year_group} =========")
        print(venue_targets)

        for venue_category, n_to_sample in venue_targets.items():

            if n_to_sample <= 0:
                continue

            subset = year_df[year_df["venue_category"] == venue_category]

            sampled_subset = subset.sample(
                n=int(n_to_sample),
                random_state=RANDOM_SEED
            )

            selected_indices.extend(sampled_subset.index.tolist())


    selected_df = df.loc[selected_indices].copy()

    # Safety check
    if len(selected_df) != TARGET_SAMPLE_SIZE:
        print(f"\nWARNING: sample is not exactly {TARGET_SAMPLE_SIZE}. Current size:", len(selected_df))


    selected_df = selected_df.sort_values(
        ["year_group", "venue_category", "extracted_year"]
    )


    # ============================================================
    # PRINT THE 35 PAPER TITLES ONLY
    # ============================================================

    print("\n\n====================================================")
    print(f"FINAL {TARGET_SAMPLE_SIZE} PAPERS FOR INTER-CODER RELIABILITY")
    print("Stratified by year group and balanced by venue category")
    print("====================================================\n")

    counter = 1

    for year_group in valid_year_groups:
        year_subset = selected_df[selected_df["year_group"] == year_group]

        print(f"\n### {year_group} — {len(year_subset)} papers ###\n")

        for _, row in year_subset.iterrows():
            title = row[TITLE_COL]
            year = row["extracted_year"]
            venue = row[VENUE_COL]
            venue_category = row["venue_category"]

            print(f"{counter}. {title}")
            print(f"   Year: {year}")
            print(f"   Venue: {venue}")
            print(f"   Venue category: {venue_category}")
            print()

            counter += 1


    # Optional: also copy just the titles into a Python list
    selected_titles = selected_df[TITLE_COL].tolist()

    print("\nJust the titles as a Python list:")
    print(selected_titles)


if __name__ == "__main__":
    main()

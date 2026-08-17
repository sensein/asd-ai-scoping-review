# ============================================================
# RQ1: PARTICIPANT CHARACTERISTICS + DIAGNOSTIC REPORTING
# ============================================================

import pandas as pd
import numpy as np
import os
import re

from pathlib import Path

import columns as COL
from analysis_common import RQ_QUESTIONS, RQ_TITLES
from setup_data_ import load_annotation_data, INVALID_VALUES as SHARED_INVALID_VALUES
from helper_functions_ import categories_for_age_ranges as shared_categories_for_age_ranges

RQ_NUMBER = 1
RQ_TITLE = RQ_TITLES[RQ_NUMBER]
RQ_QUESTION = RQ_QUESTIONS[RQ_NUMBER]


def main() -> None:
    # ============================================================
    # 0. SETTINGS
    # ============================================================

    SAVE_OUTPUTS = True
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUT_ROOT = Path(os.environ.get("ASD_REVIEW_OUTPUT_ROOT", PROJECT_ROOT / "output")).expanduser()
    if not OUTPUT_ROOT.is_absolute():
        OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
    OUTPUT_DIR = OUTPUT_ROOT / "rq1_results"
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    def _prefixed_output_name(filename):
        prefix = "RQ1_"
        directory, basename = Path(filename).parent, Path(filename).name
        if basename.lower().startswith(prefix.lower()):
            basename = "RQ1" + basename[len("RQ1"):]
        else:
            basename = prefix + basename
        return str(directory / basename) if str(directory) != "." else basename


    def _write_csv_safely(df_to_save, path, index=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        df_to_save.to_csv(temp_path, index=index)
        temp_path.replace(path)


    def save_df_optional(df_to_save, filename, index=False):
        if SAVE_OUTPUTS and df_to_save is not None:
            _write_csv_safely(df_to_save, OUTPUT_DIR / _prefixed_output_name(filename), index=index)

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    data = load_annotation_data()

    df = data["df"]
    df_subset = data["df_subset"]

    cols_clean_Total_Papers = data["cols_clean_Total_Papers"]
    cols_clean_ASD = data["cols_clean_ASD"]
    cols_clean_Neur = data["cols_clean_Neur"]
    cols_clean_Other = data["cols_clean_Other"]

    empty_rows_Total_Paper = data["empty_rows_Total_Paper"]
    empty_rows_ASD = data["empty_rows_ASD"]
    empty_rows_Neur = data["empty_rows_Neur"]
    empty_rows_Other = data["empty_rows_Other"]

    valid_total = data["valid_total"]
    valid_ASD = data["valid_ASD"]
    valid_Neur = data["valid_Neur"]
    valid_Other = data["valid_Other"]

    valid_papers_Total = data["valid_papers_Total"]
    valid_papers_ASD = data["valid_papers_ASD"]
    valid_papers_Neur = data["valid_papers_Neur"]
    valid_papers_Other = data["valid_papers_Other"]


    # ============================================================
    # 1. COLUMN MAP
    # ============================================================
    # Based on uploaded coding sheet:
    #
    # A  / 0   = Title
    # M  / 12  = # Autistic participants
    # N  / 13  = ASD label
    # O  / 14  = ASD age range
    # P  / 15  = ASD mean age
    # Q  / 16  = ASD age SD
    # R  / 17  = ASD male participants
    # S  / 18  = ASD assessment method
    # T  / 19  = ASD other assessments
    # U  / 20  = ASD comorbidities
    #
    # V  / 21  = # ND participants
    # W  / 22  = ND label
    # X  / 23  = ND age range
    # Y  / 24  = ND mean age
    # Z  / 25  = ND age SD
    # AA / 26  = ND male participants
    # AB / 27  = ND other assessments
    # AC / 28  = ND matched in age with autistic participants
    # AD / 29  = ND matched in gender with autistic participants
    #
    # AE / 30  = # other diagnosis participants
    # AF / 31  = other diagnosis label
    # AG / 32  = other diagnosis age range
    # AH / 33  = other diagnosis mean age
    # AI / 34  = other diagnosis age SD
    # AJ / 35  = other diagnosis male participants
    # AK / 36  = other diagnosis assessment method
    # AL / 37  = other diagnosis other assessment
    # AM / 38  = other diagnosis matched in age with autistic participants
    # AN / 39  = other diagnosis matched in gender with autistic participants

    COL_TITLE = COL.TITLE
    COL_ASD_N = COL.ASD_N
    COL_ASD_LABEL = COL.ASD_LABEL
    COL_ASD_AGE_RANGE = COL.ASD_AGE_RANGE
    COL_ASD_MEAN_AGE = COL.ASD_MEAN_AGE
    COL_ASD_SD_AGE = COL.ASD_SD_AGE
    COL_ASD_MALE = COL.ASD_MALE
    COL_ASD_ASSESSMENT = COL.ASD_ASSESSMENT
    COL_ASD_OTHER_ASSESSMENT = COL.ASD_OTHER_ASSESSMENT
    COL_ASD_COMORBIDITIES = COL.ASD_COMORBIDITIES
    COL_ND_N = COL.NEUROTYPICAL_N
    COL_ND_LABEL = COL.NEUROTYPICAL_LABEL
    COL_ND_AGE_RANGE = COL.NEUROTYPICAL_AGE_RANGE
    COL_ND_MEAN_AGE = COL.NEUROTYPICAL_MEAN_AGE
    COL_ND_SD_AGE = COL.NEUROTYPICAL_SD_AGE
    COL_ND_MALE = COL.NEUROTYPICAL_MALE
    COL_ND_OTHER_ASSESSMENT = 27
    COL_ND_MATCH_AGE = 28
    COL_ND_MATCH_GENDER = 29

    COL_OTHER_N = COL.OTHER_DIAGNOSIS_N
    COL_OTHER_LABEL = COL.OTHER_DIAGNOSIS_LABEL
    COL_OTHER_AGE_RANGE = COL.OTHER_DIAGNOSIS_AGE_RANGE
    COL_OTHER_MEAN_AGE = COL.OTHER_DIAGNOSIS_MEAN_AGE
    COL_OTHER_SD_AGE = COL.OTHER_DIAGNOSIS_SD_AGE
    COL_OTHER_MALE = COL.OTHER_DIAGNOSIS_MALE
    COL_OTHER_ASSESSMENT = 36
    COL_OTHER_OTHER_ASSESSMENT = 37
    COL_OTHER_MATCH_AGE = 38
    COL_OTHER_MATCH_GENDER = 39


    # ============================================================
    # 2. GENERAL HELPERS
    # ============================================================

    INVALID_VALUES = set(SHARED_INVALID_VALUES)


    def normalize_text(x):
        if pd.isna(x):
            return ""
        return str(x).strip().lower()

    def is_invalid(x):
        text = normalize_text(x)
        return text in INVALID_VALUES

    def safe_pct(count, denom):
        return (count / denom) * 100 if denom else 0

    def ensure_series_mask(mask, index):
        if isinstance(mask, pd.Series):
            return mask.reindex(index).fillna(False).astype(bool)
        return pd.Series(mask, index=index).fillna(False).astype(bool)


    valid_total = ensure_series_mask(valid_total, df_subset.index)
    valid_ASD = ensure_series_mask(valid_ASD, df_subset.index)
    valid_Neur = ensure_series_mask(valid_Neur, df_subset.index)
    valid_Other = ensure_series_mask(valid_Other, df_subset.index)


    # ============================================================
    # 3. COUNT PARSING FOR SAMPLE SIZE + GENDER
    # ============================================================

    def extract_count_numbers(text):
        """
        Extract participant-count numbers while ignoring percentages.

        Examples:
        '75 (83.3%)' -> [75]
        '12 (42.9%)' -> [12]
        '9 + 10' -> [9, 10]
        '26 high functioning + 24 low functioning' -> [26, 24]
        """
        if pd.isna(text):
            return []

        text = str(text)

        # Remove percentages, e.g. 83.3%, 42.9%, 81%
        text = re.sub(r"\d+\.?\d*\s*%", " ", text)

        # Remove non-count phrases
        text = re.sub(r"\bacross both groups\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\btotal\b", " ", text, flags=re.IGNORECASE)

        return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


    def parse_subgroup_count_sum(x):
        """
        Parses subgroup-specific participant count cells.

        This is appropriate for the coding sheet because each column already
        represents a participant category, e.g. ASD N, ND N, other diagnosis N,
        or male count within those groups.

        Multiple subgroup counts in the same cell are summed.
        """
        if is_invalid(x):
            return np.nan

        nums = extract_count_numbers(x)

        if len(nums) == 0:
            return np.nan

        return sum(nums)


    # ============================================================
    # 4. SAMPLE SIZE SUMMARIES
    # ============================================================

    def summarize_sample_sizes(series, raw_col, valid_mask, label):
        """
        Descriptive statistics for sample size.

        Uses parsed numeric subgroup counts, not simple pd.to_numeric,
        so cells like '20 + 30' are handled correctly.
        """
        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        numeric = series[valid_mask].dropna()
        raw_valid = raw_col[valid_mask]

        unusable_n = numeric.isna().sum()
        explicit_missing_n = raw_valid.apply(is_invalid).sum()
        nonempty_unparsed_n = raw_valid[
            raw_valid.apply(lambda x: (not is_invalid(x)) and pd.isna(parse_subgroup_count_sum(x)))
        ]

        print(f"\n=============== {label}: SAMPLE SIZE SUMMARY ===============")
        print("Total valid papers:", int(valid_mask.sum()))
        print("N with numeric sample size:", len(numeric))
        print("Explicitly missing sample size:", explicit_missing_n)
        print("Non-empty but unparsed sample size:", len(nonempty_unparsed_n))

        if len(numeric) == 0:
            print("No valid numeric sample sizes.")
            return {
                "group": label,
                "valid_papers": int(valid_mask.sum()),
                "n_numeric": 0,
                "missing_explicit": int(explicit_missing_n),
                "nonempty_unparsed": int(len(nonempty_unparsed_n))
            }

        q1 = numeric.quantile(0.25)
        median = numeric.median()
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        summary = {
            "group": label,
            "valid_papers": int(valid_mask.sum()),
            "n_numeric": int(len(numeric)),
            "missing_explicit": int(explicit_missing_n),
            "nonempty_unparsed": int(len(nonempty_unparsed_n)),
            "mean": numeric.mean(),
            "sd": numeric.std(),
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "min": numeric.min(),
            "max": numeric.max()
        }

        print("Mean:", summary["mean"])
        print("Standard deviation:", summary["sd"])
        print("Median:", summary["median"])
        print("Q1:", summary["q1"])
        print("Q3:", summary["q3"])
        print("IQR:", summary["iqr"])
        print("Minimum:", summary["min"])
        print("Maximum:", summary["max"])

        return summary


    asd_sample_raw = df_subset.iloc[:, COL_ASD_N]
    neur_sample_raw = df_subset.iloc[:, COL_ND_N]
    other_sample_raw = df_subset.iloc[:, COL_OTHER_N]

    asd_sample = asd_sample_raw.apply(parse_subgroup_count_sum)
    neur_sample = neur_sample_raw.apply(parse_subgroup_count_sum)
    other_sample = other_sample_raw.apply(parse_subgroup_count_sum)

    sample_summary_ASD = summarize_sample_sizes(
        asd_sample,
        asd_sample_raw,
        valid_ASD,
        "ASD"
    )

    sample_summary_Neur = summarize_sample_sizes(
        neur_sample,
        neur_sample_raw,
        valid_Neur,
        "NEUROTYPICALS"
    )

    sample_summary_Other = summarize_sample_sizes(
        other_sample,
        other_sample_raw,
        valid_Other,
        "OTHER DIAGNOSES"
    )

    sample_size_summary_df = pd.DataFrame([
        sample_summary_ASD,
        sample_summary_Neur,
        sample_summary_Other
    ])


    # ============================================================
    # 5. SAMPLE SIZE OUTLIER / SENSITIVITY ANALYSIS
    # ============================================================

    def print_sample_outliers(series, valid_mask, label, threshold):
        valid_mask = ensure_series_mask(valid_mask, df_subset.index)
        title_col = df_subset.iloc[:, COL_TITLE]

        outlier_mask = valid_mask & (series >= threshold)

        print(f"\n{label} sample-size outliers >= {threshold}:")
        if outlier_mask.sum() == 0:
            print("None detected.")
        else:
            for idx in df_subset[outlier_mask].index:
                print(
                    f"Row index {idx}: {title_col.loc[idx]} | "
                    f"{label} sample size = {series.loc[idx]}"
                )

        return outlier_mask


    print("\n=============== POTENTIAL INSTANCE-LEVEL OUTLIERS ===============")

    ASD_OUTLIER_THRESHOLD = 10000
    NEUR_OUTLIER_THRESHOLD = 100000
    OTHER_OUTLIER_THRESHOLD = 10000

    asd_outlier_mask = print_sample_outliers(
        asd_sample,
        valid_ASD,
        "ASD",
        ASD_OUTLIER_THRESHOLD
    )

    neur_outlier_mask = print_sample_outliers(
        neur_sample,
        valid_Neur,
        "NEUROTYPICAL",
        NEUR_OUTLIER_THRESHOLD
    )

    other_outlier_mask = print_sample_outliers(
        other_sample,
        valid_Other,
        "OTHER DIAGNOSIS",
        OTHER_OUTLIER_THRESHOLD
    )

    print("\n=============== SENSITIVITY ANALYSIS: EXCLUDING EXTREME OUTLIERS ===============")

    sample_summary_ASD_no_outlier = summarize_sample_sizes(
        asd_sample[~asd_outlier_mask],
        asd_sample_raw[~asd_outlier_mask],
        valid_ASD[~asd_outlier_mask],
        "ASD EXCLUDING OUTLIERS"
    )

    sample_summary_Neur_no_outlier = summarize_sample_sizes(
        neur_sample[~neur_outlier_mask],
        neur_sample_raw[~neur_outlier_mask],
        valid_Neur[~neur_outlier_mask],
        "NEUROTYPICALS EXCLUDING OUTLIERS"
    )

    sample_summary_Other_no_outlier = summarize_sample_sizes(
        other_sample[~other_outlier_mask],
        other_sample_raw[~other_outlier_mask],
        valid_Other[~other_outlier_mask],
        "OTHER DIAGNOSES EXCLUDING OUTLIERS"
    )


    # ============================================================
    # 6. AGE RANGE PARSING
    # ============================================================

    AGE_CATEGORIES = {
        "Infants": (0, 1),
        "Toddlers": (1, 3),
        "Pre-schoolers": (3, 6),
        "Grade-schoolers": (6, 12),
        "Teens": (12, 18),
        "Adults": (18, float("inf"))
    }

    MAX_REASONABLE_AGE_YEARS = 120


    def extract_age_numbers(text):
        return [float(n) for n in re.findall(r"\d+\.?\d*", str(text))]


    def contains_months(text):
        text = normalize_text(text)
        return "month" in text or "months" in text


    def convert_if_months(age_min, age_max, text):
        if contains_months(text):
            return age_min / 12, age_max / 12
        return age_min, age_max


    def is_reasonable_age_range(age_min, age_max):
        """
        Prevents Excel date serials or impossible ages from being counted.
        Example: 45357 should not become 45,357 years old.
        """
        if pd.isna(age_min) or pd.isna(age_max):
            return False

        if age_min < 0:
            return False

        if age_max > MAX_REASONABLE_AGE_YEARS:
            return False

        return True


    def parse_numeric_age_value(cell):
        """Parse the first numeric value from an age mean/SD cell."""
        if is_invalid(cell):
            return np.nan
        nums = extract_age_numbers(cell)
        if len(nums) == 0:
            return np.nan
        value = nums[0]
        if contains_months(cell):
            value = value / 12
        if value < 0 or value > MAX_REASONABLE_AGE_YEARS:
            return np.nan
        return value


    def parse_multiple_age_ranges(cell):
        text = normalize_text(cell)

        dash_matches = re.findall(
            r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)",
            text
        )

        to_matches = re.findall(
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
            text
        )

        ranges = []

        for start, end in dash_matches + to_matches:
            age_min = float(start)
            age_max = float(end)

            age_min, age_max = convert_if_months(age_min, age_max, text)

            age_min, age_max = min(age_min, age_max), max(age_min, age_max)

            if is_reasonable_age_range(age_min, age_max):
                ranges.append((age_min, age_max))

        return ranges


    def parse_under_over_age(cell):
        text = normalize_text(cell)
        nums = extract_age_numbers(text)

        if len(nums) != 1:
            return []

        age = nums[0]

        if contains_months(text):
            age = age / 12

        if "under" in text or "below" in text:
            if is_reasonable_age_range(0, age):
                return [(0, age)]

        if "over" in text or "above" in text:
            if age <= MAX_REASONABLE_AGE_YEARS:
                return [(age, float("inf"))]

        return []


    def parse_mean_sd_age_from_text(cell):
        """
        Handles cells like '3.7 ± 1.3 years'.
        """
        text = normalize_text(cell)

        if "±" not in text and "+/-" not in text:
            return []

        nums = extract_age_numbers(text)

        if len(nums) < 2:
            return []

        mean = nums[0]
        sd = nums[1]

        age_min = max(mean - sd, 0)
        age_max = mean + sd

        age_min, age_max = convert_if_months(age_min, age_max, text)

        if is_reasonable_age_range(age_min, age_max):
            return [(age_min, age_max)]

        return []


    def parse_regular_age(cell):
        text = normalize_text(cell)

        if is_invalid(text):
            return []

        nums = extract_age_numbers(text)

        if len(nums) == 0:
            return []

        if len(nums) == 1:
            age_min = nums[0]
            age_max = nums[0]
        else:
            age_min = min(nums)
            age_max = max(nums)

        age_min, age_max = convert_if_months(age_min, age_max, text)

        if is_reasonable_age_range(age_min, age_max):
            return [(age_min, age_max)]

        return []


    def parse_age_range_cell(cell):
        """
        Parses age range cell only.

        Returns:
        parsed_ranges, parser_type
        """
        if is_invalid(cell):
            return [], "invalid"

        text = normalize_text(cell)

        dash_count = text.count("-") + text.count("–") + text.count("—")
        to_count = len(re.findall(r"\bto\b", text))

        if dash_count >= 2 or to_count >= 2:
            result = parse_multiple_age_ranges(cell)
            if result:
                return result, "multiple_ranges"

        if any(word in text for word in ["under", "below", "over", "above"]):
            result = parse_under_over_age(cell)
            if result:
                return result, "under_over"

        result = parse_mean_sd_age_from_text(cell)
        if result:
            return result, "estimated_mean_sd_from_range_cell"

        result = parse_regular_age(cell)
        if result:
            return result, "regular"

        return [], "unparsed"


    def parse_mean_sd_columns(mean_cell, sd_cell):
        """
        Uses mean and SD columns as fallback when age range is missing,
        unparseable, or affected by Excel date conversion.

        This is an estimated age span, not a true reported range.
        """
        mean = parse_numeric_age_value(mean_cell)
        sd = parse_numeric_age_value(sd_cell)

        if pd.isna(mean):
            return [], "unparsed"

        if mean > MAX_REASONABLE_AGE_YEARS:
            return [], "unparsed"

        if pd.isna(sd):
            age_min = mean
            age_max = mean
            parser_type = "estimated_mean_only"
        else:
            age_min = max(mean - sd, 0)
            age_max = mean + sd
            parser_type = "estimated_mean_sd_columns"

        if is_reasonable_age_range(age_min, age_max):
            return [(age_min, age_max)], parser_type

        return [], "unparsed"


    def parse_age_with_fallback(range_cell, mean_cell, sd_cell):
        """
        First tries the age range column.
        If that fails, uses mean/SD columns as an estimated fallback.
        """
        parsed_ranges, parser_type = parse_age_range_cell(range_cell)

        if parsed_ranges:
            return parsed_ranges, parser_type

        fallback_ranges, fallback_type = parse_mean_sd_columns(mean_cell, sd_cell)

        if fallback_ranges:
            return fallback_ranges, fallback_type

        return [], parser_type


    def add_range_to_age_counts(age_min, age_max, counts, categories):
        """
        Counts overlap with half-open age bins:
        [low, high), except Adults = [18, infinity).

        Age categories are not mutually exclusive across studies.
        """
        for category, (low, high) in categories.items():
            if high == float("inf"):
                if age_max >= low:
                    counts[category] += 1
            else:
                if age_max >= low and age_min < high:
                    counts[category] += 1


    def compute_age_ranges(
        df_subset,
        range_col_index,
        mean_col_index,
        sd_col_index,
        valid_mask,
        group_name,
        categories=AGE_CATEGORIES
    ):
        print(f"\n============= AGE RANGE: {group_name.upper()} =============")

        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        counts = {key: 0 for key in categories}
        not_given = 0
        audit_rows = []

        parser_usage = {
            "regular": 0,
            "multiple_ranges": 0,
            "under_over": 0,
            "estimated_mean_sd_from_range_cell": 0,
            "estimated_mean_sd_columns": 0,
            "estimated_mean_only": 0,
            "invalid": 0,
            "unparsed": 0
        }

        for idx, row in df_subset[valid_mask].iterrows():
            range_cell = row.iloc[range_col_index]
            mean_cell = row.iloc[mean_col_index]
            sd_cell = row.iloc[sd_col_index]

            parsed_ranges, parser_type = parse_age_with_fallback(
                range_cell,
                mean_cell,
                sd_cell
            )

            if parser_type not in parser_usage:
                parser_usage[parser_type] = 0

            parser_usage[parser_type] += 1

            if len(parsed_ranges) == 0:
                not_given += 1

                audit_rows.append({
                    "index": idx,
                    "group": group_name,
                    "raw_age_range": range_cell,
                    "raw_mean_age": mean_cell,
                    "raw_sd_age": sd_cell,
                    "parser_type": parser_type,
                    "parsed_ranges_years": parsed_ranges,
                    "manual_review_needed": not (
                        is_invalid(range_cell) and is_invalid(mean_cell) and is_invalid(sd_cell)
                    )
                })

                continue

            for category in shared_categories_for_age_ranges(parsed_ranges, categories):
                counts[category] += 1

            audit_rows.append({
                "index": idx,
                "group": group_name,
                "raw_age_range": range_cell,
                "raw_mean_age": mean_cell,
                "raw_sd_age": sd_cell,
                "parser_type": parser_type,
                "parsed_ranges_years": parsed_ranges,
                "manual_review_needed": parser_type.startswith("estimated")
            })

        counts["Not given"] = not_given

        total_valid = int(valid_mask.sum())

        percentages = {
            key: safe_pct(value, total_valid)
            for key, value in counts.items()
        }

        audit_df = pd.DataFrame(audit_rows)

        print("Total valid papers:", total_valid)

        print("\n=== AGE CATEGORY COUNTS ===")
        for key, value in counts.items():
            print(f"{key}: {value} ({percentages[key]:.2f}%)")

        print("\n=== PARSER USAGE ===")
        for key, value in parser_usage.items():
            print(f"{key}: {value} ({safe_pct(value, total_valid):.2f}%)")

        manual_review_df = audit_df[audit_df["manual_review_needed"] == True]
        print("\nManual-review age rows:", len(manual_review_df))

        if len(manual_review_df) > 0:
            print("\nFirst 20 age manual-review rows:")
            print(manual_review_df.head(20))

        return counts, percentages, parser_usage, audit_df


    asd_age_counts, asd_age_percentages, asd_age_parser_usage, asd_age_audit = compute_age_ranges(
        df_subset=df_subset,
        range_col_index=COL_ASD_AGE_RANGE,
        mean_col_index=COL_ASD_MEAN_AGE,
        sd_col_index=COL_ASD_SD_AGE,
        valid_mask=valid_ASD,
        group_name="ASD"
    )

    neur_age_counts, neur_age_percentages, neur_age_parser_usage, neur_age_audit = compute_age_ranges(
        df_subset=df_subset,
        range_col_index=COL_ND_AGE_RANGE,
        mean_col_index=COL_ND_MEAN_AGE,
        sd_col_index=COL_ND_SD_AGE,
        valid_mask=valid_Neur,
        group_name="Neurotypical"
    )

    other_age_counts, other_age_percentages, other_age_parser_usage, other_age_audit = compute_age_ranges(
        df_subset=df_subset,
        range_col_index=COL_OTHER_AGE_RANGE,
        mean_col_index=COL_OTHER_MEAN_AGE,
        sd_col_index=COL_OTHER_SD_AGE,
        valid_mask=valid_Other,
        group_name="Other diagnoses"
    )

    age_audit_all = pd.concat(
        [asd_age_audit, neur_age_audit, other_age_audit],
        ignore_index=True
    )

    age_manual_review = age_audit_all[
        age_audit_all["manual_review_needed"] == True
    ]


    # ============================================================
    # 7. GENDER RATIOS
    # ============================================================

    def compute_gender_ratios(
        df_subset,
        total_col,
        male_col,
        valid_mask,
        label
    ):
        print(f"\n============= GENDER: {label.upper()} =============")

        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        male_ratios = []
        female_ratios = []

        used_totals = []
        used_males = []

        skipped_invalid_total = 0
        skipped_male_missing = 0
        skipped_male_gt_total = 0
        missing_gender = 0

        audit_rows = []

        for idx, row in df_subset[valid_mask].iterrows():
            total_raw = row.iloc[total_col]
            male_raw = row.iloc[male_col]

            total = parse_subgroup_count_sum(total_raw)
            male = parse_subgroup_count_sum(male_raw)

            if pd.isna(total) or total == 0:
                skipped_invalid_total += 1

                audit_rows.append({
                    "index": idx,
                    "group": label,
                    "total_raw": total_raw,
                    "male_raw": male_raw,
                    "parsed_total": total,
                    "parsed_male": male,
                    "status": "skipped_invalid_or_zero_total",
                    "manual_review_needed": not (is_invalid(total_raw) and is_invalid(male_raw))
                })

                continue

            if pd.isna(male):
                missing_gender += 1
                skipped_male_missing += 1

                audit_rows.append({
                    "index": idx,
                    "group": label,
                    "total_raw": total_raw,
                    "male_raw": male_raw,
                    "parsed_total": total,
                    "parsed_male": male,
                    "status": "skipped_missing_male_count",
                    "manual_review_needed": not is_invalid(male_raw)
                })

                continue

            if male > total:
                skipped_male_gt_total += 1

                audit_rows.append({
                    "index": idx,
                    "group": label,
                    "total_raw": total_raw,
                    "male_raw": male_raw,
                    "parsed_total": total,
                    "parsed_male": male,
                    "status": "skipped_male_count_exceeds_group_total_possible_whole_sample_count",
                    "manual_review_needed": True
                })

                continue

            female = total - male

            male_ratio = male / total
            female_ratio = female / total

            male_ratios.append(male_ratio)
            female_ratios.append(female_ratio)

            used_totals.append(total)
            used_males.append(male)

            audit_rows.append({
                "index": idx,
                "group": label,
                "total_raw": total_raw,
                "male_raw": male_raw,
                "parsed_total": total,
                "parsed_male": male,
                "parsed_female": female,
                "male_ratio": male_ratio,
                "female_ratio": female_ratio,
                "status": "used",
                "manual_review_needed": False
            })

        used_n = len(male_ratios)
        total_valid = int(valid_mask.sum())

        pooled_total_n = sum(used_totals)
        pooled_male_n = sum(used_males)
        pooled_female_n = pooled_total_n - pooled_male_n

        summary = {
            "group": label,
            "valid_papers": total_valid,
            "used_in_ratio_calc": used_n,

            "mean_male_ratio_paper_level": np.mean(male_ratios) if used_n else np.nan,
            "sd_male_ratio_paper_level": np.std(male_ratios, ddof=1) if used_n > 1 else np.nan,
            "mean_female_ratio_paper_level": np.mean(female_ratios) if used_n else np.nan,
            "sd_female_ratio_paper_level": np.std(female_ratios, ddof=1) if used_n > 1 else np.nan,

            "pooled_total_n": pooled_total_n,
            "pooled_male_n": pooled_male_n,
            "pooled_female_n": pooled_female_n,
            "pooled_male_ratio": pooled_male_n / pooled_total_n if pooled_total_n else np.nan,
            "pooled_female_ratio": pooled_female_n / pooled_total_n if pooled_total_n else np.nan,

            "missing_gender": missing_gender,
            "skipped_invalid_total": skipped_invalid_total,
            "skipped_male_missing": skipped_male_missing,
            "skipped_male_gt_total": skipped_male_gt_total
        }

        audit_df = pd.DataFrame(audit_rows)

        print("Valid papers:", total_valid)
        print("Used in ratio calculation:", used_n)

        print("\n--- Paper-level gender ratios ---")
        print("Mean male ratio:", summary["mean_male_ratio_paper_level"])
        print("SD male ratio:", summary["sd_male_ratio_paper_level"])
        print("Mean female ratio:", summary["mean_female_ratio_paper_level"])
        print("SD female ratio:", summary["sd_female_ratio_paper_level"])

        print("\n--- Pooled participant-level gender ratios ---")
        print("Pooled total N:", pooled_total_n)
        print("Pooled male N:", pooled_male_n)
        print("Pooled female N:", pooled_female_n)
        print("Pooled male ratio:", summary["pooled_male_ratio"])
        print("Pooled female ratio:", summary["pooled_female_ratio"])

        print("\n--- Missing / skipped breakdown ---")
        print("Missing gender:", missing_gender)
        print("Skipped invalid or zero total:", skipped_invalid_total)
        print("Skipped missing male count:", skipped_male_missing)
        print("Skipped male > total:", skipped_male_gt_total)

        manual_review_df = audit_df[audit_df["manual_review_needed"] == True]

        print("\nManual-review gender rows:", len(manual_review_df))

        if len(manual_review_df) > 0:
            print("\nFirst 20 gender manual-review rows:")
            print(manual_review_df.head(20))

        return summary, audit_df


    ASD_gender_summary, ASD_gender_audit = compute_gender_ratios(
        df_subset=df_subset,
        total_col=COL_ASD_N,
        male_col=COL_ASD_MALE,
        valid_mask=valid_ASD,
        label="ASD"
    )

    Neur_gender_summary, Neur_gender_audit = compute_gender_ratios(
        df_subset=df_subset,
        total_col=COL_ND_N,
        male_col=COL_ND_MALE,
        valid_mask=valid_Neur,
        label="NEUROTYPICALS"
    )

    Other_gender_summary, Other_gender_audit = compute_gender_ratios(
        df_subset=df_subset,
        total_col=COL_OTHER_N,
        male_col=COL_OTHER_MALE,
        valid_mask=valid_Other,
        label="OTHER DIAGNOSES"
    )

    gender_summary_df = pd.DataFrame([
        ASD_gender_summary,
        Neur_gender_summary,
        Other_gender_summary
    ])

    gender_audit_all = pd.concat(
        [ASD_gender_audit, Neur_gender_audit, Other_gender_audit],
        ignore_index=True
    )

    gender_manual_review = gender_audit_all[
        gender_audit_all["manual_review_needed"] == True
    ]


    # ============================================================
    # 8. ASD DIAGNOSIS METHODS
    # ============================================================

    def compute_diagnosis_methods(col, valid_mask, label):
        print(f"\n============= DIAGNOSIS METHODS: {label.upper()} =============")

        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        col_filtered = col[valid_mask].astype(str).str.lower().str.strip()
        total_valid = int(valid_mask.sum())

        patterns = {
            "DSM": (
                r"\bdsm\w*"
                r"|\bdsm[-\s]?(iv|v|4|5)\b"
                r"|diagnostic\s*and\s*statistical\s*manual\s*of\s*mental\s*disorders"
            ),
            "ICD": (
                r"\bicd\w*"
                r"|\bicd[-\s]?(9|10|11)\b"
                r"|international\s*classification\s*of\s*diseases"
            ),
            "ADOS": (
                r"\bados\w*"
                r"|\bados[-\s]?\d+\b"
                r"|autism\s*diagnostic\s*observation\s*schedule"
            ),
            "CARS": (
                r"\bcars\w*"
                r"|\bcars[-\s]?\d+\b"
                r"|\bk-cars\w*"
                r"|childhood\s*autism\s*rating\s*scale"
            ),
            "ADI": (
                r"\badi\w*"
                r"|\badi[-\s]?\w+\b"
                r"|autism\s*diagnostic\s*interview"
            ),
            "Other": (
                r"\bSRS\w*"
                r"|\bsocial responsiveness scale\b"
                r"|\bclinical diagnosisb"
                r"|\bclinicians\b"
                r"|\bphysician\b"
                r"|\bpsychiatrist\w*"
                r"|\bcharity organisation\w*"
                r"|\bclinical diagnosis\b"
                r"|\bclinically diagnosed\b"
                r"|\bquestionnaire\b"
                r"|\bpsychoeducational profile\b"
                r"|\bClassification and Diagnostic Criteria of Mental Disorders\b"
                r"|\bself-reported\b"
                r"|\bGilliam Autism Rating Scale-Second Edition\b"
                r"|\bself reported"
                r"|\bsocial communication questionnaire\b"
                r"|autism\s*diagnostic\s*interview"

            )
        }

        results = []

        for method, pattern in patterns.items():
            count = col_filtered.str.contains(pattern, regex=True, na=False).sum()
            results.append({
                "group": label,
                "method": method,
                "count": int(count),
                "percentage": safe_pct(count, total_valid),
                "denominator": total_valid
            })

            print(f"{method}: {count} ({safe_pct(count, total_valid):.2f}%)")

        not_given = col[valid_mask].apply(is_invalid).sum()

        results.append({
            "group": label,
            "method": "Not given",
            "count": int(not_given),
            "percentage": safe_pct(not_given, total_valid),
            "denominator": total_valid
        })

        print(f"Not given: {not_given} ({safe_pct(not_given, total_valid):.2f}%)")

        return pd.DataFrame(results)


    diagnosis_methods_df = compute_diagnosis_methods(
        df_subset.iloc[:, COL_ASD_ASSESSMENT],
        valid_ASD,
        "ASD"
    )

    # ============================================================
    # 9. PARTICIPANT-GROUP TERMINOLOGY CATEGORIZATION
    # ============================================================
    # Notes:
    # - Terminology categories are NOT mutually exclusive.
    #   A study can be counted in more than one terminology category
    #   if the label contains multiple types of wording.
    # - Denominators:
    #   ASD terminology = valid_ASD
    #   Neurotypical terminology = valid_Neur
    #   Other-diagnosis terminology = valid_Other


    def row_matches_any_pattern(text, patterns):
        """
        Returns True if normalized text matches at least one regex pattern.
        """
        text = normalize_text(text)

        if is_invalid(text):
            return False

        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


    def categorize_terminology(
        df_subset,
        label_col,
        valid_mask,
        group_name,
        category_patterns
    ):
        """
        Categorizes participant-label terminology using predefined regex categories.

        Parameters
        ----------
        df_subset : pandas DataFrame
            Main dataframe.

        label_col : int
            Column index containing participant-group label.

        valid_mask : pandas Series
            Valid mask for that participant group.

        group_name : str
            Name of participant group.

        category_patterns : dict
            Dictionary where keys are category names and values are lists of regex patterns.

        Returns
        -------
        summary_df : pandas DataFrame
            Category-level count table.

        audit_df : pandas DataFrame
            Row-level table showing which categories each study matched.
        """
        print(f"\n============= TERMINOLOGY: {group_name.upper()} =============")

        valid_mask = ensure_series_mask(valid_mask, df_subset.index)
        denom = int(valid_mask.sum())

        summary_rows = []
        audit_rows = []

        for idx, raw_label in df_subset.loc[valid_mask].iloc[:, label_col].items():
            text = normalize_text(raw_label)

            matched_categories = []

            for category, patterns in category_patterns.items():
                if row_matches_any_pattern(text, patterns):
                    matched_categories.append(category)

            audit_rows.append({
                "index": idx,
                "group": group_name,
                "raw_label": raw_label,
                "normalized_label": text,
                "matched_categories": matched_categories,
                "no_category_match": len(matched_categories) == 0 and not is_invalid(raw_label),
                "missing_or_invalid_label": is_invalid(raw_label)
            })

        audit_df = pd.DataFrame(audit_rows)

        for category in category_patterns.keys():
            count = audit_df["matched_categories"].apply(lambda cats: category in cats).sum()

            summary_rows.append({
                "group": group_name,
                "category": category,
                "count": int(count),
                "percentage": safe_pct(count, denom),
                "denominator": denom
            })

            print(f"{category}: {count} ({safe_pct(count, denom):.2f}%)")

        missing_count = audit_df["missing_or_invalid_label"].sum()
        uncategorized_nonmissing = audit_df["no_category_match"].sum()

        summary_rows.append({
            "group": group_name,
            "category": "Missing / not reported",
            "count": int(missing_count),
            "percentage": safe_pct(missing_count, denom),
            "denominator": denom
        })

        summary_rows.append({
            "group": group_name,
            "category": "Non-missing but uncategorized",
            "count": int(uncategorized_nonmissing),
            "percentage": safe_pct(uncategorized_nonmissing, denom),
            "denominator": denom
        })

        print(f"Missing / not reported: {missing_count} ({safe_pct(missing_count, denom):.2f}%)")
        print(f"Non-missing but uncategorized: {uncategorized_nonmissing} ({safe_pct(uncategorized_nonmissing, denom):.2f}%)")

        if uncategorized_nonmissing > 0:
            print("\nFirst 20 non-missing uncategorized labels:")
            print(
                audit_df[
                    audit_df["no_category_match"] == True
                ][["index", "raw_label"]].head(20)
            )

        summary_df = pd.DataFrame(summary_rows)

        return summary_df, audit_df


    # ============================================================
    # 9A. ASD TERMINOLOGY CATEGORIES
    # ============================================================
    # Category logic:
    # 1. Official diagnostic terminology:
    #    ASD, Autism Spectrum Disorder
    # 2. Identity-first language:
    #    autistic children, autistic individuals, autistic participants, etc.
    # 3. Alternative / non-standard terminology:
    #    ASC, high autistic traits, children with autism, high-functioning autism, etc.
    asd_terminology_categories = {
        "Official diagnostic terminology": [
            r"\basd\b",
            r"\basds\b",
            r"\bas\b",  # use carefully; included because your sheet has "non-AS" in control labels
            r"autism\s*spectrum\s*disorder[s]?"
        ],

        "Identity-first language": [
            r"\bautistic\b",
            r"\bautstic\b",  # common typo found in audit
            r"autistic\s+(children|child|individuals?|people|persons?|participants?|adults?|toddlers?|infants?|subjects?)",
            r"autstic\s+(children|child|individuals?|people|persons?|participants?|adults?|toddlers?|infants?|subjects?)"
        ],

        "Alternative / non-standard terminology": [
            r"\basc\b",
            r"\bascs\b",
            r"with\s+ascs?",
            r"autism\s*spectrum\s*condition[s]?",
            r"high\s+autistic\s+traits?",
            r"elevated\s+autistic\s+traits?",
            r"low\s+autistic\s+traits?",
            r"autistic\s+traits?",
            r"children\s+with\s+autism",
            r"child\s+with\s+autism",
            r"individuals?\s+with\s+autism",
            r"people\s+with\s+autism",
            r"participants?\s+with\s+autism",
            r"subjects?\s+with\s+autism",
            r"patients?\s+with\s+autism",
            r"high[-\s]?functioning\s+autism",
            r"\bhfa\b",
            r"\bautism\b"
        ]
    }

    asd_terminology_df, asd_terminology_audit = categorize_terminology(
        df_subset=df_subset,
        label_col=COL_ASD_LABEL,
        valid_mask=valid_ASD,
        group_name="ASD",
        category_patterns=asd_terminology_categories
    )


    # ============================================================
    # 9B. NEUROTYPICAL / CONTROL TERMINOLOGY CATEGORIES
    # ============================================================
    # Category logic:
    # 1. Standard control group terms:
    #    TD, NT, typically developing, neurotypical
    # 2. ASD-specific contrast terms:
    #    non-ASD, non-autistic, ASD negative, children without ASD
    # 3. Vague / non-specific language:
    #    healthy, typical, normal, controls

    neurotypical_terminology_categories = {
        "Official neurotypical/control terminology": [
            r"\btd\b",
            r"\bnt\b",
            r"\bnd\b",
            r"typically\s+developing",
            r"typically\s+developed",
            r"typical\s+development",
            r"neurotypical[s]?",
            r"neurodivergent",
            r"characteristically\s+developing"
        ],

        "ASD-specific contrast terminology": [
            r"non[-\s]?asd",
            r"non[-\s]?as\b",
            r"non[-\s]?autistic",
            r"non[-\s]?autism",
            r"without\s+asd",
            r"without\s+autism",
            r"children\s+without\s+asd",
            r"children\s+without\s+autism",
            r"individuals?\s+without\s+asd",
            r"individuals?\s+without\s+autism",
            r"asd[-\s]?negative",
            r"autism[-\s]?negative",
            r"no\s+asd",
            r"no\s+autism",
            r"low\s+autistic\s+traits?",
            r"low\s+autism\s+traits?",
            r"\bnone[-\s]?asd\b"
        ],

        "Vague / non-specific control terminology": [
            r"\bhealthy\b",
            r"healthy\s+controls?",
            r"\bcontrols?\b",
            r"control\s+group",
            r"\btypical\b",
            r"\bnormal\b",
            r"normally\s+developing",
            r"healthy\s+children",
            r"healthy\s+individuals?"
        ]
    }

    neurotypical_terminology_df, neurotypical_terminology_audit = categorize_terminology(
        df_subset=df_subset,
        label_col=COL_ND_LABEL,
        valid_mask=valid_Neur,
        group_name="Neurotypical / control",
        category_patterns=neurotypical_terminology_categories
    )


    # ============================================================
    # 9C. OTHER-DIAGNOSIS TERMINOLOGY CATEGORIES
    # ============================================================
    # Category logic:
    # 1. Official / specific diagnostic labels:
    #    ADHD, schizophrenia, developmental delay, Down syndrome, etc.
    # 2. Umbrella / broad clinical labels:
    #    other language delays, other cognitive delays, developmental disorders, clinical controls
    # 3. ASD-risk labels:
    #    at risk for ASD, low-risk ASD, high-risk ASD

    other_diagnosis_terminology_categories = {
        "Official / specific diagnostic terminology": [
            r"\badhd\b",
            r"attention[-\s]?deficit[/\s-]*hyperactivity\s+disorder",
            r"attention\s+deficit\s+hyperactivity\s+disorder",
            r"\bdd\b",
            r"developmental\s+delay",
            r"developmentally\s+delayed",
            r"down\s+syndrome",
            r"\bds\b",
            r"schizophrenia",
            r"intellectual\s+disabilit(y|ies)",
            r"\bid\b",
            r"language\s+delay",
            r"specific\s+language\s+impairment",
            r"\bsli\b",
            r"\bslc\b",
            r"speech\s+and\s+language\s+condition",
            r"speech\s+language\s+condition",
            r"speech\s+and\s+language\s+conditions",
            r"global\s+developmental\s+delay",
            r"developmental\s+language\s+disorder",
            r"\bdld\b",
            r"\bond\b",
            r"\bpd\b",
            r"anxiety",
            r"depression",
            r"epilepsy",
            r"cerebral\s+palsy",
            r"fragile\s+x",
            r"tourette",
            r"obsessive[-\s]?compulsive",
            r"\bocd\b"
        ],

        "Umbrella / broad other-diagnosis terminology": [
            r"other\s+diagnos",
            r"other\s+clinical",
            r"clinical\s+controls?",
            r"clinical\s+group",
            r"psychiatric",
            r"neurodevelopmental\s+disorders?",
            r"developmental\s+disorders?",
            r"other\s+developmental",
            r"other\s+language\s+delays?",
            r"other\s+cognitive\s+delays?",
            r"cognitive\s+delay",
            r"language\s+impairment",
            r"learning\s+disabilit(y|ies)",
            r"special\s+needs"
        ],

        "ASD-risk terminology": [
            r"at[-\s]?risk",
            r"at[-\s]?risk\s+children",
            r"at[-\s]?risk\s+for\s+asd",
            r"at[-\s]?risk\s+for\s+autism",
            r"risk\s+for\s+asd",
            r"risk\s+for\s+autism",
            r"high[-\s]?risk\s+asd",
            r"high[-\s]?risk\s+autism",
            r"low[-\s]?risk\s+asd",
            r"low[-\s]?risk\s+autism",
            r"infant[s]?\s+at\s+risk",
            r"familial\s+risk",
            r"younger\s+sibling[s]?",
            r"sibling[s]?\s+of\s+children\s+with\s+asd",
            r"sibling[s]?\s+of\s+children\s+with\s+autism",
            r"baby\s+sibs?",
            r"high[-\s]?risk\s+sibling[s]?"
        ]
    }

    other_diagnosis_terminology_df, other_diagnosis_terminology_audit = categorize_terminology(
        df_subset=df_subset,
        label_col=COL_OTHER_LABEL,
        valid_mask=valid_Other,
        group_name="Other diagnoses",
        category_patterns=other_diagnosis_terminology_categories
    )


    # ============================================================
    # 10. ADDITIONAL ASSESSMENTS / DIAGNOSTIC REPORTING
    # ============================================================

    def count_nonmissing_rows(col, valid_mask, label):
        """
        Counts rows with any non-missing value among valid rows.
        """
        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        count = 0
        matched_indices = []

        for idx, x in col[valid_mask].items():
            if not is_invalid(x):
                count += 1
                matched_indices.append(idx)

        denom = int(valid_mask.sum())

        print(f"\n{label}: {count} ({safe_pct(count, denom):.2f}%)")

        return {
            "label": label,
            "count": int(count),
            "percentage": safe_pct(count, denom),
            "denominator": denom,
            "matched_indices": matched_indices
        }


    additional_asd_assessments_summary = count_nonmissing_rows(
        df_subset.iloc[:, COL_ASD_OTHER_ASSESSMENT],
        valid_ASD,
        "Additional ASD-related assessments reported"
    )

    additional_neurotypical_assessments_summary = count_nonmissing_rows(
        df_subset.iloc[:, COL_ND_OTHER_ASSESSMENT],
        valid_Neur,
        "Additional neurotypical/control assessments reported"
    )

    other_diagnosis_method_summary = count_nonmissing_rows(
        df_subset.iloc[:, COL_OTHER_ASSESSMENT],
        valid_Other,
        "Diagnostic methods reported for other-diagnosis group"
    )

    additional_other_diagnosis_assessments_summary = count_nonmissing_rows(
        df_subset.iloc[:, COL_OTHER_OTHER_ASSESSMENT],
        valid_Other,
        "Additional other-diagnosis assessments reported"
    )


    # ============================================================
    # 11. MATCHING WITH ASD GROUP
    # ============================================================

    def count_yes_rows(col, valid_mask, label):
        """
        Counts exact 'yes' responses among valid rows.
        This avoids counting ambiguous values like 'not yes' or explanatory text.
        """
        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        count = 0
        matched_indices = []

        for idx, x in col[valid_mask].items():
            text = normalize_text(x)

            if text == "yes":
                count += 1
                matched_indices.append(idx)

        denom = int(valid_mask.sum())

        print(f"\n{label}: {count} ({safe_pct(count, denom):.2f}%)")

        return {
            "label": label,
            "count": int(count),
            "percentage": safe_pct(count, denom),
            "denominator": denom,
            "matched_indices": matched_indices
        }


    matched_neur_age_summary = count_yes_rows(
        df_subset.iloc[:, COL_ND_MATCH_AGE],
        valid_Neur,
        "Neurotypical group matched for age with autistic group"
    )

    matched_neur_gender_summary = count_yes_rows(
        df_subset.iloc[:, COL_ND_MATCH_GENDER],
        valid_Neur,
        "Neurotypical group matched for gender with autistic group"
    )

    matched_other_age_summary = count_yes_rows(
        df_subset.iloc[:, COL_OTHER_MATCH_AGE],
        valid_Other,
        "Other-diagnosis group matched for age with autistic group"
    )

    matched_other_gender_summary = count_yes_rows(
        df_subset.iloc[:, COL_OTHER_MATCH_GENDER],
        valid_Other,
        "Other-diagnosis group matched for gender with autistic group"
    )


    # ============================================================
    # 12. COMORBIDITIES
    # ============================================================

    def count_comorbidities(col, valid_mask, label):
        """
        Counts studies that report comorbidities for autistic participants.
        Excludes explicit negative/missing values.
        """
        valid_mask = ensure_series_mask(valid_mask, df_subset.index)

        local_invalid = INVALID_VALUES.union({
            "no",
            "none reported",
            "not mentioned",
            "not assessed",
            "no comorbidities",
            "none"
        })

        count = 0
        matched_indices = []

        for idx, x in col[valid_mask].items():
            if pd.isna(x):
                continue

            text = normalize_text(x)

            if text not in local_invalid:
                count += 1
                matched_indices.append(idx)

        denom = int(valid_mask.sum())

        print(f"\n{label}: {count} ({safe_pct(count, denom):.2f}%)")

        return {
            "label": label,
            "count": int(count),
            "percentage": safe_pct(count, denom),
            "denominator": denom,
            "matched_indices": matched_indices
        }


    comorbidities_summary = count_comorbidities(
        df_subset.iloc[:, COL_ASD_COMORBIDITIES],
        valid_ASD,
        "Comorbidities reported for autistic participants"
    )


    # ============================================================
    # 13. COMBINE TERMINOLOGY TABLES
    # ============================================================

    terminology_summary_df = pd.concat(
        [
            asd_terminology_df,
            neurotypical_terminology_df,
            other_diagnosis_terminology_df
        ],
        ignore_index=True
    )

    terminology_audit_all = pd.concat(
        [
            asd_terminology_audit,
            neurotypical_terminology_audit,
            other_diagnosis_terminology_audit
        ],
        ignore_index=True
    )

    terminology_uncategorized_review = terminology_audit_all[
        terminology_audit_all["no_category_match"] == True
    ]


    # ============================================================
    # 14. COMBINE RQ1 SIMPLE COUNTS
    # ============================================================

    rq1_simple_counts = pd.DataFrame([
        additional_asd_assessments_summary,
        additional_neurotypical_assessments_summary,
        other_diagnosis_method_summary,
        additional_other_diagnosis_assessments_summary,
        matched_neur_age_summary,
        matched_neur_gender_summary,
        matched_other_age_summary,
        matched_other_gender_summary,
        comorbidities_summary
    ]).drop(columns=["matched_indices"], errors="ignore")


    # ============================================================
    # 15. COMBINE AGE SUMMARY OUTPUT
    # ============================================================

    age_summary_df = pd.DataFrame([
        {
            "group": "ASD",
            "category": key,
            "count": value,
            "percentage": asd_age_percentages[key],
            "denominator": int(valid_ASD.sum())
        }
        for key, value in asd_age_counts.items()
    ] + [
        {
            "group": "Neurotypical",
            "category": key,
            "count": value,
            "percentage": neur_age_percentages[key],
            "denominator": int(valid_Neur.sum())
        }
        for key, value in neur_age_counts.items()
    ] + [
        {
            "group": "Other diagnoses",
            "category": key,
            "count": value,
            "percentage": other_age_percentages[key],
            "denominator": int(valid_Other.sum())
        }
        for key, value in other_age_counts.items()
    ])


    # ============================================================
    # 16. FINAL RQ1 OUTPUTS
    # ============================================================

    print("\n\n==================== FINAL RQ1 SUMMARY TABLES ====================")

    print("\n--- Sample size summary ---")
    print(sample_size_summary_df)

    print("\n--- Age summary ---")
    print(age_summary_df)

    print("\n--- Gender summary ---")
    print(gender_summary_df)

    print("\n--- ASD diagnosis methods ---")
    print(diagnosis_methods_df)

    print("\n--- Terminology summary ---")
    print(terminology_summary_df)

    print("\n--- Other RQ1 counts ---")
    print(rq1_simple_counts)

    print("\n--- Manual review counts ---")
    print("Age manual-review rows:", len(age_manual_review))
    print("Gender manual-review rows:", len(gender_manual_review))
    print("Terminology uncategorized non-missing rows:", len(terminology_uncategorized_review))

    if len(terminology_uncategorized_review) > 0:
        print("\nFirst 30 uncategorized terminology rows:")
        print(
            terminology_uncategorized_review[
                ["index", "group", "raw_label", "normalized_label"]
            ].head(30)
        )


    # ============================================================
    # 18. CLEANED SCRIPT EXPORTS
    # ============================================================

    save_df_optional(sample_size_summary_df, "sample_size_outputs.csv")
    save_df_optional(age_summary_df, "age_outputs.csv")
    save_df_optional(gender_summary_df, "gender_outputs.csv")
    save_df_optional(diagnosis_methods_df, "asd_diagnosis_method_outputs.csv")
    save_df_optional(terminology_summary_df, "terminology_outputs.csv")
    save_df_optional(terminology_audit_all, "terminology_audit_outputs.csv")
    save_df_optional(terminology_uncategorized_review, "terminology_uncategorized_review_outputs.csv")
    save_df_optional(rq1_simple_counts, "simple_count_outputs.csv")
    save_df_optional(age_audit_all, "age_audit_outputs.csv")
    save_df_optional(gender_audit_all, "gender_audit_outputs.csv")
    save_df_optional(age_manual_review, "age_manual_review_outputs.csv")
    save_df_optional(gender_manual_review, "gender_manual_review_outputs.csv")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

from setup_data_ import load_annotation_data, INVALID_VALUES
from helper_functions_ import (
    OTHER_BEHAVIORAL_CATEGORY_PATTERNS,
    clean_text_series as shared_clean_text_series,
    is_no as shared_is_no,
    is_yes as shared_is_yes,
    set_output_name_prefix,
)


# ============================================================
# 0. SETTINGS
# ============================================================

SAVE_OUTPUTS = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("ASD_REVIEW_OUTPUT_ROOT", PROJECT_ROOT / "output")).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
OUTPUT_DIR = OUTPUT_ROOT / "rq3_results"
set_output_name_prefix("RQ3")


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


if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. COLUMN MAP
# ============================================================
# Based on uploaded coding sheet:
#
# A  / 0   = Title
# E  / 4   = Publication year
#
# AO / 40  = Gaze data
# AP / 41  = Speech/language/audio data
# AQ / 42  = Motor data
# AR / 43  = Other behavioral data
# AS / 44  = Other type of data
# AT / 45  = Feature fusion / feature combination method

COL_TITLE = 0
COL_PUBLICATION_YEAR = 4

COL_GAZE = 40
COL_SPEECH = 41
COL_MOTOR = 42
COL_OTHER_BEHAVIORAL = 43
COL_OTHER_TYPE_DATA = 44
COL_FEATURE_FUSION = 45


COLUMN_MAP = pd.DataFrame([
    {"Variable": "Title", "Excel Column": "A", "Python iloc Index": COL_TITLE},
    {"Variable": "Publication year", "Excel Column": "E", "Python iloc Index": COL_PUBLICATION_YEAR},
    {"Variable": "Gaze data", "Excel Column": "AO", "Python iloc Index": COL_GAZE},
    {"Variable": "Speech/language/audio data", "Excel Column": "AP", "Python iloc Index": COL_SPEECH},
    {"Variable": "Motor data", "Excel Column": "AQ", "Python iloc Index": COL_MOTOR},
    {"Variable": "Other behavioral data", "Excel Column": "AR", "Python iloc Index": COL_OTHER_BEHAVIORAL},
    {"Variable": "Other type of data", "Excel Column": "AS", "Python iloc Index": COL_OTHER_TYPE_DATA},
    {"Variable": "Feature fusion / feature combination", "Excel Column": "AT", "Python iloc Index": COL_FEATURE_FUSION},
])



def _prefixed_output_name(filename):
    prefix = "RQ3_"
    directory, basename = Path(filename).parent, Path(filename).name
    if basename.lower().startswith(prefix.lower()):
        basename = "RQ3" + basename[len("RQ3"):]
    else:
        basename = prefix + basename
    return str(directory / basename) if str(directory) != "." else basename


def _write_csv_safely(df_to_save, path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    df_to_save.to_csv(temp_path, index=index)
    temp_path.replace(path)


print("\n=============COLUMN MAP=============")
print(COLUMN_MAP.to_string(index=False))

if SAVE_OUTPUTS:
    _write_csv_safely(COLUMN_MAP, OUTPUT_DIR / _prefixed_output_name("behavioral_modality_column_map.csv"), index=False)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================


def save_df_optional(df_to_save, filename):
    if SAVE_OUTPUTS:
        _write_csv_safely(df_to_save, OUTPUT_DIR / _prefixed_output_name(filename), index=False)


def clean_text_series(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


def is_yes(value):
    text = str(value).lower().strip()
    return bool(re.search(r"^\s*yes\b", text))


def is_no(value):
    text = str(value).lower().strip()
    return bool(re.search(r"^\s*no\b", text))


clean_text_series = shared_clean_text_series
is_yes = shared_is_yes
is_no = shared_is_no


def extract_year(value):
    text = str(value).strip()
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)

    if match:
        return int(match.group(1))

    return np.nan


def classify_year_group(year):
    if pd.isna(year):
        return "Missing / unreadable year"

    year = int(year)

    if 2013 <= year <= 2017:
        return "2013-2017"
    elif 2018 <= year <= 2023:
        return "2018-2023"
    elif 2024 <= year <= 2026:
        return "2024-2026"
    else:
        return "Outside expected range"


# ============================================================
# 4. GAZE, MOTOR, AND SPEECH DATA
# ============================================================

def yes_no_modality_summary(df, valid_mask, gaze_col, motor_col, speech_col):
    print("\n=============Gaze, Motor, and Speech Data=============")

    total_valid = valid_mask.sum()

    modality_cols = {
        "gaze": gaze_col,
        "motor": motor_col,
        "speech": speech_col
    }

    summary_rows = []
    match_table = pd.DataFrame(index=df.loc[valid_mask].index)

    for modality, col_index in modality_cols.items():
        col_filtered = clean_text_series(df.iloc[:, col_index][valid_mask])

        yes_mask = col_filtered.apply(is_yes)
        no_mask = col_filtered.apply(is_no)
        unclear_mask = ~(yes_mask | no_mask)

        yes_count = yes_mask.sum()
        no_count = no_mask.sum()
        unclear_count = unclear_mask.sum()

        match_table[f"{modality}_text"] = col_filtered
        match_table[f"{modality}_yes"] = yes_mask
        match_table[f"{modality}_no"] = no_mask
        match_table[f"{modality}_unclear"] = unclear_mask

        rows = [
            {"Modality": modality, "Category": "Yes", "Count": yes_count, "Total Valid Papers": total_valid,
             "Percentage": round((yes_count / total_valid) * 100, 2) if total_valid else 0},
            {"Modality": modality, "Category": "No", "Count": no_count, "Total Valid Papers": total_valid,
             "Percentage": round((no_count / total_valid) * 100, 2) if total_valid else 0},
            {"Modality": modality, "Category": "Unclear / missing", "Count": unclear_count, "Total Valid Papers": total_valid,
             "Percentage": round((unclear_count / total_valid) * 100, 2) if total_valid else 0},
        ]

        summary_rows.extend(rows)

        print(f"\n{modality}:")
        print("Yes Count:", yes_count)
        print("Yes Percentage:", round((yes_count / total_valid) * 100, 2) if total_valid else 0)

        print("No Count:", no_count)
        print("No Percentage:", round((no_count / total_valid) * 100, 2) if total_valid else 0)

        print("Unclear Count:", unclear_count)
        print("Unclear Percentage:", round((unclear_count / total_valid) * 100, 2) if total_valid else 0)

    summary_df = pd.DataFrame(summary_rows)

    save_df_optional(summary_df, "gaze_motor_speech_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "gaze_motor_speech_match_table.csv")

    return summary_df, match_table


modality_summary, modality_match_table = yes_no_modality_summary(
    df_subset,
    valid_total,
    gaze_col=COL_GAZE,
    motor_col=COL_MOTOR,
    speech_col=COL_SPEECH
)


# ============================================================
# 5. OTHER BEHAVIORAL / OTHER DATA KEYWORD IDENTIFICATION
# ============================================================

def behavioral_keyword_identification(df, valid_mask, col_indices):
    print("\n=============Other Behavioral / Modality Keyword Identification=============")

    total_valid = valid_mask.sum()

    text = (
        df.loc[valid_mask, df.columns[col_indices]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.strip()
    )

    category_patterns = OTHER_BEHAVIORAL_CATEGORY_PATTERNS

    summary_rows = []
    match_table = pd.DataFrame(index=text.index)
    match_table["combined_other_behavioral_text"] = text

    for category, pattern in category_patterns.items():
        category_mask = text.str.contains(pattern, regex=True, na=False)
        count = category_mask.sum()
        percent = (count / total_valid) * 100 if total_valid else 0

        match_table[category] = category_mask

        summary_rows.append({
            "Category": category,
            "Count": count,
            "Total Valid Papers": total_valid,
            "Percentage": round(percent, 2)
        })

        print(f"\n{category}:")
        print("Count:", count)
        print("Percentage:", round(percent, 2))

    category_cols = list(category_patterns.keys())
    match_table["any_category_matched"] = match_table[category_cols].any(axis=1)

    unmatched_rows = match_table.loc[
        ~match_table["any_category_matched"]
        & match_table["combined_other_behavioral_text"].ne("")
    ].copy()

    print("\n=============Unmatched Non-Empty Other Behavioral Rows=============")

    if unmatched_rows.empty:
        print("No unmatched non-empty rows.")
    else:
        print(unmatched_rows[["combined_other_behavioral_text"]].to_string())

    summary_df = pd.DataFrame(summary_rows)

    save_df_optional(summary_df, "other_behavioral_keyword_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "other_behavioral_keyword_match_table.csv")
    save_df_optional(unmatched_rows.reset_index().rename(columns={"index": "row_index"}), "other_behavioral_unmatched_rows.csv")

    return summary_df, match_table, unmatched_rows


behavioral_keyword_summary, behavioral_keyword_match_table, behavioral_keyword_unmatched_rows = behavioral_keyword_identification(
    df_subset,
    valid_total,
    col_indices=[COL_OTHER_BEHAVIORAL, COL_OTHER_TYPE_DATA]
)


# ============================================================
# 6. FEATURE FUSION / FEATURE COMBINATION METHODS
# ============================================================

def feature_fusion_summary(df, valid_mask, col_index):
    print("\n=============Feature Fusion / Feature Combination Methods=============")

    total_valid = valid_mask.sum()

    col_filtered = clean_text_series(df.iloc[:, col_index][valid_mask])

    placeholder_mask = col_filtered.str.fullmatch(
        r"\s*"
        r"|[-\s]+"
        r"|no"
        r"|none"
        r"|not specified"
        r"|not reported"
        r"|unclear"
        r"|unknown"
        r"|na"
        r"|n/a"
        r"|nd"
        r"|n/d"
        r"|nan",
        na=False
    )

    category_patterns = {
        "concatenating_features": (
            r"\bconcatenate\b"
            r"|\bconcatenated\b"
            r"|\bconcatenation\b"
            r"|\bfeature concatenation\b"
            r"|\bcombine\b"
            r"|\bcombining\b"
            r"|\bcombination\b"

        ),

        "early_fusion": (
            r"\bearly\b"
            r"|\bearly-stage fusion\b"
        ),

        "late_fusion": (
            r"\blate\b"
            r"|\bdecision\b"
            r"|\bscore\b"
        ),

        "hybrid_fusion": (
            r"\bhybrid\b"
            r"|\bmultimodal\b"
            r"|\bmulti-modal\b"
            r"|\bmultiscale\b"
            r"|\bmulti-scale\b"
            r"|\bmulti scale\b"
        ),

        "feature_level_fusion": (
            r"\bfeature-level fusion\b"
            r"|\bfeature level fusion\b"
            r"|\bfeature fusion\b"
            r"|\bfeature combination\b"
            r"|\bcombined features\b"
            r"|\bcombining features\b"
            r"|\bscene-\b"
        ),

        "graph_based_feature_combination": (
            r"\bfeature graph\b"
            r"|\bfeature-graph\b"
            r"|\bgraph\b"
        ),

        "recursive_feature_selection": (
            r"\brecursive feature elimination\b"
            r"|\brfe\b"
            r"|\brecursive\b"
        ),

        "model_based_feature_combination": (
            r"\bpomdp\b"
            r"|\bensemble\b"
            r"|\bstacking\b"
            r"|\bweighted\b"
            r"|\bweighted naive bayes\b"
            r"|\bnaive bayes\b"
            r"|\blstm\b"
            r"|\bcnn\b"
            r"|\bautoencoder\b"
            r"|\bauto-encoder\b"
            r"|\bauto encoder\b"
            r"|\bdecision tree\b"
            r"|\bdecision-tree\b"
            r"|\bmse\b"
            r"|\beuclidean\b"
        ),
    }

    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["feature_fusion_text"] = col_filtered
    match_table["placeholder_no_fusion"] = placeholder_mask

    summary_rows = []

    for category, pattern in category_patterns.items():
        category_mask = ~placeholder_mask & col_filtered.str.contains(pattern, regex=True, na=False)
        match_table[category] = category_mask

        count = category_mask.sum()
        percent_total = (count / total_valid) * 100 if total_valid else 0

        summary_rows.append({
            "Feature Fusion Category": category,
            "Count": count,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": round(percent_total, 2)
        })

    category_cols = list(category_patterns.keys())

    match_table["any_fusion_category_matched"] = match_table[category_cols].any(axis=1)
    match_table["used_fusion_or_feature_combination"] = (
        ~match_table["placeholder_no_fusion"]
        & match_table["any_fusion_category_matched"]
    )

    match_table["manual_review_non_placeholder"] = (
        ~match_table["placeholder_no_fusion"]
        & ~match_table["any_fusion_category_matched"]
    )

    used_count = match_table["used_fusion_or_feature_combination"].sum()
    no_count = match_table["placeholder_no_fusion"].sum()
    manual_count = match_table["manual_review_non_placeholder"].sum()

    overview_rows = [
        {
            "Category": "Used fusion / feature-combination method",
            "Count": used_count,
            "Total Valid Papers": total_valid,
            "Percentage": round((used_count / total_valid) * 100, 2) if total_valid else 0
        },
        {
            "Category": "No fusion / not reported",
            "Count": no_count,
            "Total Valid Papers": total_valid,
            "Percentage": round((no_count / total_valid) * 100, 2) if total_valid else 0
        },
        {
            "Category": "Manual review, non-placeholder",
            "Count": manual_count,
            "Total Valid Papers": total_valid,
            "Percentage": round((manual_count / total_valid) * 100, 2) if total_valid else 0
        },
    ]

    overview_df = pd.DataFrame(overview_rows)
    summary_df = pd.DataFrame(summary_rows)

    print("\n=============Feature Fusion Overview=============")
    print(overview_df.to_string(index=False))

    print("\n=============Feature Fusion Categories=============")
    print(summary_df.to_string(index=False))

    manual_rows = match_table.loc[match_table["manual_review_non_placeholder"]].copy()

    print("\n=============Feature Fusion Rows Needing Manual Review=============")

    if manual_rows.empty:
        print("No rows need manual review.")
    else:
        print(manual_rows[["feature_fusion_text"]].to_string())

    save_df_optional(overview_df, "feature_fusion_overview.csv")
    save_df_optional(summary_df, "feature_fusion_category_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "feature_fusion_match_table.csv")
    save_df_optional(manual_rows.reset_index().rename(columns={"index": "row_index"}), "feature_fusion_manual_review_rows.csv")

    return overview_df, summary_df, match_table, manual_rows


feature_fusion_overview, feature_fusion_category_summary, feature_fusion_match_table, feature_fusion_manual_rows = feature_fusion_summary(
    df_subset,
    valid_total,
    col_index=COL_FEATURE_FUSION
)


# ============================================================
# 7. BEHAVIORAL MODALITY BY PUBLICATION YEAR GROUP
# ============================================================
# this is for RQ5
def compute_behavioral_modality_by_year(
    data_df,
    data_valid_mask,
    year_col,
    gaze_col,
    motor_col,
    speech_col,
    other_behavior_cols
):
    print("\n=============Behavioral Modality by Publication Year Group=============")

    max_len = len(data_df)

    data_df = data_df.iloc[:max_len].copy()
    data_valid_mask = data_valid_mask.iloc[:max_len]

    years = data_df.iloc[:, year_col].apply(extract_year)
    year_groups = years.apply(classify_year_group)

    combined_other_text = (
        data_df.iloc[:, other_behavior_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.strip()
    )

    extra_category_patterns = OTHER_BEHAVIORAL_CATEGORY_PATTERNS

    modality_table = pd.DataFrame(index=data_df.index)

    modality_table["year"] = years
    modality_table["year_group"] = year_groups
    modality_table["valid_annotation"] = data_valid_mask
    modality_table["included_in_modality_analysis"] = data_valid_mask

    modality_table["gaze"] = data_df.iloc[:, gaze_col].apply(is_yes)
    modality_table["motor"] = data_df.iloc[:, motor_col].apply(is_yes)
    modality_table["speech"] = data_df.iloc[:, speech_col].apply(is_yes)

    for category, pattern in extra_category_patterns.items():
        modality_table[category] = combined_other_text.str.contains(
            pattern,
            regex=True,
            na=False
        )

    year_group_order = [
        "2013-2017",
        "2018-2023",
        "2024-2026",
        "Missing / unreadable year",
        "Outside expected range"
    ]

    modality_columns = [
        "gaze",
        "motor",
        "speech",
        "facial_expression_emotion_recognition",
        "nonverbal_other_speech_language",
        "social_interaction",
        "joint_attention",
        "video_analysis_data",
        "decision_making",
        "sensor_data",
        "other_movement_data",
        "other_gaze_data",
        "eeg"
    ]

    summary_rows = []

    for year_group in year_group_order:
        group_mask = (
            (modality_table["year_group"] == year_group)
            & modality_table["included_in_modality_analysis"]
        )

        total_group_papers = group_mask.sum()

        if total_group_papers == 0:
            continue

        for modality in modality_columns:
            count = modality_table.loc[group_mask, modality].sum()
            percentage = (count / total_group_papers) * 100 if total_group_papers else 0

            summary_rows.append({
                "Year Group": year_group,
                "Behavioral Modality": modality,
                "Count": count,
                "Total Papers in Year Group": total_group_papers,
                "Percentage": round(percentage, 2)
            })

    modality_by_year_summary = pd.DataFrame(summary_rows)

    year_coverage_summary = (
        modality_table.loc[modality_table["included_in_modality_analysis"]]
        .groupby("year_group")
        .size()
        .reset_index(name="Valid Papers")
    )

    print("\n=============Valid Papers by Year Group=============")
    print(year_coverage_summary.to_string(index=False))

    print("\n=============Behavioral Modality by Year Group=============")
    print(modality_by_year_summary.to_string(index=False))

    save_df_optional(year_coverage_summary, "valid_papers_by_year_group.csv")
    save_df_optional(modality_by_year_summary, "behavioral_modality_by_year_summary.csv")
    save_df_optional(modality_table.reset_index().rename(columns={"index": "row_index"}), "behavioral_modality_by_year_match_table.csv")

    return modality_by_year_summary, year_coverage_summary, modality_table


modality_by_year_summary, valid_papers_by_year_group, modality_year_match_table = compute_behavioral_modality_by_year(
    data_df=df_subset,
    data_valid_mask=valid_total,
    year_col=COL_PUBLICATION_YEAR,
    gaze_col=COL_GAZE,
    motor_col=COL_MOTOR,
    speech_col=COL_SPEECH,
    other_behavior_cols=[COL_OTHER_BEHAVIORAL, COL_OTHER_TYPE_DATA]
)


# ============================================================
# 8. FINAL CHECK
# ============================================================

print("\n=============FINAL CHECK=============")
print("Total valid papers:", valid_total.sum())
print("Script completed successfully.")

if SAVE_OUTPUTS:
    print(f"Output files saved in: {OUTPUT_DIR.resolve()}")
else:
    print("SAVE_OUTPUTS is False, so no output files were saved.")
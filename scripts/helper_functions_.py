"""
helper_functions_.py
Shared coding and summary functions for cleaned RQ analyses.

Purpose:
- Code each study characteristic once.
- Reuse the same match tables for descriptive summaries and performance subgroup analyses.
- Avoid inconsistent denominators across task type, modality, setting, and ASD age analyses.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from setup_data_ import INVALID_VALUES

OUTPUT_NAME_PREFIX = ""


def set_output_name_prefix(prefix: Optional[str] = None) -> None:
    """Set a process-local output filename prefix such as RQ3 or RQ4."""
    global OUTPUT_NAME_PREFIX
    OUTPUT_NAME_PREFIX = (prefix or "").strip()


def apply_output_name_prefix(filename: str) -> str:
    """Prefix output files by RQ while preserving an existing RQ prefix."""
    if not OUTPUT_NAME_PREFIX:
        return filename

    directory, basename = os.path.split(filename)
    expected = OUTPUT_NAME_PREFIX + "_"
    lower_expected = expected.lower()

    if basename.lower().startswith(lower_expected):
        basename = OUTPUT_NAME_PREFIX + basename[len(OUTPUT_NAME_PREFIX):]
    else:
        basename = expected + basename

    return os.path.join(directory, basename) if directory else basename



# ============================================================
# 0. GENERAL HELPERS
# ============================================================

INVALID_TEXT_VALUES = INVALID_VALUES




def normalize_text(value) -> str:
    """Lowercase and normalize a free-text cell for regex matching."""
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\xa0", " ")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series for consistent regex matching."""
    return series.apply(normalize_text)


def ensure_series_mask(mask, index: pd.Index) -> pd.Series:
    """Ensure a boolean mask is a pandas Series aligned to a target index."""
    if isinstance(mask, pd.Series):
        return mask.reindex(index).fillna(False).astype(bool)
    return pd.Series(mask, index=index).fillna(False).astype(bool)


def is_invalid(value) -> bool:
    """True if a cell is empty or a standard missing/invalid placeholder."""
    text = normalize_text(value)
    return text in INVALID_TEXT_VALUES


def is_yes(value) -> bool:
    """Robust yes detector for binary yes/no modality columns."""
    text = normalize_text(value)
    if is_invalid(text):
        return False
    return bool(re.search(r"^(yes|y|true|1|used|present|included|reported)\b", text))


def is_no(value) -> bool:
    """Robust no detector for binary yes/no modality columns."""
    text = normalize_text(value)
    if text in {"no", "n", "false", "0", "not used", "absent", "not included"}:
        return True
    return bool(re.search(r"^(no|n|false|0)\b", text))


def safe_pct(count: int | float, total: int | float) -> float:
    if total is None or total == 0 or pd.isna(total):
        return 0.0
    return round((float(count) / float(total)) * 100, 2)


def pct(count: int | float, total: int | float) -> float:
    return safe_pct(count, total)


def count_percent_rows(counts: Dict[str, int], total_valid: int, category_col: str = "Category") -> pd.DataFrame:
    rows = []
    for category, count in counts.items():
        rows.append({
            category_col: category,
            "Count": int(count),
            "Total Valid Papers": int(total_valid),
            "Percentage": safe_pct(count, total_valid),
        })
    return pd.DataFrame(rows)


def save_df_optional(df: pd.DataFrame, filename: str, output_dir: Optional[str] = None, index: bool = False) -> None:
    """Save a DataFrame through a temporary file, then atomically replace the target."""
    if df is None:
        return
    filename = apply_output_name_prefix(filename)
    path = os.path.join(output_dir, filename) if output_dir else filename
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = f"{path}.tmp"
    df.to_csv(temp_path, index=index)
    os.replace(temp_path, path)


def summarize_accuracy_by_flags(
    match_table: pd.DataFrame,
    accuracy_series: pd.Series,
    flag_cols: Sequence[str],
    category_col: str = "Category",
    accuracy_col: str = "extracted_accuracy_percent",
    save_prefix: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize accuracy for non-exclusive boolean categories.

    Each study can contribute to multiple categories if multiple flag columns are True.
    The denominator for 'Total n' is rows with that category flag=True.
    The denominator for accuracy summaries is rows with flag=True and non-missing accuracy.
    """
    accuracy_aligned = accuracy_series.reindex(match_table.index)

    rows = []
    for col in flag_cols:
        if col not in match_table.columns:
            print(f"Warning: {col} not found in match_table; skipping.")
            continue

        mask = match_table[col].fillna(False).astype(bool)
        values = pd.to_numeric(accuracy_aligned[mask], errors="coerce").dropna()

        rows.append({
            category_col: col,
            "Total n": int(mask.sum()),
            "Accuracy n": int(values.shape[0]),
            "Mean": values.mean(),
            "Median": values.median(),
            "SD": values.std(),
            "Min": values.min(),
            "Max": values.max(),
        })

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("Mean", ascending=False, na_position="last")

    if save_prefix:
        save_df_optional(summary_df, f"{save_prefix}.csv", output_dir=output_dir, index=False)

    return summary_df


def summarize_accuracy_by_group(
    group_series: pd.Series,
    accuracy_series: pd.Series,
    group_col: str = "Group",
    accuracy_col: str = "extracted_accuracy_percent",
    save_prefix: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize accuracy by a mutually exclusive group label."""
    df = pd.DataFrame({
        group_col: group_series,
        accuracy_col: pd.to_numeric(accuracy_series.reindex(group_series.index), errors="coerce"),
    })

    rows_with_accuracy = (
        df.dropna(subset=[accuracy_col])
        .groupby(group_col)[accuracy_col]
        .agg(**{
            "Accuracy n": "count",
            "Mean": "mean",
            "Median": "median",
            "SD": "std",
            "Min": "min",
            "Max": "max",
        })
    )

    total_rows = df.groupby(group_col).size().rename("Total n")
    summary_df = rows_with_accuracy.join(total_rows)
    summary_df = summary_df[["Total n", "Accuracy n", "Mean", "Median", "SD", "Min", "Max"]]
    summary_df = summary_df.sort_values("Mean", ascending=False, na_position="last")
    summary_df = summary_df.reset_index()

    if save_prefix:
        save_df_optional(summary_df, f"{save_prefix}.csv", output_dir=output_dir, index=False)

    return summary_df


# ============================================================
# 1. BEHAVIORAL MODALITY: GAZE, MOTOR, SPEECH
# ============================================================

def build_modality_match_table(df: pd.DataFrame, valid_mask, gaze_col: int, motor_col: int, speech_col: int) -> pd.DataFrame:
    """Build reusable yes/no/unclear match table for behavioral modality columns."""
    valid_mask = ensure_series_mask(valid_mask, df.index)

    modality_cols = {
        "gaze": gaze_col,
        "motor": motor_col,
        "speech": speech_col,
    }

    match_table = pd.DataFrame(index=df.loc[valid_mask].index)

    for modality, col_index in modality_cols.items():
        col_filtered = clean_text_series(df.iloc[:, col_index][valid_mask])
        yes_mask = col_filtered.apply(is_yes)
        no_mask = col_filtered.apply(is_no)
        unclear_mask = ~(yes_mask | no_mask)

        match_table[f"{modality}_text"] = col_filtered
        match_table[f"{modality}_yes"] = yes_mask
        match_table[f"{modality}_no"] = no_mask
        match_table[f"{modality}_unclear"] = unclear_mask

    return match_table


def summarize_modality_match_table(match_table: pd.DataFrame, total_valid: Optional[int] = None) -> pd.DataFrame:
    if total_valid is None:
        total_valid = int(len(match_table))

    rows = []
    for modality in ["gaze", "motor", "speech"]:
        for category, suffix in [("Yes", "yes"), ("No", "no"), ("Unclear / missing", "unclear")]:
            col = f"{modality}_{suffix}"
            count = int(match_table[col].sum()) if col in match_table.columns else 0
            rows.append({
                "Modality": modality,
                "Category": category,
                "Count": count,
                "Total Valid Papers": int(total_valid),
                "Percentage": safe_pct(count, total_valid),
            })
    return pd.DataFrame(rows)


def yes_no_modality_summary(df, valid_mask, gaze_col, motor_col, speech_col, output_dir: Optional[str] = None):
    print("\n============= Gaze, Motor, and Speech Data =============")
    valid_mask = ensure_series_mask(valid_mask, df.index)
    total_valid = int(valid_mask.sum())

    match_table = build_modality_match_table(df, valid_mask, gaze_col, motor_col, speech_col)
    summary_df = summarize_modality_match_table(match_table, total_valid)

    print(summary_df.to_string(index=False))

    save_df_optional(summary_df, "gaze_motor_speech_summary.csv", output_dir=output_dir)
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "gaze_motor_speech_match_table.csv", output_dir=output_dir)

    return summary_df, match_table


# ============================================================
# 1B. OTHER BEHAVIORAL / OTHER DATA KEYWORD CATEGORIES
# ============================================================

OTHER_BEHAVIORAL_CATEGORY_PATTERNS = {
    "facial_expression_emotion_recognition": (
        r"\bfacial\b"
        r"|\bfacial expression\b"
        r"|\bfacial expressions\b"
        r"|\bemotion\b"
        r"|\bemotions\b"
        r"|\bemotion recognition\b"
        r"|\bface\b"
        r"|\bfaces\b"
    ),
    "nonverbal_other_speech_language": (
        r"\btranscript\b"
        r"|\btranscripts\b"
        r"|\btext\b"
        r"|\btweets\b"
        r"|\btweet\b"
        r"|\bques\w*"
        r"|\bq-chat\b"
        r"|\bqchat\b"
        r"|\bq chat\b"
        r"|\bspeech\b"
        r"|\baudio-visual data\b"
        r"|\baudiovisual data\b"
        r"|\becholalia\b"
        r"|\bados\b"
        r"|\bcars\b"
        r"|\bdemographic\b"
        r"|\blanguage\b"
    ),
    "social_interaction": (
        r"\binteraction\b"
        r"|\binteractions\b"
        r"|\binteract\b"
        r"|\binteracting\b"
        r"|\bgesture\b"
        r"|\bgestures\b"
        r"|\bsocial\b"
        r"|\bvocalization\b"
        r"|\bvocalizations\b"
        r"|\bcommunication\b"
        r"|\bengaging\b"
    ),
    "joint_attention": (
        r"\bjoint attention\b"
        r"|\bjointattention\b"
    ),
    "video_analysis_data": (
        r"\bvideo\b"
        r"|\bvideos\b"
        r"|\bvideo frame\b"
        r"|\bvideo frames\b"
        r"|\baudio-visual data\b"
        r"|\baudiovisual data\b"
    ),
    "decision_making": (
        r"\bdecision making\b"
        r"|\bdecision-making\b"
    ),
    "sensor_data": (
        r"\bsensor\b"
        r"|\bsensors\b"
    ),
    "other_movement_data": (
        r"\binertial\b"
        r"|\bkinematic\b"
        r"|\bkinematics\b"
        r"|\bgrasp\b"
        r"|\bgrasping\b"
        r"|\bpose\b"
        r"|\bposes\b"
        r"|\bangle\b"
        r"|\bangles\b"
        r"|\bfine-motor\b"
        r"|\bfine motor\b"
        r"|\bmotor abnormalities\b"
        r"|\bgait\b"
        r"|\brotation\b"
    ),
    "other_gaze_data": (
        r"\beye-gaze\b"
        r"|\beye gaze\b"
        r"|\bscan-path\b"
        r"|\bscan-paths\b"
        r"|\bscanpath\b"
        r"|\bscanpaths\b"
        r"|\bsaccade\b"
        r"|\bsaccades\b"
        r"|\beye-tracking\b"
        r"|\beye tracking\b"
    ),
    "eeg": r"\beeg\b",
}


def build_other_behavioral_match_table(df: pd.DataFrame, valid_mask, col_indices: Sequence[int]) -> pd.DataFrame:
    """Build reusable match table for RQ3 other behavioral / other data categories."""
    valid_mask = ensure_series_mask(valid_mask, df.index)
    combined_text = (
        df.loc[valid_mask, df.columns[list(col_indices)]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .apply(normalize_text)
    )

    match_table = pd.DataFrame(index=combined_text.index)
    match_table["combined_other_behavioral_text"] = combined_text

    for category, pattern in OTHER_BEHAVIORAL_CATEGORY_PATTERNS.items():
        match_table[category] = combined_text.str.contains(pattern, regex=True, na=False)

    category_cols = list(OTHER_BEHAVIORAL_CATEGORY_PATTERNS.keys())
    match_table["any_category_matched"] = match_table[category_cols].any(axis=1)
    return match_table


def summarize_other_behavioral_match_table(match_table: pd.DataFrame, total_valid: Optional[int] = None) -> pd.DataFrame:
    if total_valid is None:
        total_valid = int(len(match_table))

    rows = []
    for category in OTHER_BEHAVIORAL_CATEGORY_PATTERNS:
        count = int(match_table[category].sum())
        rows.append({
            "Category": category,
            "Count": count,
            "Total Valid Papers": int(total_valid),
            "Percentage": safe_pct(count, total_valid),
        })
    return pd.DataFrame(rows)


def compute_other_behavioral_keywords(df: pd.DataFrame, valid_mask, col_indices: Sequence[int], output_dir: Optional[str] = None):
    print("\n============= Other Behavioral / Modality Keyword Identification =============")
    valid_mask = ensure_series_mask(valid_mask, df.index)
    total_valid = int(valid_mask.sum())
    match_table = build_other_behavioral_match_table(df, valid_mask, col_indices)
    summary_df = summarize_other_behavioral_match_table(match_table, total_valid)

    unmatched_rows = match_table.loc[
        ~match_table["any_category_matched"]
        & match_table["combined_other_behavioral_text"].ne("")
    ].copy()

    print(summary_df.to_string(index=False))
    if unmatched_rows.empty:
        print("No unmatched non-empty rows.")
    else:
        print("\n============= Unmatched Non-Empty Other Behavioral Rows =============")
        print(unmatched_rows[["combined_other_behavioral_text"]].to_string())

    save_df_optional(summary_df, "other_behavioral_keyword_summary.csv", output_dir=output_dir)
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "other_behavioral_keyword_match_table.csv", output_dir=output_dir)
    save_df_optional(unmatched_rows.reset_index().rename(columns={"index": "row_index"}), "other_behavioral_unmatched_rows.csv", output_dir=output_dir)

    return summary_df, match_table, unmatched_rows


# ============================================================
# 2. STUDY SETTING
# ============================================================

NOT_REPORTED_PATTERN = (
    r"^\s*$"
    r"|^\s*-$"
    r"|^\s*--$"
    r"|^\s*na\s*$"
    r"|^\s*n/a\s*$"
    r"|^\s*n\.a\s*$"
    r"|^\s*n/d\s*$"
    r"|^\s*nd\s*$"
    r"|^\s*n\.d\s*$"
    r"|^\s*nan\s*$"
    r"|^\s*not specified\s*$"
    r"|^\s*not reported\s*$"
    r"|^\s*not given\s*$"
    r"|^\s*none\s*$"
    r"|^\s*no\s*$"
)

CONTROLLED_SETTING_PATTERN = (
    r"\bclinic\w*\b"
    r"|\bclinical\b"
    r"|\bcontrolled\b"
    r"|\blab\w*\b"
    r"|\blaboratory\b"
    r"|\bhospital\w*\b"
    r"|\bschool\w*\b"
    r"|\bcenter\w*\b"
    r"|\bcentre\w*\b"
    r"|\buniversity\b"
    r"|\bresearch facility\b"
    r"|\bvr\b"
    r"|\bvirtual reality\b"
    r"|\bvirtual[- ]?reality\b"
    r"|\bvirtual environment\w*\b"
    r"|\bimmersive environment\w*\b"
    r"|\bkindergarden\b"
)

UNCONTROLLED_SETTING_PATTERN = (
    r"\bremote\b"
    r"|\buncontrolled\b"
    r"|\bonline\b"
    r"|\bhome\w*\b"
    r"|\bnaturalistic\b"
    r"|\bin[- ]?the[- ]?wild\b"
    r"|\breal[- ]?world\b"
    r"|\bhouse\w*\b"
)

EXPLICIT_BOTH_SETTING_PATTERN = (
    r"\bboth\b"
    r"|\bclinic.*home\b"
    r"|\bhome.*clinic\b"
    r"|\blab.*home\b"
    r"|\bhome.*lab\b"
    r"|\bcontrolled.*uncontrolled\b"
    r"|\buncontrolled.*controlled\b"
    r"|\bclinic.*online\b"
    r"|\bonline.*clinic\b"
    r"|\blab.*online\b"
    r"|\bonline.*lab\b"
    r"|\bmulti-site\b"
)

STUDY_SETTING_COLS = [
    "controlled_setting",
    "uncontrolled_naturalistic_remote",
    "both_controlled_and_uncontrolled",
    "not_reported",
    "unclear",
]


def build_study_setting_match_table(col: pd.Series, valid_mask) -> pd.DataFrame:
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])

    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["setting_text"] = col_filtered

    not_reported_mask = col_filtered.str.contains(NOT_REPORTED_PATTERN, regex=True, na=False)
    controlled_mask = ~not_reported_mask & col_filtered.str.contains(CONTROLLED_SETTING_PATTERN, regex=True, na=False)
    uncontrolled_mask = ~not_reported_mask & col_filtered.str.contains(UNCONTROLLED_SETTING_PATTERN, regex=True, na=False)
    explicit_both_mask = ~not_reported_mask & col_filtered.str.contains(EXPLICIT_BOTH_SETTING_PATTERN, regex=True, na=False)

    both_mask = explicit_both_mask | (controlled_mask & uncontrolled_mask)

    match_table["not_reported"] = not_reported_mask
    match_table["both_controlled_and_uncontrolled"] = both_mask
    match_table["controlled_setting"] = controlled_mask & ~both_mask
    match_table["uncontrolled_naturalistic_remote"] = uncontrolled_mask & ~both_mask
    match_table["unclear"] = ~not_reported_mask & ~both_mask & ~controlled_mask & ~uncontrolled_mask

    return match_table


def compute_study_setting(col, valid_mask, output_dir: Optional[str] = None):
    print("\n============= Study Setting =============")

    valid_mask = ensure_series_mask(valid_mask, col.index)
    total_valid = int(valid_mask.sum())
    match_table = build_study_setting_match_table(col, valid_mask)

    counts = {category: int(match_table[category].sum()) for category in STUDY_SETTING_COLS}
    summary_df = count_percent_rows(counts, total_valid, "Study Setting Category")

    print(summary_df.to_string(index=False))
    print("\n============= TOTAL CHECK =============")
    print("Total valid papers:", total_valid)
    print("Sum of mutually exclusive categories:", sum(counts.values()))

    unclear_rows = match_table.loc[match_table["unclear"]].copy()
    if not unclear_rows.empty:
        print("\n============= Unclear Study Setting Rows =============")
        print(unclear_rows[["setting_text"]].to_string())

    save_df_optional(summary_df, "study_setting_summary.csv", output_dir=output_dir)
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "study_setting_match_table.csv", output_dir=output_dir)

    return summary_df, match_table


# ============================================================
# 3. TASK TYPE / TASK DESIGN
# ============================================================

TASK_TYPE_PATTERNS = {
    "gaze_visual_attention_task": (
        r"\bjoint[- ]?attention\b"
        r"|\bjointattention\b"
        r"|\bgaze\b"
        r"|\beye[- ]?tracking\b"
        r"|\bsaccade\w*\b"
        r"|\bscan[- ]?path\w*\b"
        r"|\bscanpath\w*\b"
        r"|\bwatch\w*\b"
        r"|\bvideo\w*\b"
        r"|\bmovie\w*\b"
        r"|\bmovie clip\w*\b"
        r"|\bpicture\w*\b"
        r"|\bscene\w*\b"
        r"|\blook\w*\b"
        r"|\bview\w*\b"
        r"|\bobserve\w*\b"
        r"|\bbrows\w*\b"
        r"|\bweb[- ]?search\w*\b"
        r"|\bwebsite\w*\b"
        r"|\bface viewing\b"
        r"|\bvisual attention\b"
        r"|\bvisual exploration\b"
        r"|\bobserving images\b"
        r"|\bimage classification\b"
    ),
    "motor_movement_task": (
        r"\bplay\w*\b"
        r"|\btoy\w*\b"
        r"|\bwalk\w*\b"
        r"|\bgait\b"
        r"|\bmove\w*\b"
        r"|\bmovement\w*\b"
        r"|\bstand\w*\b"
        r"|\bstood\b"
        r"|\breach\w*\b"
        r"|\bgrasp\w*\b"
        r"|\bpose\w*\b"
        r"|\bgesture\w*\b"
        r"|\bimitation\b"
        r"|\bimitat\w*\b"
        r"|\bmotor\b"
        r"|\bdrag\b"
    ),
    "language_speech_audio_task": (
        r"\brepl\w*\b"
        r"|\bspeak\w*\b"
        r"|\bspoke\b"
        r"|\blisten\w*\b"
        r"|\bdiscuss\w*\b"
        r"|\bconversation\w*\b"
        r"|\baudio\w*\b"
        r"|\bsound\w*\b"
        r"|\bspeech\b"
        r"|\bdialog\w*\b"
        r"|\bread\w*\b"
        r"|\binterview\w*\b"
        r"|\bvocal\w*\b"
        r"|\bvoice\b"
    ),
    "questionnaire_survey_task": (
        r"\bquestionnaire\w*\b"
        r"|\bquestionnare\w*\b"
        r"|\bquesstionnaire\w*\b"
        r"|\bquesstionaire\w*\b"
        r"|\bques\w*"
        r"|\bsurvey\w*\b"
        r"|\bself[- ]?report\b"
        r"|\bparent[- ]?report\b"
        r"|\bcaregiver[- ]?report\b"
        r"|\brating scale\w*\b"
        r"|\bq-chat-10\b"
        r"|\bq-chat\b"
        r"|\behr\b"
    ),
    "facial_emotion_expression_task": (
        r"\bfacial emotion\w*\b"
        r"|\bfacial expression\w*\b"
        r"|\bemotion recognition\b"
        r"|\bemotion identification\b"
        r"|\bidentify\w* emotion\w*\b"
        r"|\bidentifying emotion\w*\b"
        r"|\brecogniz\w* emotion\w*\b"
        r"|\bemotions? from facial image\w*\b"
        r"|\bfacial image\w*\b"
        r"|\bface recognition\b"
        r"|\bfaze recognition\b"
        r"|\bfacial affect\b"
        r"|\bimages of participants\b"
        r"|\bface\b"
    ),
    "social_interaction_task": (
        r"\binteract\w*\b"
        r"|\bsocial\b"
        r"|\brelation\w*\b"
        r"|\brobot\w*\b"
        r"|\bvirtual reality\b"
        r"|\bvirtualreality\b"
        r"|\bvr\b"
        r"|\bsocial interaction\b"
        r"|\bjoint activity\b"
        r"|\btweets\b"
        r"|\bcommunication\b"
        r"|\bresponse to name\b"
    ),
    "decision_making_cognitive_task": (
        r"\bdecision[- ]?making\b"
        r"|\bdecision making\b"
        r"|\bchoice task\b"
        r"|\bcognitive task\b"
        r"|\brisk task\b"
        r"|\breward task\b"
        r"|\breaction time\b"
    ),
    "clinical_observation_assessment_task": (
        r"\bados\b"
        r"|\bados[- ]?2\b"
        r"|\bclinical observation\b"
        r"|\bdiagnostic observation\b"
        r"|\bassessment task\b"
        r"|\bstructured assessment\b"
    ),
    "neurophysiology_neuroimaging_task": (
        r"\beeg\b"
        r"|\berp\b"
        r"|\bmri\b"
        r"|\bfmri\b"
        r"|\bpet\b"
        r"|\bemg\b"
        r"|\becg\b"
        r"|\bbrain activity\b"
        r"|\bbrain imaging\b"
        r"|\bneuroimaging\b"
        r"|\bphysiolog\w*\b"
        r"|\bbiosignal\w*\b"
        r"|\bresting[- ]?state\b"
        r"|\btask[- ]?state\b"
        r"|\belectrode\w*\b"
    ),
}

TASK_TYPE_COLS = list(TASK_TYPE_PATTERNS.keys())

NOT_GIVEN_TASK_PATTERN = (
    r"^\s*$"
    r"|^\s*-$"
    r"|^\s*--$"
    r"|^\s*not given\s*$"
    r"|^\s*not reported\s*$"
    r"|^\s*not specified\s*$"
    r"|^\s*none\s*$"
    r"|^\s*no\s*$"
    r"|^\s*na\s*$"
    r"|^\s*n/a\s*$"
    r"|^\s*n\.a\s*$"
    r"|^\s*nd\s*$"
    r"|^\s*n/d\s*$"
    r"|^\s*n\.d\s*$"
    r"|^\s*nan\s*$"
)


def build_task_type_match_table(col: pd.Series, valid_mask) -> pd.DataFrame:
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])

    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["task_text"] = col_filtered

    not_given_mask = col_filtered.str.contains(NOT_GIVEN_TASK_PATTERN, regex=True, na=False)
    match_table["not_given"] = not_given_mask

    for category, pattern in TASK_TYPE_PATTERNS.items():
        match_table[category] = ~not_given_mask & col_filtered.str.contains(pattern, regex=True, na=False)

    match_table["task_category_count"] = match_table[TASK_TYPE_COLS].sum(axis=1)
    match_table["multiple_task_types"] = match_table["task_category_count"] >= 2
    match_table["unclear"] = ~match_table["not_given"] & (match_table["task_category_count"] == 0)

    return match_table


def compute_task_type(col, valid_mask, output_dir: Optional[str] = None):
    print("\n============= Task Type =============")

    valid_mask = ensure_series_mask(valid_mask, col.index)
    total_valid = int(valid_mask.sum())
    match_table = build_task_type_match_table(col, valid_mask)

    counts = {category: int(match_table[category].sum()) for category in TASK_TYPE_COLS}
    counts["not_given"] = int(match_table["not_given"].sum())
    counts["multiple_task_types"] = int(match_table["multiple_task_types"].sum())
    counts["unclear"] = int(match_table["unclear"].sum())

    summary_df = count_percent_rows(counts, total_valid, "Task Type Category")

    unclear_rows = match_table.loc[match_table["unclear"]].copy()

    print(summary_df.to_string(index=False))
    print("\nUnclear task type row numbers:", match_table.index[match_table["unclear"]].tolist())
    if not unclear_rows.empty:
        print("\n============= Unclear Task Type Rows =============")
        print(unclear_rows[["task_text"]].to_string())

    save_df_optional(summary_df, "task_type_summary.csv", output_dir=output_dir)
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "task_type_match_table.csv", output_dir=output_dir)
    save_df_optional(unclear_rows.reset_index().rename(columns={"index": "row_index"}), "task_type_unclear_rows.csv", output_dir=output_dir)

    return summary_df, match_table, unclear_rows


def make_exclusive_task_group(match_table: pd.DataFrame) -> pd.Series:
    """
    Optional exclusive task grouping. Use only for a sensitivity/compact table;
    primary task summaries should remain non-exclusive.
    """
    def classify(row):
        if row.get("multiple_task_types", False):
            return "multiple_task_types"
        for col in TASK_TYPE_COLS:
            if bool(row.get(col, False)):
                return col
        if bool(row.get("not_given", False)):
            return "not_given"
        return "unclear"

    return match_table.apply(classify, axis=1)


# ============================================================
# 4. ASD AGE RANGE PARSING AND AGE-GROUP MATCH TABLE
# ============================================================

AGE_CATEGORIES = {
    "Infants": (0, 1),
    "Toddlers": (1, 3),
    "Pre-schoolers": (3, 6),
    "Grade-schoolers": (6, 12),
    "Teens": (12, 18),
    "Adults": (18, float("inf")),
}

AGE_CATEGORY_COLS = list(AGE_CATEGORIES.keys())
MAX_REASONABLE_AGE_YEARS = 120


def extract_age_numbers(text) -> List[float]:
    return [float(n) for n in re.findall(r"\d+\.?\d*", str(text))]


def contains_months(text) -> bool:
    text = normalize_text(text)
    return "month" in text or "months" in text


def convert_if_months(age_min: float, age_max: float, text) -> Tuple[float, float]:
    if contains_months(text):
        return age_min / 12, age_max / 12
    return age_min, age_max


def is_reasonable_age_range(age_min: float, age_max: float) -> bool:
    """Prevent Excel date serials or impossible ages from being counted."""
    if pd.isna(age_min) or pd.isna(age_max):
        return False
    if age_min < 0:
        return False
    if age_max > MAX_REASONABLE_AGE_YEARS:
        return False
    return True


def parse_numeric_age_value(cell) -> float:
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


def parse_multiple_age_ranges(cell) -> List[Tuple[float, float]]:
    text = normalize_text(cell)

    dash_matches = re.findall(r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)", text)
    to_matches = re.findall(r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)", text)

    ranges = []
    for start, end in dash_matches + to_matches:
        age_min = float(start)
        age_max = float(end)
        age_min, age_max = convert_if_months(age_min, age_max, text)
        age_min, age_max = min(age_min, age_max), max(age_min, age_max)
        if is_reasonable_age_range(age_min, age_max):
            ranges.append((age_min, age_max))

    return ranges


def parse_under_over_age(cell) -> List[Tuple[float, float]]:
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


def parse_mean_sd_age_from_text(cell) -> List[Tuple[float, float]]:
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


def parse_regular_age(cell) -> List[Tuple[float, float]]:
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


def parse_age_range_cell(cell) -> Tuple[List[Tuple[float, float]], str]:
    """Parse age range cell only. Returns parsed_ranges, parser_type."""
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


def parse_mean_sd_columns(mean_cell, sd_cell) -> Tuple[List[Tuple[float, float]], str]:
    """
    Uses mean and SD columns as fallback when age range is missing/unparseable.
    This is an estimated age span, not a true reported range.
    """
    mean = parse_numeric_age_value(mean_cell)
    sd = parse_numeric_age_value(sd_cell)

    if pd.isna(mean):
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


def parse_age_with_fallback(range_cell, mean_cell=None, sd_cell=None) -> Tuple[List[Tuple[float, float]], str]:
    """First tries age range; if that fails, uses mean/SD columns if provided."""
    parsed_ranges, parser_type = parse_age_range_cell(range_cell)
    if parsed_ranges:
        return parsed_ranges, parser_type

    if mean_cell is not None or sd_cell is not None:
        fallback_ranges, fallback_type = parse_mean_sd_columns(mean_cell, sd_cell)
        if fallback_ranges:
            return fallback_ranges, fallback_type

    return [], parser_type


def age_range_overlaps_category(age_min: float, age_max: float, low: float, high: float) -> bool:
    """Half-open age bins: [low, high), except Adults = [18, infinity)."""
    if high == float("inf"):
        return age_max >= low
    return age_max >= low and age_min < high


def categories_for_age_ranges(parsed_ranges: Sequence[Tuple[float, float]], categories: Dict[str, Tuple[float, float]] = AGE_CATEGORIES) -> List[str]:
    matched = set()
    for age_min, age_max in parsed_ranges:
        for category, (low, high) in categories.items():
            if age_range_overlaps_category(age_min, age_max, low, high):
                matched.add(category)
    return sorted(matched, key=list(categories.keys()).index)


def build_asd_age_match_table(
    df_subset: pd.DataFrame,
    range_col_index: int,
    mean_col_index: int,
    sd_col_index: int,
    valid_asd_mask,
    categories: Dict[str, Tuple[float, float]] = AGE_CATEGORIES,
) -> pd.DataFrame:
    """Build reusable ASD age-group match table for descriptive and performance analyses."""
    valid_asd_mask = ensure_series_mask(valid_asd_mask, df_subset.index)

    rows = []
    for idx, row in df_subset.loc[valid_asd_mask].iterrows():
        range_cell = row.iloc[range_col_index]
        mean_cell = row.iloc[mean_col_index]
        sd_cell = row.iloc[sd_col_index]

        parsed_ranges, parser_type = parse_age_with_fallback(range_cell, mean_cell, sd_cell)
        matched_categories = categories_for_age_ranges(parsed_ranges, categories) if parsed_ranges else []

        out = {
            "row_index": idx,
            "raw_asd_age_range": range_cell,
            "raw_asd_mean_age": mean_cell,
            "raw_asd_sd_age": sd_cell,
            "parser_type": parser_type,
            "parsed_ranges_years": parsed_ranges,
            "matched_age_categories": "; ".join(matched_categories),
            "not_given": len(parsed_ranges) == 0,
            "manual_review_needed": (len(parsed_ranges) == 0 and not (is_invalid(range_cell) and is_invalid(mean_cell) and is_invalid(sd_cell))) or parser_type.startswith("estimated"),
        }

        for category in categories:
            out[category] = category in matched_categories

        out["age_category_count"] = len(matched_categories)
        out["multiple_age_groups"] = len(matched_categories) >= 2
        rows.append(out)

    match_table = pd.DataFrame(rows)
    if match_table.empty:
        return pd.DataFrame(columns=["row_index", *categories.keys(), "not_given", "manual_review_needed"]).set_index("row_index")

    return match_table.set_index("row_index")


def summarize_asd_age_match_table(
    age_match_table: pd.DataFrame,
    categories: Dict[str, Tuple[float, float]] = AGE_CATEGORIES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize ASD age groups from a prebuilt ASD age match table."""
    total_valid = int(len(age_match_table))
    counts = {category: int(age_match_table[category].sum()) for category in categories}
    counts["Not given"] = int(age_match_table["not_given"].sum())
    counts["multiple_age_groups"] = int(age_match_table["multiple_age_groups"].sum()) if "multiple_age_groups" in age_match_table.columns else 0

    summary_df = count_percent_rows(counts, total_valid, "ASD Age Group")

    parser_usage = (
        age_match_table["parser_type"]
        .value_counts(dropna=False)
        .rename_axis("parser_type")
        .reset_index(name="Count")
    )
    parser_usage["Total Valid Papers"] = total_valid
    parser_usage["Percentage"] = parser_usage["Count"].apply(lambda x: safe_pct(x, total_valid))

    return summary_df, parser_usage


def compute_asd_age_ranges(
    df_subset: pd.DataFrame,
    range_col_index: int,
    mean_col_index: int,
    sd_col_index: int,
    valid_asd_mask,
    output_dir: Optional[str] = None,
):
    """ASD-only age range summary. Use this for RQ4.5 age subgroup analyses."""
    print("\n============= AGE RANGE: ASD =============")

    age_match_table = build_asd_age_match_table(
        df_subset=df_subset,
        range_col_index=range_col_index,
        mean_col_index=mean_col_index,
        sd_col_index=sd_col_index,
        valid_asd_mask=valid_asd_mask,
    )

    summary_df, parser_usage_df = summarize_asd_age_match_table(age_match_table)
    manual_review_df = age_match_table[age_match_table["manual_review_needed"] == True].copy()

    print("\n=== ASD AGE CATEGORY COUNTS ===")
    print(summary_df.to_string(index=False))
    print("\n=== ASD AGE PARSER USAGE ===")
    print(parser_usage_df.to_string(index=False))
    print("\nManual-review ASD age rows:", len(manual_review_df))

    save_df_optional(summary_df, "asd_age_summary.csv", output_dir=output_dir)
    save_df_optional(parser_usage_df, "asd_age_parser_usage.csv", output_dir=output_dir)
    save_df_optional(age_match_table.reset_index(), "asd_age_match_table.csv", output_dir=output_dir)
    save_df_optional(manual_review_df.reset_index(), "asd_age_manual_review.csv", output_dir=output_dir)

    return summary_df, parser_usage_df, age_match_table, manual_review_df


# ============================================================
# 5. READY-TO-USE RQ4.5 PERFORMANCE SUMMARIES
# ============================================================

def accuracy_by_behavioral_modality(modality_match_table: pd.DataFrame, accuracy_series: pd.Series, output_dir: Optional[str] = None) -> pd.DataFrame:
    return summarize_accuracy_by_flags(
        modality_match_table,
        accuracy_series,
        flag_cols=["gaze_yes", "motor_yes", "speech_yes"],
        category_col="Behavioral modality",
        save_prefix="rq4_accuracy_by_behavioral_modality",
        output_dir=output_dir,
    )


def accuracy_by_study_setting(study_setting_match_table: pd.DataFrame, accuracy_series: pd.Series, output_dir: Optional[str] = None) -> pd.DataFrame:
    return summarize_accuracy_by_flags(
        study_setting_match_table,
        accuracy_series,
        flag_cols=STUDY_SETTING_COLS,
        category_col="Study setting",
        save_prefix="rq4_accuracy_by_study_setting",
        output_dir=output_dir,
    )


def accuracy_by_task_type(task_type_match_table: pd.DataFrame, accuracy_series: pd.Series, output_dir: Optional[str] = None) -> pd.DataFrame:
    flag_cols = TASK_TYPE_COLS + ["multiple_task_types", "not_given", "unclear"]
    return summarize_accuracy_by_flags(
        task_type_match_table,
        accuracy_series,
        flag_cols=flag_cols,
        category_col="Task type",
        save_prefix="rq4_accuracy_by_task_type_nonexclusive",
        output_dir=output_dir,
    )


def accuracy_by_asd_age_group(asd_age_match_table: pd.DataFrame, accuracy_series: pd.Series, output_dir: Optional[str] = None) -> pd.DataFrame:
    # Use ASD rows only because asd_age_match_table is indexed only to valid ASD rows.
    return summarize_accuracy_by_flags(
        asd_age_match_table,
        accuracy_series,
        flag_cols=AGE_CATEGORY_COLS + ["not_given", "multiple_age_groups"],
        category_col="ASD age group",
        save_prefix="rq4_accuracy_by_asd_age_group",
        output_dir=output_dir,
    )

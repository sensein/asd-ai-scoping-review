import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

import columns as COL
from analysis_common import RQ_QUESTIONS, RQ_TITLES
from codebook import NOT_GIVEN_TASK_PATTERN, TASK_TYPE_PATTERNS
from setup_data_ import load_annotation_data, INVALID_VALUES
from helper_functions_ import (
    clean_text_series as shared_clean_text_series,
    is_no as shared_is_no,
    is_yes as shared_is_yes,
)

RQ_NUMBER = 2
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
    OUTPUT_DIR = OUTPUT_ROOT / "rq2_results"

    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    invalid_values = INVALID_VALUES

    valid_papers_Total = data["valid_papers_Total"]
    valid_papers_ASD = data["valid_papers_ASD"]
    valid_papers_Neur = data["valid_papers_Neur"]
    valid_papers_Other = data["valid_papers_Other"]


    # ============================================================
    # 2. COLUMN MAP
    # ============================================================
    # Python iloc indices are zero-based.
    #
    # A  / 0   = Title
    # E  / 4   = Publication year
    # K  / 10  = Limitation/source text column 1, if present in coding sheet
    # L  / 11  = Limitation/source text column 2, if present in coding sheet
    # BH / 59  = Study setting
    # BI / 60  = Study goal
    # BJ / 61  = New dataset / data collected for study
    # BK / 62  = Dataset name / dataset source
    # BL / 63  = Data open-source access
    # BN / 65  = Data collection time frame / longitudinal status
    # BO / 66  = Data collection tool
    # BP / 67  = Participant task / task type
    # BQ / 68  = Code open-source access
    # BR / 69  = Study limitations
    # BT / 71  = Main findings
    # BU / 72  = BIDS data structure
    # BV / 73  = Novel analysis pipeline
    # BW / 74  = Future research pipelines
    # BX / 75  = Sensitive data / type of sensitive data
    # BY / 76  = Measures taken to protect sensitive data
    # CB / 79  = Included / final inclusion flag, if present

    COL_TITLE = COL.TITLE
    COL_PUBLICATION_YEAR = COL.PUBLICATION_YEAR
    COL_LIMITATION_SOURCE_1 = 10
    COL_LIMITATION_SOURCE_2 = 11

    COL_STUDY_SETTING = COL.STUDY_SETTING
    COL_STUDY_GOAL = COL.STUDY_GOAL
    COL_NEW_DATASET = COL.NEW_DATASET
    COL_DATASET_NAME = COL.DATASET_NAME
    COL_DATA_OPEN_SOURCE = COL.DATA_OPEN_SOURCE
    COL_LONGITUDINAL_TIME = COL.LONGITUDINAL_TIME
    COL_DATA_COLLECTION_TOOL = COL.DATA_COLLECTION_TOOL
    COL_PARTICIPANT_TASK = COL.PARTICIPANT_TASK
    COL_CODE_OPEN_SOURCE = COL.CODE_OPEN_SOURCE
    COL_STUDY_LIMITATIONS = COL.STUDY_LIMITATIONS
    COL_MAIN_FINDINGS = COL.MAIN_FINDINGS
    COL_BIDS_STRUCTURE = COL.BIDS_STRUCTURE
    COL_NOVEL_ANALYSIS_PIPELINE = COL.NOVEL_ANALYSIS_PIPELINE
    COL_FUTURE_RESEARCH_PIPELINES = COL.FUTURE_RESEARCH_PIPELINES
    COL_SENSITIVE_DATA = COL.SENSITIVE_DATA
    COL_SENSITIVE_DATA_PROTECTION = COL.SENSITIVE_DATA_PROTECTION
    COL_INCLUDED = None  # outside A:BY and intentionally excluded from cleaned scripts

    COLUMN_MAP = pd.DataFrame([
        {"Variable": "Title", "Excel Column": "A", "Python iloc Index": COL_TITLE},
        {"Variable": "Publication year", "Excel Column": "E", "Python iloc Index": COL_PUBLICATION_YEAR},
        {"Variable": "Limitation/source text 1", "Excel Column": "K", "Python iloc Index": COL_LIMITATION_SOURCE_1},
        {"Variable": "Limitation/source text 2", "Excel Column": "L", "Python iloc Index": COL_LIMITATION_SOURCE_2},
        {"Variable": "Study setting", "Excel Column": "BH", "Python iloc Index": COL_STUDY_SETTING},
        {"Variable": "Study goal", "Excel Column": "BI", "Python iloc Index": COL_STUDY_GOAL},
        {"Variable": "New dataset / data collected", "Excel Column": "BJ", "Python iloc Index": COL_NEW_DATASET},
        {"Variable": "Dataset name / source", "Excel Column": "BK", "Python iloc Index": COL_DATASET_NAME},
        {"Variable": "Data open-source access", "Excel Column": "BL", "Python iloc Index": COL_DATA_OPEN_SOURCE},
        {"Variable": "Longitudinal / time frame", "Excel Column": "BN", "Python iloc Index": COL_LONGITUDINAL_TIME},
        {"Variable": "Data collection tool", "Excel Column": "BO", "Python iloc Index": COL_DATA_COLLECTION_TOOL},
        {"Variable": "Participant task", "Excel Column": "BP", "Python iloc Index": COL_PARTICIPANT_TASK},
        {"Variable": "Code open-source access", "Excel Column": "BQ", "Python iloc Index": COL_CODE_OPEN_SOURCE},
        {"Variable": "Study limitations", "Excel Column": "BR", "Python iloc Index": COL_STUDY_LIMITATIONS},
        {"Variable": "Main findings", "Excel Column": "BT", "Python iloc Index": COL_MAIN_FINDINGS},
        {"Variable": "BIDS data structure", "Excel Column": "BU", "Python iloc Index": COL_BIDS_STRUCTURE},
        {"Variable": "Novel analysis pipeline", "Excel Column": "BV", "Python iloc Index": COL_NOVEL_ANALYSIS_PIPELINE},
        {"Variable": "Future research pipelines", "Excel Column": "BW", "Python iloc Index": COL_FUTURE_RESEARCH_PIPELINES},
        {"Variable": "Sensitive data / type", "Excel Column": "BX", "Python iloc Index": COL_SENSITIVE_DATA},
        {"Variable": "Sensitive data protection measures", "Excel Column": "BY", "Python iloc Index": COL_SENSITIVE_DATA_PROTECTION},
    ])

    print("\n=============COLUMN MAP=============")
    print(COLUMN_MAP.to_string(index=False))


    # ============================================================
    # 3. HELPERS
    # ============================================================


    def _prefixed_output_name(filename):
        prefix = "RQ2_"
        directory, basename = Path(filename).parent, Path(filename).name
        if basename.lower().startswith(prefix.lower()):
            basename = "RQ2" + basename[len("RQ2"):]
        else:
            basename = prefix + basename
        return str(directory / basename) if str(directory) != "." else basename

    def _write_csv_safely(df_to_save, path, index=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        df_to_save.to_csv(temp_path, index=index)
        temp_path.replace(path)


    def save_df_optional(df_to_save, filename, index=False):
        if SAVE_OUTPUTS:
            _write_csv_safely(df_to_save, OUTPUT_DIR / _prefixed_output_name(filename), index=index)


    def clean_text_series(series):
        return shared_clean_text_series(series)


    def pct(count, denominator):
        return round((count / denominator) * 100, 2) if denominator else 0


    def count_percent_rows(count_dict, total_valid, category_col="Category"):
        return pd.DataFrame([
            {
                category_col: category,
                "Count": int(count),
                "Total Valid Papers": int(total_valid),
                "Percentage": pct(count, total_valid),
            }
            for category, count in count_dict.items()
        ])


    def is_yes(value):
        return shared_is_yes(value)


    def is_no(value):
        return shared_is_no(value)


    save_df_optional(COLUMN_MAP, "data_column_map.csv")


    # ============================================================
    # 4. STUDY GOAL
    # ============================================================

    def compute_study_goals(col, valid_mask):
        print("\n=============Study Goal=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        patterns = {
            "prediction_of_outcome": r"\bprediction\b|\bpredict\w*\b|\bforecast\w*\b|\bpredicting\b|\boutcome prediction\b",
            "screening_detection": r"\bscreen\w*\b|\brecognition\b|\bdetection\b|\bdetect\w*\b|\bidentification\b|\bidentify\w*\b",
            "severity_detection": r"\bseverity\b|\bsevere\b|\bseverely\b|\bsymptom severity\b",
            "classification": r"\bclassif\w*\b|\bclassifier\b|\bdistinguish\w*\b|\bdifferentiat\w*\b",
            "diagnosis": r"\bdiagnos\w*\b|\bdiagnostic\b",
            "identifying_symptoms_biomarkers": r"\bindicator\w*\b|\bsymptom\w*\b|\binvestigat\w*\b|\bbiomarker\w*\b|\bmarker\w*\b",
            "other": r"\battention\b|\bstratification\b|\bintervention\b|\btreatment\b|\btherapy\b|\bfeasibility\b",
        }

        match_table = pd.DataFrame(index=col_filtered.index)
        match_table["study_goal_text"] = col_filtered

        counts = {}
        for category, pattern in patterns.items():
            mask = col_filtered.str.contains(pattern, regex=True, na=False)
            match_table[category] = mask
            counts[category] = int(mask.sum())
            print(f"\n{category}:")
            print("Count:", counts[category])
            print("Percentage:", pct(counts[category], total_valid))

        match_table["any_goal_category_matched"] = match_table[list(patterns.keys())].any(axis=1)
        match_table["unclear"] = ~match_table["any_goal_category_matched"]

        unclear_rows = match_table.loc[match_table["unclear"]].copy()
        print("\nunclear:")
        print("Count:", int(match_table["unclear"].sum()))
        print("Percentage:", pct(match_table["unclear"].sum(), total_valid))

        summary_df = count_percent_rows({**counts, "unclear": int(match_table["unclear"].sum())}, total_valid, "Study Goal Category")
        save_df_optional(summary_df, "study_goal_summary.csv")
        save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "study_goal_match_table.csv")
        save_df_optional(unclear_rows.reset_index().rename(columns={"index": "row_index"}), "study_goal_unclear_rows.csv")
        return summary_df, match_table, unclear_rows


    study_goal_summary, study_goal_match_table, study_goal_unclear_rows = compute_study_goals(
        df_subset.iloc[:, COL_STUDY_GOAL],
        valid_total,
    )


    # ============================================================
    # 5. STUDY SETTING
    # ============================================================
    def compute_study_setting(col, valid_mask):
        print("\n=============Study Setting=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        not_reported_pattern = (
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

        controlled_pattern = (
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

        uncontrolled_pattern = (
            r"\bremote\b"
            r"|\buncontrolled\b"
            r"|\bonline\b"
            r"|\bhome\w*\b"
            r"|\bnaturalistic\b"
            r"|\bin[- ]?the[- ]?wild\b"
            r"|\breal[- ]?world\b"
            r"|\bhouse\w*\b"

        )

        explicit_both_pattern = (
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
            r"|\bmulti-site\b"
            r"|\bonline.*lab\b"
        )

        match_table = pd.DataFrame(index=col_filtered.index)
        match_table["setting_text"] = col_filtered

        not_reported_mask = col_filtered.str.contains(not_reported_pattern, regex=True, na=False)

        controlled_mask = (
            ~not_reported_mask
            & col_filtered.str.contains(controlled_pattern, regex=True, na=False)
        )

        uncontrolled_mask = (
            ~not_reported_mask
            & col_filtered.str.contains(uncontrolled_pattern, regex=True, na=False)
        )

        explicit_both_mask = (
            ~not_reported_mask
            & col_filtered.str.contains(explicit_both_pattern, regex=True, na=False)
        )

        both_mask = explicit_both_mask | (controlled_mask & uncontrolled_mask)

        # Mutually exclusive final classification
        match_table["not_reported"] = not_reported_mask
        match_table["both_controlled_and_uncontrolled"] = both_mask
        match_table["controlled_setting"] = controlled_mask & ~both_mask
        match_table["uncontrolled_naturalistic_remote"] = uncontrolled_mask & ~both_mask
        match_table["unclear"] = (
            ~not_reported_mask
            & ~both_mask
            & ~controlled_mask
            & ~uncontrolled_mask
        )

        counts = {
            "controlled_setting": int(match_table["controlled_setting"].sum()),
            "uncontrolled_naturalistic_remote": int(match_table["uncontrolled_naturalistic_remote"].sum()),
            "both_controlled_and_uncontrolled": int(match_table["both_controlled_and_uncontrolled"].sum()),
            "not_reported": int(match_table["not_reported"].sum()),
            "unclear": int(match_table["unclear"].sum()),
        }

        for category, count in counts.items():
            print(f"\n{category}:")
            print("Count:", count)
            print("Percentage:", pct(count, total_valid))

        summary_df = count_percent_rows(
            counts,
            total_valid,
            "Study Setting Category"
        )

        print("\n=============Study Setting Summary Table=============")
        print(summary_df.to_string(index=False))

        print("\n=============TOTAL CHECK=============")
        print("Total valid papers:", total_valid)
        print("Sum of mutually exclusive categories:", sum(counts.values()))

        print("\n=============Unclear Study Setting Rows=============")
        unclear_rows = match_table.loc[match_table["unclear"]].copy()

        if unclear_rows.empty:
            print("No unclear study setting rows.")
        else:
            print(unclear_rows[["setting_text"]].to_string())

        save_df_optional(summary_df, "study_setting_summary.csv")
        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            "study_setting_match_table.csv"
        )

        return summary_df, match_table


    study_setting_summary, study_setting_match_table = compute_study_setting(
        df_subset.iloc[:, COL_STUDY_SETTING],
        valid_total,
    )

    # ============================================================
    # 6. DATASET TYPE / NEW DATASET
    # ============================================================

    def compute_dataset_type(col, valid_mask):
        print("\n=============Dataset Type / New Dataset=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        placeholder_mask = col_filtered.str.fullmatch(r"\s*|-+|n/a|na|nd|n/d|nan|none|not reported|unclear|unknown", na=False)

        new_mask = ~placeholder_mask & col_filtered.str.contains(r"^\s*yes\b|\bnew\b|\bcollected\b|\bcreated\b", regex=True, na=False)
        existing_mask = ~placeholder_mask & ~new_mask & col_filtered.str.contains(r"^\s*no\b|\bexisting\b|\bprevious\b|\bpreviously\b|\bearlier\b|\bpublic\b|\bopen dataset\b|\bsecondary\b", regex=True, na=False)
        unclear_mask = ~(new_mask | existing_mask | placeholder_mask)

        counts = {
            "new_dataset_or_primary_data_collection": int(new_mask.sum()),
            "existing_dataset_or_secondary_data": int(existing_mask.sum()),
            "not_reported_or_placeholder": int(placeholder_mask.sum()),
            "manual_review_unclear": int(unclear_mask.sum()),
        }

        for category, count in counts.items():
            print(f"\n{category}:")
            print("Count:", count)
            print("Percentage:", pct(count, total_valid))

        match_table = pd.DataFrame({
            "dataset_type_text": col_filtered,
            "new_dataset_or_primary_data_collection": new_mask,
            "existing_dataset_or_secondary_data": existing_mask,
            "not_reported_or_placeholder": placeholder_mask,
            "manual_review_unclear": unclear_mask,
        })

        summary_df = count_percent_rows(counts, total_valid, "Dataset Type Category")
        save_df_optional(summary_df, "dataset_type_summary.csv")
        save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "dataset_type_match_table.csv")
        return summary_df, match_table


    dataset_type_summary, dataset_type_match_table = compute_dataset_type(
        df_subset.iloc[:, COL_NEW_DATASET],
        valid_total,
    )

    # ============================================================
    # 7. TASK TYPE
    # Broad categories + finer task/protocol subcategories
    # ============================================================

    def compute_task_type(col, valid_mask):
        print("\n=============Task Type=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        # ============================================================
        # BROAD TASK CATEGORIES
        # ============================================================

        patterns = {**TASK_TYPE_PATTERNS, "not_given": NOT_GIVEN_TASK_PATTERN}

        # ============================================================
        # FINER TASK / PROTOCOL SUBCATEGORIES
        # Non-mutually exclusive. These are for richer RQ2.4 reporting.
        # They do NOT replace the broad categories above.
        # ============================================================

        subcategory_patterns = {
            "passive_visual_stimulus_viewing": (
                r"\bwatch\w*\b"
                r"|\bview\w*\b"
                r"|\bobser\w*\b"
                r"|\blook\w*\b"
                r"|\bvideo\w*\b"
                r"|\bmovie\w*\b"
                r"|\bmovie clip\w*\b"
                r"|\bimages?\b"
                r"|\bpictures?\b"
                r"|\bphotos?\b"
                r"|\bstimuli\b"
                r"|\bvisual stimuli\b"
                r"|\bnatural images?\b"
                r"|\bsocial and non[- ]?social\b"
                r"|\bsocial stimuli\b"
                r"|\bnon[- ]?social stimuli\b"
                r"|\bmonitor\b"
                r"|\bscreen\b"
                r"|\btablet\b"
            ),

            "active_gaze_visual_orienting_search_task": (
                r"\bvisual[- ]?orient\w*\b"
                r"|\borienting task\b"
                r"|\bguided task\w*\b"
                r"|\bsaccade\w*\b"
                r"|\bscan[- ]?path\w*\b"
                r"|\bscanpath\w*\b"
                r"|\bfixation\w*\b"
                r"|\beye[- ]?tracking\b"
                r"|\bgaze\b"
                r"|\bbrows\w*\b"
                r"|\bweb[- ]?search\w*\b"
                r"|\bsearch task\b"
                r"|\blocate specific\b"
                r"|\bvisual exploration\b"
                r"|\bvisual attention\b"
                r"|\bcenter bias\b"
                r"|\barea[s]? of interest\b"
                r"|\baoi\b"
            ),

            "joint_attention_social_cue_response_task": (
                r"\bjoint[- ]?attention\b"
                r"|\bjointattention\b"
                r"|\bresponse to name\b"
                r"|\bname call\b"
                r"|\bcalled up to\b"
                r"|\bija\b"
                r"|\brja\b"
                r"|\binitiation of joint attention\b"
                r"|\bresponse to joint attention\b"
                r"|\bsocial cue\w*\b"
                r"|\bvisual cue\w*\b"
                r"|\bspeech cue\w*\b"
                r"|\bmotion cue\w*\b"
                r"|\bfollow the cued direction\b"
                r"|\bsocial attention\b"
            ),

            "motor_gait_posture_kinematic_task": (
                r"\bwalk\w*\b"
                r"|\bgait\b"
                r"|\bpostur\w*\b"
                r"|\bstand\w*\b"
                r"|\bstood\b"
                r"|\bstanding\b"
                r"|\bbalance\b"
                r"|\bcenter of pressure\b"
                r"|\bcop\b"
                r"|\bforce plate\w*\b"
                r"|\breach\w*\b"
                r"|\bgrasp\w*\b"
                r"|\bdrop\b"
                r"|\bhand\w*\b"
                r"|\bupper[- ]?limb\b"
                r"|\bkinematic\w*\b"
                r"|\btrajectory\w*\b"
                r"|\bpose estimation\b"
                r"|\bbody movement\w*\b"
                r"|\bmotor pattern\w*\b"
                r"|\bfine motor\b"
                r"|\bgross motor\b"
                r"|\bdrag\b"
                r"|\bmove\w* card\w*\b"
                r"|\bmanipulate virtual objects\b"
            ),

            "imitation_task": (
                r"\bimitation\b"
                r"|\bimitat\w*\b"
                r"|\bmove forward\b"
                r"|\bmove backward\b"
                r"|\braise hands\b"
                r"|\bhands down\b"
                r"|\bdance[- ]?like\b"
                r"|\bpre[- ]?recorded sentences\b"
            ),

            "social_interaction_play_task": (
                r"\bfree play\b"
                r"|\bplay\b"
                r"|\bplaying\b"
                r"|\btoy\w*\b"
                r"|\bparent[- ]?child\b"
                r"|\bmother\b"
                r"|\bcaregiver\b"
                r"|\bclinician\b"
                r"|\binvestigator\b"
                r"|\bexperimenter\b"
                r"|\bconfederate\b"
                r"|\bface[- ]?to[- ]?face\b"
                r"|\bsocial interaction\b"
                r"|\binteract\w*\b"
                r"|\bnormal interactions\b"
                r"|\bstill[- ]?face\b"
                r"|\breunion episode\b"
                r"|\bjoint activity\b"
                r"|\bturn[- ]?taking\b"
                r"|\bcommunication action\b"
                r"|\bresponsive social smile\b"
                r"|\bcontextual assessment of social skills\b"
                r"|\bcass\b"
            ),

            "robot_virtual_game_app_task": (
                r"\brobot\w*\b"
                r"|\bnao\b"
                r"|\broboparrot\b"
                r"|\brobot[- ]?assisted\b"
                r"|\bhuman[- ]?robot\b"
                r"|\bvirtual reality\b"
                r"|\bvirtualreality\b"
                r"|\bvr\b"
                r"|\bvirtual environment\b"
                r"|\bvirtual agent\w*\b"
                r"|\bunity3d\b"
                r"|\bgame\w*\b"
                r"|\bapp\b"
                r"|\bguess what\b"
                r"|\bguesswhat\b"
                r"|\bcharades\b"
                r"|\bquiz game\b"
                r"|\bsmart tablet\b"
                r"|\btablet game\w*\b"
                r"|\btouch me\b"
                r"|\bdance with me\b"
                r"|\bmemory game\w*\b"
            ),


            "digital_trace_online_behavior_task": (
                r"\btweet\w*\b"
                r"|\bsocial media\b"
                r"|\bonline\b"
                r"|\bposting tweets\b"
                r"|\baac\b"
                r"|\bapp for communication\b"
                r"|\bcommunication app\b"
                r"|\busage pattern\w*\b"
                r"|\bweb[- ]?based\b"
                r"|\bwebsites?\b"
            ),

        }

        # ============================================================
        # MATCH TABLE
        # ============================================================

        match_table = pd.DataFrame(index=col_filtered.index)
        match_table["task_text"] = col_filtered

        # First identify not-given rows
        not_given_mask = col_filtered.str.contains(
            patterns["not_given"],
            regex=True,
            na=False
        )

        match_table["not_given"] = not_given_mask

        counts = {}

        # ============================================================
        # BROAD CATEGORY COUNTS
        # ============================================================

        task_categories = [
            category for category in patterns.keys()
            if category != "not_given"
        ]

        for category in task_categories:
            pattern = patterns[category]

            # Do not allow placeholder/not-given rows to also count as task categories
            mask = (
                ~not_given_mask
                & col_filtered.str.contains(pattern, regex=True, na=False)
            )

            match_table[category] = mask
            counts[category] = int(mask.sum())

            print(f"\n{category}:")
            print("Count:", counts[category])
            print("Percentage:", pct(counts[category], total_valid))

        counts["not_given"] = int(match_table["not_given"].sum())

        # Multiple broad task types should only consider real broad task categories
        match_table["task_category_count"] = match_table[task_categories].sum(axis=1)
        match_table["multiple_task_types"] = match_table["task_category_count"] >= 2

        # Unclear means: text is present and not placeholder, but it still did not match any broad task category
        match_table["unclear"] = (
            ~match_table["not_given"]
            & (match_table["task_category_count"] == 0)
        )

        counts["multiple_task_types"] = int(match_table["multiple_task_types"].sum())
        counts["unclear"] = int(match_table["unclear"].sum())

        # ============================================================
        # SUBCATEGORY COUNTS
        # ============================================================

        subcategory_counts = {}
        subcategory_records = []

        for subcategory, pattern in subcategory_patterns.items():
            col_name = f"subcat__{subcategory}"

            mask = (
                ~not_given_mask
                & col_filtered.str.contains(pattern, regex=True, na=False)
            )

            match_table[col_name] = mask
            count = int(mask.sum())
            subcategory_counts[subcategory] = count

            # Save a few example text entries for manual checking
            examples = (
                col_filtered[mask]
                .dropna()
                .drop_duplicates()
                .head(3)
                .tolist()
            )

            subcategory_records.append({
                "Task subcategory": subcategory,
                "n": count,
                "% of studies": pct(count, total_valid),
                "Example task text": " | ".join(examples)
            })

        subcategory_cols = [f"subcat__{x}" for x in subcategory_patterns.keys()]

        match_table["task_subcategory_count"] = match_table[subcategory_cols].sum(axis=1)
        match_table["multiple_task_subcategories"] = match_table["task_subcategory_count"] >= 2

        # Rows that matched at least one broad category but no finer subcategory
        match_table["broad_task_but_no_subcategory"] = (
            ~match_table["not_given"]
            & (match_table["task_category_count"] > 0)
            & (match_table["task_subcategory_count"] == 0)
        )

        # Rows that did not match broad task or subcategory, but were not blank/not-given
        match_table["unclear_after_subcategories"] = (
            ~match_table["not_given"]
            & (match_table["task_category_count"] == 0)
            & (match_table["task_subcategory_count"] == 0)
        )

        subcategory_summary_df = pd.DataFrame(subcategory_records)

        subcategory_summary_df = subcategory_summary_df.sort_values(
            by=["n", "Task subcategory"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # ============================================================
        # PRINT BROAD TASK RESULTS
        # ============================================================

        print("\nnot_given:")
        print("Count:", counts["not_given"])
        print("Percentage:", pct(counts["not_given"], total_valid))

        print("\nmultiple_task_types:")
        print("Count:", counts["multiple_task_types"])
        print("Percentage:", pct(counts["multiple_task_types"], total_valid))

        print("\nunclear:")
        print("Count:", counts["unclear"])
        print("Percentage:", pct(counts["unclear"], total_valid))
        print("Row numbers:", match_table.index[match_table["unclear"]].tolist())

        unclear_rows = match_table.loc[match_table["unclear"]].copy()

        print("\n=============Unclear Broad Task Type Rows=============")

        if unclear_rows.empty:
            print("No unclear broad task type rows.")
        else:
            print(unclear_rows[["task_text"]].to_string())

        summary_df = count_percent_rows(
            counts,
            total_valid,
            "Task Type Category"
        )

        print("\n=============Task Type Summary Table=============")
        print(summary_df.to_string(index=False))

        # ============================================================
        # PRINT SUBCATEGORY RESULTS
        # ============================================================

        print("\n=============Task Subcategory Summary Table=============")
        print(subcategory_summary_df.to_string(index=False))

        print("\n=============Most Common Task Subcategory=============")

        if subcategory_summary_df.empty:
            print("No task subcategories identified.")
        else:
            top_subcat = subcategory_summary_df.iloc[0]
            print("Subcategory:", top_subcat["Task subcategory"])
            print("Count:", top_subcat["n"])
            print("Percentage:", top_subcat["% of studies"])

        print("\nmultiple_task_subcategories:")
        print("Count:", int(match_table["multiple_task_subcategories"].sum()))
        print("Percentage:", pct(int(match_table["multiple_task_subcategories"].sum()), total_valid))

        print("\nbroad_task_but_no_subcategory:")
        broad_no_subcat_rows = match_table.loc[match_table["broad_task_but_no_subcategory"]].copy()
        print("Count:", int(match_table["broad_task_but_no_subcategory"].sum()))
        print("Percentage:", pct(int(match_table["broad_task_but_no_subcategory"].sum()), total_valid))
        print("Row numbers:", match_table.index[match_table["broad_task_but_no_subcategory"]].tolist())

        if not broad_no_subcat_rows.empty:
            print(broad_no_subcat_rows[["task_text"]].to_string())

        print("\nunclear_after_subcategories:")
        unclear_after_subcat_rows = match_table.loc[match_table["unclear_after_subcategories"]].copy()
        print("Count:", int(match_table["unclear_after_subcategories"].sum()))
        print("Percentage:", pct(int(match_table["unclear_after_subcategories"].sum()), total_valid))
        print("Row numbers:", match_table.index[match_table["unclear_after_subcategories"]].tolist())

        if not unclear_after_subcat_rows.empty:
            print(unclear_after_subcat_rows[["task_text"]].to_string())

        # ============================================================
        # SAVE OUTPUTS
        # ============================================================

        save_df_optional(summary_df, "task_type_summary.csv")
        save_df_optional(subcategory_summary_df, "task_type_subcategory_summary.csv")

        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            "task_type_match_table.csv"
        )

        save_df_optional(
            unclear_rows.reset_index().rename(columns={"index": "row_index"}),
            "task_type_unclear_rows.csv"
        )

        save_df_optional(
            broad_no_subcat_rows.reset_index().rename(columns={"index": "row_index"}),
            "task_type_broad_but_no_subcategory_rows.csv"
        )

        save_df_optional(
            unclear_after_subcat_rows.reset_index().rename(columns={"index": "row_index"}),
            "task_type_unclear_after_subcategories_rows.csv"
        )

        return (
            summary_df,
            subcategory_summary_df,
            match_table,
            unclear_rows,
            broad_no_subcat_rows,
            unclear_after_subcat_rows
        )


    task_type_summary, task_type_subcategory_summary, task_type_match_table, task_type_unclear_rows, task_type_broad_but_no_subcategory_rows, task_type_unclear_after_subcategories_rows = compute_task_type(
        df_subset.iloc[:, COL_PARTICIPANT_TASK],
        valid_total,
    )

    # Denominators for tool sub-analyses, based on broad task-type coding.
    gaze_count = int(task_type_match_table["gaze_visual_attention_task"].sum())
    motor_count = int(task_type_match_table["motor_movement_task"].sum())
    language_count = int(task_type_match_table["language_speech_audio_task"].sum())
    questionnaire_count = int(task_type_match_table["questionnaire_survey_task"].sum())
    facial_emotion_count = int(task_type_match_table["facial_emotion_expression_task"].sum())
    other_social_tasks_count = int(task_type_match_table["social_interaction_task"].sum())
    decision_making_count = int(task_type_match_table["decision_making_cognitive_task"].sum())
    clinical_observation_count = int(task_type_match_table["clinical_observation_assessment_task"].sum())
    neurophysiology_neuroimaging_count = int(task_type_match_table["neurophysiology_neuroimaging_task"].sum())
    not_given_task_count = int(task_type_match_table["not_given"].sum())
    unclear_task_count = int(task_type_match_table["unclear"].sum())

    # subcategory-level counts for reporting
    passive_visual_viewing_count = int(task_type_match_table["subcat__passive_visual_stimulus_viewing"].sum())
    active_gaze_visual_search_count = int(task_type_match_table["subcat__active_gaze_visual_orienting_search_task"].sum())
    joint_attention_social_cue_count = int(task_type_match_table["subcat__joint_attention_social_cue_response_task"].sum())
    motor_gait_posture_kinematic_count = int(task_type_match_table["subcat__motor_gait_posture_kinematic_task"].sum())
    imitation_task_count = int(task_type_match_table["subcat__imitation_task"].sum())
    social_interaction_play_count = int(task_type_match_table["subcat__social_interaction_play_task"].sum())
    robot_virtual_game_app_count = int(task_type_match_table["subcat__robot_virtual_game_app_task"].sum())
    digital_trace_online_behavior_count = int(task_type_match_table["subcat__digital_trace_online_behavior_task"].sum())

    # ============================================================
    # PRINT ALL TASK TYPE OUTPUTS CLEARLY
    # ============================================================

    def print_all_task_type_outputs(
        task_type_summary,
        task_type_subcategory_summary,
        task_type_match_table,
        total_valid
    ):
        print("\n\n============================================================")
        print("FULL TASK TYPE OUTPUTS")
        print("============================================================")

        # Make pandas print all rows/columns without truncation
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 120)

        # ------------------------------------------------------------
        # Broad task categories
        # ------------------------------------------------------------
        print("\n================ BROAD TASK TYPE CATEGORIES ================")

        broad_category_cols = [
            "gaze_visual_attention_task",
            "motor_movement_task",
            "language_speech_audio_task",
            "questionnaire_survey_task",
            "facial_emotion_expression_task",
            "social_interaction_task",
            "decision_making_cognitive_task",
            "clinical_observation_assessment_task",
            "neurophysiology_neuroimaging_task",
            "not_given",
            "multiple_task_types",
            "unclear",
        ]

        broad_rows = []

        for category in broad_category_cols:
            if category in task_type_match_table.columns:
                count = int(task_type_match_table[category].sum())
            elif category in task_type_summary.iloc[:, 0].astype(str).values:
                # fallback if category exists only in summary table
                count = int(
                    task_type_summary.loc[
                        task_type_summary.iloc[:, 0].astype(str) == category
                    ].iloc[0, 1]
                )
            else:
                count = 0

            broad_rows.append({
                "Category type": "Broad task category",
                "Category": category,
                "Count": count,
                "Total valid studies": total_valid,
                "Percentage": pct(count, total_valid),
            })

        broad_output_df = pd.DataFrame(broad_rows)

        print(broad_output_df.to_string(index=False))

        # ------------------------------------------------------------
        # Finer task/protocol subcategories
        # ------------------------------------------------------------
        print("\n================ TASK / PROTOCOL SUBCATEGORIES ================")

        subcategory_cols = [
            col for col in task_type_match_table.columns
            if col.startswith("subcat__")
        ]

        subcategory_rows = []

        for col in subcategory_cols:
            clean_name = col.replace("subcat__", "")
            count = int(task_type_match_table[col].sum())

            subcategory_rows.append({
                "Category type": "Task/protocol subcategory",
                "Category": clean_name,
                "Count": count,
                "Total valid studies": total_valid,
                "Percentage": pct(count, total_valid),
            })

        subcategory_output_df = pd.DataFrame(subcategory_rows).sort_values(
            by=["Count", "Category"],
            ascending=[False, True]
        ).reset_index(drop=True)

        print(subcategory_output_df.to_string(index=False))

        # ------------------------------------------------------------
        # Derived subcategory summary counts
        # ------------------------------------------------------------
        print("\n================ DERIVED TASK SUBCATEGORY COUNTS ================")

        derived_cols = [
            "task_category_count",
            "task_subcategory_count",
            "multiple_task_subcategories",
            "broad_task_but_no_subcategory",
            "unclear_after_subcategories",
        ]

        derived_rows = []

        for category in derived_cols:
            if category in task_type_match_table.columns:
                if task_type_match_table[category].dtype == bool:
                    count = int(task_type_match_table[category].sum())
                    percentage = pct(count, total_valid)
                else:
                    count = "numeric count column"
                    percentage = "not applicable"

                derived_rows.append({
                    "Derived variable": category,
                    "Count": count,
                    "Total valid studies": total_valid,
                    "Percentage": percentage,
                })

        derived_output_df = pd.DataFrame(derived_rows)

        print(derived_output_df.to_string(index=False))

        # ------------------------------------------------------------
        # Most common categories
        # ------------------------------------------------------------
        print("\n================ MOST COMMON TASK CATEGORIES ================")

        top_broad = broad_output_df[
            broad_output_df["Category"].isin([
                "gaze_visual_attention_task",
                "motor_movement_task",
                "language_speech_audio_task",
                "questionnaire_survey_task",
                "facial_emotion_expression_task",
                "social_interaction_task",
                "decision_making_cognitive_task",
                "clinical_observation_assessment_task",
                "neurophysiology_neuroimaging_task",
            ])
        ].sort_values(
            by="Count",
            ascending=False
        ).head(1)

        top_subcat = subcategory_output_df.sort_values(
            by="Count",
            ascending=False
        ).head(1)

        print("\nMost common broad task category:")
        print(top_broad.to_string(index=False))

        print("\nMost common task/protocol subcategory:")
        print(top_subcat.to_string(index=False))

        # ------------------------------------------------------------
        # Optional: save clean all-output tables
        # ------------------------------------------------------------
        save_df_optional(broad_output_df, "task_type_all_broad_counts.csv")
        save_df_optional(subcategory_output_df, "task_type_all_subcategory_counts.csv")
        save_df_optional(derived_output_df, "task_type_all_derived_counts.csv")

        return broad_output_df, subcategory_output_df, derived_output_df


    # ============================================================
    # RUN FULL TASK TYPE PRINT OUTPUTS
    # ============================================================

    task_type_all_broad_counts, task_type_all_subcategory_counts, task_type_all_derived_counts = print_all_task_type_outputs(
        task_type_summary=task_type_summary,
        task_type_subcategory_summary=task_type_subcategory_summary,
        task_type_match_table=task_type_match_table,
        total_valid=int(valid_total.sum())
    )

    # ============================================================
    # 8. DATA COLLECTION TIME FRAME
    # ============================================================

    def compute_time_frame(col, valid_mask):
        print("\n=============Time Frame=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        # ---------------- PATTERNS ---------------- #

        not_reported_pattern = (
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
            r"|^\s*not explicitly stated\s*$"
            r"|^\s*none\s*$"

        )

        cross_sectional_pattern = (
            r"\bcross[- ]?sectional\b"
            r"|\bcross sectional\b"
        )

        longitudinal_pattern = (
            r"\blongitudinal\b"
            r"|\bfollow[- ]?up\b"
            r"|\bover time\b"
            r"|\brepeated\b"
            r"|\bmultiple time\b"
            r"|\bmultiple session\w*\b"
            r"|\bmultiple visit\w*\b"
            r"|\b\d+\s*(day|days|week|weeks|month|months|year|years)\b"
        )

        single_time_pattern = (
            r"\bsingle session\b"
            r"|\bsingle-session\b"
            r"|\bsingle time\b"
            r"|\bsingle-time\b"
            r"|\bone session\b"
            r"|\bone-time\b"
            r"|\bone time\b"
            r"|\bnot longitudinal\b"
            r"|\bno longitudinal\b"
            r"|^\s*no\s*$"

        )

        # ---------------- MASKS ---------------- #

        not_reported_mask = col_filtered.str.contains(
            not_reported_pattern,
            regex=True,
            na=False
        )

        longitudinal_mask_raw = col_filtered.str.contains(
            longitudinal_pattern,
            regex=True,
            na=False
        )

        single_time_mask_raw = col_filtered.str.contains(
            single_time_pattern,
            regex=True,
            na=False
        )

        # ---------------- MUTUALLY EXCLUSIVE PRIORITY ---------------- #
        # Priority:
        # 1. not reported
        # 2. longitudinal/repeated time points
        # 3. single-session/not longitudinal
        # 4. unreported/unclear

        not_reported_final = not_reported_mask

        longitudinal_final = (
            ~not_reported_final
            & longitudinal_mask_raw
        )

        single_time_final = (
            ~not_reported_final
            & ~longitudinal_final
            & single_time_mask_raw
        )

        unclear_final = (
            ~not_reported_final
            & ~longitudinal_final
            & ~single_time_final
        )

        counts = {
            "longitudinal_or_repeated_time_points": int(longitudinal_final.sum()),
            "not_longitudinal_or_single_time_point": int(single_time_final.sum()),
            "unreported_or_unclear": int((not_reported_final | unclear_final).sum()),
        }

        for category, count in counts.items():
            print(f"\n{category}:")
            print("Count:", count)
            print("Percentage:", pct(count, total_valid))

        match_table = pd.DataFrame({
            "time_frame_text": col_filtered,
            "longitudinal_or_repeated_time_points": longitudinal_final,
            "not_longitudinal_or_single_time_point": single_time_final,
            "not_reported": not_reported_final,
            "unclear_non_placeholder": unclear_final,
            "unreported_or_unclear": not_reported_final | unclear_final,
        })


        print("\n=============Unclear Non-Placeholder Rows=============")
        unclear_rows = match_table.loc[match_table["unclear_non_placeholder"]]

        if unclear_rows.empty:
            print("No unclear non-placeholder rows.")
        else:
            print(unclear_rows[["time_frame_text"]].to_string())

        summary_df = count_percent_rows(
            counts,
            total_valid,
            "Time Frame Category"
        )

        print("\n=============Time Frame Summary Table=============")
        print(summary_df.to_string(index=False))

        print("\n=============TOTAL CHECK=============")
        print("Total valid papers:", total_valid)
        print("Sum of mutually exclusive categories:", sum(counts.values()))

        save_df_optional(summary_df, "time_frame_summary.csv")
        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            "time_frame_match_table.csv"
        )

        return summary_df, match_table


    time_frame_summary, time_frame_match_table = compute_time_frame(
        df_subset.iloc[:, COL_LONGITUDINAL_TIME],
        valid_total,
    )
    # ============================================================
    # 8B. EXTRACT LONGITUDINAL TIME DURATIONS
    # ============================================================

    def extract_duration_to_days(text):
        """
        Extracts time-duration values from one cell and converts them to days.

        Handles:
        - single values: "6 weeks", "11 years", "3 months", "14 days"
        - ranges: "6-9 weeks", "6 to 9 weeks", "6–9 weeks"
        - common typos: daya, mont, uyear, yrs, yr

        For ranges, uses the midpoint:
        - "6-9 weeks" -> 7.5 weeks -> 52.5 days
        """

        original_text = str(text)
        text = original_text.lower().strip()

        # ---------------- PLACEHOLDERS ---------------- #

        placeholder_pattern = (
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
            r"|^\s*none\s*$"
            r"|^\s*no\s*$"
            r"|^\s*not reported\s*$"
            r"|^\s*not specified\s*$"
            r"|^\s*not given\s*$"
        )

        if re.search(placeholder_pattern, text):
            return {
                "duration_days": np.nan,
                "extracted_value": np.nan,
                "extracted_unit": "",
                "extraction_type": "not_extractable",
                "matched_text": ""
            }

        # ---------------- NORMALIZE COMMON TYPOS ---------------- #

        text = text.replace("daya", "days")
        text = text.replace("monts", "months")
        text = text.replace("mont", "month")
        text = text.replace("monthes", "months")
        text = text.replace("uyears", "years")
        text = text.replace("uyear", "year")
        text = text.replace("yrs", "years")
        text = text.replace("yr", "year")

        # normalize dash types
        text = text.replace("–", "-").replace("—", "-")

        unit_to_days = {
            "day": 1,
            "days": 1,
            "week": 7,
            "weeks": 7,
            "month": 30.5,
            "months": 30.5,
            "year": 365.25,
            "years": 365.25,
        }

        unit_pattern = r"(day|days|week|weeks|month|months|year|years)"

        # ---------------- RANGE PATTERN ---------------- #
        # Examples:
        # 6-9 weeks
        # 6 to 9 weeks
        # 6 and 9 weeks

        range_pattern = (
            rf"(\d+(?:\.\d+)?)\s*"
            rf"(?:-|to|and)\s*"
            rf"(\d+(?:\.\d+)?)\s*"
            rf"{unit_pattern}\b"
        )

        range_match = re.search(range_pattern, text)

        if range_match:
            lower = float(range_match.group(1))
            upper = float(range_match.group(2))
            unit = range_match.group(3)

            midpoint = (lower + upper) / 2
            duration_days = midpoint * unit_to_days[unit]

            return {
                "duration_days": duration_days,
                "extracted_value": midpoint,
                "extracted_unit": unit,
                "extraction_type": "range_midpoint",
                "matched_text": range_match.group(0)
            }

        # ---------------- SINGLE VALUE PATTERN ---------------- #
        # Examples:
        # 6 weeks
        # 11 years
        # 3 months
        # 14 days

        single_pattern = rf"(\d+(?:\.\d+)?)\s*{unit_pattern}\b"

        single_match = re.search(single_pattern, text)

        if single_match:
            value = float(single_match.group(1))
            unit = single_match.group(2)
            duration_days = value * unit_to_days[unit]

            return {
                "duration_days": duration_days,
                "extracted_value": value,
                "extracted_unit": unit,
                "extraction_type": "single_value",
                "matched_text": single_match.group(0)
            }

        return {
            "duration_days": np.nan,
            "extracted_value": np.nan,
            "extracted_unit": "",
            "extraction_type": "not_extractable",
            "matched_text": ""
        }


    def summarize_longitudinal_time_durations(col, valid_mask, time_frame_match_table=None):
        print("\n=============Longitudinal Duration Extraction=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        extraction_rows = []

        for idx, text in col_filtered.items():
            extracted = extract_duration_to_days(text)

            row = {
                "row_index": idx,
                "time_frame_text": text,
                "matched_text": extracted["matched_text"],
                "extracted_value": extracted["extracted_value"],
                "extracted_unit": extracted["extracted_unit"],
                "duration_days": extracted["duration_days"],
                "extraction_type": extracted["extraction_type"]
            }

            extraction_rows.append(row)

        extraction_df = pd.DataFrame(extraction_rows).set_index("row_index")

        # ------------------------------------------------------------
        # OPTIONAL: attach classification from compute_time_frame()
        # ------------------------------------------------------------

        if time_frame_match_table is not None:
            shared_index = extraction_df.index.intersection(time_frame_match_table.index)

            time_cols_to_attach = [
                "longitudinal_or_repeated_time_points",
                "not_longitudinal_or_single_time_point",
                "cross_sectional",
                "not_reported",
                "unclear_non_placeholder",
                "unreported_or_unclear"
            ]

            for col_name in time_cols_to_attach:
                if col_name in time_frame_match_table.columns:
                    extraction_df.loc[shared_index, col_name] = time_frame_match_table.loc[shared_index, col_name]

        # ------------------------------------------------------------
        # Extract rows with real duration values
        # ------------------------------------------------------------

        extracted_duration_rows = extraction_df[
            extraction_df["duration_days"].notna()
        ].copy()

        n_extractable = len(extracted_duration_rows)

        # ------------------------------------------------------------
        # Summary statistics
        # ------------------------------------------------------------

        if n_extractable > 0:
            duration_days = extracted_duration_rows["duration_days"]

            raw_summary = pd.DataFrame([{
                "Total Valid Papers": total_valid,
                "Rows with Extractable Duration": n_extractable,
                "Mean Duration Days": round(duration_days.mean(), 2),
                "Median Duration Days": round(duration_days.median(), 2),
                "SD Duration Days": round(duration_days.std(ddof=1), 2) if n_extractable > 1 else np.nan,
                "Minimum Duration Days": round(duration_days.min(), 2),
                "Maximum Duration Days": round(duration_days.max(), 2),
                "Q1 Duration Days": round(duration_days.quantile(0.25), 2),
                "Q3 Duration Days": round(duration_days.quantile(0.75), 2),
                "IQR Duration Days": round(duration_days.quantile(0.75) - duration_days.quantile(0.25), 2),
                "Mean Duration Weeks": round(duration_days.mean() / 7, 2),
                "Median Duration Weeks": round(duration_days.median() / 7, 2),
                "Mean Duration Months": round(duration_days.mean() / 30.5, 2),
                "Median Duration Months": round(duration_days.median() / 30.5, 2),
                "Mean Duration Years": round(duration_days.mean() / 365.25, 2),
                "Median Duration Years": round(duration_days.median() / 365.25, 2),
            }])

        else:
            raw_summary = pd.DataFrame([{
                "Total Valid Papers": total_valid,
                "Rows with Extractable Duration": 0,
                "Mean Duration Days": np.nan,
                "Median Duration Days": np.nan,
                "SD Duration Days": np.nan,
                "Minimum Duration Days": np.nan,
                "Maximum Duration Days": np.nan,
                "Q1 Duration Days": np.nan,
                "Q3 Duration Days": np.nan,
                "IQR Duration Days": np.nan,
                "Mean Duration Weeks": np.nan,
                "Median Duration Weeks": np.nan,
                "Mean Duration Months": np.nan,
                "Median Duration Months": np.nan,
                "Mean Duration Years": np.nan,
                "Median Duration Years": np.nan,
            }])

        # ------------------------------------------------------------
        # Readable summary table
        # ------------------------------------------------------------

        readable_summary = pd.DataFrame([
            {
                "Metric": "Total valid papers",
                "Value": int(raw_summary.loc[0, "Total Valid Papers"])
            },
            {
                "Metric": "Rows with extractable duration",
                "Value": int(raw_summary.loc[0, "Rows with Extractable Duration"])
            },
            {
                "Metric": "Mean duration",
                "Value": (
                    f'{raw_summary.loc[0, "Mean Duration Days"]} days '
                    f'({raw_summary.loc[0, "Mean Duration Weeks"]} weeks; '
                    f'{raw_summary.loc[0, "Mean Duration Months"]} months; '
                    f'{raw_summary.loc[0, "Mean Duration Years"]} years)'
                )
            },
            {
                "Metric": "Median duration",
                "Value": (
                    f'{raw_summary.loc[0, "Median Duration Days"]} days '
                    f'({raw_summary.loc[0, "Median Duration Weeks"]} weeks; '
                    f'{raw_summary.loc[0, "Median Duration Months"]} months; '
                    f'{raw_summary.loc[0, "Median Duration Years"]} years)'
                )
            },
            {
                "Metric": "Standard deviation",
                "Value": f'{raw_summary.loc[0, "SD Duration Days"]} days'
            },
            {
                "Metric": "Minimum duration",
                "Value": f'{raw_summary.loc[0, "Minimum Duration Days"]} days'
            },
            {
                "Metric": "Maximum duration",
                "Value": f'{raw_summary.loc[0, "Maximum Duration Days"]} days'
            },
            {
                "Metric": "Q1 duration",
                "Value": f'{raw_summary.loc[0, "Q1 Duration Days"]} days'
            },
            {
                "Metric": "Q3 duration",
                "Value": f'{raw_summary.loc[0, "Q3 Duration Days"]} days'
            },
            {
                "Metric": "IQR duration",
                "Value": f'{raw_summary.loc[0, "IQR Duration Days"]} days'
            },
        ])

        # ------------------------------------------------------------
        # Print clean outputs only
        # ------------------------------------------------------------

        print("\n=============Extracted Longitudinal Duration Rows=============")

        if extracted_duration_rows.empty:
            print("No extractable duration rows found.")
        else:
            extracted_display = extracted_duration_rows[
                [
                    "time_frame_text",
                    "matched_text",
                    "extracted_value",
                    "extracted_unit",
                    "duration_days",
                    "extraction_type"
                ]
            ].copy()

            extracted_display["duration_days"] = extracted_display["duration_days"].round(2)

            print(extracted_display.to_string())

        print("\n=============Longitudinal Duration Summary=============")
        print(readable_summary.to_string(index=False))

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------

        save_df_optional(raw_summary, "longitudinal_duration_summary.csv")
        save_df_optional(readable_summary, "longitudinal_duration_summary_readable.csv")
        save_df_optional(
            extraction_df.reset_index(),
            "longitudinal_duration_extraction_match_table.csv"
        )
        save_df_optional(
            extracted_duration_rows.reset_index(),
            "longitudinal_duration_extracted_rows.csv"
        )

        return raw_summary, readable_summary, extraction_df, extracted_duration_rows


    longitudinal_duration_summary, longitudinal_duration_summary_readable, longitudinal_duration_extraction_table, longitudinal_duration_extracted_rows = summarize_longitudinal_time_durations(
        df_subset.iloc[:, COL_LONGITUDINAL_TIME],
        valid_total,
        time_frame_match_table=time_frame_match_table
    )
    # ============================================================
    # 9. DATA COLLECTION TOOLS
    # ============================================================

    def data_collection_tool_summary(col, valid_mask, patterns, denominator, label):
        print(f"\n============={label} Tool Summary=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())
        denominator = int(denominator)

        safe_label = label.lower().replace(" ", "_").replace("/", "_")

        match_table = pd.DataFrame(index=col_filtered.index)
        match_table[f"{safe_label}_tool_text"] = col_filtered

        summary_rows = []
        tool_cols = []

        # ---------------- NOT GIVEN / UNCLEAR TOOL COLUMN ---------------- #

        not_given_unclear_pattern = (
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
            r"|^\s*none\s*$"
            r"|^\s*no\s*$"
            r"|^\s*not reported\s*$"
            r"|^\s*not specified\s*$"
            r"|^\s*not given\s*$"
            r"|^\s*unclear\s*$"
            r"|^\s*unknown\s*$"
        )

        not_given_unclear_mask = col_filtered.str.contains(
            not_given_unclear_pattern,
            regex=True,
            na=False
        )

        match_table[f"{safe_label}_tool_column_not_given_or_unclear"] = not_given_unclear_mask

        # ---------------- INDIVIDUAL TOOL COUNTS ---------------- #
        # IMPORTANT: this preserves your original logic.
        # It counts across all valid papers, not only rows marked as that task type.

        for tool_name, pattern in patterns.items():
            mask = col_filtered.str.contains(pattern, regex=True, na=False)

            match_table[tool_name] = mask
            tool_cols.append(tool_name)

            count = int(mask.sum())
            percent_denom = pct(count, denominator) if denominator else 0
            percent_total = pct(count, total_valid)

            summary_rows.append({
                "Tool Group": label,
                "Tool Category": tool_name,
                "Count": count,
                "Relevant Task Papers": denominator,
                "Percentage of Relevant Task Papers": percent_denom,
                "Total Valid Papers": total_valid,
                "Percentage of Total Papers": percent_total,
            })

            print(f"\n{tool_name}:")
            print("Count:", count)
            print("Percentage of relevant task papers:", percent_denom)
            print("Percentage of total valid papers:", percent_total)

        # ---------------- ANY NAMED TOOL IN MAJOR GROUP ---------------- #
        # This gives unique papers with any named tool in this group.
        # This avoids summing tools and double-counting papers with multiple tools.

        if tool_cols:
            match_table[f"any_named_{safe_label}_tool"] = match_table[tool_cols].sum(axis=1) >= 1
        else:
            match_table[f"any_named_{safe_label}_tool"] = False

        any_tool_count = int(match_table[f"any_named_{safe_label}_tool"].sum())
        any_tool_percent_denom = pct(any_tool_count, denominator) if denominator else 0
        any_tool_percent_total = pct(any_tool_count, total_valid)

        print(f"\nany_named_{safe_label}_tool:")
        print("Count:", any_tool_count)
        print("Percentage of relevant task papers:", any_tool_percent_denom)
        print("Percentage of total valid papers:", any_tool_percent_total)
        print("Row numbers:", match_table.index[match_table[f"any_named_{safe_label}_tool"]].tolist())

        summary_rows.append({
            "Tool Group": label,
            "Tool Category": f"any_named_{safe_label}_tool",
            "Count": any_tool_count,
            "Relevant Task Papers": denominator,
            "Percentage of Relevant Task Papers": any_tool_percent_denom,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": any_tool_percent_total,
        })

        # ---------------- TOOL COLUMN NOT GIVEN / UNCLEAR ---------------- #

        not_given_unclear_count = int(not_given_unclear_mask.sum())
        not_given_unclear_percent_total = pct(not_given_unclear_count, total_valid)

        print(f"\n{safe_label}_tool_column_not_given_or_unclear:")
        print("Count:", not_given_unclear_count)
        print("Percentage of total valid papers:", not_given_unclear_percent_total)

        summary_rows.append({
            "Tool Group": label,
            "Tool Category": f"{safe_label}_tool_column_not_given_or_unclear",
            "Count": not_given_unclear_count,
            "Relevant Task Papers": denominator,
            "Percentage of Relevant Task Papers": "",
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": not_given_unclear_percent_total,
        })

        # ---------------- SUMMARY TABLE ---------------- #

        summary_df = pd.DataFrame(summary_rows)

        print(f"\n============={label} Tool Summary Table=============")
        print(summary_df.to_string(index=False))

        save_df_optional(summary_df, f"{safe_label}_tool_summary.csv")
        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            f"{safe_label}_tool_match_table.csv"
        )

        return summary_df, match_table


    # ---------------- EXACT SAME TOOL PATTERNS AS BEFORE ---------------- #

    eye_tracking_patterns = {
        "tobii": r"\btobii\b",
        "smi": r"\bsmi\b|\bsensomotoric\b",
        "nao_robot": r"\bnao\b",
        "gazefinder": r"\bgazefinder\b",
        "eyelink": r"\beyelink\b|\beye\s*link\b",
        "gazepoint": r"\bgazepoint\b",
        "scieye": r"\bscieye\b",
        "xmi": r"\bxmi\b",
    }

    motor_tool_patterns = {
        "force_plates": r"\bforce plate\w*\b|\bplates\b",
        "kinect": r"\bkinect\b",
        "openpose": r"\bopenpose\b|\bopen pose\b",
        "openface": r"\bopenface\b|\bopen face\b",
        "accelerometer": r"\baccelerometer\w*\b|\bimu\b|\binertial measurement\b",
    }

    neuroimaging_tool_patterns = {
        "eeg_electrodes_or_sensor_net": r"\belectrode\w*\b|\bhydrocel\b|\bsensor net\b|\bgeodesic\b",
        "wearable_eeg_system": r"\bemotiv\b",
        "mri_scanner": r"\bmri\b|\bfmri\b|\bscanner\b",
    }

    video_audio_tool_patterns = {
        "camera_or_webcam": r"\bcamera\w*\b|\bwebcam\w*\b|\bvideo camera\b",
        "microphone_or_audio_recorder": r"\bmic\b|\bmics\b|\bmicrophone\w*\b|\baudio recorder\b|\brecorder\b",
    }


    # ---------------- RUN TOOL SUMMARIES ---------------- #
    # This preserves original denominators and original counting behavior.

    eye_tracking_tools_summary, eye_tracking_tools_match_table = data_collection_tool_summary(
        df_subset.iloc[:, COL_DATA_COLLECTION_TOOL],
        valid_total,
        eye_tracking_patterns,
        gaze_count,
        "Eye Tracking"
    )

    motor_tools_summary, motor_tools_match_table = data_collection_tool_summary(
        df_subset.iloc[:, COL_DATA_COLLECTION_TOOL],
        valid_total,
        motor_tool_patterns,
        motor_count,
        "Motor"
    )

    neuroimaging_tools_summary, neuroimaging_tools_match_table = data_collection_tool_summary(
        df_subset.iloc[:, COL_DATA_COLLECTION_TOOL],
        valid_total,
        neuroimaging_tool_patterns,
        neurophysiology_neuroimaging_count,
        "Neuroimaging"
    )

    # ---------------- VIDEO/AUDIO TOOL DENOMINATOR ---------------- #
    # Use unique papers, not task-category occurrences.
    # A paper is counted once if it has ANY of these task types:
    # social interaction OR motor/movement OR language/speech/audio.

    video_audio_task_mask = (
        task_type_match_table["social_interaction_task"]
        | task_type_match_table["motor_movement_task"]
        | task_type_match_table["language_speech_audio_task"]
    )

    video_audio_denominator = int(video_audio_task_mask.sum())

    print("\n=============Video/Audio Denominator Check=============")
    print("Social interaction task papers:", int(task_type_match_table["social_interaction_task"].sum()))
    print("Motor/movement task papers:", int(task_type_match_table["motor_movement_task"].sum()))
    print("Language/speech/audio task papers:", int(task_type_match_table["language_speech_audio_task"].sum()))
    print("Unique video/audio-relevant papers:", video_audio_denominator)
    print("Rows included:", task_type_match_table.index[video_audio_task_mask].tolist())

    video_audio_tools_summary, video_audio_tools_match_table = data_collection_tool_summary(
        df_subset.iloc[:, COL_DATA_COLLECTION_TOOL],
        valid_total,
        video_audio_tool_patterns,
        video_audio_denominator,
        "Video Audio"
    )



    # ---------------- MAJOR TOOL GROUP SUMMARY ---------------- #
    # This gives one clean summary row per major tool group.

    total_valid = int(valid_total.sum()) if hasattr(valid_total, "sum") else int(valid_total)

    major_tool_group_summary = pd.DataFrame([
        {
            "Major Tool Group": "Eye Tracking",
            "Relevant Task Papers": gaze_count,
            "Any Named Tool Count": int(eye_tracking_tools_match_table["any_named_eye_tracking_tool"].sum()),
            "Percentage of Relevant Task Papers": pct(
                int(eye_tracking_tools_match_table["any_named_eye_tracking_tool"].sum()),
                gaze_count
            ) if gaze_count else 0,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(
                int(eye_tracking_tools_match_table["any_named_eye_tracking_tool"].sum()),
                total_valid
            ),
            "Tool Column Not Given/Unclear": int(
                eye_tracking_tools_match_table["eye_tracking_tool_column_not_given_or_unclear"].sum()
            ),
        },
        {
            "Major Tool Group": "Motor",
            "Relevant Task Papers": motor_count,
            "Any Named Tool Count": int(motor_tools_match_table["any_named_motor_tool"].sum()),
            "Percentage of Relevant Task Papers": pct(
                int(motor_tools_match_table["any_named_motor_tool"].sum()),
                motor_count
            ) if motor_count else 0,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(
                int(motor_tools_match_table["any_named_motor_tool"].sum()),
                total_valid
            ),
            "Tool Column Not Given/Unclear": int(
                motor_tools_match_table["motor_tool_column_not_given_or_unclear"].sum()
            ),
        },
        {
            "Major Tool Group": "Neuroimaging",
            "Relevant Task Papers": neurophysiology_neuroimaging_count,
            "Any Named Tool Count": int(neuroimaging_tools_match_table["any_named_neuroimaging_tool"].sum()),
            "Percentage of Relevant Task Papers": pct(
                int(neuroimaging_tools_match_table["any_named_neuroimaging_tool"].sum()),
                neurophysiology_neuroimaging_count
            ) if neurophysiology_neuroimaging_count else 0,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(
                int(neuroimaging_tools_match_table["any_named_neuroimaging_tool"].sum()),
                total_valid
            ),
            "Tool Column Not Given/Unclear": int(
                neuroimaging_tools_match_table["neuroimaging_tool_column_not_given_or_unclear"].sum()
            ),
        },
        {
            "Major Tool Group": "Video Audio",
            "Relevant Task Papers": video_audio_denominator,
            "Any Named Tool Count": int(video_audio_tools_match_table["any_named_video_audio_tool"].sum()),
            "Percentage of Relevant Task Papers": pct(
                int(video_audio_tools_match_table["any_named_video_audio_tool"].sum()),
                video_audio_denominator
            ) if video_audio_denominator else 0,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(
                int(video_audio_tools_match_table["any_named_video_audio_tool"].sum()),
                total_valid
            ),
            "Tool Column Not Given/Unclear": int(
                video_audio_tools_match_table["video_audio_tool_column_not_given_or_unclear"].sum()
            ),
        },
    ])

    print("\n=============Major Tool Group Summary=============")
    print(major_tool_group_summary.to_string(index=False))

    save_df_optional(major_tool_group_summary, "major_tool_group_summary.csv")
    #===============DATA and CODE ACCESS+============#

    def open_source_access_summary(col, valid_mask, label="Data"):
        print(f"\n============={label} Availability / Access=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())
        safe_label = label.lower().replace(" ", "_").replace("/", "_")

        # ---------------- SIMPLE CATEGORY RULES ---------------- #
        # yes = only "yes" / misspelled "yest" OR public repository/code terms
        # no = only "no"
        # request = contains request
        # limited_or_pseudocode = contains limited/minimal/pseudocode/partial-only language
        # placeholder = empty / missing / unclear
        # manual = anything else non-empty

        yes_mask = (
            col_filtered.str.fullmatch(r"\s*yes\s*", na=False)
            | col_filtered.str.fullmatch(r"\s*yest\s*", na=False)
            | col_filtered.str.contains(
                r"\bkaggle\b"
                r"|\bpython\b"
                r"|\bgithub\b"
                r"|\bscripts?\b",
                regex=True,
                na=False
            )
        )

        no_mask = col_filtered.str.contains(
        r"\bno\b|confidential|actually\s+no",
        case=False,
        na=False,
        regex=True
    )

        placeholder_mask = col_filtered.str.fullmatch(
            r"\s*"
            r"|-+"
            r"|n/d"
            r"|nd"
            r"|n/a"
            r"|na"
            r"|nan"
            r"|none"
            r"|not reported"
            r"|not given"
            r"|not specified"
            r"|unclear"
            r"|unknown",
            na=False
        )

        request_mask = (
            ~yes_mask
            & ~no_mask
            & ~placeholder_mask
            & col_filtered.str.contains(
                r"\brequest\w*\b",
                regex=True,
                na=False
            )
        )

        limited_or_pseudocode_mask = (
            ~yes_mask
            & ~no_mask
            & ~placeholder_mask
            & ~request_mask
            & col_filtered.str.contains(
                r"\blimit\w*\b"
                r"|\bminimal\b"
                r"|\bonly\b"
                r"|\bjust\b"
                r"|\bfeatures?\b"
                r"|\bopenface features?\b"
                r"|\bnot the original\b"
                r"|\bnot original\b"
                r"|\boriginal videos?\b"
                r"|\braw coordinates?\b"
                r"|\bpseudo[- ]?code\b"
                r"|\bpseudocode\b",
                regex=True,
                na=False
            )
        )

        manual_review_mask = ~(
            yes_mask
            | no_mask
            | placeholder_mask
            | request_mask
            | limited_or_pseudocode_mask
        )

              # ---------------- OVERLAP CHECK ---------------- #
        # Checks whether any row matched more than one category.
        # Do NOT include manual_review_mask here because it is defined as
        # the inverse of all other categories.

        overlap_check_df = pd.DataFrame({
            f"{safe_label}_availability_text": col_filtered,
            "yes_publicly_available": yes_mask,
            "no_not_publicly_available": no_mask,
            "available_on_request": request_mask,
            "limited_or_pseudocode": limited_or_pseudocode_mask,
            "not_reported_or_placeholder": placeholder_mask,
        })

        category_cols_for_overlap = [
            "yes_publicly_available",
            "no_not_publicly_available",
            "available_on_request",
            "limited_or_pseudocode",
            "not_reported_or_placeholder",
        ]

        overlap_check_df["category_match_count"] = overlap_check_df[
            category_cols_for_overlap
        ].sum(axis=1)

        overlap_rows = overlap_check_df[
            overlap_check_df["category_match_count"] > 1
        ].copy()

        print(f"\n============={label} Availability Rows in Multiple Categories=============")

        if overlap_rows.empty:
            print("No rows matched multiple categories.")
        else:
            overlap_rows.insert(0, "excel_row", overlap_rows.index + 3)

            overlap_rows["matched_categories"] = overlap_rows[
                category_cols_for_overlap
            ].apply(
                lambda row: ", ".join(row.index[row].tolist()),
                axis=1
            )

            print(
                overlap_rows[
                    [
                        "excel_row",
                        f"{safe_label}_availability_text",
                        "matched_categories",
                        "category_match_count",
                    ]
                ].to_string(index=False)
            )

        # ---------------- COUNTS ---------------- #

        counts = {
            "yes_publicly_available": int(yes_mask.sum()),
            "no_not_publicly_available": int(no_mask.sum()),
            "available_on_request": int(request_mask.sum()),
            "limited_or_pseudocode": int(limited_or_pseudocode_mask.sum()),
            "not_reported_or_placeholder": int(placeholder_mask.sum()),
            "manual_review_non_placeholder": int(manual_review_mask.sum()),
        }

        for category, count in counts.items():
            print(f"\n{category}:")
            print("Count:", count)
            print("Percentage:", pct(count, total_valid))

        # ---------------- MATCH TABLE ---------------- #


        match_table = pd.DataFrame({
            f"{safe_label}_availability_text": col_filtered,
            "yes_publicly_available": yes_mask,
            "no_not_publicly_available": no_mask,
            "available_on_request": request_mask,
            "limited_or_pseudocode": limited_or_pseudocode_mask,
            "not_reported_or_placeholder": placeholder_mask,
            "manual_review_non_placeholder": manual_review_mask,
        })

        manual_rows = match_table.loc[manual_review_mask].copy()

        print(f"\n============={label} Availability Rows Needing Manual Review=============")

        if manual_rows.empty:
            print("No rows need manual annotation.")
        else:
            print(manual_rows[[f"{safe_label}_availability_text"]].to_string())

        # ---------------- SUMMARY TABLE ---------------- #

        summary_df = count_percent_rows(
            counts,
            total_valid,
            f"{label} Availability Category"
        )

        print(f"\n============={label} Availability Summary Table=============")
        print(summary_df.to_string(index=False))

        save_df_optional(summary_df, f"{safe_label}_availability_summary.csv")
        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            f"{safe_label}_availability_match_table.csv"
        )
        save_df_optional(
            manual_rows.reset_index().rename(columns={"index": "row_index"}),
            f"{safe_label}_availability_manual_rows.csv"
        )

        return summary_df, manual_rows, match_table


    # ---------------- RUN DATA AVAILABILITY ---------------- #
    # BL / Python index 63 = Data open-source access

    data_availability_summary, data_availability_manual_rows, data_availability_match_table = open_source_access_summary(
        df_subset.iloc[:, COL_DATA_OPEN_SOURCE],
        valid_total,
        label="Data"
    )


    # ---------------- RUN CODE AVAILABILITY ---------------- #
    # BQ / Python index 68 = Code open-source access

    code_availability_summary, code_availability_manual_rows, code_availability_match_table = open_source_access_summary(
        df_subset.iloc[:, COL_CODE_OPEN_SOURCE],
        valid_total,
        label="Code"
    )


    # ============================================================
    # 11. LIMITATION CATEGORIES
    # ============================================================

    def rq_limitation_categories(df, valid_mask, col_indices):
        print("\n=============Limitation Categories=============")

        total_valid = int(valid_mask.sum())
        safe_col_indices = [c for c in col_indices if c < df.shape[1]]

        text = (
            df.loc[valid_mask, df.columns[safe_col_indices]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
            .str.strip()
        )

        category_patterns = {
            "diagnosis_limitations": r"ados|adi[- ]?r|adir|cars|clinical diagnos|standardi[sz]ed diagnostic|validated diagnostic|not confirmed|diagnostic tool|diagnos.*self.report|diagnos.*parent|diagnosis",
            "small_sample_size": r"small sample|sample size|limited sample|limited number|limited data|limited dataset|limited training dataset|few subject|few participant|larger dataset|larger sample|more data|small cohort|pilot analysis|small size|small dataset|dataset can be increased|limited public speech dataset| samples",
            "lack_of_demographic_diversity": r"cultur\w*|demographic|divers\w*|race|racial|ethnic|ethnicity|socioeconomic|socio-economic|language|linguistic|geographic|country|region|homogene\w*|selection bias|population bias",
            "lack_of_sex_gender_balance": r"\bsex\b|gender|female|females|girls|male|males|single gender|sex imbalance|gender imbalance|few females|predominantly male",
            "limited_age_generalizability": r"age group|age range|age variability|younger than|very young children|only adults|only adolescents|only children|children aged|developmental period|specific.*age|not representative.*developmental",
            "lack_of_iq_adaptive_behavior_measures": r"\biq\b|intelligence quotient|cognitive ability|cognition|adaptive behaviou?r|functional level|functioning|high[- ]?functioning|low[- ]?functioning|developmental level|not measured|participant information|participants.*unclear|participants.*not clear",
            "single_site_sample": r"single site|single-site|single cent(?:er|re)|one clinic|one school|one hospital|one university|single institution|one location|convenience sample|same school|same clinic",
            "lack_of_external_validation": r"external validation|independent data ?set|external data ?set|cross[- ]?corpus|cross[- ]?site|replication|overfitting|risk of overfitting|standard dataset",
            "missing_comparison_group": r"no control group|comparison group|control group|typically developing|td children|td group|neurotypical|non[- ]?asd|not matched|mismatched|groups were not matched",
            "measurement_tool_limitations": r"tool|device|instrument|eye tracker|eye-tracking system|camera|microphone|audio quality|speech-to-text|transcription|robot|\bvr\b|headset|not evaluated|not described|not specified",
            "task_paradigm_limitations": r"task|paradigm|stimuli|stimulus|images as stimuli|web-searching tasks|video scenarios|short duration|short time window|single images|structured setting|real-world|real world|not suitable|not ideal|formal clinical assessments|false information",
            "child_task_feasibility_limitations": r"attention|young children|shy.*children|didn.*interact|bored|fatigue|discomfort|headset|immersive environment|rejection of vr|tolerat|sensor|eye tracker.*affect",
            "model_interpretability_limitations": r"interpret|explain|black[- ]?box|which particular aspects|identify which|features.*crucial|feature importance|unclear.*model|conclusive explanation|weak evaluation|minimal evaluation",
            "data_quality_missing_data_limitations": r"unbalanced|imbalanced|class imbalance|missing data|data quality|quality limitations|non[- ]?stationary|variability in data|individual variability|camera motion|poor accuracy|false negatives|data availability|lack of quality|demographic data|age|info|balancing dataset|balancing datasets|own data|new data|independent data",
            "experimental_setup_limitations": r"setting|experimental setup|experiment|procedure|protocol|setup|not strictly controlled|manual|manually|human oversight|therapist|room|environment|labeling prompts|supervising image capture|presence of researcher",
            "analysis_setup_limitations": r"computational cost|autoencoder|supervised|pre[- ]?processing|diarization|segmentation|scoresheets|feature extraction|features.*not clear|performance metrics|ml models|simple classifiers|decision trees|logistic regression|transfer learning|not clear.*analysis|not enough info|minimum duration|comparison model|comparison models|other models|baseline models|one model|2 models|small number of models used|different models|no cross[- ]?validation|without cross[- ]?validation|lack of cross[- ]?validation|cross[- ]?validation was not used|no k[- ]?fold|without k[- ]?fold",
        }

        category_match_table = pd.DataFrame(index=text.index)
        category_match_table["combined_limitation_text"] = text
        counts = {}

        for category, pattern in category_patterns.items():
            category_mask = text.str.contains(pattern, regex=True, na=False)
            category_match_table[category] = category_mask
            count = int(category_mask.sum())
            counts[category] = count
            print(f"\n{category}:")
            print("Count:", count)
            print("Percentage:", pct(count, total_valid))

        category_cols = list(category_patterns.keys())
        any_match = category_match_table[category_cols].any(axis=1)

        placeholder_mask = text.str.fullmatch(
            r"\s*|[-\s]+|(?:no|yes|n/d|nd|n/a|na|nan|none|no limitation|no limitations|not reported|not applicable|n\.a\.)(?:\s+(?:no|yes|n/d|nd|n/a|na|nan|none|no limitation|no limitations|not reported|not applicable|n\.a\.|-))*\s*",
            na=False,
        )

        manual_revision_mask = ~any_match & text.ne("") & ~placeholder_mask
        empty_mask = text.eq("") | placeholder_mask

        counts["need_manual_revision"] = int(manual_revision_mask.sum())
        counts["empty_or_not_applicable"] = int(empty_mask.sum())

        print("\nneed_manual_revision:")
        print("Count:", counts["need_manual_revision"])
        print("Percentage:", pct(counts["need_manual_revision"], total_valid))

        print("\nempty_or_not_applicable:")
        print("Count:", counts["empty_or_not_applicable"])
        print("Percentage:", pct(counts["empty_or_not_applicable"], total_valid))

        manual_revision_rows = category_match_table.loc[manual_revision_mask].copy()

        print("\n=============Rows Needing Manual Coding=============")
        if manual_revision_rows.empty:
            print("No rows need manual coding.")
        else:
            print(manual_revision_rows[["combined_limitation_text"]].to_string())

        summary_df = count_percent_rows(counts, total_valid, "Limitation Category")
        save_df_optional(summary_df, "limitation_category_summary.csv")
        save_df_optional(category_match_table.reset_index().rename(columns={"index": "row_index"}), "limitation_category_match_table.csv")
        save_df_optional(manual_revision_rows.reset_index().rename(columns={"index": "row_index"}), "manual_revision_limitations.csv")
        return summary_df, manual_revision_rows, category_match_table


    limitation_summary, manual_revision_rows, limitation_category_matches = rq_limitation_categories(
        df_subset,
        valid_total,
        col_indices=[COL_LIMITATION_SOURCE_1, COL_LIMITATION_SOURCE_2, COL_STUDY_LIMITATIONS],
    )
    # ============================================================
    # 12. RQ2.15 GOAL-FINDING ALIGNMENT + MAIN FINDING SUBCATEGORIES
    # ============================================================


    def goal_finding_hybrid_summary(df, valid_mask, goal_col_index, finding_col_index):
        print("\n=============RQ2.15 Goal-Finding Hybrid Summary=============")

        goal_text = clean_text_series(df.loc[valid_mask, df.columns[goal_col_index]])
        finding_text = clean_text_series(df.loc[valid_mask, df.columns[finding_col_index]])

        total_valid = int(valid_mask.sum())

        # ------------------------------------------------------------
        # Broad categories used for BOTH stated goals and main findings
        # ------------------------------------------------------------
        # Screening, detection, classification, diagnosis, recognition,
        # and identification are merged because ASD ML papers often use
        # these terms interchangeably.

        category_patterns = {
            "screening_detection_classification_diagnosis": (
                r"\bscreen\w*\b"
                r"|\bdetection\b|\bdetect\w*\b"
                r"|\brecognition\b|\brecogniz\w*\b|\brecognis\w*\b"
                r"|\bidentification\b|\bidentify\w*\b"
                r"|\bclassif\w*\b|\bclassifier\b"
                r"|\bdistinguish\w*\b|\bdifferentiat\w*\b"
                r"|\bdiscriminat\w*\b"
                r"|\bdiagnos\w*\b|\bdiagnostic\b"
                r"|\basd\s*vs\b"
                r"|\bautism\s*vs\b"
                r"|\bautistic\s*vs\b"
                r"|\bnon[- ]?autism\b"
                r"|\bnon[- ]?asd\b"
                r"|\basd\s+and\s+td\b"
                r"|\basd\s+and\s+nt\b"
                r"|\basd\s+from\s+td\b"
                r"|\basd\s+from\s+non[- ]?asd\b"
            ),

            "severity_or_symptom_estimation": (
                r"\bseverity\b|\bsevere\b|\bseverely\b"
                r"|\bsymptom severity\b|\bsymptom\w*\b"
                r"|\btrait\w*\b|\bscore\w*\b|\bscale\w*\b"
                r"|\bados score\w*\b"
                r"|\bcars score\w*\b"
            ),

            "prediction_of_outcome_or_risk": (
                r"\bprediction\b|\bpredict\w*\b|\bforecast\w*\b"
                r"|\boutcome prediction\b|\brisk\b|\brisk assessment\b"
                r"|\bearly risk\b"
                r"|\blater diagnos\w*\b"
                r"|\bfollow[- ]?up\b"
            ),

            "biomarker_feature_or_behavioral_marker_identification": (
                r"\bbiomarker\w*\b|\bmarker\w*\b|\bindicator\w*\b"
                r"|\bfeature\w*\b|\bfeatures associated\b"
                r"|\bbehavioral marker\w*\b|\bbehavioural marker\w*\b"
                r"|\binvestigat\w*\b|\bcharacteri[sz]\w*\b"
                r"|\bpattern\w*\b"
                r"|\bhallmark\w*\b"
                r"|\bdifference\w*\b"
                r"|\bsignificant difference\w*\b"
                r"|\bcorrelation\w*\b"
                r"|\bcombin\w*\b"
            ),

            "intervention_feasibility_or_other": (
                r"\battention\b|\bstratification\b"
                r"|\bintervention\b|\btreatment\b|\btherapy\b"
                r"|\bfeasibility\b|\bfeasible\b"
                r"|\bmonitor\w*\b|\btracking\b"
                r"|\bclinical workflow\b"
                r"|\bclinical implementation\b"
                r"|\bprimary care\b"
                r"|\bclinician\w*\b"
            ),
        }

        priority_order = [
            "screening_detection_classification_diagnosis",
            "severity_or_symptom_estimation",
            "prediction_of_outcome_or_risk",
            "biomarker_feature_or_behavioral_marker_identification",
            "intervention_feasibility_or_other",
        ]

        # ------------------------------------------------------------
        # Main finding subcategories
        # These are ONLY for the main findings column.
        # They are non-mutually exclusive.
        # ------------------------------------------------------------

        finding_subcategory_patterns = {
            "high_model_performance_or_feasibility": (
                r"\bhigh accuracy\b"
                r"|\bhigh diagnostic accuracy\b"
                r"|\bhigh classification accuracy\b"
                r"|\baccuracy\b"
                r"|\baccurate\w*\b"
                r"|\bauc\b"
                r"|\broc[- ]?auc\b"
                r"|\bf1\b"
                r"|\bf1[- ]?score\b"
                r"|\bprecision\b"
                r"|\brecall\b"
                r"|\bsensitivity\b"
                r"|\bspecificity\b"
                r"|\bppv\b"
                r"|\bnpv\b"
                r"|\brecognition rate\b"
                r"|\bclassification rate\b"
                r"|\bsuperior performance\b"
                r"|\bpromising\b"
                r"|\beffective\b"
                r"|\bfeasible\b"
                r"|\bfeasibility\b"
                r"|\bsuccessful\b"
                r"|\boutperform\w*\b"
                r"|\bimproved performance\b"
                r"|\bstrong potential\b"
            ),

            "behavioral_feature_differences_between_groups": (
                r"\bdifference\w*\b"
                r"|\bsignificant difference\w*\b"
                r"|\bdifferent patterns?\b"
                r"|\bdistinct patterns?\b"
                r"|\bgroup difference\w*\b"
                r"|\basd .* showed\b"
                r"|\bautistic .* showed\b"
                r"|\bchildren with asd .* exhibited\b"
                r"|\bexhibited\b"
                r"|\bcharacteri[sz]ed by\b"
                r"|\bmore .* than\b"
                r"|\bless .* than\b"
                r"|\blower\b"
                r"|\bhigher\b"
                r"|\bgreater\b"
                r"|\breduced\b"
                r"|\bimpaired\b"
                r"|\bdeficit\w*\b"
            ),

            "specific_predictive_or_discriminative_features_identified": (
                r"\bfeature\w*\b"
                r"|\btop .* feature\w*\b"
                r"|\bdiscriminative feature\w*\b"
                r"|\bpredictive feature\w*\b"
                r"|\bmost significant\b"
                r"|\bimportant feature\w*\b"
                r"|\bselected feature\w*\b"
                r"|\bfeature selection\b"
                r"|\bfixation\w*\b"
                r"|\bsaccade\w*\b"
                r"|\bscan[- ]?path\w*\b"
                r"|\bscanpath\w*\b"
                r"|\bhead pose\b"
                r"|\bshoulder rotation\b"
                r"|\bkinematic\w*\b"
                r"|\btrajectory\w*\b"
                r"|\bcop\b"
                r"|\bcenter of pressure\b"
                r"|\bprosod\w*\b"
                r"|\bacoustic\w*\b"
                r"|\bsemantic\w*\b"
                r"|\bpragmatic\w*\b"
                r"|\bfacial dynamic\w*\b"
                r"|\bbody movement\w*\b"
                r"|\bmotor pattern\w*\b"
                r"|\bgaze pattern\w*\b"
            ),

            "multimodal_or_feature_fusion_improved_performance": (
                r"\bmultimodal\b"
                r"|\bmulti[- ]?modal\b"
                r"|\bfusion\b"
                r"|\bfus\w*\b"
                r"|\bcombining\b"
                r"|\bcombined\b"
                r"|\bcombination\b"
                r"|\bjoint analysis\b"
                r"|\bjoint optimization\b"
                r"|\bcomplementary information\b"
                r"|\bsingle data source\b"
                r"|\bsingle modality\b"
                r"|\bmultiple modalities\b"
                r"|\bfeature sets?\b"
            ),

            "model_algorithm_comparison_or_optimization": (
                r"\bbest model\b"
                r"|\bbest result\b"
                r"|\bbest performance\b"
                r"|\boptimal\b"
                r"|\boutperform\w*\b"
                r"|\bcompared\b"
                r"|\bcomparison\b"
                r"|\bclassifier\w*\b"
                r"|\bmodel\w*\b"
                r"|\balgorithm\w*\b"
                r"|\boptimizer\w*\b"
                r"|\bparameter\w*\b"
                r"|\bsvm\b"
                r"|\brandom forest\b"
                r"|\brf\b"
                r"|\bgradient boosting\b"
                r"|\bxgboost\b"
                r"|\bknn\b"
                r"|\blogistic regression\b"
                r"|\bann\b"
                r"|\bdnn\b"
                r"|\bcnn\b"
                r"|\blstm\b"
                r"|\bgru\b"
                r"|\bbert\b"
                r"|\btransformer\b"
                r"|\bauto[- ]?encoder\b"
                r"|\bgan\b"
                r"|\bhmm\b"
                r"|\bpomdp\b"
                r"|\bensemble\b"
                r"|\bhybrid\b"
                r"|\bsmote\b"
            ),

            "clinical_screening_or_diagnostic_support": (
                r"\bclinical\b"
                r"|\bclinician\w*\b"
                r"|\bdiagnostic support\b"
                r"|\bsupport clinicians\b"
                r"|\bassist\w* diagnosis\b"
                r"|\bcomputer[- ]?aided diagnosis\b"
                r"|\bobjective tool\b"
                r"|\bscreening tool\b"
                r"|\bdiagnostic tool\b"
                r"|\bprimary care\b"
                r"|\bclinical assessment\b"
                r"|\bclinical experts?\b"
                r"|\bearlier intervention\b"
                r"|\bearly intervention\b"
                r"|\blow[- ]?cost\b"
                r"|\beasy[- ]?to[- ]?implement\b"
                r"|\bdiagnostic evaluations?\b"
            ),

            "severity_symptom_trait_or_score_estimation": (
                r"\bseverity\b"
                r"|\bseverity level\w*\b"
                r"|\bsymptom severity\b"
                r"|\bsymptom\w*\b"
                r"|\btrait\w*\b"
                r"|\bautistic trait\w*\b"
                r"|\basd trait\w*\b"
                r"|\bados score\w*\b"
                r"|\bados\b"
                r"|\bcars\b"
                r"|\bscore\w*\b"
                r"|\bscale\w*\b"
                r"|\bsubscore\w*\b"
                r"|\bseverity estimation\b"
            ),

            "task_or_protocol_specific_effect": (
                r"\btask affected\b"
                r"|\btask type\b"
                r"|\bprotocol\b"
                r"|\bsetup\b"
                r"|\bsearch task\b"
                r"|\bbrowse task\b"
                r"|\bjoint attention task\b"
                r"|\bimitation task\b"
                r"|\bvisual orientation task\b"
                r"|\bvisual[- ]?orienting task\b"
                r"|\battention task\b"
                r"|\bgame[- ]?based\b"
                r"|\brobot\b"
                r"|\bvr\b"
                r"|\bvirtual reality\b"
                r"|\btablet\b"
                r"|\bhome environment\b"
                r"|\btask\b.*\bclassification\b"
            ),

            "negative_cautious_or_generalizability_limitation": (
                r"\bnot able\b"
                r"|\bfailed\b"
                r"|\bfail\w*\b"
                r"|\bno difference\b"
                r"|\bno clear\b"
                r"|\bnot sufficient\b"
                r"|\binsufficient\b"
                r"|\blimited\b"
                r"|\blimitation\w*\b"
                r"|\bpreliminary\b"
                r"|\bsmall sample\b"
                r"|\bsample size limitation\b"
                r"|\bheterogeneity\b"
                r"|\bheterogeneous\b"
                r"|\boverfitting\b"
                r"|\bgeneralise\b"
                r"|\bgeneralize\b"
                r"|\bnot representative\b"
                r"|\binterpret.*cautious\b"
                r"|\bshould be interpreted cautiously\b"
            ),

            "human_clinician_or_rater_comparison": (
                r"\bhuman rater\w*\b"
                r"|\bnon[- ]?expert human\b"
                r"|\bexpert\b"
                r"|\brater\w*\b"
                r"|\bclinician diagnosis\b"
                r"|\bclinician\w*\b"
                r"|\bclinical experts?\b"
                r"|\bagreement with clinician\b"
                r"|\bcomparable to clinician\w*\b"
                r"|\bcompared .* human\b"
                r"|\bcompared .* clinician\b"
            ),

            "gaze_visual_attention_finding": (
                r"\bgaze\b"
                r"|\beye[- ]?tracking\b"
                r"|\beye movement\w*\b"
                r"|\bvisual attention\b"
                r"|\bvisual exploration\b"
                r"|\bfixation\w*\b"
                r"|\bsaccade\w*\b"
                r"|\bscan[- ]?path\w*\b"
                r"|\bscanpath\w*\b"
                r"|\bvisual processing\b"
                r"|\bface scanning\b"
                r"|\bvisual preference\b"
                r"|\bjoint attention\b"
                r"|\barea[s]? of interest\b"
                r"|\baoi\b"
            ),

            "motor_movement_kinematic_finding": (
                r"\bmotor\b"
                r"|\bmovement\w*\b"
                r"|\bkinematic\w*\b"
                r"|\bgait\b"
                r"|\bpostur\w*\b"
                r"|\bpose\b"
                r"|\bpose estimation\b"
                r"|\bhead movement\w*\b"
                r"|\bbody movement\w*\b"
                r"|\bgesture\w*\b"
                r"|\btrajectory\w*\b"
                r"|\bwalking\b"
                r"|\bwalk\b"
                r"|\bgrasp\w*\b"
                r"|\breach\w*\b"
                r"|\bforce plate\b"
                r"|\bcenter of pressure\b"
                r"|\bcop\b"
                r"|\bpostural control\b"
            ),

            "speech_language_acoustic_finding": (
                r"\bspeech\b"
                r"|\bvoice\b"
                r"|\baudio\b"
                r"|\bacoustic\w*\b"
                r"|\bprosod\w*\b"
                r"|\bvocali[sz]ation\w*\b"
                r"|\bcry\b"
                r"|\btranscript\w*\b"
                r"|\blanguage\b"
                r"|\blinguistic\b"
                r"|\bsemantic\b"
                r"|\bpragmatic\b"
                r"|\bnarrative\b"
                r"|\bconversation\w*\b"
                r"|\bintonation\b"
                r"|\brhythm\b"
                r"|\bpitch\b"
                r"|\bloudness\b"
                r"|\bmfcc\b"
            ),

            "facial_expression_or_image_finding": (
                r"\bfacial\b"
                r"|\bface\b"
                r"|\bfacial expression\w*\b"
                r"|\bfacial image\w*\b"
                r"|\bface image\w*\b"
                r"|\bfacial gesture\w*\b"
                r"|\bemotion recognition\b"
                r"|\bemotion identification\b"
                r"|\bfacial affect\b"
                r"|\bresponsive social smile\b"
                r"|\bsmile\b"
                r"|\bimages of participants\b"
            ),

            "questionnaire_or_clinical_score_finding": (
                r"\bquestionnaire\w*\b"
                r"|\bquestionnare\w*\b"
                r"|\bquesstionnaire\w*\b"
                r"|\bquesstionaire\w*\b"
                r"|\bq[- ]?chat\b"
                r"|\bq[- ]?chat[- ]?10\b"
                r"|\baq[- ]?10\b"
                r"|\baq\b"
                r"|\bados\b"
                r"|\bcars\b"
                r"|\bsrs\b"
                r"|\bscq\b"
                r"|\brater score\w*\b"
                r"|\bclinical questionnaire\b"
                r"|\bdemographic\b"
                r"|\behr\b"
            ),

            "neurophysiology_or_biosignal_finding": (
                r"\beeg\b"
                r"|\berp\b"
                r"|\bmri\b"
                r"|\bfmri\b"
                r"|\bbrain activity\b"
                r"|\bbio[- ]?signal\w*\b"
                r"|\bbiosignal\w*\b"
                r"|\bskin temperature\b"
                r"|\bheart rate\b"
                r"|\bphysiological\b"
            ),

            "social_interaction_or_play_finding": (
                r"\bsocial interaction\b"
                r"|\binteraction\b"
                r"|\bplay\b"
                r"|\bfree play\b"
                r"|\bparent[- ]?child\b"
                r"|\bhuman[- ]?robot\b"
                r"|\brobot interaction\b"
                r"|\bturn[- ]?taking\b"
                r"|\bsocial behavior\w*\b"
                r"|\bsocialtaking\b"
                r"|\bsocial behavior\w*\b"
                r"|\bsocial behaviour\w*\b"
                r"|\bsocial biomarkers?\b"
                r"|\bsocial communication\b"
            ),
        }

        placeholder_pattern = (
            r"^\s*$|^-+$|^n/d$|^nd$|^n/a$|^na$|^nan$|"
            r"^not reported$|^not given$|^not specified$|^unclear$|^unknown$"
        )

        # ------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------

        def normalize_for_regex(x):
            if pd.isna(x):
                return ""
            return str(x).replace("\n", " ").strip()

        def assign_categories(text_value):
            text_value = normalize_for_regex(text_value)

            if re.fullmatch(placeholder_pattern, text_value, flags=re.IGNORECASE):
                return ["unclear"]

            matched = [
                cat for cat, pat in category_patterns.items()
                if re.search(pat, text_value, flags=re.IGNORECASE)
            ]

            return matched if matched else ["unclear"]

        def assign_finding_subcategories(text_value):
            text_value = normalize_for_regex(text_value)

            if re.fullmatch(placeholder_pattern, text_value, flags=re.IGNORECASE):
                return ["unclear"]

            matched = [
                subcat for subcat, pat in finding_subcategory_patterns.items()
                if re.search(pat, text_value, flags=re.IGNORECASE)
            ]

            return matched if matched else ["unclear"]

        def primary_category(categories):
            for cat in priority_order:
                if cat in categories:
                    return cat
            return "unclear"

        def simple_alignment(goal_categories, finding_categories):
            goal_set = set(goal_categories)
            finding_set = set(finding_categories)

            if "unclear" in goal_set or "unclear" in finding_set:
                return "unclear"

            if goal_set.intersection(finding_set):
                return "matched"

            return "partial_or_different"

        def short_text(x, max_len=180):
            x = normalize_for_regex(x)
            if len(x) > max_len:
                return x[:max_len] + "..."
            return x

        def summarize_primary_column(coded_df, col_name, title, total_valid):
            print(f"\n============={title}=============")

            counts = coded_df[col_name].value_counts(dropna=False)

            summary_df = pd.DataFrame([
                {
                    "Category": category,
                    "Count": int(count),
                    "Total Valid Papers": total_valid,
                    "Percentage": pct(count, total_valid),
                }
                for category, count in counts.items()
            ])

            print(summary_df.to_string(index=False))
            save_df_optional(summary_df, f"{col_name}_summary.csv")

            return summary_df

        def summarize_nonmutually_exclusive_categories(
            coded_df,
            list_col,
            category_list,
            total_valid,
            title,
            output_filename
        ):
            print(f"\n============={title}=============")

            rows = []

            for category in category_list:
                count = coded_df[list_col].apply(
                    lambda cats: category in cats
                ).sum()

                rows.append({
                    "Category": category,
                    "Count": int(count),
                    "Total Valid Papers": total_valid,
                    "Percentage": pct(count, total_valid),
                })

            summary_df = pd.DataFrame(rows).sort_values(
                by=["Count", "Category"],
                ascending=[False, True]
            ).reset_index(drop=True)

            print(summary_df.to_string(index=False))
            save_df_optional(summary_df, output_filename)

            return summary_df

        def summarize_finding_subcategories(coded_df, total_valid):
            print("\n=============Main Finding Subcategory Summary=============")

            rows = []

            for subcategory in finding_subcategory_patterns.keys():
                bool_col = f"finding_subcat__{subcategory}"
                count = int(coded_df[bool_col].sum())

                example_texts = (
                    coded_df.loc[coded_df[bool_col], "finding_text"]
                    .dropna()
                    .drop_duplicates()
                    .apply(lambda x: short_text(x, max_len=180))
                    .head(3)
                    .tolist()
                )

                rows.append({
                    "Finding Subcategory": subcategory,
                    "Count": count,
                    "Total Valid Papers": total_valid,
                    "Percentage": pct(count, total_valid),
                    "Example finding text": " | ".join(example_texts),
                })

            summary_df = pd.DataFrame(rows).sort_values(
                by=["Count", "Finding Subcategory"],
                ascending=[False, True]
            ).reset_index(drop=True)

            print(summary_df.to_string(index=False))
            save_df_optional(summary_df, "main_finding_subcategory_summary.csv")

            return summary_df

        # ------------------------------------------------------------
        # Code each paper
        # ------------------------------------------------------------

        coded_rows = []

        for idx in goal_text.index:
            goal_value = goal_text.loc[idx]
            finding_value = finding_text.loc[idx]

            goal_categories = assign_categories(goal_value)
            finding_categories = assign_categories(finding_value)

            goal_primary = primary_category(goal_categories)
            finding_primary = primary_category(finding_categories)

            alignment = simple_alignment(goal_categories, finding_categories)

            finding_subcategories = assign_finding_subcategories(finding_value)

            row = {
                "row_index": idx,
                "excel_row": idx + 3,
                "goal_text": goal_value,
                "finding_text": finding_value,
                "goal_categories": "; ".join(goal_categories),
                "finding_categories": "; ".join(finding_categories),
                "goal_categories_list": goal_categories,
                "finding_categories_list": finding_categories,
                "goal_primary_category": goal_primary,
                "finding_primary_category": finding_primary,
                "goal_finding_alignment": alignment,
                "finding_subcategories": "; ".join(finding_subcategories),
                "finding_subcategories_list": finding_subcategories,
                "finding_subcategory_count": 0 if finding_subcategories == ["unclear"] else len(finding_subcategories),
            }

            coded_rows.append(row)

        coded_df = pd.DataFrame(coded_rows)

        # Add one boolean column per main-finding subcategory
        for subcategory in finding_subcategory_patterns.keys():
            coded_df[f"finding_subcat__{subcategory}"] = coded_df[
                "finding_subcategories_list"
            ].apply(lambda cats: subcategory in cats)

        coded_df["finding_subcategory_unclear"] = coded_df[
            "finding_subcategories_list"
        ].apply(lambda cats: cats == ["unclear"])

        coded_df["multiple_finding_subcategories"] = coded_df[
            "finding_subcategory_count"
        ] >= 2

        coded_df["finding_broad_but_no_subcategory"] = (
            (coded_df["finding_primary_category"] != "unclear")
            & (coded_df["finding_subcategory_count"] == 0)
        )

        # ------------------------------------------------------------
        # Primary category summaries
        # ------------------------------------------------------------

        summary_tables = {}

        for col_name, title in [
            ("goal_primary_category", "Stated Goal Primary Category Summary"),
            ("finding_primary_category", "Main Finding Primary Category Summary"),
            ("goal_finding_alignment", "Simple Goal-Finding Alignment Summary"),
        ]:
            summary_tables[col_name] = summarize_primary_column(
                coded_df,
                col_name,
                title,
                total_valid
            )

        # ------------------------------------------------------------
        # Non-mutually exclusive broad category summaries
        # ------------------------------------------------------------

        all_broad_categories = priority_order + ["unclear"]

        goal_broad_nonexclusive_summary = summarize_nonmutually_exclusive_categories(
            coded_df=coded_df,
            list_col="goal_categories_list",
            category_list=all_broad_categories,
            total_valid=total_valid,
            title="Non-Mutually Exclusive Stated Goal Category Summary",
            output_filename="goal_broad_nonexclusive_summary.csv"
        )

        finding_broad_nonexclusive_summary = summarize_nonmutually_exclusive_categories(
            coded_df=coded_df,
            list_col="finding_categories_list",
            category_list=all_broad_categories,
            total_valid=total_valid,
            title="Non-Mutually Exclusive Main Finding Broad Category Summary",
            output_filename="finding_broad_nonexclusive_summary.csv"
        )

        summary_tables["goal_broad_nonexclusive_summary"] = goal_broad_nonexclusive_summary
        summary_tables["finding_broad_nonexclusive_summary"] = finding_broad_nonexclusive_summary

        # ------------------------------------------------------------
        # Goal vs Finding comparison table based on primary category
        # ------------------------------------------------------------

        goal_counts = coded_df["goal_primary_category"].value_counts()
        finding_counts = coded_df["finding_primary_category"].value_counts()

        comparison_rows = []

        for category in all_broad_categories:
            goal_count = int(goal_counts.get(category, 0))
            finding_count = int(finding_counts.get(category, 0))

            comparison_rows.append({
                "Category": category,
                "Goal Count": goal_count,
                "Goal Percentage": pct(goal_count, total_valid),
                "Finding Count": finding_count,
                "Finding Percentage": pct(finding_count, total_valid),
                "Finding Minus Goal Percentage": round(
                    pct(finding_count, total_valid) - pct(goal_count, total_valid),
                    2
                ),
            })

        goal_finding_comparison = pd.DataFrame(comparison_rows)

        print("\n=============Goal vs Finding Primary Category Comparison=============")
        print(goal_finding_comparison.to_string(index=False))

        save_df_optional(
            goal_finding_comparison,
            "goal_vs_finding_primary_category_comparison.csv"
        )

        summary_tables["goal_vs_finding_primary_category_comparison"] = goal_finding_comparison

        # ------------------------------------------------------------
        # Main finding subcategory summary
        # ------------------------------------------------------------

        finding_subcategory_summary = summarize_finding_subcategories(
            coded_df,
            total_valid
        )

        summary_tables["main_finding_subcategory_summary"] = finding_subcategory_summary

        print("\n=============Most Common Main Finding Subcategory=============")

        if finding_subcategory_summary.empty:
            print("No main finding subcategories identified.")
        else:
            top_subcat = finding_subcategory_summary.iloc[0]
            print("Subcategory:", top_subcat["Finding Subcategory"])
            print("Count:", top_subcat["Count"])
            print("Percentage:", top_subcat["Percentage"])

        print("\nmultiple_finding_subcategories:")
        multiple_count = int(coded_df["multiple_finding_subcategories"].sum())
        print("Count:", multiple_count)
        print("Percentage:", pct(multiple_count, total_valid))

        print("\nfinding_subcategory_unclear:")
        finding_subcat_unclear_count = int(coded_df["finding_subcategory_unclear"].sum())
        print("Count:", finding_subcat_unclear_count)
        print("Percentage:", pct(finding_subcat_unclear_count, total_valid))

        print("\nfinding_broad_but_no_subcategory:")
        broad_no_subcat_count = int(coded_df["finding_broad_but_no_subcategory"].sum())
        print("Count:", broad_no_subcat_count)
        print("Percentage:", pct(broad_no_subcat_count, total_valid))

        broad_no_subcat_rows = coded_df.loc[
            coded_df["finding_broad_but_no_subcategory"]
        ].copy()

        if not broad_no_subcat_rows.empty:
            print(
                broad_no_subcat_rows[
                    ["row_index", "excel_row", "finding_text", "finding_primary_category"]
                ].to_string(index=False)
            )

        # ------------------------------------------------------------
        # Clean manual review output
        # ------------------------------------------------------------

        manual_mask = (
            (coded_df["goal_primary_category"] == "unclear")
            | (coded_df["finding_primary_category"] == "unclear")
            | (coded_df["goal_finding_alignment"] != "matched")
            | (coded_df["finding_subcategory_unclear"])
            | (coded_df["finding_broad_but_no_subcategory"])
        )

        manual_rows = coded_df.loc[manual_mask].copy()

        def manual_review_reason(row):
            reasons = []

            if row["goal_primary_category"] == "unclear":
                reasons.append("goal unclear")

            if row["finding_primary_category"] == "unclear":
                reasons.append("finding broad category unclear")

            if row["goal_finding_alignment"] == "partial_or_different":
                reasons.append("goal and finding broad categories differ")

            if row["goal_finding_alignment"] == "unclear":
                reasons.append("alignment unclear")

            if row["finding_subcategory_unclear"]:
                reasons.append("finding subcategory unclear")

            if row["finding_broad_but_no_subcategory"]:
                reasons.append("finding has broad category but no subcategory")

            return "; ".join(reasons)

        manual_rows["manual_review_reason"] = manual_rows.apply(
            manual_review_reason,
            axis=1
        )

        manual_rows["goal_preview"] = manual_rows["goal_text"].apply(
            lambda x: short_text(x, max_len=120)
        )

        manual_rows["finding_preview"] = manual_rows["finding_text"].apply(
            lambda x: short_text(x, max_len=260)
        )

        manual_review_table = manual_rows[
            [
                "row_index",
                "excel_row",
                "manual_review_reason",
                "goal_preview",
                "finding_preview",
                "goal_primary_category",
                "finding_primary_category",
                "goal_finding_alignment",
                "goal_categories",
                "finding_categories",
                "finding_subcategories",
                "finding_subcategory_count",
            ]
        ].copy()

        # Add empty columns for manual correction
        manual_review_table["manual_goal_category"] = ""
        manual_review_table["manual_finding_category"] = ""
        manual_review_table["manual_alignment"] = ""
        manual_review_table["manual_finding_subcategories"] = ""
        manual_review_table["notes"] = ""

        print("\n=============Rows Needing Manual Review: Clean Table=============")

        if manual_review_table.empty:
            print("No rows need manual review.")
        else:
            print(manual_review_table.to_string(index=False))

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------

        # Drop list columns before saving for cleaner CSV outout
        coded_df_for_export = coded_df.drop(
            columns=[
                "goal_categories_list",
                "finding_categories_list",
                "finding_subcategories_list",
            ],
            errors="ignore"
        )

        save_df_optional(coded_df_for_export, "goal_finding_hybrid_coded_rows.csv")
        save_df_optional(manual_review_table, "goal_finding_hybrid_manual_review_table.csv")
        save_df_optional(
            broad_no_subcat_rows.drop(
                columns=[
                    "goal_categories_list",
                    "finding_categories_list",
                    "finding_subcategories_list",
                ],
                errors="ignore"
            ),
            "main_finding_broad_but_no_subcategory_rows.csv"
        )

        return (
            coded_df,
            manual_review_table,
            summary_tables,
            finding_subcategory_summary,
            broad_no_subcat_rows
        )


    goal_finding_coded_df, goal_finding_manual_review_table, goal_finding_summary_tables, main_finding_subcategory_summary, main_finding_broad_but_no_subcategory_rows = goal_finding_hybrid_summary(
        df_subset,
        valid_total,
        goal_col_index=COL_STUDY_GOAL,
        finding_col_index=COL_MAIN_FINDINGS,
    )

    #subcategory-level counts for reporting
    high_model_performance_count = int(goal_finding_coded_df["finding_subcat__high_model_performance_or_feasibility"].sum())
    behavioral_feature_differences_count = int(goal_finding_coded_df["finding_subcat__behavioral_feature_differences_between_groups"].sum())
    specific_features_identified_count = int(goal_finding_coded_df["finding_subcat__specific_predictive_or_discriminative_features_identified"].sum())
    multimodal_fusion_count = int(goal_finding_coded_df["finding_subcat__multimodal_or_feature_fusion_improved_performance"].sum())
    model_algorithm_comparison_count = int(goal_finding_coded_df["finding_subcat__model_algorithm_comparison_or_optimization"].sum())
    clinical_screening_support_count = int(goal_finding_coded_df["finding_subcat__clinical_screening_or_diagnostic_support"].sum())
    severity_symptom_trait_count = int(goal_finding_coded_df["finding_subcat__severity_symptom_trait_or_score_estimation"].sum())
    task_protocol_effect_count = int(goal_finding_coded_df["finding_subcat__task_or_protocol_specific_effect"].sum())
    negative_cautious_count = int(goal_finding_coded_df["finding_subcat__negative_cautious_or_generalizability_limitation"].sum())
    human_clinician_comparison_count = int(goal_finding_coded_df["finding_subcat__human_clinician_or_rater_comparison"].sum())

    gaze_visual_attention_finding_count = int(goal_finding_coded_df["finding_subcat__gaze_visual_attention_finding"].sum())
    motor_movement_kinematic_finding_count = int(goal_finding_coded_df["finding_subcat__motor_movement_kinematic_finding"].sum())
    speech_language_acoustic_finding_count = int(goal_finding_coded_df["finding_subcat__speech_language_acoustic_finding"].sum())
    facial_expression_image_finding_count = int(goal_finding_coded_df["finding_subcat__facial_expression_or_image_finding"].sum())
    questionnaire_clinical_score_finding_count = int(goal_finding_coded_df["finding_subcat__questionnaire_or_clinical_score_finding"].sum())
    neurophysiology_biosignal_finding_count = int(goal_finding_coded_df["finding_subcat__neurophysiology_or_biosignal_finding"].sum())
    social_interaction_play_finding_count = int(goal_finding_coded_df["finding_subcat__social_interaction_or_play_finding"].sum())

    # ============================================================
    # PRINT ALL GOAL-FINDING + MAIN FINDING SUBCATEGORY OUTPUTS
    # ============================================================

    def print_all_goal_finding_outputs(
        goal_finding_coded_df,
        goal_finding_summary_tables,
        main_finding_subcategory_summary,
        total_valid
    ):
        print("\n\n============================================================")
        print("FULL GOAL-FINDING + MAIN FINDING OUTPUTS")
        print("============================================================")

        # Make pandas print all rows/columns without truncation
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 160)

        # ------------------------------------------------------------
        # Primary broad goal/finding summaries
        # ------------------------------------------------------------
        print("\n================ PRIMARY BROAD GOAL CATEGORIES ================")

        if "goal_primary_category" in goal_finding_summary_tables:
            print(goal_finding_summary_tables["goal_primary_category"].to_string(index=False))
        else:
            goal_primary_counts = (
                goal_finding_coded_df["goal_primary_category"]
                .value_counts(dropna=False)
                .reset_index()
            )
            goal_primary_counts.columns = ["Category", "Count"]
            goal_primary_counts["Total Valid Papers"] = total_valid
            goal_primary_counts["Percentage"] = goal_primary_counts["Count"].apply(
                lambda x: pct(x, total_valid)
            )
            print(goal_primary_counts.to_string(index=False))

        print("\n================ PRIMARY BROAD FINDING CATEGORIES ================")

        if "finding_primary_category" in goal_finding_summary_tables:
            print(goal_finding_summary_tables["finding_primary_category"].to_string(index=False))
        else:
            finding_primary_counts = (
                goal_finding_coded_df["finding_primary_category"]
                .value_counts(dropna=False)
                .reset_index()
            )
            finding_primary_counts.columns = ["Category", "Count"]
            finding_primary_counts["Total Valid Papers"] = total_valid
            finding_primary_counts["Percentage"] = finding_primary_counts["Count"].apply(
                lambda x: pct(x, total_valid)
            )
            print(finding_primary_counts.to_string(index=False))

        # ------------------------------------------------------------
        # Goal-finding alignment
        # ------------------------------------------------------------
        print("\n================ GOAL-FINDING ALIGNMENT ================")

        if "goal_finding_alignment" in goal_finding_summary_tables:
            print(goal_finding_summary_tables["goal_finding_alignment"].to_string(index=False))
        else:
            alignment_counts = (
                goal_finding_coded_df["goal_finding_alignment"]
                .value_counts(dropna=False)
                .reset_index()
            )
            alignment_counts.columns = ["Category", "Count"]
            alignment_counts["Total Valid Papers"] = total_valid
            alignment_counts["Percentage"] = alignment_counts["Count"].apply(
                lambda x: pct(x, total_valid)
            )
            print(alignment_counts.to_string(index=False))

        # ------------------------------------------------------------
        # Non-mutually exclusive broad goal categories
        # ------------------------------------------------------------
        print("\n================ NON-MUTUALLY EXCLUSIVE BROAD GOAL CATEGORIES ================")

        if "goal_broad_nonexclusive_summary" in goal_finding_summary_tables:
            print(goal_finding_summary_tables["goal_broad_nonexclusive_summary"].to_string(index=False))
        else:
            print("goal_broad_nonexclusive_summary not found in summary_tables.")

        # ------------------------------------------------------------
        # Non-mutually exclusive broad finding categories
        # ------------------------------------------------------------
        print("\n================ NON-MUTUALLY EXCLUSIVE BROAD FINDING CATEGORIES ================")

        if "finding_broad_nonexclusive_summary" in goal_finding_summary_tables:
            print(goal_finding_summary_tables["finding_broad_nonexclusive_summary"].to_string(index=False))
        else:
            print("finding_broad_nonexclusive_summary not found in summary_tables.")

        # ------------------------------------------------------------
        # Goal vs finding comparison
        # ------------------------------------------------------------
        print("\n================ GOAL VS FINDING PRIMARY CATEGORY COMPARISON ================")

        if "goal_vs_finding_primary_category_comparison" in goal_finding_summary_tables:
            print(goal_finding_summary_tables["goal_vs_finding_primary_category_comparison"].to_string(index=False))
        else:
            print("goal_vs_finding_primary_category_comparison not found in summary_tables.")

        # ------------------------------------------------------------
        # Main finding subcategory summary
        # ------------------------------------------------------------
        print("\n================ MAIN FINDING SUBCATEGORIES ================")

        if main_finding_subcategory_summary is not None and not main_finding_subcategory_summary.empty:
            print(main_finding_subcategory_summary.to_string(index=False))
        else:
            print("No main finding subcategory summary found.")

        # ------------------------------------------------------------
        # Recalculate all main finding subcategory counts directly
        # from boolean columns, to make sure every subcategory prints
        # ------------------------------------------------------------
        print("\n================ ALL MAIN FINDING SUBCATEGORY BOOLEAN COUNTS ================")

        finding_subcat_cols = [
            col for col in goal_finding_coded_df.columns
            if col.startswith("finding_subcat__")
        ]

        subcat_rows = []

        for col in finding_subcat_cols:
            clean_name = col.replace("finding_subcat__", "")
            count = int(goal_finding_coded_df[col].sum())

            subcat_rows.append({
                "Category type": "Main finding subcategory",
                "Category": clean_name,
                "Count": count,
                "Total valid studies": total_valid,
                "Percentage": pct(count, total_valid),
            })

        all_finding_subcat_counts = pd.DataFrame(subcat_rows).sort_values(
            by=["Count", "Category"],
            ascending=[False, True]
        ).reset_index(drop=True)

        print(all_finding_subcat_counts.to_string(index=False))

        # ------------------------------------------------------------
        # Derived finding-subcategory counts
        # ------------------------------------------------------------
        print("\n================ DERIVED MAIN FINDING COUNTS ================")

        derived_count_specs = [
            ("multiple_finding_subcategories", "Studies with multiple finding subcategories"),
            ("finding_subcategory_unclear", "Studies with unclear finding subcategory"),
            ("finding_broad_but_no_subcategory", "Studies with broad finding category but no subcategory"),
        ]

        derived_rows = []

        for col, label in derived_count_specs:
            if col in goal_finding_coded_df.columns:
                count = int(goal_finding_coded_df[col].sum())
                derived_rows.append({
                    "Derived variable": col,
                    "Description": label,
                    "Count": count,
                    "Total valid studies": total_valid,
                    "Percentage": pct(count, total_valid),
                })

        derived_finding_counts = pd.DataFrame(derived_rows)

        print(derived_finding_counts.to_string(index=False))

        # ------------------------------------------------------------
        # Most common categories
        # ------------------------------------------------------------
        print("\n================ MOST COMMON GOAL / FINDING CATEGORIES ================")

        if "goal_primary_category" in goal_finding_coded_df.columns:
            top_goal = (
                goal_finding_coded_df["goal_primary_category"]
                .value_counts(dropna=False)
                .reset_index()
            )
            top_goal.columns = ["Category", "Count"]
            top_goal["Total valid studies"] = total_valid
            top_goal["Percentage"] = top_goal["Count"].apply(lambda x: pct(x, total_valid))
            print("\nMost common primary goal category:")
            print(top_goal.head(1).to_string(index=False))

        if "finding_primary_category" in goal_finding_coded_df.columns:
            top_finding = (
                goal_finding_coded_df["finding_primary_category"]
                .value_counts(dropna=False)
                .reset_index()
            )
            top_finding.columns = ["Category", "Count"]
            top_finding["Total valid studies"] = total_valid
            top_finding["Percentage"] = top_finding["Count"].apply(lambda x: pct(x, total_valid))
            print("\nMost common primary finding category:")
            print(top_finding.head(1).to_string(index=False))

        if not all_finding_subcat_counts.empty:
            print("\nMost common main finding subcategory:")
            print(all_finding_subcat_counts.head(1).to_string(index=False))

        # ------------------------------------------------------------
        # Optional: print rows needing review
        # ------------------------------------------------------------
        print("\n================ ROWS WITH UNCLEAR / NO SUBCATEGORY FINDINGS ================")

        review_mask = pd.Series(False, index=goal_finding_coded_df.index)

        if "finding_subcategory_unclear" in goal_finding_coded_df.columns:
            review_mask = review_mask | goal_finding_coded_df["finding_subcategory_unclear"]

        if "finding_broad_but_no_subcategory" in goal_finding_coded_df.columns:
            review_mask = review_mask | goal_finding_coded_df["finding_broad_but_no_subcategory"]

        review_rows = goal_finding_coded_df.loc[review_mask].copy()

        if review_rows.empty:
            print("No unclear/no-subcategory finding rows.")
        else:
            cols_to_print = [
                "row_index",
                "excel_row",
                "finding_text",
                "finding_primary_category",
                "finding_subcategories",
                "finding_subcategory_count",
            ]
            cols_to_print = [
                col for col in cols_to_print
                if col in review_rows.columns
            ]
            print(review_rows[cols_to_print].to_string(index=False))

        # ------------------------------------------------------------
        # Save clean all-output tables
        # ------------------------------------------------------------
        save_df_optional(all_finding_subcat_counts, "all_main_finding_subcategory_counts.csv")
        save_df_optional(derived_finding_counts, "all_main_finding_derived_counts.csv")

        if "goal_primary_category" in goal_finding_coded_df.columns:
            save_df_optional(top_goal, "all_goal_primary_counts.csv")

        if "finding_primary_category" in goal_finding_coded_df.columns:
            save_df_optional(top_finding, "all_finding_primary_counts.csv")

        save_df_optional(review_rows, "main_finding_unclear_or_no_subcategory_rows.csv")

        return (
            all_finding_subcat_counts,
            derived_finding_counts,
            review_rows
        )


    # ============================================================
    # RUN FULL GOAL-FINDING PRINT OUTPUTS
    # ============================================================

    all_main_finding_subcategory_counts, all_main_finding_derived_counts, main_finding_unclear_or_no_subcategory_rows = print_all_goal_finding_outputs(
        goal_finding_coded_df=goal_finding_coded_df,
        goal_finding_summary_tables=goal_finding_summary_tables,
        main_finding_subcategory_summary=main_finding_subcategory_summary,
        total_valid=int(valid_total.sum())
    )
    # ============================================================
    # 13. YES/NO COUNTS: BIDS, NOVEL ANALYSIS, SENSITIVE DATA
    # ============================================================

    def compute_yes_no_text_count(col, valid_mask, label):
        print(f"\n============={label}=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        yes_mask = col_filtered.str.contains(r"^\s*yes\b|^\s*yes\s*[:\-]", regex=True, na=False)
        no_mask = col_filtered.str.contains(r"^\s*no\b", regex=True, na=False)
        placeholder_mask = col_filtered.str.fullmatch(r"\s*|-+|n/a|na|nd|n/d|nan|none|not reported|not given|unclear|unknown", na=False)
        other_nonempty_mask = ~(yes_mask | no_mask | placeholder_mask)

        counts = {
            "yes": int(yes_mask.sum()),
            "no": int(no_mask.sum()),
            "empty_or_not_reported": int(placeholder_mask.sum()),
            "other_nonempty_manual_review": int(other_nonempty_mask.sum()),
        }

        for category, count in counts.items():
            print(f"\n{category}:")
            print("Count:", count)
            print("Percentage:", pct(count, total_valid))

        match_table = pd.DataFrame({
            f"{label.lower().replace(' ', '_')}_text": col_filtered,
            "yes": yes_mask,
            "no": no_mask,
            "empty_or_not_reported": placeholder_mask,
            "other_nonempty_manual_review": other_nonempty_mask,
        })
        manual_rows = match_table.loc[other_nonempty_mask].copy()

        print(f"\n============={label} Manual Review Rows=============")
        if manual_rows.empty:
            print("No rows need manual review.")
        else:
            print(manual_rows[[f"{label.lower().replace(' ', '_')}_text"]].to_string())

        summary_df = count_percent_rows(counts, total_valid, f"{label} Category")
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        save_df_optional(summary_df, f"{safe_label}_summary.csv")
        save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), f"{safe_label}_match_table.csv")
        save_df_optional(manual_rows.reset_index().rename(columns={"index": "row_index"}), f"{safe_label}_manual_review_rows.csv")
        return summary_df, match_table, manual_rows


    bids_summary, bids_match_table, bids_manual_rows = compute_yes_no_text_count(
        df_subset.iloc[:, COL_BIDS_STRUCTURE], valid_total, label="BIDS Data Structure"
    )

    novel_analysis_summary, novel_analysis_match_table, novel_analysis_manual_rows = compute_yes_no_text_count(
        df_subset.iloc[:, COL_NOVEL_ANALYSIS_PIPELINE], valid_total, label="Novel Analysis Pipeline"
    )

    sensitive_data_summary, sensitive_data_match_table, sensitive_data_manual_rows = compute_yes_no_text_count(
        df_subset.iloc[:, COL_SENSITIVE_DATA], valid_total, label="Sensitive Data"
    )


    # ============================================================
    # 14. SENSITIVE DATA PROTECTION MEASURES
    # ============================================================

    def sensitive_data_protection_summary(col, valid_mask):
        print("\n=============Sensitive Data Protection Measures=============")

        col_filtered = clean_text_series(col[valid_mask])
        total_valid = int(valid_mask.sum())

        placeholder_mask = col_filtered.str.fullmatch(r"\s*|-+|n/a|na|nd|n/d|nan|none|not reported|not given|unclear|unknown|no", na=False)

        patterns = {
            "deidentification_or_machine_learning_based_privacy_preserving_methods": r"de[- ]?identif\w*|participant id|subject id|anonymous id|coded id|personal information removed|automatic labelling|anony\w*|federated learning|privacy preserving machine learning|machine unlearning|dropped",
            "face_blurring_or_mosaicing": r"mosaic\w*|blur\w*|face.*removed|removed.*face|mask\w*.*face|face.*mask\w*",
            "consent_or_ethics_compliance": r"consent|parent\w*.*choose|choose when|permission|helsinki",
            "restricted_access_or_secure_storage": r"restricted|secure|encrypted|password|access control|stored securely|protected server|public\w*",
        }

        match_table = pd.DataFrame(index=col_filtered.index)
        match_table["sensitive_data_protection_text"] = col_filtered
        match_table["placeholder_no_measure_reported"] = placeholder_mask

        counts = {}
        for category, pattern in patterns.items():
            mask = ~placeholder_mask & col_filtered.str.contains(pattern, regex=True, na=False)
            match_table[category] = mask
            counts[category] = int(mask.sum())
            print(f"\n{category}:")
            print("Count:", counts[category])
            print("Percentage:", pct(counts[category], total_valid))

        category_cols = list(patterns.keys())
        match_table["any_protection_category_matched"] = match_table[category_cols].any(axis=1)
        match_table["manual_review_non_placeholder"] = ~placeholder_mask & ~match_table["any_protection_category_matched"]

        counts["not_reported_or_placeholder"] = int(placeholder_mask.sum())
        counts["manual_review_non_placeholder"] = int(match_table["manual_review_non_placeholder"].sum())

        print("\nnot_reported_or_placeholder:")
        print("Count:", counts["not_reported_or_placeholder"])
        print("Percentage:", pct(counts["not_reported_or_placeholder"], total_valid))

        print("\nmanual_review_non_placeholder:")
        print("Count:", counts["manual_review_non_placeholder"])
        print("Percentage:", pct(counts["manual_review_non_placeholder"], total_valid))

        manual_rows = match_table.loc[match_table["manual_review_non_placeholder"]].copy()
        print("\n=============Protection Measure Rows Needing Manual Review=============")
        if manual_rows.empty:
            print("No rows need manual review.")
        else:
            print(manual_rows[["sensitive_data_protection_text"]].to_string())

        summary_df = count_percent_rows(counts, total_valid, "Protection Measure Category")
        save_df_optional(summary_df, "sensitive_data_protection_summary.csv")
        save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "sensitive_data_protection_match_table.csv")
        save_df_optional(manual_rows.reset_index().rename(columns={"index": "row_index"}), "sensitive_data_protection_manual_rows.csv")
        return summary_df, match_table, manual_rows


    sensitive_protection_summary, sensitive_protection_match_table, sensitive_protection_manual_rows = sensitive_data_protection_summary(
        df_subset.iloc[:, COL_SENSITIVE_DATA_PROTECTION], valid_total
    )


    # ============================================================
    # 15. FUTURE GOALS / RESEARCH AIMS CATEGORIES
    # ============================================================
    def compute_future_goals_categories(col, valid_mask):
        print("\n=============Future Goals / Research Aims Categories=============")

        total_valid = int(valid_mask.sum())
        col_filtered = clean_text_series(col[valid_mask])

        category_patterns = {
            "dataset_size_and_diversity": (
                r"\blarger data\w*\b"
                r"|\bmore data\b"
                r"|\bincrease dataset\b"
                r"|\bpopulation diversity\b"
                r"|\bmore population diversity\b"
                r"|\bdiverse\b"
                r"|\bdiverse dataset\b"
                r"|\bfemale\w*\b"
                r"|\bgender differences\b"
                r"|\bbalanced groups\b"
                r"|\bbalanced sample\b"
                r"|\bmatch the comparison group\w*\b"
                r"|\bmatch the comaprison group\w*\b"
                r"|\bmatch the comparison groups better\b"
                r"|\bmatch the comaprison groups better\b"
                r"|\bmatch the groups\b"
                r"|\bgroup variability\b"
                r"|\bage\b"
                r"|\byounger\b"
                r"|\bother diagnos\w*\b"
                r"|\bother daignos\w*\b"
                r"|\bpariticipants with other daignosis\b"
                r"|\bparticipants? with other diagnos\w*\b"
                r"|\bparticipants? with other daignos\w*\b"
                r"|\bneurological\b"
                r"|\bdifferent disorders\b"
                r"|\bparticipants? with other disorders\b"
                r"|\bcollaborating with clinicians\b"
                r"|\bcollaborat\w* with clinician\w*\b"
            ),

            "feature_expansion_and_fusion": (
                r"\bmore feature\w*\b"
                r"|\bfeature fusion\b"
                r"|\bcombine features\b"
                r"|\bcombining features\b"
                r"|\bdata fusion\b"
                r"|\bbehavioral modalit\w*\b"
                r"|\bcorrelation between features\b"
                r"|\bdemographic information\b"
                r"|\bdemographics\b"
            ),

            "validation_and_generalizability": (
                r"\bvalidation\b"
                r"|\bexternal data\w*\b"
                r"|\bexternal validation\b"
                r"|\bmultiple data\w*\b"
                r"|\bother data\w*\b"
                r"|\bdifferent data\w*\b"
                r"|\bvalidate the model with other data\w*\b"
                r"|\bvalidate.*other data\w*\b"
                r"|\bvalidate.*different data\w*\b"
                r"|\blongitudinal\b"
                r"|\bfollow[- ]?up\b"
                r"|\bdifferent tasks\b"
                r"|\bmore task features\b"
                r"|\bbetter experimental setup\b"
                r"|\bdifferent setting\w*\b"
                r"|\bgeneraliz\w*\b"
                r"|\breal[- ]?world data\b"
                r"|\breal world\b"
            ),

            "model_performance_and_optimization": (
                r"\baccuracy\b"
                r"|\bclassification performance\b"
                r"|\bimprove performance\b"
                r"|\bincrease classification performance\b"
                r"|\boptimi[sz]e ml metrics\b"
                r"|\befficiency\b"
                r"|\bdifferent ml models\b"
                r"|\btest different ml models\b"
                r"|\bimprove diagnosis\b"
            ),

            "automation_tools_and_implementation": (
                r"\bfully automated process\b"
                r"|\bunified tool\b"
                r"|\bsustainability\b"
                r"|\bimplementation\b"
                r"|\bdeploy\w*\b"
                r"|\bneuroimaging data\b"
            ),

            "further_analysis_unspecified": (
                r"\bfurther\b"
                r"|\bfuther\b"
                r"|\bfurther analysis\b"
                r"|\bfuture work\b"
                r"|\bexplainable\b"
                r"|\bexplainability\b"
                r"|\btemporal\b"
                r"|\battention mechanisms\b"
            ),

            "not_given": (
                r"\bno\b"
                r"|\bnot given\b"
                r"|\bnot specified\b"
                r"|\bnot reported\b"
                r"|\bn/a\b"
                r"|\bna\b"
                r"|\bn\.a\b"
                r"|\bnd\b"
                r"|\bn\.d\b"
                r"|\bn/d\b"
            ),
        }

        match_table = pd.DataFrame(index=col_filtered.index)
        match_table["future_goal_text"] = col_filtered
        counts = {}

        for category, pattern in category_patterns.items():
            mask = col_filtered.str.contains(pattern, regex=True, na=False)
            match_table[category] = mask
            counts[category] = int(mask.sum())

            print(f"\n{category}:")
            print("Count:", counts[category])
            print("Percentage:", pct(counts[category], total_valid))

        category_cols = list(category_patterns.keys())
        match_table["any_future_goal_category_matched"] = match_table[category_cols].any(axis=1)
        match_table["no_category_matched"] = ~match_table["any_future_goal_category_matched"]
        counts["no_category_matched"] = int(match_table["no_category_matched"].sum())

        print("\nno_category_matched:")
        print("Count:", counts["no_category_matched"])
        print("Percentage:", pct(counts["no_category_matched"], total_valid))

        unmatched_rows = match_table.loc[match_table["no_category_matched"]].copy()

        print("\n=============Unmatched Future Goal Texts=============")
        if unmatched_rows.empty:
            print("No unmatched future goal texts.")
        else:
            print(unmatched_rows[["future_goal_text"]].to_string())

        summary_df = count_percent_rows(
            counts,
            total_valid,
            "Future Goal / Research Aim Category"
        )

        print("\n=============Future Goals Summary Table=============")
        print(summary_df.to_string(index=False))

        save_df_optional(summary_df, "future_goals_summary.csv")
        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            "future_goals_match_table.csv"
        )
        save_df_optional(
            unmatched_rows.reset_index().rename(columns={"index": "row_index"}),
            "future_goals_unmatched_rows.csv"
        )

        return summary_df, match_table, unmatched_rows


    future_goals_summary, future_goals_match_table, future_goals_unmatched_rows = compute_future_goals_categories(
        df_subset.iloc[:, COL_FUTURE_RESEARCH_PIPELINES],
        valid_total,
    )

    # ============================================================
    # 16. FINAL CHECK
    # ============================================================

    print("\n=============FINAL CHECK=============")
    print("Total valid papers:", int(valid_total.sum()))
    print("Script completed successfully.")
    if SAVE_OUTPUTS:
        print(f"Output files saved in: {OUTPUT_DIR.resolve()}")
    else:
        print("SAVE_OUTPUTS is False, so no output files were saved.")


if __name__ == "__main__":
    main()

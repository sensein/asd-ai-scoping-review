import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import columns as COL
from analysis_common import RQ_QUESTIONS, RQ_TITLES
from codebook import NOT_GIVEN_TASK_PATTERN, TASK_TYPE_PATTERNS
from setup_data_ import load_annotation_data
from helper_functions_ import (
    OTHER_BEHAVIORAL_CATEGORY_PATTERNS,
    clean_text_series as shared_clean_text_series,
    count_percent_rows,
    is_no as shared_is_no,
    is_yes as shared_is_yes,
    pct,
    set_output_name_prefix,
)
RQ_NUMBER = 3
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
    # BO / 66  = Data collection tool
    # BP / 67  = Participant task / task type
    COL_TITLE = COL.TITLE
    COL_PUBLICATION_YEAR = COL.PUBLICATION_YEAR
    COL_GAZE = COL.GAZE
    COL_SPEECH = COL.SPEECH
    COL_MOTOR = COL.MOTOR
    COL_OTHER_BEHAVIORAL = COL.OTHER_BEHAVIORAL
    COL_OTHER_TYPE_DATA = COL.OTHER_DATA
    COL_FEATURE_FUSION = COL.FEATURE_FUSION
    COL_DATA_COLLECTION_TOOL = COL.DATA_COLLECTION_TOOL
    COL_PARTICIPANT_TASK = COL.PARTICIPANT_TASK
    COLUMN_MAP = pd.DataFrame([
        {"Variable": "Title", "Excel Column": "A", "Python iloc Index": COL_TITLE},
        {"Variable": "Publication year", "Excel Column": "E", "Python iloc Index": COL_PUBLICATION_YEAR},
        {"Variable": "Gaze data", "Excel Column": "AO", "Python iloc Index": COL_GAZE},
        {"Variable": "Speech/language/audio data", "Excel Column": "AP", "Python iloc Index": COL_SPEECH},
        {"Variable": "Motor data", "Excel Column": "AQ", "Python iloc Index": COL_MOTOR},
        {"Variable": "Other behavioral data", "Excel Column": "AR", "Python iloc Index": COL_OTHER_BEHAVIORAL},
        {"Variable": "Other type of data", "Excel Column": "AS", "Python iloc Index": COL_OTHER_TYPE_DATA},
        {"Variable": "Feature fusion / feature combination", "Excel Column": "AT", "Python iloc Index": COL_FEATURE_FUSION},
        {"Variable": "Data collection tool", "Excel Column": "BO", "Python iloc Index": COL_DATA_COLLECTION_TOOL},
        {"Variable": "Participant task", "Excel Column": "BP", "Python iloc Index": COL_PARTICIPANT_TASK},
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


# Data collection tool analysis

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





    # RQ5 publication-trend outputs are generated by rq5_.py.



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
if __name__ == "__main__":
    main()
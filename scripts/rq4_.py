import pandas as pd
import numpy as np
import os
import os
import re
from pathlib import Path

from setup_data_ import load_annotation_data, INVALID_VALUES

from helper_functions_ import (
    ensure_series_mask,
    yes_no_modality_summary,
    compute_study_setting,
    compute_task_type,
    compute_asd_age_ranges,
    accuracy_by_behavioral_modality,
    accuracy_by_study_setting,
    accuracy_by_task_type,
    accuracy_by_asd_age_group,
    make_exclusive_task_group,
    summarize_accuracy_by_group,
    summarize_accuracy_by_flags,
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
OUTPUT_DIR = OUTPUT_ROOT / "rq4_results"

if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HELPER_OUTPUT_DIR = str(OUTPUT_DIR) if SAVE_OUTPUTS else None
set_output_name_prefix("RQ4")


# ============================================================
# 1. LOAD DATA
# ============================================================

data = load_annotation_data()

df = data["df"]
df_subset = data["df_subset"].copy()

valid_total = ensure_series_mask(data["valid_total"], df_subset.index)
valid_ASD = data.get("valid_ASD")
valid_mask_ASD = data.get("valid_mask_ASD", valid_ASD)
ASD_VALID_MASK = valid_ASD if valid_ASD is not None else valid_mask_ASD
ASD_VALID_MASK = ensure_series_mask(ASD_VALID_MASK, df_subset.index) if ASD_VALID_MASK is not None else valid_total

valid_papers_Total = int(data.get("valid_papers_Total", valid_total.sum()))


# ============================================================
# 2. RQ4 COLUMN MAP
# ============================================================
# Current worksheet: data
# Python uses zero-based iloc indexing.
#
# M  / 12 = ASD sample size
# N  / 13 = ASD label
# O  / 14 = ASD age range
# P  / 15 = ASD mean age
# Q  / 16 = ASD SD age
# AO / 40 = gaze
# AP / 41 = speech
# AQ / 42 = motor
# AR / 43 = other behavioural
# AS / 44 = other type of data
# AT / 45 = fusion technique
# AU / 46 = algorithms used
# AV / 47 = features (or E2E)
# AW / 48 = learning type (RL, SL, ...)
# AX / 49 = evaluation metrics
# AY / 50 = best performing model
# AZ / 51 = best performance
# BA / 52 = features importance technique
# BB / 53 = features importance result
# BC / 54 = balancing/unbiasing technique
# BD / 55 = cross corpus validation
# BE / 56 = X-fold cross-validation
# BF / 57 = LOXO
# BG / 58 = Real time analysis?
# BH / 59 = study setting
# BI / 60 = study goal
# BP / 67 = task for the participants

COL_ASD_LABEL = 13
COL_ASD_AGE_RANGE = 14
COL_ASD_MEAN_AGE = 15
COL_ASD_SD_AGE = 16

COL_GAZE = 40
COL_SPEECH = 41
COL_MOTOR = 42
COL_OTHER_BEHAVIOURAL = 43
COL_OTHER_DATA = 44
COL_FUSION_TECHNIQUE = 45
COL_ALGORITHMS = 46
COL_FEATURES = 47
COL_LEARNING_TYPE = 48
COL_EVALUATION_METRICS = 49
COL_BEST_MODEL = 50
COL_BEST_PERFORMANCE = 51
COL_FEATURE_IMPORTANCE_TECHNIQUE = 52
COL_FEATURE_IMPORTANCE_RESULT = 53
COL_BALANCING_TECHNIQUE = 54
COL_CROSS_CORPUS_VALIDATION = 55
COL_XFOLD_CV = 56
COL_LOXO = 57
COL_REAL_TIME_ANALYSIS = 58
COL_STUDY_SETTING = 59
COL_STUDY_GOAL = 60
COL_TASK_PARTICIPANTS = 67

RQ4_COLUMN_MAP = pd.DataFrame([
    {"Variable": "ASD label", "Excel Column": "N", "Python iloc Index": COL_ASD_LABEL},
    {"Variable": "ASD age range", "Excel Column": "O", "Python iloc Index": COL_ASD_AGE_RANGE},
    {"Variable": "ASD mean age", "Excel Column": "P", "Python iloc Index": COL_ASD_MEAN_AGE},
    {"Variable": "ASD SD age", "Excel Column": "Q", "Python iloc Index": COL_ASD_SD_AGE},
    {"Variable": "Gaze modality flag", "Excel Column": "AO", "Python iloc Index": COL_GAZE},
    {"Variable": "Speech modality flag", "Excel Column": "AP", "Python iloc Index": COL_SPEECH},
    {"Variable": "Motor modality flag", "Excel Column": "AQ", "Python iloc Index": COL_MOTOR},
    {"Variable": "Other behavioural modality flag", "Excel Column": "AR", "Python iloc Index": COL_OTHER_BEHAVIOURAL},
    {"Variable": "Other type of data", "Excel Column": "AS", "Python iloc Index": COL_OTHER_DATA},
    {"Variable": "Fusion technique", "Excel Column": "AT", "Python iloc Index": COL_FUSION_TECHNIQUE},
    {"Variable": "Algorithms used", "Excel Column": "AU", "Python iloc Index": COL_ALGORITHMS},
    {"Variable": "Features or end-to-end representation", "Excel Column": "AV", "Python iloc Index": COL_FEATURES},
    {"Variable": "Learning type", "Excel Column": "AW", "Python iloc Index": COL_LEARNING_TYPE},
    {"Variable": "Evaluation metrics", "Excel Column": "AX", "Python iloc Index": COL_EVALUATION_METRICS},
    {"Variable": "Best performing model", "Excel Column": "AY", "Python iloc Index": COL_BEST_MODEL},
    {"Variable": "Best performance", "Excel Column": "AZ", "Python iloc Index": COL_BEST_PERFORMANCE},
    {"Variable": "Feature importance technique", "Excel Column": "BA", "Python iloc Index": COL_FEATURE_IMPORTANCE_TECHNIQUE},
    {"Variable": "Feature importance result", "Excel Column": "BB", "Python iloc Index": COL_FEATURE_IMPORTANCE_RESULT},
    {"Variable": "Balancing / unbiasing technique", "Excel Column": "BC", "Python iloc Index": COL_BALANCING_TECHNIQUE},
    {"Variable": "Cross-corpus validation", "Excel Column": "BD", "Python iloc Index": COL_CROSS_CORPUS_VALIDATION},
    {"Variable": "X-fold cross-validation", "Excel Column": "BE", "Python iloc Index": COL_XFOLD_CV},
    {"Variable": "LOXO", "Excel Column": "BF", "Python iloc Index": COL_LOXO},
    {"Variable": "Real-time analysis", "Excel Column": "BG", "Python iloc Index": COL_REAL_TIME_ANALYSIS},
    {"Variable": "Study setting", "Excel Column": "BH", "Python iloc Index": COL_STUDY_SETTING},
    {"Variable": "Study goal", "Excel Column": "BI", "Python iloc Index": COL_STUDY_GOAL},
    {"Variable": "Participant task", "Excel Column": "BP", "Python iloc Index": COL_TASK_PARTICIPANTS},
])

print("\n============= RQ4 COLUMN MAP =============")
print(RQ4_COLUMN_MAP.to_string(index=False))


# ============================================================
# 3. GENERAL HELPERS
# ============================================================


def _prefixed_output_name(filename):
    prefix = "RQ4_"
    directory, basename = Path(filename).parent, Path(filename).name
    if basename.lower().startswith(prefix.lower()):
        basename = "RQ4" + basename[len("RQ4"):]
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


def clean_text_series(series):
    return series.fillna("").astype(str).str.lower().str.strip()


def pct(count, denominator):
    return round((count / denominator) * 100, 2) if denominator else 0


def count_percent_rows(count_dict, total_valid, category_col="Category"):
    return pd.DataFrame([
        {
            category_col: category,
            "Count": int(count),
            "Total Valid Papers": int(total_valid),
            "Percentage": pct(int(count), int(total_valid)),
        }
        for category, count in count_dict.items()
    ])


def invalid_or_placeholder_mask(series):
    return series.str.fullmatch(
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
        r"|not explicitly stated"
        r"|not applicable"
        r"|not clear"
        r"|unclear"
        r"|unknown",
        na=False,
    )


def summarize_patterns(col, valid_mask, patterns, label, category_col="Category", save_prefix=None):
    print(f"\n============= {label} =============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    match_table = pd.DataFrame(index=col_filtered.index)
    match_table[f"{label.lower().replace(' ', '_')}_text"] = col_filtered

    counts = {}
    for category, pattern in patterns.items():
        mask = col_filtered.str.contains(pattern, regex=True, na=False)
        match_table[category] = mask
        counts[category] = int(mask.sum())

        print(f"\n{category}:")
        print("Count:", counts[category])
        print("Percentage:", pct(counts[category], total_valid))

    summary_df = count_percent_rows(counts, total_valid, category_col=category_col)
    print(f"\n============= {label} Summary Table =============")
    print(summary_df.to_string(index=False))

    if save_prefix:
        save_df_optional(summary_df, f"{save_prefix}_summary.csv")
        save_df_optional(
            match_table.reset_index().rename(columns={"index": "row_index"}),
            f"{save_prefix}_match_table.csv",
        )

    return summary_df, match_table


save_df_optional(RQ4_COLUMN_MAP, "rq4_column_map.csv")


# ============================================================
# 4. SHARED DESCRIPTIVE CODING TABLES
# These are reused later for RQ4.5 accuracy summaries.
# ============================================================

modality_summary, modality_match_table = yes_no_modality_summary(
    df=df_subset,
    valid_mask=valid_total,
    gaze_col=COL_GAZE,
    motor_col=COL_MOTOR,
    speech_col=COL_SPEECH,
    output_dir=HELPER_OUTPUT_DIR,
)

study_setting_summary, study_setting_match_table = compute_study_setting(
    col=df_subset.iloc[:, COL_STUDY_SETTING],
    valid_mask=valid_total,
    output_dir=HELPER_OUTPUT_DIR,
)

task_type_summary, task_type_match_table, task_type_unclear_rows = compute_task_type(
    col=df_subset.iloc[:, COL_TASK_PARTICIPANTS],
    valid_mask=valid_total,
    output_dir=HELPER_OUTPUT_DIR,
)

asd_age_summary = pd.DataFrame()
asd_age_parser_usage = pd.DataFrame()
asd_age_match_table = pd.DataFrame()
asd_age_manual_review = pd.DataFrame()

if int(ASD_VALID_MASK.sum()) > 0:
    asd_age_summary, asd_age_parser_usage, asd_age_match_table, asd_age_manual_review = compute_asd_age_ranges(
        df_subset=df_subset,
        range_col_index=COL_ASD_AGE_RANGE,
        mean_col_index=COL_ASD_MEAN_AGE,
        sd_col_index=COL_ASD_SD_AGE,
        valid_asd_mask=ASD_VALID_MASK,
        output_dir=HELPER_OUTPUT_DIR,
    )
else:
    print("\n============= AGE RANGE: ASD =============")
    print("Skipped: ASD-valid mask is empty or unavailable.")


# ============================================================
# 5. MACHINE LEARNING PARADIGM
# ============================================================

def machine_learning_paradigm(col, valid_mask):
    patterns = {
        "supervised_learning": r"\bsupervised\b|\bsl\b|\bclassification\b|\bclassifier\b|\bregression\b",
        "unsupervised_learning": r"\bunsupervised\b|\bclustering\b|\bk[- ]?means\b|\bpca\b|\bautoencoder\b|\bvae\b",
        "reinforcement_learning": r"\breinforcement\b|\brl\b|\bq[- ]?learning\b|\bpomdp\b|\bmarkov decision\b",
        "semi_self_or_transfer_learning": r"\bsemi[- ]?supervised\b|\bself[- ]?supervised\b|\btransfer learning\b|\bfine[- ]?tuning\b|\bpretrained\b|\bpre-trained\b",
    }
    return summarize_patterns(
        col,
        valid_mask,
        patterns,
        label="Machine Learning Paradigm",
        category_col="Learning Paradigm",
        save_prefix="rq4_learning_paradigm",
    )


summary_learning_paradigms, learning_paradigm_match_table = machine_learning_paradigm(
    df_subset.iloc[:, COL_LEARNING_TYPE], valid_total
)


# ============================================================
# 6. FEATURES: BROAD CATEGORIES
# ============================================================

def features_broad(col, valid_mask):
    patterns = {
        "gaze_eye_tracking_features": (
            r"\baoi\b|\broi\b|area of interest|region of interest|fixation|saccade|blink|eye[- ]?movement"
            r"|eye[- ]?tracking|scanpath|scan[- ]?path|gaze|visual attention|visit count|revisit|length of gaze"
            r"|tracking ratio|point regard|gaze vector|heatmap|saliency map"
        ),
        "facial_expression_face_features": (
            r"facial landmark|face landmark|openface|open face|open-face|facial expression|face|facial"
            r"|action unit|\bau\d+\b|smile|smiling|mouth|eyes|eyebrow|lip|emotion recognition|facial dynamics"
        ),
        "motor_pose_kinematic_features": (
            r"speed|acceleration|velocity|duration|movement|motion|amplitude|deceleration|distance|displacement"
            r"|openpose|open pose|head pose|head movement|pitch|yaw|roll|rotation|joint movement|skeleton|skeletal"
            r"|keypoint|pose|grip force|sway|jerk|kinematic|gait|stride|walking|gesture|tablet|touch|wheel rotation|rmse"
        ),
        "social_interaction_behavioral_features": (
            r"reaction|latency|response latency|eye contact|social engagement|human behavior coding|observation coding"
            r"|imitation|imitate|social influence|response bias|correctness of response|turn[- ]?taking|joint attention|interaction"
        ),
        "language_speech_acoustic_features": (
            r"\bword\b|\bwords\b|word count|tf[- ]?idf|word2vec|wav2vec|bert|transformer|nlp|natural language"
            r"|universal sentence encoder|sentence embedding|text embedding|tweet|questionnaire|q[- ]?chat|aq\b|cat[- ]?q"
            r"|audio|raw audio|spectrogram|spectogram|spectral|prosody|pitch|timbre|voice|vocalization|vocalisation"
            r"|echolalia|sound response|mfcc|speech rhythm|acoustic"
        ),
        "vector_or_embedding_features": (
            r"presence vector|weighted presence vector|feature vector|behavioral vector|behavioural vector|embedding|embeddings"
            r"|latent representation|latent vector|vector\b|vectors\b"
        ),
        "image_video_visual_features": (
            r"raw video|\bvideo\b|\bvideos\b|\bimage\b|\bimages\b|frame|frames|rgb|optical flow"
            r"|visual features|video analysis|image analysis|heatmap|heat-map|scanpath image|facial image"
        ),
        "demographic_developmental_background_features": (
            r"\bage\b|\bsex\b|\bgender\b|developmental history|age of walking|age of first words|crib rocking"
            r"|pregnancy|delivery|premature|family history|parental|vaccination|sensory|adaptive behavior|iq\b"
        ),
    }
    return summarize_patterns(
        col,
        valid_mask,
        patterns,
        label="Features Broad Categories",
        category_col="Feature Category",
        save_prefix="rq4_features_broad",
    )


summary_features, features_match_table = features_broad(df_subset.iloc[:, COL_FEATURES], valid_total)


# ============================================================
# 7. FEATURES: SUBCATEGORIES
# ============================================================

def feature_subcategories(col, valid_mask):
    print("\n============= Features: Subcategories =============")

    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    parent_patterns = {
        
        "gaze_eye_tracking": (
            r"\baoi\b|\broi\b|area of interest|region of interest|fixation|saccade|blink|eye movement|eye[- ]?tracking"
            r"|scanpath|scan path|scan-path|visit count|revisits|length of gaze|gaze|visual attention|heatmap"
        ),
        "facial_face_based": (
            r"facial landmark|face landmark|openface|open face|open-face|facial expression|smile|smiling"
            r"|adult eyes|child eyes|adult mouth|child mouth|joint attention|action unit|\bau\d+\b|facial dynamics"
        ),
        "motor_pose_kinematic": (
            r"speed|acceleration|velocity|duration|movement|motion|amplitude|deceleration|distance|displacement|jerk"
            r"|kinematic|rmse|head movement|head pose|pitch|yaw|roll|rotation|openpose|skeleton|skeletal|keypoint"
            r"|joint movement|gait|grip force|sway|wheel rotation|accelerometer|gesture|tablet|touch"
        ),
        "social_interaction_behavioral": (
            r"duration of reaction|latency of reaction|latency in response|response latency|eye contact|social engagement"
            r"|imitation|imitate|response bias|correctness of response|social influence|human behavior coding|observation coding|turn[- ]?taking"
        ),
        "language_speech_acoustic": (
            r"word|word count|tf[- ]?idf|word2vec|wav2vec|bert|nlp|natural language|universal sentence encoder"
            r"|text embedding|tweet|questionnaire|q[- ]?chat|audio|raw audio|spectrogram|spectral|mfcc|timbre|prosody|pitch"
            r"|voice|vocalization|vocalisation|echolalia|sound responses|speech rhythm|acoustic"
        ),
        "vector_representations": r"presence vector|weighted presence vector|feature vector|behavioral vector|behavioural vector|vector-based|embedding|latent|vector\b|vectors\b",
        "image_video_visual": r"raw video|video|videos|image|images|frame|frames|rgb|optical flow|video analysis|image analysis|visual features|heatmap|heat-map",
        "demographic_developmental_background": (
            r"age|sex|gender|age of walking|age of first words|crib rocking|music insistence|reaction to bright lights"
            r"|reaction to colors|reaction to sounds|pregnancy|delivery|premature|oxygen after birth|miscarriage"
            r"|family history|parental|suspected deafness|vaccination|iq|adaptive behavior"
        ),
    }

    subcategory_patterns = {
        "aoi_roi_features": {"parent": "gaze_eye_tracking", "pattern": r"\baoi\b|\broi\b|area of interest|areas of interest|region of interest|regions of interest"},
        "fixation_features": {"parent": "gaze_eye_tracking", "pattern": r"fixation|fixation count|fixation duration|total fixation duration|time to first fixation|fixation map"},
        "saccade_eye_movement_features": {"parent": "gaze_eye_tracking", "pattern": r"saccade|saccades|blink|eye movement|eye movements|eye[- ]?tracking|smooth pursuit|anti[- ]?saccade"},
        "scanpath_features": {"parent": "gaze_eye_tracking", "pattern": r"scanpath|scanpaths|scan path|scan paths|scan-path|scan-paths|scanpath image"},
        "visual_attention_features": {"parent": "gaze_eye_tracking", "pattern": r"visit count|revisits|length of gaze|attentional pattern|visual attention|tracking ratio|point regard|gaze vector|heatmap|saliency"},
        "facial_landmark_features": {"parent": "facial_face_based", "pattern": r"facial landmark|facial landmarks|face landmark|openface|open face|open-face|landmark dynamics"},
        "facial_expression_features": {"parent": "facial_face_based", "pattern": r"facial expression|facial expressions|smile|smiling|smile detection|social smiling|emotion|affective expression|expression|emotion"},
        "facial_action_unit_features": {"parent": "facial_face_based", "pattern": r"action unit|\bau\d+\b|inner[- ]?brow|lip[- ]?corner|facial action"},
        "face_region_social_attention_features": {"parent": "facial_face_based", "pattern": r"adult eyes|child eyes|adult mouth|child mouth|joint attention|mouth|eyes|face region"},
        "general_kinematic_features": {"parent": "motor_pose_kinematic", "pattern": r"speed|acceleration|velocity|duration|movement amplitude|amplitude|deceleration|distance|displacement|jerk|kinematic|rmse"},
        "head_movement_features": {"parent": "motor_pose_kinematic", "pattern": r"head movement|head pose|head pose angles|head rotation|head motion|\bhead\b|pitch|yaw|roll|rotation range"},
        "pose_skeletal_features": {"parent": "motor_pose_kinematic", "pattern": r"openpose|open pose|open-pose|skeletal keypoint|skeletal keypoints|skeleton|keypoint|joint movement|pose"},
        "task_specific_motor_features": {"parent": "motor_pose_kinematic", "pattern": r"gait|stride|walking|grip force|sway area|wheel rotation|length of action|tablet|touch|gesture|grasp"},
        "interaction_timing_features": {"parent": "social_interaction_behavioral", "pattern": r"duration of reaction|latency of reaction|latency in response|response latency|reaction time"},
        "eye_contact_social_engagement_features": {"parent": "social_interaction_behavioral", "pattern": r"eye contact|duration and frequency of eye contact|social engagement|joint attention"},
        "imitation_turn_taking_features": {"parent": "social_interaction_behavioral", "pattern": r"imitation|imitate|accuracy of imitation|turn[- ]?taking|spontaneous engagement"},
        "response_task_performance_features": {"parent": "social_interaction_behavioral", "pattern": r"response bias|response-bias|correctness of response|social influence factor|task performance"},
        "human_coded_behavior_features": {"parent": "social_interaction_behavioral", "pattern": r"human behavior coding|human observation coding|observation coding|behavior coding|behaviour coding|manual coding"},
        "text_word_level_features": {"parent": "language_speech_acoustic", "pattern": r"\bword\b|\bwords\b|number of words|word count|word error rate|word correctness rate|repeated words|tf[- ]?idf"},
        "text_embedding_nlp_features": {"parent": "language_speech_acoustic", "pattern": r"\bnlp\b|natural language processing|universal sentence encoder|\buse\b|word2vec|wav2vec|bert|transformer|text embedding|language embedding|sentence embedding"},
        "text_data_sources": {"parent": "language_speech_acoustic", "pattern": r"tweets|tweet|questionnaire|questionnaires|q[- ]?chat|aq\b|cat[- ]?q|survey"},
        "audio_acoustic_features": {"parent": "language_speech_acoustic", "pattern": r"audio length|raw audio|spectrogram|spectogram|spectral|spectral entropy|mfcc|timbre|formant|jitter|shimmer|harmonic|hnr"},
        "speech_vocal_features": {"parent": "language_speech_acoustic", "pattern": r"prosody|pitch|fundamental frequency|\bf0\b|vocalization|vocalizations|vocalisation|vocalisations|echolalia|sound responses|speech rhythm|voice"},
        "presence_vector_features": {"parent": "vector_representations", "pattern": r"presence vector|presence vectors"},
        "weighted_vector_features": {"parent": "vector_representations", "pattern": r"weighted presence vector|weighted presence vectors"},
        "other_vector_based_representations": {"parent": "vector_representations", "pattern": r"feature vector|feature vectors|behavioral vector|behavioural vector|vector-based|embedding|embeddings|latent representation|latent vector|\bvector\b|\bvectors\b"},
        "raw_visual_inputs": {"parent": "image_video_visual", "pattern": r"raw video|video|videos|image|images|frames|rgb|facial image|scanpath image"},
        "image_video_analysis_features": {"parent": "image_video_visual", "pattern": r"video analysis|image analysis|image analysis features|visual features|optical flow|frame-level"},
        "spatial_visual_representations": {"parent": "image_video_visual", "pattern": r"heatmap|heatmaps|heat-map|heat-maps|saliency map|fixation map"},
        "demographic_features": {"parent": "demographic_developmental_background", "pattern": r"\bage\b|\bsex\b|\bgender\b"},
        "developmental_history_features": {"parent": "demographic_developmental_background", "pattern": r"age of walking|age of first words|crib rocking|music insistence|reaction to bright lights|reaction to colors|reaction to sounds|developmental history"},
        "prenatal_perinatal_history_features": {"parent": "demographic_developmental_background", "pattern": r"pregnancy|delivery|premature birth|oxygen after birth|miscarriage|perinatal|prenatal"},
        "family_parental_history_features": {"parent": "demographic_developmental_background", "pattern": r"family cognitive disability history|family history|parental educational|parental smoking|parental drug use|anti[- ]?depressants|antidepressants"},
        "medical_sensory_background_features": {"parent": "demographic_developmental_background", "pattern": r"suspected deafness|vaccination|sensory|medical history|adaptive behavior|iq\b"},
    }

    parent_masks = {parent: col_filtered.str.contains(pattern, regex=True, na=False) for parent, pattern in parent_patterns.items()}
    parent_counts = {parent: int(mask.sum()) for parent, mask in parent_masks.items()}

    rows = []
    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["feature_text"] = col_filtered

    for subcategory, info in subcategory_patterns.items():
        parent = info["parent"]
        pattern = info["pattern"]
        subcategory_mask = col_filtered.str.contains(pattern, regex=True, na=False)
        count = int(subcategory_mask.sum())
        parent_count = int(parent_counts[parent])
        match_table[subcategory] = subcategory_mask
        rows.append({
            "Feature Subcategory": subcategory,
            "Parent Category": parent,
            "Count": count,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(count, total_valid),
            "Parent Count": parent_count,
            "Percentage of Parent Category": pct(count, parent_count),
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_feature_subcategories_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_feature_subcategories_match_table.csv")
    return summary_df, match_table


summary_feature_subcategories, feature_subcategory_match_table = feature_subcategories(
    df_subset.iloc[:, COL_FEATURES], valid_total
)


# ============================================================
# 8. ALGORITHMS: BROAD CATEGORIES
# ============================================================

PATTERN_CLASSICAL_ML = (
    r"linear regres+sion|logistic regression|linear discriminant analysis|\blda\b|quadratic classifier"
    r"|support vector machine|\bsvm\b|\bknn\b|k[- ]?nearest neighbors?|naive[- ]?bayes|naïve[- ]?bayes|\bnb\b"
    r"|decision tree|random forest|extra trees|regulari[sz]ed greedy forest|\bcart\b|\bridge\b|elastic net"
)

PATTERN_ENSEMBLE = (
    r"gradient boost|gradient boosting|\bgb\b|gbm|gbdt|adaboost|ada boost"
    r"|xgboost|extreme gradient boosting|lightgbm|light gbm|lgbm"
    r"|catboost|cat boost|ensemble\w*|voting|bagging|boosting"
    r"|stacking|stacked ensemble"
)

PATTERN_NEURAL = (
    r"\bann\b|artificial neural network|multi[- ]?layer perceptron|multilayer perceptron|\bmlp\b|\bfnn\b|feed[- ]?forward"
    r"|fcdnn|\bdnn\b|deep neural network|\bcnn\b|convolutional neural network|resnet|resnet[- ]?50|googlenet|inception|inceptionv3"
    r"|vgg|vgg[- ]?16|vgg[- ]?19|mobilenet|efficientnet|xception|convnext|yolo|yolov8|neural network"
    r"|\brnn\b|recurrent neural network|\blstm\b|bi[- ]?lstm|\bblstm\b|\bgru\b|cnn[-+ ]?gru|cnn[-+ ]?lstm"
    r"|attention|relu|dropout|fully connected|softmax"
    r"|graph convolutional network|\bgcn\b|graph neural network|\bgnn\b|msg3d|st[- ]?gcn|ksnet"
    r"|generative adversarial network|\bgan\b|\bvae\b|sdae|stacked denoising autoencoder|autoencoder|binary classifier|pnn|transformer|bert|wav2vec"
)

PATTERN_STATISTICAL_SPECIALISED = (
    r"\blasso\b|kernel extreme learning machine|kernel extreme machine learning|\bkelm\b|extreme learning machine|\belm\b|fvelm"
    r"|markov model|\bpomdp\b|\bhmm\b|hidden markov|bayesian|gaussian process|gami[- ]?net"
    r"|giza pyramids construction|\bgpc\b|metaheuristic|genetic algorithm|particle swarm|\bpso\b"
)

INVALID_ALGORITHM_VALUES = [
    "", "nan", "none", "n/a", "na", "not applicable", "not reported",
    "not specified", "not stated", "not mentioned", "unknown",
    "unclear", "no information", "no algorithm", "not available"
]


def algorithms_broad(col, valid_mask):
    patterns = {
        "classical_machine_learning_models": PATTERN_CLASSICAL_ML,
        "ensemble_models": PATTERN_ENSEMBLE,
        "neural_network_models": PATTERN_NEURAL,
        "statistical_and_other_specialised_models": PATTERN_STATISTICAL_SPECIALISED,
    }

    summary_df, match_table = summarize_patterns(
        col,
        valid_mask,
        patterns,
        label="Algorithm Broad Categories",
        category_col="Algorithm Broad Category",
        save_prefix="rq4_algorithms_broad",
    )

    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_clean = col.astype(str).str.strip().str.lower().replace({"nan": ""})
    invalid_algorithm_mask = valid_mask & col_clean.isin(INVALID_ALGORITHM_VALUES)

    category_cols = list(patterns.keys())
    any_category_match_valid = match_table[category_cols].any(axis=1)
    no_algorithm_category_match_mask = pd.Series(False, index=col.index)
    no_algorithm_category_match_mask.loc[match_table.index] = ~any_category_match_valid

    print("\n=============== ALGORITHM INFORMATION COVERAGE ===============")
    print(f"Total valid papers: {int(valid_mask.sum())}")
    print(f"Papers with missing/invalid algorithm information: {int(invalid_algorithm_mask.sum())}")
    print(f"Valid papers with no broad algorithm category matched: {int(no_algorithm_category_match_mask.sum())}")

    if no_algorithm_category_match_mask.sum() > 0:
        print("\nUnmatched algorithm entries:")
        print(col[no_algorithm_category_match_mask].dropna().unique())

    counts = {row["Algorithm Broad Category"]: int(row["Count"]) for _, row in summary_df.iterrows()}
    summary_counts = {
        "classical": counts.get("classical_machine_learning_models", 0),
        "ensemble": counts.get("ensemble_models", 0),
        "neural": counts.get("neural_network_models", 0),
        "other_and_statistical": counts.get("statistical_and_other_specialised_models", 0),
        "missing_or_invalid_algorithm_info": int(invalid_algorithm_mask.sum()),
        "no_algorithm_category_match": int(no_algorithm_category_match_mask.sum()),
    }

    return summary_df, match_table, summary_counts, invalid_algorithm_mask, no_algorithm_category_match_mask


summary_algorithms_df, algorithms_match_table, summary_algorithms, invalid_algorithm_mask, no_algorithm_category_match_mask = algorithms_broad(
    df_subset.iloc[:, COL_ALGORITHMS], valid_total
)


# ============================================================
# 9. ALGORITHMS: NEURAL SUBCATEGORIES
# ============================================================

def algorithms_further_neural_networks(col, valid_mask, neural_denominator):
    print("\n============== Algorithms Further Classified: Neural Networks =============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    patterns = {
        "basic_neural_network_models": r"\bann\b|artificial neural network|multi[- ]?layer perceptron|multilayer perceptron|mutilayer perceptron|\bmlp\b|feed[- ]?forward neural network|\bfnn\b|fcdnn|fdcnn",
        "deep_learning_neural_networks": r"\bdnn\b|deep neural network|\bcnn\b|convolutional neural network|resnet|googlenet|inception|vgg|mobilenet|efficientnet|xception|convnext|yolo|fully connected deep neural network",
        "sequence_models": r"\brnn\b|recurrent neural network|\blstm\b|long[- ]?short[- ]?term[- ]?memory|bi[- ]?directional long short term memory|bi[- ]?lstm|\bblstm\b|\bgru\b|cnn[-+ ]?gru|cnn[-+ ]?lstm|wav2vec",
        "graph_neural_networks": r"graph convolutional network|\bgcn\b|graph neural network|\bgnn\b|msg3d|st[- ]?gcn|ksnet",
        "generative_or_autoencoder_models": r"generative adversarial network|\bgan\b|\bvae\b|sdae|stacked denoising autoencoder|autoencoder|autoencoders",
        "transformer_or_foundation_models": r"transformer|bert|wav2vec|word2vec|vision transformer|\bvit\b",
        "other_neural_hybrid_models": r"binary classification network|binary classification|binary classifier|\bpnn\b|hybrid neural|dual[- ]?stream|pairwise euclidean",
    }

    rows = []
    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["algorithm_text"] = col_filtered

    for category, pattern in patterns.items():
        mask = col_filtered.str.contains(pattern, regex=True, na=False)
        count = int(mask.sum())
        match_table[category] = mask
        rows.append({
            "Neural Network Category": category,
            "Count": count,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(count, total_valid),
            "Neural Network Parent Count": int(neural_denominator),
            "Percentage of Neural Network Papers": pct(count, neural_denominator),
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_algorithms_neural_subcategories_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_algorithms_neural_subcategories_match_table.csv")
    return summary_df, match_table


summary_algorithms_nn, algorithms_nn_match_table = algorithms_further_neural_networks(
    df_subset.iloc[:, COL_ALGORITHMS], valid_total, summary_algorithms["neural"]
)


# ============================================================
# 10. ALGORITHMS: BASIC / CLASSICAL ML SUBCATEGORIES
# ============================================================

def algorithms_further_basic_ml(col, valid_mask, classical_denominator):
    print("\n============== Algorithms Further Classified: Basic Machine Learning Models =============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    patterns = {
        "linear_and_generalised_linear_models": r"linear regres+sion|logistic regression|linear discriminant analysis|\blda\b|quadratic classifier|ridge|elastic net",
        "support_vector_methods": r"support vector machine|\bsvm\b|svr",
        "instance_based_learning": r"\bknn\b|k[- ]?nearest neighbors?|nearest neighbour|nearest neighbor",
        "probabilistic_models": r"naive[- ]?bayes|naïve[- ]?bayes|\bnb\b|bayesian",
        "tree_based_models": r"decision tree|random forest|extra trees|regulari[sz]ed greedy forest|\bcart\b",
    }

    rows = []
    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["algorithm_text"] = col_filtered

    for category, pattern in patterns.items():
        mask = col_filtered.str.contains(pattern, regex=True, na=False)
        count = int(mask.sum())
        match_table[category] = mask
        rows.append({
            "Classical ML Category": category,
            "Count": count,
            "Total Valid Papers": total_valid,
            "Percentage of Total Papers": pct(count, total_valid),
            "Classical ML Parent Count": int(classical_denominator),
            "Percentage of Classical ML Papers": pct(count, classical_denominator),
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_algorithms_classical_subcategories_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_algorithms_classical_subcategories_match_table.csv")
    return summary_df, match_table


summary_algorithms_classical, algorithms_classical_match_table = algorithms_further_basic_ml(
    df_subset.iloc[:, COL_ALGORITHMS], valid_total, summary_algorithms["classical"]
)


# ============================================================
# 11. HYBRID MODELS + MULTI-MODEL PAPERS
# ============================================================

MODEL_TERMS_FOR_HYBRID = (
    r"cnn|lstm|dnn|ann|rnn|gru|blstm|bi[- ]?lstm|mlp|svm|pnn"
    r"|autoencoder|ae|vgg|vgg[- ]?16|vgg[- ]?19|resnet|googlenet|inception"
    r"|gcn|gnn|gan|vae|bert|transformer|xgboost|random forest"
    r"|decision tree|dt|knn|naive bayes|nb|kelm|elm"
)

PATTERN_HYBRID_MODEL = (
    rf"\b(?:{MODEL_TERMS_FOR_HYBRID})\b\s*(?:\+|&|and|with|/|-)\s*\b(?:{MODEL_TERMS_FOR_HYBRID})\b"
    r"|hybrid model|hybrid framework|hybrid architecture|hybrid approach|\bhybrid\b"
    r"|dual[- ]?stream|multi[- ]?stream|two[- ]?stream"
    r"|ensemble of|combination of models|combined model|combined models"
)


def algorithms_hybrid(col, valid_mask):
    print("\n=========== Hybrid Models =============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    hybrid_mask = col_filtered.str.contains(PATTERN_HYBRID_MODEL, regex=True, na=False)
    hybrid_count = int(hybrid_mask.sum())

    summary_df = count_percent_rows({"hybrid_or_multi_model_architectures": hybrid_count}, total_valid, category_col="Algorithm Category")
    print(summary_df.to_string(index=False))

    match_table = pd.DataFrame({"algorithm_text": col_filtered, "hybrid_or_multi_model_architecture": hybrid_mask})

    save_df_optional(summary_df, "rq4_hybrid_models_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_hybrid_models_match_table.csv")
    return summary_df, match_table, hybrid_mask


hybrid_models_summary, hybrid_models_match_table, hybrid_model_mask = algorithms_hybrid(
    df_subset.iloc[:, COL_ALGORITHMS], valid_total
)


def detect_algorithm_families(value):
    text = str(value).lower().strip()
    families = []
    if re.search(PATTERN_CLASSICAL_ML, text):
        families.append("classical machine learning")
    if re.search(PATTERN_ENSEMBLE, text):
        families.append("ensemble model")
    if re.search(PATTERN_NEURAL, text):
        families.append("neural network")
    if re.search(PATTERN_STATISTICAL_SPECIALISED, text):
        families.append("statistical / specialised model")
    return families


def multi_model_papers(col, valid_mask):
    print("\n============= Multi-Model Papers: Multiple Model Families =============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    family_lists = col_filtered.apply(detect_algorithm_families)
    family_count = family_lists.apply(len)
    multi_model_mask = family_count > 1
    multi_model_count = int(multi_model_mask.sum())

    summary_df = count_percent_rows({"multi_model_papers": multi_model_count}, total_valid, category_col="Model Use Category")
    print(summary_df.to_string(index=False))

    match_table = pd.DataFrame({
        "algorithm_text": col_filtered,
        "algorithm_families_detected": family_lists.apply(lambda x: "; ".join(x)),
        "number_algorithm_families": family_count,
        "multi_model_paper": multi_model_mask,
    })

    save_df_optional(summary_df, "rq4_multi_model_papers_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_multi_model_papers_match_table.csv")
    return summary_df, match_table


multi_use_model_summary, multi_use_model_match_table = multi_model_papers(
    df_subset.iloc[:, COL_ALGORITHMS], valid_total
)

combined_algorithm_summary = pd.concat(
    [summary_algorithms_df, hybrid_models_summary.rename(columns={"Algorithm Category": "Algorithm Broad Category"})],
    ignore_index=True,
)
save_df_optional(combined_algorithm_summary, "rq4_algorithms_combined_summary.csv")


# ============================================================
# 12. EVALUATION METRICS
# ============================================================

def evaluation_metrics(col, valid_mask):
    patterns = {
        "accuracy": r"accuracy|balanced accuracy|classification accuracy",
        "specificity": r"specificity|\btnr\b|true negative rate",
        "sensitivity_recall": r"sensitivity|\btpr\b|true positive rate|\brecall\b",
        "precision_ppv": r"\bprecision\b|positive predictive value|\bppv\b",
        "f1_score": r"f[- ]?1|f1 score|f[- ]?measure|f measure",
        "auc_roc": r"\bauc\b|\broc\b|auc[- ]?roc|au[- ]?roc|auroc|area under the curve|receiver operating characteristic",
        "other_evaluation_reporting_metrics": (
            r"confusion matrix|confusion|classification report|error matrix"
            r"|error rate|error-rate|classification error|mae|mse|rmse|loss|cross[- ]?entropy"
            r"|matthews correlation coefficient|\bmcc\b|negative predictive value|\bnpv\b|\buar\b|kappa|g[- ]?mean|balanced error|diagnostic validity"
        ),
    }
    return summarize_patterns(
        col,
        valid_mask,
        patterns,
        label="Evaluation Metrics",
        category_col="Evaluation Metric",
        save_prefix="rq4_evaluation_metrics",
    )


evaluation_metrics_summary, evaluation_metrics_match_table = evaluation_metrics(
    df_subset.iloc[:, COL_EVALUATION_METRICS], valid_total
)


# ============================================================
# 13. ACCURACY EXTRACTION
# ============================================================

OTHER_METRIC_TERMS = [
    "sensitivity", "specificity", "recall", "precision", "auc", "auroc", "roc", "auc-roc", "auc roc", "au-roc",
    "confusion matrix", "f-1 score", "f-1", "f 1", "f1", "matthews correlation coefficient", "mcc",
    "error-rate", "error rate", "positive predictive value", "ppv", "negative predictive value", "npv",
    "uar", "tpr", "tnr", "kappa", "diagnostic validity", "f-measure", "f measure", "g-mean", "g mean",
    "loss", "mae", "mse", "rmse"
]


def has_other_metric_terms(text):
    text = str(text).lower()
    return any(term in text for term in OTHER_METRIC_TERMS)


def normalize_accuracy_number(x):
    try:
        value = float(x)
    except Exception:
        return np.nan
    if 0 < value <= 1:
        return value * 100
    if 1 < value <= 100:
        return value
    return np.nan


def extract_accuracy_from_row(row):
    performance_text = str(row.iloc[COL_BEST_PERFORMANCE]).lower()
    metric_text = str(row.iloc[COL_EVALUATION_METRICS]).lower()

    if "accuracy" in performance_text or re.search(r"\bacc\b", performance_text):
        patterns = [
            r"(?:balanced\s+accuracy|classification\s+accuracy|accuracy|\bacc\b)[^0-9]{0,80}(\d+(?:\.\d+)?)\s*%",
            r"(?:balanced\s+accuracy|classification\s+accuracy|accuracy|\bacc\b)[^0-9]{0,80}(0?\.\d+)",
            r"(?:balanced\s+accuracy|classification\s+accuracy|accuracy|\bacc\b)[^0-9]{0,80}(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*%\s*(?:balanced\s+accuracy|classification\s+accuracy|accuracy|\bacc\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, performance_text)
            if match:
                return normalize_accuracy_number(match.group(1))

    if "accuracy" in metric_text and not has_other_metric_terms(performance_text):
        numbers = re.findall(r"\d+(?:\.\d+)?", performance_text)
        if numbers:
            return normalize_accuracy_number(numbers[0])

    return np.nan


df_subset["extracted_accuracy_percent"] = df_subset.apply(extract_accuracy_from_row, axis=1)
df_valid = df_subset.loc[valid_total].copy()
accuracy_series = df_subset["extracted_accuracy_percent"]

accuracy_extraction_summary = pd.DataFrame([{
    "Valid rows": int(len(df_valid)),
    "Rows with extracted accuracy": int(df_valid["extracted_accuracy_percent"].notna().sum()),
    "Percentage with extracted accuracy": pct(int(df_valid["extracted_accuracy_percent"].notna().sum()), int(len(df_valid))),
}])

print("\n============= Accuracy Extraction Summary =============")
print(accuracy_extraction_summary.to_string(index=False))
save_df_optional(accuracy_extraction_summary, "rq4_accuracy_extraction_summary.csv")
save_df_optional(
    df_valid[[df_valid.columns[COL_EVALUATION_METRICS], df_valid.columns[COL_BEST_PERFORMANCE], "extracted_accuracy_percent"]]
    .reset_index()
    .rename(columns={"index": "row_index"}),
    "rq4_accuracy_extraction_rows.csv",
)


# ============================================================
# 14. RQ4.5 ACCURACY BY BEHAVIORAL MODALITY
# ============================================================

accuracy_by_modality_df = accuracy_by_behavioral_modality(
    modality_match_table=modality_match_table,
    accuracy_series=accuracy_series,
    output_dir=HELPER_OUTPUT_DIR,
)

print("\n============= Accuracy by Behavioral Modality =============")
print(accuracy_by_modality_df.to_string(index=False))


# ============================================================
# 15. RQ4.5 ACCURACY BY STUDY SETTING
# ============================================================

accuracy_by_setting_df = accuracy_by_study_setting(
    study_setting_match_table=study_setting_match_table,
    accuracy_series=accuracy_series,
    output_dir=HELPER_OUTPUT_DIR,
)

print("\n============= Accuracy by Study Setting =============")
print(accuracy_by_setting_df.to_string(index=False))


# ============================================================
# 16. RQ4.5 ACCURACY BY AI TECHNIQUE
# ============================================================

def classify_algorithm_group_for_accuracy(row_index, value):
    text = str(value).lower().strip()
    families = detect_algorithm_families(text)
    hybrid = bool(hybrid_model_mask.reindex([row_index]).fillna(False).iloc[0])

    if hybrid:
        return "hybrid model"
    if len(families) > 1:
        return "multiple model families"
    if len(families) == 1:
        return families[0]
    return "unclear"


df_valid["algorithm_families_detected"] = df_valid.iloc[:, COL_ALGORITHMS].apply(detect_algorithm_families)
df_valid["algorithm_group"] = [
    classify_algorithm_group_for_accuracy(idx, value)
    for idx, value in df_valid.iloc[:, COL_ALGORITHMS].items()
]
df_valid["number_algorithm_families"] = df_valid["algorithm_families_detected"].apply(len)

algorithm_group_accuracy_summary = summarize_accuracy_by_group(
    group_series=df_valid["algorithm_group"],
    accuracy_series=accuracy_series,
    group_col="AI technique group",
    save_prefix="rq4_accuracy_by_ai_technique_group",
    output_dir=HELPER_OUTPUT_DIR,
)

print("\n============= Accuracy by AI Technique Group =============")
print(algorithm_group_accuracy_summary.to_string(index=False))

all_algorithm_families = sorted({family for family_list in df_valid["algorithm_families_detected"] for family in family_list})
algorithm_family_match_table = pd.DataFrame(index=df_valid.index)

for family in all_algorithm_families:
    algorithm_family_match_table[family] = df_valid["algorithm_families_detected"].apply(lambda detected: family in detected)

algorithm_family_accuracy_summary = summarize_accuracy_by_flags(
    match_table=algorithm_family_match_table,
    accuracy_series=accuracy_series,
    flag_cols=all_algorithm_families,
    category_col="AI technique family",
    save_prefix="rq4_accuracy_by_ai_technique_family",
    output_dir=HELPER_OUTPUT_DIR,
)

print("\n============= Accuracy by AI Technique Family =============")
print(algorithm_family_accuracy_summary.to_string(index=False))

df_valid["is_multi_model_family"] = df_valid["number_algorithm_families"] > 1

multi_family_accuracy_summary = summarize_accuracy_by_group(
    group_series=df_valid["is_multi_model_family"].map({
        True: "multiple model families",
        False: "not multiple model families"
    }),
    accuracy_series=accuracy_series,
    group_col="Multiple model family status",
    save_prefix="rq4_accuracy_multiple_family_vs_not",
    output_dir=HELPER_OUTPUT_DIR,
)

hybrid_accuracy_summary = summarize_accuracy_by_group(
    group_series=(df_valid["algorithm_group"] == "hybrid model").map({True: "hybrid model", False: "not hybrid model"}),
    accuracy_series=accuracy_series,
    group_col="Hybrid model status",
    save_prefix="rq4_accuracy_hybrid_vs_nonhybrid",
    output_dir=HELPER_OUTPUT_DIR,
)



print("\n============= Hybrid Model Accuracy =============")
print(hybrid_accuracy_summary.to_string(index=False))

print("\n============= Multiple Model Families Accuracy =============")
print(multi_family_accuracy_summary.to_string(index=False))


# ============================================================
# 17. RQ4.5 ACCURACY BY TASK DESIGN / TASK TYPE
# ============================================================

accuracy_by_task_type_df = accuracy_by_task_type(
    task_type_match_table=task_type_match_table,
    accuracy_series=accuracy_series,
    output_dir=HELPER_OUTPUT_DIR,
)

print("\n============= Accuracy by Task Design / Task Type =============")
print(accuracy_by_task_type_df.to_string(index=False))

task_match_valid = task_type_match_table.loc[df_valid.index].copy()
exclusive_task_group = make_exclusive_task_group(task_match_valid)

exclusive_task_accuracy_summary = summarize_accuracy_by_group(
    group_series=exclusive_task_group,
    accuracy_series=accuracy_series,
    group_col="Exclusive task group",
    save_prefix="rq4_accuracy_by_task_design_exclusive",
    output_dir=HELPER_OUTPUT_DIR,
)

print("\n============= Exclusive Task Design Accuracy Summary =============")
print(exclusive_task_accuracy_summary.to_string(index=False))


# ============================================================
# 18. RQ4.5 ACCURACY BY ASD AGE GROUP
# ============================================================

if not asd_age_match_table.empty:
    accuracy_by_asd_age_df = accuracy_by_asd_age_group(
        asd_age_match_table=asd_age_match_table,
        accuracy_series=accuracy_series,
        output_dir=HELPER_OUTPUT_DIR,
    )
    print("\n============= Accuracy by ASD Age Group =============")
    print(accuracy_by_asd_age_df.to_string(index=False))
else:
    accuracy_by_asd_age_df = pd.DataFrame()
    print("\n============= Accuracy by ASD Age Group =============")
    print("Skipped: ASD age match table is empty.")


# ============================================================
# 19. FEATURE SELECTION / INTERPRETATION TECHNIQUE
# ============================================================

def compute_interpretation_methods(col, valid_mask):
    print("\n========= Feature Selection / Interpretation Counts ==========")
    patterns = {
        "statistical_feature_evaluation": r"t[- ]?test|ttest|anova|kruskal|wallis|kolmogorov|smirnov|mann|whitney|mwu|pearson|spearman|correlation|permutation|discriminative|p[- ]?value|statistical test",
        "filter_based_feature_selection": r"relief|relieff|information gain|info gain|mutual information|\bmic\b|maximal information coefficient|mrmr|cfs|fisher|fdr|chi[- ]?square|chi2|correlation[- ]?based",
        "wrapper_based_feature_selection": r"forward|backward|recursive|\brfe\b|svm[- ]?rfe|stepwise|swda|genetic|wrapper|boruta|sequential feature",
        "embedded_feature_importance": r"svm weights?|rf weights?|random forest|feature weights?|weights?|weighted|gini importance|tree importance|xgboost importance|lightgbm importance|lasso",
        "explainable_ai_model_interpretation": r"shap|shapley|lime|grad[- ]?cam|cam\b|integrated gradients|saliency|attention|attention map|feature importance|importance|ablation|leave[- ]?one[- ]?out|permutation importance|pdp|partial dependence|explainable|xai|interpretability",
        "dimensionality_reduction_representation_learning": r"pca|principal component|vae|autoencoder|latent|representation|dimensionality|embedding visualization|t[- ]?sne|umap",
        "ensemble_hybrid_feature_selection": r"ensemble|ensembled|voting|combined|combining|combination|hybrid|fusion based feature selection",
        "feature_engineering_exploratory_analysis": r"spatial|temporal|spatiotemporal|spatio-temporal|k[- ]?means|clustering|lmem|linear mixed|exploratory|feature engineering",
    }

    summary_df, match_table = summarize_patterns(
        col,
        valid_mask,
        patterns,
        label="Feature Selection / Interpretation Methods",
        category_col="Interpretation Method Category",
        save_prefix="rq4_feature_interpretation_methods",
    )

    total_valid = int(ensure_series_mask(valid_mask, col.index).sum())
    any_mask = match_table[list(patterns.keys())].any(axis=1)
    any_count = int(any_mask.sum())
    not_reported_count = total_valid - any_count

    overview = pd.DataFrame({
        "Category": ["Any interpretation / feature importance method reported", "Not reported / unclear"],
        "Count": [any_count, not_reported_count],
        "Total Valid Papers": [total_valid, total_valid],
        "Percentage of Total Papers": [pct(any_count, total_valid), pct(not_reported_count, total_valid)],
    })
    print("\n============= Feature Interpretation Reporting Overview =============")
    print(overview.to_string(index=False))
    save_df_optional(overview, "rq4_feature_interpretation_reporting_overview.csv")
    return summary_df, overview, match_table


interpretation_counts, interpretation_reporting_overview, interpretation_match_table = compute_interpretation_methods(
    df_subset.iloc[:, COL_FEATURE_IMPORTANCE_TECHNIQUE], valid_total
)


# ============================================================
# 20. FEATURE IMPORTANCE RESULTS
# ============================================================

def compute_feature_importance_result(col, valid_mask):
    print("\n========== Feature Importance Result ==================")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    total_valid_papers = int(valid_mask.sum())
    col_filtered = clean_text_series(col[valid_mask])

    valid_feature_result_mask = ~invalid_or_placeholder_mask(col_filtered)
    feature_importance_result_count = int(valid_feature_result_mask.sum())
    col_feature_results = col_filtered[valid_feature_result_mask]

    patterns = {
        "motor_and_kinematic_features": r"speed|velocity|acceleration|peak velocity|peak acceleration|movement|motion|kinematic|gait|stride|walking|grip|force|supination|gesture|grasp|head rotation|head motion|yaw|roll|amplitude|tablet|touch",
        "gaze_and_visual_attention_features": r"gaze|fixation|saccade|eye movement|eye tracking|scanpath|visual focus|attention|\baoi\b|object of interest|mouth|wholebody|monitor screen|gaze away|head-eye|eye contact|heatmap|saliency",
        "speech_and_acoustic_features": r"speech|acoustic|voice|vocal|prosody|pitch|fundamental frequency|\bf0\b|mfcc|rasta|mcep|hmpdm|hmpdd|rhythm|harmony-to-noise|hnr|sentiment|emotional|liwc|text feature|word|embedding",
        "facial_and_social_features": r"\bau\d+\b|inner[- ]?brow|lip[- ]?corner|facial|smiling|social smiling|facial mimicry|non[- ]?verbal|social|interaction|presence of face|presence of people|emotion",
        "multimodal_combination_features": r"fusion|fused|fusing|combined|combining|concatenat|integrat|multimodal|eeg and eye|spatial\+temporal|all features|feature combination|dual[- ]?stream|pairwise euclidean",
        "other_behavioral_features": r"entropy|visual focus|saliency|steerable|rgb|color|colour|intensity|orientation|horizon|frame center|scene center|biological movement|geometrical movement|session length|questionnaire|q[- ]?chat|score",
    }

    match_table = pd.DataFrame(index=col_feature_results.index)
    match_table["feature_importance_result_text"] = col_feature_results

    counts = {}
    for category, pattern in patterns.items():
        mask = col_feature_results.str.contains(pattern, regex=True, na=False)
        counts[category] = int(mask.sum())
        match_table[category] = mask

    rows = []
    for category, count in counts.items():
        rows.append({
            "Category": category,
            "Count": count,
            "Total Valid Papers": total_valid_papers,
            "Percentage out of total valid papers": pct(count, total_valid_papers),
            "Papers reporting feature importance results": feature_importance_result_count,
            "Percentage out of papers reporting feature importance results": pct(count, feature_importance_result_count),
        })

    rows.append({
        "Category": "Any valid feature importance result reported",
        "Count": feature_importance_result_count,
        "Total Valid Papers": total_valid_papers,
        "Percentage out of total valid papers": pct(feature_importance_result_count, total_valid_papers),
        "Papers reporting feature importance results": feature_importance_result_count,
        "Percentage out of papers reporting feature importance results": 100.00 if feature_importance_result_count else 0,
    })

    rows.append({
        "Category": "No valid feature importance result reported",
        "Count": total_valid_papers - feature_importance_result_count,
        "Total Valid Papers": total_valid_papers,
        "Percentage out of total valid papers": pct(total_valid_papers - feature_importance_result_count, total_valid_papers),
        "Papers reporting feature importance results": feature_importance_result_count,
        "Percentage out of papers reporting feature importance results": None,
    })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_feature_importance_results_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_feature_importance_results_match_table.csv")
    return summary_df, match_table


feature_importance_results_summary, feature_importance_results_match_table = compute_feature_importance_result(
    df_subset.iloc[:, COL_FEATURE_IMPORTANCE_RESULT], valid_total
)

# ============================================================
# 20. BIAS MITIGATION / BALANCING TECHNIQUES
# ============================================================

def compute_bias_mitigation(col, valid_mask):
    print("\n========= Bias Mitigation / Balancing Techniques ============")

    total_valid_papers = int(valid_mask.sum())
    col_filtered = clean_text_series(col[valid_mask])

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
        r"|^\s*none\s*$"
        r"|^\s*not reported\s*$"
        r"|^\s*not given\s*$"
        r"|^\s*not specified\s*$"
        r"|^\s*not applicable\s*$"
        r"|^\s*unclear\s*$"
        r"|^\s*unknown\s*$"
    )

    reported_no_pattern = (
        r"^\s*no\s*$"
        r"|^\s*no[,.]"
        r"|^\s*no "
        r"|not used"
        r"|not applied"
        r"|not performed"
        r"|no balancing"
        r"|no bias mitigation"
        r"|no class balancing"
        r"|did not use"
        r"|without balancing"
        r"|without bias mitigation"
    )

    patterns = {
        "smote": (
            r"\bsmote\b"
            r"|synthetic minority"
        ),

        "adasyn_or_synthetic_sampling": (
            r"\badasyn\b"
            r"|synthetic sample\w*"
            r"|synthetic data"
            r"|synthetic example\w*"
            r"|data synthesis"
        ),

        "class_weights_or_cost_sensitive_learning": (
            r"class weight\w*"
            r"|weighted loss"
            r"|cost[- ]?sensitive"
            r"|sample weight\w*"
            r"|focal loss"
            r"|balanced loss"
            r"|weighted cross[- ]?entropy"
        ),

        "under_or_over_sampling": (
            r"under[- ]?sampling"
            r"|undersampling"
            r"|over[- ]?sampling"
            r"|oversampling"
            r"|random under"
            r"|random over"
            r"|minority oversampling"
            r"|majority undersampling"
        ),

        "data_augmentation": (
            r"data augmentation"
            r"|augment\w*"
            r"|rotation"
            r"|flip"
            r"|noise injection"
            r"|cropping"
            r"|gan augmentation"
        ),

        "balanced_split_or_matching": (
            r"balanced dataset"
            r"|balanced data set"
            r"|balanced groups"
            r"|matched groups"
            r"|age[- ]?matched"
            r"|gender[- ]?matched"
            r"|sex[- ]?matched"
            r"|matched controls"
            r"|similar ratio"
        ),
    }

    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["bias_mitigation_text"] = col_filtered

    not_reported_mask = col_filtered.str.contains(
        not_reported_pattern,
        regex=True,
        na=False
    )

    reported_no_mask = (
        ~not_reported_mask
        & col_filtered.str.contains(reported_no_pattern, regex=True, na=False)
    )

    match_table["not_reported_or_missing"] = not_reported_mask
    match_table["reported_no_bias_mitigation"] = reported_no_mask

    counts = {}

    for category, pattern in patterns.items():
        mask = (
            ~not_reported_mask
            & ~reported_no_mask
            & col_filtered.str.contains(pattern, regex=True, na=False)
        )

        match_table[category] = mask
        counts[category] = int(mask.sum())

    technique_cols = list(patterns.keys())

    any_technique_mask = match_table[technique_cols].any(axis=1)

    valid_text_but_uncategorized_mask = (
        ~not_reported_mask
        & ~reported_no_mask
        & ~any_technique_mask
    )

    match_table["any_bias_mitigation_or_balancing_technique"] = any_technique_mask
    match_table["valid_text_but_uncategorized"] = valid_text_but_uncategorized_mask

    any_technique_count = int(any_technique_mask.sum())
    not_reported_count = int(not_reported_mask.sum())
    reported_no_count = int(reported_no_mask.sum())
    uncategorized_count = int(valid_text_but_uncategorized_mask.sum())

    rows = []

    for category, count in counts.items():
        rows.append({
            "Category": category,
            "Count": count,
            "Total Valid Papers": total_valid_papers,
            "Percentage out of total valid papers": pct(count, total_valid_papers),
            "Papers reporting any bias mitigation / balancing technique": any_technique_count,
            "Percentage out of papers reporting any technique": pct(count, any_technique_count),
        })

    rows.extend([
        {
            "Category": "Any bias mitigation / balancing technique reported",
            "Count": any_technique_count,
            "Total Valid Papers": total_valid_papers,
            "Percentage out of total valid papers": pct(any_technique_count, total_valid_papers),
            "Papers reporting any bias mitigation / balancing technique": any_technique_count,
            "Percentage out of papers reporting any technique": 100.00 if any_technique_count else 0,
        },
        {
            "Category": "Reported no bias mitigation / balancing",
            "Count": reported_no_count,
            "Total Valid Papers": total_valid_papers,
            "Percentage out of total valid papers": pct(reported_no_count, total_valid_papers),
            "Papers reporting any bias mitigation / balancing technique": any_technique_count,
            "Percentage out of papers reporting any technique": None,
        },
        {
            "Category": "Not reported / missing",
            "Count": not_reported_count,
            "Total Valid Papers": total_valid_papers,
            "Percentage out of total valid papers": pct(not_reported_count, total_valid_papers),
            "Papers reporting any bias mitigation / balancing technique": any_technique_count,
            "Percentage out of papers reporting any technique": None,
        },
        {
            "Category": "Valid text but uncategorized",
            "Count": uncategorized_count,
            "Total Valid Papers": total_valid_papers,
            "Percentage out of total valid papers": pct(uncategorized_count, total_valid_papers),
            "Papers reporting any bias mitigation / balancing technique": any_technique_count,
            "Percentage out of papers reporting any technique": None,
        },
    ])

    summary_df = pd.DataFrame(rows)

    print(summary_df.to_string(index=False))

    print("\n========= Valid but Uncategorized Bias Mitigation Rows =========")
    uncategorized_rows = match_table.loc[
        match_table["valid_text_but_uncategorized"],
        ["bias_mitigation_text"]
    ]

    if uncategorized_rows.empty:
        print("No uncategorized rows.")
    else:
        print(uncategorized_rows.to_string())

    save_df_optional(summary_df, "rq4_bias_mitigation_summary.csv")
    save_df_optional(
        match_table.reset_index().rename(columns={"index": "row_index"}),
        "rq4_bias_mitigation_match_table.csv"
    )
    save_df_optional(
        uncategorized_rows.reset_index().rename(columns={"index": "row_index"}),
        "rq4_bias_mitigation_uncategorized_rows.csv"
    )

    return summary_df, match_table, uncategorized_rows


bias_mitigation_summary, bias_mitigation_match_table, bias_mitigation_uncategorized_rows = compute_bias_mitigation(
    df_subset.iloc[:, COL_BALANCING_TECHNIQUE],
    valid_total
)

# ============================================================
# 22. CROSS-CORPUS / EXTERNAL DATASET VALIDATION
# ============================================================

def compute_cross_dataset_validation(col, valid_mask):
    print("\n========= Cross-Corpus / External Dataset Validation ============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    total_valid_papers = int(valid_mask.sum())
    col_filtered = clean_text_series(col[valid_mask])

    valid_cross_dataset_value_mask = ~invalid_or_placeholder_mask(col_filtered)
    cross_dataset_value_count = int(valid_cross_dataset_value_mask.sum())
    col_cross_dataset_results = col_filtered[valid_cross_dataset_value_mask]

    pattern_strict_cross_corpus = (
        r"^yes$|cross[-\s]?corpus|cross[-\s]?dataset|cross[-\s]?site|leave[-\s]?one[-\s]?dataset[-\s]?out|leave[-\s]?one[-\s]?site[-\s]?out|loxo|loso"
        r"|train(?:ed)?\s+on\s+.*test(?:ed)?\s+on|train(?:ed)?\s+on\s+.*evaluat(?:ed|ion)\s+on|external validation"
    )
    pattern_external_multidataset = (
        r"second dataset|secondary dataset|independent dataset|external dataset|separate dataset|another dataset|two datasets|multiple datasets|multi[- ]?dataset"
        r"|preliminary dataset|retrain|retrained|retraining|compare results|compared results|combining|combined dataset|separate cohort|independent cohort|different cohort"
        r"|actual autistic population|asd population|never seen during train|never seen during training|held[- ]?out dataset|unseen dataset|unseen site|out[- ]?of[- ]?sample|generalization dataset|generalisation dataset"
    )
    pattern_no = r"^no$|^no,|^no\.|^no "

    strict_mask = col_cross_dataset_results.str.contains(pattern_strict_cross_corpus, regex=True, na=False)
    external_mask = col_cross_dataset_results.str.contains(pattern_external_multidataset, regex=True, na=False)
    no_mask = col_cross_dataset_results.str.contains(pattern_no, regex=True, na=False)
    any_mask = strict_mask | external_mask
    categorized_mask = strict_mask | external_mask | no_mask

    summary_df = pd.DataFrame({
        "Category": [
            "Strict cross-corpus validation",
            "External dataset / multi-dataset validation",
            "Any cross-dataset generalizability evaluation",
            "Reported no cross-dataset validation",
            "Other valid but unclear entry",
            "Any valid entry in cross-corpus column",
            "No valid entry in cross-corpus column",
        ],
        "Count": [
            int(strict_mask.sum()),
            int(external_mask.sum()),
            int(any_mask.sum()),
            int(no_mask.sum()),
            int((~categorized_mask).sum()),
            cross_dataset_value_count,
            total_valid_papers - cross_dataset_value_count,
        ],
    })
    summary_df["Percentage out of total valid papers"] = (summary_df["Count"] / total_valid_papers * 100).round(2) if total_valid_papers else 0
    summary_df["Percentage out of papers with valid cross-dataset entries"] = (
        summary_df["Count"] / cross_dataset_value_count * 100 if cross_dataset_value_count else 0
    ).round(2)
    summary_df.loc[summary_df["Category"] == "Any valid entry in cross-corpus column", "Percentage out of papers with valid cross-dataset entries"] = 100.00
    summary_df.loc[summary_df["Category"] == "No valid entry in cross-corpus column", "Percentage out of papers with valid cross-dataset entries"] = None

    match_table = pd.DataFrame({
        "cross_dataset_text": col_cross_dataset_results,
        "strict_cross_corpus_validation": strict_mask,
        "external_dataset_multi_dataset_validation": external_mask,
        "reported_no_cross_dataset_validation": no_mask,
        "other_valid_unclear": ~categorized_mask,
    })

    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_cross_dataset_validation_summary.csv")
    save_df_optional(match_table.reset_index().rename(columns={"index": "row_index"}), "rq4_cross_dataset_validation_match_table.csv")
    return summary_df, match_table


cross_dataset_summary, cross_dataset_match_table = compute_cross_dataset_validation(
    df_subset.iloc[:, COL_CROSS_CORPUS_VALIDATION], valid_total
)


# ============================================================
# 23. X-FOLD CROSS-VALIDATION
# ============================================================

def compute_cross_validation(col, valid_mask):
    print("\n========= Cross Validation =============")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    pattern_cross_validation = r"^yes$|cross[- ]?validation|\bcv\b|k[- ]?fold|\d+[- ]?fold|folds?\b|leave[- ]?one[- ]?out|loocv|stratified k[- ]?fold|nested cross[- ]?validation|subject[- ]?independent cross[- ]?validation"
    cross_validation_mask = col_filtered.str.contains(pattern_cross_validation, regex=True, na=False)
    cross_validation_count = int(cross_validation_mask.sum())

    summary_df = count_percent_rows({"cross_validation_reported": cross_validation_count}, total_valid, category_col="Validation Category")
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_cross_validation_summary.csv")
    save_df_optional(pd.DataFrame({"cross_validation_text": col_filtered, "cross_validation_reported": cross_validation_mask}).reset_index().rename(columns={"index": "row_index"}), "rq4_cross_validation_match_table.csv")
    return summary_df, cross_validation_mask


cross_validation_summary, cross_validation_mask = compute_cross_validation(
    df_subset.iloc[:, COL_XFOLD_CV], valid_total
)


# ============================================================
# 24. LOXO
# ============================================================

def compute_LOXO(col, valid_mask):
    print("\n=========== LOXO ===========")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    pattern_LOXO = r"^yes$|\bloxo\b|\bloocv\b|leave[- ]?one[- ]?out|leave[- ]?one[- ]?subject[- ]?out|leave[- ]?one[- ]?site[- ]?out|leave[- ]?one[- ]?dataset[- ]?out|loso|lopo"
    loxo_mask = col_filtered.str.contains(pattern_LOXO, regex=True, na=False)
    loxo_count = int(loxo_mask.sum())

    summary_df = count_percent_rows({"LOXO_reported": loxo_count}, total_valid, category_col="LOXO Category")
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_loxo_summary.csv")
    save_df_optional(pd.DataFrame({"loxo_text": col_filtered, "loxo_reported": loxo_mask}).reset_index().rename(columns={"index": "row_index"}), "rq4_loxo_match_table.csv")
    return summary_df, loxo_mask


LOXO_summary, LOXO_mask = compute_LOXO(df_subset.iloc[:, COL_LOXO], valid_total)


# ============================================================
# 25. REAL-TIME ANALYSIS
# ============================================================

def compute_real_time_analysis(col, valid_mask):
    print("\n=========== Real Time Analysis ===========")
    valid_mask = ensure_series_mask(valid_mask, col.index)
    col_filtered = clean_text_series(col[valid_mask])
    total_valid = int(valid_mask.sum())

    pattern_rta = r"^yes$|real[- ]?time|online inference|live|real time analysis|real[- ]?time analysis|real[- ]?time detection|real[- ]?time screening|app[- ]?based|mobile app|deployed|deployment"
    rta_mask = col_filtered.str.contains(pattern_rta, regex=True, na=False)
    rta_count = int(rta_mask.sum())

    summary_df = count_percent_rows({"real_time_analysis_reported": rta_count}, total_valid, category_col="Real-Time Analysis Category")
    print(summary_df.to_string(index=False))
    save_df_optional(summary_df, "rq4_real_time_analysis_summary.csv")
    save_df_optional(pd.DataFrame({"real_time_text": col_filtered, "real_time_analysis_reported": rta_mask}).reset_index().rename(columns={"index": "row_index"}), "rq4_real_time_analysis_match_table.csv")
    return summary_df, rta_mask


Real_Time_Analysis_summary, Real_Time_Analysis_mask = compute_real_time_analysis(
    df_subset.iloc[:, COL_REAL_TIME_ANALYSIS], valid_total
)


# ============================================================
# 26. FINAL CHECK
# ============================================================

print("\n============= RQ4 FINAL CHECK =============")
print("Total valid papers:", int(valid_total.sum()))
print("RQ4 output directory:", OUTPUT_DIR.resolve())
print("Column map saved as:", OUTPUT_DIR / "rq4_column_map.csv")

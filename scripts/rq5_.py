
import os
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from setup_data_ import load_annotation_data, INVALID_VALUES
from helper_functions_ import (
    AGE_CATEGORY_COLS,
    clean_text_series,
    compute_asd_age_ranges,
    compute_task_type,
    ensure_series_mask,
    save_df_optional,
    TASK_TYPE_COLS,
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
OUTPUT_DIR = OUTPUT_ROOT / "rq5_results"
if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_output_name_prefix("RQ5")


def save_df(df_to_save: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Save only when SAVE_OUTPUTS=True."""
    if SAVE_OUTPUTS:
        save_df_optional(df_to_save, filename, output_dir=str(OUTPUT_DIR), index=index)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = load_annotation_data()

df = data["df"]
df_subset = data["df_subset"].copy()

valid_total = ensure_series_mask(data["valid_total"], df_subset.index)
valid_ASD = ensure_series_mask(data.get("valid_ASD", data.get("valid_mask_ASD")), df_subset.index)
valid_Neur = ensure_series_mask(data.get("valid_Neur", pd.Series(False, index=df_subset.index)), df_subset.index)
valid_Other = ensure_series_mask(data.get("valid_Other", pd.Series(False, index=df_subset.index)), df_subset.index)

valid_papers_Total = int(data.get("valid_papers_Total", valid_total.sum()))
valid_papers_ASD = int(data.get("valid_papers_ASD", valid_ASD.sum()))
valid_papers_Neur = int(data.get("valid_papers_Neur", valid_Neur.sum()))
valid_papers_Other = int(data.get("valid_papers_Other", valid_Other.sum()))


# ============================================================
# 2. CURRENT DATA-WORKSHEET COLUMN MAP
# ============================================================
# Python uses zero-based iloc indexing.
# Everything below is read from the current `data` worksheet / df_subset.
#
# A  / 0  = Title
# C  / 2  = Keywords
# D  / 3  = Journal / venue
# E  / 4  = Publication year
# N  / 13 = ASD label
# O  / 14 = ASD age range
# P  / 15 = ASD mean age
# Q  / 16 = ASD SD age
# AO / 40 = Gaze data
# AP / 41 = Speech/language/audio data
# AQ / 42 = Motor data
# AR / 43 = Other behavioral data
# AS / 44 = Other type of data
# AT / 45 = Feature fusion / feature combination method
# BP / 67 = Participant task / protocol

COL_TITLE = 0
COL_KEYWORDS = 2
COL_JOURNAL = 3
COL_PUBLICATION_YEAR = 4
COL_ASD_LABEL = 13
COL_ASD_AGE_RANGE = 14
COL_ASD_MEAN_AGE = 15
COL_ASD_SD_AGE = 16
COL_GAZE = 40
COL_SPEECH = 41
COL_MOTOR = 42
COL_OTHER_BEHAVIORAL = 43
COL_OTHER_TYPE_DATA = 44
COL_FEATURE_FUSION = 45
COL_TASK_PARTICIPANTS = 67

COLUMN_MAP = pd.DataFrame([
    {"Variable": "Title", "Excel Column": "A", "Python iloc Index": COL_TITLE},
    {"Variable": "Keywords", "Excel Column": "C", "Python iloc Index": COL_KEYWORDS},
    {"Variable": "Journal / venue", "Excel Column": "D", "Python iloc Index": COL_JOURNAL},
    {"Variable": "Publication year", "Excel Column": "E", "Python iloc Index": COL_PUBLICATION_YEAR},
    {"Variable": "ASD label", "Excel Column": "N", "Python iloc Index": COL_ASD_LABEL},
    {"Variable": "ASD age range", "Excel Column": "O", "Python iloc Index": COL_ASD_AGE_RANGE},
    {"Variable": "ASD mean age", "Excel Column": "P", "Python iloc Index": COL_ASD_MEAN_AGE},
    {"Variable": "ASD SD age", "Excel Column": "Q", "Python iloc Index": COL_ASD_SD_AGE},
    {"Variable": "Gaze data", "Excel Column": "AO", "Python iloc Index": COL_GAZE},
    {"Variable": "Speech/language/audio data", "Excel Column": "AP", "Python iloc Index": COL_SPEECH},
    {"Variable": "Motor data", "Excel Column": "AQ", "Python iloc Index": COL_MOTOR},
    {"Variable": "Other behavioral data", "Excel Column": "AR", "Python iloc Index": COL_OTHER_BEHAVIORAL},
    {"Variable": "Other type of data", "Excel Column": "AS", "Python iloc Index": COL_OTHER_TYPE_DATA},
    {"Variable": "Feature fusion / feature combination", "Excel Column": "AT", "Python iloc Index": COL_FEATURE_FUSION},
    {"Variable": "Participant task / protocol", "Excel Column": "BP", "Python iloc Index": COL_TASK_PARTICIPANTS},
])

print("\n============= COLUMN MAP =============")
print(COLUMN_MAP.to_string(index=False))
save_df(COLUMN_MAP, "column_map.csv")


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

GLOBAL_INVALID_VALUES = set(INVALID_VALUES)


def pct(count, denominator):
    return round((int(count) / int(denominator)) * 100, 2) if int(denominator) else 0.0


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\xa0", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def is_global_invalid(value) -> bool:
    return normalize_text(value) in GLOBAL_INVALID_VALUES


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


def extract_publication_year(value):
    text = str(value).strip()
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else np.nan


def classify_year_group(year):
    if pd.isna(year):
        return "Missing / unreadable year"
    year = int(year)
    if 2013 <= year <= 2017:
        return "2013-2017"
    if 2018 <= year <= 2023:
        return "2018-2023"
    if 2024 <= year <= 2026:
        return "2024-2026"
    return "Outside expected range"


YEAR_GROUP_ORDER = [
    "2013-2017",
    "2018-2023",
    "2024-2026",
    "Missing / unreadable year",
    "Outside expected range",
]


# ============================================================
# 4. KEYWORD SUBGROUP FREQUENCY ANALYSIS
# ============================================================


def clean_keyword_text(text):
    """Clean keyword cells so regex matching is more accurate."""
    text = str(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    replacements = {
        "disorderobject": "disorder object",
        "disordermotor": "disorder motor",
        "engineering controlled termsclassification": "engineering controlled terms classification",
        "speechengineering": "speech engineering",
        "deeplearning": "deep learning",
        "machine-learning": "machine learning",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[;,\n\r\t·]+", " ; ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_keyword_subgroup_counts_accurate(col, valid_mask):
    print("\n========= Keyword Subgroup Frequency Analysis ============")

    valid_mask = ensure_series_mask(valid_mask, col.index)
    total_valid_papers = int(valid_mask.sum())
    col_filtered = col[valid_mask].astype(str).apply(clean_keyword_text)

    keyword_subgroups = {
        "Core autism terms": r"\bautism\b|\basd\b|\bautism spectrum disorder(?:s)?\b|\bautism spectrum condition\b|\bhigh[-\s]?functioning autism\b",
        "High-risk / development-related terms": r"\bhigh[-\s]?risk\b|\bdevelopmental disorder\b|\bdevelopmental delay\b|\bneurodevelopmental disorder\b|\btypical development\b|\btypically developing\b",
        "Other diagnostic or comparison-group terms": r"\badhd\b|\bdown syndrome\b|\bmental retardation\b|\bintellectual disability\b",
        "Combined autism + task terms": r"\bautism screening\b|\basd screening\b|\basd diagnosis\b|\bautism diagnosis\b|\bautism detection\b|\bscreening of asd\b",
        "Participant age / population terms": r"\bchildren\b|\bchild\b|\badolescent\b|\badolescence\b|\bschool[-\s]?age\b|\binfant\b|\btoddler\b|\badult\b",
        "General AI / ML terms": r"\bmachine learning\b|\bml\b|\bai\b|\bdeep learning\b|\bartificial intelligence\b|\bsupervised machine learning\b|\bcomputer vision\b|\bcomputational intelligence\b|\brepresentation learning\b|\bpattern recognition\b|\bmultimodal\b|\btransfer learning\b",
        "ML model types / algorithm names": r"\bartificial neural network\b|\bneural network(?:s)?\b|\bdeep neural network(?:s)?\b|\bsupport vector machine(?:s)?\b|\bsvm\b|\brandom forest\b|\bdecision tree(?:s)?\b|\bcnn\b|\bconvolutional neural network\b|\blstm\b|\bbert\b|\bpomdp\b|\bk[-\s]?means\b|\bhmm\b|\bgan\b|\bauto[-\s]?encoder\b|\bpnn\b|\bkelm\b|\bensemble\b",
        "Feature extraction / signal-processing / computational-analysis terms": r"\bfeature selection\b|\bfeature extraction\b|\bdimensionality reduction\b|\bdata mining\b|\battention mechanism\b|\blinear discriminant analysis\b|\bindependent component analysis\b|\bentropy\b|\bspectrum analysis\b|\bsignal processing\b|\btime[-\s]?frequency\b|\bsaliency\b|\bdynamic viewing patterns\b|\bfixation time\b",
        "Data generation / synthetic-data terms": r"\bvirtual sample generation\b|\bsynthetic\b|\bnormal gaussian distribution\b|\bmega trend diffusion\b|\bfunctional virtual population\b",
        "Model evaluation / validation terms": r"\bnested cross[-\s]?validation\b|\bcross[-\s]?validation\b|\bloocv\b|\bloxo\b|\bleave[-\s]?one",
        "Human-computer / digital interaction terms": r"\bhuman[-\s]?computer interaction\b|\bhuman[-\s]?robot interaction\b|\biot\b|\bvirtual reality\b|\bvr\b|\bsemantic web\b",
        "Mobile health / digital health / assistive technology terms": r"\bmhealth\b|\bmobile health\b|\behealth\b|\bmobile diagnostics\b|\bmobile app\b|\bapp\b|\bandroid\b|\bdigital phenotyping\b|\bwearable\b|\bgaming\b",
        "Robotics / robot-mediated intervention terms": r"\brobot\b|\brobotics\b|\bhumanoid robot\b|\bnao\b|\bchild[-\s]?robot interaction\b",
        "Gaze / eye-movement feature terms": r"\bscanpath\b|\bscan path\b|\bsaccade\b|\beye[-\s]?gaze\b|\beye movement\b|\bgaze\b|\beye contact\b|\bvisual fixation\b|\bfixation\b|\bvisual attention\b|\bpupil\b|\bjoint attention\b|\bdynamic viewing patterns\b",
        "Eye-tracking method terms": r"\beye[-\s]?tracking\b|\beye tracking\b",
        "Motion / motor / gait terms": r"\bmotor\b|\bkinematic\b|\bkinematics\b|\bmotion\b|\bbody movement\b|\bhead movement\b|\bhand gesture\b|\bpostural\b|\bbalance\b|\bforce plate\b|\bgait\b|\bwalking\b",
        "Speech / audio / language terms": r"\bspeech\b|\baudio\b|\bvoice\b|\bpitch\b|\bprosody\b|\bacoustic\b|\blinguistic\b|\blanguage\b|\bcommunication\b|\bvocali[sz]ation\b|\btweet\b|\bdisfluency\b",
        "Social interaction / social behavior terms": r"\bresponse to name\b|\brtn\b|\bimitation\b|\bsocial behavior\b|\bsocial behaviour\b|\brepetitive behavior\b|\bparent[-\s]?child interaction\b|\bengagement\b|\bhuman behaviou?r\b",
        "Face / emotion / affect terms": r"\bface\b|\bfacial\b|\bemotion recognition\b|\bfacial emotion\b|\bstill[-\s]?face\b|\bopensmile\b|\begemaps\b|\baffective\b",
        "Neuroimaging / physiological signal terms": r"\beeg\b|\belectroencephalography\b|\belectroencephalogram\b|\bmri\b|\bfmri\b|\bphysiological\b",
        "Video / image / digital behavioral data terms": r"\bvideo\b|\bvideos\b|\bimage processing\b|\bcamera\b|\bcomputer games\b|\bweb\b|\bdigital data\b|\bbehavioral biomarker\b|\bbiomarker\b",
        "General diagnosis / screening / assessment terms": r"\bdiagnosis\b|\bdiagnostic\b|\bscreening\b|\bdetection\b|\bclassification\b|\bidentification\b|\bprediction\b|\bassessment\b|\btherapy\b|\bintervention\b|\bseverity\b",
        "Specific diagnosis / screening terms": r"\bearly detection\b|\bearly diagnosis\b|\basd diagnosis\b|\bai[-\s]?assisted diagnosis\b|\bautism screening\b|\bautomatic diagnosis\b|\bautomatic classification\b|\bauxiliary diagnosis\b|\bclassifier\b",
        "Diagnosis-related tools / instruments": r"\bados\b|\bautism diagnostic observation schedule\b|\bmobile diagnostics\b",
        "Terms combining ML and clinical aim": r"\bai[-\s]?assisted diagnosis\b|\bearly detection model\b|\bmodel for early screening\b|\bautomatic classification\b|\bautomatic diagnosis\b|\bauxiliary diagnosis\b|\bscreening and machine learning\b",
        "Data / dataset / sample-size terms": r"\bdatabase\b|\bdataset\b|\bdata visualization\b|\bdata processing\b|\bsmall dataset\b|\bsmall data\b|\blarge dataset\b|\bheterogeneity\b",
        "Other / extra terms": r"\btrajectory\b|\bnarrative\b|\bobject detection\b|\boutcome measure\b|\bpattern\b|\battention\b|\btask analysis\b|\bvisualization\b",
    }

    match_table = pd.DataFrame(index=col_filtered.index)
    match_table["keyword_text"] = col_filtered

    rows = []
    for subgroup, pattern in keyword_subgroups.items():
        mask = col_filtered.str.contains(pattern, regex=True, na=False)
        match_table[subgroup] = mask
        count = int(mask.sum())
        rows.append({
            "Subgroup": subgroup,
            "Count": count,
            "Total Valid Papers": total_valid_papers,
            "Percentage": pct(count, total_valid_papers),
        })

    summary_df = pd.DataFrame(rows).sort_values("Count", ascending=False)

    print("\nTotal valid papers:", total_valid_papers)
    print(summary_df.to_string(index=False))

    save_df(summary_df, "keyword_subgroup_summary.csv")
    save_df(match_table.reset_index().rename(columns={"index": "row_index"}), "keyword_subgroup_match_table.csv")
    return summary_df, match_table


keyword_subgroup_summary, keyword_subgroup_match_table = compute_keyword_subgroup_counts_accurate(
    col=df_subset.iloc[:, COL_KEYWORDS],
    valid_mask=valid_total,
)
# ============================================================
# 5. VENUE / JOURNAL CATEGORY COUNTS
# ============================================================

def clean_venue_text(text):
    text = normalize_text(text)
    for ch in ["&", ".", ",", ":", ";", "-", "(", ")"]:
        text = text.replace(ch, " and " if ch == "&" else " ")
    return re.sub(r"\s+", " ", text).strip()


def compute_venue_category_counts_exclusive(col, valid_mask):
    print("\n========= Mutually Exclusive Venue Category Counts ============")

    valid_mask = ensure_series_mask(valid_mask, col.index)
    total_valid_papers = int(valid_mask.sum())

    # Keep BOTH raw and cleaned venue text
    col_raw = col[valid_mask].copy()
    col_cleaned = col_raw.astype(str).apply(clean_venue_text)

    venue_categories = [
        (
            "Autism-specific venues",
            r"\bjournal of autism and developmental disorders\b"
            r"|\bj autism dev disord\b"
            r"|\bautism research\b"
            r"|\bmolecular autism\b"
            r"|\binternational society for autism research\b"
            r"|\binsar\b",
        ),
        (
            "Broader clinical / psychology / neuroscience venues",
            r"\bfrontiers in psychology\b"
            r"|\bfrontiers in psychiatry\b"
            r"|\bfrontiers in pediatrics\b"
            r"|\bfrontiers in human neuroscience\b"
            r"|\bfrontiers in neurology\b"
            r"|\bfrontiers in medicine\b"
            r"|\bfrontiers in computational neuroscience\b"
            r"|\bfrontiers in neuroscience\b"
            r"|\bpsychology\b"
            r"|\bpsychiatry\b"
            r"|\bpediatrics\b"
            r"|\bpediatr\b"
            r"|\bbiological psychiatry\b"
            r"|\bcognitive neuroscience\b"
            r"|\bneuroscience\b"
            r"|\bjournal of clinical medicine\b"
            r"|\bbmj open\b"
            r"|\bplos one\b"
            r"|\bscientific reports\b"
            r"|\bj child adolesc psychiatr nurs\b"
            r"|\bchild adolesc psychiatr nurs\b"
            r"|\bpsych journal\b"
            r"|\bjama pediatr\b"
            r"|\bjama pediatrics\b"
            r"|\bjama network open\b"
            r"|\binternational journal of language and communication disorders\b"
            r"|\binternational journal of psychophysiology\b"
            r"|\bcognition\b"
            r"|\bjournal of pioneering medical sciences\b"
            r"|\bjournal of neurodevelopmental disorders\b"
            r"|\bchildren\b",
        ),
        (
            "Digital health / biomedical engineering venues",
            r"\bjournal of medical internet research\b"
            r"|\bj med internet res\b"
            r"|\bjmir\b"
            r"|\bnpj digital medicine\b"
            r"|\bfrontiers in digital health\b"
            r"|\bfront digit health\b"
            r"|\bdigital medicine\b"
            r"|\bhealth informatics\b"
            r"|\bjournal of healthcare engineering\b"
            r"|\bmedical image computing\b"
            r"|\bbiomedical engineering\b"
            r"|\bengineering in medicine and biology\b"
            r"|\bbioinformatics\b"
            r"|\bcomputational biology\b"
            r"|\bdiagnostics\b"
            r"|\bsensors\b"
            r"|\bcomputers in biology and medicine\b"
            r"|\bcomputers in biology and medicin\b"
            r"|\bjournal of biomechanics\b"
            r"|\bintelligence based medicine\b"
            r"|\birbm\b",
        ),
        (
            "Robotics / HRI venues",
            r"\bhuman robot interaction\b"
            r"|\bhri\b"
            r"|\bfrontiers in robotics and ai\b"
            r"|\binternational journal of social robotics\b"
            r"|\brobotics\b"
            r"|\bintelligent robotic\b"
            r"|\bjournal of intelligent and robotic systems\b",
        ),
        (
            "Technology / AI / engineering venues",
            r"\bieee\b"
            r"|\bacm\b"
            r"|\bmachine learning\b"
            r"|\bartificial intelligence\b"
            r"|\bneurocomputing\b"
            r"|\bapplied soft computing\b"
            r"|\balgorithms\b"
            r"|\bcomputer science\b"
            r"|\bcomputing\b"
            r"|\bintelligent systems\b"
            r"|\bpattern recognition\b"
            r"|\bsignal processing\b"
            r"|\bsignal image and video processing\b"
            r"|\bsignal process image commun\b"
            r"|\bmultimedia\b"
            r"|\bcomputer vision\b"
            r"|\binterspeech\b"
            r"|\belectronics\b"
            r"|\bapplied sciences\b"
            r"|\binternational journal of advanced computer science and applications\b"
            r"|\bsound and vibration\b"
            r"|\binternational journal of control and automation\b"
            r"|\bexpert systems with applications\b"
            r"|\balexandria engineering journal\b"
            r"|\bcomputers materials and continua\b"
            r"|\bjournal of computational mathematics and data science\b"
            r"|\btraitement du signal\b"
            r"|\bsadhana\b"
            r"|\badvances in human computer interaction\b"
            r"|\bengineering research express\b"
            r"|\binformation\b"
            r"|\bintelligent decision technologies\b"
            r"|\bbulletin of electrical engineering and informatics\b"
            r"|\bengineering technology and applied science research\b"
            r"|\binternational journal of intelligent engineering systems\b"
            r"|\bcomputer modeling in engineering and sciences\b"
            r"|\barray\b",
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
            r"|\blrec\b"
            r"|\bmlsp\b"
            r"|\btencon\b",
        ),
        (
            "Other / multidisciplinary venues",
            r"\bmdpi sustainability\b"
            r"|\bsustainability\b",
        ),
    ]

    non_venue_pattern = (
        r"\beye tracking\b.*\bfeature selection\b"
        r"|\bfeature selection\b.*\bneural network\b"
        r"|\bgroup selection\b.*\bstimuli selection\b"
        r"|\bparadigm selection\b.*\bneural network\b"
        r"|\bscience direct\b"
        r"|\bsciencedirect\b"
        r"|\bweb of science\b"
    )

    category_names = [name for name, _ in venue_categories] + [
        "Other / unclear venues",
        "Likely non-venue row / keyword string",
    ]

    counts = {category: 0 for category in category_names}
    assigned_rows = []

    for original_index, venue_cleaned in col_cleaned.items():
        venue_raw = col_raw.loc[original_index]

        if is_global_invalid(venue_cleaned):
            category = "Other / unclear venues"
        else:
            category = None

            for category_name, pattern in venue_categories:
                if re.search(pattern, venue_cleaned):
                    category = category_name
                    break

            if category is None and re.search(non_venue_pattern, venue_cleaned):
                category = "Likely non-venue row / keyword string"

            if category is None:
                category = "Other / unclear venues"

        counts[category] += 1

        assigned_rows.append({
            "row_index": original_index,
            "original_venue_cell": venue_raw,
            "cleaned_venue": venue_cleaned,
            "category": category,
        })

    summary_df = count_percent_rows(
        counts,
        total_valid_papers,
        category_col="Venue Category"
    ).sort_values("Count", ascending=False)

    assigned_df = pd.DataFrame(assigned_rows)

    manual_check_categories = [
        "Other / unclear venues",
        "Likely non-venue row / keyword string",
    ]

    manual_check_df = assigned_df[
        assigned_df["category"].isin(manual_check_categories)
    ].copy()

    manual_check_df["manual_category"] = ""
    manual_check_df["manual_notes"] = ""

    print("\nTotal valid papers:", total_valid_papers)
    print(summary_df.to_string(index=False))
    print("Total assigned:", int(summary_df["Count"].sum()))

    print("\n========= MANUAL CHECK ROWS: UNCLEAR / POSSIBLE NON-VENUE =========")
    print(f"Rows needing manual venue review: {len(manual_check_df)} / {total_valid_papers}")

    if len(manual_check_df) > 0:
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.width", 300,
        ):
            print(
                manual_check_df[
                    [
                        "row_index",
                        "original_venue_cell",
                        "cleaned_venue",
                        "category",
                        "manual_category",
                        "manual_notes",
                    ]
                ].to_string(index=False)
            )
    else:
        print("No unclear or likely non-venue rows found.")

    save_df(summary_df, "venue_category_summary.csv")
    save_df(assigned_df, "venue_assigned_rows.csv")
    save_df(manual_check_df, "venue_manual_check_rows.csv")

    return summary_df, assigned_df, manual_check_df


venue_summary, assigned_venues, venue_manual_check = compute_venue_category_counts_exclusive(
    col=df_subset.iloc[:, COL_JOURNAL],
    valid_mask=valid_total,
)

# ============================================================
# 7. TASK TYPE / METHOD BY PUBLICATION YEAR GROUP
# ============================================================


def extract_publication_year_robust(value):
    """
    Robustly extract a 4-digit publication year from a cell.

    Handles:
    - integers: 2024
    - floats: 2024.0
    - strings: "2024", "2024 ", "Published 2024", "2024\n"
    - missing values / blank strings
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text.lower() in ["", "nan", "none", "null", "na", "n/a"]:
        return np.nan

    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)

    if match:
        return int(match.group(1))

    return np.nan


def make_row_text_preview(row, max_chars=800):
    """
    Returns a compact text preview of the full row so you can identify
    which paper has the unreadable year.
    """
    parts = []

    for col_name, value in row.items():
        if pd.isna(value):
            continue

        value_text = str(value).strip()

        if value_text == "":
            continue

        parts.append(f"{col_name}: {value_text}")

    row_text = " | ".join(parts)

    if len(row_text) > max_chars:
        row_text = row_text[:max_chars] + "..."

    return row_text


def debug_unreadable_publication_years(data_df, valid_mask, year_col, years=None, label="valid_papers"):
    print(f"\n============= DEBUG: Missing / Unreadable Publication Years ({label}) =============")

    valid_mask = ensure_series_mask(valid_mask, data_df.index)

    raw_year = data_df.iloc[:, year_col].copy()

    if years is None:
        years = raw_year.apply(extract_publication_year_robust)

    unreadable_mask = valid_mask & years.isna()

    debug_rows = []

    for row_index in data_df.index[unreadable_mask]:
        raw_value = raw_year.loc[row_index]
        cleaned_value = "" if pd.isna(raw_value) else str(raw_value).strip()

        debug_rows.append({
            "row_index": row_index,
            "raw_year_cell": raw_value,
            "raw_year_repr": repr(raw_value),
            "cleaned_year_repr": repr(cleaned_value),
            "parsed_year": years.loc[row_index],
            "row_text_preview": make_row_text_preview(data_df.loc[row_index]),
        })

    debug_df = pd.DataFrame(debug_rows)

    print(f"Rows counted as missing/unreadable year: {len(debug_df)}")

    if len(debug_df) > 0:
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.width", 300,
        ):
            print(debug_df.to_string(index=False))
    else:
        print("No missing or unreadable publication-year rows found.")

    save_df(debug_df, f"debug_unreadable_publication_years_{label}.csv")

    return debug_df


def compute_publication_year_group_summary(data_df, valid_mask, year_col, label="total_valid"):
    print(f"\n============= Publication Year Group Summary ({label}) =============")

    valid_mask = ensure_series_mask(valid_mask, data_df.index)

    raw_year = data_df.iloc[:, year_col].copy()
    years = raw_year.apply(extract_publication_year_robust)
    year_groups = years.apply(classify_year_group)

    total_valid_papers = int(valid_mask.sum())

    # Year-group counts
    year_group_summary = (
        pd.DataFrame({
            "Year Group": year_groups[valid_mask],
        })
        .value_counts("Year Group")
        .reindex(YEAR_GROUP_ORDER, fill_value=0)
        .rename_axis("Year Group")
        .reset_index(name="Count")
    )

    year_group_summary["Total Valid Papers"] = total_valid_papers
    year_group_summary["Percentage"] = year_group_summary["Count"].apply(
        lambda x: pct(x, total_valid_papers)
    )

    print("\n============= Papers per Year Group =============")
    print(year_group_summary.to_string(index=False))

    # Exact year counts
    exact_year_summary = (
        pd.DataFrame({
            "Publication Year": years[valid_mask],
        })
        .dropna(subset=["Publication Year"])
        .assign(**{"Publication Year": lambda x: x["Publication Year"].astype(int)})
        .value_counts("Publication Year")
        .sort_index()
        .rename_axis("Publication Year")
        .reset_index(name="Count")
    )

    exact_year_summary["Total Valid Papers"] = total_valid_papers
    exact_year_summary["Percentage"] = exact_year_summary["Count"].apply(
        lambda x: pct(x, total_valid_papers)
    )

    print("\n============= Papers per Exact Publication Year =============")
    print(exact_year_summary.to_string(index=False))

    unreadable_debug_df = debug_unreadable_publication_years(
        data_df=data_df,
        valid_mask=valid_mask,
        year_col=year_col,
        years=years,
        label=label,
    )

    save_df(year_group_summary, f"publication_year_group_summary_{label}.csv")
    save_df(exact_year_summary, f"publication_year_exact_summary_{label}.csv")

    return year_group_summary, exact_year_summary, unreadable_debug_df, years, year_groups


def compute_task_type_by_year(data_df, valid_mask, year_col, task_col):
    print("\n============= Task Type / Method by Publication Year Group =============")

    valid_mask = ensure_series_mask(valid_mask, data_df.index)

    publication_year_group_summary, publication_year_exact_summary, unreadable_year_debug_df, years, year_groups = (
        compute_publication_year_group_summary(
            data_df=data_df,
            valid_mask=valid_mask,
            year_col=year_col,
            label="total_valid",
        )
    )

    task_type_summary, task_type_match_table, task_type_unclear_rows = compute_task_type(
        col=data_df.iloc[:, task_col],
        valid_mask=valid_mask,
        output_dir=str(OUTPUT_DIR) if SAVE_OUTPUTS else None,
    )

    task_type_table = task_type_match_table.copy()
    task_type_table["year"] = years.reindex(task_type_table.index)
    task_type_table["year_group"] = year_groups.reindex(task_type_table.index)

    task_columns = TASK_TYPE_COLS + ["multiple_task_types", "not_given", "unclear"]

    summary_rows = []

    for year_group in YEAR_GROUP_ORDER:
        group_mask = task_type_table["year_group"] == year_group
        total_group_papers = int(group_mask.sum())

        if total_group_papers == 0:
            continue

        for task in task_columns:
            count = int(task_type_table.loc[group_mask, task].sum())

            summary_rows.append({
                "Year Group": year_group,
                "Task Type / Method": task,
                "Count": count,
                "Total Papers in Year Group": total_group_papers,
                "Percentage": pct(count, total_group_papers),
            })

    task_type_by_year_summary = pd.DataFrame(summary_rows)

    print("\n============= Task Type / Method by Year Group =============")
    print(task_type_by_year_summary.to_string(index=False))

    save_df(task_type_by_year_summary, "task_type_by_year_summary.csv")
    save_df(task_type_table.reset_index().rename(columns={"index": "row_index"}), "task_type_by_year_match_table.csv")
    save_df(task_type_summary, "task_type_overall_summary.csv")
    save_df(task_type_unclear_rows.reset_index().rename(columns={"index": "row_index"}), "task_type_unclear_rows.csv")
    save_df(publication_year_group_summary, "publication_year_group_summary.csv")
    save_df(publication_year_exact_summary, "publication_year_exact_summary.csv")
    save_df(unreadable_year_debug_df, "publication_year_unreadable_debug_total_valid.csv")

    return (
        task_type_by_year_summary,
        task_type_summary,
        task_type_table,
        publication_year_group_summary,
        publication_year_exact_summary,
        unreadable_year_debug_df,
    )


(
    task_type_by_year_summary,
    task_type_overall_summary,
    task_type_year_match_table,
    publication_year_group_summary,
    publication_year_exact_summary,
    publication_year_unreadable_debug,
) = compute_task_type_by_year(
    data_df=df_subset,
    valid_mask=valid_total,
    year_col=COL_PUBLICATION_YEAR,
    task_col=COL_TASK_PARTICIPANTS,
)


# ============================================================
# 8. PUBLICATION YEAR GROUP BY ASD AGE GROUP
# ============================================================


def compute_year_by_asd_age_group(data_df, valid_asd_mask, year_col, asd_range_col, asd_mean_col, asd_sd_col):
    print("\n============= Publication Year Group by ASD Age Group =============")

    valid_asd_mask = ensure_series_mask(valid_asd_mask, data_df.index)

    asd_year_group_summary, asd_exact_year_summary, asd_unreadable_year_debug_df, years, year_groups = (
        compute_publication_year_group_summary(
            data_df=data_df,
            valid_mask=valid_asd_mask,
            year_col=year_col,
            label="asd_valid",
        )
    )

    asd_age_summary, asd_age_parser_usage, asd_age_match_table, asd_age_manual_review = compute_asd_age_ranges(
        df_subset=data_df,
        range_col_index=asd_range_col,
        mean_col_index=asd_mean_col,
        sd_col_index=asd_sd_col,
        valid_asd_mask=valid_asd_mask,
        output_dir=str(OUTPUT_DIR) if SAVE_OUTPUTS else None,
    )

    age_year_table = asd_age_match_table.copy()
    age_year_table["year"] = years.reindex(age_year_table.index)
    age_year_table["year_group"] = year_groups.reindex(age_year_table.index)

    age_group_cols = AGE_CATEGORY_COLS + ["not_given", "multiple_age_groups"]

    summary_rows = []

    for year_group in YEAR_GROUP_ORDER:
        group_mask = age_year_table["year_group"] == year_group
        total_papers_in_year_group = int(group_mask.sum())

        if total_papers_in_year_group == 0:
            continue

        for age_group in age_group_cols:
            count = int(age_year_table.loc[group_mask, age_group].sum())

            summary_rows.append({
                "Year Group": year_group,
                "ASD Age Group": age_group,
                "Count": count,
                "Total ASD Papers in Year Group": total_papers_in_year_group,
                "Percentage": pct(count, total_papers_in_year_group),
            })

    year_by_asd_age_summary = pd.DataFrame(summary_rows)

    coverage_summary = asd_year_group_summary.rename(columns={
        "Count": "ASD-valid papers",
        "Total Valid Papers": "Total ASD-valid papers",
        "Percentage": "Percentage of ASD-valid papers",
    })

    print("\n============= ASD-valid Coverage by Year Group =============")
    print(coverage_summary.to_string(index=False))

    print("\n============= Year Group by ASD Age Group Summary =============")
    print(year_by_asd_age_summary.to_string(index=False))

    save_df(year_by_asd_age_summary, "year_by_asd_age_summary.csv")
    save_df(age_year_table.reset_index().rename(columns={"index": "row_index"}), "asd_age_year_match_table.csv")
    save_df(coverage_summary, "asd_age_year_coverage_summary.csv")
    save_df(asd_year_group_summary, "publication_year_group_summary_asd_valid.csv")
    save_df(asd_exact_year_summary, "publication_year_exact_summary_asd_valid.csv")
    save_df(asd_unreadable_year_debug_df, "publication_year_unreadable_debug_asd_valid.csv")
    save_df(asd_age_summary, "asd_age_overall_summary.csv")
    save_df(asd_age_parser_usage, "asd_age_parser_usage.csv")
    save_df(asd_age_manual_review.reset_index().rename(columns={"index": "row_index"}), "asd_age_manual_review.csv")

    return (
        year_by_asd_age_summary,
        age_year_table,
        coverage_summary,
        asd_age_summary,
        asd_year_group_summary,
        asd_exact_year_summary,
        asd_unreadable_year_debug_df,
    )


(
    year_by_asd_age_summary,
    asd_age_year_match_table,
    age_year_coverage_summary,
    asd_age_overall_summary,
    asd_publication_year_group_summary,
    asd_publication_year_exact_summary,
    asd_publication_year_unreadable_debug,
) = compute_year_by_asd_age_group(
    data_df=df_subset,
    valid_asd_mask=valid_ASD,
    year_col=COL_PUBLICATION_YEAR,
    asd_range_col=COL_ASD_AGE_RANGE,
    asd_mean_col=COL_ASD_MEAN_AGE,
    asd_sd_col=COL_ASD_SD_AGE,
)


# ============================================================
# 9. FINAL CHECK
# ============================================================

print("\n============= FINAL CHECK =============")
print("Total valid papers:", int(valid_total.sum()))
print("Valid ASD papers:", int(valid_ASD.sum()))
print("Output directory:", OUTPUT_DIR.resolve())

print("\nTotal valid papers by year group:")
print(publication_year_group_summary.to_string(index=False))

print("\nASD-valid papers by year group:")
print(age_year_coverage_summary.to_string(index=False))

print("\nUnreadable publication-year rows, total-valid:")
print(len(publication_year_unreadable_debug))

if len(publication_year_unreadable_debug) > 0:
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.width", 300,
    ):
        print(publication_year_unreadable_debug.to_string(index=False))

print("\nUnreadable publication-year rows, ASD-valid:")
print(len(asd_publication_year_unreadable_debug))

if len(asd_publication_year_unreadable_debug) > 0:
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.width", 300,
    ):
        print(asd_publication_year_unreadable_debug.to_string(index=False))

print("\nSaved files include:")
print("- column_map.csv")
print("- keyword_subgroup_summary.csv")
print("- venue_category_summary.csv")
print("- publication_year_volume_summary.csv")
print("- publication_year_group_summary.csv")
print("- publication_year_exact_summary.csv")
print("- publication_year_unreadable_debug_total_valid.csv")
print("- task_type_by_year_summary.csv")
print("- year_by_asd_age_summary.csv")
print("- asd_age_year_coverage_summary.csv")
print("- publication_year_unreadable_debug_asd_valid.csv")
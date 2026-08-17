
"""

The analysis is project-specific. It reads the final annotated workbook, applies
the same category definitions used by rq2_.py and rq4_.py, and produces
modality-specific summaries for gaze, speech/language/audio, and motor data.

Shared categories are imported from ``codebook.py``. Remaining project-specific
subcategories are read from the guarded RQ scripts without executing analysis.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import pandas as pd

from analysis_common import resolve_root
from codebook import (
    ALGORITHM_FAMILY_PATTERNS,
    EVALUATION_METRIC_PATTERNS,
    HYBRID_MODEL_PATTERN,
    LEARNING_TYPE_PATTERNS,
    NOT_GIVEN_TASK_PATTERN,
    TASK_TYPE_PATTERNS,
)
from helper_functions_ import extract_accuracy_percent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = resolve_root("ASD_REVIEW_DATA_ROOT", "data") / "final_annotation_sheet_.xlsx"
DEFAULT_OUTPUT_DIR = resolve_root("ASD_REVIEW_OUTPUT_ROOT", "output") / "mapping_across_research_questions"
DEFAULT_REFERENCE_OUTPUT_ROOT = resolve_root("ASD_REVIEW_OUTPUT_ROOT", "output")
DEFAULT_SHEET = "final_data"
OUTPUT_FILENAMES = [
    "mapping_underlying_counts.csv",
    "mapping_performance_statistics.csv",
    "mapping_study_membership.csv",
    "mapping_examples.csv",
    "mapping_summary_table.csv",
    "mapping_validation.csv",
    "mapping_report.md",
]

N_RELEVANT_COLUMNS = 77
N_DATA_ROWS = 172

COL_TITLE = 0
COL_GAZE = 40
COL_SPEECH = 41
COL_MOTOR = 42
COL_ALGORITHMS = 46
COL_LEARNING_TYPE = 48
COL_EVALUATION_METRICS = 49
COL_BEST_MODEL = 50
COL_BEST_PERFORMANCE = 51
COL_BIAS_MITIGATION = 54
COL_PARTICIPANT_TASK = 67
COL_STUDY_LIMITATIONS = 69
COL_MAIN_FINDINGS = 71
COL_FUTURE_DIRECTIONS = 74
LIMITATION_SOURCE_COLUMNS = [10, 11, COL_STUDY_LIMITATIONS]

MODALITIES = {
    "gaze": COL_GAZE,
    "speech_language_audio": COL_SPEECH,
    "motor": COL_MOTOR,
}

DISPLAY_MODALITY = {
    "gaze": "Gaze / eye tracking",
    "speech_language_audio": "Speech / language / audio",
    "motor": "Motor / movement / pose / kinematic",
}

INVALID_VALUES = {
    "",
    "-",
    "--",
    "nan",
    "na",
    "n/a",
    "n.a",
    "n.a.",
    "n.d",
    "n.d.",
    "nd",
    "n/d",
    "nr",
    "n.r",
    "n.r.",
    "n/r",
    "not given",
    "not reported",
    "not specified",
    "not stated",
    "not mentioned",
    "not available",
    "not provided",
    "not applicable",
    "unknown",
    "unkown",
    "unclear",
    "no information",
    "not clear",
    "not explicitly stated",
    "none",
    "none specified",
    "not givven",
}

REQUIRED_FINDING_CATEGORIES = [
    "specific_predictive_or_discriminative_features_identified",
    "multimodal_or_feature_fusion_improved_performance",
    "model_algorithm_comparison_or_optimization",
    "high_model_performance_or_feasibility",
    "task_or_protocol_specific_effect",
]

# These examples were manually verified against the current final_data sheet.
# Title fragments are used instead of row numbers so row insertions do not
# silently point to a different study.
CURATED_TASK_EXAMPLES = {
    "gaze": [
        ("Application of Machine Learning Techniques to Detect", "passive social/non-social video viewing"),
        ("Detecting High-Functioning Autism in Adults", "website browsing and information-search tasks"),
        ("Toward Continuous Social Phenotyping", "facial-emotion identification with gaze recording"),
    ],
    "speech_language_audio": [
        ("Automatic Autism Spectrum Disorder Detection Using Everyday Vocalizations", "picture description, reading, story reading, and everyday vocalization"),
        ("Automatic detection of autism spectrum disorder in children using acoustic", "unstructured prompted conversation"),
        ("Detecting autism from picture book narratives", "ADOS-2 story-from-a-book task"),
    ],
    "motor": [
        ("Applying Machine Learning to Kinematic and Eye Movement Features", "observed and imitated hand-movement sequences"),
        ("Effects of Intra-Subject Variation in Gait Analysis", "repeated walking tasks"),
        ("Toward the Autism Motor Signature", "smart-tablet gameplay with touch and gesture capture"),
        ("Use of Machine Learning to Identify Children with Autism and Their Motor", "reach, grasp, and drop task"),
    ],
}

OTHER_METRIC_TERMS_FALLBACK = [
    "sensitivity",
    "specificity",
    "recall",
    "precision",
    "auc",
    "auroc",
    "roc",
    "auc-roc",
    "auc roc",
    "au-roc",
    "confusion matrix",
    "f-1 score",
    "f-1",
    "f 1",
    "f1",
    "matthews correlation coefficient",
    "mcc",
    "error-rate",
    "error rate",
    "positive predictive value",
    "ppv",
    "negative predictive value",
    "npv",
    "uar",
    "tpr",
    "tnr",
    "kappa",
    "diagnostic validity",
    "f-measure",
    "f measure",
    "g-mean",
    "g mean",
    "loss",
    "mae",
    "mse",
    "rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map behavioral modalities across the established RQ2/RQ4 coding framework."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Annotated workbook (default: data/final_annotation_sheet_.xlsx).",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help="Annotated worksheet name (default: final_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for new mapping outputs.",
    )
    parser.add_argument(
        "--reference-output-root",
        type=Path,
        default=DEFAULT_REFERENCE_OUTPUT_ROOT,
        help="Existing RQ output root used only for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's named output files if they already exist.",
    )
    return parser.parse_args()


def resolve_from_project(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def flatten_columns(columns: Iterable[tuple[object, object]]) -> list[str]:
    flattened = []
    for column in columns:
        parts = [
            str(value).strip()
            for value in column
            if pd.notna(value)
            and not str(value).startswith("Unnamed")
            and str(value).strip()
        ]
        flattened.append(" | ".join(parts) if parts else "Unnamed")
    return flattened


def load_annotation_data(path: Path, sheet: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotated workbook not found: {path}")

    raw = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        usecols="A:BY",
        nrows=N_DATA_ROWS + 2,
    )
    if raw.shape[1] != N_RELEVANT_COLUMNS:
        raise ValueError(
            f"Expected A:BY ({N_RELEVANT_COLUMNS} columns), found {raw.shape[1]}."
        )
    if raw.shape[0] < N_DATA_ROWS + 2:
        raise ValueError(
            f"Expected two header rows plus {N_DATA_ROWS} data rows, found {raw.shape[0]} rows."
        )

    header_rows = raw.iloc[0:2].ffill(axis=1)
    columns = []
    seen: dict[tuple[object, object], int] = {}
    for top, bottom in zip(header_rows.iloc[0].tolist(), header_rows.iloc[1].tolist()):
        key = (top, bottom)
        duplicate_number = seen.get(key, 0)
        seen[key] = duplicate_number + 1
        if duplicate_number:
            bottom = f"{bottom}.{duplicate_number}"
        columns.append((top, bottom))

    frame = raw.iloc[2 : N_DATA_ROWS + 2, :N_RELEVANT_COLUMNS].copy()
    frame.columns = flatten_columns(columns)
    frame = frame.reset_index(drop=True)

    title_clean = frame.iloc[:, COL_TITLE].apply(normalize_text)
    valid_mask = ~title_clean.isin(INVALID_VALUES)
    if int(valid_mask.sum()) != N_DATA_ROWS:
        raise ValueError(
            "The established title-based denominator changed: "
            f"expected {N_DATA_ROWS}, found {int(valid_mask.sum())}."
        )
    return frame, valid_mask


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\xa0", " ")
    )
    return re.sub(r"\s+", " ", text)


def rq4_clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def is_yes(value: object) -> bool:
    text = normalize_text(value)
    if text in INVALID_VALUES:
        return False
    return bool(re.search(r"^(yes|y|true|1|used|present|included|reported)\b", text))


def is_no(value: object) -> bool:
    text = normalize_text(value)
    if text in {"no", "n", "false", "0", "not used", "absent", "not included"}:
        return True
    return bool(re.search(r"^(no|n|false|0)\b", text))


def pct(count: int | float, denominator: int | float) -> float:
    return round((float(count) / float(denominator)) * 100, 2) if denominator else 0.0


def display_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def find_function(tree: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise KeyError(f"Function {function_name!r} not found.")


def extract_literal_assignment(
    tree: ast.Module,
    variable_name: str,
    function_name: str | None = None,
) -> object:
    scope: ast.AST = find_function(tree, function_name) if function_name else tree
    for node in ast.walk(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in targets):
            return ast.literal_eval(node.value)
    location = f" in {function_name}" if function_name else ""
    raise KeyError(f"Assignment {variable_name!r}{location} not found.")


def load_established_rules(script_dir: Path) -> tuple[dict[str, object], dict[str, str]]:
    rq2_path = script_dir / "rq2_.py"
    rq4_path = script_dir / "rq4_.py"
    for path in [rq2_path, rq4_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required source script not found: {path}")

    rq2_tree = ast.parse(rq2_path.read_text(encoding="utf-8"), filename=str(rq2_path))
    rq4_tree = ast.parse(rq4_path.read_text(encoding="utf-8"), filename=str(rq4_path))

    rules = {
        "task_broad": {**TASK_TYPE_PATTERNS, "not_given": NOT_GIVEN_TASK_PATTERN},
        "task_subcategory": extract_literal_assignment(
            rq2_tree, "subcategory_patterns", "compute_task_type"
        ),
        "limitations": extract_literal_assignment(
            rq2_tree, "category_patterns", "rq_limitation_categories"
        ),
        "main_findings": extract_literal_assignment(
            rq2_tree, "finding_subcategory_patterns", "goal_finding_hybrid_summary"
        ),
        "future_directions": extract_literal_assignment(
            rq2_tree, "category_patterns", "compute_future_goals_categories"
        ),
        "paradigm": LEARNING_TYPE_PATTERNS,
        "algorithm_broad": ALGORITHM_FAMILY_PATTERNS,
        "algorithm_classical": extract_literal_assignment(
            rq4_tree, "patterns", "algorithms_further_basic_ml"
        ),
        "algorithm_neural": extract_literal_assignment(
            rq4_tree, "patterns", "algorithms_further_neural_networks"
        ),
        "evaluation_metrics": EVALUATION_METRIC_PATTERNS,
        "bias_mitigation": extract_literal_assignment(
            rq4_tree, "patterns", "compute_bias_mitigation"
        ),
        "bias_not_reported": extract_literal_assignment(
            rq4_tree, "not_reported_pattern", "compute_bias_mitigation"
        ),
        "bias_reported_no": extract_literal_assignment(
            rq4_tree, "reported_no_pattern", "compute_bias_mitigation"
        ),
        "hybrid_algorithm": HYBRID_MODEL_PATTERN,
    }

    rules["other_metric_terms"] = OTHER_METRIC_TERMS_FALLBACK

    hashes = {
        display_path(rq2_path): sha256_file(rq2_path),
        display_path(rq4_path): sha256_file(rq4_path),
        display_path(script_dir / "codebook.py"): sha256_file(script_dir / "codebook.py"),
    }
    return rules, hashes


def build_modality_table(
    frame: pd.DataFrame, valid_mask: pd.Series
) -> pd.DataFrame:
    table = pd.DataFrame(index=frame.index[valid_mask])
    for modality, column_index in MODALITIES.items():
        cleaned = frame.iloc[:, column_index][valid_mask].apply(normalize_text)
        yes_mask = cleaned.apply(is_yes)
        no_mask = cleaned.apply(is_no)
        table[f"{modality}_text"] = cleaned
        table[f"{modality}_yes"] = yes_mask
        table[f"{modality}_no"] = no_mask
        table[f"{modality}_unclear"] = ~(yes_mask | no_mask)
    return table


def match_patterns(
    text: pd.Series,
    patterns: Mapping[str, str],
    prefix: str,
) -> pd.DataFrame:
    table = pd.DataFrame(index=text.index)
    for category, pattern in patterns.items():
        table[f"{prefix}__{category}"] = text.str.contains(
            pattern, regex=True, na=False
        )
    return table


def build_task_table(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    broad_patterns: Mapping[str, str],
    subcategory_patterns: Mapping[str, str],
) -> pd.DataFrame:
    text = frame.iloc[:, COL_PARTICIPANT_TASK][valid_mask].apply(normalize_text)
    table = pd.DataFrame({"task_text": text})
    not_given = text.str.contains(broad_patterns["not_given"], regex=True, na=False)
    table["task__not_given"] = not_given

    real_categories = [
        category for category in broad_patterns if category != "not_given"
    ]
    for category in real_categories:
        table[f"task__{category}"] = (
            ~not_given
            & text.str.contains(broad_patterns[category], regex=True, na=False)
        )

    category_columns = [f"task__{category}" for category in real_categories]
    table["task__multiple_task_types"] = table[category_columns].sum(axis=1) >= 2
    table["task__unclear"] = ~not_given & (table[category_columns].sum(axis=1) == 0)

    for category, pattern in subcategory_patterns.items():
        table[f"task_subcategory__{category}"] = (
            ~not_given & text.str.contains(pattern, regex=True, na=False)
        )
    return table


def build_limitation_table(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    patterns: Mapping[str, str],
) -> pd.DataFrame:
    text = (
        frame.loc[valid_mask, frame.columns[LIMITATION_SOURCE_COLUMNS]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.strip()
    )
    table = pd.DataFrame({"limitation_text": text})
    for category, pattern in patterns.items():
        table[f"limitation__{category}"] = text.str.contains(
            pattern, regex=True, na=False
        )

    category_columns = [f"limitation__{category}" for category in patterns]
    any_match = table[category_columns].any(axis=1)
    placeholder = text.str.fullmatch(
        r"\s*|[-\s]+|(?:no|yes|n/d|nd|n/a|na|nan|none|no limitation|"
        r"no limitations|not reported|not applicable|n\.a\.)(?:\s+(?:no|yes|"
        r"n/d|nd|n/a|na|nan|none|no limitation|no limitations|not reported|"
        r"not applicable|n\.a\.|-))*\s*",
        na=False,
    )
    table["limitation__need_manual_revision"] = ~any_match & text.ne("") & ~placeholder
    table["limitation__empty_or_not_applicable"] = text.eq("") | placeholder
    return table


def build_finding_table(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    patterns: Mapping[str, str],
) -> pd.DataFrame:
    text = (
        frame.iloc[:, COL_MAIN_FINDINGS][valid_mask]
        .apply(lambda value: "" if pd.isna(value) else str(value).replace("\n", " ").strip())
    )
    table = pd.DataFrame({"main_finding_text": text})
    for category, pattern in patterns.items():
        table[f"main_finding__{category}"] = text.apply(
            lambda value, regex=pattern: bool(re.search(regex, value, flags=re.IGNORECASE))
        )
    placeholder = text.apply(
        lambda value: bool(
            re.fullmatch(
                r"^\s*$|^-+$|^n/d$|^nd$|^n/a$|^na$|^nan$|"
                r"^not reported$|^not given$|^not specified$|^unclear$|^unknown$",
                value,
                flags=re.IGNORECASE,
            )
        )
    )
    category_columns = [f"main_finding__{category}" for category in patterns]
    table["main_finding__unclear"] = placeholder | ~table[category_columns].any(axis=1)
    return table


def build_future_table(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    patterns: Mapping[str, str],
) -> pd.DataFrame:
    text = frame.iloc[:, COL_FUTURE_DIRECTIONS][valid_mask].apply(normalize_text)
    table = pd.DataFrame({"future_direction_text": text})
    for category, pattern in patterns.items():
        table[f"future_direction__{category}"] = text.str.contains(
            pattern, regex=True, na=False
        )
    category_columns = [f"future_direction__{category}" for category in patterns]
    table["future_direction__no_category_matched"] = ~table[category_columns].any(axis=1)
    return table


def build_bias_mitigation_table(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    patterns: Mapping[str, str],
    not_reported_pattern: str,
    reported_no_pattern: str,
) -> pd.DataFrame:
    text = rq4_clean_text(frame.iloc[:, COL_BIAS_MITIGATION][valid_mask])
    table = pd.DataFrame({"bias_mitigation_text": text})
    not_reported = text.str.contains(not_reported_pattern, regex=True, na=False)
    reported_no = (
        ~not_reported
        & text.str.contains(reported_no_pattern, regex=True, na=False)
    )
    table["bias_mitigation__not_reported_or_missing"] = not_reported
    table["bias_mitigation__reported_no_bias_mitigation"] = reported_no

    technique_columns = []
    for category, pattern in patterns.items():
        column = f"bias_mitigation__{category}"
        table[column] = (
            ~not_reported
            & ~reported_no
            & text.str.contains(pattern, regex=True, na=False)
        )
        technique_columns.append(column)
    any_technique = table[technique_columns].any(axis=1)
    table["bias_mitigation__any_technique_reported"] = any_technique
    table["bias_mitigation__valid_text_but_uncategorized"] = (
        ~not_reported & ~reported_no & ~any_technique
    )
    return table


def extract_accuracy_from_row(
    row: pd.Series, other_metric_terms: Sequence[str]
) -> float:
    del other_metric_terms  # retained for CLI/output compatibility
    return extract_accuracy_percent(
        row.iloc[COL_BEST_PERFORMANCE], row.iloc[COL_EVALUATION_METRICS]
    )


def add_count_rows(
    rows: list[dict[str, object]],
    modality: str,
    section: str,
    table: pd.DataFrame,
    category_columns: Sequence[str],
    modality_mask: pd.Series,
    source_script: str,
    source_section: str,
    source_column: str,
    denominator_scope: str = "all studies assigned to this modality",
    nonexclusive: bool = True,
    category_prefix: str | None = None,
) -> None:
    denominator = int(modality_mask.sum())
    for column in category_columns:
        count = int((table[column].reindex(modality_mask.index).fillna(False) & modality_mask).sum())
        category = (
            column[len(category_prefix) :]
            if category_prefix and column.startswith(category_prefix)
            else column
        )
        rows.append(
            {
                "modality": modality,
                "modality_label": DISPLAY_MODALITY[modality],
                "section": section,
                "category": category,
                "count": count,
                "denominator": denominator,
                "percentage": pct(count, denominator),
                "denominator_scope": denominator_scope,
                "source_script": source_script,
                "source_code_section": source_section,
                "source_column": source_column,
                "nonexclusive": nonexclusive,
            }
        )


def select_row_example(
    frame: pd.DataFrame,
    eligible_mask: pd.Series,
    source_series: pd.Series,
    excluded_indices: set[int] | None = None,
) -> int | None:
    excluded_indices = excluded_indices or set()
    candidates = [
        index
        for index in source_series.index
        if bool(eligible_mask.get(index, False))
        and index not in excluded_indices
        and normalize_text(source_series.loc[index]) not in INVALID_VALUES
    ]
    return candidates[0] if candidates else None


def build_curated_task_examples(
    frame: pd.DataFrame,
    modality_table: pd.DataFrame,
) -> list[dict[str, object]]:
    examples = []
    normalized_titles = frame.iloc[:, COL_TITLE].apply(normalize_text)
    for modality, specifications in CURATED_TASK_EXAMPLES.items():
        modality_mask = modality_table[f"{modality}_yes"]
        for title_fragment, operationalization in specifications:
            fragment = normalize_text(title_fragment)
            candidates = normalized_titles[
                normalized_titles.str.contains(re.escape(fragment), regex=True, na=False)
            ].index.tolist()
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one study for curated title fragment {title_fragment!r}; "
                    f"found {len(candidates)}."
                )
            index = candidates[0]
            if not bool(modality_mask.get(index, False)):
                raise ValueError(
                    f"Curated example {title_fragment!r} is not assigned to {modality}."
                )
            examples.append(
                {
                    "modality": modality,
                    "modality_label": DISPLAY_MODALITY[modality],
                    "section": "task_operationalization",
                    "category": operationalization,
                    "row_index": index,
                    "excel_row": index + 3,
                    "study_title": frame.iloc[index, COL_TITLE],
                    "source_text": frame.iloc[index, COL_PARTICIPANT_TASK],
                }
            )
    return examples


def build_dynamic_examples(
    frame: pd.DataFrame,
    modality_table: pd.DataFrame,
    count_table: pd.DataFrame,
    section_tables: Mapping[str, tuple[pd.DataFrame, str, str]],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for modality in MODALITIES:
        modality_mask = modality_table[f"{modality}_yes"]
        for section, maximum_categories in [
            ("algorithm_family", 3),
            ("main_findings", 2),
            ("future_directions", 1),
        ]:
            section_counts = count_table[
                (count_table["modality"] == modality)
                & (count_table["section"] == section)
                & (count_table["count"] > 0)
            ].sort_values(["count", "category"], ascending=[False, True])
            if section == "future_directions":
                section_counts = section_counts[
                    ~section_counts["category"].isin(["not_given", "no_category_matched"])
                ]
            selected_categories = section_counts.head(maximum_categories)["category"].tolist()
            match_table, prefix, source_text_column = section_tables[section]
            used_indices: set[int] = set()
            for category in selected_categories:
                flag_column = f"{prefix}{category}"
                if flag_column not in match_table:
                    continue
                eligible = (
                    match_table[flag_column].reindex(modality_mask.index).fillna(False)
                    & modality_mask
                )
                index = select_row_example(
                    frame,
                    eligible,
                    match_table[source_text_column],
                    excluded_indices=used_indices,
                )
                if index is None:
                    continue
                used_indices.add(index)
                examples.append(
                    {
                        "modality": modality,
                        "modality_label": DISPLAY_MODALITY[modality],
                        "section": section,
                        "category": category,
                        "row_index": index,
                        "excel_row": index + 3,
                        "study_title": frame.iloc[index, COL_TITLE],
                        "source_text": match_table.loc[index, source_text_column],
                    }
                )
    return examples


def category_counts(table: pd.DataFrame, prefix: str) -> dict[str, int]:
    return {
        column[len(prefix) :]: int(table[column].sum())
        for column in table.columns
        if column.startswith(prefix) and table[column].dtype == bool
    }


def compare_reference_summary(
    validation_rows: list[dict[str, object]],
    reference_path: Path,
    generated_counts: Mapping[str, int],
    generated_denominator: int,
    group: str,
    category_column: str,
    count_column: str = "Count",
    denominator_column: str | None = "Total Valid Papers",
    percentage_column: str | None = "Percentage",
) -> None:
    if not reference_path.is_file():
        validation_rows.append(
            {
                "validation_group": group,
                "key": "reference_file",
                "reference_file": display_path(reference_path),
                "reference_value": "",
                "generated_value": "",
                "absolute_difference": "",
                "status": "REFERENCE_NOT_AVAILABLE",
                "explanation": "Existing RQ output was not available; no comparison was possible.",
            }
        )
        return

    reference = pd.read_csv(reference_path)
    for category, generated_count in generated_counts.items():
        matched = reference[reference[category_column].astype(str) == str(category)]
        if matched.empty:
            validation_rows.append(
                {
                    "validation_group": group,
                    "key": category,
                    "reference_file": display_path(reference_path),
                    "reference_value": "",
                    "generated_value": generated_count,
                    "absolute_difference": "",
                    "status": "CATEGORY_NOT_IN_REFERENCE",
                    "explanation": "Generated support/derived category has no direct reference row.",
                }
            )
            continue

        reference_count = float(matched.iloc[0][count_column])
        count_difference = abs(reference_count - generated_count)
        validation_rows.append(
            {
                "validation_group": group,
                "key": f"{category}::count",
                "reference_file": display_path(reference_path),
                "reference_value": reference_count,
                "generated_value": generated_count,
                "absolute_difference": count_difference,
                "status": "MATCH" if count_difference == 0 else "DISCREPANCY",
                "explanation": "Count comparison.",
            }
        )

        if denominator_column and denominator_column in reference:
            reference_denominator = float(matched.iloc[0][denominator_column])
            denominator_difference = abs(reference_denominator - generated_denominator)
            validation_rows.append(
                {
                    "validation_group": group,
                    "key": f"{category}::denominator",
                    "reference_file": display_path(reference_path),
                    "reference_value": reference_denominator,
                    "generated_value": generated_denominator,
                    "absolute_difference": denominator_difference,
                    "status": "MATCH" if denominator_difference == 0 else "DISCREPANCY",
                    "explanation": "Denominator comparison.",
                }
            )

        if percentage_column and percentage_column in reference:
            reference_percentage = float(matched.iloc[0][percentage_column])
            generated_percentage = pct(generated_count, generated_denominator)
            percentage_difference = abs(reference_percentage - generated_percentage)
            validation_rows.append(
                {
                    "validation_group": group,
                    "key": f"{category}::percentage",
                    "reference_file": display_path(reference_path),
                    "reference_value": reference_percentage,
                    "generated_value": generated_percentage,
                    "absolute_difference": percentage_difference,
                    "status": "MATCH" if percentage_difference <= 0.01 else "DISCREPANCY",
                    "explanation": "Percentage comparison, allowing only display-rounding tolerance.",
                }
            )


def build_validation_table(
    reference_root: Path,
    total_valid: int,
    modality_table: pd.DataFrame,
    task_table: pd.DataFrame,
    paradigm_table: pd.DataFrame,
    algorithm_broad_table: pd.DataFrame,
    algorithm_classical_table: pd.DataFrame,
    algorithm_neural_table: pd.DataFrame,
    metric_table: pd.DataFrame,
    limitation_table: pd.DataFrame,
    finding_table: pd.DataFrame,
    future_table: pd.DataFrame,
    bias_table: pd.DataFrame,
    performance_table: pd.DataFrame,
    count_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    modality_reference = reference_root / "rq3_results" / "RQ3_gaze_motor_speech_summary.csv"
    if modality_reference.is_file():
        reference = pd.read_csv(modality_reference)
        reference = reference[reference["Category"] == "Yes"]
        reference_map = dict(zip(reference["Modality"], reference["Count"]))
        for modality, reference_name in [
            ("gaze", "gaze"),
            ("motor", "motor"),
            ("speech_language_audio", "speech"),
        ]:
            generated = int(modality_table[f"{modality}_yes"].sum())
            expected = int(reference_map[reference_name])
            rows.append(
                {
                    "validation_group": "modality_assignment",
                    "key": f"{modality}::yes_count",
                    "reference_file": display_path(modality_reference),
                    "reference_value": expected,
                    "generated_value": generated,
                    "absolute_difference": abs(expected - generated),
                    "status": "MATCH" if expected == generated else "DISCREPANCY",
                    "explanation": "Exact RQ3/RQ4 modality yes-count comparison.",
                }
            )
    else:
        rows.append(
            {
                "validation_group": "modality_assignment",
                "key": "reference_file",
                "reference_file": display_path(modality_reference),
                "reference_value": "",
                "generated_value": "",
                "absolute_difference": "",
                "status": "REFERENCE_NOT_AVAILABLE",
                "explanation": "Existing modality summary was not available.",
            }
        )

    comparisons = [
        (
            reference_root / "rq2_results" / "RQ2_task_type_summary.csv",
            category_counts(task_table, "task__"),
            "task_broad",
            "Task Type Category",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
        (
            reference_root / "rq2_results" / "RQ2_task_type_subcategory_summary.csv",
            category_counts(task_table, "task_subcategory__"),
            "task_subcategory",
            "Task subcategory",
            "n",
            None,
            "% of studies",
        ),
        (
            reference_root / "rq4_results" / "RQ4_learning_paradigm_summary.csv",
            category_counts(paradigm_table, "paradigm__"),
            "paradigm",
            "Learning Paradigm",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
        (
            reference_root / "rq4_results" / "RQ4_algorithms_broad_summary.csv",
            {
                category: count
                for category, count in category_counts(
                    algorithm_broad_table, "algorithm_family__"
                ).items()
                if category != "hybrid_or_multi_model_architectures"
            },
            "algorithm_family",
            "Algorithm Broad Category",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
        (
            reference_root
            / "rq4_results"
            / "RQ4_algorithms_classical_subcategories_summary.csv",
            category_counts(algorithm_classical_table, "algorithm_classical__"),
            "algorithm_classical_subcategory",
            "Classical ML Category",
            "Count",
            "Total Valid Papers",
            "Percentage of Total Papers",
        ),
        (
            reference_root
            / "rq4_results"
            / "RQ4_algorithms_neural_subcategories_summary.csv",
            category_counts(algorithm_neural_table, "algorithm_neural__"),
            "algorithm_neural_subcategory",
            "Neural Network Category",
            "Count",
            "Total Valid Papers",
            "Percentage of Total Papers",
        ),
        (
            reference_root / "rq4_results" / "RQ4_evaluation_metrics_summary.csv",
            category_counts(metric_table, "metric__"),
            "evaluation_metrics",
            "Evaluation Metric",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
        (
            reference_root / "rq2_results" / "RQ2_limitation_category_summary.csv",
            category_counts(limitation_table, "limitation__"),
            "limitations",
            "Limitation Category",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
        (
            reference_root
            / "rq2_results"
            / "RQ2_main_finding_subcategory_summary.csv",
            {
                category: count
                for category, count in category_counts(
                    finding_table, "main_finding__"
                ).items()
                if category != "unclear"
            },
            "main_findings",
            "Finding Subcategory",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
        (
            reference_root / "rq2_results" / "RQ2_future_goals_summary.csv",
            category_counts(future_table, "future_direction__"),
            "future_directions",
            "Future Goal / Research Aim Category",
            "Count",
            "Total Valid Papers",
            "Percentage",
        ),
    ]
    for (
        path,
        counts,
        group,
        category_column,
        count_column,
        denominator_column,
        percentage_column,
    ) in comparisons:
        compare_reference_summary(
            rows,
            path,
            counts,
            total_valid,
            group,
            category_column,
            count_column,
            denominator_column,
            percentage_column,
        )

    compare_reference_summary(
        rows,
        reference_root / "rq4_results" / "RQ4_hybrid_models_summary.csv",
        {
            "hybrid_or_multi_model_architectures": category_counts(
                algorithm_broad_table, "algorithm_family__"
            )["hybrid_or_multi_model_architectures"]
        },
        total_valid,
        "algorithm_hybrid",
        "Algorithm Category",
    )
    compare_reference_summary(
        rows,
        reference_root
        / "rq2_results"
        / "RQ2_all_main_finding_derived_counts.csv",
        {
            "finding_subcategory_unclear": category_counts(
                finding_table, "main_finding__"
            )["unclear"]
        },
        total_valid,
        "main_findings_derived",
        "Derived variable",
        denominator_column="Total valid studies",
    )

    bias_counts = category_counts(bias_table, "bias_mitigation__")
    bias_reference_counts = {
        **{
            category: bias_counts[category]
            for category in [
                "smote",
                "adasyn_or_synthetic_sampling",
                "class_weights_or_cost_sensitive_learning",
                "under_or_over_sampling",
                "data_augmentation",
                "balanced_split_or_matching",
            ]
        },
        "Any bias mitigation / balancing technique reported": bias_counts[
            "any_technique_reported"
        ],
        "Reported no bias mitigation / balancing": bias_counts[
            "reported_no_bias_mitigation"
        ],
        "Not reported / missing": bias_counts["not_reported_or_missing"],
        "Valid text but uncategorized": bias_counts["valid_text_but_uncategorized"],
    }
    compare_reference_summary(
        rows,
        reference_root / "rq4_results" / "RQ4_bias_mitigation_summary.csv",
        bias_reference_counts,
        total_valid,
        "bias_mitigation",
        "Category",
        percentage_column="Percentage out of total valid papers",
    )

    performance_reference = (
        reference_root
        / "rq4_results"
        / "RQ4_accuracy_by_behavioral_modality.csv"
    )
    if performance_reference.is_file():
        reference = pd.read_csv(performance_reference).set_index("Behavioral modality")
        reference_names = {
            "gaze": "gaze_yes",
            "speech_language_audio": "speech_yes",
            "motor": "motor_yes",
        }
        numeric_columns = ["Total n", "Accuracy n", "Mean", "Median", "SD", "Min", "Max"]
        for modality, reference_name in reference_names.items():
            generated_row = performance_table.loc[
                performance_table["modality"] == modality
            ].iloc[0]
            for column in numeric_columns:
                expected = float(reference.loc[reference_name, column])
                generated = float(generated_row[column])
                difference = abs(expected - generated)
                rows.append(
                    {
                        "validation_group": "outcome_performance",
                        "key": f"{modality}::{column}",
                        "reference_file": display_path(performance_reference),
                        "reference_value": expected,
                        "generated_value": generated,
                        "absolute_difference": difference,
                        "status": "MATCH" if difference <= 1e-10 else "DISCREPANCY",
                        "explanation": "Exact RQ4.5 accuracy-by-modality statistic comparison.",
                    }
                )
    else:
        rows.append(
            {
                "validation_group": "outcome_performance",
                "key": "reference_file",
                "reference_file": display_path(performance_reference),
                "reference_value": "",
                "generated_value": "",
                "absolute_difference": "",
                "status": "REFERENCE_NOT_AVAILABLE",
                "explanation": "Existing RQ4.5 output was not available.",
            }
        )

    directly_validated_sections = {"modality_prevalence", "outcome_performance"}
    for _, result in count_table.iterrows():
        if result["section"] in directly_validated_sections:
            continue
        rows.append(
            {
                "validation_group": f"modality_cross_tab::{result['section']}",
                "key": f"{result['modality']}::{result['category']}",
                "reference_file": "row-level masks generated from established source rules",
                "reference_value": "",
                "generated_value": result["count"],
                "absolute_difference": "",
                "status": "DERIVED_FROM_VALIDATED_MASKS",
                "explanation": (
                    "No pre-existing modality-specific cross-tab exists. The result "
                    "intersects the validated non-exclusive modality mask with a category "
                    "mask whose overall total was compared with the existing RQ output."
                ),
            }
        )

    return pd.DataFrame(rows)


def compact_category_list(
    counts: pd.DataFrame,
    modality: str,
    section: str,
    limit: int | None = None,
    exclude: Sequence[str] = (),
    positive_only: bool = False,
) -> str:
    subset = counts[
        (counts["modality"] == modality)
        & (counts["section"] == section)
        & (~counts["category"].isin(exclude))
    ].sort_values(["count", "category"], ascending=[False, True])
    if positive_only:
        subset = subset[subset["count"] > 0]
    if limit is not None:
        subset = subset.head(limit)
    result = "; ".join(
        f"{row.category.replace('_', ' ')} {int(row['count'])}/{int(row.denominator)} "
        f"({row.percentage:.2f}%)"
        for _, row in subset.iterrows()
    )
    return result or "none"


def build_summary_table(
    counts: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for modality in MODALITIES:
        performance_row = performance.loc[performance["modality"] == modality].iloc[0]
        rows.append(
            {
                "Behavioral modality": DISPLAY_MODALITY[modality],
                "Study prevalence": compact_category_list(
                    counts, modality, "modality_prevalence"
                ),
                "Common task operationalizations": compact_category_list(
                    counts,
                    modality,
                    "task_broad",
                    limit=3,
                    exclude=["not_given", "multiple_task_types", "unclear"],
                    positive_only=True,
                ),
                "Dominant ML paradigm": compact_category_list(
                    counts, modality, "ml_paradigm", limit=2, positive_only=True
                ),
                "Common algorithm families": compact_category_list(
                    counts, modality, "algorithm_family", limit=3, positive_only=True
                ),
                "Common evaluation metrics": compact_category_list(
                    counts, modality, "evaluation_metrics", limit=4, positive_only=True
                ),
                "Outcome-performance pattern": (
                    f"Accuracy extracted for {int(performance_row['Accuracy n'])}/"
                    f"{int(performance_row['Total n'])} studies "
                    f"({performance_row['Accuracy availability percentage']:.2f}%); "
                    f"mean {performance_row['Mean']:.2f}%, "
                    f"median {performance_row['Median']:.2f}%"
                ),
                "Common main findings": compact_category_list(
                    counts,
                    modality,
                    "main_findings",
                    limit=3,
                    exclude=["unclear"],
                    positive_only=True,
                ),
                "Common limitations": compact_category_list(
                    counts,
                    modality,
                    "limitations",
                    limit=3,
                    exclude=["need_manual_revision", "empty_or_not_applicable"],
                    positive_only=True,
                ),
                "Most common future direction": compact_category_list(
                    counts,
                    modality,
                    "future_directions",
                    limit=1,
                    exclude=["not_given", "no_category_matched"],
                    positive_only=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    headers = [str(column) for column in frame.columns]
    rows = [
        [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def build_markdown_report(
    input_path: Path,
    sheet: str,
    source_hashes: Mapping[str, str],
    counts: pd.DataFrame,
    performance: pd.DataFrame,
    summary: pd.DataFrame,
    examples: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    lines = [
        "# Cross-research-question mapping of behavioral AI studies",
        "",
        "## 1. Methods and modality assignment",
        "",
        (
            f"The analysis used `{display_path(input_path)}` worksheet "
            f"`{sheet}` and read only Excel columns A:BY. As in the established "
            "analyses, rows 3-174 were retained and a study was valid when column A "
            "(title) was non-missing, giving N=172."
        ),
        "",
        (
            "Gaze, speech/language/audio, and motor membership used the exact "
            "non-exclusive yes/no logic applied to AO, AP, and AQ by RQ3 and RQ4. "
            "A multimodal study may therefore contribute to more than one modality. "
            "All other categories are non-exclusive where the source RQ analysis is "
            "non-exclusive."
        ),
        "",
        (
            "Category regex definitions were read from `scripts/rq2_.py` and "
            "`scripts/rq4_.py` with Python's AST parser. The scripts were not imported "
            "or executed. Source SHA-256 values: "
            + "; ".join(f"`{path}` `{digest}`" for path, digest in source_hashes.items())
            + "."
        ),
        "",
        (
            "The workbook's actual source worksheet is `final_data`; there is no "
            "worksheet named `final_annotated_data` in the inspected workbook."
        ),
        "",
        "## 2. Cross-question mapping table",
        "",
        dataframe_to_markdown(summary),
        "",
        "## 3. Modality-specific mappings",
        "",
    ]

    for modality in MODALITIES:
        modality_label = DISPLAY_MODALITY[modality]
        performance_row = performance.loc[performance["modality"] == modality].iloc[0]
        lines.extend(
            [
                f"### {modality_label}",
                "",
                "**A. Input and operationalization.** "
                + compact_category_list(counts, modality, "modality_prevalence")
                + ". Reliable RQ2 task categories in this modality were: "
                + compact_category_list(
                    counts,
                    modality,
                    "task_broad",
                    exclude=["multiple_task_types"],
                )
                + ".",
                "",
            ]
        )

        task_examples = examples[
            (examples["modality"] == modality)
            & (examples["section"] == "task_operationalization")
        ]
        for _, example in task_examples.iterrows():
            lines.append(
                f"- {display_text(example['category'])}: "
                f"**{display_text(example['study_title'])}** "
                f"(Excel row {int(example['excel_row'])}) - "
                f"{display_text(example['source_text'])}"
            )
        lines.extend(
            [
                "",
                "**B. Machine-learning paradigm.** "
                + compact_category_list(counts, modality, "ml_paradigm")
                + ".",
                "",
                "**C. Algorithm families.** Broad families: "
                + compact_category_list(counts, modality, "algorithm_family")
                + ". Prominent classical/neural subfamilies (modality denominator): "
                + compact_category_list(
                    pd.concat(
                        [
                            counts.assign(
                                section=counts["section"].replace(
                                    {
                                        "algorithm_classical_subcategory": "algorithm_subcategory",
                                        "algorithm_neural_subcategory": "algorithm_subcategory",
                                    }
                                )
                            )
                        ],
                        ignore_index=True,
                    ),
                    modality,
                    "algorithm_subcategory",
                    limit=6,
                )
                + ".",
                "",
            ]
        )
        algorithm_examples = examples[
            (examples["modality"] == modality)
            & (examples["section"] == "algorithm_family")
        ]
        for _, example in algorithm_examples.iterrows():
            lines.append(
                f"- {example['category'].replace('_', ' ')} example: "
                f"**{display_text(example['study_title'])}** - "
                f"{display_text(example['source_text'])}"
            )

        lines.extend(
            [
                "",
                "**D. Performance metrics used.** "
                + compact_category_list(counts, modality, "evaluation_metrics")
                + ". These counts describe metrics named by studies; they are distinct "
                "from the extracted best-accuracy outcome below.",
                "",
                (
                    "**E. Outcome performance.** RQ4.5 extracted an accuracy value for "
                    f"{int(performance_row['Accuracy n'])}/"
                    f"{int(performance_row['Total n'])} studies "
                    f"({performance_row['Accuracy availability percentage']:.2f}%). "
                    f"Mean={performance_row['Mean']:.2f}%, "
                    f"median={performance_row['Median']:.2f}%, "
                    f"SD={performance_row['SD']:.2f}, "
                    f"range={performance_row['Min']:.2f}-{performance_row['Max']:.2f}%. "
                    "RQ4.5 models performance as a continuous extracted accuracy, so no "
                    "new high/medium/low performance categories were created."
                ),
                "",
                "**F. Main reported findings.** Required finding categories: "
                + compact_category_list(
                    counts.loc[counts["category"].isin(REQUIRED_FINDING_CATEGORIES)],
                    modality,
                    "main_findings",
                )
                + ". All other positive established finding categories: "
                + compact_category_list(
                    counts.loc[
                        ~counts["category"].isin(REQUIRED_FINDING_CATEGORIES + ["unclear"])
                    ],
                    modality,
                    "main_findings",
                    positive_only=True,
                )
                + ".",
                "",
            ]
        )
        finding_examples = examples[
            (examples["modality"] == modality)
            & (examples["section"] == "main_findings")
        ]
        for _, example in finding_examples.iterrows():
            lines.append(
                f"- {example['category'].replace('_', ' ')}: "
                f"**{display_text(example['study_title'])}** - "
                f"{display_text(example['source_text'])}"
            )

        lines.extend(
            [
                "",
                "**G. Limitations and biases.** Established limitation categories: "
                + compact_category_list(counts, modality, "limitations")
                + ". Bias mitigation/balancing reporting: "
                + compact_category_list(counts, modality, "bias_mitigation")
                + ". The project does not define a separate study-level bias taxonomy; "
                "bias-related concerns remain in the limitation categories, while the "
                "RQ4 field records mitigation/balancing techniques.",
                "",
                "**H. Future directions.** "
                + compact_category_list(counts, modality, "future_directions")
                + ".",
                "",
            ]
        )
        future_examples = examples[
            (examples["modality"] == modality)
            & (examples["section"] == "future_directions")
        ]
        for _, example in future_examples.iterrows():
            lines.append(
                f"- Representative future-direction example "
                f"({example['category'].replace('_', ' ')}): "
                f"**{display_text(example['study_title'])}** - "
                f"{display_text(example['source_text'])}"
            )
        lines.append("")

    discrepancy_count = int((validation["status"] == "DISCREPANCY").sum())
    unavailable_count = int(
        (validation["status"] == "REFERENCE_NOT_AVAILABLE").sum()
    )
    lines.extend(
        [
            "## 4. Validation and denominator notes",
            "",
            (
                f"The validation table contains {len(validation)} checks. "
                f"Unexplained discrepancies: {discrepancy_count}. "
                f"Missing reference-output checks: {unavailable_count}."
            ),
            "",
            (
                "Direct checks compare modality totals, overall category counts, "
                "denominators, percentages, and RQ4.5 accuracy statistics with existing "
                "RQ outputs. New modality-specific cross-tabs have no prior direct "
                "output; each is marked `DERIVED_FROM_VALIDATED_MASKS` because it "
                "intersects row-level modality and category masks whose source totals "
                "were validated. Percentages use the non-exclusive modality study count "
                "as N. Consequently, modality Ns and category percentages are not "
                "expected to sum to 172 or 100% across modalities/categories."
            ),
            "",
            "## 5. Files generated",
            "",
            "- `mapping_underlying_counts.csv`: all modality-specific n/N percentages.",
            "- `mapping_performance_statistics.csv`: exact RQ4.5 accuracy summaries.",
            "- `mapping_study_membership.csv`: row-level modality and category flags.",
            "- `mapping_examples.csv`: auditable task, algorithm, finding, and future-direction examples.",
            "- `mapping_summary_table.csv`: compact manuscript/supplement table.",
            "- `mapping_validation.csv`: direct and derived validation checks.",
            "- `mapping_report.md`: this synthesis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataframe(
    frame: pd.DataFrame,
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {path}"
        )
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(text: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {path}"
        )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def preflight_output_targets(output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [output_dir / filename for filename in OUTPUT_FILENAMES if (output_dir / filename).exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing mapping outputs without --overwrite:\n"
            + formatted
        )


def main() -> int:
    args = parse_args()
    input_path = resolve_from_project(args.input)
    output_dir = resolve_from_project(args.output_dir)
    reference_root = resolve_from_project(args.reference_output_root)
    preflight_output_targets(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, valid_mask = load_annotation_data(input_path, args.sheet)
    total_valid = int(valid_mask.sum())
    rules, source_hashes = load_established_rules(Path(__file__).resolve().parent)

    modality_table = build_modality_table(frame, valid_mask)
    task_table = build_task_table(
        frame,
        valid_mask,
        rules["task_broad"],
        rules["task_subcategory"],
    )
    paradigm_text = rq4_clean_text(frame.iloc[:, COL_LEARNING_TYPE][valid_mask])
    paradigm_table = match_patterns(paradigm_text, rules["paradigm"], "paradigm")

    algorithm_text = rq4_clean_text(frame.iloc[:, COL_ALGORITHMS][valid_mask])
    algorithm_broad_table = pd.DataFrame({"algorithm_text": algorithm_text})
    algorithm_broad_table = algorithm_broad_table.join(
        match_patterns(algorithm_text, rules["algorithm_broad"], "algorithm_family")
    )
    algorithm_broad_table["algorithm_family__hybrid_or_multi_model_architectures"] = (
        algorithm_text.str.contains(rules["hybrid_algorithm"], regex=True, na=False)
    )
    algorithm_classical_table = pd.DataFrame({"algorithm_text": algorithm_text}).join(
        match_patterns(
            algorithm_text, rules["algorithm_classical"], "algorithm_classical"
        )
    )
    algorithm_neural_table = pd.DataFrame({"algorithm_text": algorithm_text}).join(
        match_patterns(algorithm_text, rules["algorithm_neural"], "algorithm_neural")
    )

    metric_text = rq4_clean_text(frame.iloc[:, COL_EVALUATION_METRICS][valid_mask])
    metric_table = match_patterns(
        metric_text, rules["evaluation_metrics"], "metric"
    )
    limitation_table = build_limitation_table(
        frame, valid_mask, rules["limitations"]
    )
    finding_table = build_finding_table(
        frame, valid_mask, rules["main_findings"]
    )
    future_table = build_future_table(
        frame, valid_mask, rules["future_directions"]
    )
    bias_table = build_bias_mitigation_table(
        frame,
        valid_mask,
        rules["bias_mitigation"],
        rules["bias_not_reported"],
        rules["bias_reported_no"],
    )

    accuracy_series = frame.apply(
        extract_accuracy_from_row,
        axis=1,
        other_metric_terms=rules["other_metric_terms"],
    )

    count_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    for modality in MODALITIES:
        modality_mask = modality_table[f"{modality}_yes"]
        modality_count = int(modality_mask.sum())
        count_rows.append(
            {
                "modality": modality,
                "modality_label": DISPLAY_MODALITY[modality],
                "section": "modality_prevalence",
                "category": "yes",
                "count": modality_count,
                "denominator": total_valid,
                "percentage": pct(modality_count, total_valid),
                "denominator_scope": "all title-valid studies",
                "source_script": "scripts/rq3_.py and scripts/rq4_.py",
                "source_code_section": "gaze/motor/speech yes-no modality summary",
                "source_column": {
                    "gaze": "AO",
                    "speech_language_audio": "AP",
                    "motor": "AQ",
                }[modality],
                "nonexclusive": True,
            }
        )

        add_count_rows(
            count_rows,
            modality,
            "task_broad",
            task_table,
            [column for column in task_table if column.startswith("task__")],
            modality_mask,
            "scripts/rq2_.py",
            "compute_task_type broad categories",
            "BP",
            category_prefix="task__",
        )
        add_count_rows(
            count_rows,
            modality,
            "task_subcategory",
            task_table,
            [
                column
                for column in task_table
                if column.startswith("task_subcategory__")
            ],
            modality_mask,
            "scripts/rq2_.py",
            "compute_task_type finer task/protocol subcategories",
            "BP",
            category_prefix="task_subcategory__",
        )
        add_count_rows(
            count_rows,
            modality,
            "ml_paradigm",
            paradigm_table,
            list(paradigm_table.columns),
            modality_mask,
            "scripts/rq4_.py",
            "machine_learning_paradigm",
            "AW",
            category_prefix="paradigm__",
        )
        add_count_rows(
            count_rows,
            modality,
            "algorithm_family",
            algorithm_broad_table,
            [
                column
                for column in algorithm_broad_table
                if column.startswith("algorithm_family__")
            ],
            modality_mask,
            "scripts/rq4_.py",
            "algorithms_broad and algorithms_hybrid",
            "AU",
            category_prefix="algorithm_family__",
        )
        add_count_rows(
            count_rows,
            modality,
            "algorithm_classical_subcategory",
            algorithm_classical_table,
            [
                column
                for column in algorithm_classical_table
                if column.startswith("algorithm_classical__")
            ],
            modality_mask,
            "scripts/rq4_.py",
            "algorithms_further_basic_ml",
            "AU",
            category_prefix="algorithm_classical__",
        )
        add_count_rows(
            count_rows,
            modality,
            "algorithm_neural_subcategory",
            algorithm_neural_table,
            [
                column
                for column in algorithm_neural_table
                if column.startswith("algorithm_neural__")
            ],
            modality_mask,
            "scripts/rq4_.py",
            "algorithms_further_neural_networks",
            "AU",
            category_prefix="algorithm_neural__",
        )
        add_count_rows(
            count_rows,
            modality,
            "evaluation_metrics",
            metric_table,
            list(metric_table.columns),
            modality_mask,
            "scripts/rq4_.py",
            "evaluation_metrics",
            "AX",
            category_prefix="metric__",
        )
        add_count_rows(
            count_rows,
            modality,
            "main_findings",
            finding_table,
            [
                column
                for column in finding_table
                if column.startswith("main_finding__")
            ],
            modality_mask,
            "scripts/rq2_.py",
            "goal_finding_hybrid_summary main finding subcategories",
            "BT",
            category_prefix="main_finding__",
        )
        add_count_rows(
            count_rows,
            modality,
            "limitations",
            limitation_table,
            [
                column
                for column in limitation_table
                if column.startswith("limitation__")
            ],
            modality_mask,
            "scripts/rq2_.py",
            "rq_limitation_categories",
            "K, L, BR",
            category_prefix="limitation__",
        )
        add_count_rows(
            count_rows,
            modality,
            "bias_mitigation",
            bias_table,
            [
                column
                for column in bias_table
                if column.startswith("bias_mitigation__")
            ],
            modality_mask,
            "scripts/rq4_.py",
            "compute_bias_mitigation",
            "BC",
            category_prefix="bias_mitigation__",
        )
        add_count_rows(
            count_rows,
            modality,
            "future_directions",
            future_table,
            [
                column
                for column in future_table
                if column.startswith("future_direction__")
            ],
            modality_mask,
            "scripts/rq2_.py",
            "compute_future_goals_categories",
            "BW",
            category_prefix="future_direction__",
        )

        values = pd.to_numeric(
            accuracy_series.reindex(modality_mask.index)[modality_mask],
            errors="coerce",
        ).dropna()
        performance_rows.append(
            {
                "modality": modality,
                "modality_label": DISPLAY_MODALITY[modality],
                "Total n": modality_count,
                "Accuracy n": int(values.shape[0]),
                "Accuracy availability percentage": pct(
                    int(values.shape[0]), modality_count
                ),
                "Mean": values.mean(),
                "Median": values.median(),
                "SD": values.std(),
                "Min": values.min(),
                "Max": values.max(),
                "source_script": "scripts/rq4_.py",
                "source_code_section": "RQ4.5 accuracy by behavioral modality",
                "denominator_note": (
                    "Total n is modality membership; Accuracy n is non-missing "
                    "RQ4-extracted accuracy within that modality."
                ),
            }
        )
        count_rows.extend(
            [
                {
                    "modality": modality,
                    "modality_label": DISPLAY_MODALITY[modality],
                    "section": "outcome_performance",
                    "category": "extracted_accuracy_available",
                    "count": int(values.shape[0]),
                    "denominator": modality_count,
                    "percentage": pct(int(values.shape[0]), modality_count),
                    "denominator_scope": "all studies assigned to this modality",
                    "source_script": "scripts/rq4_.py",
                    "source_code_section": "RQ4.5 accuracy extraction and modality summary",
                    "source_column": "AX and AZ",
                    "nonexclusive": False,
                },
                {
                    "modality": modality,
                    "modality_label": DISPLAY_MODALITY[modality],
                    "section": "outcome_performance",
                    "category": "no_extracted_accuracy",
                    "count": modality_count - int(values.shape[0]),
                    "denominator": modality_count,
                    "percentage": pct(
                        modality_count - int(values.shape[0]), modality_count
                    ),
                    "denominator_scope": "all studies assigned to this modality",
                    "source_script": "scripts/rq4_.py",
                    "source_code_section": "RQ4.5 accuracy extraction and modality summary",
                    "source_column": "AX and AZ",
                    "nonexclusive": False,
                },
            ]
        )

    counts = pd.DataFrame(count_rows)
    performance = pd.DataFrame(performance_rows)

    membership = pd.DataFrame(
        {
            "row_index": frame.index[valid_mask],
            "excel_row": frame.index[valid_mask] + 3,
            "study_title": frame.iloc[:, COL_TITLE][valid_mask],
            "participant_task": frame.iloc[:, COL_PARTICIPANT_TASK][valid_mask],
            "algorithm_text": frame.iloc[:, COL_ALGORITHMS][valid_mask],
            "learning_type_text": frame.iloc[:, COL_LEARNING_TYPE][valid_mask],
            "evaluation_metric_text": frame.iloc[:, COL_EVALUATION_METRICS][valid_mask],
            "best_model_text": frame.iloc[:, COL_BEST_MODEL][valid_mask],
            "best_performance_text": frame.iloc[:, COL_BEST_PERFORMANCE][valid_mask],
            "extracted_accuracy_percent": accuracy_series[valid_mask],
            "main_finding_text": frame.iloc[:, COL_MAIN_FINDINGS][valid_mask],
            "future_direction_text": frame.iloc[:, COL_FUTURE_DIRECTIONS][valid_mask],
        }
    ).set_index("row_index")
    for table in [
        modality_table,
        task_table.drop(columns=["task_text"]),
        paradigm_table,
        algorithm_broad_table.drop(columns=["algorithm_text"]),
        algorithm_classical_table.drop(columns=["algorithm_text"]),
        algorithm_neural_table.drop(columns=["algorithm_text"]),
        metric_table,
        finding_table.drop(columns=["main_finding_text"]),
        limitation_table,
        bias_table,
        future_table.drop(columns=["future_direction_text"]),
    ]:
        new_columns = [column for column in table if column not in membership.columns]
        membership = membership.join(table[new_columns], how="left")
    membership = membership.reset_index()

    example_rows = build_curated_task_examples(frame, modality_table)
    dynamic_examples = build_dynamic_examples(
        frame,
        modality_table,
        counts,
        {
            "algorithm_family": (
                algorithm_broad_table,
                "algorithm_family__",
                "algorithm_text",
            ),
            "main_findings": (
                finding_table,
                "main_finding__",
                "main_finding_text",
            ),
            "future_directions": (
                future_table,
                "future_direction__",
                "future_direction_text",
            ),
        },
    )
    examples = pd.DataFrame(example_rows + dynamic_examples)

    validation = build_validation_table(
        reference_root,
        total_valid,
        modality_table,
        task_table,
        paradigm_table,
        algorithm_broad_table,
        algorithm_classical_table,
        algorithm_neural_table,
        metric_table,
        limitation_table,
        finding_table,
        future_table,
        bias_table,
        performance,
        counts,
    )
    discrepancy_count = int((validation["status"] == "DISCREPANCY").sum())
    if discrepancy_count:
        discrepancies = validation.loc[
            validation["status"] == "DISCREPANCY",
            ["validation_group", "key", "reference_value", "generated_value"],
        ]
        raise RuntimeError(
            "Validation found unexplained discrepancies:\n"
            + discrepancies.to_string(index=False)
        )

    summary = build_summary_table(counts, performance)
    report = build_markdown_report(
        input_path,
        args.sheet,
        source_hashes,
        counts,
        performance,
        summary,
        examples,
        validation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(
        counts,
        output_dir / "mapping_underlying_counts.csv",
        args.overwrite,
    )
    write_dataframe(
        performance,
        output_dir / "mapping_performance_statistics.csv",
        args.overwrite,
    )
    write_dataframe(
        membership,
        output_dir / "mapping_study_membership.csv",
        args.overwrite,
    )
    write_dataframe(
        examples,
        output_dir / "mapping_examples.csv",
        args.overwrite,
    )
    write_dataframe(
        summary,
        output_dir / "mapping_summary_table.csv",
        args.overwrite,
    )
    write_dataframe(
        validation,
        output_dir / "mapping_validation.csv",
        args.overwrite,
    )
    write_text(report, output_dir / "mapping_report.md", args.overwrite)

    print(f"Valid studies: {total_valid}")
    print(
        "Modality counts: "
        + ", ".join(
            f"{modality}={int(modality_table[f'{modality}_yes'].sum())}"
            for modality in MODALITIES
        )
    )
    print(f"Validation checks: {len(validation)}; discrepancies: 0")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

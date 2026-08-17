"""Shared, non-scientific configuration and I/O helpers for review analyses."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RQ_TITLES = {
    1: "Participants",
    2: "Study design",
    3: "Behaviors",
    4: "AI techniques",
    5: "Paper writing and publishing trends",
}

RQ_QUESTIONS = {
    1: "What are the characteristics of participants included in AI-based autism prediction studies using behavioral data?",
    2: "How are AI-based autism prediction studies using behavioral data designed, conducted, and reported?",
    3: "How is behavioral data conceptualized and used in AI-based autism prediction?",
    4: "How are AI and machine learning techniques applied to behavioral data for autism prediction?",
    5: "How has the literature on AI-based autism prediction using behavioral data evolved over time?",
}

# Publication periods used throughout the Results and sampling scripts.
YEAR_GROUP_ORDER = [
    "2013-2017",
    "2018-2023",
    "2024-2026",
    "Missing / unreadable year",
    "Outside expected range",
]


def resolve_root(env_name: str, default_dirname: str) -> Path:
    """Resolve a shared data/output root, including relative environment values."""
    raw = os.environ.get(env_name)
    path = Path(raw).expanduser() if raw else PROJECT_ROOT / default_dirname
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_publication_year(value: Any) -> float:
    """Extract the first plausible four-digit publication year."""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value).strip())
    return int(match.group(1)) if match else np.nan


def classify_year_group(year: Any) -> str:
    """Assign the manuscript's publication-year periods."""
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


def rq_output_path(rq_number: int) -> Path:
    return resolve_root("ASD_REVIEW_OUTPUT_ROOT", "output") / f"rq{rq_number}_results"


def prefix_output_name(rq_number: int, filename: str) -> str:
    path = Path(filename)
    prefix = f"RQ{rq_number}"
    basename = path.name
    if basename.lower().startswith(prefix.lower() + "_"):
        basename = prefix + basename[len(prefix) :]
    else:
        basename = f"{prefix}_{basename}"
    return str(path.parent / basename) if path.parent != Path(".") else basename


def write_csv_atomic(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def save_rq_csv(frame: pd.DataFrame, rq_number: int, filename: str, index: bool = False) -> None:
    write_csv_atomic(frame, rq_output_path(rq_number) / prefix_output_name(rq_number, filename), index=index)

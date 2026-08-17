"""Shared Krippendorff alpha implementation for all ICR analyses."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value == ""


def krippendorff_alpha(unit_values: Iterable[Iterable[Any]], level: str = "nominal") -> tuple[float, str]:
    """Return alpha and an explicit undefined reason.

    ``unit_values`` contains one iterable of coder values per unit. Missing values
    are omitted. Nominal and interval distance functions are supported.
    """
    if level not in {"nominal", "interval"}:
        raise ValueError(f"Unsupported measurement level: {level}")
    groups = [[value for value in values if not is_missing(value)] for values in unit_values]
    groups = [values for values in groups if len(values) >= 2]
    if len(groups) < 2:
        return np.nan, "too_few_paired_observations"
    all_values = [value for values in groups for value in values]
    if len(set(map(str, all_values))) <= 1:
        return np.nan, "no_variation"

    def distance(left: Any, right: Any) -> float:
        return float(left - right) ** 2 if level == "interval" else float(str(left) != str(right))

    observed = [distance(values[i], values[j]) for values in groups for i in range(len(values)) for j in range(i + 1, len(values))]
    expected = [distance(all_values[i], all_values[j]) for i in range(len(all_values)) for j in range(i + 1, len(all_values))]
    if not observed:
        return np.nan, "too_few_paired_observations"
    if not expected:
        return np.nan, "too_few_values"
    expected_mean = float(np.mean(expected))
    if expected_mean == 0:
        return np.nan, "no_variation"
    return 1.0 - (float(np.mean(observed)) / expected_mean), ""

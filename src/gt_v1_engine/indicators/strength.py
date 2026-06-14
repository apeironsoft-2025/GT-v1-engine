import math

import pandas as pd

from gt_v1_engine.core.errors import DataValidationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import IndicatorDirection, IndicatorResult

_VALID_STRENGTH_VALUES: set[float] = {0.0, 0.25, 0.5, 0.75, 1.0}


def _is_invalid_number(value: float | None) -> bool:
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(numeric)


def normalize_raw_score(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    if _is_invalid_number(value) or _is_invalid_number(min_value) or _is_invalid_number(max_value):
        return 0.0
    if max_value <= min_value:
        return 0.0

    numeric = float(value)
    if numeric <= min_value:
        return 0.0
    if numeric >= max_value:
        return 1.0
    return (numeric - min_value) / (max_value - min_value)


def bucket_strength(value: float) -> float:
    if _is_invalid_number(value):
        return 0.0

    numeric = float(value)
    if numeric <= 0:
        return 0.0
    if numeric <= 0.25:
        return 0.25
    if numeric <= 0.50:
        return 0.5
    if numeric <= 0.75:
        return 0.75
    return 1.0


def normalize_and_bucket(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return bucket_strength(normalize_raw_score(value, min_value, max_value))


def direction_from_score(score: float, dead_zone: float = 0.0) -> str:
    if _is_invalid_number(score) or _is_invalid_number(dead_zone):
        return IndicatorDirection.NO_SIGNAL.value

    numeric_score = float(score)
    numeric_dead_zone = abs(float(dead_zone))
    if numeric_score > numeric_dead_zone:
        return IndicatorDirection.UP.value
    if numeric_score < -numeric_dead_zone:
        return IndicatorDirection.DOWN.value
    return IndicatorDirection.NO_SIGNAL.value


def validate_strength_series(series: pd.Series, column_name: str) -> None:
    non_null = series.dropna()
    invalid_values: set[object] = set()
    for value in non_null:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_values.add(value)
            continue
        if numeric not in _VALID_STRENGTH_VALUES:
            invalid_values.add(numeric)
    if invalid_values:
        raise DataValidationError(
            f"{column_name} contains invalid strength value(s): "
            + ", ".join(str(value) for value in invalid_values)
        )


def summarize_indicator_output(df: pd.DataFrame, indicator_name: str) -> IndicatorResult:
    direction_column = f"{indicator_name}_TD"
    strength_column = f"{indicator_name}_TS"
    require_columns(
        df,
        [direction_column, strength_column],
        f"{indicator_name} indicator output",
    )
    validate_strength_series(df[strength_column], strength_column)

    directions = df[direction_column]
    strength = df[strength_column].dropna()
    min_strength = None if strength.empty else float(strength.min())
    max_strength = None if strength.empty else float(strength.max())
    average_strength = None if strength.empty else float(strength.mean())

    return IndicatorResult(
        name=indicator_name,
        direction_column=direction_column,
        strength_column=strength_column,
        row_count=len(df),
        up_count=int((directions == IndicatorDirection.UP.value).sum()),
        down_count=int((directions == IndicatorDirection.DOWN.value).sum()),
        no_signal_count=int((directions == IndicatorDirection.NO_SIGNAL.value).sum()),
        min_strength=min_strength,
        max_strength=max_strength,
        average_strength=average_strength,
    )

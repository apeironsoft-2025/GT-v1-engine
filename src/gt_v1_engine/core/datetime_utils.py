import pandas as pd

from gt_v1_engine.core.errors import DataValidationError


def parse_utc_datetime(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise DataValidationError(f"Invalid datetime value: {value}")
    return parsed


def normalize_datetime_series(series: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(series, utc=True, errors="coerce")
    if normalized.isna().any():
        bad_count = int(normalized.isna().sum())
        raise DataValidationError(f"DateTime contains {bad_count} invalid value(s)")
    return normalized

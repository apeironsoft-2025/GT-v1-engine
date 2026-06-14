from pathlib import Path

import pandas as pd

from gt_v1.core.constants import REQUIRED_MARKET_COLUMNS, STANDARD_MARKET_COLUMNS
from gt_v1.core.datetime_utils import normalize_datetime_series
from gt_v1.core.errors import UnsupportedFormatError
from gt_v1.core.io_utils import ensure_file_exists
from gt_v1.core.validation import require_columns, require_non_empty_dataframe
from gt_v1.data.schema import COLUMN_ALIASES


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        normalized = COLUMN_ALIASES.get(str(column).strip().lower())
        if normalized:
            rename_map[column] = normalized
    return df.rename(columns=rename_map)


def load_market_data(input_path: Path) -> pd.DataFrame:
    resolved = ensure_file_exists(input_path)
    suffix = resolved.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(resolved)
    elif suffix == ".parquet":
        df = pd.read_parquet(resolved)
    else:
        raise UnsupportedFormatError(f"Unsupported market data format: {resolved.suffix}")

    df = _normalize_columns(df)
    if "SRP" not in df.columns and "Close" in df.columns:
        df["SRP"] = df["Close"]

    require_columns(df, REQUIRED_MARKET_COLUMNS, f"Market data {resolved}")
    require_non_empty_dataframe(df, f"Market data {resolved}")

    df["DateTime"] = normalize_datetime_series(df["DateTime"])
    df = df.sort_values("DateTime", ascending=True)
    df = df.drop_duplicates(subset=["DateTime"], keep="last")

    require_columns(df, STANDARD_MARKET_COLUMNS, f"Market data {resolved}")
    return df[STANDARD_MARKET_COLUMNS].reset_index(drop=True)

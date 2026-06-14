from collections.abc import Iterable

import pandas as pd

from gt_v1.core.errors import DataValidationError


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DataValidationError(f"{context} missing required column(s): {', '.join(missing)}")


def require_non_empty_dataframe(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise DataValidationError(f"{context} is empty")

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gt_v1_engine.core.constants import REQUIRED_MARKET_COLUMNS
from gt_v1_engine.core.errors import DataValidationError
from gt_v1_engine.core.io_utils import ensure_file_exists, write_dataframe_csv, write_json
from gt_v1_engine.data.schema import COLUMN_ALIASES


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _base_summary(input_path: Path, output_path: Path, summary_path: Path, started_at: str) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "original_row_count": 0,
        "cleaned_row_count": 0,
        "removed_empty_rows": 0,
        "removed_invalid_datetime_rows": 0,
        "removed_duplicate_datetime_rows": 0,
        "removed_invalid_ohlc_rows": 0,
        "required_columns": list(REQUIRED_MARKET_COLUMNS),
        "available_columns": [],
        "status": "FAILED",
        "error_message": None,
        "started_at": started_at,
        "finished_at": None,
    }


def _detect_delimiter(header_line: str) -> str:
    return "\t" if header_line.count("\t") > header_line.count(",") else ","


def _read_raw_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        header_line = file.readline()
        delimiter = _detect_delimiter(header_line)
        file.seek(0)
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader, [])
        first_data_row = next((row for row in reader if any(cell.strip() for cell in row)), [])

    if first_data_row and len(first_data_row) > len(header):
        extra_columns = [f"Extra_{index}" for index in range(1, len(first_data_row) - len(header) + 1)]
        return pd.read_csv(path, sep=delimiter, skiprows=1, names=[*header, *extra_columns], engine="python")

    return pd.read_csv(path, sep=delimiter, engine="python")


def _normalize_column_name(column: object) -> str:
    trimmed = str(column).strip()
    return COLUMN_ALIASES.get(trimmed.lower(), trimmed)


def clean_raw_market_csv(input_path: Path | str, output_path: Path | str, summary_path: Path | str) -> dict[str, Any]:
    started_at = _utc_now_iso()
    resolved_input = Path(input_path)
    resolved_output = Path(output_path)
    resolved_summary = Path(summary_path)
    summary = _base_summary(resolved_input, resolved_output, resolved_summary, started_at)

    try:
        ensure_file_exists(resolved_input)
        df = _read_raw_csv(resolved_input)
        summary["original_row_count"] = int(len(df))

        df = df.rename(columns={column: _normalize_column_name(column) for column in df.columns})
        summary["available_columns"] = [str(column) for column in df.columns]

        missing_columns = [column for column in REQUIRED_MARKET_COLUMNS if column not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Raw CSV missing required column(s): {', '.join(missing_columns)}")

        empty_mask = df.isna().all(axis=1)
        summary["removed_empty_rows"] = int(empty_mask.sum())
        df = df.loc[~empty_mask].copy()

        df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
        invalid_datetime_mask = df["DateTime"].isna()
        summary["removed_invalid_datetime_rows"] = int(invalid_datetime_mask.sum())
        df = df.loc[~invalid_datetime_mask].copy()

        df = df.sort_values("DateTime", ascending=True)
        duplicate_datetime_mask = df.duplicated(subset=["DateTime"], keep="first")
        summary["removed_duplicate_datetime_rows"] = int(duplicate_datetime_mask.sum())
        df = df.loc[~duplicate_datetime_mask].copy()

        for column in REQUIRED_MARKET_COLUMNS[1:]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        missing_ohlc_mask = df[REQUIRED_MARKET_COLUMNS[1:]].isna().any(axis=1)
        logic_mask = (
            (df["High"] >= df[["Open", "Close"]].max(axis=1))
            & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            & (df["High"] >= df["Low"])
        )
        invalid_ohlc_mask = missing_ohlc_mask | ~logic_mask
        summary["removed_invalid_ohlc_rows"] = int(invalid_ohlc_mask.sum())
        df = df.loc[~invalid_ohlc_mask].copy()

        summary["cleaned_row_count"] = int(len(df))
        summary["status"] = "SUCCESS"
        summary["error_message"] = None
        summary["finished_at"] = _utc_now_iso()

        write_dataframe_csv(df.reset_index(drop=True), resolved_output)
        write_json(resolved_summary, summary)
        return summary
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["error_message"] = str(exc)
        summary["finished_at"] = _utc_now_iso()
        write_json(resolved_summary, summary)
        raise

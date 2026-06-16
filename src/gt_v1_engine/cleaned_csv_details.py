from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console

from gt_v1_engine.core.constants import REQUIRED_MARKET_COLUMNS
from gt_v1_engine.core.io_utils import write_json

console = Console()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _base_summary(root_path: Path, relative_path: Path, absolute_path: Path, output_json: Path, started_at: str) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "error_message": None,
        "root_path": str(root_path),
        "relative_path": str(relative_path),
        "absolute_path": str(absolute_path),
        "file_name": absolute_path.name,
        "row_count": 0,
        "start_datetime": None,
        "end_datetime": None,
        "max_price_from_ohlc": None,
        "min_price_from_ohlc": None,
        "average_price_from_ohlc": None,
        "median_price_from_ohlc": None,
        "first_2_rows": [],
        "last_2_rows": [],
        "mid_2_rows": [],
        "required_columns": list(REQUIRED_MARKET_COLUMNS),
        "available_columns": [],
        "started_at": started_at,
        "finished_at": None,
        "output_json": str(output_json),
    }


def _resolve_cleaned_csv_path(root_path: Path | str, relative_path: Path | str) -> tuple[Path, Path, Path]:
    resolved_root = Path(root_path).resolve(strict=False)
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("relative_path must be relative to root_path")

    resolved_absolute = (resolved_root / relative).resolve(strict=False)
    try:
        resolved_absolute.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("relative_path escapes root_path") from exc

    return resolved_root, relative, resolved_absolute


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value) if isinstance(value, pd.Timestamp) else value


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return records


def _mid_two_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.head(0)
    start_index = max((len(df) // 2) - 1, 0)
    return df.iloc[start_index : start_index + 2]


def _apply_ohlc_aggregates(summary: dict[str, Any], df: pd.DataFrame) -> None:
    ohlc_columns = REQUIRED_MARKET_COLUMNS[1:]
    values = pd.to_numeric(df[ohlc_columns].stack(), errors="coerce").dropna()
    if values.empty:
        return

    summary["max_price_from_ohlc"] = float(values.max())
    summary["min_price_from_ohlc"] = float(values.min())
    summary["average_price_from_ohlc"] = float(values.mean())
    summary["median_price_from_ohlc"] = float(values.median())


def build_cleaned_csv_details(
    root_path: Path | str,
    relative_path: Path | str,
    output_json: Path | str,
) -> dict[str, Any]:
    started_at = _utc_now_iso()
    resolved_output = Path(output_json)

    try:
        resolved_root, relative, resolved_absolute = _resolve_cleaned_csv_path(root_path, relative_path)
    except Exception as exc:
        resolved_root = Path(root_path).resolve(strict=False)
        relative = Path(relative_path)
        resolved_absolute = (resolved_root / relative).resolve(strict=False)
        summary = _base_summary(resolved_root, relative, resolved_absolute, resolved_output, started_at)
        summary["error_message"] = str(exc)
        summary["finished_at"] = _utc_now_iso()
        write_json(resolved_output, summary)
        return summary

    summary = _base_summary(resolved_root, relative, resolved_absolute, resolved_output, started_at)

    try:
        if not resolved_absolute.exists() or not resolved_absolute.is_file():
            raise FileNotFoundError(f"{resolved_absolute} not found")

        df = pd.read_csv(resolved_absolute)
        summary["row_count"] = int(len(df))
        summary["available_columns"] = [str(column) for column in df.columns]

        missing_columns = [column for column in REQUIRED_MARKET_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Cleaned CSV missing required column(s): {', '.join(missing_columns)}")

        if not df.empty:
            summary["start_datetime"] = _json_safe_value(df["DateTime"].iloc[0])
            summary["end_datetime"] = _json_safe_value(df["DateTime"].iloc[-1])

        _apply_ohlc_aggregates(summary, df)
        summary["first_2_rows"] = _records_for_json(df.head(2))
        summary["last_2_rows"] = _records_for_json(df.tail(2))
        summary["mid_2_rows"] = _records_for_json(_mid_two_rows(df))
        summary["status"] = "SUCCESS"
        summary["error_message"] = None
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["error_message"] = str(exc)
    finally:
        summary["finished_at"] = _utc_now_iso()
        write_json(resolved_output, summary)

    return summary


def main(
    root_path: Path = typer.Option(..., "--root-path", help="Root directory for cleaned shared-storage CSV files."),
    relative_path: Path = typer.Option(..., "--relative-path", help="Cleaned CSV path relative to root path."),
    output_json: Path = typer.Option(..., "--output-json", help="Path for details JSON output."),
) -> None:
    """Analyze a cleaned shared-storage market CSV and write details JSON."""
    summary = build_cleaned_csv_details(root_path, relative_path, output_json)
    console.print(f"status: {summary['status']}")
    console.print(f"row_count: {summary['row_count']}")
    console.print(f"output_json: {summary['output_json']}")
    if summary["error_message"]:
        console.print(f"error_message: {summary['error_message']}")


if __name__ == "__main__":
    typer.run(main)

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gt_v1_engine.core.constants import NOT_ACTIVE
from gt_v1_engine.core.errors import DataValidationError, GTV1EngineError, IndicatorCalculationError
from gt_v1_engine.core.io_utils import write_dataframe_csv, write_dataframe_parquet, write_json
from gt_v1_engine.data.market_data_loader import load_market_data
from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.registry import (
    create_indicator,
    get_default_indicator_order,
    validate_indicator_order,
)
from gt_v1_engine.indicators.strength import summarize_indicator_output, validate_strength_series

VALID_DIRECTIONS = {
    IndicatorDirection.UP.value,
    IndicatorDirection.DOWN.value,
    IndicatorDirection.NO_SIGNAL.value,
}


@dataclass
class IndicatorExecutionSummary:
    pair: str
    timeframe: str
    input_path: str
    output_csv_path: str | None
    output_parquet_path: str | None
    selected_indicators: list[str]
    indicator_order: list[str]
    row_count: int
    first_datetime: str | None
    last_datetime: str | None
    generated_columns: list[str]
    indicator_summaries: dict[str, dict[str, Any]]
    validation_status: str
    production_activation_status: str
    broker_order_allowed: bool
    live_trading_allowed: bool
    generated_at_utc: str


def run_indicator_executor(
    input_path: Path,
    pair: str,
    timeframe: str,
    indicators: list[str],
    output_csv: Path,
    output_parquet: Path | None = None,
    summary_json: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_pair_timeframe(pair, timeframe)
    indicator_order = get_default_indicator_order()
    selected_indicators = (
        validate_indicator_order(indicators, indicator_order) if indicators else indicator_order
    )

    try:
        df = load_market_data(input_path)
        original_columns = list(df.columns)
        indicator_summaries: dict[str, dict[str, Any]] = {}

        for indicator_name in selected_indicators:
            indicator = create_indicator(indicator_name)
            df = indicator.calculate(df)
            _validate_indicator_columns(df, indicator_name)
            indicator_summaries[indicator_name] = summarize_indicator_output(
                df,
                indicator_name,
            ).model_dump()

        generated_columns = [column for column in df.columns if column not in original_columns]
        summary = IndicatorExecutionSummary(
            pair=pair.strip(),
            timeframe=timeframe.strip(),
            input_path=str(Path(input_path)),
            output_csv_path=str(Path(output_csv)) if output_csv else None,
            output_parquet_path=str(Path(output_parquet)) if output_parquet else None,
            selected_indicators=selected_indicators,
            indicator_order=indicator_order,
            row_count=len(df),
            first_datetime=_datetime_to_string(df["DateTime"].iloc[0]) if not df.empty else None,
            last_datetime=_datetime_to_string(df["DateTime"].iloc[-1]) if not df.empty else None,
            generated_columns=generated_columns,
            indicator_summaries=indicator_summaries,
            validation_status="PASS",
            production_activation_status=NOT_ACTIVE,
            broker_order_allowed=False,
            live_trading_allowed=False,
            generated_at_utc=datetime.now(UTC).isoformat(),
        )
        summary_dict = asdict(summary)

        _write_outputs(df, output_csv, output_parquet, summary_json, summary_dict)
        return df, summary_dict
    except GTV1EngineError:
        raise
    except Exception as exc:
        raise IndicatorCalculationError(f"Indicator executor failed: {exc}") from exc


def _validate_pair_timeframe(pair: str, timeframe: str) -> None:
    if not isinstance(pair, str) or not pair.strip():
        raise DataValidationError("pair cannot be empty")
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise DataValidationError("timeframe cannot be empty")


def _validate_indicator_columns(df: pd.DataFrame, indicator_name: str) -> None:
    direction_column = f"{indicator_name}_TD"
    strength_column = f"{indicator_name}_TS"
    missing = [column for column in (direction_column, strength_column) if column not in df.columns]
    if missing:
        raise DataValidationError(
            f"{indicator_name} output missing required column(s): {', '.join(missing)}"
        )
    if df[direction_column].isna().any():
        raise DataValidationError(f"{direction_column} contains null values")
    if df[strength_column].isna().any():
        raise DataValidationError(f"{strength_column} contains null values")

    invalid_directions = sorted(set(df[direction_column]) - VALID_DIRECTIONS)
    if invalid_directions:
        raise DataValidationError(
            f"{direction_column} contains invalid direction value(s): "
            + ", ".join(str(value) for value in invalid_directions)
        )
    validate_strength_series(df[strength_column], strength_column)


def _write_outputs(
    df: pd.DataFrame,
    output_csv: Path,
    output_parquet: Path | None,
    summary_json: Path | None,
    summary: dict[str, Any],
) -> None:
    try:
        write_dataframe_csv(df, output_csv)
        if output_parquet is not None:
            write_dataframe_parquet(df, output_parquet)
        if summary_json is not None:
            write_json(summary_json, summary)
    except Exception as exc:
        raise IndicatorCalculationError(f"Failed to write indicator executor output: {exc}") from exc


def _datetime_to_string(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

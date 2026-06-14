from pathlib import Path
from typing import Any

import pandas as pd

from gt_v1_engine.backtesting.candle_outcome import (
    BacktestResult,
    TradeSide,
    evaluate_future_candles,
)
from gt_v1_engine.backtesting.summary import (
    best_indicator_by_metric,
    generated_at_utc,
    safety_fields,
)
from gt_v1_engine.core.errors import DataValidationError, GTV1EngineError, IndicatorCalculationError
from gt_v1_engine.core.io_utils import write_dataframe_csv, write_json
from gt_v1_engine.core.pip_utils import resolve_pip_size
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.data.market_data_loader import load_market_data
from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.registry import (
    direction_column_for,
    get_default_indicator_order,
    strength_column_for,
    validate_indicator_names,
    validate_indicator_order,
)


def backtest_indicator(
    input_path: Path,
    pair: str,
    timeframe: str,
    indicator: str,
    start: str | None,
    end: str | None,
    horizon_candles: int,
    target_pips: float,
    stop_pips: float,
    pip_size: float | None,
    output_csv: Path,
    include_no_signal_rows: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        _validate_backtest_inputs(pair, timeframe, horizon_candles, target_pips, stop_pips, pip_size)
        normalized_indicator = validate_indicator_names([indicator])[0]
        resolved_pip_size = resolve_pip_size(pair, pip_size)
        if resolved_pip_size <= 0:
            raise DataValidationError("pip_size must be greater than 0")

        df = load_market_data(input_path)
        direction_column = direction_column_for(normalized_indicator)
        strength_column = strength_column_for(normalized_indicator)
        require_columns(df, [direction_column, strength_column], f"{normalized_indicator} backtest input")

        filtered = _filter_by_datetime(df, start, end)
        rows = _build_backtest_rows(
            df=df,
            filtered=filtered,
            pair=pair.strip(),
            timeframe=timeframe.strip(),
            indicator=normalized_indicator,
            direction_column=direction_column,
            strength_column=strength_column,
            horizon_candles=horizon_candles,
            target_pips=target_pips,
            stop_pips=stop_pips,
            pip_size=resolved_pip_size,
            include_no_signal_rows=include_no_signal_rows,
        )
        result_df = pd.DataFrame(rows, columns=_output_columns())
        summary = _indicator_summary(
            result_df=result_df,
            filtered=filtered,
            pair=pair.strip(),
            timeframe=timeframe.strip(),
            indicator=normalized_indicator,
            input_path=input_path,
            output_csv=output_csv,
            start=start,
            end=end,
            horizon_candles=horizon_candles,
            target_pips=target_pips,
            stop_pips=stop_pips,
            pip_size=resolved_pip_size,
        )
        write_dataframe_csv(result_df, output_csv)
        return result_df, summary
    except GTV1EngineError:
        raise
    except Exception as exc:
        raise IndicatorCalculationError(f"Indicator backtest failed: {exc}") from exc


def backtest_all_indicators(
    input_path: Path,
    pair: str,
    timeframe: str,
    indicators: list[str],
    start: str | None,
    end: str | None,
    horizon_candles: int,
    target_pips: float,
    stop_pips: float,
    pip_size: float | None,
    output_dir: Path,
    summary_json: Path,
    include_no_signal_rows: bool = False,
) -> dict[str, Any]:
    try:
        selected = (
            validate_indicator_order(validate_indicator_names(indicators), get_default_indicator_order())
            if indicators
            else get_default_indicator_order()
        )
        indicator_summaries: dict[str, dict[str, Any]] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        for indicator in selected:
            output_csv = output_dir / f"{pair}_{timeframe}_{indicator}_backtest.csv"
            _, summary = backtest_indicator(
                input_path=input_path,
                pair=pair,
                timeframe=timeframe,
                indicator=indicator,
                start=start,
                end=end,
                horizon_candles=horizon_candles,
                target_pips=target_pips,
                stop_pips=stop_pips,
                pip_size=pip_size,
                output_csv=output_csv,
                include_no_signal_rows=include_no_signal_rows,
            )
            indicator_summaries[indicator] = summary

        combined_summary = {
            "pair": pair.strip(),
            "timeframe": timeframe.strip(),
            "selected_indicators": selected,
            "indicator_summaries": indicator_summaries,
            "best_indicator_by_net_pips": best_indicator_by_metric(
                indicator_summaries,
                "total_realized_pips",
            ),
            "best_indicator_by_win_rate": best_indicator_by_metric(indicator_summaries, "win_rate"),
            "validation_status": "PASS",
            **safety_fields(),
            "generated_at_utc": generated_at_utc(),
        }
        write_json(summary_json, combined_summary)
        return combined_summary
    except GTV1EngineError:
        raise
    except Exception as exc:
        raise IndicatorCalculationError(f"All-indicator backtest failed: {exc}") from exc


def _validate_backtest_inputs(
    pair: str,
    timeframe: str,
    horizon_candles: int,
    target_pips: float,
    stop_pips: float,
    pip_size: float | None,
) -> None:
    if not isinstance(pair, str) or not pair.strip():
        raise DataValidationError("pair cannot be empty")
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise DataValidationError("timeframe cannot be empty")
    if horizon_candles <= 0:
        raise DataValidationError("horizon_candles must be greater than 0")
    if target_pips <= 0:
        raise DataValidationError("target_pips must be greater than 0")
    if stop_pips <= 0:
        raise DataValidationError("stop_pips must be greater than 0")
    if pip_size is not None and pip_size <= 0:
        raise DataValidationError("pip_size must be greater than 0")


def _filter_by_datetime(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    filtered = df
    try:
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None
    except Exception as exc:
        raise DataValidationError(f"Invalid date range: {exc}") from exc
    if start_ts is not None and start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts is not None and end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise DataValidationError("start must be before or equal to end")
    if start_ts is not None:
        filtered = filtered[filtered["DateTime"] >= start_ts]
    if end_ts is not None:
        filtered = filtered[filtered["DateTime"] <= end_ts]
    return filtered.sort_values("DateTime").reset_index(drop=False).rename(columns={"index": "source_index"})


def _build_backtest_rows(
    df: pd.DataFrame,
    filtered: pd.DataFrame,
    pair: str,
    timeframe: str,
    indicator: str,
    direction_column: str,
    strength_column: str,
    horizon_candles: int,
    target_pips: float,
    stop_pips: float,
    pip_size: float,
    include_no_signal_rows: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in filtered.itertuples(index=False):
        indicator_td = getattr(row, direction_column)
        if indicator_td == IndicatorDirection.NO_SIGNAL.value and not include_no_signal_rows:
            continue

        signal_side = _side_from_direction(indicator_td)
        entry_price = float(getattr(row, "SRP") if hasattr(row, "SRP") else getattr(row, "Close"))
        source_index = int(getattr(row, "source_index"))
        future_df = df.iloc[source_index + 1 : source_index + 1 + horizon_candles]

        if signal_side is None:
            outcome = _no_signal_outcome()
        else:
            outcome = evaluate_future_candles(
                future_df=future_df,
                side=signal_side,
                entry_price=entry_price,
                target_pips=target_pips,
                stop_pips=stop_pips,
                pip_size=pip_size,
            )

        rows.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "indicator": indicator,
                "signal_datetime": _datetime_to_string(getattr(row, "DateTime")),
                "signal_side": signal_side,
                "indicator_td": indicator_td,
                "indicator_ts": float(getattr(row, strength_column)),
                "entry_price": entry_price,
                "pip_size": pip_size,
                "target_pips": target_pips,
                "stop_pips": stop_pips,
                "horizon_candles": horizon_candles,
                **outcome,
            }
        )
    return rows


def _side_from_direction(direction: str) -> str | None:
    if direction == IndicatorDirection.UP.value:
        return TradeSide.BUY.value
    if direction == IndicatorDirection.DOWN.value:
        return TradeSide.SELL.value
    if direction == IndicatorDirection.NO_SIGNAL.value:
        return None
    raise DataValidationError(f"Invalid indicator direction: {direction}")


def _no_signal_outcome() -> dict[str, Any]:
    return {
        "take_profit_price": None,
        "stop_loss_price": None,
        "result": None,
        "realized_pips": 0.0,
        "close_datetime": None,
        "close_candle_offset": None,
        "close_price": None,
        "close_reason": None,
        "both_hit_same_candle": False,
    }


def _indicator_summary(
    result_df: pd.DataFrame,
    filtered: pd.DataFrame,
    pair: str,
    timeframe: str,
    indicator: str,
    input_path: Path,
    output_csv: Path,
    start: str | None,
    end: str | None,
    horizon_candles: int,
    target_pips: float,
    stop_pips: float,
    pip_size: float,
) -> dict[str, Any]:
    tested = result_df[result_df["signal_side"].notna()]
    tested_trades = len(tested)
    win_count = int((tested["result"] == BacktestResult.WIN.value).sum()) if tested_trades else 0
    loss_count = int((tested["result"] == BacktestResult.LOSS.value).sum()) if tested_trades else 0
    no_hit_count = int((tested["result"] == BacktestResult.NO_HIT.value).sum()) if tested_trades else 0
    no_future_data_count = (
        int((tested["result"] == BacktestResult.NO_FUTURE_DATA.value).sum()) if tested_trades else 0
    )
    total_realized_pips = float(tested["realized_pips"].sum()) if tested_trades else 0.0
    signal_rows = int((filtered[f"{indicator}_TD"] != IndicatorDirection.NO_SIGNAL.value).sum())
    no_signal_rows = int((filtered[f"{indicator}_TD"] == IndicatorDirection.NO_SIGNAL.value).sum())

    return {
        "pair": pair,
        "timeframe": timeframe,
        "indicator": indicator,
        "input_path": str(Path(input_path)),
        "output_csv_path": str(Path(output_csv)),
        "start_datetime": start,
        "end_datetime": end,
        "horizon_candles": horizon_candles,
        "target_pips": target_pips,
        "stop_pips": stop_pips,
        "pip_size": pip_size,
        "total_rows_in_period": len(filtered),
        "signal_rows": signal_rows,
        "no_signal_rows": no_signal_rows,
        "tested_trades": tested_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "no_hit_count": no_hit_count,
        "no_future_data_count": no_future_data_count,
        "both_hit_same_candle_count": int(tested["both_hit_same_candle"].sum()) if tested_trades else 0,
        "total_realized_pips": total_realized_pips,
        "average_pips_per_trade": total_realized_pips / tested_trades if tested_trades else 0.0,
        "win_rate": win_count / tested_trades if tested_trades else 0.0,
        "loss_rate": loss_count / tested_trades if tested_trades else 0.0,
        "validation_status": "PASS",
        **safety_fields(),
        "generated_at_utc": generated_at_utc(),
    }


def _output_columns() -> list[str]:
    return [
        "pair",
        "timeframe",
        "indicator",
        "signal_datetime",
        "signal_side",
        "indicator_td",
        "indicator_ts",
        "entry_price",
        "pip_size",
        "target_pips",
        "stop_pips",
        "horizon_candles",
        "take_profit_price",
        "stop_loss_price",
        "result",
        "realized_pips",
        "close_datetime",
        "close_candle_offset",
        "close_price",
        "close_reason",
        "both_hit_same_candle",
    ]


def _datetime_to_string(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

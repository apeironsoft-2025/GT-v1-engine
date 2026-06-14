from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gt_v1_engine.backtesting.summary import generated_at_utc
from gt_v1_engine.core.constants import NOT_ACTIVE, RULE171_NAME
from gt_v1_engine.core.errors import DataValidationError, GTV1EngineError, IndicatorCalculationError
from gt_v1_engine.core.io_utils import write_dataframe_csv, write_json
from gt_v1_engine.core.pip_utils import resolve_pip_size
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.data.market_data_loader import load_market_data
from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.registry import (
    direction_column_for,
    strength_column_for,
    validate_indicator_order,
)
from gt_v1_engine.rules.rule_config import Rule171Config, load_rule171_config

WIN_CLOSE = "WIN_CLOSE"
LOSS_CLOSE = "LOSS_CLOSE"


@dataclass
class Rule171ExecutionOverrides:
    pair: str | None = None
    timeframe: str | None = None
    indicators: list[str] | None = None
    start: str | None = None
    end: str | None = None
    pip_size: float | None = None
    strength_threshold: float | None = None
    entry_confirmation_required: int | None = None
    take_profit_pips: float | None = None
    stop_loss_pips: float | None = None
    max_holding_candles: int | None = None


@dataclass
class _RuntimeSettings:
    pair: str
    timeframe: str
    selected_indicators: list[str]
    indicator_order: list[str]
    start: str
    end: str
    pip_size: float
    strength_threshold: float
    entry_confirmation_required: int
    take_profit_pips: float
    stop_loss_pips: float
    max_holding_candles: int


def run_rule171_backtest(
    input_path: Path,
    config_path: Path,
    output_csv: Path,
    output_summary: Path,
    overrides: Rule171ExecutionOverrides | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        config = load_rule171_config(config_path)
        _validate_research_only(config)
        settings = _runtime_settings(config, overrides)
        df = load_market_data(input_path)
        period_df = _filter_period(df, settings.start, settings.end)
        _validate_rule171_input(period_df, config, settings)

        trade_rows, unresolved_signal_candidates, blocked_rows_while_open = _execute_rule171(
            period_df,
            config,
            settings,
        )
        output_df = pd.DataFrame(trade_rows, columns=_output_columns(settings.selected_indicators))
        summary = _summary(
            output_df=output_df,
            config=config,
            settings=settings,
            input_path=input_path,
            output_csv=output_csv,
            output_summary=output_summary,
            unresolved_signal_candidates=unresolved_signal_candidates,
            blocked_rows_while_open=blocked_rows_while_open,
        )

        try:
            write_dataframe_csv(output_df, output_csv)
            write_json(output_summary, summary)
        except Exception as exc:
            raise IndicatorCalculationError(f"Failed to write Rule171 output: {exc}") from exc

        return output_df, summary
    except GTV1EngineError:
        raise
    except Exception as exc:
        raise IndicatorCalculationError(f"Rule171 backtest failed: {exc}") from exc


def _runtime_settings(
    config: Rule171Config,
    overrides: Rule171ExecutionOverrides | None,
) -> _RuntimeSettings:
    overrides = overrides or Rule171ExecutionOverrides()
    pair = overrides.pair if overrides.pair is not None else config.market.default_pair
    timeframe = overrides.timeframe if overrides.timeframe is not None else config.market.default_timeframe
    start = overrides.start if overrides.start is not None else config.market.default_start
    end = overrides.end if overrides.end is not None else config.market.default_end
    indicator_order = validate_indicator_order(config.indicators.order, config.indicators.order)
    selected = overrides.indicators if overrides.indicators is not None else config.indicators.selected
    selected_indicators = validate_indicator_order(selected, indicator_order)
    pip_size = overrides.pip_size
    if pip_size is None:
        pip_size = resolve_pip_size(pair, config.pip_size.override)

    settings = _RuntimeSettings(
        pair=pair.strip() if isinstance(pair, str) else "",
        timeframe=timeframe.strip() if isinstance(timeframe, str) else "",
        selected_indicators=selected_indicators,
        indicator_order=indicator_order,
        start=start,
        end=end,
        pip_size=float(pip_size),
        strength_threshold=(
            float(overrides.strength_threshold)
            if overrides.strength_threshold is not None
            else float(config.entry.strength_threshold)
        ),
        entry_confirmation_required=(
            int(overrides.entry_confirmation_required)
            if overrides.entry_confirmation_required is not None
            else int(config.entry.entry_confirmation_required)
        ),
        take_profit_pips=(
            float(overrides.take_profit_pips)
            if overrides.take_profit_pips is not None
            else float(config.trade_management.take_profit_pips)
        ),
        stop_loss_pips=(
            float(overrides.stop_loss_pips)
            if overrides.stop_loss_pips is not None
            else float(config.trade_management.stop_loss_pips)
        ),
        max_holding_candles=(
            int(overrides.max_holding_candles)
            if overrides.max_holding_candles is not None
            else int(config.trade_management.max_holding_candles)
        ),
    )
    _validate_runtime_settings(settings)
    return settings


def _validate_runtime_settings(settings: _RuntimeSettings) -> None:
    if not settings.pair:
        raise DataValidationError("pair cannot be empty")
    if not settings.timeframe:
        raise DataValidationError("timeframe cannot be empty")
    if settings.pip_size <= 0:
        raise DataValidationError("pip_size must be greater than 0")
    if settings.strength_threshold <= 0:
        raise DataValidationError("strength_threshold must be greater than 0")
    if settings.entry_confirmation_required <= 0:
        raise DataValidationError("entry_confirmation_required must be greater than 0")
    if settings.take_profit_pips <= 0:
        raise DataValidationError("take_profit_pips must be greater than 0")
    if settings.stop_loss_pips <= 0:
        raise DataValidationError("stop_loss_pips must be greater than 0")
    if settings.max_holding_candles <= 0:
        raise DataValidationError("max_holding_candles must be greater than 0")


def _validate_research_only(config: Rule171Config) -> None:
    if config.production_activation_status != NOT_ACTIVE:
        raise DataValidationError("production_activation_status must be NOT_ACTIVE")
    if config.safety.live_trading_allowed:
        raise DataValidationError("live_trading_allowed must be false")
    if config.safety.broker_order_allowed:
        raise DataValidationError("broker_order_allowed must be false")


def _filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except Exception as exc:
        raise DataValidationError(f"Invalid start/end datetime: {exc}") from exc
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    if start_ts > end_ts:
        raise DataValidationError("start must be before or equal to end")
    return df[(df["DateTime"] >= start_ts) & (df["DateTime"] <= end_ts)].reset_index(drop=True)


def _validate_rule171_input(
    df: pd.DataFrame,
    config: Rule171Config,
    settings: _RuntimeSettings,
) -> None:
    ohlc = config.data.ohlc_columns
    require_columns(
        df,
        [
            config.data.datetime_column,
            config.data.entry_price_column,
            ohlc["open"],
            ohlc["high"],
            ohlc["low"],
            ohlc["close"],
        ],
        "Rule171 input",
    )
    for indicator in settings.selected_indicators:
        require_columns(
            df,
            [direction_column_for(indicator), strength_column_for(indicator)],
            f"Rule171 {indicator} input",
        )


def _execute_rule171(
    df: pd.DataFrame,
    config: Rule171Config,
    settings: _RuntimeSettings,
) -> tuple[list[dict[str, Any]], int, int]:
    trades: list[dict[str, Any]] = []
    confirmation_side: str | None = None
    confirmation_count = 0
    confirmation_datetimes: list[str] = []
    blocked_until_index = -1
    blocked_rows_while_open = 0
    unresolved_signal_candidates = 0
    signal_sequence = 1

    buy_patterns, sell_patterns = _patterns_for_selection(config, settings)

    for index, row in df.iterrows():
        if index <= blocked_until_index:
            blocked_rows_while_open += 1
            continue

        indicator_pattern = _indicator_pattern(row, settings.selected_indicators)
        candidate_side = _candidate_side(indicator_pattern, buy_patterns, sell_patterns)
        if candidate_side is None:
            if not config.entry.preserve_cycle_on_no_signal:
                confirmation_side = None
                confirmation_count = 0
                confirmation_datetimes = []
            continue

        agreeing_strength_sum = _agreeing_strength_sum(
            row,
            settings.selected_indicators,
            candidate_side,
        )
        if agreeing_strength_sum < settings.strength_threshold:
            if not config.entry.preserve_cycle_on_no_signal:
                confirmation_side = None
                confirmation_count = 0
                confirmation_datetimes = []
            continue

        if confirmation_side == candidate_side:
            confirmation_count += 1
            confirmation_datetimes.append(_datetime_to_string(row[config.data.datetime_column]))
        elif confirmation_side is None or config.entry.reset_cycle_on_opposite_signal:
            confirmation_side = candidate_side
            confirmation_count = 1
            confirmation_datetimes = [_datetime_to_string(row[config.data.datetime_column])]

        if confirmation_count < settings.entry_confirmation_required:
            continue

        future_df = df.iloc[index + 1 : index + 1 + settings.max_holding_candles]
        if future_df.empty:
            unresolved_signal_candidates += 1
            confirmation_side = None
            confirmation_count = 0
            confirmation_datetimes = []
            continue

        entry_price = float(row[config.data.entry_price_column])
        close_outcome = _close_rule171_trade(
            future_df=future_df,
            side=candidate_side,
            entry_price=entry_price,
            take_profit_pips=settings.take_profit_pips,
            stop_loss_pips=settings.stop_loss_pips,
            pip_size=settings.pip_size,
            max_holding_candles=settings.max_holding_candles,
            timeframe=settings.timeframe,
        )
        blocked_until_index = index + int(close_outcome["close_candle_offset"])

        trades.append(
            _trade_row(
                signal_sequence=signal_sequence,
                row=row,
                config=config,
                settings=settings,
                signal_side=candidate_side,
                matched_release_pattern=_release_pattern(candidate_side, indicator_pattern),
                indicator_pattern=indicator_pattern,
                agreeing_strength_sum=agreeing_strength_sum,
                confirmation_count=confirmation_count,
                confirmation_datetimes=confirmation_datetimes,
                entry_price=entry_price,
                close_outcome=close_outcome,
                blocked_rows_while_open=blocked_rows_while_open,
            )
        )
        signal_sequence += 1
        confirmation_side = None
        confirmation_count = 0
        confirmation_datetimes = []

    return trades, unresolved_signal_candidates, blocked_rows_while_open


def _patterns_for_selection(
    config: Rule171Config,
    settings: _RuntimeSettings,
) -> tuple[set[str], set[str]]:
    expected = len(settings.selected_indicators)
    buy = {pattern for pattern in config.patterns.BUY if len(pattern.split("|")) == expected}
    sell = {pattern for pattern in config.patterns.SELL if len(pattern.split("|")) == expected}
    return buy, sell


def _indicator_pattern(row: pd.Series, indicators: list[str]) -> str:
    return "|".join(str(row[direction_column_for(indicator)]) for indicator in indicators)


def _candidate_side(pattern: str, buy_patterns: set[str], sell_patterns: set[str]) -> str | None:
    if pattern in buy_patterns:
        return "BUY"
    if pattern in sell_patterns:
        return "SELL"
    return None


def _agreeing_strength_sum(row: pd.Series, indicators: list[str], side: str) -> float:
    direction = IndicatorDirection.UP.value if side == "BUY" else IndicatorDirection.DOWN.value
    total = 0.0
    for indicator in indicators:
        if row[direction_column_for(indicator)] == direction:
            total += float(row[strength_column_for(indicator)])
    return total


def _close_rule171_trade(
    future_df: pd.DataFrame,
    side: str,
    entry_price: float,
    take_profit_pips: float,
    stop_loss_pips: float,
    pip_size: float,
    max_holding_candles: int,
    timeframe: str,
) -> dict[str, Any]:
    take_profit_price, stop_loss_price = _target_stop_prices(
        side,
        entry_price,
        take_profit_pips,
        stop_loss_pips,
        pip_size,
    )
    for offset, candle in enumerate(future_df.itertuples(index=False), start=1):
        high = float(getattr(candle, "High"))
        low = float(getattr(candle, "Low"))
        close_datetime = _datetime_to_string(getattr(candle, "DateTime"))
        if side == "BUY":
            tp_hit = high >= take_profit_price
            sl_hit = low <= stop_loss_price
        else:
            tp_hit = low <= take_profit_price
            sl_hit = high >= stop_loss_price

        if tp_hit and sl_hit:
            return _close_outcome(
                close_datetime=close_datetime,
                close_candle_offset=offset,
                close_price=stop_loss_price,
                close_price_source="Low" if side == "BUY" else "High",
                close_reason="BOTH_HIT_SAME_CANDLE",
                trade_result=LOSS_CLOSE,
                realized_pips=-stop_loss_pips,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                closed_at_take_profit=False,
                closed_at_stop_loss=True,
                closed_at_time_limit=False,
                closed_at_12_hours=False,
                closed_at_end_of_data=False,
                both_hit_same_candle=True,
            )
        if tp_hit:
            return _close_outcome(
                close_datetime=close_datetime,
                close_candle_offset=offset,
                close_price=take_profit_price,
                close_price_source="High" if side == "BUY" else "Low",
                close_reason="TAKE_PROFIT",
                trade_result=WIN_CLOSE,
                realized_pips=take_profit_pips,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                closed_at_take_profit=True,
                closed_at_stop_loss=False,
                closed_at_time_limit=False,
                closed_at_12_hours=False,
                closed_at_end_of_data=False,
                both_hit_same_candle=False,
            )
        if sl_hit:
            return _close_outcome(
                close_datetime=close_datetime,
                close_candle_offset=offset,
                close_price=stop_loss_price,
                close_price_source="Low" if side == "BUY" else "High",
                close_reason="STOP_LOSS",
                trade_result=LOSS_CLOSE,
                realized_pips=-stop_loss_pips,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                closed_at_take_profit=False,
                closed_at_stop_loss=True,
                closed_at_time_limit=False,
                closed_at_12_hours=False,
                closed_at_end_of_data=False,
                both_hit_same_candle=False,
            )

    last = future_df.iloc[-1]
    close_price = float(last["Close"])
    realized = _realized_pips(side, entry_price, close_price, pip_size)
    full_horizon = len(future_df) >= max_holding_candles
    close_reason = "TIME_LIMIT_CLOSE" if full_horizon else "END_OF_DATA_CLOSE"
    closed_at_12_hours = full_horizon and max_holding_candles == 144 and timeframe.upper() == "M5"
    return _close_outcome(
        close_datetime=_datetime_to_string(last["DateTime"]),
        close_candle_offset=len(future_df),
        close_price=close_price,
        close_price_source="Close",
        close_reason=close_reason,
        trade_result=WIN_CLOSE if realized > 0 else LOSS_CLOSE,
        realized_pips=realized,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        closed_at_take_profit=False,
        closed_at_stop_loss=False,
        closed_at_time_limit=full_horizon,
        closed_at_12_hours=closed_at_12_hours,
        closed_at_end_of_data=not full_horizon,
        both_hit_same_candle=False,
    )


def _target_stop_prices(
    side: str,
    entry_price: float,
    take_profit_pips: float,
    stop_loss_pips: float,
    pip_size: float,
) -> tuple[float, float]:
    if side == "BUY":
        return entry_price + take_profit_pips * pip_size, entry_price - stop_loss_pips * pip_size
    return entry_price - take_profit_pips * pip_size, entry_price + stop_loss_pips * pip_size


def _realized_pips(side: str, entry_price: float, close_price: float, pip_size: float) -> float:
    if side == "BUY":
        return (close_price - entry_price) / pip_size
    return (entry_price - close_price) / pip_size


def _close_outcome(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def _release_pattern(side: str, pattern: str) -> str:
    return f"{side}:{pattern}"


def _trade_row(
    signal_sequence: int,
    row: pd.Series,
    config: Rule171Config,
    settings: _RuntimeSettings,
    signal_side: str,
    matched_release_pattern: str,
    indicator_pattern: str,
    agreeing_strength_sum: float,
    confirmation_count: int,
    confirmation_datetimes: list[str],
    entry_price: float,
    close_outcome: dict[str, Any],
    blocked_rows_while_open: int,
) -> dict[str, Any]:
    trade = {
        "signal_sequence": signal_sequence,
        "pair": settings.pair,
        "timeframe": settings.timeframe,
        "rule_name": RULE171_NAME,
        "signal_datetime": _datetime_to_string(row[config.data.datetime_column]),
        "signal_side": signal_side,
        "selected_indicators": ",".join(settings.selected_indicators),
        "matched_release_pattern": matched_release_pattern,
        "indicator_pattern": indicator_pattern,
        "agreeing_strength_sum": agreeing_strength_sum,
        "strength_threshold": settings.strength_threshold,
        "strength_filter_passed": True,
        "entry_confirmation_count": confirmation_count,
        "entry_confirmation_required": settings.entry_confirmation_required,
        "entry_confirmation_1_datetime": confirmation_datetimes[0] if len(confirmation_datetimes) > 0 else None,
        "entry_confirmation_2_datetime": confirmation_datetimes[1] if len(confirmation_datetimes) > 1 else None,
        "entry_confirmation_3_datetime": confirmation_datetimes[2] if len(confirmation_datetimes) > 2 else None,
        "entry_price_column": config.data.entry_price_column,
        "entry_price": entry_price,
        "pip_size": settings.pip_size,
        "take_profit_pips": settings.take_profit_pips,
        "stop_loss_pips": settings.stop_loss_pips,
        "max_holding_candles": settings.max_holding_candles,
        "blocked_rows_while_open": blocked_rows_while_open,
        **close_outcome,
    }
    for indicator in settings.selected_indicators:
        trade[direction_column_for(indicator)] = row[direction_column_for(indicator)]
        trade[strength_column_for(indicator)] = float(row[strength_column_for(indicator)])
    return trade


def _summary(
    output_df: pd.DataFrame,
    config: Rule171Config,
    settings: _RuntimeSettings,
    input_path: Path,
    output_csv: Path,
    output_summary: Path,
    unresolved_signal_candidates: int,
    blocked_rows_while_open: int,
) -> dict[str, Any]:
    released = len(output_df)
    win_count = int((output_df["trade_result"] == WIN_CLOSE).sum()) if released else 0
    loss_count = int((output_df["trade_result"] == LOSS_CLOSE).sum()) if released else 0
    total_pips = float(output_df["realized_pips"].sum()) if released else 0.0
    return {
        "rule_name": config.rule_name,
        "production_activation_status": NOT_ACTIVE,
        "pair": settings.pair,
        "timeframe": settings.timeframe,
        "input_path": str(Path(input_path)),
        "output_csv_path": str(Path(output_csv)),
        "output_summary_path": str(Path(output_summary)),
        "selected_indicators": settings.selected_indicators,
        "indicator_order": settings.indicator_order,
        "start_datetime": settings.start,
        "end_datetime": settings.end,
        "pip_size": settings.pip_size,
        "strength_threshold": settings.strength_threshold,
        "entry_confirmation_required": settings.entry_confirmation_required,
        "take_profit_pips": settings.take_profit_pips,
        "stop_loss_pips": settings.stop_loss_pips,
        "max_holding_candles": settings.max_holding_candles,
        "released_signals": released,
        "buy_signals": int((output_df["signal_side"] == "BUY").sum()) if released else 0,
        "sell_signals": int((output_df["signal_side"] == "SELL").sum()) if released else 0,
        "win_close_count": win_count,
        "loss_close_count": loss_count,
        "take_profit_closes": int(output_df["closed_at_take_profit"].sum()) if released else 0,
        "stop_loss_closes": int(output_df["closed_at_stop_loss"].sum()) if released else 0,
        "time_limit_closes": int(output_df["closed_at_time_limit"].sum()) if released else 0,
        "twelve_hour_closes": int(output_df["closed_at_12_hours"].sum()) if released else 0,
        "twelve_hour_win_closes": int(((output_df["closed_at_12_hours"]) & (output_df["trade_result"] == WIN_CLOSE)).sum()) if released else 0,
        "twelve_hour_loss_closes": int(((output_df["closed_at_12_hours"]) & (output_df["trade_result"] == LOSS_CLOSE)).sum()) if released else 0,
        "end_of_data_closes": int(output_df["closed_at_end_of_data"].sum()) if released else 0,
        "both_hit_same_candle_count": int(output_df["both_hit_same_candle"].sum()) if released else 0,
        "total_realized_pips": total_pips,
        "average_pips_per_signal": total_pips / released if released else 0.0,
        "win_close_rate": win_count / released if released else 0.0,
        "loss_close_rate": loss_count / released if released else 0.0,
        "blocked_rows_while_open": blocked_rows_while_open,
        "unresolved_signal_candidates": unresolved_signal_candidates,
        "validation_status": "PASS",
        "live_trading_allowed": False,
        "broker_order_allowed": False,
        "generated_at_utc": generated_at_utc(),
    }


def _output_columns(selected_indicators: list[str]) -> list[str]:
    base_columns = [
        "signal_sequence",
        "pair",
        "timeframe",
        "rule_name",
        "signal_datetime",
        "signal_side",
        "selected_indicators",
        "matched_release_pattern",
        "indicator_pattern",
        "agreeing_strength_sum",
        "strength_threshold",
        "strength_filter_passed",
        "entry_confirmation_count",
        "entry_confirmation_required",
        "entry_confirmation_1_datetime",
        "entry_confirmation_2_datetime",
        "entry_confirmation_3_datetime",
        "entry_price_column",
        "entry_price",
        "pip_size",
        "take_profit_pips",
        "stop_loss_pips",
        "take_profit_price",
        "stop_loss_price",
        "max_holding_candles",
        "close_datetime",
        "close_candle_offset",
        "close_price",
        "close_price_source",
        "close_reason",
        "trade_result",
        "realized_pips",
        "closed_at_take_profit",
        "closed_at_stop_loss",
        "closed_at_time_limit",
        "closed_at_12_hours",
        "closed_at_end_of_data",
        "both_hit_same_candle",
        "blocked_rows_while_open",
    ]
    indicator_columns: list[str] = []
    for indicator in selected_indicators:
        indicator_columns.extend([direction_column_for(indicator), strength_column_for(indicator)])
    return [*base_columns, *indicator_columns]


def _datetime_to_string(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

from enum import Enum
from typing import Any

import pandas as pd

from gt_v1_engine.core.errors import DataValidationError
from gt_v1_engine.core.validation import require_columns


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class BacktestResult(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    NO_HIT = "NO_HIT"
    NO_FUTURE_DATA = "NO_FUTURE_DATA"


class CloseReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    BOTH_HIT_SAME_CANDLE = "BOTH_HIT_SAME_CANDLE"
    NO_HIT_WITHIN_HORIZON = "NO_HIT_WITHIN_HORIZON"
    NO_FUTURE_DATA = "NO_FUTURE_DATA"


def evaluate_future_candles(
    future_df: pd.DataFrame,
    side: str,
    entry_price: float,
    target_pips: float,
    stop_pips: float,
    pip_size: float,
) -> dict[str, Any]:
    normalized_side = _normalize_side(side)
    _validate_price_inputs(entry_price, target_pips, stop_pips, pip_size)
    require_columns(future_df, ["DateTime", "High", "Low"], "Future candle data")

    take_profit_price, stop_loss_price = _target_stop_prices(
        normalized_side,
        entry_price,
        target_pips,
        stop_pips,
        pip_size,
    )

    if future_df.empty:
        return _outcome(
            result=BacktestResult.NO_FUTURE_DATA.value,
            realized_pips=0.0,
            close_reason=CloseReason.NO_FUTURE_DATA.value,
            close_datetime=None,
            close_candle_offset=None,
            close_price=None,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            both_hit_same_candle=False,
        )

    for offset, row in enumerate(future_df.itertuples(index=False), start=1):
        high = float(getattr(row, "High"))
        low = float(getattr(row, "Low"))
        close_datetime = getattr(row, "DateTime")

        if normalized_side == TradeSide.BUY.value:
            tp_hit = high >= take_profit_price
            sl_hit = low <= stop_loss_price
        else:
            tp_hit = low <= take_profit_price
            sl_hit = high >= stop_loss_price

        if tp_hit and sl_hit:
            return _outcome(
                result=BacktestResult.LOSS.value,
                realized_pips=-float(stop_pips),
                close_reason=CloseReason.BOTH_HIT_SAME_CANDLE.value,
                close_datetime=_datetime_to_string(close_datetime),
                close_candle_offset=offset,
                close_price=stop_loss_price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                both_hit_same_candle=True,
            )
        if tp_hit:
            return _outcome(
                result=BacktestResult.WIN.value,
                realized_pips=float(target_pips),
                close_reason=CloseReason.TAKE_PROFIT.value,
                close_datetime=_datetime_to_string(close_datetime),
                close_candle_offset=offset,
                close_price=take_profit_price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                both_hit_same_candle=False,
            )
        if sl_hit:
            return _outcome(
                result=BacktestResult.LOSS.value,
                realized_pips=-float(stop_pips),
                close_reason=CloseReason.STOP_LOSS.value,
                close_datetime=_datetime_to_string(close_datetime),
                close_candle_offset=offset,
                close_price=stop_loss_price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                both_hit_same_candle=False,
            )

    return _outcome(
        result=BacktestResult.NO_HIT.value,
        realized_pips=0.0,
        close_reason=CloseReason.NO_HIT_WITHIN_HORIZON.value,
        close_datetime=None,
        close_candle_offset=None,
        close_price=None,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        both_hit_same_candle=False,
    )


def _normalize_side(side: str) -> str:
    normalized = side.strip().upper() if isinstance(side, str) else ""
    if normalized not in {TradeSide.BUY.value, TradeSide.SELL.value}:
        raise DataValidationError(f"Invalid trade side: {side}")
    return normalized


def _validate_price_inputs(
    entry_price: float,
    target_pips: float,
    stop_pips: float,
    pip_size: float,
) -> None:
    if target_pips <= 0:
        raise DataValidationError("target_pips must be greater than 0")
    if stop_pips <= 0:
        raise DataValidationError("stop_pips must be greater than 0")
    if pip_size <= 0:
        raise DataValidationError("pip_size must be greater than 0")
    float(entry_price)


def _target_stop_prices(
    side: str,
    entry_price: float,
    target_pips: float,
    stop_pips: float,
    pip_size: float,
) -> tuple[float, float]:
    target_distance = target_pips * pip_size
    stop_distance = stop_pips * pip_size
    if side == TradeSide.BUY.value:
        return entry_price + target_distance, entry_price - stop_distance
    return entry_price - target_distance, entry_price + stop_distance


def _outcome(
    result: str,
    realized_pips: float,
    close_reason: str,
    close_datetime: str | None,
    close_candle_offset: int | None,
    close_price: float | None,
    take_profit_price: float,
    stop_loss_price: float,
    both_hit_same_candle: bool,
) -> dict[str, Any]:
    return {
        "result": result,
        "realized_pips": realized_pips,
        "close_reason": close_reason,
        "close_datetime": close_datetime,
        "close_candle_offset": close_candle_offset,
        "close_price": close_price,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "both_hit_same_candle": both_hit_same_candle,
    }


def _datetime_to_string(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

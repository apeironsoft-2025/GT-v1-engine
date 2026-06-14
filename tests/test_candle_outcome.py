import pandas as pd

from gt_v1_engine.backtesting.candle_outcome import (
    BacktestResult,
    CloseReason,
    TradeSide,
    evaluate_future_candles,
)


def _future(*candles: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DateTime": pd.date_range("2025-01-01", periods=len(candles), freq="5min", tz="UTC"),
            "High": [high for high, _ in candles],
            "Low": [low for _, low in candles],
        }
    )


def test_buy_take_profit_hit_by_high_is_win() -> None:
    outcome = evaluate_future_candles(_future((160.31, 159.95)), TradeSide.BUY.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.WIN.value
    assert outcome["close_reason"] == CloseReason.TAKE_PROFIT.value
    assert outcome["realized_pips"] == 30


def test_buy_stop_loss_hit_by_low_is_loss() -> None:
    outcome = evaluate_future_candles(_future((160.10, 159.59)), TradeSide.BUY.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.LOSS.value
    assert outcome["close_reason"] == CloseReason.STOP_LOSS.value


def test_buy_both_hit_same_candle_is_loss() -> None:
    outcome = evaluate_future_candles(_future((160.31, 159.59)), TradeSide.BUY.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.LOSS.value
    assert outcome["close_reason"] == CloseReason.BOTH_HIT_SAME_CANDLE.value
    assert outcome["both_hit_same_candle"] is True
    assert outcome["realized_pips"] == -40


def test_sell_take_profit_hit_by_low_is_win() -> None:
    outcome = evaluate_future_candles(_future((160.05, 159.69)), TradeSide.SELL.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.WIN.value
    assert outcome["close_reason"] == CloseReason.TAKE_PROFIT.value


def test_sell_stop_loss_hit_by_high_is_loss() -> None:
    outcome = evaluate_future_candles(_future((160.41, 159.90)), TradeSide.SELL.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.LOSS.value
    assert outcome["close_reason"] == CloseReason.STOP_LOSS.value


def test_sell_both_hit_same_candle_is_loss() -> None:
    outcome = evaluate_future_candles(_future((160.41, 159.69)), TradeSide.SELL.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.LOSS.value
    assert outcome["close_reason"] == CloseReason.BOTH_HIT_SAME_CANDLE.value


def test_no_hit_when_neither_touched() -> None:
    outcome = evaluate_future_candles(_future((160.10, 159.95), (160.20, 159.90)), TradeSide.BUY.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.NO_HIT.value
    assert outcome["close_reason"] == CloseReason.NO_HIT_WITHIN_HORIZON.value


def test_no_future_data_when_future_df_empty() -> None:
    outcome = evaluate_future_candles(_future(), TradeSide.BUY.value, 160, 30, 40, 0.01)
    assert outcome["result"] == BacktestResult.NO_FUTURE_DATA.value
    assert outcome["close_reason"] == CloseReason.NO_FUTURE_DATA.value


def test_close_candle_offset_is_correct() -> None:
    outcome = evaluate_future_candles(_future((160.10, 159.95), (160.31, 159.95)), TradeSide.BUY.value, 160, 30, 40, 0.01)
    assert outcome["close_candle_offset"] == 2

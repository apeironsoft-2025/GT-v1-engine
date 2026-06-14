import json
from pathlib import Path

import pytest

from gt_v1_engine.backtesting.indicator_backtester import (
    backtest_all_indicators,
    backtest_indicator,
)
from gt_v1_engine.core.errors import DataValidationError, IndicatorRegistryError


def _indicator_ready_csv(tmp_path: Path, include_ts: bool = True) -> Path:
    rows = []
    directions = ["UP", "DOWN", "NO_SIGNAL", "UP", "DOWN", "UP"]
    for index, direction in enumerate(directions):
        close = 160.0 + index * 0.05
        rows.append(
            {
                "DateTime": f"2025-01-01T00:{index * 5:02d}:00+00:00",
                "Open": close,
                "High": close + 0.08,
                "Low": close - 0.08,
                "Close": close,
                "SRP": close,
                "MACD_TD": direction,
                "RSI_TD": direction,
                "MACD_TS": 0.5,
                "RSI_TS": 0.5,
            }
        )
    path = tmp_path / "indicator_ready.csv"
    import pandas as pd

    df = pd.DataFrame(rows)
    if not include_ts:
        df = df.drop(columns=["MACD_TS"])
    df.to_csv(path, index=False)
    return path


def test_backtest_indicator_creates_csv_and_buy_sell_rows(tmp_path: Path) -> None:
    output_csv = tmp_path / "MACD_backtest.csv"
    result_df, summary = backtest_indicator(
        input_path=_indicator_ready_csv(tmp_path),
        pair="USDJPY",
        timeframe="M5",
        indicator="MACD",
        start=None,
        end=None,
        horizon_candles=2,
        target_pips=5,
        stop_pips=8,
        pip_size=0.01,
        output_csv=output_csv,
    )
    assert output_csv.exists()
    assert {"BUY", "SELL"}.issubset(set(result_df["signal_side"]))
    assert "NO_SIGNAL" not in set(result_df["indicator_td"])
    assert summary["validation_status"] == "PASS"
    assert summary["signal_rows"] == 5
    assert summary["no_signal_rows"] == 1
    assert summary["tested_trades"] == 5


def test_include_no_signal_rows_includes_no_signal(tmp_path: Path) -> None:
    result_df, summary = backtest_indicator(
        input_path=_indicator_ready_csv(tmp_path),
        pair="USDJPY",
        timeframe="M5",
        indicator="MACD",
        start=None,
        end=None,
        horizon_candles=2,
        target_pips=5,
        stop_pips=8,
        pip_size=0.01,
        output_csv=tmp_path / "MACD_backtest.csv",
        include_no_signal_rows=True,
    )
    assert "NO_SIGNAL" in set(result_df["indicator_td"])
    assert summary["tested_trades"] == 5


def test_unknown_indicator_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(IndicatorRegistryError):
        backtest_indicator(
            input_path=_indicator_ready_csv(tmp_path),
            pair="USDJPY",
            timeframe="M5",
            indicator="UNKNOWN",
            start=None,
            end=None,
            horizon_candles=2,
            target_pips=5,
            stop_pips=8,
            pip_size=0.01,
            output_csv=tmp_path / "out.csv",
        )


def test_missing_td_ts_column_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        backtest_indicator(
            input_path=_indicator_ready_csv(tmp_path, include_ts=False),
            pair="USDJPY",
            timeframe="M5",
            indicator="MACD",
            start=None,
            end=None,
            horizon_candles=2,
            target_pips=5,
            stop_pips=8,
            pip_size=0.01,
            output_csv=tmp_path / "out.csv",
        )


def test_invalid_horizon_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        backtest_indicator(
            input_path=_indicator_ready_csv(tmp_path),
            pair="USDJPY",
            timeframe="M5",
            indicator="MACD",
            start=None,
            end=None,
            horizon_candles=0,
            target_pips=5,
            stop_pips=8,
            pip_size=0.01,
            output_csv=tmp_path / "out.csv",
        )


def test_backtest_all_indicators_creates_csvs_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "backtests"
    summary_json = tmp_path / "summary.json"
    summary = backtest_all_indicators(
        input_path=_indicator_ready_csv(tmp_path),
        pair="USDJPY",
        timeframe="M5",
        indicators=["MACD", "RSI"],
        start=None,
        end=None,
        horizon_candles=2,
        target_pips=5,
        stop_pips=8,
        pip_size=0.01,
        output_dir=output_dir,
        summary_json=summary_json,
    )
    assert (output_dir / "USDJPY_M5_MACD_backtest.csv").exists()
    assert (output_dir / "USDJPY_M5_RSI_backtest.csv").exists()
    assert summary_json.exists()
    saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved_summary["validation_status"] == "PASS"
    assert summary["selected_indicators"] == ["MACD", "RSI"]

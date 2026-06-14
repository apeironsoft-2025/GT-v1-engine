from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gt_v1_engine.core.errors import FileMissingError, IndicatorRegistryError
from gt_v1_engine.pipeline.smoke_pipeline import run_smoke_pipeline

INDICATORS = ["MACD", "RSI", "ADX", "ATR", "BOLLINGER", "EMA_STACK"]


def _raw_csv(tmp_path: Path, rows: int = 180) -> Path:
    index = np.arange(rows)
    close = 160.0 + np.linspace(0, 2.0, rows) + 0.08 * np.sin(index / 4)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    df = pd.DataFrame(
        {
            "DateTime": pd.date_range("2025-12-01", periods=rows, freq="5min", tz="UTC"),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "SRP": close,
        }
    )
    path = tmp_path / "USDJPY_M5.csv"
    df.to_csv(path, index=False)
    return path


def _run_pipeline(tmp_path: Path, indicators: list[str] | None = None) -> dict:
    return run_smoke_pipeline(
        input_path=_raw_csv(tmp_path),
        pair="USDJPY",
        timeframe="M5",
        start="2025-12-01T00:00:00+00:00",
        end="2025-12-01T12:00:00+00:00",
        indicators=indicators or INDICATORS,
        config_path=Path("configs/rules/rule171.yaml"),
        output_root=tmp_path / "out",
        horizon_candles=48,
        indicator_target_pips=30,
        indicator_stop_pips=40,
        rule171_pip_size=0.01,
        rule171_strength_threshold=4.5,
        rule171_entry_confirmation_required=3,
        rule171_take_profit_pips=30,
        rule171_stop_loss_pips=40,
        rule171_max_holding_candles=48,
    )


def test_run_smoke_pipeline_creates_all_outputs(tmp_path: Path) -> None:
    summary = _run_pipeline(tmp_path)
    assert Path(summary["indicator_dataset_csv"]).exists()
    assert Path(summary["indicator_dataset_parquet"]).exists()
    assert Path(summary["indicator_summary_json"]).exists()
    assert Path(summary["indicator_backtest_summary_json"]).exists()
    assert Path(summary["rule171_output_csv"]).exists()
    assert Path(summary["rule171_summary_json"]).exists()
    assert Path(summary["markdown_report_path"]).exists()
    assert summary["validation_status"] == "PASS"
    assert summary["production_activation_status"] == "NOT_ACTIVE"
    assert summary["live_trading_allowed"] is False
    assert summary["broker_order_allowed"] is False


def test_run_smoke_pipeline_invalid_indicator_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(IndicatorRegistryError):
        _run_pipeline(tmp_path, indicators=["MACD", "UNKNOWN"])


def test_run_smoke_pipeline_missing_input_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(FileMissingError):
        run_smoke_pipeline(
            input_path=tmp_path / "missing.csv",
            pair="USDJPY",
            timeframe="M5",
            start="2025-12-01T00:00:00+00:00",
            end="2025-12-01T12:00:00+00:00",
            indicators=INDICATORS,
            config_path=Path("configs/rules/rule171.yaml"),
            output_root=tmp_path / "out",
            horizon_candles=48,
            indicator_target_pips=30,
            indicator_stop_pips=40,
            rule171_pip_size=0.01,
            rule171_strength_threshold=4.5,
            rule171_entry_confirmation_required=3,
            rule171_take_profit_pips=30,
            rule171_stop_loss_pips=40,
            rule171_max_holding_candles=48,
        )

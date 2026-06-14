import json
from pathlib import Path

import pytest

from gt_v1_engine.core.errors import DataValidationError, FileMissingError, IndicatorRegistryError
from gt_v1_engine.indicators.executor import run_indicator_executor
from gt_v1_engine.indicators.registry import get_default_indicator_order
from tests.indicator_test_helpers import VALID_DIRECTIONS, VALID_STRENGTHS, make_ohlc_dataframe


def _write_market_csv(tmp_path: Path, rows: int = 180) -> Path:
    path = tmp_path / "USDJPY_M5.csv"
    make_ohlc_dataframe(rows=rows).to_csv(path, index=False)
    return path


def _assert_td_ts_contract(output_columns: list[str], df_columns: list[str]) -> None:
    for indicator in output_columns:
        assert f"{indicator}_TD" in df_columns
        assert f"{indicator}_TS" in df_columns


def test_executor_runs_all_six_indicators(tmp_path: Path) -> None:
    input_path = _write_market_csv(tmp_path)
    output_csv = tmp_path / "USDJPY_M5_6I.csv"
    output_parquet = tmp_path / "USDJPY_M5_6I.parquet"
    summary_json = tmp_path / "USDJPY_M5_6I_summary.json"

    df, summary = run_indicator_executor(
        input_path=input_path,
        pair="USDJPY",
        timeframe="M5",
        indicators=get_default_indicator_order(),
        output_csv=output_csv,
        output_parquet=output_parquet,
        summary_json=summary_json,
    )

    _assert_td_ts_contract(get_default_indicator_order(), list(df.columns))
    assert summary["validation_status"] == "PASS"
    assert summary["row_count"] == len(make_ohlc_dataframe(rows=180))
    assert output_csv.exists()
    assert output_parquet.exists()
    assert summary_json.exists()

    saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved_summary["validation_status"] == "PASS"
    assert saved_summary["production_activation_status"] == "NOT_ACTIVE"
    assert saved_summary["broker_order_allowed"] is False
    assert saved_summary["live_trading_allowed"] is False

    for indicator in get_default_indicator_order():
        assert set(df[f"{indicator}_TD"].unique()).issubset(VALID_DIRECTIONS)
        assert set(df[f"{indicator}_TS"].unique()).issubset(VALID_STRENGTHS)


def test_executor_selected_indicator_subset(tmp_path: Path) -> None:
    input_path = _write_market_csv(tmp_path)
    output_csv = tmp_path / "USDJPY_M5_MACD_RSI.csv"

    df, summary = run_indicator_executor(
        input_path=input_path,
        pair="USDJPY",
        timeframe="M5",
        indicators=["MACD", "RSI"],
        output_csv=output_csv,
    )

    assert summary["selected_indicators"] == ["MACD", "RSI"]
    _assert_td_ts_contract(["MACD", "RSI"], list(df.columns))
    assert "ADX_TD" not in df.columns
    assert "ADX_TS" not in df.columns
    assert output_csv.exists()


def test_executor_unknown_indicator_raises_clean_error(tmp_path: Path) -> None:
    input_path = _write_market_csv(tmp_path)
    with pytest.raises(IndicatorRegistryError):
        run_indicator_executor(
            input_path=input_path,
            pair="USDJPY",
            timeframe="M5",
            indicators=["MACD", "UNKNOWN"],
            output_csv=tmp_path / "out.csv",
        )


def test_executor_missing_input_file_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(FileMissingError):
        run_indicator_executor(
            input_path=tmp_path / "missing.csv",
            pair="USDJPY",
            timeframe="M5",
            indicators=["MACD"],
            output_csv=tmp_path / "out.csv",
        )


def test_executor_rejects_empty_pair_or_timeframe(tmp_path: Path) -> None:
    input_path = _write_market_csv(tmp_path)
    with pytest.raises(DataValidationError, match="pair cannot be empty"):
        run_indicator_executor(
            input_path=input_path,
            pair="",
            timeframe="M5",
            indicators=["MACD"],
            output_csv=tmp_path / "out.csv",
        )

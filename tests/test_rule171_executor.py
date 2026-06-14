from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gt_v1_engine.core.errors import DataValidationError, FileMissingError, RuleConfigError
from gt_v1_engine.rules.rule171 import Rule171ExecutionOverrides, run_rule171_backtest

INDICATORS = ["MACD", "RSI", "ADX", "ATR", "BOLLINGER", "EMA_STACK"]
REQUIRED_COLUMNS = [
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


def _write_indicator_csv(
    tmp_path: Path,
    directions: list[str],
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    strength: float = 1.0,
    drop_column: str | None = None,
) -> Path:
    rows = []
    closes = closes or [160.0 + index * 0.05 for index in range(len(directions))]
    highs = highs or [close + 0.05 for close in closes]
    lows = lows or [close - 0.05 for close in closes]
    for index, direction in enumerate(directions):
        row = {
            "DateTime": (pd.Timestamp("2025-12-01T00:00:00+00:00") + pd.Timedelta(minutes=5 * index)).isoformat(),
            "Open": closes[index],
            "High": highs[index],
            "Low": lows[index],
            "Close": closes[index],
            "SRP": closes[index],
        }
        for indicator in INDICATORS:
            row[f"{indicator}_TD"] = direction
            row[f"{indicator}_TS"] = strength
        rows.append(row)
    df = pd.DataFrame(rows)
    if drop_column:
        df = df.drop(columns=[drop_column])
    path = tmp_path / "rule171_input.csv"
    df.to_csv(path, index=False)
    return path


def _run(
    tmp_path: Path,
    input_path: Path,
    **override_kwargs,
) -> tuple[pd.DataFrame, dict]:
    return run_rule171_backtest(
        input_path=input_path,
        config_path=Path("configs/rules/rule171.yaml"),
        output_csv=tmp_path / "rule171.csv",
        output_summary=tmp_path / "rule171_summary.json",
        overrides=Rule171ExecutionOverrides(
            pair="USDJPY",
            timeframe="M5",
            start="2025-12-01T00:00:00+00:00",
            end="2025-12-02T00:00:00+00:00",
            pip_size=0.01,
            strength_threshold=override_kwargs.pop("strength_threshold", 4.5),
            entry_confirmation_required=override_kwargs.pop("entry_confirmation_required", 3),
            take_profit_pips=override_kwargs.pop("take_profit_pips", 5),
            stop_loss_pips=override_kwargs.pop("stop_loss_pips", 8),
            max_holding_candles=override_kwargs.pop("max_holding_candles", 2),
            **override_kwargs,
        ),
    )


def test_three_same_side_buy_signals_release_one_trade(tmp_path: Path) -> None:
    output, summary = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"]))
    assert len(output) == 1
    assert output.iloc[0]["signal_side"] == "BUY"
    assert summary["released_signals"] == 1


def test_two_same_side_signals_do_not_release_trade(tmp_path: Path) -> None:
    output, summary = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP"]))
    assert output.empty
    assert summary["released_signals"] == 0


def test_no_signal_preserves_confirmation_cycle(tmp_path: Path) -> None:
    output, _ = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "NO_SIGNAL", "UP", "NO_SIGNAL"]))
    assert len(output) == 1
    assert output.iloc[0]["entry_confirmation_1_datetime"].endswith("00:00:00+00:00")
    assert output.iloc[0]["entry_confirmation_3_datetime"].endswith("00:15:00+00:00")


def test_opposite_accepted_signal_resets_confirmation_cycle(tmp_path: Path) -> None:
    output, _ = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "DOWN", "UP", "UP", "UP", "NO_SIGNAL"]))
    assert len(output) == 1
    assert output.iloc[0]["signal_side"] == "BUY"
    assert output.iloc[0]["entry_confirmation_1_datetime"].endswith("00:15:00+00:00")


def test_strength_below_threshold_does_not_count_as_accepted_signal(tmp_path: Path) -> None:
    output, summary = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "UP"], strength=0.25),
    )
    assert output.empty
    assert summary["released_signals"] == 0


def test_buy_tp_hit_by_high_gives_win_close(tmp_path: Path) -> None:
    output, _ = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"]))
    assert output.iloc[0]["trade_result"] == "WIN_CLOSE"
    assert output.iloc[0]["close_reason"] == "TAKE_PROFIT"


def test_buy_sl_hit_by_low_gives_loss_close(tmp_path: Path) -> None:
    closes = [160.0, 160.05, 160.10, 160.0]
    lows = [159.95, 160.0, 160.05, 160.01]
    lows[3] = 160.01
    # Entry is 160.10, stop is 160.02.
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"], closes=closes, lows=lows),
    )
    assert output.iloc[0]["trade_result"] == "LOSS_CLOSE"
    assert output.iloc[0]["close_reason"] == "STOP_LOSS"


def test_sell_tp_hit_by_low_gives_win_close(tmp_path: Path) -> None:
    closes = [160.2, 160.15, 160.10, 160.0]
    lows = [160.15, 160.10, 160.05, 160.04]
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["DOWN", "DOWN", "DOWN", "NO_SIGNAL"], closes=closes, lows=lows),
    )
    assert output.iloc[0]["signal_side"] == "SELL"
    assert output.iloc[0]["trade_result"] == "WIN_CLOSE"


def test_sell_sl_hit_by_high_gives_loss_close(tmp_path: Path) -> None:
    closes = [160.2, 160.15, 160.10, 160.2]
    highs = [160.25, 160.20, 160.15, 160.19]
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["DOWN", "DOWN", "DOWN", "NO_SIGNAL"], closes=closes, highs=highs),
    )
    assert output.iloc[0]["trade_result"] == "LOSS_CLOSE"
    assert output.iloc[0]["close_reason"] == "STOP_LOSS"


def test_both_hit_same_candle_gives_loss_close(tmp_path: Path) -> None:
    closes = [160.0, 160.05, 160.10, 160.10]
    highs = [160.05, 160.10, 160.15, 160.16]
    lows = [159.95, 160.0, 160.05, 160.01]
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"], closes=closes, highs=highs, lows=lows),
    )
    assert output.iloc[0]["trade_result"] == "LOSS_CLOSE"
    assert output.iloc[0]["close_reason"] == "BOTH_HIT_SAME_CANDLE"
    assert bool(output.iloc[0]["both_hit_same_candle"]) is True


def test_time_limit_close_gives_win_when_pips_positive(tmp_path: Path) -> None:
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL", "NO_SIGNAL"], closes=[160.0, 160.01, 160.02, 160.03, 160.04]),
        take_profit_pips=30,
        stop_loss_pips=40,
    )
    assert output.iloc[0]["close_reason"] == "TIME_LIMIT_CLOSE"
    assert output.iloc[0]["trade_result"] == "WIN_CLOSE"


def test_time_limit_close_gives_loss_when_pips_nonpositive(tmp_path: Path) -> None:
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL", "NO_SIGNAL"], closes=[160.0, 160.01, 160.02, 160.01, 160.0]),
        take_profit_pips=30,
        stop_loss_pips=40,
    )
    assert output.iloc[0]["close_reason"] == "TIME_LIMIT_CLOSE"
    assert output.iloc[0]["trade_result"] == "LOSS_CLOSE"


def test_rows_while_trade_open_are_counted_as_blocked(tmp_path: Path) -> None:
    _, summary = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "UP", "UP", "UP"], closes=[160.0, 160.01, 160.02, 160.03, 160.04, 160.05]),
        take_profit_pips=30,
        stop_loss_pips=40,
    )
    assert summary["blocked_rows_while_open"] == 2


def test_output_csv_contains_required_columns(tmp_path: Path) -> None:
    output, _ = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"]))
    for column in REQUIRED_COLUMNS:
        assert column in output.columns
    for indicator in INDICATORS:
        assert f"{indicator}_TD" in output.columns
        assert f"{indicator}_TS" in output.columns


def test_summary_json_contains_required_metrics(tmp_path: Path) -> None:
    _, summary = _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"]))
    assert summary["validation_status"] == "PASS"
    assert summary["production_activation_status"] == "NOT_ACTIVE"
    assert summary["broker_order_allowed"] is False
    assert (tmp_path / "rule171_summary.json").exists()


def test_override_strength_threshold_changes_behavior(tmp_path: Path) -> None:
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "UP", "NO_SIGNAL"], strength=0.25),
        strength_threshold=1.0,
    )
    assert len(output) == 1


def test_override_confirmation_count_changes_behavior(tmp_path: Path) -> None:
    output, _ = _run(
        tmp_path,
        _write_indicator_csv(tmp_path, ["UP", "UP", "NO_SIGNAL"]),
        entry_confirmation_required=2,
    )
    assert len(output) == 1


def test_missing_td_ts_column_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        _run(tmp_path, _write_indicator_csv(tmp_path, ["UP", "UP", "UP"], drop_column="MACD_TS"))


def test_missing_config_file_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(FileMissingError):
        run_rule171_backtest(
            input_path=_write_indicator_csv(tmp_path, ["UP", "UP", "UP"]),
            config_path=tmp_path / "missing.yaml",
            output_csv=tmp_path / "out.csv",
            output_summary=tmp_path / "summary.json",
        )


def test_unsafe_config_flags_fail(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/rules/rule171.yaml").read_text(encoding="utf-8"))
    unsafe = deepcopy(payload)
    unsafe["safety"]["broker_order_allowed"] = True
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(yaml.safe_dump(unsafe), encoding="utf-8")
    with pytest.raises(RuleConfigError):
        run_rule171_backtest(
            input_path=_write_indicator_csv(tmp_path, ["UP", "UP", "UP"]),
            config_path=config_path,
            output_csv=tmp_path / "out.csv",
            output_summary=tmp_path / "summary.json",
        )

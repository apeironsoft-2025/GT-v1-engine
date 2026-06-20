import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_indicator_backtest_rule_01.py"
ACTIVE_CALLS = {
    "BUY_CALL",
    "SELL_CALL",
    "HOLD_BUY_CALL",
    "HOLD_SELL_CALL",
    "CLOSE_BUY_CALL",
    "CLOSE_SELL_CALL",
}
ACTIVE_FIELDS = [
    "call_id",
    "entry_price",
    "entry_side",
    "highest_price_after_call",
    "lowest_price_after_call",
    "max_win_side_pips",
    "max_loss_side_pips",
    "candle_count_from_call",
    "is_open_call",
]


def load_script():
    spec = importlib.util.spec_from_file_location("run_indicator_backtest_rule_01", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def indicator_df(
    directions: list[str],
    closes: list[float] | None = None,
    include_datetime: bool = True,
) -> pd.DataFrame:
    closes = closes or [100.0 + index for index in range(len(directions))]
    rows = []
    for index, (direction, close) in enumerate(zip(directions, closes)):
        row = {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.25,
            "Close": close,
            "EMA_STACK_TD": direction,
            "EMA_STACK_TS": 0.5,
            "source_marker": f"row-{index}",
        }
        if include_datetime:
            row["DateTime"] = f"2026-01-01 00:{index:02d}:00"
        rows.append(row)
    return pd.DataFrame(rows)


def test_builds_input_path_from_root_and_file_name() -> None:
    script = load_script()

    path = script.build_input_path(
        r"F:\GT-v1-shared-storage\indicators",
        "USDJPY_M5_ema_stack_td_ts.csv",
    )

    assert str(path) == r"F:\GT-v1-shared-storage\indicators\USDJPY_M5_ema_stack_td_ts.csv"


def test_detects_single_td_column() -> None:
    script = load_script()

    td_column, indicator_name = script.detect_td_column(indicator_df(["UP"]))

    assert td_column == "EMA_STACK_TD"
    assert indicator_name == "EMA_STACK"


def test_missing_td_column_fails_safely() -> None:
    script = load_script()
    df = indicator_df(["UP"]).drop(columns=["EMA_STACK_TD"])

    with pytest.raises(ValueError, match="No _TD column found"):
        script.run_backtest(df)


def test_multiple_td_columns_fail_safely() -> None:
    script = load_script()
    df = indicator_df(["UP"])
    df["MACD_TD"] = "UP"

    with pytest.raises(ValueError, match="Multiple _TD columns found"):
        script.run_backtest(df)


def test_buy_call_opens_on_up_and_preserves_original_columns() -> None:
    script = load_script()

    output, summary = script.run_backtest(indicator_df(["UP"]), pip_size=0.01)

    assert output.iloc[0]["CALL"] == "BUY_CALL"
    assert output.iloc[0]["entry_side"] == "BUY_CALL"
    assert output.iloc[0]["highest_price_after_call"] == 100.0
    assert output.iloc[0]["lowest_price_after_call"] == 100.0
    assert output.iloc[0]["max_win_side_pips"] == 0.0
    assert output.iloc[0]["max_loss_side_pips"] == 0.0
    assert output.iloc[0]["candle_count_from_call"] == 1
    assert "EMA_STACK_TS" in output.columns
    assert "source_marker" in output.columns
    assert summary["buy_call_count"] == 1


def test_sell_call_opens_on_down() -> None:
    script = load_script()

    output, summary = script.run_backtest(indicator_df(["DOWN"]), pip_size=0.01)

    assert output.iloc[0]["CALL"] == "SELL_CALL"
    assert output.iloc[0]["entry_side"] == "SELL_CALL"
    assert output.iloc[0]["candle_count_from_call"] == 1
    assert summary["sell_call_count"] == 1


def test_buy_call_holds_while_td_remains_up() -> None:
    script = load_script()

    output, _ = script.run_backtest(indicator_df(["UP", "UP"]), pip_size=0.01)

    assert output["CALL"].tolist() == ["BUY_CALL", "HOLD_BUY_CALL"]
    assert output.iloc[1]["call_id"] == 1
    assert output["candle_count_from_call"].tolist() == [1, 2]


def test_sell_call_holds_while_td_remains_down() -> None:
    script = load_script()

    output, _ = script.run_backtest(indicator_df(["DOWN", "DOWN"]), pip_size=0.01)

    assert output["CALL"].tolist() == ["SELL_CALL", "HOLD_SELL_CALL"]
    assert output.iloc[1]["call_id"] == 1
    assert output["candle_count_from_call"].tolist() == [1, 2]


def test_buy_call_closes_on_down() -> None:
    script = load_script()

    output, summary = script.run_backtest(
        indicator_df(["UP", "DOWN"], [100.0, 102.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == ["BUY_CALL", "CLOSE_BUY_CALL"]
    assert output.iloc[1]["close_reason"] == "TD_REVERSAL"
    assert output.iloc[1]["realized_pips"] == 200.0
    assert output.iloc[1]["candle_count_from_call"] == 2
    assert summary["win_count"] == 1


def test_buy_call_closes_on_no_direction() -> None:
    script = load_script()

    output, summary = script.run_backtest(
        indicator_df(["UP", "NO_SIGNAL"], [100.0, 99.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == ["BUY_CALL", "CLOSE_BUY_CALL"]
    assert output.iloc[1]["close_reason"] == "NO_DIRECTION_CLOSE"
    assert output.iloc[1]["realized_pips"] == -100.0
    assert summary["loss_count"] == 1


def test_sell_call_closes_on_up() -> None:
    script = load_script()

    output, summary = script.run_backtest(
        indicator_df(["DOWN", "UP"], [100.0, 98.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == ["SELL_CALL", "CLOSE_SELL_CALL"]
    assert output.iloc[1]["close_reason"] == "TD_REVERSAL"
    assert output.iloc[1]["realized_pips"] == 200.0
    assert output.iloc[1]["candle_count_from_call"] == 2
    assert summary["win_count"] == 1


def test_sell_call_closes_on_no_direction() -> None:
    script = load_script()

    output, summary = script.run_backtest(
        indicator_df(["DOWN", "UNKNOWN"], [100.0, 101.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == ["SELL_CALL", "CLOSE_SELL_CALL"]
    assert output.iloc[1]["close_reason"] == "NO_DIRECTION_CLOSE"
    assert output.iloc[1]["realized_pips"] == -100.0
    assert summary["loss_count"] == 1


def test_one_candle_after_close_is_interval_skip() -> None:
    script = load_script()

    output, summary = script.run_backtest(
        indicator_df(["UP", "DOWN", "UP"], [100.0, 101.0, 102.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == ["BUY_CALL", "CLOSE_BUY_CALL", "INTERVAL_SKIP"]
    assert bool(output.iloc[2]["interval_skip"])
    assert pd.isna(output.iloc[2]["candle_count_from_call"])
    assert summary["interval_skip_count"] == 1


def test_after_interval_skip_no_direction_rows_become_no_direction_skip() -> None:
    script = load_script()

    output, summary = script.run_backtest(
        indicator_df(["UP", "DOWN", "", "NO_SIGNAL", "DOWN"], [100, 101, 102, 103, 104]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == [
        "BUY_CALL",
        "CLOSE_BUY_CALL",
        "INTERVAL_SKIP",
        "NO_DIRECTION_SKIP",
        "SELL_CALL",
    ]
    assert pd.isna(output.iloc[2]["candle_count_from_call"])
    assert pd.isna(output.iloc[3]["candle_count_from_call"])
    assert summary["no_direction_skip_count"] == 1


def test_no_direction_has_empty_candle_count() -> None:
    script = load_script()

    output, _ = script.run_backtest(indicator_df(["NO_SIGNAL"]), pip_size=0.01)

    assert output.iloc[0]["CALL"] == "NO_DIRECTION"
    assert pd.isna(output.iloc[0]["candle_count_from_call"])


def test_candle_count_tracks_full_buy_call_cycle() -> None:
    script = load_script()

    output, _ = script.run_backtest(
        indicator_df(["UP", "UP", "UP", "DOWN"], [100.0, 101.0, 102.0, 103.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == [
        "BUY_CALL",
        "HOLD_BUY_CALL",
        "HOLD_BUY_CALL",
        "CLOSE_BUY_CALL",
    ]
    assert output["candle_count_from_call"].tolist() == [1, 2, 3, 4]


def test_candle_count_tracks_full_sell_call_cycle() -> None:
    script = load_script()

    output, _ = script.run_backtest(
        indicator_df(["DOWN", "DOWN", "DOWN", "UP"], [100.0, 99.0, 98.0, 97.0]),
        pip_size=0.01,
    )

    assert output["CALL"].tolist() == [
        "SELL_CALL",
        "HOLD_SELL_CALL",
        "HOLD_SELL_CALL",
        "CLOSE_SELL_CALL",
    ]
    assert output["candle_count_from_call"].tolist() == [1, 2, 3, 4]


def test_high_low_and_buy_side_pip_formulas_are_cumulative() -> None:
    script = load_script()
    df = indicator_df(["UP", "UP", "DOWN"], [100.0, 101.0, 102.0])
    df.loc[0, ["High", "Low"]] = [105.0, 95.0]
    df.loc[1, ["High", "Low"]] = [103.0, 99.0]
    df.loc[2, ["High", "Low"]] = [102.5, 98.0]

    output, summary = script.run_backtest(df, pip_size=0.01)
    close_row = output.iloc[2]

    assert output.iloc[0]["highest_price_after_call"] == 100.0
    assert output.iloc[0]["lowest_price_after_call"] == 100.0
    assert close_row["highest_price_after_call"] == 103.0
    assert close_row["lowest_price_after_call"] == 98.0
    assert close_row["max_win_side_pips"] == 300.0
    assert close_row["max_loss_side_pips"] == 200.0
    assert summary["max_win_side_pips"] == 300.0
    assert summary["max_loss_side_pips"] == 200.0


def test_high_low_and_sell_side_pip_formulas_are_cumulative() -> None:
    script = load_script()
    df = indicator_df(["DOWN", "DOWN", "UP"], [100.0, 99.0, 98.0])
    df.loc[0, ["High", "Low"]] = [105.0, 95.0]
    df.loc[1, ["High", "Low"]] = [101.0, 97.0]
    df.loc[2, ["High", "Low"]] = [99.0, 96.0]

    output, summary = script.run_backtest(df, pip_size=0.01)
    close_row = output.iloc[2]

    assert output.iloc[0]["highest_price_after_call"] == 100.0
    assert output.iloc[0]["lowest_price_after_call"] == 100.0
    assert close_row["highest_price_after_call"] == 101.0
    assert close_row["lowest_price_after_call"] == 96.0
    assert close_row["max_win_side_pips"] == 400.0
    assert close_row["max_loss_side_pips"] == 100.0
    assert summary["max_win_side_pips"] == 400.0
    assert summary["max_loss_side_pips"] == 100.0


def test_active_call_rows_do_not_leave_evaluation_cells_empty() -> None:
    script = load_script()

    output, _ = script.run_backtest(indicator_df(["UP", "UP", "DOWN"]), pip_size=0.01)
    active_rows = output[output["CALL"].isin(ACTIVE_CALLS)]

    for field in ACTIVE_FIELDS:
        assert active_rows[field].notna().all(), field


def test_closed_calls_summary_contains_only_close_rows_and_required_metrics() -> None:
    script = load_script()
    output, _ = script.run_backtest(
        indicator_df(
            ["UP", "UP", "DOWN", "UP", "DOWN", "DOWN", "UP"],
            [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0],
        ),
        pip_size=0.01,
    )

    summary = script.build_closed_calls_summary(output)

    assert summary["CALL"].tolist() == ["CLOSE_BUY_CALL", "CLOSE_SELL_CALL"]
    assert set(summary["CALL"].unique()) == {"CLOSE_BUY_CALL", "CLOSE_SELL_CALL"}
    for column in [
        "max_win_side_pips",
        "max_loss_side_pips",
        "realized_pips",
        "result",
        "candle_count_from_call",
    ]:
        assert column in summary.columns
        assert summary[column].notna().all(), column


def test_datetime_is_optional_and_file_order_is_used_without_datetime() -> None:
    script = load_script()

    output, _ = script.run_backtest(indicator_df(["UP", "DOWN"], include_datetime=False), pip_size=0.01)

    assert "DateTime" not in output.columns
    assert output.iloc[0]["entry_datetime"] is None
    assert output.iloc[1]["close_datetime"] is None


def test_cli_writes_outputs_and_prints_success_json(tmp_path: Path) -> None:
    input_root = tmp_path / "indicators"
    output_dir = tmp_path / "backtests"
    input_root.mkdir()
    input_csv = input_root / "USDJPY_M5_ema_stack_td_ts.csv"
    indicator_df(["UP", "DOWN", "UP"]).to_csv(input_csv, index=False)
    legacy_summary_json = output_dir / "USDJPY_M5_ema_stack_td_ts_BACKTEST_RULE_01_summary.json"
    output_dir.mkdir()
    legacy_summary_json.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--file-name",
            input_csv.name,
            "--indicator-root-path",
            str(input_root),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    response = json.loads(completed.stdout)
    assert response["status"] == "SUCCESS"
    assert response["indicator_name"] == "EMA_STACK"
    assert Path(response["output_csv_path"]).exists()
    assert "closed_calls_summary_csv_path" in response
    assert "output_summary_path" not in response
    assert response["closed_call_count"] == 1
    assert Path(response["closed_calls_summary_csv_path"]).exists()
    assert not legacy_summary_json.exists()

    full_output = pd.read_csv(response["output_csv_path"])
    closed_summary = pd.read_csv(response["closed_calls_summary_csv_path"])

    assert "candle_count_from_call" in full_output.columns
    assert closed_summary["CALL"].tolist() == ["CLOSE_BUY_CALL"]

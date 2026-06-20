import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rules" / "run_rule_gt_td_3_9_tlpsl_18_22.py"
REQUIRED_TD_COLUMNS = [
    "ADX_TD",
    "ATR_TD",
    "BOLLINGER_TD",
    "EMA_STACK_TD",
    "ICHIMOKU_TD",
    "MACD_TD",
    "PARABOLIC_SAR_TD",
    "RSI_TD",
    "STOCHASTIC_TD",
]
SUMMARY_FIELDS = [
    "rule_name",
    "rule_description",
    "source_file_name",
    "source_file_path",
    "output_calls_csv",
    "output_summary_json",
    "row_count",
    "td_column_count",
    "required_td_columns",
    "first_datetime",
    "last_datetime",
    "total_call_count",
    "total_sell_call_count",
    "total_buy_call_count",
    "total_sell_call_tp_hit_count",
    "total_sell_call_sl_hit_count",
    "total_buy_call_tp_hit_count",
    "total_buy_call_sl_hit_count",
    "total_sell_call_close_loss_count",
    "total_sell_call_close_loss_pips",
    "total_sell_call_close_win_count",
    "total_sell_call_close_win_pips",
    "total_buy_call_close_loss_count",
    "total_buy_call_close_loss_pips",
    "total_buy_call_close_win_count",
    "total_buy_call_close_win_pips",
    "total_tp_hit_count",
    "total_sl_hit_count",
    "total_win_call_count",
    "total_loss_call_count",
    "total_win_pip",
    "total_loss_pip",
    "net_pip",
    "actual_signed_pips",
    "created_at_utc",
]


def load_script():
    spec = importlib.util.spec_from_file_location("run_rule_gt_td_3_9_tlpsl_18_22", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def td_values(default: str, **overrides: str) -> dict[str, str]:
    values = {column: default for column in REQUIRED_TD_COLUMNS}
    values.update(overrides)
    return values


def row(index: int, close: float, high: float, low: float, default_td: str, **overrides: str):
    result = {
        "DateTime": f"2026-01-01 00:{index:02d}:00",
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
    }
    result.update(td_values(default_td, **overrides))
    for column in REQUIRED_TD_COLUMNS:
        result[column.replace("_TD", "_TS")] = 1.0
    return result


def df_from_rows(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def run(rows: list[dict]):
    script = load_script()
    return script.run_backtest(pd.DataFrame(rows), "sample.csv")


def test_buy_call_opens_when_all_9_td_are_up() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.22, 100.22, 100.05, "UP"),
    ])

    assert calls.iloc[0]["call_type"] == "BUY_CALL"
    assert calls.iloc[0]["close_reason"] == "TP_HIT"


def test_sell_call_opens_when_all_9_td_are_down() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.78, 99.95, 99.78, "DOWN"),
    ])

    assert calls.iloc[0]["call_type"] == "SELL_CALL"
    assert calls.iloc[0]["close_reason"] == "TP_HIT"


def test_no_call_opens_when_one_td_is_no_direction() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP", ADX_TD="NO_DIRECTION"),
    ])

    assert calls.empty


def test_buy_call_tp_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.22, 99.95, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["result"] == "WIN_CALL"
    assert calls.iloc[0]["signed_pips"] == 22.0


def test_buy_call_sl_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 99.9, 100.05, 99.82, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["result"] == "LOSS_CALL"
    assert calls.iloc[0]["signed_pips"] == -18.0


def test_sell_call_tp_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.9, 100.05, 99.78, "DOWN"),
    ])

    assert calls.iloc[0]["call_type"] == "SELL_CALL"
    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["signed_pips"] == 22.0


def test_sell_call_sl_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 100.1, 100.18, 99.95, "DOWN"),
    ])

    assert calls.iloc[0]["call_type"] == "SELL_CALL"
    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -18.0


def test_buy_call_closes_as_close_win_on_two_opposition_td_values() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.1, 100.0, "UP", ADX_TD="DOWN", ATR_TD="NO_DIRECTION"),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_WIN"
    assert calls.iloc[0]["result"] == "WIN_CALL"
    assert calls.iloc[0]["td_opposition_count"] == 2


def test_buy_call_closes_as_close_loss_on_two_opposition_td_values() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 99.9, 100.0, 99.9, "UP", ADX_TD="DOWN", ATR_TD="NO_DIRECTION"),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_LOSS"
    assert calls.iloc[0]["result"] == "LOSS_CALL"


def test_sell_call_closes_as_close_win_on_two_opposition_td_values() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.9, 100.0, 99.9, "DOWN", ADX_TD="UP", ATR_TD="NO_DIRECTION"),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_WIN"
    assert calls.iloc[0]["result"] == "WIN_CALL"


def test_sell_call_closes_as_close_loss_on_two_opposition_td_values() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 100.1, 100.1, 100.0, "DOWN", ADX_TD="UP", ATR_TD="NO_DIRECTION"),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_LOSS"
    assert calls.iloc[0]["result"] == "LOSS_CALL"


def test_same_candle_tp_sl_conflict_closes_as_sl_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.0, 100.22, 99.82, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -18.0


def test_end_of_file_open_call_force_closes() -> None:
    calls, _ = run([row(0, 100.0, 100.05, 99.95, "UP")])

    assert calls.iloc[0]["close_reason"] == "FORCE_CLOSE_END_OF_FILE"
    assert calls.iloc[0]["close_row_index"] == 0
    assert calls.iloc[0]["result"] == "LOSS_CALL"


def test_summary_json_contains_all_required_fields() -> None:
    calls, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.22, 99.95, "UP"),
    ])

    assert list(summary.keys()) == SUMMARY_FIELDS
    assert summary["total_call_count"] == len(calls)
    assert summary["total_win_call_count"] == 1
    assert summary["total_win_pip"] == 21


def test_cli_writes_outputs_and_markdown_documentation() -> None:
    temp_root = PROJECT_ROOT / ".tmp_rule_gt_td_3_9_tests" / uuid.uuid4().hex
    experiments_root = temp_root / "experiments"
    output_dir = temp_root / "rules"
    try:
        experiments_root.mkdir(parents=True)
        input_csv = experiments_root / "sample_indicator_collection.csv"
        df_from_rows(
            row(0, 100.0, 100.05, 99.95, "UP"),
            row(1, 100.1, 100.22, 99.95, "UP"),
        ).to_csv(input_csv, index=False)

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--file-name",
                input_csv.name,
                "--experiments-root-path",
                str(experiments_root),
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        response = json.loads(completed.stdout)
        calls_csv = Path(response["output_calls_csv"])
        summary_json = Path(response["output_summary_json"])
        markdown_path = Path(response["output_markdown"])

        assert calls_csv.exists()
        assert summary_json.exists()
        assert markdown_path == output_dir / "Rule-GT-TD-3-9-TLPSL-18-22.md"
        assert markdown_path.exists()
        assert "Rule-GT-TD-3-9-TLPSL-18-22" in markdown_path.read_text(encoding="utf-8")
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)

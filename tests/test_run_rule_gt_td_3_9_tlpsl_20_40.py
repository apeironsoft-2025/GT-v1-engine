import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rules" / "run_rule_gt_td_3_9_tlpsl_20_40.py"
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
    "first_datetime",
    "last_datetime",
    "total_call_count",
    "total_buy_call_count",
    "total_sell_call_count",
    "total_win_count",
    "total_loss_count",
    "total_tp_hit_count",
    "total_sl_hit_count",
    "total_close_win_count",
    "total_close_loss_count",
    "total_force_close_count",
    "total_buy_call_tp_hit_count",
    "total_sell_call_tp_hit_count",
    "total_buy_call_sl_hit_count",
    "total_sell_call_sl_hit_count",
    "total_buy_call_tp_hit_pips",
    "total_sell_call_tp_hit_pips",
    "total_buy_call_sl_hit_pips",
    "total_sell_call_sl_hit_pips",
    "total_close_win_pips",
    "total_close_loss_pips",
    "total_force_close_win_pips",
    "total_force_close_loss_pips",
    "total_win_pip",
    "total_loss_pip",
    "net_pip",
    "is_profitable",
    "created_at_utc",
]


def load_script():
    spec = importlib.util.spec_from_file_location("run_rule_gt_td_3_9_tlpsl_20_40", SCRIPT_PATH)
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
        row(1, 100.2, 100.2, 100.05, "UP"),
    ])

    assert calls.iloc[0]["call_type"] == "BUY_CALL"
    assert calls.iloc[0]["close_reason"] == "TP_HIT"


def test_sell_call_opens_when_all_9_td_are_down() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.8, 99.95, 99.8, "DOWN"),
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
        row(1, 100.1, 100.2, 99.95, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["result"] == "WIN_CALL"
    assert calls.iloc[0]["signed_pips"] == 20.0


def test_buy_call_sl_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 99.8, 100.05, 99.6, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["result"] == "LOSS_CALL"
    assert calls.iloc[0]["signed_pips"] == -40.0


def test_sell_call_tp_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.9, 100.05, 99.8, "DOWN"),
    ])

    assert calls.iloc[0]["call_type"] == "SELL_CALL"
    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["signed_pips"] == 20.0


def test_sell_call_sl_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 100.1, 100.4, 99.95, "DOWN"),
    ])

    assert calls.iloc[0]["call_type"] == "SELL_CALL"
    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -40.0


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
        row(1, 100.0, 100.2, 99.6, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -40.0


def test_end_of_file_open_call_force_closes() -> None:
    calls, _ = run([row(0, 100.0, 100.05, 99.95, "UP")])

    assert calls.iloc[0]["close_reason"] == "FORCE_CLOSE_END_OF_FILE"
    assert calls.iloc[0]["close_row_index"] == 0
    assert calls.iloc[0]["result"] == "LOSS_CALL"


def test_summary_json_contains_all_required_fields() -> None:
    calls, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.2, 99.95, "UP"),
    ])

    assert list(summary.keys()) == SUMMARY_FIELDS
    assert summary["total_call_count"] == len(calls)
    assert summary["total_win_count"] == 1
    assert summary["total_loss_count"] == 0
    assert summary["total_buy_call_tp_hit_pips"] == 20.0
    assert summary["total_win_pip"] == 20.0
    assert summary["total_loss_pip"] == 0.0
    assert summary["net_pip"] == 20.0
    assert summary["is_profitable"] is True


def test_clean_summary_uses_result_counts_and_actual_pip_magnitudes() -> None:
    _, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.1, 100.0, "UP", ADX_TD="DOWN", ATR_TD="NO_DIRECTION"),
        row(2, 100.0, 100.05, 99.95, "UP"),
        row(3, 99.9, 100.0, 99.9, "UP", ADX_TD="DOWN", ATR_TD="NO_DIRECTION"),
    ])

    assert summary["total_win_count"] == 1
    assert summary["total_loss_count"] == 1
    assert summary["total_close_win_pips"] == 10.0
    assert summary["total_close_loss_pips"] == 10.0
    assert summary["total_win_pip"] == 10.0
    assert summary["total_loss_pip"] == 10.0
    assert summary["net_pip"] == summary["total_win_pip"] - summary["total_loss_pip"]
    assert summary["is_profitable"] is False


def test_clean_summary_has_no_old_fields_or_fixed_formula_logic() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")
    _, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.2, 99.95, "UP"),
        row(2, 100.0, 100.05, 99.95, "DOWN"),
        row(3, 100.1, 100.4, 99.95, "DOWN"),
    ])

    assert "actual_signed_pips" not in summary
    assert "actual_signed_pips" not in script_text
    assert "* 19" not in script_text
    assert "* 41" not in script_text
    assert summary["total_win_count"] == 1
    assert summary["total_loss_count"] == 1
    assert summary["total_win_pip"] == 20.0
    assert summary["total_loss_pip"] == 40.0


def test_cli_writes_outputs_and_markdown_documentation() -> None:
    temp_root = PROJECT_ROOT / ".tmp_rule_gt_td_3_9_tests" / uuid.uuid4().hex
    experiments_root = temp_root / "experiments"
    output_dir = temp_root / "rules"
    try:
        experiments_root.mkdir(parents=True)
        input_csv = experiments_root / "sample_indicator_collection.csv"
        df_from_rows(
            row(0, 100.0, 100.05, 99.95, "UP"),
            row(1, 100.1, 100.4, 99.95, "UP"),
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
        assert markdown_path == output_dir / "Rule-GT-TD-3-9-TLPSL-20-40.md"
        assert markdown_path.exists()
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "Rule-GT-TD-3-9-TLPSL-20-40" in markdown
        assert "BUY_CALL TP is High - entry_price >= 0.200" in markdown
        assert "BUY_CALL SL is entry_price - Low >= 0.400" in markdown
        assert "net_pip is total_win_pip - total_loss_pip" in markdown
        assert "not WIN_CALL count multiplied by 19" in markdown
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


def test_output_dir_parent_creates_child_rule_folder() -> None:
    script = load_script()
    output_dir = Path(r"F:\GT-v1-shared-storage\rules")
    input_path = Path("sample_indicator_collection.csv")

    calls_csv, summary_json, markdown_path = script.output_paths(input_path, output_dir)

    expected_rule_dir = output_dir / "Rule-GT-TD-3-9-TLPSL-20-40"
    assert calls_csv.parent == expected_rule_dir
    assert summary_json.parent == expected_rule_dir
    assert markdown_path == output_dir / "Rule-GT-TD-3-9-TLPSL-20-40.md"


def test_output_dir_rule_folder_is_used_directly_without_nested_duplicate() -> None:
    script = load_script()
    output_dir = Path(r"F:\GT-v1-shared-storage\rules\Rule-GT-TD-3-9-TLPSL-20-40")
    input_path = Path("sample_indicator_collection.csv")

    calls_csv, summary_json, markdown_path = script.output_paths(input_path, output_dir)

    assert calls_csv.parent == output_dir
    assert summary_json.parent == output_dir
    assert "Rule-GT-TD-3-9-TLPSL-20-40\\Rule-GT-TD-3-9-TLPSL-20-40" not in str(calls_csv)
    assert markdown_path == output_dir.parent / "Rule-GT-TD-3-9-TLPSL-20-40.md"

import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rules" / "run_rule_gt_td_4_9_tlpsl_30_30.py"
RULE_NAME = "Rule-GT-TD-4-9-TLPSL-30-30"
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
    "total_call_count",
    "total_buy_call_tp_hit_count",
    "total_sell_call_tp_hit_count",
    "total_buy_call_sl_hit_count",
    "total_sell_call_sl_hit_count",
    "total_close_loss_count",
    "total_close_win_count",
    "total_buy_call_tp_hit_pips",
    "total_sell_call_tp_hit_pips",
    "total_buy_call_sl_hit_pips",
    "total_sell_call_sl_hit_pips",
    "total_close_loss_pips",
    "total_close_win_pips",
    "total_win_pip",
    "total_loss_pip",
    "net_pip",
    "is_profitable",
]


def load_script():
    spec = importlib.util.spec_from_file_location("run_rule_gt_td_4_9_tlpsl_30_30", SCRIPT_PATH)
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


def four_buy_opposition() -> dict[str, str]:
    return {
        "ADX_TD": "DOWN",
        "ATR_TD": "DOWN",
        "BOLLINGER_TD": "NO_DIRECTION",
        "EMA_STACK_TD": "NO_DIRECTION",
    }


def three_buy_opposition() -> dict[str, str]:
    return {"ADX_TD": "DOWN", "ATR_TD": "DOWN", "BOLLINGER_TD": "NO_DIRECTION"}


def four_sell_opposition() -> dict[str, str]:
    return {
        "ADX_TD": "UP",
        "ATR_TD": "UP",
        "BOLLINGER_TD": "NO_DIRECTION",
        "EMA_STACK_TD": "NO_DIRECTION",
    }


def test_rule_name_is_new_rule_name() -> None:
    script = load_script()
    assert script.RULE_NAME == RULE_NAME


def test_output_folder_is_new_rule_folder() -> None:
    script = load_script()
    output_dir = Path(r"F:\GT-v1-shared-storage\rules")
    calls_csv, summary_json, markdown_path = script.output_paths(Path("sample.csv"), output_dir)

    expected_rule_dir = output_dir / RULE_NAME
    assert calls_csv.parent == expected_rule_dir
    assert summary_json.parent == expected_rule_dir
    assert markdown_path == output_dir / f"{RULE_NAME}.md"


def test_buy_call_opens_when_all_9_td_are_up() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.2, 100.2, 100.05, "UP"),
    ])

    assert calls.iloc[0]["call_type"] == "BUY_CALL"


def test_sell_call_opens_when_all_9_td_are_down() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.8, 99.95, 99.8, "DOWN"),
    ])

    assert calls.iloc[0]["call_type"] == "SELL_CALL"


def test_no_call_opens_when_one_td_is_no_direction() -> None:
    calls, _ = run([row(0, 100.0, 100.05, 99.95, "UP", ADX_TD="NO_DIRECTION")])

    assert calls.empty


def test_buy_call_tp_hit_uses_30_pips() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.3, 99.95, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["signed_pips"] == 30.0


def test_sell_call_tp_hit_uses_30_pips() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.9, 100.05, 99.7, "DOWN"),
    ])

    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["signed_pips"] == 30.0


def test_buy_call_sl_hit_uses_30_pips() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 99.8, 100.05, 99.7, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -30.0


def test_sell_call_sl_hit_uses_30_pips() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 100.1, 100.3, 99.95, "DOWN"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -30.0


def test_same_candle_tp_sl_conflict_closes_as_sl_hit() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.0, 100.3, 99.7, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "SL_HIT"
    assert calls.iloc[0]["signed_pips"] == -30.0


def test_td_opposition_does_not_close_when_only_3_of_9_are_opposite() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.05, 100.10, 99.95, "UP", **three_buy_opposition()),
        row(2, 100.3, 100.3, 100.0, "UP"),
    ])

    assert calls.iloc[0]["close_reason"] == "TP_HIT"
    assert calls.iloc[0]["close_row_index"] == 2
    assert calls.iloc[0]["td_opposition_count"] == 0


def test_td_opposition_closes_when_4_of_9_are_opposite() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.1, 100.0, "UP", **four_buy_opposition()),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_WIN"
    assert calls.iloc[0]["td_opposition_count"] == 4


def test_buy_call_closes_as_close_win_with_4_opposition_and_close_above_entry() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.1, 100.0, "UP", **four_buy_opposition()),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_WIN"
    assert calls.iloc[0]["result"] == "WIN_CALL"


def test_buy_call_closes_as_close_loss_with_4_opposition_and_close_below_entry() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 99.9, 100.0, 99.9, "UP", **four_buy_opposition()),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_LOSS"
    assert calls.iloc[0]["result"] == "LOSS_CALL"


def test_sell_call_closes_as_close_win_with_4_opposition_and_close_below_entry() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 99.9, 100.0, 99.9, "DOWN", **four_sell_opposition()),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_WIN"
    assert calls.iloc[0]["result"] == "WIN_CALL"


def test_sell_call_closes_as_close_loss_with_4_opposition_and_close_above_entry() -> None:
    calls, _ = run([
        row(0, 100.0, 100.05, 99.95, "DOWN"),
        row(1, 100.1, 100.1, 100.0, "DOWN", **four_sell_opposition()),
    ])

    assert calls.iloc[0]["close_reason"] == "CLOSE_LOSS"
    assert calls.iloc[0]["result"] == "LOSS_CALL"


def test_summary_json_contains_only_clean_fields() -> None:
    _, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.3, 99.95, "UP"),
    ])

    assert list(summary.keys()) == SUMMARY_FIELDS
    assert "actual_signed_pips" not in summary
    assert "created_at_utc" not in summary


def test_summary_totals_use_tp_close_win_sl_and_close_loss_pips() -> None:
    _, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.3, 99.95, "UP"),
        row(2, 100.0, 100.05, 99.95, "DOWN"),
        row(3, 99.9, 100.05, 99.7, "DOWN"),
        row(4, 100.0, 100.05, 99.95, "UP"),
        row(5, 99.8, 100.05, 99.7, "UP"),
        row(6, 100.0, 100.05, 99.95, "UP"),
        row(7, 100.1, 100.1, 100.0, "UP", **four_buy_opposition()),
        row(8, 100.0, 100.05, 99.95, "DOWN"),
        row(9, 100.1, 100.1, 100.0, "DOWN", **four_sell_opposition()),
    ])

    assert summary["total_buy_call_tp_hit_pips"] == 30.0
    assert summary["total_sell_call_tp_hit_pips"] == 30.0
    assert summary["total_buy_call_sl_hit_pips"] == 30.0
    assert summary["total_sell_call_sl_hit_pips"] == 0.0
    assert summary["total_close_win_pips"] == 10.0
    assert summary["total_close_loss_pips"] == 10.0
    assert summary["total_win_pip"] == 70.0
    assert summary["total_loss_pip"] == 40.0
    assert summary["net_pip"] == summary["total_win_pip"] - summary["total_loss_pip"]
    assert summary["is_profitable"] is True


def test_is_profitable_true_only_when_net_pip_positive() -> None:
    _, summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 100.1, 100.3, 99.95, "UP"),
    ])

    assert summary["net_pip"] > 0
    assert summary["is_profitable"] is True

    _, loss_summary = run([
        row(0, 100.0, 100.05, 99.95, "UP"),
        row(1, 99.8, 100.05, 99.7, "UP"),
    ])

    assert loss_summary["net_pip"] <= 0
    assert loss_summary["is_profitable"] is False


def test_markdown_documentation_is_created() -> None:
    temp_root = PROJECT_ROOT / ".tmp_rule_gt_td_4_9_tests" / uuid.uuid4().hex
    experiments_root = temp_root / "experiments"
    output_dir = temp_root / "rules"
    try:
        experiments_root.mkdir(parents=True)
        input_csv = experiments_root / "sample_indicator_collection.csv"
        df_from_rows(
            row(0, 100.0, 100.05, 99.95, "UP"),
            row(1, 100.1, 100.3, 99.95, "UP"),
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
        markdown_path = Path(response["output_markdown"])
        summary = json.loads(Path(response["output_summary_json"]).read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")

        assert markdown_path == output_dir / f"{RULE_NAME}.md"
        assert markdown_path.exists()
        assert RULE_NAME in markdown
        assert "at least 4 required TD columns" in markdown
        assert "Clean Summary Fields" in markdown
        assert list(summary.keys()) == SUMMARY_FIELDS
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


def test_output_dir_parent_creates_child_rule_folder() -> None:
    script = load_script()
    output_dir = Path(r"F:\GT-v1-shared-storage\rules")

    calls_csv, summary_json, markdown_path = script.output_paths(Path("sample.csv"), output_dir)

    assert calls_csv.parent == output_dir / RULE_NAME
    assert summary_json.parent == output_dir / RULE_NAME
    assert markdown_path == output_dir / f"{RULE_NAME}.md"


def test_output_dir_rule_folder_is_used_directly_without_nested_duplicate() -> None:
    script = load_script()
    output_dir = Path(r"F:\GT-v1-shared-storage\rules\Rule-GT-TD-4-9-TLPSL-30-30")

    calls_csv, summary_json, markdown_path = script.output_paths(Path("sample.csv"), output_dir)

    assert calls_csv.parent == output_dir
    assert summary_json.parent == output_dir
    assert f"{RULE_NAME}\\{RULE_NAME}" not in str(calls_csv)
    assert markdown_path == output_dir.parent / f"{RULE_NAME}.md"


def test_validation_errors_are_clear() -> None:
    script = load_script()
    valid = df_from_rows(row(0, 100.0, 100.05, 99.95, "UP"))

    with pytest.raises(ValueError, match="missing required OHLC columns"):
        script.validate_input_frame(valid.drop(columns=["Close"]))
    with pytest.raises(ValueError, match="missing required TD columns"):
        script.validate_input_frame(valid.drop(columns=["ADX_TD"]))
    with pytest.raises(ValueError, match="Input CSV is empty"):
        script.validate_input_frame(valid.iloc[0:0])
    bad_numeric = valid.astype({"High": "object"}).copy()
    bad_numeric.loc[0, "High"] = "bad"
    with pytest.raises(ValueError, match="Invalid numeric OHLC values"):
        script.validate_input_frame(bad_numeric)
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        script.execute("missing.csv", str(PROJECT_ROOT), str(PROJECT_ROOT / ".tmp_missing_rule"))

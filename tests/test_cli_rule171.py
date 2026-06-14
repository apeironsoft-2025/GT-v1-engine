import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

INDICATORS = ["MACD", "RSI", "ADX", "ATR", "BOLLINGER", "EMA_STACK"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH)
    return subprocess.run(
        [sys.executable, "-m", "gt_v1_engine.cli", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _indicator_csv(tmp_path: Path, drop_column: str | None = None) -> Path:
    rows = []
    directions = ["UP", "UP", "UP", "NO_SIGNAL"]
    for index, direction in enumerate(directions):
        close = 160.0 + index * 0.05
        row = {
            "DateTime": (pd.Timestamp("2025-12-01T00:00:00+00:00") + pd.Timedelta(minutes=5 * index)).isoformat(),
            "Open": close,
            "High": close + 0.05,
            "Low": close - 0.05,
            "Close": close,
            "SRP": close,
        }
        for indicator in INDICATORS:
            row[f"{indicator}_TD"] = direction
            row[f"{indicator}_TS"] = 1.0
        rows.append(row)
    df = pd.DataFrame(rows)
    if drop_column:
        df = df.drop(columns=[drop_column])
    path = tmp_path / "rule171_input.csv"
    df.to_csv(path, index=False)
    return path


def test_backtest_rule171_help_exits_zero() -> None:
    assert _run_cli("backtest-rule171", "--help").returncode == 0


def test_cli_backtest_rule171_works(tmp_path: Path) -> None:
    output_csv = tmp_path / "rule171.csv"
    output_summary = tmp_path / "rule171_summary.json"
    completed = _run_cli(
        "backtest-rule171",
        "--input",
        str(_indicator_csv(tmp_path)),
        "--config",
        "configs/rules/rule171.yaml",
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicators",
        "MACD,RSI,ADX,ATR,BOLLINGER,EMA_STACK",
        "--start",
        "2025-12-01T00:00:00+00:00",
        "--end",
        "2025-12-02T00:00:00+00:00",
        "--pip-size",
        "0.01",
        "--strength-threshold",
        "4.5",
        "--entry-confirmation-required",
        "3",
        "--take-profit-pips",
        "5",
        "--stop-loss-pips",
        "8",
        "--max-holding-candles",
        "2",
        "--output-csv",
        str(output_csv),
        "--output-summary",
        str(output_summary),
    )
    assert completed.returncode == 0, completed.stderr
    assert output_csv.exists()
    assert output_summary.exists()
    assert "PASS" in completed.stdout


def test_cli_unknown_indicator_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    completed = _run_cli(
        "backtest-rule171",
        "--input",
        str(_indicator_csv(tmp_path)),
        "--config",
        "configs/rules/rule171.yaml",
        "--indicators",
        "MACD,UNKNOWN",
        "--output-csv",
        str(tmp_path / "out.csv"),
        "--output-summary",
        str(tmp_path / "summary.json"),
    )
    assert completed.returncode != 0
    assert "GT-v1-engine ERROR" in completed.stdout + completed.stderr


def test_cli_missing_td_ts_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    completed = _run_cli(
        "backtest-rule171",
        "--input",
        str(_indicator_csv(tmp_path, drop_column="MACD_TS")),
        "--config",
        "configs/rules/rule171.yaml",
        "--start",
        "2025-12-01T00:00:00+00:00",
        "--end",
        "2025-12-02T00:00:00+00:00",
        "--output-csv",
        str(tmp_path / "out.csv"),
        "--output-summary",
        str(tmp_path / "summary.json"),
    )
    assert completed.returncode != 0
    assert "GT-v1-engine ERROR" in completed.stdout + completed.stderr

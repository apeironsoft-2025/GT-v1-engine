import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

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


def _indicator_ready_csv(tmp_path: Path) -> Path:
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
                "MACD_TS": 0.5,
                "RSI_TD": direction,
                "RSI_TS": 0.5,
            }
        )
    path = tmp_path / "indicator_ready.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_backtest_indicator_help_exits_zero() -> None:
    assert _run_cli("backtest-indicator", "--help").returncode == 0


def test_backtest_all_indicators_help_exits_zero() -> None:
    assert _run_cli("backtest-all-indicators", "--help").returncode == 0


def test_cli_backtest_indicator_works(tmp_path: Path) -> None:
    output_csv = tmp_path / "MACD_backtest.csv"
    completed = _run_cli(
        "backtest-indicator",
        "--input",
        str(_indicator_ready_csv(tmp_path)),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicator",
        "MACD",
        "--horizon-candles",
        "2",
        "--target-pips",
        "5",
        "--stop-pips",
        "8",
        "--pip-size",
        "0.01",
        "--output-csv",
        str(output_csv),
    )
    assert completed.returncode == 0, completed.stderr
    assert output_csv.exists()
    assert "PASS" in completed.stdout


def test_cli_backtest_all_indicators_works(tmp_path: Path) -> None:
    output_dir = tmp_path / "backtests"
    summary_json = tmp_path / "summary.json"
    completed = _run_cli(
        "backtest-all-indicators",
        "--input",
        str(_indicator_ready_csv(tmp_path)),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicators",
        "MACD,RSI",
        "--horizon-candles",
        "2",
        "--target-pips",
        "5",
        "--stop-pips",
        "8",
        "--pip-size",
        "0.01",
        "--output-dir",
        str(output_dir),
        "--summary-json",
        str(summary_json),
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "USDJPY_M5_MACD_backtest.csv").exists()
    assert (output_dir / "USDJPY_M5_RSI_backtest.csv").exists()
    assert summary_json.exists()


def test_cli_unknown_indicator_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    completed = _run_cli(
        "backtest-indicator",
        "--input",
        str(_indicator_ready_csv(tmp_path)),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicator",
        "UNKNOWN",
        "--output-csv",
        str(tmp_path / "out.csv"),
    )
    assert completed.returncode != 0
    assert "GT-v1-engine ERROR" in completed.stdout + completed.stderr

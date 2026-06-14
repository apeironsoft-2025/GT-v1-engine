import os
import subprocess
import sys
from pathlib import Path

from tests.indicator_test_helpers import make_ohlc_dataframe

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


def _write_market_csv(tmp_path: Path) -> Path:
    path = tmp_path / "USDJPY_M5.csv"
    make_ohlc_dataframe(rows=180).to_csv(path, index=False)
    return path


def test_run_indicators_help_exits_zero() -> None:
    assert _run_cli("run-indicators", "--help").returncode == 0


def test_cli_run_indicators_writes_outputs(tmp_path: Path) -> None:
    input_path = _write_market_csv(tmp_path)
    output_csv = tmp_path / "USDJPY_M5_6I.csv"
    summary_json = tmp_path / "USDJPY_M5_6I_summary.json"

    completed = _run_cli(
        "run-indicators",
        "--input",
        str(input_path),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicators",
        "MACD,RSI,ADX,ATR,BOLLINGER,EMA_STACK",
        "--output-csv",
        str(output_csv),
        "--summary-json",
        str(summary_json),
    )

    assert completed.returncode == 0, completed.stderr
    assert output_csv.exists()
    assert summary_json.exists()
    assert "validation status" in completed.stdout
    assert "PASS" in completed.stdout


def test_cli_unknown_indicator_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    input_path = _write_market_csv(tmp_path)
    completed = _run_cli(
        "run-indicators",
        "--input",
        str(input_path),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicators",
        "MACD,UNKNOWN",
        "--output-csv",
        str(tmp_path / "out.csv"),
    )

    assert completed.returncode != 0
    assert "GT-v1-engine ERROR" in completed.stdout + completed.stderr

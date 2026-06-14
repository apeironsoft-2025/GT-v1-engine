import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def _raw_csv(tmp_path: Path, rows: int = 180) -> Path:
    index = np.arange(rows)
    close = 160.0 + np.linspace(0, 2.0, rows) + 0.08 * np.sin(index / 4)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    df = pd.DataFrame(
        {
            "DateTime": pd.date_range("2025-12-01", periods=rows, freq="5min", tz="UTC"),
            "Open": open_,
            "High": np.maximum(open_, close) + 0.08,
            "Low": np.minimum(open_, close) - 0.08,
            "Close": close,
            "SRP": close,
        }
    )
    path = tmp_path / "USDJPY_M5.csv"
    df.to_csv(path, index=False)
    return path


def test_run_smoke_pipeline_help_exits_zero() -> None:
    assert _run_cli("run-smoke-pipeline", "--help").returncode == 0


def test_cli_run_smoke_pipeline_creates_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    completed = _run_cli(
        "run-smoke-pipeline",
        "--input",
        str(_raw_csv(tmp_path)),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--start",
        "2025-12-01T00:00:00+00:00",
        "--end",
        "2025-12-01T12:00:00+00:00",
        "--indicators",
        "MACD,RSI,ADX,ATR,BOLLINGER,EMA_STACK",
        "--config",
        "configs/rules/rule171.yaml",
        "--output-root",
        str(output_root),
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_root / "indicators" / "USDJPY_M5_6I.csv").exists()
    assert (output_root / "indicators" / "USDJPY_M5_6I.parquet").exists()
    assert (output_root / "indicators" / "USDJPY_M5_6I_summary.json").exists()
    assert (output_root / "backtests" / "indicators" / "USDJPY_M5_all_indicators_summary.json").exists()
    assert (output_root / "backtests" / "rules" / "rule171_USDJPY_M5_20251201_20251201.csv").exists()
    assert (
        output_root
        / "backtests"
        / "rules"
        / "rule171_USDJPY_M5_20251201_20251201_summary.json"
    ).exists()
    assert (output_root / "reports" / "smoke_pipeline_USDJPY_M5_20251201_20251201.md").exists()
    assert "PASS" in completed.stdout


def test_cli_invalid_indicator_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    completed = _run_cli(
        "run-smoke-pipeline",
        "--input",
        str(_raw_csv(tmp_path)),
        "--pair",
        "USDJPY",
        "--timeframe",
        "M5",
        "--indicators",
        "MACD,UNKNOWN",
        "--config",
        "configs/rules/rule171.yaml",
        "--output-root",
        str(tmp_path / "out"),
    )
    assert completed.returncode != 0
    assert "GT-v1-engine ERROR" in completed.stdout + completed.stderr

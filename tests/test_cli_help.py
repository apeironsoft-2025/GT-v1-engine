import os
import subprocess
import sys
from pathlib import Path


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


def test_cli_help_exits_zero() -> None:
    assert _run_cli("--help").returncode == 0


def test_validate_config_help_exits_zero() -> None:
    assert _run_cli("validate-config", "--help").returncode == 0


def test_validate_data_help_exits_zero() -> None:
    assert _run_cli("validate-data", "--help").returncode == 0


def test_show_defaults_help_exits_zero() -> None:
    assert _run_cli("show-defaults", "--help").returncode == 0


def test_list_indicators_help_exits_zero() -> None:
    assert _run_cli("list-indicators", "--help").returncode == 0


def test_validate_indicators_config_help_exits_zero() -> None:
    assert _run_cli("validate-indicators-config", "--help").returncode == 0


def test_run_indicators_help_exits_zero() -> None:
    assert _run_cli("run-indicators", "--help").returncode == 0


def test_backtest_indicator_help_exits_zero() -> None:
    assert _run_cli("backtest-indicator", "--help").returncode == 0


def test_backtest_all_indicators_help_exits_zero() -> None:
    assert _run_cli("backtest-all-indicators", "--help").returncode == 0


def test_backtest_rule171_help_exits_zero() -> None:
    assert _run_cli("backtest-rule171", "--help").returncode == 0


def test_run_smoke_pipeline_help_exits_zero() -> None:
    assert _run_cli("run-smoke-pipeline", "--help").returncode == 0


def test_compare_rule171_baseline_help_exits_zero() -> None:
    assert _run_cli("compare-rule171-baseline", "--help").returncode == 0

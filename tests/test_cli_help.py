import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gt_v1_engine.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help_exits_zero() -> None:
    assert _run_cli("--help").returncode == 0


def test_validate_config_help_exits_zero() -> None:
    assert _run_cli("validate-config", "--help").returncode == 0


def test_validate_data_help_exits_zero() -> None:
    assert _run_cli("validate-data", "--help").returncode == 0


def test_show_defaults_help_exits_zero() -> None:
    assert _run_cli("show-defaults", "--help").returncode == 0

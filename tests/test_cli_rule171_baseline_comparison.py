import json
import os
import subprocess
import sys
from pathlib import Path

from gt_v1_engine.baselines.rule171_baseline import get_rule171_old_baseline

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


def _write_summary(tmp_path: Path, released_delta: int = 0) -> Path:
    summary = {
        **get_rule171_old_baseline(),
        "live_trading_allowed": False,
        "broker_order_allowed": False,
        "validation_status": "PASS",
    }
    summary["released_signals"] += released_delta
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_compare_rule171_baseline_help_exits_zero() -> None:
    assert _run_cli("compare-rule171-baseline", "--help").returncode == 0


def test_cli_works_with_exact_baseline_summary(tmp_path: Path) -> None:
    report = tmp_path / "comparison.md"
    completed = _run_cli(
        "compare-rule171-baseline",
        "--summary-json",
        str(_write_summary(tmp_path)),
        "--output-report",
        str(report),
    )
    assert completed.returncode == 0, completed.stderr
    assert report.exists()
    assert "PASS" in completed.stdout


def test_cli_writes_optional_json(tmp_path: Path) -> None:
    report = tmp_path / "comparison.md"
    output_json = tmp_path / "comparison.json"
    completed = _run_cli(
        "compare-rule171-baseline",
        "--summary-json",
        str(_write_summary(tmp_path)),
        "--output-report",
        str(report),
        "--output-json",
        str(output_json),
    )
    assert completed.returncode == 0, completed.stderr
    assert output_json.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))["validation_status"] == "PASS"


def test_cli_mismatch_exits_zero_with_mismatch_status(tmp_path: Path) -> None:
    report = tmp_path / "comparison.md"
    output_json = tmp_path / "comparison.json"
    completed = _run_cli(
        "compare-rule171-baseline",
        "--summary-json",
        str(_write_summary(tmp_path, released_delta=1)),
        "--output-report",
        str(report),
        "--output-json",
        str(output_json),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "MISMATCH"
    assert "MISMATCH" in completed.stdout


def test_cli_missing_file_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    completed = _run_cli(
        "compare-rule171-baseline",
        "--summary-json",
        str(tmp_path / "missing.json"),
        "--output-report",
        str(tmp_path / "comparison.md"),
    )
    assert completed.returncode != 0
    assert "GT-v1-engine ERROR" in completed.stdout + completed.stderr

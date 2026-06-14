from pathlib import Path

import pytest

from gt_v1_engine.baselines.comparison import (
    compare_rule171_summary_to_baseline,
    write_rule171_baseline_comparison_report,
)
from gt_v1_engine.baselines.rule171_baseline import get_rule171_old_baseline
from gt_v1_engine.core.errors import DataValidationError


def _baseline_summary() -> dict:
    baseline = get_rule171_old_baseline()
    return {
        **baseline,
        "input_path": "input.csv",
        "output_csv_path": "rule171.csv",
        "output_summary_path": "summary.json",
        "live_trading_allowed": False,
        "broker_order_allowed": False,
        "validation_status": "PASS",
    }


def test_exact_baseline_summary_exact_match_true() -> None:
    comparison = compare_rule171_summary_to_baseline(_baseline_summary())
    assert comparison["exact_match"] is True


def test_exact_baseline_summary_tolerance_match_true() -> None:
    comparison = compare_rule171_summary_to_baseline(_baseline_summary())
    assert comparison["tolerance_match"] is True
    assert comparison["validation_status"] == "PASS"


def test_one_count_difference_exact_match_false() -> None:
    summary = _baseline_summary()
    summary["released_signals"] += 1
    comparison = compare_rule171_summary_to_baseline(summary, tolerance_count=1)
    assert comparison["exact_match"] is False


def test_one_count_difference_outside_tolerance_is_mismatch() -> None:
    summary = _baseline_summary()
    summary["released_signals"] += 1
    comparison = compare_rule171_summary_to_baseline(summary, tolerance_count=0)
    assert comparison["validation_status"] == "MISMATCH"
    assert comparison["tolerance_match"] is False


def test_one_pip_difference_within_tolerance_passes() -> None:
    summary = _baseline_summary()
    summary["total_realized_pips"] += 0.05
    comparison = compare_rule171_summary_to_baseline(summary, tolerance_pips=0.1)
    assert comparison["validation_status"] == "PASS"


def test_missing_metric_raises_clean_error() -> None:
    summary = _baseline_summary()
    summary.pop("released_signals")
    with pytest.raises(DataValidationError):
        compare_rule171_summary_to_baseline(summary)


def test_markdown_report_is_written(tmp_path: Path) -> None:
    summary = _baseline_summary()
    comparison = compare_rule171_summary_to_baseline(summary)
    output_path = tmp_path / "comparison.md"
    write_rule171_baseline_comparison_report(comparison, summary, output_path)
    assert output_path.exists()
    assert "Rule171 Baseline Comparison Report" in output_path.read_text(encoding="utf-8")


def test_comparison_includes_recommended_next_action() -> None:
    comparison = compare_rule171_summary_to_baseline(_baseline_summary())
    assert comparison["recommended_next_action"]


def test_safety_fields_are_research_only() -> None:
    comparison = compare_rule171_summary_to_baseline(_baseline_summary())
    assert comparison["production_activation_status"] == "NOT_ACTIVE"
    assert comparison["live_trading_allowed"] is False
    assert comparison["broker_order_allowed"] is False

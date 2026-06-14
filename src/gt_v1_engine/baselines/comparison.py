from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gt_v1_engine.baselines.rule171_baseline import get_rule171_old_baseline
from gt_v1_engine.core.constants import NOT_ACTIVE
from gt_v1_engine.core.errors import DataValidationError, IndicatorCalculationError
from gt_v1_engine.core.io_utils import ensure_parent_dir

COUNT_METRICS = {
    "released_signals",
    "buy_signals",
    "sell_signals",
    "win_close_count",
    "loss_close_count",
    "take_profit_closes",
    "stop_loss_closes",
    "twelve_hour_closes",
    "twelve_hour_win_closes",
    "twelve_hour_loss_closes",
}
PIP_METRICS = {"total_realized_pips", "average_pips_per_signal"}
RATE_METRICS = {"win_close_rate", "loss_close_rate"}
COMPARED_METRICS = [
    "released_signals",
    "buy_signals",
    "sell_signals",
    "win_close_count",
    "loss_close_count",
    "take_profit_closes",
    "stop_loss_closes",
    "twelve_hour_closes",
    "twelve_hour_win_closes",
    "twelve_hour_loss_closes",
    "total_realized_pips",
    "average_pips_per_signal",
    "win_close_rate",
    "loss_close_rate",
]

LIKELY_MISMATCH_REASON = (
    "Fresh GT-v1-engine indicators may not match the old Ghost Trader recovered "
    "TD/TS indicator columns exactly. Rule171 executor output can differ when "
    "indicator formulas or strength bucketing differ."
)
PASS_ACTION = "Fresh GT-v1-engine Rule171 output matches the old baseline within tolerance."
MISMATCH_ACTION = (
    "Investigate indicator parity first. Compare GT-v1-engine TD/TS columns with "
    "the old indicator-ready dataset before changing Rule171."
)


def compare_rule171_summary_to_baseline(
    current_summary: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    tolerance_pips: float = 0.1,
    tolerance_rate: float = 0.01,
    tolerance_count: int = 0,
) -> dict[str, Any]:
    baseline = baseline or get_rule171_old_baseline()
    _validate_summary_metrics(current_summary)

    metric_comparisons = [
        _compare_metric(
            metric,
            baseline[metric],
            current_summary[metric],
            _tolerance_for_metric(metric, tolerance_pips, tolerance_rate, tolerance_count),
        )
        for metric in COMPARED_METRICS
    ]
    exact_match = all(item["difference"] == 0 for item in metric_comparisons)
    tolerance_match = all(item["within_tolerance"] for item in metric_comparisons)
    validation_status = "PASS" if tolerance_match else "MISMATCH"

    return {
        "comparison_name": "Rule171 old baseline comparison",
        "baseline_name": baseline["baseline_name"],
        "pair": current_summary.get("pair", baseline.get("pair")),
        "timeframe": current_summary.get("timeframe", baseline.get("timeframe")),
        "start_datetime": current_summary.get("start_datetime", baseline.get("start_datetime")),
        "end_datetime": current_summary.get("end_datetime", baseline.get("end_datetime")),
        "validation_status": validation_status,
        "exact_match": exact_match,
        "tolerance_match": tolerance_match,
        "metric_comparisons": metric_comparisons,
        "largest_differences": sorted(
            metric_comparisons,
            key=lambda item: item["absolute_difference"],
            reverse=True,
        )[:5],
        "likely_mismatch_reason": LIKELY_MISMATCH_REASON,
        "recommended_next_action": PASS_ACTION if tolerance_match else MISMATCH_ACTION,
        "production_activation_status": NOT_ACTIVE,
        "live_trading_allowed": False,
        "broker_order_allowed": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def write_rule171_baseline_comparison_report(
    comparison: dict[str, Any],
    current_summary: dict[str, Any],
    output_path: Path,
) -> None:
    try:
        ensure_parent_dir(output_path)
        output_path.write_text(_render_report(comparison, current_summary), encoding="utf-8")
    except Exception as exc:
        raise IndicatorCalculationError(
            f"Failed to write Rule171 baseline comparison report: {exc}"
        ) from exc


def _validate_summary_metrics(current_summary: dict[str, Any]) -> None:
    missing = [metric for metric in COMPARED_METRICS if metric not in current_summary]
    if missing:
        raise DataValidationError(
            "Rule171 current summary missing required metric(s): " + ", ".join(missing)
        )


def _tolerance_for_metric(
    metric: str,
    tolerance_pips: float,
    tolerance_rate: float,
    tolerance_count: int,
) -> float:
    if metric in COUNT_METRICS:
        return float(tolerance_count)
    if metric in RATE_METRICS:
        return float(tolerance_rate)
    return float(tolerance_pips)


def _compare_metric(metric: str, baseline_value: Any, current_value: Any, tolerance: float) -> dict[str, Any]:
    baseline_numeric = float(baseline_value)
    current_numeric = float(current_value)
    difference = current_numeric - baseline_numeric
    absolute_difference = abs(difference)
    return {
        "metric": metric,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "difference": difference,
        "absolute_difference": absolute_difference,
        "tolerance": tolerance,
        "within_tolerance": absolute_difference <= tolerance,
    }


def _render_report(comparison: dict[str, Any], current_summary: dict[str, Any]) -> str:
    baseline = get_rule171_old_baseline()
    lines = [
        "# Rule171 Baseline Comparison Report",
        "",
        "## Comparison status",
        "",
        f"- exact_match: {str(comparison['exact_match']).lower()}",
        f"- tolerance_match: {str(comparison['tolerance_match']).lower()}",
        f"- validation_status: {comparison['validation_status']}",
        "",
        "## Run identity",
        "",
        f"- pair: {comparison['pair']}",
        f"- timeframe: {comparison['timeframe']}",
        f"- period: {comparison['start_datetime']} to {comparison['end_datetime']}",
        "",
        "## Baseline metrics",
        "",
        _metrics_table({metric: baseline[metric] for metric in COMPARED_METRICS}),
        "",
        "## Current metrics",
        "",
        _metrics_table({metric: current_summary[metric] for metric in COMPARED_METRICS}),
        "",
        "## Metric differences",
        "",
        "| metric | baseline | current | difference | tolerance | within tolerance |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in comparison["metric_comparisons"]:
        lines.append(
            f"| {item['metric']} | {item['baseline_value']} | {item['current_value']} | "
            f"{item['difference']} | {item['tolerance']} | {str(item['within_tolerance']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Largest differences",
            "",
            _metrics_table(
                {
                    item["metric"]: item["absolute_difference"]
                    for item in comparison["largest_differences"]
                },
                value_header="absolute_difference",
            ),
            "",
            "## Interpretation",
            "",
            comparison["likely_mismatch_reason"],
            "",
            "Rule171 executor behavior may be correct even if metrics differ, because indicator TD/TS generation can differ.",
            "",
            "## Recommended next action",
            "",
            comparison["recommended_next_action"],
            "",
            "## Safety",
            "",
            f"- production_activation_status: {comparison['production_activation_status']}",
            f"- live_trading_allowed: {str(comparison['live_trading_allowed']).lower()}",
            f"- broker_order_allowed: {str(comparison['broker_order_allowed']).lower()}",
            "- research only",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics_table(metrics: dict[str, Any], value_header: str = "value") -> str:
    lines = [f"| metric | {value_header} |", "|---|---:|"]
    for metric, value in metrics.items():
        lines.append(f"| {metric} | {value} |")
    return "\n".join(lines)

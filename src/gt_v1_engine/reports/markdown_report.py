from pathlib import Path
from typing import Any

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.io_utils import ensure_parent_dir


def write_smoke_pipeline_report(
    summary: dict[str, Any],
    indicator_summary: dict[str, Any],
    indicator_backtest_summary: dict[str, Any],
    rule171_summary: dict[str, Any],
    output_path: Path,
) -> None:
    try:
        ensure_parent_dir(output_path)
        output_path.write_text(
            _render_report(summary, indicator_summary, indicator_backtest_summary, rule171_summary),
            encoding="utf-8",
        )
    except Exception as exc:
        raise IndicatorCalculationError(f"Failed to write smoke pipeline report: {exc}") from exc


def _render_report(
    summary: dict[str, Any],
    indicator_summary: dict[str, Any],
    indicator_backtest_summary: dict[str, Any],
    rule171_summary: dict[str, Any],
) -> str:
    generated_columns = [
        column
        for column in indicator_summary.get("generated_columns", [])
        if column.endswith("_TD") or column.endswith("_TS")
    ]
    lines = [
        "# GT-v1-engine Smoke Pipeline Report",
        "",
        "## Run parameters",
        "",
        f"- pair: {summary['pair']}",
        f"- timeframe: {summary['timeframe']}",
        f"- period: {summary['start_datetime']} to {summary['end_datetime']}",
        f"- selected indicators: {', '.join(summary['selected_indicators'])}",
        f"- input path: {summary['input_path']}",
        "",
        "## Output files",
        "",
        f"- indicator CSV: {summary['indicator_dataset_csv']}",
        f"- indicator Parquet: {summary['indicator_dataset_parquet']}",
        f"- indicator summary JSON: {summary['indicator_summary_json']}",
        f"- individual indicator summary JSON: {summary['indicator_backtest_summary_json']}",
        f"- Rule171 CSV: {summary['rule171_output_csv']}",
        f"- Rule171 summary JSON: {summary['rule171_summary_json']}",
        "",
        "## Indicator executor summary",
        "",
        f"- row count: {indicator_summary.get('row_count')}",
        f"- first datetime: {indicator_summary.get('first_datetime')}",
        f"- last datetime: {indicator_summary.get('last_datetime')}",
        f"- generated TD/TS columns: {', '.join(generated_columns)}",
        f"- validation status: {indicator_summary.get('validation_status')}",
        "",
        "## Individual indicator backtest summary",
        "",
        f"- selected indicators: {', '.join(indicator_backtest_summary.get('selected_indicators', []))}",
        f"- best indicator by net pips: {indicator_backtest_summary.get('best_indicator_by_net_pips')}",
        f"- best indicator by win rate: {indicator_backtest_summary.get('best_indicator_by_win_rate')}",
        f"- validation status: {indicator_backtest_summary.get('validation_status')}",
        "",
        "## Rule171 summary",
        "",
        f"- released signals: {rule171_summary.get('released_signals')}",
        f"- BUY signals: {rule171_summary.get('buy_signals')}",
        f"- SELL signals: {rule171_summary.get('sell_signals')}",
        f"- WIN_CLOSE: {rule171_summary.get('win_close_count')}",
        f"- LOSS_CLOSE: {rule171_summary.get('loss_close_count')}",
        f"- total realized pips: {rule171_summary.get('total_realized_pips')}",
        f"- average pips per signal: {rule171_summary.get('average_pips_per_signal')}",
        f"- win close rate: {rule171_summary.get('win_close_rate')}",
        f"- blocked rows while open: {rule171_summary.get('blocked_rows_while_open')}",
        "",
        "## Safety",
        "",
        f"- production_activation_status: {summary['production_activation_status']}",
        f"- live_trading_allowed: {str(summary['live_trading_allowed']).lower()}",
        f"- broker_order_allowed: {str(summary['broker_order_allowed']).lower()}",
        "- research only",
        "",
    ]
    return "\n".join(lines)

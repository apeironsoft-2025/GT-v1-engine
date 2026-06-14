import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from gt_v1_engine import __version__
from gt_v1_engine.baselines.comparison import (
    compare_rule171_summary_to_baseline,
    write_rule171_baseline_comparison_report,
)
from gt_v1_engine.backtesting.indicator_backtester import (
    backtest_all_indicators,
    backtest_indicator,
)
from gt_v1_engine.core.errors import ConfigError, GTV1EngineError
from gt_v1_engine.core.io_utils import ensure_file_exists, write_json
from gt_v1_engine.core.paths import resolve_project_path
from gt_v1_engine.data.market_data_loader import load_market_data
from gt_v1_engine.indicators.executor import run_indicator_executor
from gt_v1_engine.indicators.registry import get_registered_indicators, validate_indicator_order
from gt_v1_engine.indicators.selection import load_default_indicator_config, parse_indicator_list
from gt_v1_engine.pipeline.smoke_pipeline import run_smoke_pipeline
from gt_v1_engine.rules.rule171 import Rule171ExecutionOverrides, run_rule171_backtest
from gt_v1_engine.rules.rule_config import Rule171Config, load_rule171_config

app = typer.Typer(help="GT-v1-engine research CLI.")
console = Console()


def _handle_cli_error(exc: Exception, debug: bool) -> None:
    if debug:
        raise exc
    if isinstance(exc, GTV1EngineError):
        console.print(f"[bold red][GT-v1-engine ERROR][/bold red] {exc.__class__.__name__}: {exc}")
        raise typer.Exit(code=1) from None
    console.print(f"[bold red][GT-v1-engine ERROR][/bold red] Unexpected error: {exc}")
    if debug:
        raise
    raise typer.Exit(code=1) from None


def _resolve(path: Path) -> Path:
    return resolve_project_path(path)


@app.command()
def version() -> None:
    """Print project version."""
    console.print(__version__)


@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(
        Path("configs/rules/rule171.yaml"),
        "--config",
        help="Path to Rule171 YAML config.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for unexpected errors."),
) -> None:
    """Validate Rule171 configuration."""
    try:
        rule_config = load_rule171_config(_resolve(config))
        console.print("[bold green]Rule171 config validation passed.[/bold green]")
        _print_config_summary(rule_config)
    except Exception as exc:
        _handle_cli_error(exc, debug)


@app.command("validate-data")
def validate_data(
    input_path: Path = typer.Option(
        Path("data/raw/USDJPY_M5.csv"),
        "--input",
        help="Path to CSV or parquet market data.",
    ),
    pair: str = typer.Option("USDJPY", "--pair", help="Market pair."),
    timeframe: str = typer.Option("M5", "--timeframe", help="Market timeframe."),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for unexpected errors."),
) -> None:
    """Validate and summarize market data."""
    try:
        df = load_market_data(_resolve(input_path))
        table = Table(title="Market Data Validation")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("pair", pair)
        table.add_row("timeframe", timeframe)
        table.add_row("row_count", str(len(df)))
        table.add_row("first DateTime", str(df["DateTime"].iloc[0]))
        table.add_row("last DateTime", str(df["DateTime"].iloc[-1]))
        table.add_row("columns", ", ".join(df.columns))
        console.print(table)
    except Exception as exc:
        _handle_cli_error(exc, debug)


@app.command("show-defaults")
def show_defaults(
    config: Path = typer.Option(
        Path("configs/rules/rule171.yaml"),
        "--config",
        help="Path to Rule171 YAML config.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for unexpected errors."),
) -> None:
    """Print Rule171 defaults from config."""
    try:
        rule_config = load_rule171_config(_resolve(config))
        _print_config_summary(rule_config)
    except Exception as exc:
        _handle_cli_error(exc, debug)


@app.command("list-indicators")
def list_indicators(
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """List registered indicators and framework status."""
    try:
        table = Table(title="Registered Indicators")
        table.add_column("Indicator")
        table.add_column("TD column")
        table.add_column("TS column")
        table.add_column("implemented")
        table.add_column("enabled")

        for metadata in get_registered_indicators().values():
            table.add_row(
                metadata.name,
                metadata.direction_column,
                metadata.strength_column,
                str(metadata.implemented).lower(),
                str(metadata.enabled).lower(),
            )
        console.print(table)
    except Exception as exc:
        _handle_cli_error(exc, debug)


@app.command("validate-indicators-config")
def validate_indicators_config(
    config: Path = typer.Option(
        Path("configs/indicators/default_indicators.yaml"),
        "--config",
        help="Path to indicator YAML config.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Validate indicator defaults and enabled indicator order."""
    try:
        indicator_config = load_default_indicator_config(_resolve(config))
        default_order = validate_indicator_order(
            indicator_config["default_order"],
            indicator_config["default_order"],
        )
        enabled = validate_indicator_order(
            indicator_config["enabled"],
            indicator_config["default_order"],
        )

        console.print("[bold green]Indicator config validation passed.[/bold green]")
        table = Table(title="Indicator Config")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("default order", ", ".join(default_order))
        table.add_row("enabled", ", ".join(enabled))
        console.print(table)
    except Exception as exc:
        _handle_cli_error(exc, debug)


@app.command("run-indicators")
def run_indicators(
    input_path: Path = typer.Option(..., "--input", help="Path to CSV or parquet market data."),
    pair: str = typer.Option(..., "--pair", help="Market pair."),
    timeframe: str = typer.Option(..., "--timeframe", help="Market timeframe."),
    indicators: str | None = typer.Option(
        None,
        "--indicators",
        help="Comma-separated indicators. Defaults to registry order.",
    ),
    output_csv: Path = typer.Option(..., "--output-csv", help="Output CSV path."),
    output_parquet: Path | None = typer.Option(None, "--output-parquet", help="Output parquet path."),
    summary_json: Path | None = typer.Option(None, "--summary-json", help="Summary JSON path."),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Run selected indicators over market data and write research output."""
    try:
        selected_indicators = parse_indicator_list(indicators)
        _, summary = run_indicator_executor(
            input_path=_resolve(input_path),
            pair=pair,
            timeframe=timeframe,
            indicators=selected_indicators,
            output_csv=_resolve(output_csv),
            output_parquet=_resolve(output_parquet) if output_parquet else None,
            summary_json=_resolve(summary_json) if summary_json else None,
        )
        _print_indicator_execution_summary(
            summary,
            _resolve(summary_json) if summary_json else None,
        )
    except Exception as exc:
        _handle_cli_error(exc, debug)


def _print_indicator_execution_summary(summary: dict, summary_json: Path | None) -> None:
    table = Table(title="Indicator Execution")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("pair", summary["pair"])
    table.add_row("timeframe", summary["timeframe"])
    table.add_row("row count", str(summary["row_count"]))
    table.add_row("selected indicators", ", ".join(summary["selected_indicators"]))
    table.add_row("output CSV", str(summary["output_csv_path"]))
    if summary.get("output_parquet_path"):
        table.add_row("output Parquet", str(summary["output_parquet_path"]))
    if summary_json is not None:
        table.add_row("summary JSON", str(summary_json))
    table.add_row("validation status", summary["validation_status"])
    console.print(table)


@app.command("backtest-indicator")
def backtest_indicator_command(
    input_path: Path = typer.Option(..., "--input", help="Path to indicator-ready CSV or parquet."),
    pair: str = typer.Option(..., "--pair", help="Market pair."),
    timeframe: str = typer.Option(..., "--timeframe", help="Market timeframe."),
    indicator: str = typer.Option(..., "--indicator", help="Indicator name."),
    start: str | None = typer.Option(None, "--start", help="Inclusive start datetime."),
    end: str | None = typer.Option(None, "--end", help="Inclusive end datetime."),
    horizon_candles: int = typer.Option(48, "--horizon-candles", help="Future candle horizon."),
    target_pips: float = typer.Option(30, "--target-pips", help="Take-profit distance in pips."),
    stop_pips: float = typer.Option(40, "--stop-pips", help="Stop-loss distance in pips."),
    pip_size: float | None = typer.Option(None, "--pip-size", help="Override pip size."),
    output_csv: Path = typer.Option(..., "--output-csv", help="Backtest output CSV path."),
    include_no_signal_rows: bool = typer.Option(
        False,
        "--include-no-signal-rows",
        help="Include NO_SIGNAL rows in output.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Backtest one indicator using High/Low TP/SL touch logic."""
    try:
        _, summary = backtest_indicator(
            input_path=_resolve(input_path),
            pair=pair,
            timeframe=timeframe,
            indicator=indicator,
            start=start,
            end=end,
            horizon_candles=horizon_candles,
            target_pips=target_pips,
            stop_pips=stop_pips,
            pip_size=pip_size,
            output_csv=_resolve(output_csv),
            include_no_signal_rows=include_no_signal_rows,
        )
        _print_backtest_indicator_summary(summary)
    except Exception as exc:
        _handle_cli_error(exc, debug)


@app.command("backtest-all-indicators")
def backtest_all_indicators_command(
    input_path: Path = typer.Option(..., "--input", help="Path to indicator-ready CSV or parquet."),
    pair: str = typer.Option(..., "--pair", help="Market pair."),
    timeframe: str = typer.Option(..., "--timeframe", help="Market timeframe."),
    indicators: str | None = typer.Option(
        None,
        "--indicators",
        help="Comma-separated indicators. Defaults to registry order.",
    ),
    start: str | None = typer.Option(None, "--start", help="Inclusive start datetime."),
    end: str | None = typer.Option(None, "--end", help="Inclusive end datetime."),
    horizon_candles: int = typer.Option(48, "--horizon-candles", help="Future candle horizon."),
    target_pips: float = typer.Option(30, "--target-pips", help="Take-profit distance in pips."),
    stop_pips: float = typer.Option(40, "--stop-pips", help="Stop-loss distance in pips."),
    pip_size: float | None = typer.Option(None, "--pip-size", help="Override pip size."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for per-indicator CSVs."),
    summary_json: Path = typer.Option(..., "--summary-json", help="Combined summary JSON path."),
    include_no_signal_rows: bool = typer.Option(
        False,
        "--include-no-signal-rows",
        help="Include NO_SIGNAL rows in output.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Backtest selected indicators and write one CSV per indicator."""
    try:
        selected_indicators = parse_indicator_list(indicators)
        summary = backtest_all_indicators(
            input_path=_resolve(input_path),
            pair=pair,
            timeframe=timeframe,
            indicators=selected_indicators,
            start=start,
            end=end,
            horizon_candles=horizon_candles,
            target_pips=target_pips,
            stop_pips=stop_pips,
            pip_size=pip_size,
            output_dir=_resolve(output_dir),
            summary_json=_resolve(summary_json),
            include_no_signal_rows=include_no_signal_rows,
        )
        _print_backtest_all_summary(summary, _resolve(summary_json))
    except Exception as exc:
        _handle_cli_error(exc, debug)


def _print_backtest_indicator_summary(summary: dict) -> None:
    table = Table(title="Indicator Backtest")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("pair", summary["pair"])
    table.add_row("timeframe", summary["timeframe"])
    table.add_row("indicator", summary["indicator"])
    table.add_row("tested trades", str(summary["tested_trades"]))
    table.add_row("wins", str(summary["win_count"]))
    table.add_row("losses", str(summary["loss_count"]))
    table.add_row("no hits", str(summary["no_hit_count"]))
    table.add_row("total realized pips", str(summary["total_realized_pips"]))
    table.add_row("output CSV", summary["output_csv_path"])
    table.add_row("validation status", summary["validation_status"])
    console.print(table)


def _print_backtest_all_summary(summary: dict, summary_json: Path) -> None:
    table = Table(title="All Indicator Backtests")
    table.add_column("Field")
    table.add_column("Value")
    totals = summary["indicator_summaries"].values()
    table.add_row("pair", summary["pair"])
    table.add_row("timeframe", summary["timeframe"])
    table.add_row("selected indicators", ", ".join(summary["selected_indicators"]))
    table.add_row("tested trades", str(sum(item["tested_trades"] for item in totals)))
    table.add_row("wins", str(sum(item["win_count"] for item in summary["indicator_summaries"].values())))
    table.add_row("losses", str(sum(item["loss_count"] for item in summary["indicator_summaries"].values())))
    table.add_row("no hits", str(sum(item["no_hit_count"] for item in summary["indicator_summaries"].values())))
    table.add_row(
        "total realized pips",
        str(sum(item["total_realized_pips"] for item in summary["indicator_summaries"].values())),
    )
    table.add_row("summary JSON", str(summary_json))
    table.add_row("validation status", summary["validation_status"])
    console.print(table)


@app.command("backtest-rule171")
def backtest_rule171_command(
    input_path: Path = typer.Option(..., "--input", help="Path to indicator-ready CSV or parquet."),
    config: Path = typer.Option(
        Path("configs/rules/rule171.yaml"),
        "--config",
        help="Path to Rule171 YAML config.",
    ),
    pair: str | None = typer.Option(None, "--pair", help="Market pair override."),
    timeframe: str | None = typer.Option(None, "--timeframe", help="Market timeframe override."),
    indicators: str | None = typer.Option(None, "--indicators", help="Comma-separated indicators."),
    start: str | None = typer.Option(None, "--start", help="Inclusive start datetime override."),
    end: str | None = typer.Option(None, "--end", help="Inclusive end datetime override."),
    pip_size: float | None = typer.Option(None, "--pip-size", help="Pip size override."),
    strength_threshold: float | None = typer.Option(
        None,
        "--strength-threshold",
        help="Strength threshold override.",
    ),
    entry_confirmation_required: int | None = typer.Option(
        None,
        "--entry-confirmation-required",
        help="Confirmation count override.",
    ),
    take_profit_pips: float | None = typer.Option(
        None,
        "--take-profit-pips",
        help="Take-profit pips override.",
    ),
    stop_loss_pips: float | None = typer.Option(
        None,
        "--stop-loss-pips",
        help="Stop-loss pips override.",
    ),
    max_holding_candles: int | None = typer.Option(
        None,
        "--max-holding-candles",
        help="Max holding candles override.",
    ),
    output_csv: Path = typer.Option(..., "--output-csv", help="Rule171 output CSV path."),
    output_summary: Path = typer.Option(..., "--output-summary", help="Rule171 summary JSON path."),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Backtest Rule171 over an indicator-ready dataset."""
    try:
        selected_indicators = parse_indicator_list(indicators) if indicators is not None else None
        _, summary = run_rule171_backtest(
            input_path=_resolve(input_path),
            config_path=_resolve(config),
            output_csv=_resolve(output_csv),
            output_summary=_resolve(output_summary),
            overrides=Rule171ExecutionOverrides(
                pair=pair,
                timeframe=timeframe,
                indicators=selected_indicators,
                start=start,
                end=end,
                pip_size=pip_size,
                strength_threshold=strength_threshold,
                entry_confirmation_required=entry_confirmation_required,
                take_profit_pips=take_profit_pips,
                stop_loss_pips=stop_loss_pips,
                max_holding_candles=max_holding_candles,
            ),
        )
        _print_rule171_summary(summary)
    except Exception as exc:
        _handle_cli_error(exc, debug)


def _print_rule171_summary(summary: dict) -> None:
    table = Table(title="Rule171 Backtest")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("rule name", summary["rule_name"])
    table.add_row("pair", summary["pair"])
    table.add_row("timeframe", summary["timeframe"])
    table.add_row("selected indicators", ", ".join(summary["selected_indicators"]))
    table.add_row("period", f"{summary['start_datetime']} to {summary['end_datetime']}")
    table.add_row("released signals", str(summary["released_signals"]))
    table.add_row("wins", str(summary["win_close_count"]))
    table.add_row("losses", str(summary["loss_close_count"]))
    table.add_row("net pips", str(summary["total_realized_pips"]))
    table.add_row("win rate", str(summary["win_close_rate"]))
    table.add_row("blocked rows", str(summary["blocked_rows_while_open"]))
    table.add_row("output CSV", summary["output_csv_path"])
    table.add_row("output summary", summary["output_summary_path"])
    table.add_row("validation status", summary["validation_status"])
    console.print(table)


@app.command("run-smoke-pipeline")
def run_smoke_pipeline_command(
    input_path: Path = typer.Option(..., "--input", help="Path to raw OHLC CSV or parquet."),
    pair: str = typer.Option(..., "--pair", help="Market pair."),
    timeframe: str = typer.Option(..., "--timeframe", help="Market timeframe."),
    start: str | None = typer.Option(None, "--start", help="Inclusive start datetime."),
    end: str | None = typer.Option(None, "--end", help="Inclusive end datetime."),
    indicators: str | None = typer.Option(
        "MACD,RSI,ADX,ATR,BOLLINGER,EMA_STACK",
        "--indicators",
        help="Comma-separated indicators.",
    ),
    config: Path = typer.Option(
        Path("configs/rules/rule171.yaml"),
        "--config",
        help="Path to Rule171 YAML config.",
    ),
    output_root: Path = typer.Option(Path("data"), "--output-root", help="Output root directory."),
    horizon_candles: int = typer.Option(48, "--horizon-candles", help="Indicator backtest horizon."),
    indicator_target_pips: float = typer.Option(
        30,
        "--indicator-target-pips",
        help="Indicator backtest take-profit pips.",
    ),
    indicator_stop_pips: float = typer.Option(
        40,
        "--indicator-stop-pips",
        help="Indicator backtest stop-loss pips.",
    ),
    rule171_pip_size: float | None = typer.Option(None, "--rule171-pip-size", help="Rule171 pip size."),
    rule171_strength_threshold: float | None = typer.Option(
        None,
        "--rule171-strength-threshold",
        help="Rule171 strength threshold.",
    ),
    rule171_entry_confirmation_required: int | None = typer.Option(
        None,
        "--rule171-entry-confirmation-required",
        help="Rule171 confirmation count.",
    ),
    rule171_take_profit_pips: float | None = typer.Option(
        None,
        "--rule171-take-profit-pips",
        help="Rule171 take-profit pips.",
    ),
    rule171_stop_loss_pips: float | None = typer.Option(
        None,
        "--rule171-stop-loss-pips",
        help="Rule171 stop-loss pips.",
    ),
    rule171_max_holding_candles: int | None = typer.Option(
        None,
        "--rule171-max-holding-candles",
        help="Rule171 max holding candles.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Run the full research smoke pipeline and write a validation report."""
    try:
        selected_indicators = parse_indicator_list(indicators)
        summary = run_smoke_pipeline(
            input_path=_resolve(input_path),
            pair=pair,
            timeframe=timeframe,
            start=start,
            end=end,
            indicators=selected_indicators,
            config_path=_resolve(config),
            output_root=_resolve(output_root),
            horizon_candles=horizon_candles,
            indicator_target_pips=indicator_target_pips,
            indicator_stop_pips=indicator_stop_pips,
            rule171_pip_size=rule171_pip_size,
            rule171_strength_threshold=rule171_strength_threshold,
            rule171_entry_confirmation_required=rule171_entry_confirmation_required,
            rule171_take_profit_pips=rule171_take_profit_pips,
            rule171_stop_loss_pips=rule171_stop_loss_pips,
            rule171_max_holding_candles=rule171_max_holding_candles,
        )
        _print_smoke_pipeline_summary(summary)
    except Exception as exc:
        _handle_cli_error(exc, debug)


def _print_smoke_pipeline_summary(summary: dict) -> None:
    table = Table(title="Smoke Pipeline")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("pipeline status", summary["validation_status"])
    table.add_row("pair/timeframe", f"{summary['pair']} {summary['timeframe']}")
    table.add_row("selected indicators", ", ".join(summary["selected_indicators"]))
    table.add_row("indicator CSV", summary["indicator_dataset_csv"])
    table.add_row("all-indicator summary JSON", summary["indicator_backtest_summary_json"])
    table.add_row("Rule171 CSV", summary["rule171_output_csv"])
    table.add_row("Rule171 summary JSON", summary["rule171_summary_json"])
    table.add_row("report path", summary["markdown_report_path"])
    table.add_row("Rule171 released signals", str(summary.get("rule171_released_signals", "")))
    table.add_row("Rule171 net pips", str(summary.get("rule171_total_realized_pips", "")))
    table.add_row("validation status", summary["validation_status"])
    console.print(table)


@app.command("compare-rule171-baseline")
def compare_rule171_baseline_command(
    summary_json: Path = typer.Option(..., "--summary-json", help="Rule171 summary JSON path."),
    output_report: Path = typer.Option(..., "--output-report", help="Markdown comparison report path."),
    output_json: Path | None = typer.Option(None, "--output-json", help="Optional comparison JSON path."),
    tolerance_pips: float = typer.Option(0.1, "--tolerance-pips", help="Pip metric tolerance."),
    tolerance_rate: float = typer.Option(0.01, "--tolerance-rate", help="Rate metric tolerance."),
    tolerance_count: int = typer.Option(0, "--tolerance-count", help="Count metric tolerance."),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Compare a Rule171 summary JSON with the old baseline metrics."""
    try:
        current_summary = _read_json(_resolve(summary_json))
        comparison = compare_rule171_summary_to_baseline(
            current_summary,
            tolerance_pips=tolerance_pips,
            tolerance_rate=tolerance_rate,
            tolerance_count=tolerance_count,
        )
        resolved_report = _resolve(output_report)
        write_rule171_baseline_comparison_report(comparison, current_summary, resolved_report)
        resolved_output_json = _resolve(output_json) if output_json else None
        if resolved_output_json is not None:
            write_json(resolved_output_json, comparison)
        _print_rule171_baseline_comparison_summary(comparison, resolved_report, resolved_output_json)
    except Exception as exc:
        _handle_cli_error(exc, debug)


def _read_json(path: Path) -> dict:
    resolved = ensure_file_exists(path)
    try:
        with resolved.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{resolved} must contain a JSON object")
    return payload


def _print_rule171_baseline_comparison_summary(
    comparison: dict,
    output_report: Path,
    output_json: Path | None,
) -> None:
    table = Table(title="Rule171 Baseline Comparison")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("comparison status", comparison["validation_status"])
    table.add_row("exact match", str(comparison["exact_match"]).lower())
    table.add_row("tolerance match", str(comparison["tolerance_match"]).lower())
    largest = ", ".join(
        f"{item['metric']}={item['absolute_difference']}"
        for item in comparison["largest_differences"]
    )
    table.add_row("largest differences", largest)
    table.add_row("report path", str(output_report))
    if output_json is not None:
        table.add_row("output JSON", str(output_json))
    table.add_row("recommended next action", comparison["recommended_next_action"])
    console.print(table)


def _print_config_summary(rule_config: Rule171Config) -> None:
    table = Table(title="Rule171 Defaults")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("rule name", rule_config.rule_name)
    table.add_row("pair", rule_config.market.default_pair)
    table.add_row("timeframe", rule_config.market.default_timeframe)
    table.add_row("selected indicators", ", ".join(rule_config.indicators.selected))
    table.add_row("strength threshold", str(rule_config.entry.strength_threshold))
    table.add_row("confirmation required", str(rule_config.entry.entry_confirmation_required))
    table.add_row("TP", str(rule_config.trade_management.take_profit_pips))
    table.add_row("SL", str(rule_config.trade_management.stop_loss_pips))
    table.add_row("max holding candles", str(rule_config.trade_management.max_holding_candles))
    table.add_row("production status", rule_config.production_activation_status)
    table.add_row("live trading allowed", str(rule_config.safety.live_trading_allowed))
    table.add_row("broker order allowed", str(rule_config.safety.broker_order_allowed))
    console.print(table)


if __name__ == "__main__":
    app()

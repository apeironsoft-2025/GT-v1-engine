from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from gt_v1_engine import __version__
from gt_v1_engine.core.errors import GTV1EngineError
from gt_v1_engine.core.paths import resolve_project_path
from gt_v1_engine.data.market_data_loader import load_market_data
from gt_v1_engine.indicators.registry import get_registered_indicators, validate_indicator_order
from gt_v1_engine.indicators.selection import load_default_indicator_config
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

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from gt_v1_engine import __version__
from gt_v1_engine.core.errors import GTV1EngineError
from gt_v1_engine.core.paths import resolve_project_path
from gt_v1_engine.data.market_data_loader import load_market_data
from gt_v1_engine.rules.rule_config import Rule171Config, load_rule171_config

app = typer.Typer(help="GT-v1-engine research CLI.")
console = Console()


def _handle_cli_error(exc: Exception, debug: bool) -> None:
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

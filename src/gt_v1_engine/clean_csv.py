from pathlib import Path

import typer
from rich.console import Console

from gt_v1_engine.core.errors import GTV1EngineError
from gt_v1_engine.data.raw_csv_cleaner import clean_raw_market_csv

console = Console()


def main(
    input_path: Path = typer.Option(..., "--input", help="Path to raw market CSV."),
    output_path: Path = typer.Option(..., "--output", help="Path for cleaned CSV output."),
    summary_path: Path = typer.Option(..., "--summary", help="Path for cleaning summary JSON."),
    debug: bool = typer.Option(False, "--debug", help="Show traceback for errors."),
) -> None:
    """Clean and validate a raw shared-storage market CSV."""
    try:
        summary = clean_raw_market_csv(input_path, output_path, summary_path)
        console.print("[bold green]CSV cleaning completed.[/bold green]")
        console.print(f"status: {summary['status']}")
        console.print(f"cleaned_row_count: {summary['cleaned_row_count']}")
        console.print(f"output_path: {summary['output_path']}")
        console.print(f"summary_path: {summary['summary_path']}")
    except Exception as exc:
        if debug:
            raise
        if isinstance(exc, GTV1EngineError):
            console.print(f"[bold red][GT-v1-engine ERROR][/bold red] {exc.__class__.__name__}: {exc}")
        else:
            console.print(f"[bold red][GT-v1-engine ERROR][/bold red] Unexpected error: {exc}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    typer.run(main)

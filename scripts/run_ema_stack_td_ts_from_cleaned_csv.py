import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EMA_PERIODS = (20, 50, 100, 200)
DEFAULT_CLEANED_ROOT_PATH = r"F:\GT-v1-shared-storage\cleaned"
DEFAULT_OUTPUT_DIR = r"F:\GT-v1-shared-storage\indicators"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EMA Stack TD/TS CSV from a cleaned CSV file."
    )

    parser.add_argument(
        "--file-name",
        required=True,
        help="Cleaned CSV file name only. Example: USDJPY_M5_cleaned.csv",
    )

    parser.add_argument(
        "--cleaned-root-path",
        default=DEFAULT_CLEANED_ROOT_PATH,
        help="Directory containing cleaned CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for EMA Stack indicator CSV.",
    )

    return parser.parse_args()


def validate_file_name(file_name: str) -> None:
    invalid_tokens = ("..", "/", "\\")
    if any(token in file_name for token in invalid_tokens):
        raise ValueError(
            "--file-name must be a file name only, not a path. "
            "Rejecting values containing '..', '/', or '\\'."
        )
    if not file_name.strip():
        raise ValueError("--file-name must not be empty.")


def find_close_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if str(column).lower() == "close":
            return str(column)
    available_columns = ", ".join(str(column) for column in df.columns)
    raise ValueError(
        "Close column is missing. Expected one of Close, close, or CLOSE. "
        f"Available columns: {available_columns}"
    )


def bucket_ema_stack_strength(raw_strength: float, trend_direction: str) -> float:
    if trend_direction == "NO_SIGNAL" or pd.isna(raw_strength) or raw_strength <= 0:
        return 0.0
    if raw_strength < 0.50:
        return 0.25
    if raw_strength < 1.00:
        return 0.50
    if raw_strength < 1.50:
        return 0.75
    return 1.0


def output_file_name(file_name: str) -> str:
    if file_name.endswith("_cleaned.csv"):
        return file_name.replace("_cleaned.csv", "_ema_stack_td_ts.csv")
    return f"{Path(file_name).stem}_ema_stack_td_ts.csv"


def calculate_ema_stack_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    close_column = find_close_column(df)
    close = pd.to_numeric(df[close_column], errors="coerce")
    if close.notna().sum() == 0:
        raise ValueError(
            f"Close column '{close_column}' could not be converted to numeric values."
        )

    result = df.copy()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_100 = close.ewm(span=100, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()

    result["EMA_20"] = ema_20
    result["EMA_50"] = ema_50
    result["EMA_100"] = ema_100
    result["EMA_200"] = ema_200

    up_stack = (ema_20 > ema_50) & (ema_50 > ema_100) & (ema_100 > ema_200)
    down_stack = (ema_20 < ema_50) & (ema_50 < ema_100) & (ema_100 < ema_200)
    result["EMA_STACK_TD"] = np.select(
        [up_stack, down_stack],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )

    spread_20_50 = (ema_20 - ema_50).abs()
    spread_50_100 = (ema_50 - ema_100).abs()
    spread_100_200 = (ema_100 - ema_200).abs()
    stack_spread = (spread_20_50 + spread_50_100 + spread_100_200) / 3
    rolling_avg_stack_spread = stack_spread.rolling(window=50, min_periods=1).mean()
    raw_strength = (stack_spread / rolling_avg_stack_spread).replace(
        [np.inf, -np.inf], np.nan
    )

    result["EMA_STACK_TS"] = [
        bucket_ema_stack_strength(strength, trend_direction)
        for strength, trend_direction in zip(raw_strength, result["EMA_STACK_TD"])
    ]
    result["EMA_STACK_TS"] = result["EMA_STACK_TS"].astype(float)
    return result


def main() -> None:
    args = parse_args()
    validate_file_name(args.file_name)

    cleaned_root_path = Path(args.cleaned_root_path)
    output_dir = Path(args.output_dir)
    input_csv = cleaned_root_path / args.file_name
    output_csv = output_dir / output_file_name(args.file_name)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory could not be created: {output_dir}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    result = calculate_ema_stack_td_ts(df)
    result.to_csv(output_csv, index=False)
    if not output_csv.exists():
        raise FileNotFoundError(f"Output CSV was not written: {output_csv}")

    print("EMA Stack TD/TS generation completed.")
    print(f"Input CSV : {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows      : {len(result)}")
    print()
    print("EMA_STACK_TD count:")
    print(result["EMA_STACK_TD"].value_counts(dropna=False).to_string())
    print()
    print("EMA_STACK_TS count:")
    print(result["EMA_STACK_TS"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()

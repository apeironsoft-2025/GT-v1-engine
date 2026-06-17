import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MACD TD/TS CSV from a cleaned CSV file."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Absolute path to cleaned input CSV. Example: F:/GT-v1-shared-storage/cleaned/EURUSD_M5_cleaned.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=r"F:\GT-v1-shared-storage\indicators",
        help="Output directory for MACD indicator CSV.",
    )

    return parser.parse_args()


def find_close_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if str(column).lower() == "close":
            return str(column)
    available_columns = ", ".join(str(column) for column in df.columns)
    raise ValueError(
        "Close column is missing. Expected one of Close, close, or CLOSE. "
        f"Available columns: {available_columns}"
    )


def bucket_macd_strength(raw_strength: float) -> float:
    if pd.isna(raw_strength) or raw_strength <= 0:
        return 0.0
    if raw_strength < 0.50:
        return 0.25
    if raw_strength < 1.00:
        return 0.50
    if raw_strength < 1.50:
        return 0.75
    return 1.0


def calculate_macd_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    close_column = find_close_column(df)
    close = pd.to_numeric(df[close_column], errors="coerce")
    if close.notna().sum() == 0:
        raise ValueError(f"Close column '{close_column}' could not be converted to numeric values.")

    result = df.copy()
    ema_fast = close.ewm(span=FAST_PERIOD, adjust=False).mean()
    ema_slow = close.ewm(span=SLOW_PERIOD, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=SIGNAL_PERIOD, adjust=False).mean()
    histogram = macd_line - signal_line

    result["MACD_LINE"] = macd_line
    result["MACD_SIGNAL"] = signal_line
    result["MACD_HIST"] = histogram
    result["MACD_TD"] = np.select(
        [histogram > 0, histogram < 0],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )

    abs_hist = histogram.abs()
    rolling_avg_abs_hist = abs_hist.rolling(window=50, min_periods=1).mean()
    raw_strength = (abs_hist / rolling_avg_abs_hist).replace([np.inf, -np.inf], np.nan)
    result["MACD_TS"] = raw_strength.map(bucket_macd_strength).astype(float)
    return result


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).to_absolute() if hasattr(Path(args.input_csv), "to_absolute") else Path(args.input_csv).absolute()
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = input_csv.stem

    if input_stem.endswith("_cleaned"):
        output_stem = input_stem.replace("_cleaned", "_macd_td_ts")
    else:
        output_stem = f"{input_stem}_macd_td_ts"

    output_csv = output_dir / f"{output_stem}.csv"

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    result = calculate_macd_td_ts(df)
    result.to_csv(output_csv, index=False)
    if not output_csv.exists():
        raise FileNotFoundError(f"Output CSV was not written: {output_csv}")

    print("MACD TD/TS CSV generated")
    print(f"Input CSV : {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Row count: {len(result)}")
    print("MACD_TD value counts:")
    print(result["MACD_TD"].value_counts(dropna=False).to_string())
    print("MACD_TS value counts:")
    print(result["MACD_TS"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()

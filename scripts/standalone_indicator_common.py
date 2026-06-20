import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CLEANED_ROOT_PATH = r"F:\GT-v1-shared-storage\cleaned"
DEFAULT_OUTPUT_DIR = r"F:\GT-v1-shared-storage\indicators"
REQUIRED_COLUMNS = ["DateTime", "Open", "High", "Low", "Close"]
TD_VALUES = {"UP", "DOWN", "NO_SIGNAL"}
TS_VALUES = {0.0, 0.25, 0.5, 0.75, 1.0}


def parse_file_name_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--file-name",
        "--fileName",
        dest="fileName",
        required=True,
        help="Cleaned CSV file name only. Example: USDJPY_M5_cleaned.csv",
    )
    parser.add_argument(
        "--cleaned-root-path",
        dest="cleanedRootPath",
        default=DEFAULT_CLEANED_ROOT_PATH,
        help="Directory containing cleaned CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        dest="outputDir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for indicator CSV.",
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


def input_csv_path(file_name: str, cleaned_root_path: str = DEFAULT_CLEANED_ROOT_PATH) -> Path:
    validate_file_name(file_name)
    return Path(cleaned_root_path) / file_name


def output_csv_path(
    file_name: str,
    indicator_name: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Path:
    validate_file_name(file_name)
    if file_name.endswith("_cleaned.csv"):
        output_file_name = file_name.replace(
            "_cleaned.csv", f"_{indicator_name}_td_ts.csv"
        )
    else:
        output_file_name = f"{Path(file_name).stem}_{indicator_name}_td_ts.csv"
    return Path(output_dir) / output_file_name


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        available_columns = ", ".join(str(column) for column in df.columns)
        raise ValueError(
            "Input CSV missing required columns: "
            + ", ".join(missing_columns)
            + f". Required columns: {', '.join(REQUIRED_COLUMNS)}. "
            + f"Available columns: {available_columns}"
        )


def load_cleaned_csv(
    file_name: str,
    cleaned_root_path: str = DEFAULT_CLEANED_ROOT_PATH,
) -> tuple[Path, pd.DataFrame]:
    input_csv = input_csv_path(file_name, cleaned_root_path)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")
    validate_required_columns(df)
    return input_csv, df


def base_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df)
    return df[REQUIRED_COLUMNS].copy()


def numeric_ohlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    validate_required_columns(df)
    open_price = pd.to_numeric(df["Open"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    if close.notna().sum() == 0:
        raise ValueError("Close column could not be converted to numeric values.")
    return open_price, high, low, close


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)


def rolling_quantile_strength(
    values: pd.Series,
    signal_mask: pd.Series,
    window: int = 100,
) -> pd.Series:
    q40 = values.rolling(window=window, min_periods=1).quantile(0.40)
    q60 = values.rolling(window=window, min_periods=1).quantile(0.60)
    q80 = values.rolling(window=window, min_periods=1).quantile(0.80)

    strengths = pd.Series(0.0, index=values.index, dtype=float)
    valid_signal = values.notna() & signal_mask.fillna(False)
    strengths.loc[valid_signal] = 1.0
    strengths.loc[valid_signal & (values <= q80)] = 0.75
    strengths.loc[valid_signal & (values <= q60)] = 0.5
    strengths.loc[valid_signal & (values <= q40)] = 0.25
    return strengths


def write_indicator_csv(
    result: pd.DataFrame,
    file_name: str,
    indicator_name: str,
    td_column: str,
    ts_column: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Path:
    output_csv = output_csv_path(file_name, indicator_name, output_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    invalid_td = set(result[td_column].dropna().unique()) - TD_VALUES
    invalid_ts = set(float(value) for value in result[ts_column].dropna().unique()) - TS_VALUES
    if invalid_td:
        raise ValueError(f"{td_column} contains invalid values: {sorted(invalid_td)}")
    if invalid_ts:
        raise ValueError(f"{ts_column} contains invalid values: {sorted(invalid_ts)}")

    result.to_csv(output_csv, index=False)
    if not output_csv.exists():
        raise FileNotFoundError(f"Output CSV was not written: {output_csv}")
    return output_csv


def print_success(
    input_csv: Path,
    output_csv: Path,
    result: pd.DataFrame,
    td_column: str,
    ts_column: str,
) -> None:
    print(f"Input CSV : {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Row count : {len(result)}")
    print(f"{td_column} value counts:")
    print(result[td_column].value_counts(dropna=False).to_string())
    print(f"{ts_column} value counts:")
    print(result[ts_column].value_counts(dropna=False).sort_index().to_string())
    print("status SUCCESS")

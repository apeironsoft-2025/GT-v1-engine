import numpy as np
import pandas as pd

from standalone_indicator_common import (
    base_output_frame,
    load_cleaned_csv,
    numeric_ohlc,
    parse_file_name_args,
    print_success,
    write_indicator_csv,
)


BOLLINGER_PERIOD = 20
STD_MULTIPLIER = 2
TD_COLUMN = "BOLLINGER_TD"
TS_COLUMN = "BOLLINGER_TS"
INDICATOR_NAME = "bollinger"


def bucket_bollinger_strength(distance_ratio: float, trend_direction: str) -> float:
    if trend_direction == "NO_SIGNAL" or pd.isna(distance_ratio) or distance_ratio <= 0:
        return 0.0
    if distance_ratio < 0.25:
        return 0.25
    if distance_ratio < 0.5:
        return 0.5
    if distance_ratio < 1.0:
        return 0.75
    return 1.0


def calculate_bollinger_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, _, _, close = numeric_ohlc(df)
    middle = close.rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).mean()
    rolling_std = close.rolling(window=BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).std()
    upper = middle + (STD_MULTIPLIER * rolling_std)
    lower = middle - (STD_MULTIPLIER * rolling_std)
    half_width = (upper - lower) / 2
    distance_ratio = ((close - middle).abs() / half_width).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = base_output_frame(df)
    result[TD_COLUMN] = np.select(
        [close > middle, close < middle],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    result.loc[middle.isna() | half_width.isna() | (half_width == 0), TD_COLUMN] = "NO_SIGNAL"
    result[TS_COLUMN] = [
        bucket_bollinger_strength(value, trend_direction)
        for value, trend_direction in zip(distance_ratio, result[TD_COLUMN])
    ]
    result[TS_COLUMN] = result[TS_COLUMN].astype(float)
    return result


def main() -> None:
    args = parse_file_name_args("Generate Bollinger TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName, args.cleanedRootPath)
    result = calculate_bollinger_td_ts(df)
    output_csv = write_indicator_csv(
        result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN, args.outputDir
    )
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

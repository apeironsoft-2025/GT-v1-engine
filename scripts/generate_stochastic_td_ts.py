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


K_PERIOD = 14
D_PERIOD = 3
SMOOTH_K = 3
TD_COLUMN = "STOCHASTIC_TD"
TS_COLUMN = "STOCHASTIC_TS"
INDICATOR_NAME = "stochastic"


def bucket_stochastic_strength(slow_k: float, trend_direction: str) -> float:
    if trend_direction == "NO_SIGNAL" or pd.isna(slow_k):
        return 0.0
    if trend_direction == "UP":
        if slow_k < 40:
            return 0.25
        if slow_k < 60:
            return 0.5
        if slow_k < 80:
            return 0.75
        return 1.0
    if slow_k <= 20:
        return 1.0
    if slow_k <= 40:
        return 0.75
    if slow_k <= 60:
        return 0.5
    return 0.25


def calculate_stochastic_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, high, low, close = numeric_ohlc(df)
    lowest_low = low.rolling(window=K_PERIOD, min_periods=K_PERIOD).min()
    highest_high = high.rolling(window=K_PERIOD, min_periods=K_PERIOD).max()
    raw_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    raw_k = raw_k.replace([np.inf, -np.inf], np.nan)
    slow_k = raw_k.rolling(window=SMOOTH_K, min_periods=SMOOTH_K).mean()
    slow_d = slow_k.rolling(window=D_PERIOD, min_periods=D_PERIOD).mean()

    previous_slow_k = slow_k.shift(1)
    cross_above_20 = (previous_slow_k < 20) & (slow_k >= 20)
    cross_below_80 = (previous_slow_k > 80) & (slow_k <= 80)
    up_signal = cross_above_20 | (~cross_below_80 & (slow_k > slow_d) & (slow_k >= 50))
    down_signal = cross_below_80 | (~cross_above_20 & (slow_k < slow_d) & (slow_k <= 50))

    result = base_output_frame(df)
    result[TD_COLUMN] = np.select(
        [up_signal, down_signal],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    result.loc[slow_k.isna() | slow_d.isna(), TD_COLUMN] = "NO_SIGNAL"
    result[TS_COLUMN] = [
        bucket_stochastic_strength(value, trend_direction)
        for value, trend_direction in zip(slow_k, result[TD_COLUMN])
    ]
    result[TS_COLUMN] = result[TS_COLUMN].astype(float)
    return result


def main() -> None:
    args = parse_file_name_args("Generate Stochastic TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName, args.cleanedRootPath)
    result = calculate_stochastic_td_ts(df)
    output_csv = write_indicator_csv(
        result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN, args.outputDir
    )
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

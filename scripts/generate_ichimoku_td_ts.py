import numpy as np
import pandas as pd

from standalone_indicator_common import (
    base_output_frame,
    load_cleaned_csv,
    numeric_ohlc,
    parse_file_name_args,
    print_success,
    rolling_quantile_strength,
    write_indicator_csv,
)


CONVERSION_PERIOD = 9
BASELINE_PERIOD = 26
SPAN_B_PERIOD = 52
DISPLACEMENT = 1
TD_COLUMN = "ICHIMOKU_TD"
TS_COLUMN = "ICHIMOKU_TS"
INDICATOR_NAME = "ichimoku"


def midpoint(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return (highest_high + lowest_low) / 2


def calculate_ichimoku_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, high, low, close = numeric_ohlc(df)
    conversion = midpoint(high, low, CONVERSION_PERIOD)
    baseline = midpoint(high, low, BASELINE_PERIOD)
    span_a = ((conversion + baseline) / 2).shift(DISPLACEMENT)
    span_b = midpoint(high, low, SPAN_B_PERIOD).shift(DISPLACEMENT)
    _ = (span_a, span_b)

    previous_conversion = conversion.shift(1)
    previous_baseline = baseline.shift(1)
    cross_above = (previous_conversion <= previous_baseline) & (conversion > baseline)
    cross_below = (previous_conversion >= previous_baseline) & (conversion < baseline)
    up_signal = cross_above | (~cross_below & (conversion > baseline))
    down_signal = cross_below | (~cross_above & (conversion < baseline))

    result = base_output_frame(df)
    result[TD_COLUMN] = np.select(
        [up_signal, down_signal],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    result.loc[conversion.isna() | baseline.isna(), TD_COLUMN] = "NO_SIGNAL"
    distance_pct = ((conversion - baseline).abs() / close).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    result[TS_COLUMN] = rolling_quantile_strength(
        distance_pct,
        result[TD_COLUMN] != "NO_SIGNAL",
        window=100,
    )
    return result


def main() -> None:
    args = parse_file_name_args("Generate Ichimoku TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName, args.cleanedRootPath)
    result = calculate_ichimoku_td_ts(df)
    output_csv = write_indicator_csv(
        result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN, args.outputDir
    )
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

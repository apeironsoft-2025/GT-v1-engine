import numpy as np
import pandas as pd

from standalone_indicator_common import (
    base_output_frame,
    calculate_atr,
    load_cleaned_csv,
    numeric_ohlc,
    parse_file_name_args,
    print_success,
    write_indicator_csv,
)


STEP = 0.02
INCREMENT = 0.02
MAXIMUM = 0.2
TD_COLUMN = "PARABOLIC_SAR_TD"
TS_COLUMN = "PARABOLIC_SAR_TS"
INDICATOR_NAME = "parabolic_sar"


def calculate_parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    sar = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) < 2:
        return sar

    uptrend = bool(close.iloc[1] >= close.iloc[0])
    acceleration = STEP
    extreme_point = high.iloc[0] if uptrend else low.iloc[0]
    sar.iloc[1] = low.iloc[0] if uptrend else high.iloc[0]

    for index in range(2, len(close)):
        previous_sar = sar.iloc[index - 1]
        current_sar = previous_sar + acceleration * (extreme_point - previous_sar)

        if uptrend:
            current_sar = min(current_sar, low.iloc[index - 1], low.iloc[index - 2])
            if low.iloc[index] < current_sar:
                uptrend = False
                current_sar = extreme_point
                extreme_point = low.iloc[index]
                acceleration = STEP
            elif high.iloc[index] > extreme_point:
                extreme_point = high.iloc[index]
                acceleration = min(acceleration + INCREMENT, MAXIMUM)
        else:
            current_sar = max(current_sar, high.iloc[index - 1], high.iloc[index - 2])
            if high.iloc[index] > current_sar:
                uptrend = True
                current_sar = extreme_point
                extreme_point = high.iloc[index]
                acceleration = STEP
            elif low.iloc[index] < extreme_point:
                extreme_point = low.iloc[index]
                acceleration = min(acceleration + INCREMENT, MAXIMUM)

        sar.iloc[index] = current_sar

    return sar


def bucket_distance_atr(distance_atr: float, trend_direction: str) -> float:
    if trend_direction == "NO_SIGNAL" or pd.isna(distance_atr):
        return 0.0
    if distance_atr < 0.5:
        return 0.25
    if distance_atr < 1.0:
        return 0.5
    if distance_atr < 1.5:
        return 0.75
    return 1.0


def calculate_parabolic_sar_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, high, low, close = numeric_ohlc(df)
    parabolic_sar = calculate_parabolic_sar(high, low, close)
    atr = calculate_atr(high, low, close, 14)
    previous_close = close.shift(1)
    previous_sar = parabolic_sar.shift(1)

    cross_above = (previous_close <= previous_sar) & (close > parabolic_sar)
    cross_below = (previous_close >= previous_sar) & (close < parabolic_sar)
    above_sar = close > parabolic_sar
    below_sar = close < parabolic_sar

    result = base_output_frame(df)
    result[TD_COLUMN] = np.select(
        [cross_above | (~cross_below & above_sar), cross_below | (~cross_above & below_sar)],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    result.loc[parabolic_sar.isna(), TD_COLUMN] = "NO_SIGNAL"
    distance_atr = ((close - parabolic_sar).abs() / atr).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    result[TS_COLUMN] = [
        bucket_distance_atr(value, trend_direction)
        for value, trend_direction in zip(distance_atr, result[TD_COLUMN])
    ]
    result[TS_COLUMN] = result[TS_COLUMN].astype(float)
    return result


def main() -> None:
    args = parse_file_name_args("Generate Parabolic SAR TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName, args.cleanedRootPath)
    result = calculate_parabolic_sar_td_ts(df)
    output_csv = write_indicator_csv(
        result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN, args.outputDir
    )
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

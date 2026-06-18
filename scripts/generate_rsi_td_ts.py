import numpy as np
import pandas as pd

from standalone_indicator_common import (
    base_output_frame,
    load_cleaned_csv,
    numeric_ohlc,
    parse_file_name_args,
    print_success,
    wilder_smooth,
    write_indicator_csv,
)


RSI_PERIOD = 14
TD_COLUMN = "RSI_TD"
TS_COLUMN = "RSI_TS"
INDICATOR_NAME = "rsi"


def bucket_rsi_strength(rsi: float, trend_direction: str) -> float:
    if trend_direction == "NO_SIGNAL" or pd.isna(rsi):
        return 0.0
    if rsi >= 70 or rsi <= 30:
        return 1.0
    if rsi >= 65 or rsi <= 35:
        return 0.75
    if rsi >= 60 or rsi <= 40:
        return 0.5
    return 0.25


def calculate_rsi_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, _, _, close = numeric_ohlc(df)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = wilder_smooth(gain, RSI_PERIOD)
    average_loss = wilder_smooth(loss, RSI_PERIOD)
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(average_loss != 0, 100.0)
    rsi = rsi.where(average_gain != 0, 0.0)

    result = base_output_frame(df)
    result[TD_COLUMN] = np.select(
        [rsi >= 55, rsi <= 45],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    result.loc[rsi.isna(), TD_COLUMN] = "NO_SIGNAL"
    result[TS_COLUMN] = [
        bucket_rsi_strength(value, trend_direction)
        for value, trend_direction in zip(rsi, result[TD_COLUMN])
    ]
    result[TS_COLUMN] = result[TS_COLUMN].astype(float)
    return result


def main() -> None:
    args = parse_file_name_args("Generate RSI TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName)
    result = calculate_rsi_td_ts(df)
    output_csv = write_indicator_csv(result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN)
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

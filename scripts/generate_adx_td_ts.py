import numpy as np
import pandas as pd

from standalone_indicator_common import (
    base_output_frame,
    load_cleaned_csv,
    numeric_ohlc,
    parse_file_name_args,
    print_success,
    true_range,
    wilder_smooth,
    write_indicator_csv,
)


ADX_PERIOD = 14
TD_COLUMN = "ADX_TD"
TS_COLUMN = "ADX_TS"
INDICATOR_NAME = "adx"


def bucket_adx_strength(adx: float, trend_direction: str) -> float:
    if trend_direction == "NO_SIGNAL" or pd.isna(adx):
        return 0.0
    if adx < 25:
        return 0.25
    if adx < 30:
        return 0.5
    if adx < 40:
        return 0.75
    return 1.0


def calculate_adx_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, high, low, close = numeric_ohlc(df)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    atr = wilder_smooth(true_range(high, low, close), ADX_PERIOD)
    plus_di = 100 * (wilder_smooth(plus_dm, ADX_PERIOD) / atr)
    minus_di = 100 * (wilder_smooth(minus_dm, ADX_PERIOD) / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = wilder_smooth(dx.replace([np.inf, -np.inf], np.nan), ADX_PERIOD)

    result = base_output_frame(df)
    up_signal = (plus_di > minus_di) & (adx >= 20)
    down_signal = (minus_di > plus_di) & (adx >= 20)
    result[TD_COLUMN] = np.select(
        [up_signal, down_signal],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    result[TS_COLUMN] = [
        bucket_adx_strength(value, trend_direction)
        for value, trend_direction in zip(adx, result[TD_COLUMN])
    ]
    result[TS_COLUMN] = result[TS_COLUMN].astype(float)
    return result


def main() -> None:
    args = parse_file_name_args("Generate ADX TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName, args.cleanedRootPath)
    result = calculate_adx_td_ts(df)
    output_csv = write_indicator_csv(
        result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN, args.outputDir
    )
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

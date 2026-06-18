import numpy as np
import pandas as pd

from standalone_indicator_common import (
    base_output_frame,
    calculate_atr,
    load_cleaned_csv,
    numeric_ohlc,
    parse_file_name_args,
    print_success,
    rolling_quantile_strength,
    write_indicator_csv,
)


ATR_PERIOD = 14
TD_COLUMN = "ATR_TD"
TS_COLUMN = "ATR_TS"
INDICATOR_NAME = "atr"


def calculate_atr_td_ts(df: pd.DataFrame) -> pd.DataFrame:
    _, high, low, close = numeric_ohlc(df)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    previous_close = close.shift(1)

    result = base_output_frame(df)
    result[TD_COLUMN] = np.select(
        [(close > previous_close) & atr.notna(), (close < previous_close) & atr.notna()],
        ["UP", "DOWN"],
        default="NO_SIGNAL",
    )
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
    result[TS_COLUMN] = rolling_quantile_strength(
        atr_pct,
        result[TD_COLUMN] != "NO_SIGNAL",
        window=100,
    )
    return result


def main() -> None:
    args = parse_file_name_args("Generate ATR TD/TS CSV from a cleaned CSV file.")
    input_csv, df = load_cleaned_csv(args.fileName)
    result = calculate_atr_td_ts(df)
    output_csv = write_indicator_csv(result, args.fileName, INDICATOR_NAME, TD_COLUMN, TS_COLUMN)
    print_success(input_csv, output_csv, result, TD_COLUMN, TS_COLUMN)


if __name__ == "__main__":
    main()

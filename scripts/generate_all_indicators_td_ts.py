import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from generate_adx_td_ts import (
    INDICATOR_NAME as ADX_NAME,
    TD_COLUMN as ADX_TD,
    TS_COLUMN as ADX_TS,
    calculate_adx_td_ts,
)
from generate_atr_td_ts import (
    INDICATOR_NAME as ATR_NAME,
    TD_COLUMN as ATR_TD,
    TS_COLUMN as ATR_TS,
    calculate_atr_td_ts,
)
from generate_bollinger_td_ts import (
    INDICATOR_NAME as BOLLINGER_NAME,
    TD_COLUMN as BOLLINGER_TD,
    TS_COLUMN as BOLLINGER_TS,
    calculate_bollinger_td_ts,
)
from generate_ichimoku_td_ts import (
    INDICATOR_NAME as ICHIMOKU_NAME,
    TD_COLUMN as ICHIMOKU_TD,
    TS_COLUMN as ICHIMOKU_TS,
    calculate_ichimoku_td_ts,
)
from generate_parabolic_sar_td_ts import (
    INDICATOR_NAME as PARABOLIC_SAR_NAME,
    TD_COLUMN as PARABOLIC_SAR_TD,
    TS_COLUMN as PARABOLIC_SAR_TS,
    calculate_parabolic_sar_td_ts,
)
from generate_rsi_td_ts import (
    INDICATOR_NAME as RSI_NAME,
    TD_COLUMN as RSI_TD,
    TS_COLUMN as RSI_TS,
    calculate_rsi_td_ts,
)
from generate_stochastic_td_ts import (
    INDICATOR_NAME as STOCHASTIC_NAME,
    TD_COLUMN as STOCHASTIC_TD,
    TS_COLUMN as STOCHASTIC_TS,
    calculate_stochastic_td_ts,
)
from run_ema_stack_td_ts_from_cleaned_csv import (
    calculate_ema_stack_td_ts,
    output_file_name as ema_stack_output_file_name,
)
from run_macd_td_ts_from_cleaned_csv import calculate_macd_td_ts
from standalone_indicator_common import (
    DEFAULT_CLEANED_ROOT_PATH,
    DEFAULT_OUTPUT_DIR,
    load_cleaned_csv,
    validate_file_name,
    write_indicator_csv,
)


@dataclass(frozen=True)
class IndicatorJob:
    display_name: str
    indicator_name: str
    td_column: str
    ts_column: str
    calculate: Callable[[pd.DataFrame], pd.DataFrame]


STANDALONE_JOBS = (
    IndicatorJob("ADX", ADX_NAME, ADX_TD, ADX_TS, calculate_adx_td_ts),
    IndicatorJob("ATR", ATR_NAME, ATR_TD, ATR_TS, calculate_atr_td_ts),
    IndicatorJob(
        "Bollinger", BOLLINGER_NAME, BOLLINGER_TD, BOLLINGER_TS, calculate_bollinger_td_ts
    ),
    IndicatorJob(
        "Ichimoku", ICHIMOKU_NAME, ICHIMOKU_TD, ICHIMOKU_TS, calculate_ichimoku_td_ts
    ),
    IndicatorJob(
        "Parabolic SAR",
        PARABOLIC_SAR_NAME,
        PARABOLIC_SAR_TD,
        PARABOLIC_SAR_TS,
        calculate_parabolic_sar_td_ts,
    ),
    IndicatorJob("RSI", RSI_NAME, RSI_TD, RSI_TS, calculate_rsi_td_ts),
    IndicatorJob(
        "Stochastic",
        STOCHASTIC_NAME,
        STOCHASTIC_TD,
        STOCHASTIC_TS,
        calculate_stochastic_td_ts,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all TD/TS indicator CSV files from one cleaned CSV file."
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
        help="Output directory for all indicator CSV files.",
    )
    return parser.parse_args()


def macd_output_file_name(file_name: str) -> str:
    input_stem = Path(file_name).stem
    if input_stem.endswith("_cleaned"):
        return f"{input_stem.replace('_cleaned', '_macd_td_ts')}.csv"
    return f"{input_stem}_macd_td_ts.csv"


def write_full_indicator_csv(
    result: pd.DataFrame,
    output_csv: Path,
    td_column: str,
    ts_column: str,
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    if not output_csv.exists():
        raise FileNotFoundError(f"Output CSV was not written: {output_csv}")
    if td_column not in result.columns or ts_column not in result.columns:
        raise ValueError(f"Missing expected columns: {td_column}, {ts_column}")
    return output_csv


def print_counts(result: pd.DataFrame, td_column: str, ts_column: str) -> None:
    print(f"{td_column} count:")
    print(result[td_column].value_counts(dropna=False).to_string())
    print(f"{ts_column} count:")
    print(result[ts_column].value_counts(dropna=False).sort_index().to_string())


def main() -> None:
    args = parse_args()
    validate_file_name(args.file_name)

    output_dir = Path(args.output_dir)
    input_csv, df = load_cleaned_csv(args.file_name, args.cleaned_root_path)
    outputs: list[tuple[str, Path, int]] = []

    print("All indicator TD/TS generation started.")
    print(f"Input CSV : {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Rows      : {len(df)}")
    print()

    ema_result = calculate_ema_stack_td_ts(df)
    ema_output = write_full_indicator_csv(
        ema_result,
        output_dir / ema_stack_output_file_name(args.file_name),
        "EMA_STACK_TD",
        "EMA_STACK_TS",
    )
    outputs.append(("EMA Stack", ema_output, len(ema_result)))
    print(f"Generated EMA Stack: {ema_output}")
    print_counts(ema_result, "EMA_STACK_TD", "EMA_STACK_TS")
    print()

    macd_result = calculate_macd_td_ts(df)
    macd_output = write_full_indicator_csv(
        macd_result,
        output_dir / macd_output_file_name(args.file_name),
        "MACD_TD",
        "MACD_TS",
    )
    outputs.append(("MACD", macd_output, len(macd_result)))
    print(f"Generated MACD: {macd_output}")
    print_counts(macd_result, "MACD_TD", "MACD_TS")
    print()

    for job in STANDALONE_JOBS:
        result = job.calculate(df)
        output_csv = write_indicator_csv(
            result,
            args.file_name,
            job.indicator_name,
            job.td_column,
            job.ts_column,
            args.output_dir,
        )
        outputs.append((job.display_name, output_csv, len(result)))
        print(f"Generated {job.display_name}: {output_csv}")
        print_counts(result, job.td_column, job.ts_column)
        print()

    print("All indicator TD/TS generation completed.")
    for display_name, output_csv, row_count in outputs:
        print(f"{display_name}: {output_csv} ({row_count} rows)")


if __name__ == "__main__":
    main()

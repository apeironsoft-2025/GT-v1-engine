"""Rule-GT-TD-4-8-TLPSL-30-30 is a TD-only 9-indicator rule.

It opens one call only when there is no active call and the 9 required *_TD columns match one of the accepted BUY or SELL patterns.

BUY_CALL open condition:
Case 1: All 9 required *_TD columns are UP.
Case 2: Exactly 8 required *_TD columns are UP and the remaining 1 required *_TD column is DOWN or NO_SIGNAL.
NO_DIRECTION is not allowed for opening.

SELL_CALL open condition:
Case 1: All 9 required *_TD columns are DOWN.
Case 2: Exactly 8 required *_TD columns are DOWN and the remaining 1 required *_TD column is UP or NO_SIGNAL.
NO_DIRECTION is not allowed for opening.

It closes an open call by TP, SL, or TD opposition logic.

TP/SL price distances:
SELL_CALL TP = entry_price - Low >= 0.300
BUY_CALL TP = High - entry_price >= 0.300
SELL_CALL SL = High - entry_price >= 0.300
BUY_CALL SL = entry_price - Low >= 0.300

TD opposition close:
For BUY_CALL, close when at least 4 of the 9 *_TD columns are DOWN or NO_DIRECTION.
For SELL_CALL, close when at least 4 of the 9 *_TD columns are UP or NO_DIRECTION.

Only one open call is allowed at a time.
Rows while a call is open are used only for close management.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RULE_NAME = "Rule-GT-TD-4-8-TLPSL-30-30"
RULE_DESCRIPTION = (
    "Rule-GT-TD-4-8-TLPSL-30-30 is a TD-only 9-indicator rule. "
    "It opens BUY_CALL when all 9 required TD columns are UP, or when "
    "exactly 8 required TD columns are UP and the remaining TD column is "
    "DOWN or NO_SIGNAL. It opens SELL_CALL when all 9 required TD columns "
    "are DOWN, or when exactly 8 required TD columns are DOWN and the "
    "remaining TD column is UP or NO_SIGNAL. It closes by TP, SL, or TD "
    "opposition logic with a 4-of-9 opposition threshold."
)
SCRIPT_RELATIVE_PATH = "scripts\\rules\\run_rule_gt_td_4_8_tlpsl_30_30.py"
REQUIRED_OHLC_COLUMNS = ["DateTime", "Open", "High", "Low", "Close"]
REQUIRED_TD_COLUMNS = [
    "ADX_TD",
    "ATR_TD",
    "BOLLINGER_TD",
    "EMA_STACK_TD",
    "ICHIMOKU_TD",
    "MACD_TD",
    "PARABOLIC_SAR_TD",
    "RSI_TD",
    "STOCHASTIC_TD",
]
ACCEPTED_TD_VALUES = {"UP", "DOWN", "NO_DIRECTION", "NO_SIGNAL"}
TD_OPPOSITION_THRESHOLD = 4
BUY_TP_DISTANCE = 0.300
BUY_SL_DISTANCE = 0.300
SELL_TP_DISTANCE = 0.300
SELL_SL_DISTANCE = 0.300
BUY_TP_PIPS = 30.0
BUY_SL_PIPS = 30.0
SELL_TP_PIPS = 30.0
SELL_SL_PIPS = 30.0
PRICE_EPSILON = 1e-9

TRACE_COLUMNS = [
    "rule_name",
    "source_file_name",
    "call_sequence",
    "call_type",
    "open_datetime",
    "open_price",
    "close_datetime",
    "close_price",
    "close_reason",
    "result",
    "pip_count",
    "signed_pips",
    "open_row_index",
    "close_row_index",
    "holding_rows",
    "td_opposition_count",
    "td_up_count_at_open",
    "td_down_count_at_open",
    "td_no_direction_count_at_open",
    "td_no_signal_count_at_open",
    "td_up_count_at_close",
    "td_down_count_at_close",
    "td_no_direction_count_at_close",
    "td_no_signal_count_at_close",
    *REQUIRED_TD_COLUMNS,
]

SUMMARY_FIELDS = [
    "rule_name",
    "total_call_count",
    "total_buy_call_tp_hit_count",
    "total_sell_call_tp_hit_count",
    "total_buy_call_sl_hit_count",
    "total_sell_call_sl_hit_count",
    "total_close_loss_count",
    "total_close_win_count",
    "total_buy_call_tp_hit_pips",
    "total_sell_call_tp_hit_pips",
    "total_buy_call_sl_hit_pips",
    "total_sell_call_sl_hit_pips",
    "total_close_loss_pips",
    "total_close_win_pips",
    "total_win_pip",
    "total_loss_pip",
    "net_pip",
    "is_profitable",
]


@dataclass
class OpenCall:
    sequence: int
    call_type: str
    open_datetime: Any
    open_price: float
    open_row_index: int
    open_td_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Backtest {RULE_NAME}.")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--experiments-root-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def validate_file_name(file_name: str) -> None:
    if not file_name or not file_name.strip():
        raise ValueError("--file-name must not be empty.")
    if any(token in file_name for token in ("..", "/", "\\")):
        raise ValueError(
            "--file-name must be a file name only, not a path. "
            "Rejecting values containing '..', '/', or '\\'."
        )


def build_input_path(experiments_root_path: str, file_name: str) -> Path:
    validate_file_name(file_name)
    return Path(experiments_root_path) / file_name


def rule_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.name == RULE_NAME:
        return output_path
    return output_path / RULE_NAME


def output_paths(input_path: Path, output_dir: str | Path) -> tuple[Path, Path, Path]:
    rule_dir = rule_output_dir(output_dir)
    calls_csv = rule_dir / f"{input_path.stem}_{RULE_NAME}_CALLS.csv"
    summary_json = rule_dir / f"{RULE_NAME}_SUMMARY.json"
    markdown_path = rule_dir.parent / f"{RULE_NAME}.md"
    return calls_csv, summary_json, markdown_path


def normalize_td_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().upper()


def validate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input CSV is empty.")

    missing_ohlc = [column for column in REQUIRED_OHLC_COLUMNS if column not in df.columns]
    if missing_ohlc:
        raise ValueError(
            "Input CSV missing required OHLC columns: "
            + ", ".join(missing_ohlc)
            + f". Required columns: {', '.join(REQUIRED_OHLC_COLUMNS)}."
        )

    missing_td = [column for column in REQUIRED_TD_COLUMNS if column not in df.columns]
    if missing_td:
        raise ValueError(
            "Input CSV missing required TD columns: "
            + ", ".join(missing_td)
            + f". Required columns: {', '.join(REQUIRED_TD_COLUMNS)}."
        )

    validated = df.copy()
    for column in ["Open", "High", "Low", "Close"]:
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any():
            bad_count = int(numeric.isna().sum())
            raise ValueError(f"Invalid numeric OHLC values in column {column}: {bad_count} row(s).")
        validated[column] = numeric.astype(float)

    for column in REQUIRED_TD_COLUMNS:
        validated[column] = validated[column].map(normalize_td_value)

    return validated


def td_counts_from_values(values: list[str]) -> dict[str, int]:
    return {
        "UP": sum(1 for value in values if value == "UP"),
        "DOWN": sum(1 for value in values if value == "DOWN"),
        "NO_DIRECTION": sum(1 for value in values if value == "NO_DIRECTION"),
        "NO_SIGNAL": sum(1 for value in values if value == "NO_SIGNAL"),
    }


def td_counts(row: pd.Series) -> dict[str, int]:
    return td_counts_from_values([normalize_td_value(row[column]) for column in REQUIRED_TD_COLUMNS])


def open_call_type(td_values: list[str]) -> str | None:
    if any(value not in ACCEPTED_TD_VALUES for value in td_values):
        return None
    counts = td_counts_from_values(td_values)
    if counts["NO_DIRECTION"] != 0:
        return None
    if counts["UP"] == 9 and counts["DOWN"] == 0 and counts["NO_SIGNAL"] == 0:
        return "BUY_CALL"
    if counts["UP"] == 8 and counts["DOWN"] + counts["NO_SIGNAL"] == 1:
        return "BUY_CALL"
    if counts["DOWN"] == 9 and counts["UP"] == 0 and counts["NO_SIGNAL"] == 0:
        return "SELL_CALL"
    if counts["DOWN"] == 8 and counts["UP"] + counts["NO_SIGNAL"] == 1:
        return "SELL_CALL"
    return None


def opposition_count(call_type: str, row: pd.Series) -> int:
    values = [normalize_td_value(row[column]) for column in REQUIRED_TD_COLUMNS]
    if call_type == "BUY_CALL":
        return sum(1 for value in values if value in {"DOWN", "NO_DIRECTION"})
    return sum(1 for value in values if value in {"UP", "NO_DIRECTION"})


def calculate_signed_pips(call_type: str, entry_price: float, close_price: float) -> float:
    if call_type == "BUY_CALL":
        return (close_price - entry_price) * 100
    return (entry_price - close_price) * 100


def rounded(value: float) -> float:
    return round(float(value), 6)


def rounded_pip(value: float) -> float:
    return round(float(value), 1)


def result_from_reason(close_reason: str, signed_pips: float) -> str:
    if close_reason in {"TP_HIT", "CLOSE_WIN"}:
        return "WIN_CALL"
    if close_reason in {"SL_HIT", "CLOSE_LOSS"}:
        return "LOSS_CALL"
    return "WIN_CALL" if signed_pips > 0 else "LOSS_CALL"


def close_reason_for_td(call_type: str, entry_price: float, close_price: float) -> str:
    if call_type == "BUY_CALL":
        return "CLOSE_WIN" if close_price > entry_price else "CLOSE_LOSS"
    return "CLOSE_WIN" if close_price < entry_price else "CLOSE_LOSS"


def trace_record(
    source_file_name: str,
    open_call: OpenCall,
    close_row: pd.Series,
    close_row_index: int,
    close_price: float,
    close_reason: str,
    signed_pips: float,
    td_opposition_count: int,
) -> dict[str, Any]:
    close_counts = td_counts(close_row)
    record = {
        "rule_name": RULE_NAME,
        "source_file_name": source_file_name,
        "call_sequence": open_call.sequence,
        "call_type": open_call.call_type,
        "open_datetime": open_call.open_datetime,
        "open_price": rounded(open_call.open_price),
        "close_datetime": close_row["DateTime"],
        "close_price": rounded(close_price),
        "close_reason": close_reason,
        "result": result_from_reason(close_reason, signed_pips),
        "pip_count": rounded(abs(signed_pips)),
        "signed_pips": rounded(signed_pips),
        "open_row_index": open_call.open_row_index,
        "close_row_index": close_row_index,
        "holding_rows": close_row_index - open_call.open_row_index,
        "td_opposition_count": td_opposition_count,
        "td_up_count_at_open": open_call.open_td_counts["UP"],
        "td_down_count_at_open": open_call.open_td_counts["DOWN"],
        "td_no_direction_count_at_open": open_call.open_td_counts["NO_DIRECTION"],
        "td_no_signal_count_at_open": open_call.open_td_counts["NO_SIGNAL"],
        "td_up_count_at_close": close_counts["UP"],
        "td_down_count_at_close": close_counts["DOWN"],
        "td_no_direction_count_at_close": close_counts["NO_DIRECTION"],
        "td_no_signal_count_at_close": close_counts["NO_SIGNAL"],
    }
    for column in REQUIRED_TD_COLUMNS:
        record[column] = normalize_td_value(close_row[column])
    return record


def evaluate_tp_sl(open_call: OpenCall, row: pd.Series) -> tuple[str, float] | None:
    high = float(row["High"])
    low = float(row["Low"])
    entry = open_call.open_price

    if open_call.call_type == "BUY_CALL":
        tp_hit = high - entry >= BUY_TP_DISTANCE - PRICE_EPSILON
        sl_hit = entry - low >= BUY_SL_DISTANCE - PRICE_EPSILON
    else:
        tp_hit = entry - low >= SELL_TP_DISTANCE - PRICE_EPSILON
        sl_hit = high - entry >= SELL_SL_DISTANCE - PRICE_EPSILON

    if tp_hit and sl_hit:
        return "SL_HIT", -BUY_SL_PIPS if open_call.call_type == "BUY_CALL" else -SELL_SL_PIPS
    if tp_hit:
        return "TP_HIT", BUY_TP_PIPS if open_call.call_type == "BUY_CALL" else SELL_TP_PIPS
    if sl_hit:
        return "SL_HIT", -BUY_SL_PIPS if open_call.call_type == "BUY_CALL" else -SELL_SL_PIPS
    return None


def tp_sl_close_price(open_call: OpenCall, close_reason: str) -> float:
    if close_reason == "TP_HIT" and open_call.call_type == "BUY_CALL":
        return open_call.open_price + BUY_TP_DISTANCE
    if close_reason == "TP_HIT":
        return open_call.open_price - SELL_TP_DISTANCE
    if open_call.call_type == "BUY_CALL":
        return open_call.open_price - BUY_SL_DISTANCE
    return open_call.open_price + SELL_SL_DISTANCE


def run_backtest(df: pd.DataFrame, source_file_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = validate_input_frame(df)
    calls: list[dict[str, Any]] = []
    open_call: OpenCall | None = None
    next_sequence = 1

    for row_index, row in data.reset_index(drop=True).iterrows():
        close = float(row["Close"])

        if open_call is None:
            td_values = [row[column] for column in REQUIRED_TD_COLUMNS]
            call_type = open_call_type(td_values)
            if call_type is None:
                continue
            open_call = OpenCall(
                sequence=next_sequence,
                call_type=call_type,
                open_datetime=row["DateTime"],
                open_price=close,
                open_row_index=int(row_index),
                open_td_counts=td_counts_from_values(td_values),
            )
            next_sequence += 1
            continue

        tp_sl_result = evaluate_tp_sl(open_call, row)
        if tp_sl_result is not None:
            close_reason, signed_pips = tp_sl_result
            calls.append(
                trace_record(
                    source_file_name,
                    open_call,
                    row,
                    int(row_index),
                    tp_sl_close_price(open_call, close_reason),
                    close_reason,
                    signed_pips,
                    opposition_count(open_call.call_type, row),
                )
            )
            open_call = None
            continue

        td_opposition_count = opposition_count(open_call.call_type, row)
        if td_opposition_count >= TD_OPPOSITION_THRESHOLD:
            signed_pips = calculate_signed_pips(open_call.call_type, open_call.open_price, close)
            close_reason = close_reason_for_td(open_call.call_type, open_call.open_price, close)
            calls.append(
                trace_record(
                    source_file_name,
                    open_call,
                    row,
                    int(row_index),
                    close,
                    close_reason,
                    signed_pips,
                    td_opposition_count,
                )
            )
            open_call = None

    if open_call is not None:
        final_row_index = len(data) - 1
        final_row = data.iloc[final_row_index]
        final_close = float(final_row["Close"])
        signed_pips = calculate_signed_pips(open_call.call_type, open_call.open_price, final_close)
        calls.append(
            trace_record(
                source_file_name,
                open_call,
                final_row,
                final_row_index,
                final_close,
                "FORCE_CLOSE_END_OF_FILE",
                signed_pips,
                opposition_count(open_call.call_type, final_row),
            )
        )

    calls_df = pd.DataFrame(calls, columns=TRACE_COLUMNS)
    summary = build_summary(calls_df)
    return calls_df, summary


def build_summary(calls_df: pd.DataFrame) -> dict[str, Any]:
    if calls_df.empty:
        calls_df = pd.DataFrame(columns=TRACE_COLUMNS)

    buy_calls = calls_df[calls_df["call_type"] == "BUY_CALL"]
    sell_calls = calls_df[calls_df["call_type"] == "SELL_CALL"]
    close_win = calls_df[calls_df["close_reason"] == "CLOSE_WIN"]
    close_loss = calls_df[calls_df["close_reason"] == "CLOSE_LOSS"]

    buy_tp_pips = buy_calls[buy_calls["close_reason"] == "TP_HIT"]["pip_count"].sum()
    sell_tp_pips = sell_calls[sell_calls["close_reason"] == "TP_HIT"]["pip_count"].sum()
    buy_sl_pips = buy_calls[buy_calls["close_reason"] == "SL_HIT"]["pip_count"].sum()
    sell_sl_pips = sell_calls[sell_calls["close_reason"] == "SL_HIT"]["pip_count"].sum()
    close_win_pips = close_win["pip_count"].sum()
    close_loss_pips = close_loss["pip_count"].sum()

    total_win_pip = buy_tp_pips + sell_tp_pips + close_win_pips
    total_loss_pip = buy_sl_pips + sell_sl_pips + close_loss_pips
    net_pip = total_win_pip - total_loss_pip

    summary = {
        "rule_name": RULE_NAME,
        "total_call_count": int(len(calls_df)),
        "total_buy_call_tp_hit_count": int(len(buy_calls[buy_calls["close_reason"] == "TP_HIT"])),
        "total_sell_call_tp_hit_count": int(len(sell_calls[sell_calls["close_reason"] == "TP_HIT"])),
        "total_buy_call_sl_hit_count": int(len(buy_calls[buy_calls["close_reason"] == "SL_HIT"])),
        "total_sell_call_sl_hit_count": int(len(sell_calls[sell_calls["close_reason"] == "SL_HIT"])),
        "total_close_loss_count": int(len(close_loss)),
        "total_close_win_count": int(len(close_win)),
        "total_buy_call_tp_hit_pips": rounded_pip(buy_tp_pips),
        "total_sell_call_tp_hit_pips": rounded_pip(sell_tp_pips),
        "total_buy_call_sl_hit_pips": rounded_pip(buy_sl_pips),
        "total_sell_call_sl_hit_pips": rounded_pip(sell_sl_pips),
        "total_close_loss_pips": rounded_pip(close_loss_pips),
        "total_close_win_pips": rounded_pip(close_win_pips),
        "total_win_pip": rounded_pip(total_win_pip),
        "total_loss_pip": rounded_pip(total_loss_pip),
        "net_pip": rounded_pip(net_pip),
        "is_profitable": bool(net_pip > 0),
    }
    return {field: summary[field] for field in SUMMARY_FIELDS}


def markdown_content(example_file_name: str, output_dir: str | Path) -> str:
    example_command = (
        f"python F:\\GT-v1-engine\\{SCRIPT_RELATIVE_PATH} `\n"
        f"  --file-name {example_file_name} `\n"
        f"  --experiments-root-path F:\\GT-v1-shared-storage\\experiments `\n"
        f"  --output-dir {output_dir}"
    )
    return f"""# {RULE_NAME}

## Rule Name
{RULE_NAME}

## Rule Description
{RULE_DESCRIPTION}

## Script Path
F:\\GT-v1-engine\\{SCRIPT_RELATIVE_PATH}

## Example PowerShell Command
```powershell
{example_command}
```

## Required Input Columns
OHLC: {", ".join(REQUIRED_OHLC_COLUMNS)}

TD: {", ".join(REQUIRED_TD_COLUMNS)}

## Open Condition
No active call. BUY_CALL opens when all 9 required TD columns are UP, or when exactly 8 required TD columns are UP and the remaining TD column is DOWN or NO_SIGNAL. SELL_CALL opens when all 9 required TD columns are DOWN, or when exactly 8 required TD columns are DOWN and the remaining TD column is UP or NO_SIGNAL. NO_DIRECTION, empty, null, unexpected values, and 7-of-9 agreement do not open a call.

## Close Condition
Close management starts from the candle after the open candle. BUY_CALL closes by TD opposition when at least 4 required TD columns are DOWN or NO_DIRECTION. SELL_CALL closes by TD opposition when at least 4 required TD columns are UP or NO_DIRECTION.

## TP/SL Condition
BUY_CALL TP is High - entry_price >= 0.300. BUY_CALL SL is entry_price - Low >= 0.300. SELL_CALL TP is entry_price - Low >= 0.300. SELL_CALL SL is High - entry_price >= 0.300. Same-candle TP/SL conflict closes conservatively as SL_HIT.

## Output Files Created
- `<INPUT_FILE_STEM>_{RULE_NAME}_CALLS.csv`
- `{RULE_NAME}_SUMMARY.json`

## Clean Summary Fields
{", ".join(SUMMARY_FIELDS)}
"""


def execute(file_name: str, experiments_root_path: str, output_dir: str) -> dict[str, Any]:
    input_path = build_input_path(experiments_root_path, file_name)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_path}")

    calls_csv_path, summary_json_path, markdown_path = output_paths(input_path, output_dir)
    calls_df, summary = run_backtest(df, source_file_name=file_name)

    calls_csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    calls_df.to_csv(calls_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_content(file_name, output_dir), encoding="utf-8")

    return {
        "status": "SUCCESS",
        "rule_name": RULE_NAME,
        "output_calls_csv": str(calls_csv_path),
        "output_summary_json": str(summary_json_path),
        "output_markdown": str(markdown_path),
        "summary": summary,
    }


def main() -> int:
    try:
        args = parse_args()
        response = execute(args.file_name, args.experiments_root_path, args.output_dir)
        print(json.dumps(response))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RULE_NAME = "INDICATOR_BACKTEST_RULE_01"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]
VALID_CALL_VALUES = {
    "BUY_CALL",
    "SELL_CALL",
    "HOLD_BUY_CALL",
    "HOLD_SELL_CALL",
    "CLOSE_BUY_CALL",
    "CLOSE_SELL_CALL",
    "INTERVAL_SKIP",
    "NO_DIRECTION",
    "NO_DIRECTION_SKIP",
}
EVALUATION_COLUMNS = [
    "indicator_name",
    "td_column",
    "td_value",
    "CALL",
    "call_id",
    "entry_datetime",
    "entry_price",
    "entry_side",
    "close_datetime",
    "close_price",
    "close_reason",
    "realized_pips",
    "result",
    "highest_price_after_call",
    "lowest_price_after_call",
    "max_win_side_pips",
    "max_loss_side_pips",
    "candle_count_from_call",
    "is_open_call",
    "interval_skip",
]
SUMMARY_COLUMNS = [
    "indicator_name",
    "td_column",
    "call_id",
    "entry_side",
    "CALL",
    "entry_datetime",
    "close_datetime",
    "entry_price",
    "close_price",
    "realized_pips",
    "result",
    "close_reason",
    "highest_price_after_call",
    "lowest_price_after_call",
    "max_win_side_pips",
    "max_loss_side_pips",
    "candle_count_from_call",
    "td_value",
    "DateTime",
    "Open",
    "High",
    "Low",
    "Close",
]


@dataclass
class OpenCall:
    call_id: int
    side: str
    entry_datetime: Any
    entry_price: float
    highest_price_after_call: float
    lowest_price_after_call: float
    candle_count_from_call: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest one indicator TD column using CALL evaluation rule 01."
    )
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--indicator-root-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pip-size", type=float, default=0.01)
    return parser.parse_args()


def validate_file_name(file_name: str) -> None:
    if not file_name or not file_name.strip():
        raise ValueError("--file-name must not be empty.")
    if any(token in file_name for token in ("..", "/", "\\")):
        raise ValueError(
            "--file-name must be a file name only, not a path. "
            "Rejecting values containing '..', '/', or '\\'."
        )


def build_input_path(indicator_root_path: str, file_name: str) -> Path:
    validate_file_name(file_name)
    return Path(indicator_root_path) / file_name


def detect_td_column(df: pd.DataFrame) -> tuple[str, str]:
    td_columns = [str(column) for column in df.columns if str(column).endswith("_TD")]
    if not td_columns:
        raise ValueError("No _TD column found. This script expects one indicator file.")
    if len(td_columns) > 1:
        raise ValueError(
            "Multiple _TD columns found: "
            + ", ".join(td_columns)
            + ". This script expects one indicator file only."
        )
    td_column = td_columns[0]
    indicator_name = td_column[: -len("_TD")]
    if not indicator_name:
        raise ValueError(f"Invalid TD column name: {td_column}")
    return td_column, indicator_name


def validate_input_frame(df: pd.DataFrame) -> tuple[str, str]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV missing required columns: "
            + ", ".join(missing)
            + f". Required columns: {', '.join(REQUIRED_COLUMNS)}."
        )
    return detect_td_column(df)


def normalize_td_value(value: Any) -> str:
    if value is None:
        return "NO_DIRECTION"
    if isinstance(value, float) and math.isnan(value):
        return "NO_DIRECTION"
    normalized = str(value).strip().upper()
    if normalized in {"UP", "DOWN"}:
        return normalized
    return "NO_DIRECTION"


def rounded_pips(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 1)


def result_from_pips(realized_pips: float) -> str:
    if realized_pips > 0:
        return "WIN"
    if realized_pips < 0:
        return "LOSS"
    return "BREAKEVEN"


def row_datetime(row: pd.Series) -> Any:
    return row["DateTime"] if "DateTime" in row.index else None


def side_pips(open_call: OpenCall, pip_size: float) -> tuple[float, float]:
    if open_call.side == "BUY_CALL":
        max_win = (open_call.highest_price_after_call - open_call.entry_price) / pip_size
        max_loss = (open_call.entry_price - open_call.lowest_price_after_call) / pip_size
    else:
        max_win = (open_call.entry_price - open_call.lowest_price_after_call) / pip_size
        max_loss = (open_call.highest_price_after_call - open_call.entry_price) / pip_size
    return max_win, max_loss


def base_record(row: pd.Series, indicator_name: str, td_column: str, td_value: str) -> dict[str, Any]:
    record = row.to_dict()
    record.update(
        {
            "indicator_name": indicator_name,
            "td_column": td_column,
            "td_value": td_value,
            "CALL": "NO_DIRECTION",
            "call_id": None,
            "entry_datetime": None,
            "entry_price": None,
            "entry_side": None,
            "close_datetime": None,
            "close_price": None,
            "close_reason": None,
            "realized_pips": None,
            "result": None,
            "highest_price_after_call": None,
            "lowest_price_after_call": None,
            "max_win_side_pips": None,
            "max_loss_side_pips": None,
            "candle_count_from_call": None,
            "is_open_call": False,
            "interval_skip": False,
        }
    )
    return record


def fill_call_fields(record: dict[str, Any], open_call: OpenCall, pip_size: float) -> None:
    max_win, max_loss = side_pips(open_call, pip_size)
    record["call_id"] = open_call.call_id
    record["entry_datetime"] = open_call.entry_datetime
    record["entry_price"] = open_call.entry_price
    record["entry_side"] = open_call.side
    record["highest_price_after_call"] = open_call.highest_price_after_call
    record["lowest_price_after_call"] = open_call.lowest_price_after_call
    record["max_win_side_pips"] = rounded_pips(max_win)
    record["max_loss_side_pips"] = rounded_pips(max_loss)
    record["candle_count_from_call"] = open_call.candle_count_from_call


def call_side_for_td(td_value: str) -> str | None:
    if td_value == "UP":
        return "BUY_CALL"
    if td_value == "DOWN":
        return "SELL_CALL"
    return None


def run_backtest(df: pd.DataFrame, pip_size: float = 0.01) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pip_size <= 0:
        raise ValueError("--pip-size must be greater than 0.")

    td_column, indicator_name = validate_input_frame(df)
    if "DateTime" in df.columns:
        ordered_df = df.sort_values("DateTime", kind="stable").reset_index(drop=True)
    else:
        ordered_df = df.reset_index(drop=True)

    records: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    open_call: OpenCall | None = None
    skip_next_candle = False
    completed_call_cycle = False
    next_call_id = 1
    interval_skip_count = 0
    no_direction_count = 0
    no_direction_skip_count = 0

    for _, row in ordered_df.iterrows():
        td_value = normalize_td_value(row[td_column])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        record = base_record(row, indicator_name, td_column, td_value)

        if skip_next_candle:
            record["CALL"] = "INTERVAL_SKIP"
            record["interval_skip"] = True
            interval_skip_count += 1
            skip_next_candle = False
            completed_call_cycle = True
            records.append(record)
            continue

        if open_call is None:
            side = call_side_for_td(td_value)
            if side is None:
                if completed_call_cycle:
                    record["CALL"] = "NO_DIRECTION_SKIP"
                    no_direction_skip_count += 1
                else:
                    record["CALL"] = "NO_DIRECTION"
                    no_direction_count += 1
                records.append(record)
                continue

            open_call = OpenCall(
                call_id=next_call_id,
                side=side,
                entry_datetime=row_datetime(row),
                entry_price=close,
                highest_price_after_call=close,
                lowest_price_after_call=close,
                candle_count_from_call=1,
            )
            next_call_id += 1
            record["CALL"] = side
            record["is_open_call"] = True
            fill_call_fields(record, open_call, pip_size)
            calls.append(
                {
                    "call_id": open_call.call_id,
                    "side": open_call.side,
                    "closed": False,
                    "realized_pips": None,
                    "result": None,
                    "max_win_side_pips": None,
                    "max_loss_side_pips": None,
                }
            )
            records.append(record)
            continue

        open_call.highest_price_after_call = max(open_call.highest_price_after_call, high)
        open_call.lowest_price_after_call = min(open_call.lowest_price_after_call, low)
        open_call.candle_count_from_call += 1
        same_direction = (
            (open_call.side == "BUY_CALL" and td_value == "UP")
            or (open_call.side == "SELL_CALL" and td_value == "DOWN")
        )
        if same_direction:
            record["CALL"] = "HOLD_BUY_CALL" if open_call.side == "BUY_CALL" else "HOLD_SELL_CALL"
            record["is_open_call"] = True
            fill_call_fields(record, open_call, pip_size)
            records.append(record)
            continue

        close_reason = "NO_DIRECTION_CLOSE" if td_value == "NO_DIRECTION" else "TD_REVERSAL"
        realized_pips = (
            (close - open_call.entry_price) / pip_size
            if open_call.side == "BUY_CALL"
            else (open_call.entry_price - close) / pip_size
        )
        max_win, max_loss = side_pips(open_call, pip_size)
        record["CALL"] = "CLOSE_BUY_CALL" if open_call.side == "BUY_CALL" else "CLOSE_SELL_CALL"
        fill_call_fields(record, open_call, pip_size)
        record["close_datetime"] = row_datetime(row)
        record["close_price"] = close
        record["close_reason"] = close_reason
        record["realized_pips"] = rounded_pips(realized_pips)
        record["result"] = result_from_pips(realized_pips)
        record["is_open_call"] = False

        for call in reversed(calls):
            if call["call_id"] == open_call.call_id:
                call["closed"] = True
                call["realized_pips"] = realized_pips
                call["result"] = record["result"]
                call["max_win_side_pips"] = max_win
                call["max_loss_side_pips"] = max_loss
                break

        if td_value == "NO_DIRECTION":
            no_direction_count += 1
        open_call = None
        skip_next_candle = True
        records.append(record)

    if open_call is not None:
        max_win, max_loss = side_pips(open_call, pip_size)
        for call in reversed(calls):
            if call["call_id"] == open_call.call_id:
                call["max_win_side_pips"] = max_win
                call["max_loss_side_pips"] = max_loss
                break

    output_df = pd.DataFrame(records)
    output_df = output_df[[*ordered_df.columns, *EVALUATION_COLUMNS]]
    invalid_calls = set(output_df["CALL"].dropna().unique()) - VALID_CALL_VALUES
    if invalid_calls:
        raise ValueError(f"Internal error: output contains invalid CALL values: {sorted(invalid_calls)}")

    closed_calls = [call for call in calls if call["closed"]]
    realized_values = [float(call["realized_pips"]) for call in closed_calls]
    max_win_side_values = [
        float(call["max_win_side_pips"])
        for call in calls
        if call.get("max_win_side_pips") is not None
    ]
    max_loss_side_values = [
        float(call["max_loss_side_pips"])
        for call in calls
        if call.get("max_loss_side_pips") is not None
    ]
    summary = {
        "rule_name": RULE_NAME,
        "indicator_name": indicator_name,
        "td_column": td_column,
        "row_count": len(output_df),
        "call_count": len(calls),
        "buy_call_count": sum(1 for call in calls if call["side"] == "BUY_CALL"),
        "sell_call_count": sum(1 for call in calls if call["side"] == "SELL_CALL"),
        "win_count": sum(1 for call in closed_calls if call.get("result") == "WIN"),
        "loss_count": sum(1 for call in closed_calls if call.get("result") == "LOSS"),
        "breakeven_count": sum(1 for call in closed_calls if call.get("result") == "BREAKEVEN"),
        "open_unclosed_count": sum(1 for call in calls if not call["closed"]),
        "total_realized_pips": rounded_pips(sum(realized_values)) or 0.0,
        "average_realized_pips": rounded_pips(sum(realized_values) / len(realized_values))
        if realized_values
        else 0.0,
        "max_win_pips": rounded_pips(max([value for value in realized_values if value > 0], default=0.0)),
        "max_loss_pips": rounded_pips(min([value for value in realized_values if value < 0], default=0.0)),
        "max_win_side_pips": rounded_pips(max(max_win_side_values, default=0.0)),
        "max_loss_side_pips": rounded_pips(max(max_loss_side_values, default=0.0)),
        "interval_skip_count": interval_skip_count,
        "no_direction_count": no_direction_count,
        "no_direction_skip_count": no_direction_skip_count,
        "status": "SUCCESS",
    }
    return output_df, summary


def build_closed_calls_summary(output_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = output_df[
        output_df["CALL"].isin({"CLOSE_BUY_CALL", "CLOSE_SELL_CALL"})
    ].copy()
    for column in SUMMARY_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = None
    return summary_df[SUMMARY_COLUMNS]


def output_paths(input_path: Path, output_dir: str) -> tuple[Path, Path, Path]:
    output_root = Path(output_dir)
    output_csv = output_root / f"{input_path.stem}_BACKTEST_RULE_01.csv"
    closed_calls_summary_csv = (
        output_root / f"{input_path.stem}_BACKTEST_RULE_01_CLOSED_CALLS_SUMMARY.csv"
    )
    legacy_summary_json = output_root / f"{input_path.stem}_BACKTEST_RULE_01_summary.json"
    return output_csv, closed_calls_summary_csv, legacy_summary_json


def execute(
    file_name: str,
    indicator_root_path: str,
    output_dir: str,
    pip_size: float = 0.01,
) -> dict[str, Any]:
    input_path = build_input_path(indicator_root_path, file_name)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_path}")

    output_df, summary = run_backtest(df, pip_size)
    closed_calls_summary_df = build_closed_calls_summary(output_df)
    output_csv_path, closed_calls_summary_csv_path, legacy_summary_json_path = output_paths(
        input_path, output_dir
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    closed_calls_summary_df.to_csv(closed_calls_summary_csv_path, index=False)

    summary.update(
        {
            "input_file_name": file_name,
            "input_path": str(input_path),
            "output_csv_path": str(output_csv_path),
            "closed_calls_summary_csv_path": str(closed_calls_summary_csv_path),
            "indicator_root_path": str(Path(indicator_root_path)),
            "output_dir": str(Path(output_dir)),
        }
    )
    if legacy_summary_json_path.exists():
        legacy_summary_json_path.unlink()

    return {
        "status": "SUCCESS",
        "rule_name": RULE_NAME,
        "input_path": str(input_path),
        "output_csv_path": str(output_csv_path),
        "closed_calls_summary_csv_path": str(closed_calls_summary_csv_path),
        "indicator_name": summary["indicator_name"],
        "closed_call_count": summary["call_count"] - summary["open_unclosed_count"],
        "total_realized_pips": summary["total_realized_pips"],
    }


def main() -> int:
    try:
        args = parse_args()
        response = execute(
            file_name=args.file_name,
            indicator_root_path=args.indicator_root_path,
            output_dir=args.output_dir,
            pip_size=args.pip_size,
        )
        print(json.dumps(response))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = [
    "direction",
    "entry_datetime",
    "close_datetime",
    "close_status",
    "collected_pips",
    "cc",
]

QUARTERS = [
    ("Q1_00_00_to_05_59", range(0, 6)),
    ("Q2_06_00_to_11_59", range(6, 12)),
    ("Q3_12_00_to_17_59", range(12, 18)),
    ("Q4_18_00_to_23_59", range(18, 24)),
]

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

FUTURE_ML_DL_FEATURE_SUGGESTIONS = [
    "Add rolling last N trade win/loss streak features.",
    "Add hour-of-day and quarter-of-day features.",
    "Add weekday performance features.",
    "Add direction-specific performance features: UP win rate, DOWN win rate, UP net pips, DOWN net pips.",
    "Add close_status distribution as rule behavior features.",
    "Add holding candle count cc as an exit timing feature.",
    "Add market session labels: Asia, London, New York, Overlap.",
    "Add volatility features from candle range: High-Low and ATR if available.",
    "Add drawdown and consecutive loss features.",
    "Add risk-reward features: tpPip, slPip, tp/sl ratio.",
    "Add previous trade outcome feature.",
    "Add previous trade collected_pips feature.",
    "Add cumulative daily pips before entry.",
    "Add daily trade count before entry.",
    "Add per-rule comparison features across multiple TP/SL versions.",
    "For DL models, export sequence windows before each entry using OHLC and indicator columns.",
    "For classification, label future trade result as WIN, LOSS, ZERO.",
    "For regression, label future collected_pips.",
    "For ranking, compare multiple rule versions and rank best expected net pips.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a GT rule trade CSV summary JSON.")
    parser.add_argument("--input-trades", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--rule-id", required=True)
    parser.add_argument("--rule-name", required=True)
    return parser.parse_args()


def round_number(value):
    return round(float(value), 3)


def empty_pip_summary():
    return {
        "orderCount": 0,
        "totalPips": 0.0,
        "averagePips": 0.0,
    }


def empty_count_pip_summary():
    return {
        "orderCount": 0,
        "totalPips": 0.0,
    }


def add_trade(summary, pips):
    summary["orderCount"] += 1
    summary["totalPips"] = round_number(summary["totalPips"] + pips)


def finalize_average(summary):
    count = summary["orderCount"]
    summary["totalPips"] = round_number(summary["totalPips"])
    summary["averagePips"] = round_number(summary["totalPips"] / count) if count else 0.0


def parse_float(value, column, row_number):
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in column {column} at CSV row {row_number}: {value!r}") from exc


def parse_int(value, column, row_number):
    try:
        return int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(f"Invalid integer value in column {column} at CSV row {row_number}: {value!r}") from exc


def parse_datetime(value, column, row_number):
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid datetime value in column {column} at CSV row {row_number}: {value!r}")


def read_trades(input_trades_path):
    path = Path(input_trades_path)
    if not path.exists():
        raise FileNotFoundError(f"Input trade CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input trade CSV path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input trade CSV is empty: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input trade CSV has no header: {path}")

        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required trade CSV columns: {missing}")

        trades = []
        for index, row in enumerate(reader, start=2):
            entry_datetime = parse_datetime(row["entry_datetime"], "entry_datetime", index)
            parse_datetime(row["close_datetime"], "close_datetime", index)
            trades.append({
                "direction": str(row["direction"]).strip().upper(),
                "entry_datetime": entry_datetime,
                "close_status": str(row["close_status"]).strip(),
                "collected_pips": parse_float(row["collected_pips"], "collected_pips", index),
                "cc": parse_int(row["cc"], "cc", index),
            })

    return trades


def quarter_name_for_hour(hour):
    for name, hours in QUARTERS:
        if hour in hours:
            return name
    raise ValueError(f"Invalid hour value: {hour}")


def best_key_by_total(summary, highest=True):
    if not summary:
        return ""
    return sorted(
        summary.items(),
        key=lambda item: (item[1]["totalPips"], item[0]),
        reverse=highest,
    )[0][0]


def build_summary(rule_id, rule_name, input_trades_path, trades):
    total_trades = len(trades)
    net_pips = round_number(sum(trade["collected_pips"] for trade in trades))

    close_status_summary = {}
    direction_summary = {
        "UP": empty_pip_summary(),
        "DOWN": empty_pip_summary(),
    }
    win_orders = empty_pip_summary()
    loss_orders = empty_pip_summary()
    zero_orders = empty_pip_summary()
    day_quarter_summary = {name: empty_pip_summary() for name, _ in QUARTERS}
    weekday_summary = {day: empty_pip_summary() for day in WEEKDAYS}
    direction_profit_loss_summary = {
        "UP_PROFIT": empty_count_pip_summary(),
        "UP_LOSS": empty_count_pip_summary(),
        "DOWN_PROFIT": empty_count_pip_summary(),
        "DOWN_LOSS": empty_count_pip_summary(),
    }
    cc_values = []
    winning_cc_values = []
    losing_cc_values = []

    for trade in trades:
        pips = trade["collected_pips"]
        direction = trade["direction"]
        close_status = trade["close_status"]
        entry_datetime = trade["entry_datetime"]
        cc = trade["cc"]

        close_status_summary.setdefault(close_status, empty_pip_summary())
        add_trade(close_status_summary[close_status], pips)

        if direction not in direction_summary:
            direction_summary[direction] = empty_pip_summary()
        add_trade(direction_summary[direction], pips)

        if pips > 0:
            add_trade(win_orders, pips)
        elif pips < 0:
            add_trade(loss_orders, pips)
        else:
            add_trade(zero_orders, pips)

        quarter_name = quarter_name_for_hour(entry_datetime.hour)
        add_trade(day_quarter_summary[quarter_name], pips)

        weekday_name = entry_datetime.strftime("%A")
        add_trade(weekday_summary[weekday_name], pips)

        if direction == "UP" and pips > 0:
            add_trade(direction_profit_loss_summary["UP_PROFIT"], pips)
        elif direction == "UP" and pips < 0:
            add_trade(direction_profit_loss_summary["UP_LOSS"], pips)
        elif direction == "DOWN" and pips > 0:
            add_trade(direction_profit_loss_summary["DOWN_PROFIT"], pips)
        elif direction == "DOWN" and pips < 0:
            add_trade(direction_profit_loss_summary["DOWN_LOSS"], pips)

        cc_values.append(cc)
        if pips > 0:
            winning_cc_values.append(cc)
        elif pips < 0:
            losing_cc_values.append(cc)

    for summary_group in close_status_summary.values():
        finalize_average(summary_group)
    for summary_group in direction_summary.values():
        finalize_average(summary_group)
    for summary_group in [win_orders, loss_orders, zero_orders]:
        finalize_average(summary_group)
    for summary_group in day_quarter_summary.values():
        finalize_average(summary_group)
    for summary_group in weekday_summary.values():
        finalize_average(summary_group)
    for summary_group in direction_profit_loss_summary.values():
        summary_group["totalPips"] = round_number(summary_group["totalPips"])

    most_profitable_quarter = best_key_by_total(day_quarter_summary, highest=True)
    most_loss_quarter = best_key_by_total(day_quarter_summary, highest=False)
    most_profitable_day = best_key_by_total(weekday_summary, highest=True)
    most_loss_day = best_key_by_total(weekday_summary, highest=False)

    day_quarter_summary["mostProfitableQuarter"] = most_profitable_quarter
    day_quarter_summary["mostLossQuarter"] = most_loss_quarter

    return {
        "ruleId": rule_id,
        "ruleName": rule_name,
        "inputTradesPath": str(Path(input_trades_path).resolve()),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalTrades": total_trades,
        "netPips": net_pips,
        "averagePipsPerTrade": round_number(net_pips / total_trades) if total_trades else 0.0,
        "closeStatusSummary": close_status_summary,
        "directionSummary": direction_summary,
        "winLossSummary": {
            "winOrders": win_orders,
            "lossOrders": loss_orders,
            "zeroOrders": zero_orders,
            "winRatePercent": round_number((win_orders["orderCount"] / total_trades) * 100) if total_trades else 0.0,
            "lossRatePercent": round_number((loss_orders["orderCount"] / total_trades) * 100) if total_trades else 0.0,
        },
        "dayQuarterSummary": day_quarter_summary,
        "weekdaySummary": weekday_summary,
        "bestWorstWeekday": {
            "mostProfitableDay": most_profitable_day,
            "mostProfitableDayPips": round_number(weekday_summary[most_profitable_day]["totalPips"]) if most_profitable_day else 0.0,
            "mostLossDay": most_loss_day,
            "mostLossDayPips": round_number(weekday_summary[most_loss_day]["totalPips"]) if most_loss_day else 0.0,
        },
        "directionProfitLossSummary": direction_profit_loss_summary,
        "holdingCandleSummary": {
            "averageCc": round_number(sum(cc_values) / len(cc_values)) if cc_values else 0.0,
            "minCc": min(cc_values) if cc_values else 0,
            "maxCc": max(cc_values) if cc_values else 0,
            "winningAverageCc": round_number(sum(winning_cc_values) / len(winning_cc_values)) if winning_cc_values else 0.0,
            "losingAverageCc": round_number(sum(losing_cc_values) / len(losing_cc_values)) if losing_cc_values else 0.0,
        },
        "futureMlDlFeatureSuggestions": FUTURE_ML_DL_FEATURE_SUGGESTIONS,
    }


def write_summary(output_summary_path, summary):
    output_path = Path(output_summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return output_path


def main():
    args = parse_args()
    trades = read_trades(args.input_trades)
    summary = build_summary(args.rule_id, args.rule_name, args.input_trades, trades)
    output_path = write_summary(args.output_summary, summary)

    print("GT rule trade CSV summary completed")
    print(f"Rule ID: {args.rule_id}")
    print(f"Rule name: {args.rule_name}")
    print(f"Input trades: {Path(args.input_trades).resolve()}")
    print(f"Total trades: {summary['totalTrades']}")
    print(f"Net pips: {summary['netPips']}")
    print(f"Output summary JSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()

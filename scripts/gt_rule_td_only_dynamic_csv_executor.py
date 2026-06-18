import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


ALLOWED_DIRECTIONS = {"UP", "DOWN", "NO_SIGNAL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Execute a TD-only GT rule JSON against a dataset CSV.")
    parser.add_argument("--rule-json", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--rule-id", required=True)
    parser.add_argument("--rule-name", required=True)
    parser.add_argument("--pair")
    return parser.parse_args()


def load_rule(rule_json_path):
    path = Path(rule_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Rule JSON does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Rule JSON path is not a file: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        rule = json.load(handle)

    full_rule_conditions = rule.get("fullRuleConditions")
    if isinstance(full_rule_conditions, str):
        return json.loads(full_rule_conditions)
    if isinstance(full_rule_conditions, dict):
        return full_rule_conditions
    return rule


def require_path(rule, dotted_path):
    current = rule
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Missing rule JSON value: {dotted_path}")
        current = current[part]
    return current


def infer_pair(input_path):
    filename = Path(input_path).name.upper()
    match = re.search(r"\b([A-Z]{6})_(?:M\d+|H\d+|D\d+|W\d+)", filename)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z]{6})\b", filename)
    if match:
        return match.group(1)
    raise ValueError("Pair could not be inferred from dataset filename. Please pass --pair.")


def get_pip_size(rule, pair):
    pip_config = require_path(rule, "pipConfig")
    if "JPY" in pair.upper():
        return float(pip_config["jpyPairPipSize"])
    return float(pip_config["nonJpyPairPipSize"])


def normalize_direction(value, column, row_number):
    normalized = str(value).strip().upper()
    if normalized not in ALLOWED_DIRECTIONS:
        raise ValueError(
            f"Invalid direction value in column {column} at CSV row {row_number}: {value!r}. "
            f"Allowed values: {sorted(ALLOWED_DIRECTIONS)}"
        )
    return normalized


def normalize_float(value, column, row_number):
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in column {column} at CSV row {row_number}: {value!r}") from exc


def parse_datetime(value):
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
    raise ValueError(f"Invalid datetime value: {value!r}")


def collect_rule_columns(rule):
    ohlc_columns = require_path(rule, "input.ohlcColumns")
    indicator_columns = require_path(rule, "input.indicatorColumns")
    return {
        "required": list(require_path(rule, "input.requiredColumns")),
        "datetime": require_path(rule, "input.datetimeColumn"),
        "open": ohlc_columns["open"],
        "high": ohlc_columns["high"],
        "low": ohlc_columns["low"],
        "close": ohlc_columns["close"],
        "macd_direction": indicator_columns["macdDirection"],
        "macd_strength": indicator_columns["macdStrength"],
        "ema_stack_direction": indicator_columns["emaStackDirection"],
        "ema_stack_strength": indicator_columns["emaStackStrength"],
        "price_columns": [
            ohlc_columns["open"],
            ohlc_columns["high"],
            ohlc_columns["low"],
            ohlc_columns["close"],
            require_path(rule, "parameters.entryPriceSource"),
        ],
    }


def read_dataset(input_path, rule_columns):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input CSV path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input CSV is empty: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV is empty: {path}")

        missing = [column for column in rule_columns["required"] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV has no data rows: {path}")

    normalized_rows = []
    for original_index, row in enumerate(rows):
        row_number = original_index + 2
        normalized = dict(row)
        normalized["_original_index"] = original_index
        normalized["_sort_datetime"] = parse_datetime(row[rule_columns["datetime"]])

        for column in rule_columns["price_columns"]:
            normalized[column] = normalize_float(row[column], column, row_number)
        normalized[rule_columns["macd_direction"]] = normalize_direction(
            row[rule_columns["macd_direction"]], rule_columns["macd_direction"], row_number
        )
        normalized[rule_columns["ema_stack_direction"]] = normalize_direction(
            row[rule_columns["ema_stack_direction"]], rule_columns["ema_stack_direction"], row_number
        )
        normalized[rule_columns["macd_strength"]] = normalize_float(
            row[rule_columns["macd_strength"]], rule_columns["macd_strength"], row_number
        )
        normalized[rule_columns["ema_stack_strength"]] = normalize_float(
            row[rule_columns["ema_stack_strength"]], rule_columns["ema_stack_strength"], row_number
        )
        normalized_rows.append(normalized)

    normalized_rows.sort(key=lambda item: item["_sort_datetime"])
    return normalized_rows


def has_entry_signal(row, columns):
    macd_td = row[columns["macd_direction"]]
    ema_stack_td = row[columns["ema_stack_direction"]]
    macd_ts = row[columns["macd_strength"]]
    ema_stack_ts = row[columns["ema_stack_strength"]]
    return (
        macd_td == ema_stack_td
        and macd_td in {"UP", "DOWN"}
        and ema_stack_td in {"UP", "DOWN"}
        and 0.0 <= macd_ts <= 1.0
        and 0.0 <= ema_stack_ts <= 1.0
    )


def create_active_order(row, row_index, order_id, columns, rule_values, pair, rule_name):
    direction = row[columns["macd_direction"]]
    entry_price = row[rule_values["entry_price_source"]]
    tp_distance = rule_values["tp_pip"] * rule_values["pip_size"]
    sl_distance = rule_values["sl_pip"] * rule_values["pip_size"]

    if direction == "UP":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    return {
        "rule_name": rule_name,
        "pair": pair,
        "timeframe": rule_values["timeframe"],
        "order_id": order_id,
        "direction": direction,
        "entry_datetime": row[columns["datetime"]],
        "entry_index": row_index,
        "entry_price": entry_price,
        "tp_pip": rule_values["tp_pip"],
        "sl_pip": rule_values["sl_pip"],
        "tp_price": tp_price,
        "sl_price": sl_price,
        "cc": 0,
        "entry_macd_td": row[columns["macd_direction"]],
        "entry_macd_ts": row[columns["macd_strength"]],
        "entry_ema_stack_td": row[columns["ema_stack_direction"]],
        "entry_ema_stack_ts": row[columns["ema_stack_strength"]],
    }


def calculate_collected_pips(order, close_price, pip_size):
    if order["direction"] == "UP":
        return (close_price - order["entry_price"]) / pip_size
    return (order["entry_price"] - close_price) / pip_size


def status_from_pips(win_status, loss_status, collected_pips):
    if collected_pips > 0:
        return win_status
    return loss_status


def tp_sl_hit(order, row, columns):
    if order["direction"] == "UP":
        tp_hit = row[columns["high"]] >= order["tp_price"]
        sl_hit = row[columns["low"]] <= order["sl_price"]
    else:
        tp_hit = row[columns["low"]] <= order["tp_price"]
        sl_hit = row[columns["high"]] >= order["sl_price"]
    return tp_hit, sl_hit


def close_trade(order, row, row_index, columns, close_status, collected_pips, close_price):
    trade = dict(order)
    trade.update(
        {
            "close_datetime": row[columns["datetime"]],
            "close_index": row_index,
            "close_price": close_price,
            "close_status": close_status,
            "collected_pips": round(collected_pips, 3),
            "close_macd_td": row[columns["macd_direction"]],
            "close_macd_ts": row[columns["macd_strength"]],
            "close_ema_stack_td": row[columns["ema_stack_direction"]],
            "close_ema_stack_ts": row[columns["ema_stack_strength"]],
        }
    )
    return trade


def evaluate_active_order(order, row, row_index, columns, rule_values):
    tp_hit, sl_hit = tp_sl_hit(order, row, columns)
    if tp_hit and sl_hit:
        return close_trade(
            order,
            row,
            row_index,
            columns,
            rule_values["same_candle_tp_sl_status"],
            rule_values["same_candle_tp_sl_collected_pips"],
            row[columns["close"]],
        )
    if tp_hit:
        return close_trade(
            order,
            row,
            row_index,
            columns,
            rule_values["normal_tp_status"],
            rule_values["tp_pip"],
            order["tp_price"],
        )
    if sl_hit:
        return close_trade(
            order,
            row,
            row_index,
            columns,
            rule_values["normal_sl_status"],
            -rule_values["sl_pip"],
            order["sl_price"],
        )

    if row[columns["macd_direction"]] != row[columns["ema_stack_direction"]]:
        close_price = row[columns["close"]]
        collected_pips = round(calculate_collected_pips(order, close_price, rule_values["pip_size"]), 3)
        status = status_from_pips(
            rule_values["td_violation_win_status"],
            rule_values["td_violation_loss_status"],
            collected_pips,
        )
        return close_trade(order, row, row_index, columns, status, collected_pips, close_price)

    order["cc"] += 1
    return None


def force_close_at_end(order, row, row_index, columns, rule_values):
    close_price = row[columns["close"]]
    collected_pips = round(calculate_collected_pips(order, close_price, rule_values["pip_size"]), 3)
    status = status_from_pips(rule_values["eof_win_status"], rule_values["eof_loss_status"], collected_pips)
    return close_trade(order, row, row_index, columns, status, collected_pips, close_price)


def execute_rule(rows, columns, rule_values, pair, rule_name):
    trades = []
    active_order = None
    order_id = 1

    for row_index, row in enumerate(rows):
        if active_order:
            if row_index == active_order["entry_index"]:
                continue

            closed_trade = evaluate_active_order(active_order, row, row_index, columns, rule_values)
            if closed_trade:
                trades.append(closed_trade)
                active_order = None
                order_id += 1
            continue

        if has_entry_signal(row, columns):
            active_order = create_active_order(row, row_index, order_id, columns, rule_values, pair, rule_name)

    if active_order:
        trades.append(force_close_at_end(active_order, rows[-1], len(rows) - 1, columns, rule_values))

    return trades


def format_csv_value(value):
    if isinstance(value, float):
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return value


def write_trades(output_path, output_columns, trades):
    if not output_columns:
        raise ValueError("Missing rule JSON value: output.tradeCsvColumns")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        for trade in trades:
            writer.writerow({column: format_csv_value(trade.get(column, "")) for column in output_columns})


def collect_rule_values(rule, pair):
    return {
        "tp_pip": float(require_path(rule, "parameters.tpPip")),
        "sl_pip": float(require_path(rule, "parameters.slPip")),
        "timeframe": require_path(rule, "parameters.timeframe"),
        "entry_price_source": require_path(rule, "parameters.entryPriceSource"),
        "same_candle_tp_sl_collected_pips": float(require_path(rule, "tpSlLogic.sameCandleTpSl.collectedPips")),
        "same_candle_tp_sl_status": require_path(rule, "tpSlLogic.sameCandleTpSl.closeStatus"),
        "normal_tp_status": require_path(rule, "tpSlLogic.normalTakeProfit.closeStatus"),
        "normal_sl_status": require_path(rule, "tpSlLogic.normalStopLoss.closeStatus"),
        "td_violation_win_status": require_path(rule, "violationLogic.tdViolation.winStatus"),
        "td_violation_loss_status": require_path(rule, "violationLogic.tdViolation.lossStatus"),
        "eof_win_status": require_path(rule, "endOfFileLogic.winStatus"),
        "eof_loss_status": require_path(rule, "endOfFileLogic.lossStatus"),
        "pip_size": get_pip_size(rule, pair),
    }


def main():
    args = parse_args()
    rule = load_rule(args.rule_json)
    pair = args.pair.strip().upper() if args.pair else infer_pair(args.input)
    columns = collect_rule_columns(rule)
    rule_values = collect_rule_values(rule, pair)

    rows = read_dataset(args.input, columns)
    output_columns = list(require_path(rule, "output.tradeCsvColumns"))
    trades = execute_rule(rows, columns, rule_values, pair, args.rule_name)
    write_trades(args.output_trades, output_columns, trades)

    print("GT TD-only dynamic rule CSV execution completed")
    print(f"Rule ID: {args.rule_id}")
    print(f"Rule name: {args.rule_name}")
    print(f"Pair: {pair}")
    print(f"Timeframe: {rule_values['timeframe']}")
    print(f"Input rows: {len(rows)}")
    print(f"Trades generated: {len(trades)}")
    print(f"Output trades CSV: {args.output_trades}")


if __name__ == "__main__":
    main()

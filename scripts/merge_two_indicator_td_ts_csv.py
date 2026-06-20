import argparse
import csv
from pathlib import Path


BASE_COLUMNS = ["DateTime", "Open", "High", "Low", "Close"]
DEFAULT_INDICATORS_ROOT = r"F:\GT-v1-shared-storage\indicators"
DEFAULT_EXPERIMENTS_ROOT = r"F:\GT-v1-shared-storage\experiments"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge indicator TD/TS CSV files.")
    parser.add_argument(
        "--file-names",
        help=(
            "Comma-separated indicator CSV file names to merge, in output column order. "
            "Example: USDJPY_M5_adx_td_ts.csv, USDJPY_M5_atr_td_ts.csv"
        ),
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        help="Deprecated. Use --file-names instead.",
    )
    parser.add_argument("--first-file", help="Deprecated. Use --file-names instead.")
    parser.add_argument("--second-file", help="Deprecated. Use --file-names instead.")
    parser.add_argument(
        "--indicators-root-path",
        default=DEFAULT_INDICATORS_ROOT,
        help="Directory containing indicator CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_EXPERIMENTS_ROOT,
        help="Output directory for merged indicator CSV.",
    )
    parser.add_argument(
        "--indicators-root",
        dest="indicators_root_path",
        help="Deprecated. Use --indicators-root-path instead.",
    )
    parser.add_argument(
        "--experiments-root",
        dest="output_dir",
        help="Deprecated. Use --output-dir instead.",
    )
    args = parser.parse_args()

    if args.file_names:
        if args.input_files or args.first_file or args.second_file:
            parser.error("--file-names cannot be combined with deprecated file arguments.")
        args.input_files = parse_file_names(args.file_names)
        if len(args.input_files) < 2:
            parser.error("--file-names requires at least two CSV files.")
        return args

    if args.input_files:
        if args.first_file or args.second_file:
            parser.error("--input-files cannot be combined with --first-file or --second-file.")
        if len(args.input_files) < 2:
            parser.error("--input-files requires at least two CSV files.")
        return args

    if bool(args.first_file) != bool(args.second_file):
        parser.error("--first-file and --second-file must be provided together.")
    if not args.first_file:
        parser.error("--file-names is required.")

    args.input_files = [args.first_file, args.second_file]
    return args


def parse_file_names(value):
    return [file_name.strip() for file_name in value.split(",") if file_name.strip()]


def validate_relative_csv_path(value, label):
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative.")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not contain '..'.")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{label} must be a .csv file.")
    return path


def read_csv(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV is empty: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV is empty: {csv_path}")

        missing = [column for column in BASE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns {missing}: {csv_path}")

        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV has no data rows: {csv_path}")

    return reader.fieldnames, rows


def find_td_ts_columns(fieldnames, label):
    td_columns = [column for column in fieldnames if column.endswith("_TD")]
    ts_columns = [column for column in fieldnames if column.endswith("_TS")]

    if len(td_columns) != 1:
        raise ValueError(f"{label} must have exactly one _TD column. Found: {td_columns}")
    if len(ts_columns) != 1:
        raise ValueError(f"{label} must have exactly one _TS column. Found: {ts_columns}")

    return td_columns[0], ts_columns[0]


def validate_base_columns_match(first_rows, second_rows):
    if len(first_rows) != len(second_rows):
        raise ValueError("Input CSV base columns do not match.")

    for first_row, second_row in zip(first_rows, second_rows):
        for column in BASE_COLUMNS:
            if first_row[column] != second_row[column]:
                raise ValueError("Input CSV base columns do not match.")


def validate_all_base_columns_match(csv_items):
    first_item = csv_items[0]

    for item in csv_items[1:]:
        try:
            validate_base_columns_match(first_item["rows"], item["rows"])
        except ValueError as exc:
            raise ValueError(
                f"Input CSV base columns do not match: {first_item['path']} and {item['path']}"
            ) from exc


def validate_unique_indicator_columns(csv_items):
    seen = {}

    for item in csv_items:
        for column in item["indicator_columns"]:
            if column in seen:
                raise ValueError(
                    f"Duplicate indicator column '{column}' found in {seen[column]} and {item['path']}."
                )
            seen[column] = item["path"]


def write_output(output_path, csv_items):
    output_columns = BASE_COLUMNS + [
        column for item in csv_items for column in item["indicator_columns"]
    ]
    first_rows = csv_items[0]["rows"]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()

        for row_index, first_row in enumerate(first_rows):
            output_row = {column: first_row[column] for column in BASE_COLUMNS}

            for item in csv_items:
                item_row = item["rows"][row_index]
                for column in item["indicator_columns"]:
                    output_row[column] = item_row[column]

            writer.writerow(output_row)

    return output_columns


def main():
    args = parse_args()

    indicators_root = Path(args.indicators_root_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    relative_paths = [
        validate_relative_csv_path(file_name, f"--file-names[{index}]")
        for index, file_name in enumerate(args.input_files, start=1)
    ]

    csv_items = []
    for relative_path in relative_paths:
        csv_path = indicators_root / relative_path
        fieldnames, rows = read_csv(csv_path)
        td_column, ts_column = find_td_ts_columns(fieldnames, str(csv_path))
        csv_items.append(
            {
                "path": csv_path,
                "rows": rows,
                "indicator_columns": [td_column, ts_column],
            }
        )

    validate_all_base_columns_match(csv_items)
    validate_unique_indicator_columns(csv_items)

    output_name = f"{'_'.join(item['path'].stem for item in csv_items)}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / output_name
    output_columns = write_output(output_csv, csv_items)

    print("Indicator TD/TS merge completed.")
    for index, item in enumerate(csv_items, start=1):
        print(f"Input CSV {index}: {item['path']}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows      : {len(csv_items[0]['rows'])}")
    print(f"Columns   : {', '.join(output_columns)}")


if __name__ == "__main__":
    main()

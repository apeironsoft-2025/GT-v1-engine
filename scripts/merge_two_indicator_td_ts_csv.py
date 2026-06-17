import argparse
import csv
from pathlib import Path


BASE_COLUMNS = ["DateTime", "Open", "High", "Low", "Close"]
DEFAULT_INDICATORS_ROOT = r"F:\GT-v1-shared-storage\indicators"
DEFAULT_EXPERIMENTS_ROOT = r"F:\GT-v1-shared-storage\experiments"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge two indicator TD/TS CSV files.")
    parser.add_argument("--first-file", required=True)
    parser.add_argument("--second-file", required=True)
    parser.add_argument("--indicators-root", default=DEFAULT_INDICATORS_ROOT)
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT)
    return parser.parse_args()


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


def write_output(output_path, first_rows, second_rows, first_td, first_ts, second_td, second_ts):
    output_columns = BASE_COLUMNS + [first_td, first_ts, second_td, second_ts]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()

        for first_row, second_row in zip(first_rows, second_rows):
            output_row = {column: first_row[column] for column in BASE_COLUMNS}
            output_row[first_td] = first_row[first_td]
            output_row[first_ts] = first_row[first_ts]
            output_row[second_td] = second_row[second_td]
            output_row[second_ts] = second_row[second_ts]
            writer.writerow(output_row)

    return output_columns


def main():
    args = parse_args()

    indicators_root = Path(args.indicators_root).expanduser().resolve()
    experiments_root = Path(args.experiments_root).expanduser().resolve()
    first_relative = validate_relative_csv_path(args.first_file, "--first-file")
    second_relative = validate_relative_csv_path(args.second_file, "--second-file")

    first_csv = indicators_root / first_relative
    second_csv = indicators_root / second_relative

    first_fieldnames, first_rows = read_csv(first_csv)
    second_fieldnames, second_rows = read_csv(second_csv)

    validate_base_columns_match(first_rows, second_rows)

    first_td, first_ts = find_td_ts_columns(first_fieldnames, "First CSV")
    second_td, second_ts = find_td_ts_columns(second_fieldnames, "Second CSV")

    output_name = f"{first_csv.stem}_{second_csv.stem}.csv"
    output_csv = experiments_root / output_name
    output_columns = write_output(output_csv, first_rows, second_rows, first_td, first_ts, second_td, second_ts)

    print("Indicator TD/TS merge completed.")
    print(f"First CSV : {first_csv}")
    print(f"Second CSV: {second_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows      : {len(first_rows)}")
    print(f"Columns   : {', '.join(output_columns)}")


if __name__ == "__main__":
    main()

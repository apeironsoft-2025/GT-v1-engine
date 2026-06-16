import json
from pathlib import Path

import pandas as pd

from gt_v1_engine.cleaned_csv_details import build_cleaned_csv_details


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_valid_cleaned_csv_produces_details_json(tmp_path: Path) -> None:
    root_path = tmp_path / "cleaned"
    output_json = tmp_path / "reports" / "USDJPY_M5_cleaned_details.json"
    csv_path = root_path / "USDJPY_M5_cleaned.csv"
    root_path.mkdir()
    pd.DataFrame(
        {
            "DateTime": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
            ],
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10, 11, 12],
        }
    ).to_csv(csv_path, index=False)

    summary = build_cleaned_csv_details(root_path, "USDJPY_M5_cleaned.csv", output_json)
    persisted = _read_summary(output_json)

    assert summary["status"] == "SUCCESS"
    assert persisted["status"] == "SUCCESS"
    assert persisted["row_count"] == 3
    assert persisted["file_name"] == "USDJPY_M5_cleaned.csv"
    assert persisted["start_datetime"] == "2026-01-01T00:00:00Z"
    assert persisted["end_datetime"] == "2026-01-01T00:10:00Z"
    assert persisted["available_columns"] == ["DateTime", "Open", "High", "Low", "Close", "Volume"]


def test_missing_required_column_creates_failed_json(tmp_path: Path) -> None:
    root_path = tmp_path / "cleaned"
    output_json = tmp_path / "reports" / "details.json"
    csv_path = root_path / "USDJPY_M5_cleaned.csv"
    root_path.mkdir()
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z"],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
        }
    ).to_csv(csv_path, index=False)

    summary = build_cleaned_csv_details(root_path, "USDJPY_M5_cleaned.csv", output_json)
    persisted = _read_summary(output_json)

    assert summary["status"] == "FAILED"
    assert persisted["status"] == "FAILED"
    assert "Close" in persisted["error_message"]


def test_invalid_relative_path_outside_root_is_rejected(tmp_path: Path) -> None:
    root_path = tmp_path / "cleaned"
    output_json = tmp_path / "reports" / "details.json"
    root_path.mkdir()

    summary = build_cleaned_csv_details(root_path, "../outside.csv", output_json)
    persisted = _read_summary(output_json)

    assert summary["status"] == "FAILED"
    assert persisted["status"] == "FAILED"
    assert "escapes root_path" in persisted["error_message"]


def test_ohlc_aggregate_values_ignore_invalid_numeric_values(tmp_path: Path) -> None:
    root_path = tmp_path / "cleaned"
    output_json = tmp_path / "reports" / "details.json"
    csv_path = root_path / "USDJPY_M5_cleaned.csv"
    root_path.mkdir()
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"],
            "Open": [1, 5],
            "High": [2, "bad"],
            "Low": [3, 7],
            "Close": [4, 8],
        }
    ).to_csv(csv_path, index=False)

    summary = build_cleaned_csv_details(root_path, "USDJPY_M5_cleaned.csv", output_json)

    assert summary["status"] == "SUCCESS"
    assert summary["min_price_from_ohlc"] == 1.0
    assert summary["max_price_from_ohlc"] == 8.0
    assert summary["average_price_from_ohlc"] == 30.0 / 7.0
    assert summary["median_price_from_ohlc"] == 4.0


def test_row_samples_include_first_last_and_mid_rows(tmp_path: Path) -> None:
    root_path = tmp_path / "cleaned"
    output_json = tmp_path / "reports" / "details.json"
    csv_path = root_path / "USDJPY_M5_cleaned.csv"
    root_path.mkdir()
    pd.DataFrame(
        {
            "DateTime": [f"2026-01-01T00:{minute:02d}:00Z" for minute in range(5)],
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        }
    ).to_csv(csv_path, index=False)

    summary = build_cleaned_csv_details(root_path, "USDJPY_M5_cleaned.csv", output_json)

    assert [row["Open"] for row in summary["first_2_rows"]] == [100, 101]
    assert [row["Open"] for row in summary["last_2_rows"]] == [103, 104]
    assert [row["Open"] for row in summary["mid_2_rows"]] == [101, 102]

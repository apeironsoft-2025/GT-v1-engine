import json
from pathlib import Path

import pandas as pd
import pytest

from gt_v1_engine.core.errors import DataValidationError
from gt_v1_engine.data.raw_csv_cleaner import clean_raw_market_csv


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "raw" / "AUDJPY_M5.csv",
        tmp_path / "cleaned" / "AUDJPY_M5_cleaned.csv",
        tmp_path / "reports" / "AUDJPY_M5_cleaning_summary.json",
    )


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_valid_csv_produces_cleaned_csv_and_summary_json(tmp_path: Path) -> None:
    input_path, output_path, summary_path = _paths(tmp_path)
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            " DateTime ": ["2026-01-01T00:05:00Z", "2026-01-01T00:00:00Z"],
            " Open ": [100.2, 100.0],
            "High": [100.4, 100.3],
            "Low": [100.1, 99.9],
            "Close": [100.3, 100.2],
            "Volume": [10, 20],
        }
    ).to_csv(input_path, index=False)

    summary = clean_raw_market_csv(input_path, output_path, summary_path)

    cleaned = pd.read_csv(output_path)
    persisted_summary = _read_summary(summary_path)
    assert summary["status"] == "SUCCESS"
    assert persisted_summary["status"] == "SUCCESS"
    assert list(cleaned.columns) == ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    assert list(cleaned["Close"]) == [100.2, 100.3]
    assert persisted_summary["original_row_count"] == 2
    assert persisted_summary["cleaned_row_count"] == 2


def test_missing_required_column_fails_with_summary_json(tmp_path: Path) -> None:
    input_path, output_path, summary_path = _paths(tmp_path)
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z"],
            "Open": [100.0],
            "High": [100.3],
            "Low": [99.9],
        }
    ).to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        clean_raw_market_csv(input_path, output_path, summary_path)

    summary = _read_summary(summary_path)
    assert summary["status"] == "FAILED"
    assert "Close" in summary["error_message"]
    assert not output_path.exists()


def test_tab_delimited_time_alias_with_extra_field_cleans(tmp_path: Path) -> None:
    input_path, output_path, summary_path = _paths(tmp_path)
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        "\n".join(
            [
                "Time\tOpen\tHigh\tLow\tClose\tVolume",
                "2026-01-01 00:00:00\t100.0\t100.3\t99.9\t100.2\t900\t4",
                "2026-01-01 00:05:00\t100.2\t100.4\t100.1\t100.3\t901\t5",
            ]
        ),
        encoding="utf-8",
    )

    clean_raw_market_csv(input_path, output_path, summary_path)

    cleaned = pd.read_csv(output_path)
    summary = _read_summary(summary_path)
    assert list(cleaned.columns) == ["DateTime", "Open", "High", "Low", "Close", "Volume", "Extra_1"]
    assert cleaned["Volume"].tolist() == [900, 901]
    assert summary["available_columns"] == ["DateTime", "Open", "High", "Low", "Close", "Volume", "Extra_1"]


def test_duplicate_datetime_rows_are_removed(tmp_path: Path) -> None:
    input_path, output_path, summary_path = _paths(tmp_path)
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "Open": [100.0, 101.0],
            "High": [100.3, 101.3],
            "Low": [99.9, 100.9],
            "Close": [100.2, 101.2],
        }
    ).to_csv(input_path, index=False)

    clean_raw_market_csv(input_path, output_path, summary_path)

    cleaned = pd.read_csv(output_path)
    summary = _read_summary(summary_path)
    assert len(cleaned) == 1
    assert cleaned["Open"].iloc[0] == 100.0
    assert summary["removed_duplicate_datetime_rows"] == 1


def test_invalid_ohlc_rows_are_removed(tmp_path: Path) -> None:
    input_path, output_path, summary_path = _paths(tmp_path)
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "DateTime": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
            ],
            "Open": [100.0, 100.0, "bad"],
            "High": [100.3, 99.9, 100.3],
            "Low": [99.9, 99.8, 99.9],
            "Close": [100.2, 100.2, 100.2],
        }
    ).to_csv(input_path, index=False)

    clean_raw_market_csv(input_path, output_path, summary_path)

    cleaned = pd.read_csv(output_path)
    summary = _read_summary(summary_path)
    assert len(cleaned) == 1
    assert summary["removed_invalid_ohlc_rows"] == 2


def test_output_directories_are_created_automatically(tmp_path: Path) -> None:
    input_path, output_path, summary_path = _paths(tmp_path)
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z"],
            "Open": [100.0],
            "High": [100.3],
            "Low": [99.9],
            "Close": [100.2],
        }
    ).to_csv(input_path, index=False)

    clean_raw_market_csv(input_path, output_path, summary_path)

    assert output_path.exists()
    assert summary_path.exists()

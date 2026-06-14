from pathlib import Path

import pandas as pd
import pytest

from gt_v1_engine.core.errors import DataValidationError, UnsupportedFormatError
from gt_v1_engine.data.market_data_loader import load_market_data


def test_csv_with_standard_columns_loads(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:05:00Z", "2026-01-01T00:00:00Z"],
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.2],
        }
    ).to_csv(path, index=False)

    df = load_market_data(path)
    assert list(df.columns) == ["DateTime", "Open", "High", "Low", "Close", "SRP"]
    assert list(df["Close"]) == [1.2, 1.1]


def test_lowercase_columns_load_and_normalize(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-01-01T00:00:00Z"],
            "open": [1.0],
            "high": [1.2],
            "low": [0.9],
            "close": [1.1],
            "srp": [1.05],
        }
    ).to_csv(path, index=False)

    df = load_market_data(path)
    assert list(df.columns) == ["DateTime", "Open", "High", "Low", "Close", "SRP"]
    assert df["SRP"].iloc[0] == 1.05


def test_srp_missing_creates_srp_from_close(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z"],
            "Open": [1.0],
            "High": [1.2],
            "Low": [0.9],
            "Close": [1.1],
        }
    ).to_csv(path, index=False)

    df = load_market_data(path)
    assert df["SRP"].iloc[0] == df["Close"].iloc[0]


def test_missing_close_raises_data_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01T00:00:00Z"],
            "Open": [1.0],
            "High": [1.2],
            "Low": [0.9],
        }
    ).to_csv(path, index=False)

    with pytest.raises(DataValidationError):
        load_market_data(path)


def test_unsupported_txt_raises_unsupported_format_error(tmp_path: Path) -> None:
    path = tmp_path / "market.txt"
    path.write_text("not market data", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        load_market_data(path)

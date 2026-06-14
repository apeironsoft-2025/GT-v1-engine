import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection

VALID_DIRECTIONS = {
    IndicatorDirection.UP.value,
    IndicatorDirection.DOWN.value,
    IndicatorDirection.NO_SIGNAL.value,
}
VALID_STRENGTHS = {0.0, 0.25, 0.5, 0.75, 1.0}


def make_ohlc_dataframe(rows: int = 180, trend: str = "up") -> pd.DataFrame:
    index = np.arange(rows)
    slope = 2.0 if trend == "up" else -2.0
    baseline = 160.0 + np.linspace(0, slope, rows)
    close = baseline + 0.08 * np.sin(index / 4)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    extra_range = 0.04 + 0.03 * (np.sin(index / 9) + 1)
    high = np.maximum(open_, close) + 0.05 + extra_range
    low = np.minimum(open_, close) - 0.05 - extra_range
    return pd.DataFrame(
        {
            "DateTime": pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC"),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "SRP": close,
        }
    )


def assert_indicator_contract(indicator: BaseIndicator, df: pd.DataFrame) -> pd.DataFrame:
    original = df.copy(deep=True)
    result = indicator.calculate(df)

    assert_frame_equal(df, original)
    assert len(result) == len(df)
    for column in original.columns:
        assert column in result.columns
    assert indicator.direction_column in result.columns
    assert indicator.strength_column in result.columns
    assert result[indicator.direction_column].notna().all()
    assert result[indicator.strength_column].notna().all()
    assert set(result[indicator.direction_column].unique()).issubset(VALID_DIRECTIONS)
    assert set(result[indicator.strength_column].unique()).issubset(VALID_STRENGTHS)
    return result

import numpy as np
import pandas as pd

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection
from gt_v1_engine.indicators.strength import normalize_and_bucket


class ATRIndicator(BaseIndicator):
    name = "ATR"
    implemented = True
    description = "Average True Range volatility expansion indicator."

    def __init__(self, period: int = 14, expansion_window: int = 20) -> None:
        self.period = period
        self.expansion_window = expansion_window

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, ["DateTime", "High", "Low", "Close"], self.name)
        try:
            result = df.copy()
            high = pd.to_numeric(result["High"], errors="coerce")
            low = pd.to_numeric(result["Low"], errors="coerce")
            close = pd.to_numeric(result["Close"], errors="coerce")

            previous_close = close.shift(1)
            true_range = pd.concat(
                [
                    high - low,
                    (high - previous_close).abs(),
                    (low - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = true_range.ewm(alpha=1 / self.period, adjust=False).mean()
            rolling_atr_mean = atr.rolling(self.expansion_window, min_periods=self.period).mean()
            expansion = atr >= rolling_atr_mean

            result["ATR_VALUE"] = atr
            result["ATR_ROLLING_MEAN"] = rolling_atr_mean
            result[self.direction_column] = np.select(
                [(close > previous_close) & expansion, (close < previous_close) & expansion],
                [IndicatorDirection.UP.value, IndicatorDirection.DOWN.value],
                default=IndicatorDirection.NO_SIGNAL.value,
            )
            raw_strength = ((atr / rolling_atr_mean) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            result[self.strength_column] = (
                raw_strength.map(lambda value: normalize_and_bucket(value, 0.0, 1.0))
                .fillna(0.0)
                .astype(float)
            )
            return result
        except IndicatorCalculationError:
            raise
        except Exception as exc:
            raise IndicatorCalculationError(f"ATR calculation failed: {exc}") from exc

import numpy as np
import pandas as pd

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection
from gt_v1_engine.indicators.strength import normalize_and_bucket


class MACDIndicator(BaseIndicator):
    name = "MACD"
    implemented = True
    description = "Moving Average Convergence Divergence trend indicator."

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, ["DateTime", "Close"], self.name)
        try:
            result = df.copy()
            close = pd.to_numeric(result["Close"], errors="coerce")

            ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            result["MACD_LINE"] = macd_line
            result["MACD_SIGNAL"] = signal_line
            result["MACD_HIST"] = histogram
            result[self.direction_column] = np.select(
                [histogram > 0, histogram < 0],
                [IndicatorDirection.UP.value, IndicatorDirection.DOWN.value],
                default=IndicatorDirection.NO_SIGNAL.value,
            )

            rolling_abs_hist = histogram.abs().rolling(50, min_periods=10).mean()
            raw_strength = (histogram.abs() / rolling_abs_hist).replace([np.inf, -np.inf], np.nan)
            result[self.strength_column] = (
                raw_strength.map(lambda value: normalize_and_bucket(value, 0.0, 2.0))
                .fillna(0.0)
                .astype(float)
            )
            return result
        except IndicatorCalculationError:
            raise
        except Exception as exc:
            raise IndicatorCalculationError(f"MACD calculation failed: {exc}") from exc

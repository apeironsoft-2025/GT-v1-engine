import numpy as np
import pandas as pd

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection
from gt_v1_engine.indicators.strength import normalize_and_bucket


class BollingerIndicator(BaseIndicator):
    name = "BOLLINGER"
    implemented = True
    description = "Bollinger Bands relative-position indicator."

    def __init__(self, period: int = 20, std_multiplier: float = 2.0) -> None:
        self.period = period
        self.std_multiplier = std_multiplier

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, ["DateTime", "Close"], self.name)
        try:
            result = df.copy()
            close = pd.to_numeric(result["Close"], errors="coerce")
            middle = close.rolling(self.period, min_periods=self.period).mean()
            std = close.rolling(self.period, min_periods=self.period).std()
            upper = middle + self.std_multiplier * std
            lower = middle - self.std_multiplier * std
            band_width = upper - lower

            result["BOLLINGER_MIDDLE"] = middle
            result["BOLLINGER_UPPER"] = upper
            result["BOLLINGER_LOWER"] = lower
            result[self.direction_column] = np.select(
                [close > middle, close < middle],
                [IndicatorDirection.UP.value, IndicatorDirection.DOWN.value],
                default=IndicatorDirection.NO_SIGNAL.value,
            )
            raw_strength = ((close - middle).abs() / band_width).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            result[self.strength_column] = (
                raw_strength.map(lambda value: normalize_and_bucket(value, 0.0, 0.5))
                .fillna(0.0)
                .astype(float)
            )
            return result
        except IndicatorCalculationError:
            raise
        except Exception as exc:
            raise IndicatorCalculationError(f"BOLLINGER calculation failed: {exc}") from exc

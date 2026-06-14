import numpy as np
import pandas as pd

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection
from gt_v1_engine.indicators.strength import normalize_and_bucket


class RSIIndicator(BaseIndicator):
    name = "RSI"
    implemented = True
    description = "Relative Strength Index momentum indicator."

    def __init__(self, period: int = 14, up_threshold: float = 55, down_threshold: float = 45) -> None:
        self.period = period
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, ["DateTime", "Close"], self.name)
        try:
            result = df.copy()
            close = pd.to_numeric(result["Close"], errors="coerce")
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / self.period, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.period, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi = pd.Series(
                np.select(
                    [(loss == 0) & (gain > 0), (gain == 0) & (loss > 0)],
                    [100.0, 0.0],
                    default=rsi,
                ),
                index=result.index,
            ).fillna(50.0)

            result["RSI_VALUE"] = rsi
            result[self.direction_column] = np.select(
                [rsi >= self.up_threshold, rsi <= self.down_threshold],
                [IndicatorDirection.UP.value, IndicatorDirection.DOWN.value],
                default=IndicatorDirection.NO_SIGNAL.value,
            )
            raw_strength = ((rsi - 50).abs() / 50).replace([np.inf, -np.inf], np.nan)
            result[self.strength_column] = (
                raw_strength.map(lambda value: normalize_and_bucket(value, 0.0, 1.0))
                .fillna(0.0)
                .astype(float)
            )
            return result
        except IndicatorCalculationError:
            raise
        except Exception as exc:
            raise IndicatorCalculationError(f"RSI calculation failed: {exc}") from exc

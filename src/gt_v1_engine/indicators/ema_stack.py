import numpy as np
import pandas as pd

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection
from gt_v1_engine.indicators.strength import normalize_and_bucket


class EMAStackIndicator(BaseIndicator):
    name = "EMA_STACK"
    implemented = True
    description = "Fast, medium, and slow EMA stack trend indicator."

    def __init__(
        self,
        fast_period: int = 20,
        medium_period: int = 50,
        slow_period: int = 100,
    ) -> None:
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        require_columns(df, ["DateTime", "Close"], self.name)
        try:
            result = df.copy()
            close = pd.to_numeric(result["Close"], errors="coerce")
            ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
            ema_medium = close.ewm(span=self.medium_period, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

            result["EMA_STACK_FAST"] = ema_fast
            result["EMA_STACK_MEDIUM"] = ema_medium
            result["EMA_STACK_SLOW"] = ema_slow
            result[self.direction_column] = np.select(
                [(ema_fast > ema_medium) & (ema_medium > ema_slow), (ema_fast < ema_medium) & (ema_medium < ema_slow)],
                [IndicatorDirection.UP.value, IndicatorDirection.DOWN.value],
                default=IndicatorDirection.NO_SIGNAL.value,
            )
            raw_strength = ((ema_fast - ema_slow).abs() / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            result[self.strength_column] = (
                raw_strength.map(lambda value: normalize_and_bucket(value, 0.0, 0.01))
                .fillna(0.0)
                .astype(float)
            )
            return result
        except IndicatorCalculationError:
            raise
        except Exception as exc:
            raise IndicatorCalculationError(f"EMA_STACK calculation failed: {exc}") from exc

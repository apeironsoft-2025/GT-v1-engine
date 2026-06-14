import numpy as np
import pandas as pd

from gt_v1_engine.core.errors import IndicatorCalculationError
from gt_v1_engine.core.validation import require_columns
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorDirection
from gt_v1_engine.indicators.strength import normalize_and_bucket


class ADXIndicator(BaseIndicator):
    name = "ADX"
    implemented = True
    description = "Average Directional Index trend strength indicator."

    def __init__(self, period: int = 14, adx_threshold: float = 20) -> None:
        self.period = period
        self.adx_threshold = adx_threshold

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

            up_move = high.diff()
            down_move = low.shift(1) - low
            plus_dm = pd.Series(
                np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                index=result.index,
            )
            minus_dm = pd.Series(
                np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                index=result.index,
            )

            atr = true_range.ewm(alpha=1 / self.period, adjust=False).mean()
            smooth_plus_dm = plus_dm.ewm(alpha=1 / self.period, adjust=False).mean()
            smooth_minus_dm = minus_dm.ewm(alpha=1 / self.period, adjust=False).mean()
            plus_di = (100 * smooth_plus_dm / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            minus_di = (100 * smooth_minus_dm / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            dx = (
                100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            adx = dx.ewm(alpha=1 / self.period, adjust=False).mean().fillna(0.0)

            result["ADX_VALUE"] = adx
            result["ADX_PLUS_DI"] = plus_di
            result["ADX_MINUS_DI"] = minus_di
            result[self.direction_column] = np.select(
                [(plus_di > minus_di) & (adx >= self.adx_threshold), (minus_di > plus_di) & (adx >= self.adx_threshold)],
                [IndicatorDirection.UP.value, IndicatorDirection.DOWN.value],
                default=IndicatorDirection.NO_SIGNAL.value,
            )
            raw_strength = (adx / 50).replace([np.inf, -np.inf], np.nan)
            result[self.strength_column] = (
                raw_strength.map(lambda value: normalize_and_bucket(value, 0.0, 1.0))
                .fillna(0.0)
                .astype(float)
            )
            return result
        except IndicatorCalculationError:
            raise
        except Exception as exc:
            raise IndicatorCalculationError(f"ADX calculation failed: {exc}") from exc

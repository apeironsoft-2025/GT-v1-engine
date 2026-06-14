from abc import ABC
from enum import Enum

import pandas as pd
from pydantic import BaseModel


class IndicatorDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NO_SIGNAL = "NO_SIGNAL"


class IndicatorMetadata(BaseModel):
    name: str
    direction_column: str
    strength_column: str
    enabled: bool = True
    implemented: bool = False
    description: str | None = None


class IndicatorResult(BaseModel):
    name: str
    direction_column: str
    strength_column: str
    row_count: int
    up_count: int
    down_count: int
    no_signal_count: int
    min_strength: float | None
    max_strength: float | None
    average_strength: float | None


class BaseIndicator(ABC):
    name: str
    implemented: bool = False
    description: str | None = None

    @property
    def direction_column(self) -> str:
        return f"{self.name}_TD"

    @property
    def strength_column(self) -> str:
        return f"{self.name}_TS"

    @property
    def metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name=self.name,
            direction_column=self.direction_column,
            strength_column=self.strength_column,
            implemented=self.implemented,
            description=self.description,
        )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Indicator calculations are implemented in Step 03.")

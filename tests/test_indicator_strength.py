import math

import pytest

from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.strength import bucket_strength, direction_from_score


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0.0),
        (float("nan"), 0.0),
        (-1, 0.0),
        (0, 0.0),
        (0.1, 0.25),
        (0.25, 0.25),
        (0.26, 0.5),
        (0.5, 0.5),
        (0.51, 0.75),
        (0.75, 0.75),
        (0.76, 1.0),
        (10, 1.0),
    ],
)
def test_bucket_strength(value: float | None, expected: float) -> None:
    if value is not None and isinstance(value, float) and math.isnan(value):
        assert bucket_strength(value) == expected
    else:
        assert bucket_strength(value) == expected


def test_direction_from_score() -> None:
    assert direction_from_score(1) == IndicatorDirection.UP.value
    assert direction_from_score(-1) == IndicatorDirection.DOWN.value
    assert direction_from_score(0) == IndicatorDirection.NO_SIGNAL.value
    assert direction_from_score(0.01, dead_zone=0.05) == IndicatorDirection.NO_SIGNAL.value

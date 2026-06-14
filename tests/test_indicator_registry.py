import pytest

from gt_v1_engine.core.errors import IndicatorRegistryError
from gt_v1_engine.indicators.registry import (
    get_default_indicator_order,
    get_indicator_metadata,
    get_registered_indicators,
    validate_indicator_names,
)

EXPECTED_ORDER = ["MACD", "RSI", "ADX", "ATR", "BOLLINGER", "EMA_STACK"]


def test_default_order() -> None:
    assert get_default_indicator_order() == EXPECTED_ORDER


def test_six_indicators_are_registered() -> None:
    assert list(get_registered_indicators()) == EXPECTED_ORDER


@pytest.mark.parametrize("indicator", EXPECTED_ORDER)
def test_metadata_has_correct_columns(indicator: str) -> None:
    metadata = get_indicator_metadata(indicator)
    assert metadata.direction_column == f"{indicator}_TD"
    assert metadata.strength_column == f"{indicator}_TS"


def test_registry_lookup_is_case_insensitive() -> None:
    assert get_indicator_metadata("macd").name == "MACD"


def test_unknown_indicator_raises() -> None:
    with pytest.raises(IndicatorRegistryError):
        get_indicator_metadata("UNKNOWN")


def test_duplicate_indicator_selection_raises() -> None:
    with pytest.raises(IndicatorRegistryError):
        validate_indicator_names(["MACD", "macd"])


def test_empty_indicator_selection_raises() -> None:
    with pytest.raises(IndicatorRegistryError):
        validate_indicator_names([])

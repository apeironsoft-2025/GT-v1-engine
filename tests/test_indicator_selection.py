import pytest

from gt_v1_engine.core.errors import IndicatorRegistryError
from gt_v1_engine.indicators.registry import get_default_indicator_order
from gt_v1_engine.indicators.selection import parse_indicator_list, resolve_selected_indicators


def test_parse_indicator_list_none_returns_default_order() -> None:
    assert parse_indicator_list(None) == get_default_indicator_order()


def test_parse_indicator_list_string() -> None:
    assert parse_indicator_list("MACD,RSI") == ["MACD", "RSI"]


def test_parse_indicator_list_lowercase_string() -> None:
    assert parse_indicator_list("macd,rsi") == ["MACD", "RSI"]


def test_parse_indicator_list_unknown_indicator_raises() -> None:
    with pytest.raises(IndicatorRegistryError):
        parse_indicator_list("MACD,UNKNOWN")


def test_resolve_selected_from_cli_has_priority_over_config() -> None:
    assert resolve_selected_indicators("RSI", ["MACD"], get_default_indicator_order()) == ["RSI"]


def test_resolve_selected_from_config_when_cli_missing() -> None:
    assert resolve_selected_indicators(None, ["RSI", "MACD"], get_default_indicator_order()) == [
        "MACD",
        "RSI",
    ]


def test_resolve_selected_uses_default_when_cli_and_config_missing() -> None:
    assert resolve_selected_indicators(None, None, None) == get_default_indicator_order()

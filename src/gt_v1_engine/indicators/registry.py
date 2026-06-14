from gt_v1_engine.core.errors import IndicatorRegistryError
from gt_v1_engine.indicators.adx import ADXIndicator
from gt_v1_engine.indicators.atr import ATRIndicator
from gt_v1_engine.indicators.base import BaseIndicator, IndicatorMetadata
from gt_v1_engine.indicators.bollinger import BollingerIndicator
from gt_v1_engine.indicators.ema_stack import EMAStackIndicator
from gt_v1_engine.indicators.macd import MACDIndicator
from gt_v1_engine.indicators.rsi import RSIIndicator

_DEFAULT_INDICATOR_ORDER: list[str] = ["MACD", "RSI", "ADX", "ATR", "BOLLINGER", "EMA_STACK"]

_INDICATOR_CLASSES: dict[str, type[BaseIndicator]] = {
    "MACD": MACDIndicator,
    "RSI": RSIIndicator,
    "ADX": ADXIndicator,
    "ATR": ATRIndicator,
    "BOLLINGER": BollingerIndicator,
    "EMA_STACK": EMAStackIndicator,
}

_REGISTERED_INDICATORS: dict[str, IndicatorMetadata] = {
    name: _INDICATOR_CLASSES[name]().metadata for name in _DEFAULT_INDICATOR_ORDER
}


def _normalize_indicator_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise IndicatorRegistryError("Indicator name cannot be empty")
    return name.strip().upper()


def get_default_indicator_order() -> list[str]:
    return list(_DEFAULT_INDICATOR_ORDER)


def get_registered_indicators() -> dict[str, IndicatorMetadata]:
    return dict(_REGISTERED_INDICATORS)


def get_indicator_metadata(name: str) -> IndicatorMetadata:
    normalized = _normalize_indicator_name(name)
    try:
        return _REGISTERED_INDICATORS[normalized]
    except KeyError as exc:
        raise IndicatorRegistryError(f"Unknown indicator: {name}") from exc


def create_indicator(name: str) -> BaseIndicator:
    normalized = _normalize_indicator_name(name)
    if normalized not in _REGISTERED_INDICATORS:
        raise IndicatorRegistryError(f"Unknown indicator: {name}")
    if not _REGISTERED_INDICATORS[normalized].implemented:
        raise IndicatorRegistryError(f"Indicator implementation unavailable: {normalized}")
    indicator_class = _INDICATOR_CLASSES.get(normalized)
    if indicator_class is None:
        raise IndicatorRegistryError(f"Indicator implementation unavailable: {normalized}")
    return indicator_class()


def is_registered_indicator(name: str) -> bool:
    try:
        normalized = _normalize_indicator_name(name)
    except IndicatorRegistryError:
        return False
    return normalized in _REGISTERED_INDICATORS


def validate_indicator_names(indicators: list[str]) -> list[str]:
    if not indicators:
        raise IndicatorRegistryError("Indicator selection cannot be empty")

    normalized: list[str] = []
    seen: set[str] = set()
    for indicator in indicators:
        name = _normalize_indicator_name(indicator)
        if name in seen:
            raise IndicatorRegistryError(f"Duplicate indicator selected: {name}")
        if name not in _REGISTERED_INDICATORS:
            raise IndicatorRegistryError(f"Unknown indicator: {indicator}")
        normalized.append(name)
        seen.add(name)
    return normalized


def validate_indicator_order(selected: list[str], order: list[str]) -> list[str]:
    normalized_selected = validate_indicator_names(selected)
    normalized_order = validate_indicator_names(order)

    missing_from_order = [name for name in normalized_selected if name not in normalized_order]
    if missing_from_order:
        raise IndicatorRegistryError(
            "Selected indicator(s) missing from order: " + ", ".join(missing_from_order)
        )

    order_index = {name: index for index, name in enumerate(normalized_order)}
    return sorted(normalized_selected, key=lambda name: order_index[name])


def direction_column_for(indicator_name: str) -> str:
    return get_indicator_metadata(indicator_name).direction_column


def strength_column_for(indicator_name: str) -> str:
    return get_indicator_metadata(indicator_name).strength_column

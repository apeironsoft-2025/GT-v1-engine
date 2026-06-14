from pathlib import Path
from typing import Any

from gt_v1_engine.core.errors import IndicatorSelectionError
from gt_v1_engine.core.io_utils import read_yaml
from gt_v1_engine.indicators.registry import (
    get_default_indicator_order,
    validate_indicator_names,
    validate_indicator_order,
)


def parse_indicator_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return get_default_indicator_order()
    if isinstance(value, str):
        if not value.strip():
            return get_default_indicator_order()
        return validate_indicator_names([item.strip() for item in value.split(",")])
    return validate_indicator_names(value)


def load_default_indicator_config(path: Path) -> dict[str, Any]:
    payload = read_yaml(path)
    missing = [key for key in ("default_order", "enabled") if key not in payload]
    if missing:
        raise IndicatorSelectionError(
            f"Indicator config {path} missing required key(s): {', '.join(missing)}"
        )
    if not isinstance(payload["default_order"], list):
        raise IndicatorSelectionError(f"Indicator config {path} default_order must be a list")
    if not isinstance(payload["enabled"], list):
        raise IndicatorSelectionError(f"Indicator config {path} enabled must be a list")

    default_order = validate_indicator_names(payload["default_order"])
    enabled = validate_indicator_order(validate_indicator_names(payload["enabled"]), default_order)
    return {"default_order": default_order, "enabled": enabled}


def resolve_selected_indicators(
    cli_indicators: str | None,
    config_selected: list[str] | None,
    config_order: list[str] | None,
) -> list[str]:
    order = validate_indicator_names(config_order) if config_order else get_default_indicator_order()

    if cli_indicators is not None:
        selected = parse_indicator_list(cli_indicators)
    elif config_selected is not None:
        selected = validate_indicator_names(config_selected)
    else:
        selected = get_default_indicator_order()

    return validate_indicator_order(selected, order)

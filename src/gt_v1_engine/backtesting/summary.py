from datetime import UTC, datetime
from typing import Any

from gt_v1_engine.core.constants import NOT_ACTIVE


def safety_fields() -> dict[str, Any]:
    return {
        "production_activation_status": NOT_ACTIVE,
        "broker_order_allowed": False,
        "live_trading_allowed": False,
    }


def generated_at_utc() -> str:
    return datetime.now(UTC).isoformat()


def best_indicator_by_metric(
    indicator_summaries: dict[str, dict[str, Any]],
    metric: str,
) -> str | None:
    if not indicator_summaries:
        return None
    return max(
        indicator_summaries,
        key=lambda indicator: indicator_summaries[indicator].get(metric, 0.0),
    )

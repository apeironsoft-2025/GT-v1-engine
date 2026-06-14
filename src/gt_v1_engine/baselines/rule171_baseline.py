from copy import deepcopy
from typing import Any

RULE171_OLD_BASELINE: dict[str, Any] = {
    "baseline_name": "Rule171 First Profitable Baseline",
    "pair": "USDJPY",
    "timeframe": "M5",
    "start_datetime": "2025-12-01T00:00:00+00:00",
    "end_datetime": "2026-05-22T20:50:00+00:00",
    "released_signals": 246,
    "buy_signals": 121,
    "sell_signals": 125,
    "win_close_count": 146,
    "loss_close_count": 100,
    "take_profit_closes": 128,
    "stop_loss_closes": 77,
    "twelve_hour_closes": 41,
    "twelve_hour_win_closes": 18,
    "twelve_hour_loss_closes": 23,
    "total_realized_pips": 580.6,
    "average_pips_per_signal": 2.360163,
    "win_close_rate": 59.349593,
    "loss_close_rate": 40.650407,
    "production_activation_status": "NOT_ACTIVE",
}


def get_rule171_old_baseline() -> dict[str, Any]:
    return deepcopy(RULE171_OLD_BASELINE)

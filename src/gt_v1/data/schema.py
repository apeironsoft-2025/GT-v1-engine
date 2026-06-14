from gt_v1.core.constants import REQUIRED_MARKET_COLUMNS, STANDARD_MARKET_COLUMNS

COLUMN_ALIASES = {
    "datetime": "DateTime",
    "timestamp": "DateTime",
    "time": "DateTime",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "srp": "SRP",
}

__all__ = ["COLUMN_ALIASES", "REQUIRED_MARKET_COLUMNS", "STANDARD_MARKET_COLUMNS"]

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gt_v1.core.constants import NOT_ACTIVE, RULE171_NAME
from gt_v1.core.errors import RuleConfigError
from gt_v1.core.io_utils import read_yaml

PatternToken = Literal["UP", "DOWN", "NO_SIGNAL"]


class MarketConfig(BaseModel):
    default_pair: str = "USDJPY"
    default_timeframe: str = "M5"
    default_start: str
    default_end: str


class DataColumnsConfig(BaseModel):
    datetime_column: str = "DateTime"
    entry_price_column: str = "SRP"
    ohlc_columns: dict[str, str]


class PipSizeConfig(BaseModel):
    jpy_pair: float = 0.01
    non_jpy_pair: float = 0.0001
    override: float | None = None


class IndicatorSelectionConfig(BaseModel):
    selected: list[str]
    order: list[str]

    @field_validator("selected", "order")
    @classmethod
    def require_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("cannot be empty")
        return value

    @model_validator(mode="after")
    def selected_must_exist_in_order(self) -> "IndicatorSelectionConfig":
        missing = [indicator for indicator in self.selected if indicator not in self.order]
        if missing:
            raise ValueError(f"selected indicator(s) missing from order: {', '.join(missing)}")
        return self


class EntryConfig(BaseModel):
    strength_threshold: float = 4.5
    entry_confirmation_required: int = Field(default=3, ge=1)
    entry_opens_on_confirmation_number: int = 3
    preserve_cycle_on_no_signal: bool = True
    reset_cycle_on_opposite_signal: bool = True

    @model_validator(mode="after")
    def confirmation_numbers_match(self) -> "EntryConfig":
        if self.entry_opens_on_confirmation_number != self.entry_confirmation_required:
            raise ValueError(
                "entry_opens_on_confirmation_number must equal entry_confirmation_required"
            )
        return self


class PatternConfig(BaseModel):
    BUY: list[str]
    SELL: list[str]

    def all_patterns(self) -> list[str]:
        return [*self.BUY, *self.SELL]


class TradeManagementConfig(BaseModel):
    only_one_trade_open: bool = True
    blocked_rows_are_not_skipped: bool = True
    take_profit_pips: float = Field(default=30, gt=0)
    stop_loss_pips: float = Field(default=40, gt=0)
    max_holding_candles: int = Field(default=144, gt=0)
    allow_close_after_end: bool = False
    both_hit_same_candle_result: str = "LOSS_CLOSE"
    time_limit_win_condition: str = "realized_pips_gt_zero"


class OutputConfig(BaseModel):
    write_csv: bool = True
    write_summary_json: bool = True
    write_markdown_report: bool = True
    include_indicator_columns: bool = True
    include_entry_confirmation_audit: bool = True
    include_blocked_rows_count: bool = True


class SafetyConfig(BaseModel):
    research_only: bool = True
    live_trading_allowed: bool = False
    broker_order_allowed: bool = False
    production_activation_status: str = NOT_ACTIVE

    @model_validator(mode="after")
    def enforce_research_safety(self) -> "SafetyConfig":
        if self.live_trading_allowed:
            raise ValueError("safety.live_trading_allowed must be false")
        if self.broker_order_allowed:
            raise ValueError("safety.broker_order_allowed must be false")
        if self.production_activation_status != NOT_ACTIVE:
            raise ValueError("safety.production_activation_status must be NOT_ACTIVE")
        return self


class Rule171Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_name: str = RULE171_NAME
    production_activation_status: str = NOT_ACTIVE
    market: MarketConfig
    data: DataColumnsConfig
    pip_size: PipSizeConfig
    indicators: IndicatorSelectionConfig
    entry: EntryConfig
    patterns: PatternConfig
    trade_management: TradeManagementConfig
    outputs: OutputConfig
    safety: SafetyConfig

    @model_validator(mode="after")
    def validate_rule171(self) -> "Rule171Config":
        if self.rule_name != RULE171_NAME:
            raise ValueError("rule_name must be Rule171")
        if self.production_activation_status != NOT_ACTIVE:
            raise ValueError("production_activation_status must be NOT_ACTIVE")

        expected_length = len(self.indicators.selected)
        valid_tokens: set[str] = {"UP", "DOWN", "NO_SIGNAL"}
        for pattern in self.patterns.all_patterns():
            tokens = pattern.split("|")
            if len(tokens) != expected_length:
                raise ValueError(
                    f"pattern '{pattern}' length {len(tokens)} does not match selected "
                    f"indicator count {expected_length}"
                )
            invalid_tokens = [token for token in tokens if token not in valid_tokens]
            if invalid_tokens:
                raise ValueError(
                    f"pattern '{pattern}' contains invalid token(s): {', '.join(invalid_tokens)}"
                )
        return self


def load_rule171_config(path: Path) -> Rule171Config:
    payload = read_yaml(path)
    try:
        return Rule171Config.model_validate(payload)
    except ValidationError as exc:
        messages = "; ".join(error["msg"] for error in exc.errors())
        raise RuleConfigError(f"Invalid Rule171 config {path}: {messages}") from exc

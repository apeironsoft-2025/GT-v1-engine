from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gt_v1_engine.core.errors import RuleConfigError
from gt_v1_engine.rules.rule_config import load_rule171_config


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "rule171.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def default_payload() -> dict:
    path = Path("configs/rules/rule171.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_default_config_loads_successfully() -> None:
    config = load_rule171_config(Path("configs/rules/rule171.yaml"))
    assert config.rule_name == "Rule171"
    assert config.production_activation_status == "NOT_ACTIVE"


def test_unsafe_broker_order_allowed_true_fails(tmp_path: Path, default_payload: dict) -> None:
    payload = deepcopy(default_payload)
    payload["safety"]["broker_order_allowed"] = True
    with pytest.raises(RuleConfigError):
        load_rule171_config(_write_config(tmp_path, payload))


def test_pattern_length_mismatch_fails(tmp_path: Path, default_payload: dict) -> None:
    payload = deepcopy(default_payload)
    payload["patterns"]["BUY"] = ["UP|UP"]
    with pytest.raises(RuleConfigError):
        load_rule171_config(_write_config(tmp_path, payload))


def test_selected_indicator_missing_from_order_fails(tmp_path: Path, default_payload: dict) -> None:
    payload = deepcopy(default_payload)
    payload["indicators"]["order"].remove("MACD")
    with pytest.raises(RuleConfigError):
        load_rule171_config(_write_config(tmp_path, payload))


def test_tp_less_than_or_equal_zero_fails(tmp_path: Path, default_payload: dict) -> None:
    payload = deepcopy(default_payload)
    payload["trade_management"]["take_profit_pips"] = 0
    with pytest.raises(RuleConfigError):
        load_rule171_config(_write_config(tmp_path, payload))


def test_sl_less_than_or_equal_zero_fails(tmp_path: Path, default_payload: dict) -> None:
    payload = deepcopy(default_payload)
    payload["trade_management"]["stop_loss_pips"] = 0
    with pytest.raises(RuleConfigError):
        load_rule171_config(_write_config(tmp_path, payload))


def test_max_holding_less_than_or_equal_zero_fails(tmp_path: Path, default_payload: dict) -> None:
    payload = deepcopy(default_payload)
    payload["trade_management"]["max_holding_candles"] = 0
    with pytest.raises(RuleConfigError):
        load_rule171_config(_write_config(tmp_path, payload))

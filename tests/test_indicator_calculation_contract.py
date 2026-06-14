import subprocess
import sys

import pytest

from gt_v1_engine.indicators.registry import (
    create_indicator,
    get_default_indicator_order,
    get_registered_indicators,
)
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


@pytest.mark.parametrize("indicator_name", get_default_indicator_order())
def test_registered_indicator_calculation_contract(indicator_name: str) -> None:
    indicator = create_indicator(indicator_name)
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result[indicator.direction_column] != "NO_SIGNAL").any()


def test_registered_indicators_are_implemented() -> None:
    registered = get_registered_indicators()
    assert set(registered) == set(get_default_indicator_order())
    assert all(metadata.implemented for metadata in registered.values())


def test_list_indicators_cli_shows_implemented_true() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gt_v1_engine.cli", "list-indicators"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.count("true") >= 6
    for indicator_name in get_default_indicator_order():
        assert indicator_name in completed.stdout

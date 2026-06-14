from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.atr import ATRIndicator
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


def test_atr_calculation_contract() -> None:
    indicator = ATRIndicator()
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result["ATR_TD"] != IndicatorDirection.NO_SIGNAL.value).any()

from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.macd import MACDIndicator
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


def test_macd_calculation_contract() -> None:
    indicator = MACDIndicator()
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result["MACD_TD"] != IndicatorDirection.NO_SIGNAL.value).any()

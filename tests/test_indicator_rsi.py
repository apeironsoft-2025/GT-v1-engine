from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.rsi import RSIIndicator
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


def test_rsi_calculation_contract() -> None:
    indicator = RSIIndicator()
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result["RSI_TD"] != IndicatorDirection.NO_SIGNAL.value).any()

from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.adx import ADXIndicator
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


def test_adx_calculation_contract() -> None:
    indicator = ADXIndicator()
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result["ADX_TD"] != IndicatorDirection.NO_SIGNAL.value).any()

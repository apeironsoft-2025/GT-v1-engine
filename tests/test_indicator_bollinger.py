from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.bollinger import BollingerIndicator
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


def test_bollinger_calculation_contract() -> None:
    indicator = BollingerIndicator()
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result["BOLLINGER_TD"] != IndicatorDirection.NO_SIGNAL.value).any()

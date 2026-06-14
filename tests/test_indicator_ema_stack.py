from gt_v1_engine.indicators.base import IndicatorDirection
from gt_v1_engine.indicators.ema_stack import EMAStackIndicator
from tests.indicator_test_helpers import assert_indicator_contract, make_ohlc_dataframe


def test_ema_stack_calculation_contract() -> None:
    indicator = EMAStackIndicator()
    result = assert_indicator_contract(indicator, make_ohlc_dataframe())
    assert (result["EMA_STACK_TD"] != IndicatorDirection.NO_SIGNAL.value).any()

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TD_VALUES = {"UP", "DOWN", "NO_SIGNAL"}
TS_VALUES = {0.0, 0.25, 0.5, 0.75, 1.0}
BASE_COLUMNS = ["DateTime", "Open", "High", "Low", "Close"]

SCRIPT_CASES = [
    ("generate_rsi_td_ts.py", "calculate_rsi_td_ts", "RSI_TD", "RSI_TS"),
    ("generate_adx_td_ts.py", "calculate_adx_td_ts", "ADX_TD", "ADX_TS"),
    ("generate_atr_td_ts.py", "calculate_atr_td_ts", "ATR_TD", "ATR_TS"),
    (
        "generate_bollinger_td_ts.py",
        "calculate_bollinger_td_ts",
        "BOLLINGER_TD",
        "BOLLINGER_TS",
    ),
    (
        "generate_parabolic_sar_td_ts.py",
        "calculate_parabolic_sar_td_ts",
        "PARABOLIC_SAR_TD",
        "PARABOLIC_SAR_TS",
    ),
    (
        "generate_stochastic_td_ts.py",
        "calculate_stochastic_td_ts",
        "STOCHASTIC_TD",
        "STOCHASTIC_TS",
    ),
    (
        "generate_ichimoku_td_ts.py",
        "calculate_ichimoku_td_ts",
        "ICHIMOKU_TD",
        "ICHIMOKU_TS",
    ),
]


def load_script(script_name: str):
    script_path = SCRIPTS_DIR / script_name
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_cleaned_dataframe(row_count: int = 140) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        trend = index * 0.08
        cycle = (index % 12 - 6) * 0.015
        close = 150.0 + trend + cycle
        open_price = close - 0.04
        high = close + 0.12 + (index % 5) * 0.01
        low = close - 0.12 - (index % 4) * 0.01
        rows.append(
            {
                "DateTime": f"2026-01-01 00:{index % 60:02d}:00",
                "Open": round(open_price, 5),
                "High": round(high, 5),
                "Low": round(low, 5),
                "Close": round(close, 5),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("script_name,function_name,td_column,ts_column", SCRIPT_CASES)
def test_standalone_indicator_script_contract(
    script_name: str,
    function_name: str,
    td_column: str,
    ts_column: str,
) -> None:
    module = load_script(script_name)
    df = sample_cleaned_dataframe()

    result = getattr(module, function_name)(df)

    assert len(result) == len(df)
    assert list(result.columns) == [*BASE_COLUMNS, td_column, ts_column]
    assert set(result[td_column].unique()) <= TD_VALUES
    assert set(float(value) for value in result[ts_column].unique()) <= TS_VALUES
    assert (result.loc[result[td_column] == "NO_SIGNAL", ts_column] == 0).all()
    assert result.iloc[0][td_column] == "NO_SIGNAL"
    assert float(result.iloc[0][ts_column]) == 0.0


@pytest.mark.parametrize("script_name,function_name,td_column,ts_column", SCRIPT_CASES)
def test_standalone_indicator_handles_warmup_rows_without_crashing(
    script_name: str,
    function_name: str,
    td_column: str,
    ts_column: str,
) -> None:
    module = load_script(script_name)
    df = sample_cleaned_dataframe(row_count=5)

    result = getattr(module, function_name)(df)

    assert len(result) == len(df)
    assert set(result[td_column].unique()) <= TD_VALUES
    assert set(float(value) for value in result[ts_column].unique()) <= TS_VALUES


def test_standalone_indicator_missing_required_column_raises_clear_error() -> None:
    module = load_script("generate_rsi_td_ts.py")
    df = sample_cleaned_dataframe().drop(columns=["High"])

    with pytest.raises(ValueError, match="Input CSV missing required columns: High"):
        module.calculate_rsi_td_ts(df)

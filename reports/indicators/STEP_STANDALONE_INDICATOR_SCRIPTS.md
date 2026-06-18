# Standalone Indicator TD/TS Scripts

## Branch Name

`feature/add-standalone-indicator-scripts`

## Files Created/Changed

- `scripts/standalone_indicator_common.py`
- `scripts/generate_rsi_td_ts.py`
- `scripts/generate_adx_td_ts.py`
- `scripts/generate_atr_td_ts.py`
- `scripts/generate_bollinger_td_ts.py`
- `scripts/generate_parabolic_sar_td_ts.py`
- `scripts/generate_stochastic_td_ts.py`
- `scripts/generate_ichimoku_td_ts.py`
- `tests/test_standalone_indicator_scripts.py`
- `reports/indicators/STEP_STANDALONE_INDICATOR_SCRIPTS.md`

## Scripts

- `scripts/generate_rsi_td_ts.py`
- `scripts/generate_adx_td_ts.py`
- `scripts/generate_atr_td_ts.py`
- `scripts/generate_bollinger_td_ts.py`
- `scripts/generate_parabolic_sar_td_ts.py`
- `scripts/generate_stochastic_td_ts.py`
- `scripts/generate_ichimoku_td_ts.py`

## Input File Used

`F:\GT-v1-shared-storage\cleaned\USDJPY_M5_cleaned.csv`

## Output CSV Results

### RSI

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_rsi_td_ts.csv`
- Row count: 100000
- TD counts:
  - UP: 36160
  - NO_SIGNAL: 32552
  - DOWN: 31288
- TS counts:
  - 0.00: 32552
  - 0.25: 27428
  - 0.50: 19359
  - 0.75: 11515
  - 1.00: 9146

### ADX

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_adx_td_ts.csv`
- Row count: 100000
- TD counts:
  - NO_SIGNAL: 40030
  - DOWN: 30798
  - UP: 29172
- TS counts:
  - 0.00: 40030
  - 0.25: 21068
  - 0.50: 14955
  - 0.75: 15936
  - 1.00: 8011

### ATR

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_atr_td_ts.csv`
- Row count: 100000
- TD counts:
  - UP: 50003
  - DOWN: 48078
  - NO_SIGNAL: 1919
- TS counts:
  - 0.00: 1919
  - 0.25: 40697
  - 0.50: 18281
  - 0.75: 17485
  - 1.00: 21618

### Bollinger

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_bollinger_td_ts.csv`
- Row count: 100000
- TD counts:
  - UP: 52118
  - DOWN: 47844
  - NO_SIGNAL: 38
- TS counts:
  - 0.00: 38
  - 0.25: 22560
  - 0.50: 25120
  - 0.75: 41969
  - 1.00: 10313

### Parabolic SAR

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_parabolic_sar_td_ts.csv`
- Row count: 100000
- TD counts:
  - UP: 51712
  - DOWN: 48280
  - NO_SIGNAL: 8
- TS counts:
  - 0.00: 20
  - 0.25: 4531
  - 0.50: 11551
  - 0.75: 16736
  - 1.00: 67162

### Stochastic

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_stochastic_td_ts.csv`
- Row count: 100000
- TD counts:
  - NO_SIGNAL: 36832
  - UP: 33188
  - DOWN: 29980
- TS counts:
  - 0.00: 36832
  - 0.25: 7546
  - 0.50: 9205
  - 0.75: 20523
  - 1.00: 25894

### Ichimoku

- Output CSV: `F:\GT-v1-shared-storage\indicators\USDJPY_M5_ichimoku_td_ts.csv`
- Row count: 100000
- TD counts:
  - UP: 50225
  - DOWN: 44850
  - NO_SIGNAL: 4925
- TS counts:
  - 0.00: 4925
  - 0.25: 36330
  - 0.50: 19131
  - 0.75: 18166
  - 1.00: 21448

## Validation Commands Run

- `python -m compileall scripts tests`
- `python scripts\generate_rsi_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python scripts\generate_adx_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python scripts\generate_atr_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python scripts\generate_bollinger_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python scripts\generate_parabolic_sar_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python scripts\generate_stochastic_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python scripts\generate_ichimoku_td_ts.py --fileName USDJPY_M5_cleaned.csv`
- `python -m pytest`
- `python -m pytest --basetemp .pytest_tmp`

## Test Result

- `python -m compileall scripts tests`: PASS
- `python -m pytest`: initial run did not complete because pytest could not access `C:\Users\User\AppData\Local\Temp\pytest-of-User`; the new standalone indicator tests passed before the temp fixture errors.
- `python -m pytest --basetemp .pytest_tmp`: PASS, 173 passed, 1 warning.
- All seven real-data smoke commands: PASS, each printed `status SUCCESS`.

## Known Limitations

- This stage creates standalone Python indicator CSV generators only. No Spring Boot API, orchestration, merged datasets, or rule executor integration was added.
- ATR and Ichimoku rolling quantile strength calculations use a straightforward per-row implementation; this is clear and deterministic, but slower than a vectorized approximation on 100000-row files.
- Parabolic SAR is implemented manually without TA-Lib and uses the requested fixed step, increment, and maximum parameters.

## Final Status

PASS

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gt_v1_engine.backtesting.indicator_backtester import backtest_all_indicators
from gt_v1_engine.core.constants import NOT_ACTIVE
from gt_v1_engine.core.errors import DataValidationError, GTV1EngineError, IndicatorCalculationError
from gt_v1_engine.core.io_utils import ensure_file_exists
from gt_v1_engine.indicators.executor import run_indicator_executor
from gt_v1_engine.indicators.registry import get_default_indicator_order, validate_indicator_order
from gt_v1_engine.reports.markdown_report import write_smoke_pipeline_report
from gt_v1_engine.rules.rule171 import Rule171ExecutionOverrides, run_rule171_backtest
from gt_v1_engine.rules.rule_config import load_rule171_config


def run_smoke_pipeline(
    input_path: Path,
    pair: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    indicators: list[str],
    config_path: Path,
    output_root: Path,
    horizon_candles: int,
    indicator_target_pips: float,
    indicator_stop_pips: float,
    rule171_pip_size: float | None,
    rule171_strength_threshold: float | None,
    rule171_entry_confirmation_required: int | None,
    rule171_take_profit_pips: float | None,
    rule171_stop_loss_pips: float | None,
    rule171_max_holding_candles: int | None,
) -> dict[str, Any]:
    try:
        ensure_file_exists(input_path)
        ensure_file_exists(config_path)
        _validate_runtime(pair, timeframe, horizon_candles, indicator_target_pips, indicator_stop_pips)
        selected = validate_indicator_order(indicators or get_default_indicator_order(), get_default_indicator_order())
        config = load_rule171_config(config_path)
        effective_start = start if start is not None else config.market.default_start
        effective_end = end if end is not None else config.market.default_end
        paths = _output_paths(output_root, pair, timeframe, start, end, selected)
        for directory in (
            paths["indicator_csv"].parent,
            paths["indicator_backtest_summary"].parent,
            paths["rule171_csv"].parent,
            paths["markdown_report"].parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        _, indicator_summary = run_indicator_executor(
            input_path=input_path,
            pair=pair,
            timeframe=timeframe,
            indicators=selected,
            output_csv=paths["indicator_csv"],
            output_parquet=paths["indicator_parquet"],
            summary_json=paths["indicator_summary"],
        )
        indicator_backtest_summary = backtest_all_indicators(
            input_path=paths["indicator_csv"],
            pair=pair,
            timeframe=timeframe,
            indicators=selected,
            start=start,
            end=end,
            horizon_candles=horizon_candles,
            target_pips=indicator_target_pips,
            stop_pips=indicator_stop_pips,
            pip_size=rule171_pip_size,
            output_dir=paths["indicator_backtest_dir"],
            summary_json=paths["indicator_backtest_summary"],
        )
        _, rule171_summary = run_rule171_backtest(
            input_path=paths["indicator_csv"],
            config_path=config_path,
            output_csv=paths["rule171_csv"],
            output_summary=paths["rule171_summary"],
            overrides=Rule171ExecutionOverrides(
                pair=pair,
                timeframe=timeframe,
                indicators=selected,
                start=effective_start,
                end=effective_end,
                pip_size=rule171_pip_size,
                strength_threshold=rule171_strength_threshold,
                entry_confirmation_required=rule171_entry_confirmation_required,
                take_profit_pips=rule171_take_profit_pips,
                stop_loss_pips=rule171_stop_loss_pips,
                max_holding_candles=rule171_max_holding_candles,
            ),
        )

        summary = {
            "pipeline_name": "GT-v1-engine smoke pipeline",
            "validation_status": "PASS",
            "pair": pair.strip(),
            "timeframe": timeframe.strip(),
            "start_datetime": effective_start,
            "end_datetime": effective_end,
            "selected_indicators": selected,
            "input_path": str(Path(input_path)),
            "output_root": str(Path(output_root)),
            "indicator_dataset_csv": str(paths["indicator_csv"]),
            "indicator_dataset_parquet": str(paths["indicator_parquet"]),
            "indicator_summary_json": str(paths["indicator_summary"]),
            "indicator_backtest_summary_json": str(paths["indicator_backtest_summary"]),
            "rule171_output_csv": str(paths["rule171_csv"]),
            "rule171_summary_json": str(paths["rule171_summary"]),
            "markdown_report_path": str(paths["markdown_report"]),
            "indicator_executor_status": indicator_summary["validation_status"],
            "indicator_backtest_status": indicator_backtest_summary["validation_status"],
            "rule171_status": rule171_summary["validation_status"],
            "rule171_released_signals": rule171_summary["released_signals"],
            "rule171_total_realized_pips": rule171_summary["total_realized_pips"],
            "production_activation_status": NOT_ACTIVE,
            "live_trading_allowed": False,
            "broker_order_allowed": False,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        write_smoke_pipeline_report(
            summary,
            indicator_summary,
            indicator_backtest_summary,
            rule171_summary,
            paths["markdown_report"],
        )
        return summary
    except GTV1EngineError:
        raise
    except Exception as exc:
        raise IndicatorCalculationError(f"Smoke pipeline failed: {exc}") from exc


def _validate_runtime(
    pair: str,
    timeframe: str,
    horizon_candles: int,
    indicator_target_pips: float,
    indicator_stop_pips: float,
) -> None:
    if not isinstance(pair, str) or not pair.strip():
        raise DataValidationError("pair cannot be empty")
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise DataValidationError("timeframe cannot be empty")
    if horizon_candles <= 0:
        raise DataValidationError("horizon_candles must be greater than 0")
    if indicator_target_pips <= 0:
        raise DataValidationError("indicator_target_pips must be greater than 0")
    if indicator_stop_pips <= 0:
        raise DataValidationError("indicator_stop_pips must be greater than 0")


def _output_paths(
    output_root: Path,
    pair: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    selected: list[str],
) -> dict[str, Path]:
    pair_token = pair.strip()
    timeframe_token = timeframe.strip()
    start_token = _date_token(start, "ALL_START")
    end_token = _date_token(end, "ALL_END")
    suffix = "6I" if selected == get_default_indicator_order() else f"{len(selected)}I"
    root = Path(output_root)
    return {
        "indicator_csv": root / "indicators" / f"{pair_token}_{timeframe_token}_{suffix}.csv",
        "indicator_parquet": root / "indicators" / f"{pair_token}_{timeframe_token}_{suffix}.parquet",
        "indicator_summary": root / "indicators" / f"{pair_token}_{timeframe_token}_{suffix}_summary.json",
        "indicator_backtest_dir": root / "backtests" / "indicators",
        "indicator_backtest_summary": root
        / "backtests"
        / "indicators"
        / f"{pair_token}_{timeframe_token}_all_indicators_summary.json",
        "rule171_csv": root
        / "backtests"
        / "rules"
        / f"rule171_{pair_token}_{timeframe_token}_{start_token}_{end_token}.csv",
        "rule171_summary": root
        / "backtests"
        / "rules"
        / f"rule171_{pair_token}_{timeframe_token}_{start_token}_{end_token}_summary.json",
        "markdown_report": root
        / "reports"
        / f"smoke_pipeline_{pair_token}_{timeframe_token}_{start_token}_{end_token}.md",
    }


def _date_token(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%d")
    except ValueError:
        return fallback

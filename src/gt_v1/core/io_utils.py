import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gt_v1.core.errors import ConfigError, FileMissingError


def ensure_file_exists(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileMissingError(f"{resolved} not found")
    return resolved


def ensure_parent_dir(path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path | str) -> dict[str, Any]:
    resolved = ensure_file_exists(path)
    try:
        with resolved.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{resolved} must contain a YAML mapping")
    return payload


def write_json(path: Path | str, payload: Any) -> None:
    resolved = Path(path)
    ensure_parent_dir(resolved)
    with resolved.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
        file.write("\n")


def write_dataframe_csv(df: pd.DataFrame, path: Path | str) -> None:
    resolved = Path(path)
    ensure_parent_dir(resolved)
    df.to_csv(resolved, index=False)


def write_dataframe_parquet(df: pd.DataFrame, path: Path | str) -> None:
    resolved = Path(path)
    ensure_parent_dir(resolved)
    df.to_parquet(resolved, index=False)

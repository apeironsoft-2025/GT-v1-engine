from typing import Optional


def infer_pip_size(pair: str) -> float:
    if "JPY" in pair.upper():
        return 0.01
    return 0.0001


def resolve_pip_size(pair: str, override: Optional[float]) -> float:
    if override is not None:
        return override
    return infer_pip_size(pair)


def pips_to_price(pips: float, pip_size: float) -> float:
    return pips * pip_size

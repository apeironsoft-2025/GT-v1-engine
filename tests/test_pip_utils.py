from gt_v1.core.pip_utils import infer_pip_size, resolve_pip_size


def test_usdjpy_returns_jpy_pip_size() -> None:
    assert infer_pip_size("USDJPY") == 0.01


def test_eurusd_returns_non_jpy_pip_size() -> None:
    assert infer_pip_size("EURUSD") == 0.0001


def test_override_works() -> None:
    assert resolve_pip_size("USDJPY", 0.25) == 0.25

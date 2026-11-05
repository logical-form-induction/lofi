import pytest
from tests.support.installed import install

from lofi.registry import UnknownTagError
from lofi.run import StackBootstrapPricing
from lofi.run._cost_model import cost_model


def test_a_declared_cost_model_is_built_by_its_installed_plugin() -> None:
    assert cost_model(StackBootstrapPricing(resamples=50)).resamples == 50


def test_a_cost_model_without_exactly_one_installed_plugin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "pricing", ())

    with pytest.raises(UnknownTagError, match="'stack-bootstrap'"):
        cost_model(StackBootstrapPricing())

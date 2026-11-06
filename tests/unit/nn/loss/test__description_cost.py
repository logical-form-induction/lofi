from math import log
from typing import Final

import pytest

from lofi.nn.loss._description_cost import (
    LOG_ALPHABET_PLUGIN,
    PER_CONSTRUCTOR_PLUGIN,
    UNIFORM_PLUGIN,
    LogAlphabetDescriptionCost,
    PerConstructorDescriptionCost,
    UniformDescriptionCost,
)
from lofi.spec.losses.simplicity import (
    LogAlphabetSymbolCostSpec,
    PerConstructorSymbolCostSpec,
    UniformSymbolCostSpec,
)
from lofi.terms import Operator, OperatorIndex, Signature

SIGNATURE: Final = Signature(
    "description-cost-fixture",
    ("T",),
    (
        Operator("leaf", "T", family="atom"),
        Operator("branch", "T", arg_sorts=("T", "T"), family="constructor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)


def test_uniform_cost_prices_every_constructor_and_not_padding() -> None:
    policy = UniformDescriptionCost(0.75)

    assert policy.costs(INDEX) == (0.0, 0.75, 0.75)


def test_log_alphabet_cost_is_derived_from_the_owned_index() -> None:
    policy = LogAlphabetDescriptionCost()

    assert policy.costs(INDEX) == (0.0, log(2), log(2))


def test_per_constructor_cost_follows_operator_code_order() -> None:
    policy = PerConstructorDescriptionCost({"branch": 2.0, "leaf": 0.25})

    assert policy.costs(INDEX) == (0.0, 0.25, 2.0)


@pytest.mark.parametrize(
    ("costs", "detail"),
    [
        ({"leaf": 0.25}, "missing"),
        ({"leaf": 0.25, "branch": 2.0, "other": 1.0}, "unknown"),
    ],
)
def test_per_constructor_cost_refuses_an_inexact_inventory(
    costs: dict[str, float], detail: str
) -> None:
    policy = PerConstructorDescriptionCost(costs)

    with pytest.raises(ValueError, match=detail):
        policy.costs(INDEX)


def test_each_builtin_plugin_builds_its_exact_policy() -> None:
    uniform = UNIFORM_PLUGIN.construct(UniformSymbolCostSpec(cost=0.5))
    logarithmic = LOG_ALPHABET_PLUGIN.construct(LogAlphabetSymbolCostSpec())
    keyed = PER_CONSTRUCTOR_PLUGIN.construct(
        PerConstructorSymbolCostSpec(costs={"leaf": 0.25, "branch": 2.0})
    )

    assert isinstance(uniform, UniformDescriptionCost)
    assert isinstance(logarithmic, LogAlphabetDescriptionCost)
    assert isinstance(keyed, PerConstructorDescriptionCost)
    assert uniform.costs(INDEX) == (0.0, 0.5, 0.5)
    assert logarithmic.costs(INDEX) == (0.0, log(2), log(2))
    assert keyed.costs(INDEX) == (0.0, 0.25, 2.0)

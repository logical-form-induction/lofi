from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.simplicity import (
    LogAlphabetSymbolCostSpec,
    PerConstructorSymbolCostSpec,
    SimplicityLossSpec,
    SimplicityScope,
    UniformSymbolCostSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_tag_reaches_the_simplicity_variant_through_the_family_root() -> None:
    declared = LossSpec.model_validate(
        {"kind": "simplicity", "readout": "types", "scope": "all-cells"}
    )

    assert type(declared) is SimplicityLossSpec
    assert declared == SimplicityLossSpec(
        readout="types", scope=SimplicityScope.ALL_CELLS
    )
    assert LossSpec.__variants__["simplicity"] is SimplicityLossSpec


def test_the_lexicon_is_priced_by_default_and_the_readout_is_a_dependency() -> None:
    declared = SimplicityLossSpec(readout="types")

    assert declared.scope is SimplicityScope.LEXICON
    assert declared.symbol_cost == UniformSymbolCostSpec()
    assert declared.dependencies() == (Dependency("types", "simplicity-readout"),)
    assert declared.declared_output is None


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "simplicity"}, "readout"),
        ({"kind": "simplicity", "readout": ""}, "simplicity readout role"),
        ({"kind": "simplicity", "readout": True}, "readout"),
        ({"kind": "simplicity", "readout": "types", "scope": "interior"}, "scope"),
        ({"kind": "simplicity", "readout": "types", "codec": "grammar"}, "codec"),
        (
            {
                "kind": "simplicity",
                "readout": "types",
                "symbol_cost": {"kind": "per-constructor", "costs": {}},
            },
            "at least one",
        ),
    ],
)
def test_a_missing_empty_wrong_typed_unknown_or_extra_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "simplicity", "readout": "types", "scope": "lexicon", '
        '"symbol_cost": {"kind": "uniform", "cost": 1.0}, '
        '"declared_output": null}'
    )
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared


def test_all_builtin_symbol_cost_variants_validate_through_the_nested_union() -> None:
    uniform = SimplicityLossSpec(
        readout="types", symbol_cost=UniformSymbolCostSpec(cost=0.5)
    )
    logarithmic = SimplicityLossSpec(
        readout="types", symbol_cost=LogAlphabetSymbolCostSpec()
    )
    keyed = SimplicityLossSpec(
        readout="types",
        symbol_cost=PerConstructorSymbolCostSpec(costs={"/": 2.0, "N": 0.5}),
    )

    assert uniform.symbol_cost.cost == 0.5
    assert logarithmic.symbol_cost.kind == "log-alphabet"
    assert keyed.symbol_cost.costs == {"/": 2.0, "N": 0.5}

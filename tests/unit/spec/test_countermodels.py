import didactic.api as dx
import pytest

from lofi.spec.countermodels import CountermodelSpec


def test_countermodel_settings_lower_to_one_bounded_interior_value() -> None:
    declaration = CountermodelSpec(
        method="exhaustive",
        carriers={"e": 2},
        structures=8,
        seed=3,
        full_function_space_cap=16,
    )

    settings = declaration.settings()

    assert settings.method == "exhaustive"
    assert dict(settings.carriers) == {"e": 2}
    assert settings.structures == 8


def test_countermodel_declarations_refuse_an_empty_carrier_inventory() -> None:
    with pytest.raises(dx.ValidationError, match="carrier"):
        CountermodelSpec(
            method="random",
            carriers={},
            structures=1,
            seed=0,
            full_function_space_cap=1,
        )

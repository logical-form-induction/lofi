import pytest
from tests.language import language

from lofi.lf.models import CountermodelFilter, CountermodelSettings
from lofi.spec.language import (
    ConstantSpec,
    LanguageIso,
    PrefixNotationSpec,
    registered_language,
)


def settings(
    *,
    structures: int = 4,
    full_function_space_cap: int = 16,
) -> CountermodelSettings:
    return CountermodelSettings(
        "exhaustive",
        carriers={"e": 1, "v": 1},
        structures=structures,
        seed=5,
        full_function_space_cap=full_function_space_cap,
    )


def test_exhaustive_search_returns_a_checked_countermodel() -> None:
    declared = language(("rain", "wet"))
    search = CountermodelFilter(declared, settings())
    rain = declared.constant("rain")
    wet = declared.constant("wet")

    countermodel = search.refute(rain, wet)

    assert countermodel is not None
    assert countermodel.structure.holds(rain)
    assert countermodel.structure.holds(countermodel.negated_goal)


def test_search_uses_the_designated_negation_when_roles_are_ambiguous() -> None:
    source = registered_language("neo-davidsonian")
    declared = LanguageIso().forward(
        source.with_(
            constants=(
                *source.constants,
                ConstantSpec(
                    name="complement",
                    type="t -> t",
                    classes=("sentential", "negation"),
                    notation=PrefixNotationSpec(
                        ascii="complement",
                        unicode="∁",
                        precedence=6,
                    ),
                ),
                ConstantSpec(name="rain", type="t"),
                ConstantSpec(name="wet", type="t"),
            ),
            logic=source.logic.with_(negation="not"),
        )
    )
    search = CountermodelFilter(declared, settings())

    countermodel = search.refute(
        declared.constant("rain"),
        declared.constant("wet"),
    )

    assert countermodel is not None
    assert countermodel.negated_goal.args[0].op == declared.constant("not").op


def test_a_profile_theory_that_fails_excludes_a_candidate_structure() -> None:
    source = registered_language("neo-davidsonian")
    declared = LanguageIso().forward(
        source.with_(
            constants=(
                *source.constants,
                ConstantSpec(name="rain", type="t"),
                ConstantSpec(name="wet", type="t"),
            ),
            logic=source.logic.with_(postulates=("wet",)),
        )
    )
    search = CountermodelFilter(declared, settings())
    rain = declared.constant("rain")
    wet = declared.constant("wet")

    assert search.refute(rain, wet) is None


def test_search_refuses_a_partial_function_space_instead_of_sampling_it() -> None:
    declared = language(("dog",), types={"dog": "e -> t"})

    with pytest.raises(ValueError, match="full_function_space_cap=1"):
        CountermodelFilter(
            declared,
            settings(full_function_space_cap=1),
        )


def test_random_search_is_reproducible_from_its_declared_seed() -> None:
    declared = language(("rain", "wet"))
    random_settings = CountermodelSettings(
        "random",
        carriers={"e": 1, "v": 1},
        structures=2,
        seed=13,
        full_function_space_cap=16,
    )

    first = CountermodelFilter(declared, random_settings)
    second = CountermodelFilter(declared, random_settings)

    assert tuple(item.name for item in first.structures) == tuple(
        item.name for item in second.structures
    )
    assert tuple(item.denotations for item in first.structures) == tuple(
        item.denotations for item in second.structures
    )

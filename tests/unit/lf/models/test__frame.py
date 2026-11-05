from typing import TYPE_CHECKING

import pytest

from lofi.lf.models import Element, Frame, FrameTypeError, FunctionValue
from lofi.spec.language import LanguageIso, registered_language

if TYPE_CHECKING:
    from lofi.spec.language import LanguageAssembly


def frame(
    *,
    entity_size: int = 2,
    full_function_space_cap: int = 16,
    henkin_pool: int = 3,
    seed: int = 7,
) -> tuple[LanguageAssembly, Frame]:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    return language, Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": entity_size, "v": 1},
        full_function_space_cap=full_function_space_cap,
        henkin_pool=henkin_pool,
        seed=seed,
    )


def test_atomic_carriers_and_truth_are_finite_domains() -> None:
    language, declared = frame(entity_size=3)

    entities = declared.domain(language.read_type("e"))
    truths = declared.domain(language.read_type("t"))

    assert entities.full
    assert entities.values == tuple(Element("e", at) for at in range(3))
    assert truths.full
    assert truths.values == (False, True)


def test_a_small_function_type_uses_its_complete_function_space() -> None:
    language, declared = frame(full_function_space_cap=4)

    functions = declared.domain(language.read_type("e -> t"))
    entities = declared.domain(language.read_type("e")).values

    assert functions.full
    assert len(functions.values) == 4
    assert {
        tuple(value(argument) for argument in entities)
        for value in functions.values
        if isinstance(value, FunctionValue)
    } == {(False, False), (False, True), (True, False), (True, True)}


def test_a_large_function_type_uses_a_seeded_henkin_pool() -> None:
    language, first = frame(full_function_space_cap=3, henkin_pool=3, seed=19)
    other_language, second = frame(full_function_space_cap=3, henkin_pool=3, seed=19)
    function_type = language.read_type("e -> t")

    # Accessing another domain first must not advance a shared random stream.
    first.domain(language.read_type("v"))

    left = first.domain(function_type)
    right = second.domain(other_language.read_type("e -> t"))

    assert not left.full
    assert len(left.values) == 3
    assert left == right


def test_a_function_value_refuses_an_argument_outside_its_domain() -> None:
    function = FunctionValue(
        (Element("e", 0), Element("e", 1)),
        (False, True),
        (True, True),
    )

    assert function(Element("e", 1)) is True

    with pytest.raises(ValueError, match="outside the function domain"):
        function(Element("e", 2))


def test_a_function_value_refuses_an_unknown_result() -> None:
    function = FunctionValue(
        (Element("e", 0), Element("e", 1)),
        (False, True),
        (False, True),
    )

    with pytest.raises(ValueError, match="is unknown"):
        function(Element("e", 0))

    assert function(Element("e", 1)) is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("full_function_space_cap", 0), ("henkin_pool", 0), ("seed", -1)],
)
def test_frame_limits_are_positive_or_non_negative(field: str, value: int) -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    settings = {
        "full_function_space_cap": 8,
        "henkin_pool": 2,
        "seed": 0,
    }
    settings[field] = value

    with pytest.raises(ValueError, match=r"positive|negative"):
        Frame(
            language.store,
            function_type=language.function_type,
            truth_type="t",
            carriers={"e": 2, "v": 1},
            **settings,
        )


def test_an_atomic_type_needs_a_declared_carrier() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    declared = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": 2},
        full_function_space_cap=8,
        henkin_pool=2,
        seed=0,
    )

    with pytest.raises(ValueError, match="no carrier for atomic type 'v'"):
        declared.domain(language.read_type("v"))


def test_a_frame_refuses_a_non_ground_type_explicitly() -> None:
    language, declared = frame()

    with pytest.raises(FrameTypeError, match="atomic or functional"):
        declared.domain(language.store.fvar(language.type_sort, "a"))

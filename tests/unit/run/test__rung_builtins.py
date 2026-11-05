from types import MappingProxyType

import pytest

from lofi.run import (
    CorpusScaleRung,
    FallbackRung,
    LanguageRung,
    SeedRung,
    UnpreregisteredCellsRung,
)
from lofi.run._contraction import (
    DropInOrder,
    Fallback,
    PreregisteredCells,
    SeedFloor,
)
from lofi.run._rung import RungContext, contraction_rung

CONTEXT = RungContext(
    MappingProxyType(
        {
            "size": ("data.size", frozenset({"10"})),
            "language": ("data.language", frozenset({"en", "de"})),
        }
    ),
    seeds=5,
    minimum_seeds=2,
)


def test_each_builtin_rung_lowers_to_the_transformation_it_states() -> None:
    assert isinstance(
        contraction_rung(UnpreregisteredCellsRung(), CONTEXT), PreregisteredCells
    )
    assert isinstance(
        contraction_rung(LanguageRung(axis="data.language", order=("de",)), CONTEXT),
        DropInOrder,
    )
    assert isinstance(
        contraction_rung(CorpusScaleRung(axis="data.size", order=(10,)), CONTEXT),
        DropInOrder,
    )
    assert isinstance(contraction_rung(SeedRung(floor=3), CONTEXT), SeedFloor)
    assert isinstance(contraction_rung(FallbackRung(), CONTEXT), Fallback)


@pytest.mark.parametrize(
    ("floor", "reason"),
    [(6, "exceeds the 5 seeds"), (1, "below the 2 seeds")],
)
def test_a_seed_floor_outside_the_sweeps_seeds_is_refused(
    floor: int, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        contraction_rung(SeedRung(floor=floor), CONTEXT)

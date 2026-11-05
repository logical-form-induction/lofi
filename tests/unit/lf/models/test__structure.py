import pytest

from lofi.lf.models import Frame, Structure
from lofi.spec.language import LanguageIso, registered_language


def test_a_structure_keys_denotations_by_vocabulary_identifier() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=3,
    )
    structure = Structure("fixture", frame, {7: True})

    assert structure.denotation(7) is True

    with pytest.raises(ValueError, match="no denotation for constant 8"):
        structure.denotation(8)


def test_a_structure_refuses_a_negative_vocabulary_identifier() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=3,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        Structure("fixture", frame, {-1: False})

import didactic.api as dx
import pytest

from lofi.spec.outputs import DeclaredOutput, DtypeClass


def test_declared_output_round_trips_through_json() -> None:
    output = DeclaredOutput(width=64, rank=3, dtype_class=DtypeClass.REAL)

    assert DeclaredOutput.model_validate_json(output.model_dump_json()) == output


def test_the_wire_form_of_an_element_class_decodes_to_its_member() -> None:
    output = DeclaredOutput.model_validate(
        {"width": 1, "rank": 1, "dtype_class": "boolean"}
    )

    assert output.dtype_class is DtypeClass.BOOLEAN


def test_equal_declarations_collapse_in_a_set() -> None:
    output = DeclaredOutput(width=8, rank=2, dtype_class=DtypeClass.INTEGER)
    same = DeclaredOutput(width=8, rank=2, dtype_class=DtypeClass.INTEGER)
    wider = DeclaredOutput(width=16, rank=2, dtype_class=DtypeClass.INTEGER)

    assert {output, same, wider} == {output, wider}


def test_a_declared_output_changes_only_by_making_another() -> None:
    output = DeclaredOutput(width=8, rank=2, dtype_class=DtypeClass.REAL)
    wider = output.with_(width=16)

    assert wider.width == 16
    assert output.width == 8


@pytest.mark.parametrize(("width", "rank"), [(0, 2), (-1, 2), (4, 0), (4, -1)])
def test_an_extent_that_is_not_positive_is_refused(width: int, rank: int) -> None:
    with pytest.raises(dx.ValidationError):
        DeclaredOutput(width=width, rank=rank, dtype_class=DtypeClass.REAL)


def test_an_unknown_key_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        DeclaredOutput.model_validate(
            {"width": 4, "rank": 2, "dtype_class": "real", "depth": 1}
        )

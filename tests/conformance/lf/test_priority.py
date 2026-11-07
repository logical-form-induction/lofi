"""The priority properties by name, and the property inventory."""

from pathlib import Path
from typing import Final

import pytest
from tests.conformance.lf._fixtures import (
    PendingLogicalFormFixture,
    logical_form_fixture,
)
from tests.conformance.lf.test_c_print import print_parse_round_trips
from tests.conformance.lf.test_d_typing import (
    inference_is_sound,
    parser_and_typer_agree,
)
from tests.conformance.lf.test_e_reduction import (
    reduction_is_pure,
    substitution_avoids_capture,
)
from tests.conformance.lf.test_f_alpha import canonical_forms_decide_alpha

__all__ = ["logical_form_fixture"]

pytestmark = pytest.mark.conformance

_PROPERTY_COUNTS: Final = (
    ("A", 6),
    ("B", 8),
    ("C", 7),
    ("D", 15),
    ("E", 10),
    ("F", 6),
    ("N", 6),
)
_PROPERTY_IDS: Final = frozenset(
    f"{group}{number}"
    for group, count in _PROPERTY_COUNTS
    for number in range(1, count + 1)
)
_LF_DIRECTORY: Final = Path(__file__).parent


def test_e2_substitution_avoids_capture(lf_fixture: PendingLogicalFormFixture) -> None:
    substitution_avoids_capture(lf_fixture)


def test_e1_reduction_is_pure(lf_fixture: PendingLogicalFormFixture) -> None:
    reduction_is_pure(lf_fixture)


def test_c1_print_parse_round_trips(lf_fixture: PendingLogicalFormFixture) -> None:
    print_parse_round_trips(lf_fixture)


def test_d1_inference_is_sound(lf_fixture: PendingLogicalFormFixture) -> None:
    inference_is_sound(lf_fixture)


def test_d10_parser_and_typer_agree(lf_fixture: PendingLogicalFormFixture) -> None:
    parser_and_typer_agree(lf_fixture)


def test_f4_canonical_forms_decide_alpha(lf_fixture: PendingLogicalFormFixture) -> None:
    canonical_forms_decide_alpha(lf_fixture)


def test_every_property_is_collected_and_none_is_expected_to_fail(
    pytester: pytest.Pytester,
) -> None:
    items, _ = pytester.inline_genitems(str(_LF_DIRECTORY))
    collected: dict[str, pytest.Item] = {}

    for item in items:
        if not isinstance(item, pytest.Function):
            continue

        if (
            not item.originalname.endswith("_property")
            or item.name == item.originalname
        ):
            continue

        identifier = item.name.removeprefix(item.originalname).strip("[]")
        collected[identifier] = item

    assert frozenset(collected) == _PROPERTY_IDS

    for identifier, item in collected.items():
        markers = {marker.name for marker in item.iter_markers()}

        assert "xfail" not in markers, identifier
        assert "skip" not in markers, identifier
        assert "skipif" not in markers, identifier

from typing import TYPE_CHECKING, cast

import pytest

from lofi.probe._provenance import Provenance, provenance_from_json

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_a_provenance_keeps_what_produced_a_result_over_what_and_under_what() -> None:
    made = Provenance("eap-ig", "ccgbank-dev", 11)

    assert made.method == "eap-ig"
    assert made.dataset == "ccgbank-dev"
    assert made.seed == 11


def test_a_method_that_drew_nothing_carries_no_seed() -> None:
    assert Provenance("exact-ablation", "ccgbank-dev", None).seed is None


def test_a_method_with_no_name_is_refused() -> None:
    with pytest.raises(ValueError, match="method is empty"):
        Provenance("", "ccgbank-dev", 0)


def test_a_result_over_data_nobody_named_is_refused() -> None:
    with pytest.raises(ValueError, match="dataset is empty"):
        Provenance("eap-ig", "", 0)


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="seed is negative"):
        Provenance("eap-ig", "ccgbank-dev", -1)


def test_a_seed_that_is_a_float_is_refused() -> None:
    with pytest.raises(TypeError, match="seed is not an integer"):
        Provenance("eap-ig", "ccgbank-dev", cast("int", 1.0))


def test_two_records_of_the_same_three_facts_are_one_provenance() -> None:
    made = Provenance("eap-ig", "ccgbank-dev", 11)

    assert made == Provenance("eap-ig", "ccgbank-dev", 11)
    assert hash(made) == hash(Provenance("eap-ig", "ccgbank-dev", 11))
    assert len({made, Provenance("eap-ig", "ccgbank-dev", 11)}) == 1
    assert made != Provenance("eap-ig", "ccgbank-dev", 12)
    assert made != Provenance("acdc", "ccgbank-dev", 11)
    assert made != Provenance("eap-ig", "ccgbank-test", 11)
    assert made != ("eap-ig", "ccgbank-dev", 11)


def test_a_provenance_cannot_be_rebound() -> None:
    made = Provenance("eap-ig", "ccgbank-dev", 11)

    for held in ("method", "dataset", "seed"):
        with pytest.raises(AttributeError):
            setattr(made, held, None)

    assert not hasattr(made, "__dict__")
    assert made.seed == 11


def test_a_provenance_survives_the_round_trip_through_json() -> None:
    for seed in (None, 0, 4096):
        made = Provenance("bootstrap-maxima", "ccgbank-dev", seed)

        assert provenance_from_json(made.to_json()) == made


def test_a_serialized_provenance_missing_its_seed_is_refused() -> None:
    fields: dict[str, JsonValue] = {"method": "eap-ig", "dataset": "ccgbank-dev"}

    with pytest.raises(ValueError, match="no 'seed' field"):
        provenance_from_json(fields)


def test_a_serialized_provenance_that_is_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="not a JSON object"):
        provenance_from_json("eap-ig over ccgbank-dev")


def test_the_repr_of_a_provenance_shows_its_three_fields() -> None:
    shown = repr(Provenance("eap-ig", "ccgbank-dev", 11))

    assert "'eap-ig'" in shown
    assert "'ccgbank-dev'" in shown
    assert "seed=11" in shown

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import (
    DERIVED_SEED,
    SeedDerivationRecord,
    UnresolvedSeedError,
    data_seed,
    derive_seed,
    ensure_data_seed,
    fixed_seed,
    point_root,
    resolved_seed,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def a_record() -> dict[str, JsonValue]:
    return {
        "experiment_root": 7,
        "point_id": "0123456789abcdef",
        "seed": 2,
        "root": point_root(7, "0123456789abcdef"),
    }


def test_a_point_root_is_the_named_point_stream_of_the_experiment_root() -> None:
    assert point_root(7, "ab") == derive_seed(7, "point:ab")
    assert point_root(7, "ab") != point_root(8, "ab")
    assert point_root(7, "ab") != point_root(7, "ba")


@pytest.mark.parametrize(
    ("experiment_root", "point_id", "match"),
    [
        (-1, "ab", "negative"),
        (7, "", "empty"),
    ],
)
def test_a_point_root_refuses_an_invalid_derivation_input(
    experiment_root: int, point_id: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        point_root(experiment_root, point_id)


def test_a_record_whose_root_its_point_derives_validates() -> None:
    record = SeedDerivationRecord.model_validate(a_record())

    assert record.root == derive_seed(7, "point:0123456789abcdef")
    assert SeedDerivationRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("root", 1, "derives as"),
        ("experiment_root", 8, "derives as"),
        ("point_id", "fedcba9876543210", "derives as"),
        ("derivation", "sampled", "derivation"),
        ("seed", -1, "negative"),
        ("experiment_root", -1, "negative"),
        ("point_id", "", "empty"),
    ],
)
def test_a_record_that_does_not_derive_its_root_is_refused(
    field: str, value: JsonValue, match: str
) -> None:
    record = a_record()
    record[field] = value

    with pytest.raises(dx.ValidationError, match=match):
        SeedDerivationRecord.model_validate(record)


def test_a_record_with_an_unknown_field_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        SeedDerivationRecord.model_validate({**a_record(), "samplers": []})


def test_a_data_seed_is_a_nonnegative_integer_or_derived() -> None:
    assert ensure_data_seed("a seed", 5) == 5
    assert ensure_data_seed("a seed", DERIVED_SEED) == "derived"

    with pytest.raises(ValueError, match="'derive'"):
        ensure_data_seed("a seed", "derive")

    with pytest.raises(ValueError, match="negative"):
        ensure_data_seed("a seed", -1)


def test_a_fixed_seed_is_drawn_under_and_a_derived_one_is_refused() -> None:
    assert fixed_seed("a seed", 5) == 5

    with pytest.raises(UnresolvedSeedError, match="a seed is derived"):
        fixed_seed("a seed", DERIVED_SEED)


def test_resolution_keeps_a_fixed_seed_and_derives_a_derived_one_for_its_path() -> None:
    asked: list[str] = []

    def derive(path: str) -> int:
        asked.append(path)

        return 41

    assert resolved_seed(5, derive, "data.datasets.train.dataset.seed") == 5
    assert asked == []
    assert resolved_seed(DERIVED_SEED, derive, "data.datasets.train.dataset.seed") == 41
    assert asked == ["data.datasets.train.dataset.seed"]


def test_a_data_seed_is_the_named_data_stream_of_the_run_seed() -> None:
    assert data_seed(9, "data.datasets.train.dataset.seed") == derive_seed(
        9, "data:data.datasets.train.dataset.seed"
    )
    assert data_seed(9, "data.datasets.train.dataset.seed") != data_seed(
        10, "data.datasets.train.dataset.seed"
    )
    assert data_seed(9, "data.datasets.train.dataset.seed") != data_seed(
        9, "data.datasets.train.dataset.grammar.lexicon.seed"
    )

    with pytest.raises(ValueError, match="data-seed path"):
        data_seed(9, "")

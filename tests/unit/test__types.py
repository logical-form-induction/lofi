from collections.abc import Mapping, MutableMapping
from typing import cast, get_args

import pytest

from lofi._types import FrozenJsonValue, ItemId, JsonValue, freeze_json_value


def test_json_value_closes_over_itself() -> None:
    assert dict[str, JsonValue] in get_args(JsonValue.__value__)
    assert list[JsonValue] in get_args(JsonValue.__value__)


def test_frozen_json_value_closes_over_read_only_containers() -> None:
    assert tuple[FrozenJsonValue, ...] in get_args(FrozenJsonValue.__value__)
    assert Mapping[str, FrozenJsonValue] in get_args(FrozenJsonValue.__value__)


def test_freezing_json_copies_every_mutable_container() -> None:
    source: JsonValue = {
        "layers": [2, 3],
        "options": {"activation": "gelu", "bias": True},
    }
    frozen = freeze_json_value(source)

    assert frozen == {
        "layers": (2, 3),
        "options": {"activation": "gelu", "bias": True},
    }

    assert isinstance(source, dict)
    source["layers"] = [7]
    source["options"] = {}

    assert frozen == {
        "layers": (2, 3),
        "options": {"activation": "gelu", "bias": True},
    }


def test_every_nested_json_container_is_read_only() -> None:
    frozen = freeze_json_value({"outer": [{"inner": 1}]})

    assert isinstance(frozen, Mapping)
    mutable = cast("MutableMapping[str, FrozenJsonValue]", frozen)

    with pytest.raises(TypeError):
        mutable["outer"] = ()

    outer = frozen["outer"]
    assert isinstance(outer, tuple)
    nested = outer[0]
    assert isinstance(nested, Mapping)
    mutable_nested = cast("MutableMapping[str, FrozenJsonValue]", nested)

    with pytest.raises(TypeError):
        mutable_nested["inner"] = 2


def test_predictions_and_gold_join_on_an_item_id() -> None:
    gold = {ItemId("judgment:1"): 3.0, ItemId("judgment:2"): 6.0}
    predicted = ItemId("judgment:2")

    assert ItemId.__supertype__ is str
    assert gold[predicted] == 6.0

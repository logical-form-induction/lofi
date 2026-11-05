"""Behavioral tests for the trained-point capabilities bead consumes."""

from typing import NamedTuple

import pytest
from bead.protocol import ResponseEncoding, ScaleType

from lofi import judgment
from lofi.judgment import Choice, Cumulative
from lofi.judgment.bead import InferenceText, PointInputs, response_family_for


def encoding(scale_type: ScaleType) -> ResponseEncoding:
    """Return one real bead response encoding of the requested kind."""
    labels = ("no", "yes") if scale_type is ScaleType.BINARY else ("no", "maybe", "yes")
    return ResponseEncoding(
        name="inference",
        n_levels=len(labels),
        scale_type=scale_type,
        labels=labels,
    )


@pytest.mark.parametrize("scale_type", [ScaleType.BINARY, ScaleType.ORDINAL])
def test_ordered_bead_encodings_select_the_cumulative_family(
    scale_type: ScaleType,
) -> None:
    assert isinstance(response_family_for(encoding(scale_type)), Cumulative)


@pytest.mark.parametrize("scale_type", [ScaleType.NOMINAL, ScaleType.FORCED_CHOICE])
def test_unordered_bead_encodings_select_the_choice_family(
    scale_type: ScaleType,
) -> None:
    assert isinstance(response_family_for(encoding(scale_type)), Choice)


def test_inference_text_preserves_identity_and_refuses_blank_elements() -> None:
    pair = InferenceText("item-17", "Every dog barked.", "A dog barked.")

    assert pair.item_id == "item-17"
    assert pair.premise == "Every dog barked."
    assert pair.hypothesis == "A dog barked."
    assert pair == InferenceText("item-17", "Every dog barked.", "A dog barked.")

    with pytest.raises(ValueError, match="inference hypothesis"):
        InferenceText("item-17", "Every dog barked.", "")


def test_the_package_exposes_no_superseded_bead_adapter_surface() -> None:
    superseded = {
        "BeadInferenceItem",
        "BeadItemTemplate",
        "BeadModelAdapter",
        "BeadModelEncoding",
        "BeadPredictionAdapter",
        "TrainedPointLoader",
        "TrainedPointPersistence",
    }

    assert superseded.isdisjoint(judgment.__all__)


class _Rendered(NamedTuple):
    id: str
    rendered_elements: dict[str, str]


def test_declared_inputs_read_each_role_from_its_element() -> None:
    inputs = PointInputs({"words": "first"}, roles=("words",))
    item = _Rendered("item-1", {"first": "A dog barked.", "second": "A dog."})

    assert inputs.texts(item) == {"words": "A dog barked."}
    with pytest.raises(ValueError, match="renders no 'third' element"):
        PointInputs({"words": "third"}, roles=("words",)).texts(item)


def test_declared_inputs_map_exactly_the_roles_the_point_publishes() -> None:
    with pytest.raises(ValueError, match="reads the input roles"):
        PointInputs({"other": "first"}, roles=("words",))
    with pytest.raises(ValueError, match="reads the input roles"):
        PointInputs({}, roles=("words",))


def test_an_inference_input_is_read_through_the_published_roles() -> None:
    roles = dict(zip(InferenceText.ROLES, ("first", "second"), strict=True))
    inputs = PointInputs(roles, roles=InferenceText.ROLES)
    item = _Rendered("item-1", {"first": "A dog barked.", "second": "A dog."})

    assert InferenceText.of(item, inputs) == InferenceText(
        "item-1", "A dog barked.", "A dog."
    )

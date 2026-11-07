import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.grammaticality import GrammaticalityLossSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec.components.model import ModelSection


def test_the_tag_reaches_the_grammaticality_variant_through_the_family_root() -> None:
    declared = LossSpec.model_validate(
        {"kind": "grammaticality", "readout": "sentences", "noise": 0.1}
    )

    assert type(declared) is GrammaticalityLossSpec
    assert declared == GrammaticalityLossSpec(readout="sentences", noise=0.1)
    assert LossSpec.__variants__["grammaticality"] is GrammaticalityLossSpec


def test_the_noise_defaults_to_zero_and_the_loss_declares_no_graph_edge() -> None:
    declared = GrammaticalityLossSpec(readout="sentences")

    assert declared.noise == 0.0
    assert declared.dependencies() == ()
    assert declared.declared_output is None


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "grammaticality"}, "readout"),
        ({"kind": "grammaticality", "readout": ""}, "published readout"),
        ({"kind": "grammaticality", "readout": True}, "readout"),
        (
            {"kind": "grammaticality", "readout": "sentences", "noise": -0.1},
            r"\[0, 0\.5\)",
        ),
        (
            {"kind": "grammaticality", "readout": "sentences", "noise": 0.5},
            r"\[0, 0\.5\)",
        ),
        (
            {
                "kind": "grammaticality",
                "readout": "sentences",
                "noise": math.inf,
            },
            "not finite",
        ),
        (
            {
                "kind": "grammaticality",
                "readout": "sentences",
                "noise": -math.inf,
            },
            "not finite",
        ),
        (
            {"kind": "grammaticality", "readout": "sentences", "noise": True},
            "noise",
        ),
        (
            {
                "kind": "grammaticality",
                "readout": "sentences",
                "noise": math.nan,
            },
            "not finite",
        ),
        (
            {
                "kind": "grammaticality",
                "readout": "sentences",
                "smoothing": 0.1,
            },
            "smoothing",
        ),
    ],
)
def test_a_missing_empty_nonfinite_out_of_range_or_extra_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


def test_an_unknown_tag_is_refused_with_the_tag_named() -> None:
    with pytest.raises(dx.ValidationError, match="'acceptability'"):
        LossSpec.model_validate({"kind": "acceptability", "readout": "sentences"})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "grammaticality", "readout": "sentences", "noise": 0.1, '
        '"declared_output": null}'
    )
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared


def test_grammaticality_refuses_a_mass_no_declared_readout_publishes(
    packaged_model: ModelSection,
) -> None:
    GrammaticalityLossSpec(readout="sentences").refuse_model(
        packaged_model, "loss.terms.grammaticality"
    )

    with pytest.raises(
        ValueError,
        match=r"loss\.terms\.grammaticality\.readout: the objective reads the root "
        r"mass 'types', which no declared readout publishes; they publish "
        r"\['grammar-goal', 'sentences'\]",
    ):
        GrammaticalityLossSpec(readout="types").refuse_model(
            packaged_model, "loss.terms.grammaticality"
        )

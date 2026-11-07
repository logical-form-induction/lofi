from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
from tests.unit.spec.components.conftest import (
    ApplicationDecoder,
    ChartParser,
    SizedEncoder,
    declaration,
)

from lofi.spec._ports import Dependency
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.model import (
    ActionsSpec,
    DataModelLanguageSpec,
    InlineModelLanguageSpec,
    ModelRole,
    ModelSection,
    ParameterCapacitySpec,
    TermDecoderSpec,
)
from lofi.spec.components.run import RunSpec
from lofi.spec.controllers.mlp import MlpControllerSpec
from lofi.spec.groundings import PairwiseGroundingSpec
from lofi.spec.provers import ProofBudgetSpec
from lofi.spec.readouts import ReadoutSource, TermsReadoutSpec
from lofi.spec.scorers import ProverScorerSpec
from lofi.spec.spaces import CombinatorFamilySpec, Merge, SpaceSpec
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec

if TYPE_CHECKING:
    from collections.abc import Callable


ENCODER: Final = SizedEncoder(width=4)
DECODER: Final = ApplicationDecoder(rank=1)
LANGUAGE: Final = DataModelLanguageSpec()
READOUT: Final = TermsReadoutSpec(space="categories", reads=ReadoutSource.INSIDE)
SPACE: Final = SpaceSpec(
    sort="Cat",
    encoder="terms",
    identity="left",
    combinators=CombinatorFamilySpec(
        merge=Merge.CONCATENATE, decoders={"apply": "left"}
    ),
)


def _grounding() -> PairwiseGroundingSpec:
    return PairwiseGroundingSpec(
        premise="premise-forms",
        hypothesis="hypothesis-forms",
        scorer=ProverScorerSpec(
            budget=ProofBudgetSpec(calls_per_step=4),
        ),
    )


def test_named_model_roles_are_canonical() -> None:
    model = ModelSection(
        language=LANGUAGE,
        parser=ChartParser(),
        encoders={"zeta": ENCODER, "alpha": SizedEncoder(width=8)},
        decoders={"right": DECODER, "left": ApplicationDecoder(rank=2)},
        spaces={
            "zulu": SpaceSpec(sort="Tm", encoder="zeta", identity="right"),
            "bravo": SpaceSpec(sort="Cat", encoder="alpha", identity="left"),
        },
        readouts={
            "zeta-readout": TermsReadoutSpec(space="zulu", reads=ReadoutSource.INSIDE),
            "alpha-readout": TermsReadoutSpec(
                space="bravo", reads=ReadoutSource.INSIDE
            ),
        },
    )

    assert tuple(model.encoders) == ("alpha", "zeta")
    assert tuple(model.decoders) == ("left", "right")
    assert tuple(model.spaces) == ("bravo", "zulu")
    assert tuple(model.readouts) == ("alpha-readout", "zeta-readout")


def test_a_model_lists_only_buildable_components_under_their_roles() -> None:
    model = ModelSection(
        language=LANGUAGE,
        parser=ChartParser(),
        encoders={"terms": ENCODER},
        decoders={"left": DECODER},
        spaces={"categories": SPACE},
        readouts={"types": READOUT},
    )

    assert model.components() == {
        "parser": ChartParser(),
        "terms": ENCODER,
        "left": DECODER,
        "types": READOUT,
    }


def test_a_model_publishes_a_grounding_nested_component_under_its_state_role() -> None:
    grounding = _grounding()

    assert ModelSection(language=LANGUAGE, grounding=grounding).components() == {
        "grounding": grounding,
        "scorer": grounding.scorer,
    }


def test_a_grounding_nested_component_cannot_reuse_another_model_role() -> None:
    with pytest.raises(
        dx.ValidationError,
        match=r"encoders reuse other component roles: \['scorer'\]",
    ):
        ModelSection(
            language=LANGUAGE,
            grounding=_grounding(),
            encoders={"scorer": ENCODER},
        )


@pytest.mark.parametrize(
    ("declare", "reported", "role"),
    [
        (
            lambda: ModelSection(language=LANGUAGE, encoders={"parser": ENCODER}),
            "encoders",
            "parser",
        ),
        (
            lambda: ModelSection(
                language=LANGUAGE,
                encoders={"shared": ENCODER},
                decoders={"shared": DECODER},
            ),
            "decoders",
            "shared",
        ),
        (
            lambda: ModelSection(
                language=LANGUAGE,
                decoders={"left": DECODER},
                span_decoders={"left": RnnSpanDecoderSpec()},
            ),
            "span decoders",
            "left",
        ),
        (
            lambda: ModelSection(
                language=LANGUAGE,
                decoders={"left": DECODER},
                readouts={"left": READOUT},
            ),
            "readouts",
            "left",
        ),
    ],
)
def test_a_component_role_declared_twice_is_refused(
    declare: Callable[[], ModelSection], reported: str, role: str
) -> None:
    with pytest.raises(
        dx.ValidationError,
        match=rf"{reported} reuse other component roles: \['{role}'\]",
    ):
        declare()


@pytest.mark.parametrize(
    ("space", "reason"),
    [
        (
            SpaceSpec(sort="Cat", encoder="missing", identity="left"),
            "names encoder 'missing'",
        ),
        (
            SpaceSpec(sort="Cat", encoder="terms", identity="missing"),
            r"undeclared decoders: \['missing'\]",
        ),
    ],
)
def test_a_space_cannot_name_an_undeclared_component(
    space: SpaceSpec, reason: str
) -> None:
    with pytest.raises(dx.ValidationError, match=reason):
        ModelSection(
            language=LANGUAGE,
            encoders={"terms": ENCODER},
            decoders={"left": DECODER},
            spaces={"categories": space},
        )


def test_a_readout_cannot_name_an_undeclared_space() -> None:
    with pytest.raises(
        dx.ValidationError,
        match="readout 'types' names space 'missing'",
    ):
        ModelSection(
            language=LANGUAGE,
            readouts={
                "types": TermsReadoutSpec(space="missing", reads=ReadoutSource.INSIDE)
            },
        )


def test_two_spaces_cannot_redeclare_one_sort() -> None:
    with pytest.raises(dx.ValidationError, match="both represent sort 'Cat'"):
        ModelSection(
            language=LANGUAGE,
            encoders={"terms": ENCODER},
            decoders={"left": DECODER},
            spaces={
                "categories": SPACE,
                "again": SpaceSpec(sort="Cat", encoder="terms", identity="left"),
            },
        )


def test_a_decoder_is_reached_by_its_named_decoder_role() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "model": {
                "language": {"from": "data"},
                "parser": {"kind": "chart-parser"},
                "encoders": {"terms": {"kind": "sized", "width": 4}},
                "decoders": {
                    "forward": {"kind": "application-decoder", "rank": 1},
                    "backward": {"kind": "application-decoder", "rank": 2},
                },
                "spaces": {
                    "categories": {
                        "sort": "Cat",
                        "encoder": "terms",
                        "identity": "forward",
                    }
                },
            }
        }
    )

    assert run.model.decoders["backward"] == ApplicationDecoder(rank=2)


def test_a_run_that_declares_its_decoders_in_another_order_is_the_same_run() -> None:
    roles: dict[str, TermDecoderSpec] = {
        "forward": ApplicationDecoder(rank=1),
        "backward": ApplicationDecoder(rank=2),
    }
    reversed_roles = dict(reversed(list(roles.items())))

    assert ModelSection(language=LANGUAGE, decoders=roles) == ModelSection(
        language=LANGUAGE, decoders=reversed_roles
    )


def test_a_model_declares_an_identity_and_no_null_action_by_default() -> None:
    actions = ModelSection(language=LANGUAGE).actions

    assert (actions.identity, actions.null) == ("identity", None)


def test_a_declared_null_action_round_trips_through_the_model_section() -> None:
    model = ModelSection.model_validate(
        {"language": {"from": "data"}, "actions": {"null": "none-applies"}}
    )

    assert model.actions == ActionsSpec(null="none-applies")
    assert ModelSection.model_validate_json(model.model_dump_json()) == model


@pytest.mark.parametrize(
    ("declared", "reason"),
    [
        ({"identity": ""}, "the identity action needs a name"),
        ({"null": ""}, "a null action needs a name"),
        ({"null": "identity"}, "both named 'identity'"),
        ({"identity": "keep", "null": "keep"}, "both named 'keep'"),
    ],
)
def test_an_unnamed_or_doubly_named_action_is_refused(
    declared: dict[str, str], reason: str
) -> None:
    with pytest.raises(dx.ValidationError, match=reason):
        ActionsSpec.model_validate(declared)


def test_a_model_language_is_required_and_has_no_fallback() -> None:
    with pytest.raises(dx.ValidationError, match="language"):
        ModelSection.model_validate({})

    model = ModelSection.model_validate({"language": {"from": "data"}})

    assert isinstance(model.language, DataModelLanguageSpec)
    assert model.model_dump(by_alias=True)["language"] == {"from": "data"}


def test_an_inline_model_language_declares_its_language_boundary() -> None:
    assert tuple(InlineModelLanguageSpec.__field_specs__) == (
        "from",
        "language",
        "grammar",
        "interpretation",
    )


def test_a_model_declares_an_exact_total_parameter_target() -> None:
    model = ModelSection.model_validate(
        {
            "language": {"from": "data"},
            "capacity": {"kind": "parameter-count", "target": 1200},
        }
    )

    assert model.capacity == ParameterCapacitySpec(target=1200)
    assert model.model_dump()["capacity"] == {
        "kind": "parameter-count",
        "target": 1200,
        "declared_output": None,
    }


@pytest.mark.parametrize("target", [0, -1])
def test_a_parameter_capacity_target_is_positive(target: int) -> None:
    with pytest.raises(dx.ValidationError, match="parameter-capacity target"):
        ParameterCapacitySpec(target=target)


def test_the_capacity_role_is_reserved_for_parameter_padding() -> None:
    with pytest.raises(
        dx.ValidationError, match="encoders reuse other component roles"
    ):
        ModelSection(
            language=LANGUAGE,
            capacity=ParameterCapacitySpec(target=10),
            encoders={"capacity": ENCODER},
        )


def test_a_controller_reads_the_cells_of_the_encoder_it_is_given() -> None:
    controller = MlpControllerSpec(hidden=())

    assert controller.encoder_dependency("terms") == Dependency("terms", "cells")


def test_the_single_components_are_keyed_by_their_model_roles() -> None:
    model = ModelSection.model_validate(
        {"language": {"from": "data"}, "controller": {"kind": "mlp", "hidden": []}}
    )

    assert set(model.components()) == {ModelRole.CONTROLLER}


def test_a_grounded_model_fits_a_sentence_requirement_on_inference_pairs() -> None:
    sentences = frozenset({TrainingBatchKind.SENTENCES})
    grounded = ModelSection(language=LANGUAGE, grounding=_grounding())

    assert ModelSection(language=LANGUAGE).fitted_batches(sentences) == sentences
    assert grounded.fitted_batches(sentences) == frozenset(
        {TrainingBatchKind.INFERENCE_PAIRS}
    )
    assert grounded.fitted_batches(frozenset({TrainingBatchKind.JUDGMENTS})) == (
        frozenset({TrainingBatchKind.JUDGMENTS})
    )

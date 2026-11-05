from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi import spec
from lofi.spec import (
    ComponentSpec,
    ConstraintSourceSpec,
    ControllerSpec,
    DatasetSpec,
    EvaluatorSpec,
    FeaturizerSpec,
    GroundingSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    ProbeSpec,
    ReadoutSpec,
    SchedulerSpec,
    SinkSpec,
    SubwordPoolerSpec,
    TermDecoderSpec,
    TermEncoderSpec,
    TokenEncoderSpec,
    TrainerSpec,
    TrainingAllocationSpec,
    TypeSamplerSpec,
    _archive,
    _coerce,
    _constructor,
    _derivation,
    _errors,
    _fingerprint,
    _groups,
    _interpolation,
    _loader,
    _plugins,
    _ports,
    _rule,
    _wiring,
    approximation,
    batch_samplers,
    constraint_policy,
    decoding,
    forward,
    held_out,
    normalization,
    outputs,
    policy,
    reducers,
    semantics,
    spaces,
)
from lofi.spec.components import (
    base,
    baseline,
    counterfactuals,
    data,
    loss,
    model,
    run,
    trainer,
)
from lofi.spec.evaluators import (
    coherence_square,
    grammaticality,
    held_out_likelihood,
    inference,
    structural_diagnostics,
)
from lofi.spec.provers import ProverSpec

if TYPE_CHECKING:
    from types import ModuleType

FAMILIES: tuple[type[dx.TaggedUnion], ...] = (
    TermEncoderSpec,
    TokenEncoderSpec,
    SubwordPoolerSpec,
    TermDecoderSpec,
    ControllerSpec,
    ParserSpec,
    LossSpec,
    DatasetSpec,
    FeaturizerSpec,
    TrainerSpec,
    OptimizerSpec,
    SchedulerSpec,
    SinkSpec,
    ProbeSpec,
    EvaluatorSpec,
    GroundingSpec,
    ProverSpec,
    ConstraintSourceSpec,
    TypeSamplerSpec,
    ReadoutSpec,
)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
def test_a_family_refuses_a_kind_nothing_claims(
    family: type[dx.TaggedUnion],
) -> None:
    with pytest.raises(dx.ValidationError):
        family.model_validate({"kind": "unclaimed"})


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.__name__)
def test_a_family_refuses_a_payload_that_names_no_kind(
    family: type[dx.TaggedUnion],
) -> None:
    with pytest.raises(dx.ValidationError):
        family.model_validate({"width": 12})


def test_a_component_declares_nothing_about_its_output_until_a_variant_does() -> None:
    assert ComponentSpec().declared_output is None


def test_a_component_has_no_dependencies_until_a_variant_declares_them() -> None:
    assert ComponentSpec().dependencies() == ()


def test_clausal_adjunct_spec_is_public() -> None:
    assert spec.ClausalAdjunctSpec is _rule.ClausalAdjunctSpec


def test_training_allocation_capability_is_public() -> None:
    assert TrainingAllocationSpec is data.TrainingAllocationSpec


def test_the_package_publishes_what_its_modules_publish() -> None:
    source: dict[str, ModuleType] = {}

    for module in (
        _archive,
        _coerce,
        _constructor,
        _derivation,
        constraint_policy,
        _errors,
        _fingerprint,
        _groups,
        _interpolation,
        _loader,
        _ports,
        _rule,
        _wiring,
        approximation,
        baseline,
        base,
        model,
        loss,
        data,
        trainer,
        run,
        counterfactuals,
        batch_samplers,
        decoding,
        coherence_square,
        grammaticality,
        held_out,
        held_out_likelihood,
        inference,
        normalization,
        forward,
        outputs,
        policy,
        reducers,
        semantics,
        spaces,
        structural_diagnostics,
    ):
        source |= dict.fromkeys(module.__all__, module)

    for name in ("FAMILY_GROUP", "component_groups", "load_plugins"):
        source[name] = _plugins

    assert sorted(spec.__all__) == sorted(source)

    for name, module in source.items():
        assert vars(spec)[name] is vars(module)[name]

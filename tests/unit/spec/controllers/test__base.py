"""Tests for shared controller declarations."""

import didactic.api as dx
import pytest
from tests.unit.spec.components.conftest import declaration

from lofi.spec.components.model import DataModelLanguageSpec, ModelSection
from lofi.spec.components.run import RunSpec
from lofi.spec.controllers import (
    EntmaxControllerNormalizerSpec,
    MlpControllerSpec,
    SoftmaxControllerNormalizerSpec,
)
from lofi.spec.parsers.chart import InsideOutsideParserSpec
from lofi.spec.parsers.split import AttendSplitWeigherSpec


def test_entmax_declares_its_alpha_and_threshold_search() -> None:
    assert EntmaxControllerNormalizerSpec(alpha=1.25).settings() == (1.25, 50)
    assert EntmaxControllerNormalizerSpec(alpha=1.25, bisections=8).settings() == (
        1.25,
        8,
    )
    assert SoftmaxControllerNormalizerSpec().settings() == (1.0, 0)


def test_entmax_refuses_a_threshold_search_without_a_bisection() -> None:
    with pytest.raises(dx.ValidationError, match="bisection"):
        EntmaxControllerNormalizerSpec(bisections=0)


def test_entmax_refuses_an_alpha_below_one() -> None:
    with pytest.raises(dx.ValidationError, match="at least one"):
        EntmaxControllerNormalizerSpec(alpha=0.9)


def test_every_controller_publishes_one_policy_to_its_parser() -> None:
    assert MlpControllerSpec().publications() == frozenset({"policy"})


def test_soft_raising_is_declared_under_the_run_document_raise_key() -> None:
    declared = MlpControllerSpec.model_validate({"kind": "mlp", "raise": "soft"})

    assert declared.raising() == "soft"
    assert declared.model_dump()["raise"] == "soft"


def test_a_controller_does_not_raise_unless_declared() -> None:
    assert MlpControllerSpec().raising() == "none"


def test_span_input_is_refused_with_marginal_split_weights() -> None:
    document = declaration()
    document["model"] = {
        "language": {"from": "data"},
        "parser": {"kind": "inside-outside", "width": 4},
        "controller": {"kind": "mlp", "input": "span"},
    }

    with pytest.raises(dx.ValidationError, match="input 'span' is incompatible"):
        RunSpec.model_validate(document)


def test_span_input_composes_with_attended_split_weights() -> None:
    model = ModelSection(
        language=DataModelLanguageSpec(),
        parser=InsideOutsideParserSpec(
            width=4,
            attend=AttendSplitWeigherSpec(),
        ),
        controller=MlpControllerSpec(input="span"),
    )

    assert model.controller == MlpControllerSpec(input="span")


def test_soft_raising_refuses_span_input_at_the_run_boundary() -> None:
    document = declaration()
    document["model"] = {
        "language": {"from": "data"},
        "parser": {
            "kind": "inside-outside",
            "width": 4,
            "attend": {"kind": "attend"},
        },
        "controller": {"kind": "mlp", "input": "span", "raise": "soft"},
    }

    with pytest.raises(
        dx.ValidationError, match="soft raising requires input 'inside'"
    ):
        RunSpec.model_validate(document)


def test_a_controller_selects_installed_pair_features_by_name() -> None:
    declared = MlpControllerSpec.model_validate(
        {"kind": "mlp", "features": ["difference", "product"]}
    )

    assert declared.features == ("difference", "product")

    with pytest.raises(dx.ValidationError, match="installed features are"):
        MlpControllerSpec.model_validate({"kind": "mlp", "features": ["sum"]})

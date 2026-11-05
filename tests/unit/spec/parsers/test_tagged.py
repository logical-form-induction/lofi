"""Tests for the tagged parser declaration."""

from typing import cast

import didactic.api as dx
import pytest

from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec._ports import Dependency
from lofi.spec.components.model import ParserSpec
from lofi.spec.parsers.reduction import (
    LogSumExpReductionSettings,
    MaximumReductionSettings,
)
from lofi.spec.parsers.split import (
    AttendSplitWeigherSettings,
    MarginalSplitWeigherSettings,
)
from lofi.spec.parsers.tagged import TaggedParserSpec
from lofi.spec.span_functions import MlpSpanFunctionSettings, SpanFunctionDeclaration


def test_the_tag_reaches_the_tagged_variant_through_the_family_root() -> None:
    declared = cast(
        "TaggedParserSpec",
        ParserSpec.model_validate(
            {"kind": "tagged", "width": 4, "unary_depth": 1, "k_best": 2}
        ),
    )

    assert type(declared) is TaggedParserSpec
    assert declared.unary_depth == 1
    assert declared.k_best == 2
    assert declared.settings().reduction == LogSumExpReductionSettings()
    assert declared.settings().attend == (
        "marginal",
        MarginalSplitWeigherSettings(),
    )
    assert ParserSpec.__variants__["tagged"] is TaggedParserSpec


def test_a_tagged_chart_resolves_maximum_reduction() -> None:
    declared = cast(
        "TaggedParserSpec",
        ParserSpec.model_validate(
            {
                "kind": "tagged",
                "width": 4,
                "reduction": {"kind": "maximum"},
                "unary_depth": 1,
            }
        ),
    )

    assert declared.settings().reduction == MaximumReductionSettings()


def test_a_tagged_chart_resolves_declared_child_pair_attention() -> None:
    declared = cast(
        "TaggedParserSpec",
        ParserSpec.model_validate(
            {
                "kind": "tagged",
                "width": 4,
                "attend": {"kind": "attend"},
                "unary_depth": 1,
            }
        ),
    )

    assert declared.settings().attend == (
        "attend",
        AttendSplitWeigherSettings(
            SpanFunctionDeclaration("mlp", MlpSpanFunctionSettings(1, "tanh"))
        ),
    )


def test_a_tagged_chart_can_share_its_two_split_functions() -> None:
    declared = TaggedParserSpec(width=4, unary_depth=1, share_splits=True)

    assert declared.settings().share_splits

    with pytest.raises(dx.ValidationError, match="shared split slots"):
        TaggedParserSpec.model_validate(
            {
                "width": 4,
                "unary_depth": 1,
                "share_splits": True,
                "split_left": {"kind": "mlp", "layers": 1, "activation": "tanh"},
                "split_right": {"kind": "bilinear"},
            }
        )


@pytest.mark.parametrize(("field", "value"), [("unary_depth", -1), ("k_best", 0)])
def test_invalid_chart_bounds_are_refused(field: str, value: int) -> None:
    with pytest.raises(dx.ValidationError):
        ParserSpec.model_validate(
            {"kind": "tagged", "width": 4, "unary_depth": 1, field: value}
        )


def test_a_tagged_chart_requires_a_declared_unary_depth() -> None:
    with pytest.raises(dx.ValidationError, match="unary_depth"):
        ParserSpec.model_validate({"kind": "tagged", "width": 4})


def test_a_tagged_chart_publishes_its_unary_closure_depth() -> None:
    declared = TaggedParserSpec(width=4, unary_depth=3)

    assert declared.unary_closure_depth() == 3


def test_a_tagged_chart_publishes_category_transition_reachability() -> None:
    declared = TaggedParserSpec(width=4, unary_depth=1)

    assert declared.category_transition_reachability()


def test_the_tagged_chart_declares_its_producers_and_shared_controller() -> None:
    declared = TaggedParserSpec(width=4, unary_depth=1)

    assert declared.dependencies() == (
        Dependency("subword_pooler", "leaves"),
        Dependency("controller", "rules"),
    )
    assert declared.sharing() == {
        "rules": SubmoduleDeclaration(
            "policy", ModuleSharing(in_parameters=True, in_state_dict=True)
        )
    }
    assert declared.publications() == {"combine"}
    assert {"alpha", "mu", "spans", "outside_cells"} <= declared.chart_outputs()


def test_the_tagged_variant_round_trips_through_json() -> None:
    document = (
        '{"kind": "tagged", "width": 4, '
        '"composition": {"kind": "mean-field", '
        '"leaf": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"cell": {"kind": "mlp", "layers": 1, "activation": "tanh"}}, '
        '"reduction": {"kind": "log-sum-exp"}, '
        '"attend": {"kind": "marginal"}, "unary_depth": 1, "k_best": 2, '
        '"split_left": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"split_right": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"share_splits": false, '
        '"outside_weights": "contextual", "root_outside": "learned", '
        '"root_init_gain": null, "declared_output": null}'
    )
    declared = ParserSpec.model_validate_json(document)

    assert declared.model_dump_json() == document

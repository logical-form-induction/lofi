from typing import cast

import didactic.api as dx
import pytest

from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.formalism import BoundedConsumer
from lofi.spec._ports import Dependency
from lofi.spec.chart_compositions import (
    GatedResidualCompositionSpec,
    MeanFieldCompositionSpec,
)
from lofi.spec.components.model import ParserSpec
from lofi.spec.parsers.chart import InsideOutsideParserSpec
from lofi.spec.parsers.reduction import (
    LogSumExpReductionSettings,
    LogSumExpReductionSpec,
    MaximumReductionSettings,
    MaximumReductionSpec,
)
from lofi.spec.parsers.split import (
    AttendSplitWeigherSettings,
    AttendSplitWeigherSpec,
    MarginalSplitWeigherSettings,
)
from lofi.spec.span_functions import (
    MlpSpanFunctionSettings,
    MlpSpanFunctionSpec,
    SpanFunctionDeclaration,
)


def test_the_tag_reaches_the_chart_variant_through_the_family_root() -> None:
    declared = ParserSpec.model_validate({"kind": "inside-outside", "width": 4})

    assert type(declared) is InsideOutsideParserSpec
    assert declared == InsideOutsideParserSpec(width=4)
    assert ParserSpec.__variants__["inside-outside"] is InsideOutsideParserSpec


def test_log_sum_exp_is_the_default_tensor_reduction() -> None:
    declared = InsideOutsideParserSpec(width=4)

    assert declared.reduction == LogSumExpReductionSpec()
    assert declared.depth == 1
    assert declared.layer_sharing == "independent"
    assert declared.settings().reduction == LogSumExpReductionSettings()
    assert declared.settings().attend == (
        "marginal",
        MarginalSplitWeigherSettings(),
    )
    assert declared.chart_outputs() == {
        "inside",
        "log_z",
        "identity",
        "cells",
        "actions",
        "outside",
        "alpha",
        "mu",
        "spans",
        "outside_cells",
    }


def test_the_chart_resolves_declared_child_pair_attention() -> None:
    declared = cast(
        "InsideOutsideParserSpec",
        ParserSpec.model_validate(
            {"kind": "inside-outside", "width": 4, "attend": {"kind": "attend"}}
        ),
    )

    assert declared.attend == AttendSplitWeigherSpec(
        score=MlpSpanFunctionSpec(layers=1, activation="tanh")
    )
    assert declared.settings().attend == (
        "attend",
        AttendSplitWeigherSettings(
            SpanFunctionDeclaration("mlp", MlpSpanFunctionSettings(1, "tanh"))
        ),
    )


def test_the_chart_resolves_a_declared_maximum_reduction() -> None:
    declared = cast(
        "InsideOutsideParserSpec",
        ParserSpec.model_validate(
            {
                "kind": "inside-outside",
                "width": 4,
                "reduction": {"kind": "maximum"},
            }
        ),
    )

    assert declared == InsideOutsideParserSpec(
        width=4, reduction=MaximumReductionSpec()
    )
    assert declared.settings().reduction == MaximumReductionSettings()
    assert declared.chart_outputs() == {
        "inside",
        "log_z",
        "identity",
        "cells",
        "actions",
    }


def test_split_slots_may_share_exactly_one_declared_function() -> None:
    shared = InsideOutsideParserSpec(width=4, share_splits=True)

    assert shared.settings().share_splits

    with pytest.raises(dx.ValidationError, match="shared split slots"):
        InsideOutsideParserSpec.model_validate(
            {
                "width": 4,
                "share_splits": True,
                "split_left": {"kind": "mlp", "layers": 1, "activation": "tanh"},
                "split_right": {"kind": "bilinear"},
            }
        )


def test_the_chart_reads_pooled_leaves_and_the_weights_a_controller_gives() -> None:
    assert InsideOutsideParserSpec(width=4).dependencies() == (
        Dependency("subword_pooler", "leaves"),
        Dependency("controller", "rules"),
    )


def test_the_node_width_is_declared_once_on_the_chart() -> None:
    with pytest.raises(dx.ValidationError, match="width"):
        ParserSpec.model_validate({"kind": "inside-outside"})

    with pytest.raises(dx.ValidationError, match="node width"):
        InsideOutsideParserSpec(width=0)


@pytest.mark.parametrize("depth", [0, -1])
def test_the_chart_depth_is_positive(depth: int) -> None:
    with pytest.raises(dx.ValidationError, match="depth"):
        InsideOutsideParserSpec(width=4, depth=depth)


def test_a_stack_requires_probabilistic_inside_outside_readouts() -> None:
    with pytest.raises(dx.ValidationError, match=r"stacked.*log-sum-exp"):
        InsideOutsideParserSpec(
            width=4,
            depth=2,
            reduction=MaximumReductionSpec(),
        )


def test_the_chart_selects_a_registered_composition_family() -> None:
    direct = InsideOutsideParserSpec(width=4)
    residual = InsideOutsideParserSpec(
        width=4, composition=GatedResidualCompositionSpec()
    )

    assert direct.composition == MeanFieldCompositionSpec()
    assert direct.settings().composition.kind == "mean-field"
    assert residual.settings().composition.kind == "gated-residual"


def test_the_removed_top_level_functions_are_not_a_second_route() -> None:
    with pytest.raises(dx.ValidationError, match="unary"):
        ParserSpec.model_validate(
            {
                "kind": "inside-outside",
                "width": 4,
                "unary": {"kind": "mlp", "layers": 1, "activation": "tanh"},
            }
        )


def test_an_unknown_tag_is_refused_with_the_tag_named() -> None:
    with pytest.raises(dx.ValidationError, match="'shift-reduce'"):
        ParserSpec.model_validate({"kind": "shift-reduce"})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "inside-outside", "width": 4, '
        '"composition": {"kind": "mean-field", '
        '"leaf": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"cell": {"kind": "mlp", "layers": 1, "activation": "tanh"}}, '
        '"reduction": {"kind": "log-sum-exp"}, '
        '"attend": {"kind": "marginal"}, '
        '"split_left": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"split_right": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"share_splits": false, '
        '"transition": {"kind": "mlp", "layers": 1, "activation": "tanh"}, '
        '"depth": 1, "layer_sharing": "independent", '
        '"outside_weights": "contextual", "root_outside": "learned", '
        '"root_init_gain": null, "declared_output": null}'
    )
    declared = ParserSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert ParserSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_chart_derives_rule_arity_from_the_supplied_actions() -> None:
    declared = InsideOutsideParserSpec(width=4)

    assert not isinstance(declared, BoundedConsumer)


def test_the_chart_holds_the_controller_its_rules_port_supplies() -> None:
    assert InsideOutsideParserSpec(width=4).sharing() == {
        "rules": SubmoduleDeclaration(
            "policy", ModuleSharing(in_parameters=True, in_state_dict=True)
        )
    }


def test_a_learned_root_declares_its_initialization_gain() -> None:
    assert InsideOutsideParserSpec(width=4).settings().root_init_gain == 1.0
    assert (
        InsideOutsideParserSpec(width=4, root_init_gain=0.5).settings().root_init_gain
        == 0.5
    )

    with pytest.raises(dx.ValidationError, match="above zero"):
        InsideOutsideParserSpec(width=4, root_init_gain=0.0)

    with pytest.raises(dx.ValidationError, match="only for a learned root"):
        InsideOutsideParserSpec(width=4, root_outside="zero", root_init_gain=0.5)

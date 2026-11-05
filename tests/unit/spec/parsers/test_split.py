"""Tests for split-weigher declarations."""

from typing import cast

from lofi.spec.parsers.split import (
    AttendSplitWeigherSettings,
    AttendSplitWeigherSpec,
    DeclaredSplitWeigher,
    MarginalSplitWeigherSettings,
    MarginalSplitWeigherSpec,
    SplitWeigherDeclaration,
    SplitWeigherSpec,
)
from lofi.spec.span_functions import (
    MlpSpanFunctionSettings,
    MlpSpanFunctionSpec,
    SpanFunctionDeclaration,
)


def test_each_split_weigher_tag_reaches_one_current_declaration() -> None:
    marginal = SplitWeigherSpec.model_validate({"kind": "marginal"})
    attend = SplitWeigherSpec.model_validate({"kind": "attend"})

    assert marginal == MarginalSplitWeigherSpec()
    assert attend == AttendSplitWeigherSpec(
        score=MlpSpanFunctionSpec(layers=1, activation="tanh")
    )
    assert cast("DeclaredSplitWeigher", marginal).declaration() == (
        SplitWeigherDeclaration("marginal", MarginalSplitWeigherSettings())
    )
    assert cast("DeclaredSplitWeigher", attend).declaration() == (
        SplitWeigherDeclaration(
            "attend",
            AttendSplitWeigherSettings(
                SpanFunctionDeclaration("mlp", MlpSpanFunctionSettings(1, "tanh"))
            ),
        )
    )


def test_no_local_attend_declaration_or_variant_survives() -> None:
    assert set(SplitWeigherSpec.__variants__) == {"marginal", "attend"}

from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec import (
    LearnerInterpretationSpec,
    StructuredLearnerMeaningSourceSpec,
)
from lofi.spec.semantics import LearnerClassWeightSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _structured_source() -> dict[str, JsonValue]:
    return cast(
        "dict[str, JsonValue]",
        {
            "kind": "structured",
            "structure": {
                "joint": {
                    "spelling": "tabular",
                    "rows": (
                        {"logical_class": "predicate", "size": 4, "share": 0.5},
                        {"logical_class": "constant", "size": 2, "share": 0.5},
                    ),
                },
                "assignment": {
                    "kind": "tabular",
                    "rows": (
                        {
                            "category": "NP",
                            "classes": ({"logical_class": "constant"},),
                        },
                        {
                            "category": "S\\NP",
                            "classes": ({"logical_class": "predicate", "arity": 1},),
                        },
                    ),
                },
                "arities": {"arities": ({"arity": 1, "share": 1.0},)},
                "fragment": {"order": "first"},
                "seed": 0,
                "constants": 6,
            },
        },
    )


def test_an_inline_structured_learner_interpretation_validates_at_its_boundary() -> (
    None
):
    interpretation = LearnerInterpretationSpec.model_validate(
        cast(
            "dict[str, JsonValue]",
            {
                "entity": "e",
                "truth": "t",
                "erasure": {"kind": "signature-morphism", "atoms": {"S": "t"}},
                "constants": (),
                "meanings": {"source": _structured_source()},
            },
        )
    )

    assert isinstance(
        interpretation.meanings.source, StructuredLearnerMeaningSourceSpec
    )
    assert (interpretation.entity, interpretation.truth) == ("e", "t")
    assert interpretation.meanings.source.structure is not None
    assert interpretation.meanings.source.structure.constants == 6


def test_a_learner_interpretation_requires_its_semantic_type_names() -> None:
    with pytest.raises(dx.ValidationError, match="entity"):
        LearnerInterpretationSpec.model_validate(
            cast(
                "dict[str, JsonValue]",
                {
                    "truth": "proposition",
                    "erasure": {"kind": "signature-morphism", "atoms": {}},
                    "constants": (),
                    "meanings": {"source": {"kind": "lexical"}},
                },
            )
        )


def test_structured_learner_meanings_require_one_spelling() -> None:
    with pytest.raises(dx.ValidationError, match="one structure or preset"):
        StructuredLearnerMeaningSourceSpec.model_validate(
            {
                "kind": "structured",
                "structure": _structured_source()["structure"],
                "preset": "packaged",
            }
        )


def test_a_structured_learner_predicate_joint_requires_an_arity_law() -> None:
    structure = dict(cast("dict[str, JsonValue]", _structured_source()["structure"]))
    structure.pop("arities")

    with pytest.raises(dx.ValidationError, match="needs an arity law"):
        StructuredLearnerMeaningSourceSpec.model_validate(
            {"kind": "structured", "structure": structure}
        )


def test_a_learner_assignment_can_condition_connectives_on_arity() -> None:
    row = LearnerClassWeightSpec(logical_class="connective", arity=2)

    assert (row.logical_class, row.arity) == ("connective", 2)

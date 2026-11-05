from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest
from tests.conformance.evaluator import packaged_held_out_pass

from lofi._types import ItemId, ReadoutId
from lofi.data import (
    ByteSpan,
    Derivation,
    DerivationNode,
    Example,
    GoldDerivation,
    Item,
    Token,
    TokenSpan,
)
from lofi.eval import HeldOutEvaluationContext, InferenceEvaluationContext
from lofi.eval._held_out import (
    LexicalTrainingEvidence,
    derivation_dataset,
    surface_derivation_dataset,
)
from lofi.lf import THREE_WAY

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from lofi.data.synthetic import InterpretationFunction


def test_the_context_freezes_held_out_resources() -> None:
    held_out = packaged_held_out_pass()
    grammar = held_out.grammar
    parse = held_out.context.parse
    context = HeldOutEvaluationContext(
        grammar,
        interpretation=None,
        forms={ItemId("one"): "p(a)"},
        coherence={},
        target_sets={},
        lexical_training={"the": LexicalTrainingEvidence(4, 2)},
        parse=parse,
        training_items=12,
    )

    assert context.training_items == 12
    assert context.grammar is grammar
    assert context.parse is parse
    assert context.interpretation is None
    assert context.forms == {ItemId("one"): "p(a)"}
    assert isinstance(context.forms, MappingProxyType)
    assert context.coherence == {}
    assert isinstance(context.coherence, MappingProxyType)
    assert context.target_sets == {}
    assert isinstance(context.target_sets, MappingProxyType)
    assert context.lexical_training == {"the": LexicalTrainingEvidence(4, 2)}
    assert isinstance(context.lexical_training, MappingProxyType)

    with pytest.raises(TypeError):
        cast("MutableMapping[ItemId, str]", context.forms)[ItemId("two")] = "q(b)"

    with pytest.raises(TypeError):
        cast(
            "MutableMapping[str, LexicalTrainingEvidence]",
            context.lexical_training,
        )["a"] = LexicalTrainingEvidence(1, 1)


@pytest.mark.parametrize(
    ("frequency", "ambiguity"),
    [
        (-1, 0),
        (0, 1),
    ],
)
def test_invalid_lexical_training_evidence_is_refused(
    frequency: int,
    ambiguity: int,
) -> None:
    with pytest.raises(ValueError, match=r"negative|exceeds"):
        LexicalTrainingEvidence(frequency, ambiguity)


def test_the_inference_context_owns_the_declared_response_scale() -> None:
    held_out = packaged_held_out_pass()
    grammar = held_out.grammar
    parse = held_out.context.parse
    labels = tuple(label.value for label in THREE_WAY.labels)
    context = InferenceEvaluationContext(
        grammar,
        labels,
        interpretation=cast("InterpretationFunction", object()),
        forms={},
        tokens={},
        parse=parse,
    )

    assert context.grammar is grammar
    assert context.parse is parse
    assert context.labels == labels


def test_the_packaged_forward_publishes_the_symbolic_owner_of_root_mass() -> None:
    context = packaged_held_out_pass().context

    assert "sentences" in context.target_sets
    assert context.target_sets[ReadoutId("sentences")].categories


def test_the_packaged_context_counts_its_distinct_training_items() -> None:
    held_out = packaged_held_out_pass()
    training = held_out.context.training_items

    assert training is not None
    assert 0 < training <= len(held_out.gold)


def _retained() -> tuple[Example[GoldDerivation[str]], ...]:
    tokens = (Token(0, "a", ByteSpan(0, 1)), Token(1, "b", ByteSpan(2, 3)))
    derivation = Derivation(
        2,
        {
            0: DerivationNode(TokenSpan(0, 1), "A"),
            1: DerivationNode(TokenSpan(1, 2), "B"),
            2: DerivationNode(TokenSpan(0, 2), "S", rule="combine", children=(0, 1)),
        },
    )
    item = Item(ItemId("one"), tokens, text="a b")
    return (Example(item, GoldDerivation(derivation)),)


def test_a_derivation_dataset_holds_what_the_evaluator_reads_off_each_derivation() -> (
    None
):
    retained = _retained()
    data = derivation_dataset(
        lambda item, derivation: (item.item_id, derivation.root), retained
    )

    assert data.ids == (ItemId("one"),)
    assert data.annotations[0].value == (ItemId("one"), 2)


def test_a_surface_derivation_dataset_keeps_the_category_text_derivation() -> None:
    retained = _retained()
    data = surface_derivation_dataset(retained)

    assert data.annotations[0].value == retained[0].supervision.derivation

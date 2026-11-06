"""Frozen resources shared by held-out evaluator plugins."""

from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data import Annotation, EvalDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import ItemId, ReadoutId
    from lofi.data import Derivation, Example, GoldDerivation, Item
    from lofi.data.synthetic import InterpretationFunction
    from lofi.deduce import ParseSettings
    from lofi.nn.loss._ports import CoherenceEvaluationObjective
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import TargetSet, Term

__all__ = [
    "HeldOutEvaluationContext",
    "InferenceEvaluationContext",
    "LexicalTrainingEvidence",
    "derivation_dataset",
    "surface_derivation_dataset",
]


def derivation_dataset[Gold](
    annotate: Callable[[Item, Derivation[str]], Gold],
    examples: Sequence[Example[GoldDerivation[str]]],
    /,
) -> EvalDataset[Gold]:
    """Return retained items with the gold analyses their derivations give.

    Parameters
    ----------
    annotate : Callable[[Item, Derivation[str]], Gold]
        The analysis an evaluator reads off one item's gold derivation.
    examples : Sequence[Example[GoldDerivation[str]]]
        The retained items with their category-text derivations.
    """
    return EvalDataset(
        (example.item for example in examples),
        tuple(
            Annotation(
                example.item.item_id,
                annotate(example.item, example.supervision.derivation),
            )
            for example in examples
        ),
    )


def surface_derivation_dataset(
    examples: Sequence[Example[GoldDerivation[str]]], /
) -> EvalDataset[Derivation[str]]:
    """Return retained items paired with their category-text derivations.

    Parameters
    ----------
    examples : Sequence[Example[GoldDerivation[str]]]
        The retained items with their gold derivations.
    """
    return EvalDataset(
        (example.item for example in examples),
        tuple(
            Annotation(example.item.item_id, example.supervision.derivation)
            for example in examples
        ),
    )


@final
class LexicalTrainingEvidence(FrozenValue):
    """Training frequency and observed gold ambiguity of one token form."""

    _noun = "lexical training evidence"
    __slots__ = ("_ambiguity", "_frequency")

    def __init__(self, frequency: int, ambiguity: int, /) -> None:
        held_frequency = ensure_non_negative("a training frequency", frequency)
        held_ambiguity = ensure_non_negative("a gold lexical ambiguity", ambiguity)
        if held_ambiguity > held_frequency:
            msg = (
                f"gold lexical ambiguity {held_ambiguity} exceeds training "
                f"frequency {held_frequency}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "_frequency", held_frequency)
        object.__setattr__(self, "_ambiguity", held_ambiguity)

    @property
    def frequency(self) -> int:
        """Return the form's token count in the training partition."""
        return self._frequency

    @property
    def ambiguity(self) -> int:
        """Return its distinct gold category count in training."""
        return self._ambiguity

    def _fields(self) -> tuple[int, int]:
        """Return fields in constructor order."""
        return self._frequency, self._ambiguity

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value holds the same two counts."""
        if type(other) is not LexicalTrainingEvidence:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the two counts."""
        return hash((LexicalTrainingEvidence, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"LexicalTrainingEvidence{self._fields()!r}"


@final
class HeldOutEvaluationContext(FrozenValue):
    """Resources a held-out evaluator may read from the lowered dataset.

    Parameters
    ----------
    grammar : GrammarAssembly
        The grammar the parser was trained under.
    interpretation : InterpretationFunction | None
        The declared interpretation, or ``None`` when the dataset has none.
    forms : Mapping[ItemId, str]
        Gold logical form texts keyed by item identifier.
    lexical_training : Mapping[str, LexicalTrainingEvidence]
        Training token frequency and observed gold ambiguity by token form.
    parse : ParseSettings[object]
        The run's declared symbolic parse, which every decoding parse uses.
    training_items : int | None
        The number of distinct items the parser was trained on, or ``None``
        where the context is built without a training partition.
    """

    _noun = "held-out evaluation context"
    __slots__ = (
        "_coherence",
        "_forms",
        "_grammar",
        "_interpretation",
        "_lexical_training",
        "_parse",
        "_target_sets",
        "_training_items",
    )

    def __init__(
        self,
        grammar: GrammarAssembly,
        /,
        *,
        interpretation: InterpretationFunction | None,
        forms: Mapping[ItemId, str],
        coherence: Mapping[str, CoherenceEvaluationObjective],
        target_sets: Mapping[ReadoutId, TargetSet[Term]],
        lexical_training: Mapping[str, LexicalTrainingEvidence],
        parse: ParseSettings[object],
        training_items: int | None,
    ) -> None:
        evidence: dict[str, LexicalTrainingEvidence] = {}
        for form, counts in lexical_training.items():
            surface = ensure_name("a lexical training form", form)
            if not isinstance(counts, LexicalTrainingEvidence):
                msg = f"lexical training evidence for {surface!r} is {counts!r}"
                raise TypeError(msg)
            evidence[surface] = counts
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_interpretation", interpretation)
        object.__setattr__(self, "_forms", MappingProxyType(dict(forms)))
        object.__setattr__(self, "_coherence", MappingProxyType(dict(coherence)))
        object.__setattr__(self, "_target_sets", MappingProxyType(dict(target_sets)))
        object.__setattr__(self, "_lexical_training", MappingProxyType(evidence))
        object.__setattr__(self, "_parse", parse)
        object.__setattr__(
            self,
            "_training_items",
            None
            if training_items is None
            else ensure_non_negative("a training item count", training_items),
        )

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the grammar the parser was trained under."""
        return self._grammar

    @property
    def parse(self) -> ParseSettings[object]:
        """Return the run's declared symbolic parse."""
        return self._parse

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the declared interpretation, if the dataset has one."""
        return self._interpretation

    @property
    def forms(self) -> Mapping[ItemId, str]:
        """Return gold logical form texts keyed by item identifier."""
        return self._forms

    @property
    def coherence(self) -> Mapping[str, CoherenceEvaluationObjective]:
        """Return held-out coherence sites keyed by their objective role."""
        return self._coherence

    @property
    def target_sets(self) -> Mapping[ReadoutId, TargetSet[Term]]:
        """Return symbolic target sets keyed by their published mass id."""
        return self._target_sets

    @property
    def lexical_training(self) -> Mapping[str, LexicalTrainingEvidence]:
        """Return training frequency and gold ambiguity by token form."""
        return self._lexical_training

    @property
    def training_items(self) -> int | None:
        """Return the number of distinct training items, when it is known."""
        return self._training_items


@final
class InferenceEvaluationContext(FrozenValue):
    """Resources an inference label evaluator reads from a lowered pair dataset."""

    _noun = "inference evaluation context"
    __slots__ = (
        "_forms",
        "_grammar",
        "_interpretation",
        "_labels",
        "_parse",
        "_tokens",
    )

    def __init__(
        self,
        grammar: GrammarAssembly,
        labels: tuple[str, ...],
        /,
        *,
        interpretation: InterpretationFunction,
        forms: Mapping[ItemId, tuple[Term, Term]],
        tokens: Mapping[ItemId, tuple[tuple[str, ...], tuple[str, ...]]],
        parse: ParseSettings[object],
    ) -> None:
        declared = tuple(ensure_name("an inference label", label) for label in labels)

        if len(declared) < 2 or len(set(declared)) != len(declared):
            msg = f"an inference evaluation declares {len(declared)} label"
            raise ValueError(msg)

        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_labels", declared)
        object.__setattr__(self, "_interpretation", interpretation)
        object.__setattr__(self, "_forms", MappingProxyType(dict(forms)))
        object.__setattr__(self, "_tokens", MappingProxyType(dict(tokens)))
        object.__setattr__(self, "_parse", parse)

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the grammar the grounding was trained under."""
        return self._grammar

    @property
    def parse(self) -> ParseSettings[object]:
        """Return the run's declared symbolic parse."""
        return self._parse

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the response scale in the data's declared column order."""
        return self._labels

    @property
    def interpretation(self) -> InterpretationFunction:
        """Return the interpretation that labeled the retained pairs."""
        return self._interpretation

    @property
    def forms(self) -> Mapping[ItemId, tuple[Term, Term]]:
        """Return retained premise and hypothesis forms by pair identifier."""
        return self._forms

    @property
    def tokens(self) -> Mapping[ItemId, tuple[tuple[str, ...], tuple[str, ...]]]:
        """Return retained premise and hypothesis tokens by pair identifier."""
        return self._tokens

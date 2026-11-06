"""Objective protocols and published parser outputs."""

from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, Self, runtime_checkable

from lofi._checks import CHECKS
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from torch import Tensor

    from lofi._types import ItemId, ReadoutId
    from lofi.deduce import SpanIndex
    from lofi.judge import Action
    from lofi.lf.provers import ProofService
    from lofi.nn._values import Encoded, LogDistribution
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.grounding import GroundedBatch
    from lofi.nn.loss._constraint import (
        BatchLocalConstraint,
        BatchLocalNegativeBracket,
    )
    from lofi.nn.loss._parser import SpanMarginals, SplitPosterior
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.loss._typing import TensorSoftTypingInstance
    from lofi.nn.loss._values import CellChart, SpanReadout, SpanTerms
    from lofi.nn.ports import ForwardMode, ForwardOptions
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.nn.tokens.embedder import TokenVocabulary
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import BatchEncoding, TargetSet, Term

__all__ = [
    "AmbiguousTokens",
    "BoundObjective",
    "CoherenceEvaluation",
    "CoherenceEvaluationObjective",
    "CombinatorDecoder",
    "CombinatorScorer",
    "ConstrainedItems",
    "FormCredit",
    "FormTypingItems",
    "GoldFormItems",
    "LabeledItems",
    "LossComponentBindings",
    "LossTerm",
    "Objective",
    "ProofRewardBindings",
    "PublishedOutputs",
    "ResolvableConstraint",
    "RuleInstance",
    "RuleInstances",
    "ScheduledBatch",
    "ScheduledRuleInstances",
    "SoftTypingItems",
    "SpanTokenDecoder",
    "SurfaceItems",
    "TargetSetSource",
    "TermProbabilityReadout",
    "TokenizedItems",
    "TypeItems",
    "TypePairs",
]


type CoherenceContribution = tuple[str, int, float]
"""One action's weighted coherence at one span length."""


class CoherenceEvaluation(FrozenValue):
    """Detached per-item coherence contributions from one held-out pass."""

    _noun = "coherence evaluation"
    __slots__ = ("_contributions", "_item")

    def __init__(
        self,
        item: int,
        contributions: tuple[CoherenceContribution, ...],
        /,
    ) -> None:
        if type(item) is not int or item < 0:
            msg = f"a coherence evaluation has invalid item index {item!r}"
            raise ValueError(msg)

        checked: list[CoherenceContribution] = []
        for action, span_length, value in contributions:
            if not action:
                msg = "a coherence contribution names no action"
                raise ValueError(msg)
            if type(span_length) is not int or span_length < 1:
                msg = f"a coherence contribution has span length {span_length!r}"
                raise ValueError(msg)
            if not isfinite(value):
                msg = f"a coherence contribution is not finite: {value!r}"
                raise ValueError(msg)
            checked.append((action, span_length, value))

        object.__setattr__(self, "_item", item)
        object.__setattr__(self, "_contributions", tuple(checked))

    @property
    def item(self) -> int:
        """Return the batch position this evaluation describes."""
        return self._item

    @property
    def contributions(self) -> tuple[CoherenceContribution, ...]:
        """Return weighted action and span-length contributions."""
        return self._contributions


@runtime_checkable
class CoherenceEvaluationObjective(Protocol):
    """A coherence objective that publishes detached held-out evaluations."""

    @property
    def name(self) -> str:
        """Return the objective's declared reporting name."""

    def evaluation(
        self, outputs: PublishedOutputs[ScheduledBatch], /
    ) -> tuple[CoherenceEvaluation, ...]:
        """Return detached per-item coherence contributions."""


@runtime_checkable
class TargetSetSource(Protocol):
    """A configured site that publishes the symbolic sets behind target mass."""

    @property
    def target_sets(self) -> Mapping[ReadoutId, TargetSet[Term]]:
        """Return symbolic target sets keyed by their published mass id."""


def _require_scalar(name: str, tensor: Tensor) -> None:
    """Raise unless a tensor is zero-dimensional.

    Parameters
    ----------
    name : str
        The field the tensor was passed as, for the message.
    tensor : Tensor
        The tensor to check.

    Raises
    ------
    ValueError
        When the tensor has any dimension.
    """
    if tensor.ndim != 0:
        msg = f"{name} must be a scalar, but its shape is {tuple(tensor.shape)}"
        raise ValueError(msg)


class LossTerm:
    """Stores one objective's unweighted contribution for a batch.

    Parameters
    ----------
    value : Tensor
        The unweighted scalar contribution.
    count : int
        How many items contributed.
    residual : Tensor | None
        The probability mass an objective omitted where it truncated an infinite sum, or
        None where it truncated nothing.
    diagnostics : Mapping[str, Tensor]
        Named scalar diagnostics.

    Raises
    ------
    ValueError
        When the debug checks are on and the count is negative or one of the tensors is
        not a scalar.
    """

    __slots__ = ("_count", "_diagnostics", "_residual", "_value")

    def __init__(
        self,
        value: Tensor,
        count: int,
        *,
        residual: Tensor | None = None,
        diagnostics: Mapping[str, Tensor] = MappingProxyType({}),
    ) -> None:
        if CHECKS:
            _require_scalar("value", value)

            if count < 0:
                msg = f"count is a number of items and cannot be negative: {count}"
                raise ValueError(msg)

            if residual is not None:
                _require_scalar("residual", residual)

            for name, diagnostic in diagnostics.items():
                _require_scalar(f"diagnostics[{name!r}]", diagnostic)

        self._value = value
        self._count = count
        self._residual = residual
        self._diagnostics: Mapping[str, Tensor] = MappingProxyType(dict(diagnostics))

    @property
    def value(self) -> Tensor:
        """Return the unweighted scalar contribution for this batch."""
        return self._value

    @property
    def count(self) -> int:
        """Return the number of contributing items."""
        return self._count

    @property
    def residual(self) -> Tensor | None:
        """Return the omitted probability mass, if the sum was truncated."""
        return self._residual

    @property
    def diagnostics(self) -> Mapping[str, Tensor]:
        """Return the named scalar diagnostics as a read-only view."""
        return self._diagnostics

    @classmethod
    def inapplicable(cls, like: Tensor) -> Self:
        """Return the zero term for a batch the objective does not apply to.

        Parameters
        ----------
        like : Tensor
            A tensor from the batch the term would have scored.
        """
        return cls(like.unsqueeze(0)[:0].sum(), 0)


@runtime_checkable
class Objective[OutputsT](Protocol):
    """Computes one training objective."""

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads, by name."""

    def evaluate(self, outputs: OutputsT, /) -> LossTerm:
        """Return this objective's contribution for one step.

        Parameters
        ----------
        outputs : OutputsT
            What the graph published for the step.
        """


@runtime_checkable
class ResolvableConstraint[T, ScopeT: Hashable](Protocol):
    """A dataset constraint that can be localized to one batch."""

    def to_spans(
        self,
        batch_item_ids: Sequence[ItemId],
        /,
        *,
        readout: ReadoutId,
    ) -> tuple[BatchLocalConstraint[T, ScopeT], ...]:
        """Return this constraint localized to one batch.

        Parameters
        ----------
        batch_item_ids : Sequence[ItemId]
            Dataset item identifiers in batch order.
        readout : ReadoutId
            Published target mass assigned by the wiring layer.

        Returns
        -------
        tuple[BatchLocalConstraint[T, ScopeT], ...]
            Constraint values addressed by batch position.
        """


@runtime_checkable
class PublishedOutputs[BatchT](Protocol):
    """Parser outputs available to objectives."""

    @property
    def spans(self) -> SpanIndex:
        """Return the span addressing every chart index in the bundle is over."""

    @property
    def chart(self) -> CellChart:
        """Return per-span inside and outside scores."""

    @property
    def splits(self) -> SplitPosterior | None:
        """Return per-span split posteriors, if available."""

    @property
    def scores(self) -> LogDistribution | None:
        """Return per-item log inside scores, if available."""

    @property
    def marginals(self) -> SpanMarginals | None:
        """Return the chart's two marginals if published."""

    @property
    def actions(self) -> Mapping[Action, LogDistribution]:
        """Return the log weight of each action per span, keyed by the action itself."""

    @property
    def readouts(self) -> Mapping[ReadoutId, SpanReadout]:
        """Return what each readout published, keyed by the readout's id."""

    @property
    def choice_rows(self) -> Mapping[ReadoutId, SpanReadout]:
        """Return payload-aware decoder marginals keyed by readout id."""

    @property
    def terms(self) -> Mapping[ReadoutId, SpanTerms]:
        """Return weighted complete terms keyed by their readout id."""

    @property
    def forms(self) -> Mapping[ReadoutId, WeightedTerms]:
        """Return weighted logical forms keyed by their side id."""

    @property
    def entailment(self) -> GroundedBatch | None:
        """Return the grounded inference distribution, if one was published."""

    @property
    def cells(self) -> Encoded | None:
        """Return the representation of each span if published."""

    @property
    def outside_cells(self) -> Encoded | None:
        """Return each span's contextual outside representation if published."""

    @property
    def entropy(self) -> Encoded | None:
        """Return derivation entropy per item if published."""

    @property
    def mbr(self) -> Encoded | None:
        """Return MBR-tree constituent indicators if published."""

    @property
    def batch(self) -> BatchT:
        """Return the input batch passed through the neural layer."""


@runtime_checkable
class BoundObjective[ContextT, OutputsT](Protocol):
    """Binds an objective after dependent components are constructed."""

    @property
    def name(self) -> str:
        """Return the name the bound objective is logged and namespaced under."""

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs the bound objective reads, by name."""

    def bind(self, context: ContextT, /) -> Objective[OutputsT]:
        """Return the objective this becomes once the context can supply it.

        Parameters
        ----------
        context : ContextT
            The built component context.
        """


@runtime_checkable
class LossComponentBindings(Protocol):
    """Resolves built loss dependencies by their declared model roles."""

    @property
    def name(self) -> str:
        """Return the declared name the objective being built is reported under."""

    def terms_readout(self, role: str, /) -> TermsReadout:
        """Return the term readout installed under a role.

        Parameters
        ----------
        role : str
            Role named by the loss declaration.
        """

    def space(self, role: str, /) -> Space:
        """Return the represented space installed under a role.

        Parameters
        ----------
        role : str
            Role named by the loss declaration.
        """

    def readout_space(self, role: str, /) -> Space:
        """Return the represented space read by one term-readout role."""

    def relation(self, role: str, /) -> SpaceRelationBinding:
        """Return the cross-space relation declared by one source-space role."""

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        """Return the span decoder installed under a role.

        Parameters
        ----------
        role : str
            Role named by the loss declaration.
        """

    def proof_reward(self) -> ProofRewardBindings:
        """Return the language and proof service for shaped proof credit."""


@runtime_checkable
class ProofRewardBindings(Protocol):
    """Language-owned capabilities used to score shaped proof credit."""

    @property
    def language(self) -> LanguageAssembly:
        """Return the logical form language whose forms are scored."""

    @property
    def service(self) -> ProofService:
        """Return the shared proof service that decides equivalence."""


@runtime_checkable
class SurfaceItems(Protocol):
    """A batch naming the words of each of its items."""

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return the words of each item, in item order and then in word order."""


class _ObservedFormTyping(Protocol):
    """A parsed logical form with the typing label presented to training."""

    @property
    def term(self) -> Term:
        """Return the parsed logical form."""

    @property
    def observed_label(self) -> Literal["well-typed", "ill-typed"]:
        """Return the label presented to training."""


@runtime_checkable
class FormTypingItems(SurfaceItems, Protocol):
    """A surface batch with logical form typing targets aligned by item."""

    @property
    def form_typing(self) -> Sequence[Sequence[_ObservedFormTyping]]:
        """Return the logical form typing targets aligned to surface items."""


@runtime_checkable
class GoldFormItems(SurfaceItems, Protocol):
    """A surface batch carrying at most one certified form per item."""

    @property
    def gold_forms(self) -> Sequence[Term | None]:
        """Return each item's certified form, or none where it has no form."""


@runtime_checkable
class TermProbabilityReadout(Protocol):
    """Scores supplied complete terms from aligned represented vectors."""

    def score_terms(self, inputs: Encoded, terms: Sequence[Term], /) -> LogDistribution:
        """Return one complete term log probability per input vector."""


@runtime_checkable
class TokenizedItems(Protocol):
    """A batch naming the vocabulary row of every word of each of its items."""

    @property
    def token_ids(self) -> Tensor:
        """Return the row of every word, of rank 2 over the items and the positions.

        The row is at integer precision, and the positions past an item's own words
        hold whatever the collation padded with, which no span reaches.
        """

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return how many words each item holds, in item order."""


@runtime_checkable
class SpanTokenDecoder(Protocol):
    """Scores the words a span covers against the representation of that span."""

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the vocabulary whose rows the scored distribution is over."""

    @property
    def end(self) -> str:
        """Return the reserved token each span's words are followed by."""

    def score_tokens(self, prefix: Encoded, /, *, targets: Tensor) -> LogDistribution:
        """Return the log probability of each token at each position of each span.

        Parameters
        ----------
        prefix : Encoded
            One vector per span, over the axes ``("span", "width")``, which the decoder
            reads as the position before the span's first word rather than as a
            cross-attention memory of one row.
        targets : Tensor
            The rows to score, of rank 2 over the spans and then their positions, at
            integer precision, in the order of the prefix's span axis.

        Returns
        -------
        LogDistribution
            Rank 3 over the axes ``("span", "position", "token")``, at floating
            precision, normalized over the token axis, giving each token's log
            probability at a position from the prefix and the positions before it.
        """


@runtime_checkable
class CombinatorDecoder(Protocol):
    """A decoder of parent-symbol distributions from pairs of child cells."""

    @property
    def modes(self) -> frozenset[ForwardMode]:
        """Return the forward modes this decoder runs in."""

    def compose(
        self, left: Encoded, /, *, right: Encoded, options: ForwardOptions
    ) -> LogDistribution:
        """Return the distribution over the symbol at each parent slot.

        Parameters
        ----------
        left : Encoded
            The left child of each pair, over the axes ``("term", "width")``.
        right : Encoded
            The right child of each pair, in the order of the term axis.
        options : ForwardOptions
            The pass to run, whose mode the decoder declares support for.

        Returns
        -------
        LogDistribution
            Rank 3 over the axes ``("term", "slot", "symbol")``, at floating precision,
            normalized over the symbol axis.

        Raises
        ------
        ValueError
            If the pass names a mode this decoder does not declare.
        """


@runtime_checkable
class CombinatorScorer(Protocol):
    """A scorer of conclusions against one rule's premise tuples."""

    @property
    def arity(self) -> int:
        """Return the premise count this scorer consumes."""

    @property
    def slots(self) -> int:
        """Return the number of conclusion slots this scorer predicts."""

    @property
    def modes(self) -> frozenset[ForwardMode]:
        """Return the forward modes this scorer runs in."""

    def score(
        self,
        premises: Sequence[Sequence[Term]],
        /,
        *,
        conclusions: Sequence[Term],
        options: ForwardOptions,
    ) -> LogDistribution:
        """Return one conclusion log weight per aligned premise tuple."""

    def exact_matches(
        self,
        premises: Sequence[Sequence[Term]],
        /,
        *,
        conclusions: Sequence[Term],
        options: ForwardOptions,
    ) -> Tensor:
        """Return whether every predicted conclusion choice matches its target."""


@runtime_checkable
class TypeItems(Protocol):
    """A batch of linearized types."""

    @property
    def terms(self) -> BatchEncoding:
        """Return the linearized types, one row per item."""


@runtime_checkable
class AmbiguousTokens(SurfaceItems, Protocol):
    """A batch of items whose words each may take several types."""

    @property
    def types(self) -> BatchEncoding:
        """Return the linearized types the words may take, one row per candidate."""

    @property
    def positions(self) -> tuple[tuple[int, int], ...]:
        """Return the item and the word each candidate row belongs to, in row order."""


@runtime_checkable
class ConstrainedItems[T, ScopeT: Hashable](Protocol):
    """A batch with span constraints localized to batch positions."""

    @property
    def constraints(
        self,
    ) -> tuple[
        BatchLocalConstraint[T, ScopeT] | BatchLocalNegativeBracket[ScopeT], ...
    ]:
        """Return the constraints on this batch, each over positions in it."""


@runtime_checkable
class ScheduledBatch(Protocol):
    """A batch that names the training step."""

    @property
    def step(self) -> int:
        """Return the zero-based training step."""


@runtime_checkable
class TypePairs(Protocol):
    """A batch of pairs of types, each pair held as terms and as linearizations."""

    @property
    def pairs(self) -> tuple[tuple[Term, Term], ...]:
        """Return the two types of each pair, in item order."""

    @property
    def left(self) -> BatchEncoding:
        """Return the linearization of the first type of each pair, in item order."""

    @property
    def right(self) -> BatchEncoding:
        """Return the linearization of the second type of each pair, in item order."""


@runtime_checkable
class RuleInstance(Protocol):
    """One symbolic rule application at its declared premise arity."""

    @property
    def inputs(self) -> tuple[Term, ...]:
        """Return the premises in rule order."""

    @property
    def output(self) -> Term:
        """Return the conclusion."""

    @property
    def actions(self) -> tuple[Action, ...]:
        """Return every action whose rule admits this application."""


@runtime_checkable
class RuleInstances(Protocol):
    """A batch of rule instances at their declared premise arities."""

    @property
    def instances(self) -> Sequence[RuleInstance]:
        """Return premise tuples, conclusions and admitted actions in batch order."""


@runtime_checkable
class ScheduledRuleInstances(RuleInstances, Protocol):
    """A batch of labeled rule instances that names the step of the run it trains."""

    @property
    def step(self) -> int:
        """Return which step of the run this batch trains, counting from zero."""

    @property
    def horizon(self) -> int:
        """Return how many steps the run has."""


@runtime_checkable
class SoftTypingItems(Protocol):
    """A batch of tensor-valued typing-rule applications."""

    @property
    def typing_instances(self) -> tuple[TensorSoftTypingInstance, ...]:
        """Return the typing-rule applications to score."""


@runtime_checkable
class LabeledItems(Protocol):
    """A batch of items each labeled grammatical, ungrammatical, or left unlabeled."""

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the number of words of each item, in item order."""

    @property
    def labels(self) -> tuple[bool | None, ...]:
        """Return whether each item is grammatical, or None where it is unlabeled."""


@runtime_checkable
class FormCredit(Protocol):
    """A graded credit paid to one logical form for its shape alone.

    The credit is a function of the form and is constant with respect to every
    model parameter, so an expectation of it under a beam is differentiable in
    the beam's weights and in nothing else.
    """

    def credit(self, form: Term, /) -> float:
        """Return the credit one form earns, in the unit interval.

        Parameters
        ----------
        form : Term
            A closed logical form.
        """

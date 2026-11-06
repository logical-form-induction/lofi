"""The shaped credit a sampled logical form earns, and the objective that pays it."""

from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

import torch

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.lf import Abstain, Graded, Proved
from lofi.lf.provers import ProofBudget, ProofService, ProofSite
from lofi.nn.loss._errors import EveryItemAbstainedError
from lofi.nn.loss._ports import LossTerm
from lofi.registry import ComponentPlugin
from lofi.spec.losses.shaped_proof import (
    ShapedProofLossSettings,
    ShapedProofLossSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.lf import Refuted
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._ports import (
        GoldFormItems,
        LossComponentBindings,
        PublishedOutputs,
    )
    from lofi.nn.loss._values import SpanTerms
    from lofi.terms import OpName, Term
    from lofi.typecheck import LexicalSignature, TypingRule, TypingSystem

__all__ = [
    "FormOutcome",
    "ProofShaping",
    "ShapedAbstention",
    "ShapedCredit",
    "ShapedOutcome",
    "ShapedProofObjective",
    "ShapedProofReward",
]


class FormOutcome(StrEnum):
    """What a sampled logical form reached, in increasing order of credit."""

    #: The form leaves a bound index open, so it denotes nothing by itself.
    ILL_FORMED = "ill-formed"

    #: The form is closed, and the type checker left a constraint unsatisfied.
    WELL_FORMED = "well-formed"

    #: The form type-checks, and no proof against the reference was reached.
    TYPE_CHECKS = "type-checks"

    #: The form type-checks, and it and the reference entail each other.
    PROVED = "proved"


@final
class ProofShaping(FrozenValue):
    """The credit each outcome earns and the weight the length penalty carries.

    The four credits ladder upward, so a form that reached more is never paid
    less, and the length penalty is charged per node of the form on top of
    whichever credit it earned.

    Parameters
    ----------
    ill_formed : float
        What a form that leaves a bound index open earns.
    well_formed : float
        What a closed form the type checker rejects earns.
    type_checks : float
        What a form that type-checks without a proof against the reference
        earns, which is also what a countermodel leaves a form with.
    proved : float
        What a form that entails the reference and follows from it earns.
    length : float
        What one node of the form costs.

    Raises
    ------
    TypeError
        If a credit or the weight is not a number.
    ValueError
        If a credit or the weight is not finite, if the weight is negative, or
        if the credits do not ladder upward.
    """

    _noun = "proof shaping"

    __slots__ = ("_credits", "_length")

    def __init__(
        self,
        *,
        ill_formed: float,
        well_formed: float,
        type_checks: float,
        proved: float,
        length: float,
    ) -> None:
        earned = {
            FormOutcome.ILL_FORMED: ensure_finite("the ill-formed credit", ill_formed),
            FormOutcome.WELL_FORMED: ensure_finite(
                "the well-formed credit", well_formed
            ),
            FormOutcome.TYPE_CHECKS: ensure_finite(
                "the type-checking credit", type_checks
            ),
            FormOutcome.PROVED: ensure_finite("the proved credit", proved),
        }
        weight = ensure_finite("the length weight", length)

        if weight < 0.0:
            msg = f"the length weight is a cost and cannot be negative: {length!r}"

            raise ValueError(msg)

        _laddered(earned)

        object.__setattr__(self, "_credits", MappingProxyType(earned))
        object.__setattr__(self, "_length", weight)

    @property
    def length(self) -> float:
        """Return what one node of a form costs."""
        return self._length

    def credit(self, outcome: FormOutcome, /) -> float:
        """Return what one outcome earns, before the length penalty.

        Parameters
        ----------
        outcome : FormOutcome
            The class the form fell into.
        """
        return self._credits[outcome]

    def value(self, outcome: FormOutcome, size: int, /) -> float:
        """Return what one outcome earns at one form size.

        Parameters
        ----------
        outcome : FormOutcome
            The class the form fell into.
        size : int
            How many nodes the form has.
        """
        return self._credits[outcome] - self._length * size

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another shaping declares the same credits and weight."""
        if type(other) is not ProofShaping:
            return NotImplemented

        return self._credits == other._credits and self._length == other._length

    @override
    def __hash__(self) -> int:
        """Return a hash over the credits and the weight."""
        return hash(
            (
                ProofShaping,
                tuple(self._credits[outcome] for outcome in FormOutcome),
                self._length,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this shaping."""
        declared = ", ".join(
            f"{outcome.name.lower()}={self._credits[outcome]!r}"
            for outcome in FormOutcome
        )

        return f"ProofShaping({declared}, length={self._length!r})"


def _laddered(earned: Mapping[FormOutcome, float]) -> None:
    """Raise unless the declared credits never fall as the outcome improves.

    Parameters
    ----------
    earned : Mapping[FormOutcome, float]
        What each outcome earns.

    Raises
    ------
    ValueError
        If some outcome earns less than the one below it.
    """
    for below, above in pairwise(FormOutcome):
        if earned[above] < earned[below]:
            msg = (
                f"{above.value} earns {earned[above]} and {below.value} earns "
                f"{earned[below]}, so the shaping pays a form for reaching less"
            )

            raise ValueError(msg)


@final
class ShapedCredit(FrozenValue):
    """What one sampled form was paid, and the evidence the payment rests on.

    Parameters
    ----------
    outcome : FormOutcome
        The class the form fell into.
    credit : float
        What that outcome earns, before the length penalty.
    length : int
        How many nodes the form has, which the penalty is charged on.
    value : float
        The credit less the length penalty.
    verdict : Proved | Refuted | Graded | None
        The proof or the countermodel the outcome rests on, or None where the
        prover was never asked.
    """

    _noun = "shaped credit"

    __slots__ = ("_credit", "_length", "_outcome", "_value", "_verdict")

    def __init__(
        self,
        outcome: FormOutcome,
        credit: float,
        length: int,
        value: float,
        verdict: Proved | Refuted | Graded | None,
    ) -> None:
        object.__setattr__(self, "_outcome", outcome)
        object.__setattr__(self, "_credit", credit)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_verdict", verdict)

    @property
    def outcome(self) -> FormOutcome:
        """Return the class the form fell into."""
        return self._outcome

    @property
    def credit(self) -> float:
        """Return what the outcome earns, before the length penalty."""
        return self._credit

    @property
    def length(self) -> int:
        """Return how many nodes the form has."""
        return self._length

    @property
    def value(self) -> float:
        """Return the credit less the length penalty."""
        return self._value

    @property
    def verdict(self) -> Proved | Refuted | Graded | None:
        """Return the proof or the countermodel, or None where none was sought."""
        return self._verdict

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another payment agrees in every field."""
        if type(other) is not ShapedCredit:
            return NotImplemented

        return (
            self._outcome is other._outcome
            and self._credit == other._credit
            and self._length == other._length
            and self._value == other._value
            and self._verdict == other._verdict
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over every field."""
        return hash(
            (
                ShapedCredit,
                self._outcome,
                self._credit,
                self._length,
                self._value,
                self._verdict,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this payment."""
        return (
            f"ShapedCredit({self._outcome!r}, {self._credit!r}, {self._length!r}, "
            f"{self._value!r}, {self._verdict!r})"
        )


@final
class ShapedAbstention(FrozenValue):
    """One sampled form the prover reached no answer about.

    An abstention is uninformative rather than bad, so it is not a payment and
    holds no value. Whoever means over a batch leaves it out.

    Parameters
    ----------
    verdict : Abstain
        The stopping condition the prover named.
    """

    _noun = "shaped abstention"

    __slots__ = ("_verdict",)

    def __init__(self, verdict: Abstain) -> None:
        object.__setattr__(self, "_verdict", verdict)

    @property
    def verdict(self) -> Abstain:
        """Return the stopping condition the prover named."""
        return self._verdict

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another abstention names the same stopping condition."""
        if type(other) is not ShapedAbstention:
            return NotImplemented

        return self._verdict == other._verdict

    @override
    def __hash__(self) -> int:
        """Return a hash over the stopping condition."""
        return hash((ShapedAbstention, self._verdict))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this abstention."""
        return f"ShapedAbstention({self._verdict!r})"


type ShapedOutcome = ShapedCredit | ShapedAbstention
"""What one sampled form came back with: a payment, or no answer at all."""


@final
class ShapedProofReward[S]:
    """The shaped credit a sampled logical form earns against a reference form.

    A form is read in four steps, each of which can stop the reading. A form
    leaving a bound index open is ill-formed and stops there, since neither the
    type checker nor the prover accepts one. A closed form goes to the type
    checker, and one the checker leaves a constraint on stops at the
    well-formed credit. A form that type-checks with no reference to be judged
    against stops at the type checking credit. Otherwise the prover is asked
    whether the form and the reference entail each other, which pays the proved
    credit on a proof and leaves a countermodel at the type checking credit,
    the proof term having been earned by neither.

    An abstention is not a payment. It comes back as its own type, since a
    prover that ran out of clock said nothing about the form and scoring that
    as a zero would train against the prover's limit.

    Both directions of the proof go through the oracle. The forward direction
    settles the question unless it is proved; only then is the backward
    direction requested.

    Parameters
    ----------
    checker : TypingSystem[S]
        What decides whether a closed form type-checks.
    service : ProofService
        The shared proof service both directions of the equivalence go to.
    signature : S
        The lexical signature the checker reads the form's atoms under.
    rules : Mapping[OpName, TypingRule]
        The typing rule of each operator the form may be built from.
    shaping : ProofShaping
        What each outcome earns and what a node costs.
    budget : ProofBudget
        The use-site budget governing both proof calls.

    Raises
    ------
    TypeError
        If the service or budget has the wrong kind.
    """

    __slots__ = (
        "_budget",
        "_checker",
        "_rules",
        "_service",
        "_shaping",
        "_signature",
        "_site",
    )

    def __init__(
        self,
        checker: TypingSystem[S],
        service: ProofService,
        /,
        *,
        signature: S,
        rules: Mapping[OpName, TypingRule],
        shaping: ProofShaping,
        budget: ProofBudget,
        site: ProofSite,
    ) -> None:
        if not isinstance(service, ProofService):
            msg = f"a shaped proof reward needs a ProofService, not {service!r}"
            raise TypeError(msg)

        if not isinstance(budget, ProofBudget):
            msg = f"a shaped proof reward needs a ProofBudget, not {budget!r}"
            raise TypeError(msg)

        if not isinstance(site, ProofSite):
            msg = f"a shaped proof reward needs a ProofSite, not {site!r}"
            raise TypeError(msg)

        self._checker = checker
        self._service = service
        self._signature = signature
        self._rules = rules
        self._shaping = shaping
        self._budget = budget
        self._site = site

    @property
    def shaping(self) -> ProofShaping:
        """Return what each outcome earns and what a node costs."""
        return self._shaping

    @property
    def budget(self) -> ProofBudget:
        """Return the single use-site budget governing proof calls."""
        return self._budget

    def score(self, form: Term, /, *, reference: Term | None) -> ShapedOutcome:
        """Pay one sampled form against the form it is judged by.

        Parameters
        ----------
        form : Term
            The logical form the decoder sampled.
        reference : Term | None
            The form the proof is against, or None where the item has none, in
            which case the payment stops at the type checking credit.
        """
        if not form.is_closed:
            return self._paid(FormOutcome.ILL_FORMED, form, None)

        typing = self._checker.check(form, self._signature, self._rules)

        if not typing.is_well_typed:
            return self._paid(FormOutcome.WELL_FORMED, form, None)

        if reference is None:
            return self._paid(FormOutcome.TYPE_CHECKS, form, None)

        verdict = self._service.equivalent(
            form, reference, budget=self._budget, site=self._site
        )

        if isinstance(verdict, Abstain):
            return ShapedAbstention(verdict)

        if isinstance(verdict, Proved):
            return self._paid(FormOutcome.PROVED, form, verdict)

        return self._paid(FormOutcome.TYPE_CHECKS, form, verdict)

    def _paid(
        self,
        outcome: FormOutcome,
        form: Term,
        verdict: Proved | Refuted | Graded | None,
    ) -> ShapedCredit:
        """Return the payment one outcome makes on one form.

        Parameters
        ----------
        outcome : FormOutcome
            The class the form fell into.
        form : Term
            The form, whose node count the length penalty is charged on.
        verdict : Proved | Refuted | Graded | None
            The proof or the countermodel the outcome rests on, or None where
            the prover was never asked.
        """
        return ShapedCredit(
            outcome,
            self._shaping.credit(outcome),
            form.size,
            self._shaping.value(outcome, form.size),
            verdict,
        )

    def open_step(self) -> None:
        """Open this reward's next proof service budget window."""
        self._service.open_step(self._site)

    @override
    def __repr__(self) -> str:
        """Return the shaping, proof service and use-site budget."""
        return (
            f"ShapedProofReward(shaping={self._shaping!r}, "
            f"service={self._service!r}, budget={self._budget!r})"
        )


def _checked_items(name: str, batch: GoldFormItems) -> int:
    """Return how many items a batch holds, refusing misaligned gold forms.

    Parameters
    ----------
    name : str
        The objective reading the batch, for the message.
    batch : GoldFormItems
        The batch to count.

    Raises
    ------
    ValueError
        If the batch names another number of gold forms than surface items, or
        if it names no item at all.
    """
    items = len(batch.words)
    references = len(batch.gold_forms)

    if items != references:
        msg = (
            f"{name} reads one gold form per surface item, and the batch has "
            f"{items} items and {references} gold forms"
        )
        raise ValueError(msg)

    if items == 0:
        msg = f"{name} means over the items of a batch, and the batch names none"
        raise ValueError(msg)

    return items


@final
class ShapedProofObjective[S]:
    """Score a batch of sampled logical forms by the shaped proof reward.

    Each item names the readout its sampled forms were published under and the
    form they are judged against. Every member of the item's beam is paid, and
    the item's loss is the negative of what the beam expects to be paid, under
    the beam's own weights renormalized over the members the prover decided.
    An item whose every member abstained enters no mean and is reported as a
    rate instead. The value is the mean over the decided items and the count is
    their number.

    The payment is a function of the form alone, so the gradient reaches the
    beam weights and nothing else, which is what a score-function estimator
    over the decoder's samples needs.

    Parameters
    ----------
    reward : ShapedProofReward[S]
        What pays each form.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_name", "_readout", "_reward")

    def __init__(
        self,
        reward: ShapedProofReward[S],
        /,
        *,
        readout: str,
        name: str,
    ) -> None:
        self._reward = reward
        self._readout = ensure_name("the shaped-proof readout role", readout)
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"batch", "spans", "terms"})

    @property
    def shaping(self) -> ProofShaping:
        """Return the credits and the length weight the reward was declared with."""
        return self._reward.shaping

    def evaluate(self, outputs: PublishedOutputs[GoldFormItems], /) -> LossTerm:
        """Return the mean negative expected payment of the decided items.

        Parameters
        ----------
        outputs : PublishedOutputs[GoldFormItems]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the decided items, counted in them. The diagnostics
            report the fraction of items every member of which abstained as
            ``abstention_rate``.

        Raises
        ------
        ValueError
            If the batch's sides disagree about how many items there are, if
            the batch names no item, if an item names a readout nothing was
            published under, or if the step published no scored forms at all.
        EveryItemAbstainedError
            If the prover reached no answer about any member of any beam.
        """
        self._reward.open_step()
        batch = outputs.batch
        readout = outputs.terms.get(self._readout)

        if readout is None:
            offered = ", ".join(sorted(outputs.terms)) or "nothing"
            msg = (
                f"{self._name} reads decoded terms {self._readout!r}, and the step "
                f"published {offered}"
            )
            raise ValueError(msg)

        items = _checked_items(self._name, batch)
        sampled = [
            self._sampled(
                readout,
                item,
                outputs.spans.span_id(0, len(words)),
            )
            for item, words in enumerate(batch.words)
        ]
        like = sampled[0].log_weights
        losses: list[Tensor] = []

        for position, weighted in enumerate(sampled):
            expected = self._expected(weighted, batch.gold_forms[position])

            if expected is None:
                continue

            losses.append(-expected)

        decided = len(losses)
        diagnostics = {
            "abstention_rate": like.new_tensor((items - decided) / items),
        }

        if decided == 0:
            raise EveryItemAbstainedError(objective=self._name, items=items)

        return LossTerm(torch.stack(losses).mean(), decided, diagnostics=diagnostics)

    def _sampled(self, decoded: SpanTerms, item: int, span: int) -> WeightedTerms:
        """Return the weighted forms published at one root cell.

        Parameters
        ----------
        decoded : SpanTerms
            What the declared readout decoded by item and span.
        item : int
            Batch position.
        span : int
            Root span id.

        Raises
        ------
        ValueError
            If the readout decoded nothing at the root cell.
        """
        try:
            return decoded.at(item, span)

        except KeyError:
            msg = (
                f"{self._name} reads a decoded term beam at item {item}'s root, "
                "and the declared readout published none"
            )
            raise ValueError(msg) from None

    def _expected(
        self, weighted: WeightedTerms, reference: Term | None
    ) -> Tensor | None:
        """Return the expected payment, or None where every form abstained.

        Parameters
        ----------
        weighted : WeightedTerms
            The forms one readout sampled, with the decoder's weight of each.
        reference : Term | None
            The form the beam's members are judged against, or None where the
            item has none.

        Returns
        -------
        Tensor | None
            A scalar attached to the beam's weights, over the members the
            prover decided, or None where it decided none of them.
        """
        weights = weighted.log_weights
        decided: list[int] = []
        paid: list[float] = []

        for index, form in enumerate(weighted.terms):
            outcome = self._reward.score(form, reference=reference)

            if isinstance(outcome, ShapedAbstention):
                continue

            decided.append(index)
            paid.append(outcome.value)

        if not decided:
            return None

        kept = weights[decided]

        return ((kept - kept.logsumexp(0)).exp() * weights.new_tensor(paid)).sum()

    @override
    def __repr__(self) -> str:
        """Return the name and the reward the objective pays through."""
        return (
            f"ShapedProofObjective(name={self._name!r}, readout={self._readout!r}, "
            f"reward={self._reward!r})"
        )


def _build(
    settings: ShapedProofLossSettings,
    context: LossComponentBindings,
    /,
) -> ShapedProofObjective[LexicalSignature]:
    """Build proof-shaped supervision from declared language capabilities."""
    source = context.proof_reward()
    language = source.language
    shaping = settings.shaping
    reward = ShapedProofReward(
        language.checker,
        source.service,
        signature=language.lexicon,
        rules=language.rules,
        shaping=ProofShaping(
            ill_formed=shaping.ill_formed,
            well_formed=shaping.well_formed,
            type_checks=shaping.type_checks,
            proved=shaping.proved,
            length=shaping.length,
        ),
        budget=settings.budget,
        site=ProofSite(f"loss.{context.name}"),
    )

    return ShapedProofObjective(
        reward,
        readout=settings.readout,
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=ShapedProofLossSpec, build=_build)
"""The shaped proof-credit objective plugin."""

"""Evaluation of predicted logical forms against gold logical forms."""

from enum import StrEnum
from statistics import fmean
from typing import TYPE_CHECKING, NamedTuple, Never, Self, final, override

from lofi._checks import ensure_name
from lofi._types import ItemId
from lofi.terms import Normalization, TreeEditDistance, normalize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.terms import Calculus, OpName, Term, TermStore
    from lofi.typecheck._ports import TypingSystem
    from lofi.typecheck._values import TypingRule

__all__ = [
    "EvaluationItem",
    "EvaluationReport",
    "TypeAgreement",
    "evaluation_report",
]


class TypeAgreement(StrEnum):
    """Declared relation used to compare inferred semantic types."""

    EXACT = "exact"
    SUBSUMES = "subsumes"
    UNIFIES = "unifies"


class EvaluationItem(NamedTuple):
    """Gold and predicted logical forms compared for one dataset item.

    Attributes
    ----------
    item_id : ItemId
        Opaque stable dataset item id.
    gold : Term
        Normalized gold logical form.
    predicted : Term
        Normalized predicted logical form.
    gold_type : Term | None
        Principal type of the gold form, or None when it is ill-typed.
    predicted_type : Term | None
        Principal type of the predicted form, or None when it is ill-typed.
    agrees : bool
        Whether the inferred types stand in the report's declared relation.
    distance : int
        Ordered tree edit distance from the predicted form to the gold form.
    """

    item_id: ItemId
    gold: Term
    predicted: Term
    gold_type: Term | None
    predicted_type: Term | None
    agrees: bool
    distance: int

    @property
    def exact(self) -> bool:
        """Whether the two normal forms are the same interned term."""
        return self.gold is self.predicted


@final
class EvaluationReport:
    """Frozen per-item comparisons with their aggregate scores.

    Parameters
    ----------
    agreement : TypeAgreement
        Relation used to compare inferred semantic types.
    items : Sequence[EvaluationItem]
        Nonempty item comparisons in item id order.

    Raises
    ------
    TypeError
        If the agreement is not a declared relation.
    ValueError
        If no item is supplied.
    """

    __slots__ = ("_agreement", "_items")

    def __init__(
        self, agreement: TypeAgreement, items: Sequence[EvaluationItem]
    ) -> None:
        held = tuple(items)

        if not held:
            msg = "an evaluation report needs at least one item"
            raise ValueError(msg)

        if not isinstance(agreement, TypeAgreement):
            msg = f"an evaluation type agreement is not one: {agreement!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_agreement", agreement)
        object.__setattr__(self, "_items", held)

    @property
    def agreement(self) -> TypeAgreement:
        """Return the declared semantic type agreement relation."""
        return self._agreement

    @property
    def items(self) -> tuple[EvaluationItem, ...]:
        """Item comparisons in item id order."""
        return self._items

    @property
    def exact_match(self) -> float:
        """Fraction of items whose beta-normal forms are the same term."""
        return fmean(item.exact for item in self._items)

    @property
    def type_agreement(self) -> float:
        """Fraction of items whose types stand in the declared relation."""
        return fmean(item.agrees for item in self._items)

    @property
    def mean_edit_distance(self) -> float:
        """Mean tree edit distance from predicted to gold forms."""
        return fmean(item.distance for item in self._items)

    def __copy__(self) -> Self:
        """Preserve identity when copied."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Preserve identity when deep-copied."""
        del memo
        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an evaluation report is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an evaluation report is immutable"
        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return the item count and the three scores."""
        return (
            f"EvaluationReport(agreement={self._agreement.value!r}, "
            f"items={len(self._items)}, "
            f"exact_match={self.exact_match!r}, "
            f"type_agreement={self.type_agreement!r}, "
            f"mean_edit_distance={self.mean_edit_distance!r})"
        )


def _principal[S](
    term: Term,
    typing: TypingSystem[S],
    signature: S,
    rules: Mapping[OpName, TypingRule],
) -> Term | None:
    """Return a form's principal type, or None when it is ill-typed."""
    result = typing.check(term, signature, rules)
    scheme = result.scheme

    if scheme is None or not result.is_well_typed:
        return None

    return scheme.body


def _agrees[S](
    typing: TypingSystem[S],
    predicted: Term,
    inferred: Term,
    agreement: TypeAgreement,
    /,
) -> bool:
    """Return whether two types stand in the declared relation."""
    match agreement:
        case TypeAgreement.EXACT:
            return predicted is inferred
        case TypeAgreement.SUBSUMES:
            return typing.subsumes(predicted, inferred) is not None
        case TypeAgreement.UNIFIES:
            return typing.unifies(predicted, inferred) is not None


def evaluation_report[S](
    gold: Mapping[ItemId, Term],
    predicted: Mapping[ItemId, Term],
    typing: TypingSystem[S],
    signature: S,
    rules: Mapping[OpName, TypingRule],
    store: TermStore,
    calculus: Calculus,
    /,
    *,
    agreement: TypeAgreement,
    normalization: Normalization,
) -> EvaluationReport:
    """Compare predicted logical forms with gold logical forms item by item.

    Both forms of a pair are normalized in the supplied store, so two
    alpha-variant forms are the same interned term. Their inferred types are
    compared under the declared relation.

    Parameters
    ----------
    gold : Mapping[ItemId, Term]
        Gold logical form of each item.
    predicted : Mapping[ItemId, Term]
        Predicted logical form of each item, under the same ids.
    typing : TypingSystem[S]
        Type checker inferring principal types and matching them.
    signature : S
        Symbol environment used by the checker.
    rules : Mapping[OpName, TypingRule]
        Typing rules keyed by logical form operator.
    store : TermStore
        Store holding every form, in which the normal forms are interned.
    calculus : Calculus
        Abstraction and application operators of the forms.
    agreement : TypeAgreement
        Relation used to compare inferred semantic types.
    normalization : Normalization
        The declared normal form and reduction budget.

    Returns
    -------
    EvaluationReport
        One comparison per item, in item id order.

    Raises
    ------
    TypeError
        If an item id is not a string or the agreement is not declared.
    ValueError
        If no item is supplied, an item id is empty, or the gold and
        predicted ids differ.
    NonTerminatingError
        If a form has no normal form within the declared fuel.
    """
    checked_gold = {
        ItemId(ensure_name("an evaluation item id", item_id)): term
        for item_id, term in gold.items()
    }
    checked_predicted = {
        ItemId(ensure_name("an evaluation item id", item_id)): term
        for item_id, term in predicted.items()
    }
    if not checked_gold:
        msg = "an evaluation needs at least one gold item"
        raise ValueError(msg)

    if checked_gold.keys() != checked_predicted.keys():
        missing = tuple(sorted(checked_gold.keys() - checked_predicted.keys()))
        extra = tuple(sorted(checked_predicted.keys() - checked_gold.keys()))
        msg = (
            "predicted item ids differ from gold item ids; "
            f"missing={missing!r}, extra={extra!r}"
        )
        raise ValueError(msg)

    distance: TreeEditDistance = TreeEditDistance()
    items: list[EvaluationItem] = []

    for item_id in sorted(checked_gold):
        gold_form = normalize(
            store,
            checked_gold[item_id],
            calculus,
            normalization=normalization,
        )
        predicted_form = normalize(
            store,
            checked_predicted[item_id],
            calculus,
            normalization=normalization,
        )
        gold_type = _principal(gold_form, typing, signature, rules)
        predicted_type = _principal(predicted_form, typing, signature, rules)
        agrees = (
            gold_type is not None
            and predicted_type is not None
            and _agrees(typing, predicted_type, gold_type, agreement)
        )
        items.append(
            EvaluationItem(
                item_id,
                gold_form,
                predicted_form,
                gold_type,
                predicted_type,
                agrees,
                distance(predicted_form, gold_form),
            )
        )

    return EvaluationReport(agreement, items)

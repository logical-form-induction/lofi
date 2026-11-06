"""Linearization of constraint targets through a term codec."""

from typing import TYPE_CHECKING, NamedTuple

from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Hashable

    from lofi.nn.loss._constraint import Constraint, LogicalFormConstraint
    from lofi.terms import BatchEncoding, Term, TermCodec

__all__ = [
    "LinearizedConstraint",
    "LinearizedTargets",
    "linearize",
    "linearize_targets",
]


class LinearizedTargets(NamedTuple):
    """Integer channels for the categories of one target set.

    Attributes
    ----------
    encoded : BatchEncoding
        One encoding per category, in the target set's declaration order.
    pad : int
        The code the operator index reserves for a slot holding no node.
    matching : str
        The name of the matching each category was reduced under.
    targets : TargetSet[Term]
        The reduced symbolic target set retained by its readout owner.
    """

    encoded: BatchEncoding
    pad: int
    matching: str
    targets: TargetSet[Term]


class LinearizedConstraint(NamedTuple):
    """Integer channels for one constraint's categories, with its direction.

    Attributes
    ----------
    targets : LinearizedTargets
        The linearized categories.
    negate : bool
        Whether the categories are excluded rather than permitted.
    """

    targets: LinearizedTargets
    negate: bool


def linearize_targets(
    targets: TargetSet[Term], codec: TermCodec, /
) -> LinearizedTargets:
    """Encode a target set's categories through one codec.

    Each category is reduced to its matching's representative before it is encoded.
    The codes come from the codec's operator index and the slot order from its layout.

    Parameters
    ----------
    targets : TargetSet[Term]
        The categories and the matching they are compared under.
    codec : TermCodec
        The codec that assigns the codes and fixes the slot order.

    Raises
    ------
    EncodingOverflowError
        If a category has more nodes than the layout holds.
    UnknownOperatorError
        If a category names an operator the index does not declare.
    """
    matching = targets.matching
    reduced = tuple(matching.key(category) for category in targets.categories)

    return LinearizedTargets(
        codec.encode_batch(reduced),
        codec.index.PAD,
        matching.name,
        TargetSet(reduced, matching),
    )


def linearize[ScopeT: Hashable](
    constraint: Constraint[Term, ScopeT] | LogicalFormConstraint[ScopeT],
    codec: TermCodec,
    /,
) -> LinearizedConstraint:
    """Encode a constraint's targets through one codec.

    Parameters
    ----------
    constraint : Constraint[Term, ScopeT] | LogicalFormConstraint[ScopeT]
        The constraint whose targets are encoded, a permitted category set or one
        required form. Nothing is written to it.
    codec : TermCodec
        The codec that assigns the codes and fixes the slot order.

    Raises
    ------
    EncodingOverflowError
        If a category has more nodes than the layout holds.
    UnknownOperatorError
        If a category names an operator the index does not declare.
    """
    return LinearizedConstraint(
        linearize_targets(constraint.targets, codec), constraint.negate
    )

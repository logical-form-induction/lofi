"""High-level variables read off one gold derivation.

The type at a cell is read at every cell the derivation concludes an item on. The
split at a cell and the combinator at that split are read only where the normal-form
tags on the derivation's items show that the derivation was filtered by a normal form
with more than one tag, since a derivation no normal form constrained leaves both
undetermined under spurious ambiguity. The unconstrained arm reads all three
regardless, and the arm is named in the provenance.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_name
from lofi.probe._causal import CombinatorAt, SplitAt, TypeAt
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.deduce import Derivation
    from lofi.judge import NormalForm
    from lofi.probe._causal import CellAssignment, HighLevelVariable

__all__ = ["ARMS", "Arm", "Extraction", "extract_variables"]

type Arm = Literal["normal-form", "unconstrained"]
"""Specifies whether a split and a combinator need a constrained derivation."""

ARMS: Final[tuple[Arm, ...]] = ("normal-form", "unconstrained")
"""Extraction arms in report order."""


class Extraction:
    """Holds the high-level variables one derivation decided.

    Parameters
    ----------
    assignment : CellAssignment
        What the derivation decided, variable by variable.
    arm : Arm
        Which arm read the assignment.
    constrained : bool
        Whether the derivation's tags showed it was filtered by a normal form.
    provenance : Provenance
        Which arm read the assignment, over what data.
    """

    __slots__ = ("_arm", "_assignment", "_constrained", "_provenance")

    def __init__(
        self,
        assignment: CellAssignment,
        arm: Arm,
        *,
        constrained: bool,
        provenance: Provenance,
    ) -> None:
        self._assignment: CellAssignment = MappingProxyType(dict(assignment))
        self._arm: Arm = arm
        self._constrained = constrained
        self._provenance = provenance

    @property
    def assignment(self) -> CellAssignment:
        """Return what the derivation decided, variable by variable."""
        return self._assignment

    @property
    def arm(self) -> Arm:
        """Return which arm read the assignment."""
        return self._arm

    @property
    def constrained(self) -> bool:
        """Return whether the derivation's tags showed a normal form filtered it."""
        return self._constrained

    @property
    def provenance(self) -> Provenance:
        """Return which arm read the assignment, over what data."""
        return self._provenance

    @override
    def __repr__(self) -> str:
        """Return the arm, the outcome, the variable count and the provenance."""
        return (
            f"Extraction(arm={self._arm!r}, constrained={self._constrained!r}, "
            f"variables={len(self._assignment)}, provenance={self._provenance!r})"
        )


def extract_variables[T, W](
    derivation: Derivation[T, W],
    normal_form: NormalForm[T],
    *,
    render: Callable[[T], str],
    dataset: str,
    arm: Arm = "normal-form",
) -> Extraction:
    """Read the type, split and combinator variables off one gold derivation.

    Parameters
    ----------
    derivation : Derivation[T, W]
        The gold derivation, whose item keys hold the tags the normal form assigned.
    normal_form : NormalForm[T]
        The filter the derivation was parsed under, whose alphabet the tags are read
        against.
    render : Callable[[T], str]
        How a concluded term is written as the decision at its cell.
    dataset : str
        The handle of the data the derivation came from.
    arm : Arm
        Whether the split and the combinator are read only from a derivation the
        tags show to be constrained, or regardless.

    Returns
    -------
    Extraction
        The type at every cell, plus the split at every binary cell and the
        combinator at that split where the arm and the tags allow them.

    Raises
    ------
    ValueError
        If the arm is not a declared one, if an item's tag is neither empty nor in
        the normal form's alphabet, if a rendered category or a rule name is empty,
        or if the dataset handle is empty.
    """
    if arm not in ARMS:
        msg = f"no extraction arm is called {arm!r}, only {ARMS}"
        raise ValueError(msg)

    alphabet = normal_form.alphabet
    constrained = len(alphabet) > 1
    types: dict[HighLevelVariable, str | int] = {}
    structure: dict[HighLevelVariable, str | int] = {}
    pending = [derivation]

    while pending:
        node = pending.pop()
        tag = node.key.nf_tag

        if tag and tag not in alphabet:
            msg = (
                f"an item at {node.key.span!r} is tagged {tag!r}, which normal form "
                f"{normal_form.name!r} never assigns"
            )
            raise ValueError(msg)

        if not tag:
            constrained = False

        cell = node.key.span
        at = TypeAt(cell)

        if at not in types:
            types[at] = ensure_name(
                f"the category rendered at {cell!r}", render(node.edge.output)
            )

        split = node.edge.split

        if split is not None:
            structure[SplitAt(cell)] = split
            structure[CombinatorAt(cell, split)] = ensure_name(
                f"the rule name at {cell!r}", node.edge.rule_name
            )

        pending.extend(node.children)

    decided = {**types, **structure} if constrained or arm == "unconstrained" else types

    return Extraction(
        decided,
        arm,
        constrained=constrained,
        provenance=Provenance(f"variables:{arm}", dataset, None),
    )

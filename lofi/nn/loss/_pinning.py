"""Pin constrained spans to published target masses."""

from typing import TYPE_CHECKING, Final, NamedTuple, override

import torch

from lofi._checks import ensure_name
from lofi._types import ReadoutId
from lofi.nn.loss._constraint import (
    BatchLocalConstraint,
    BatchLocalNegativeBracket,
)
from lofi.nn.loss._errors import ImpossibleNegativeBracketError, ImproperMassError
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import MASS_AXES, log_complement, log_mass
from lofi.registry import ComponentPlugin
from lofi.spec.losses.pinning import PinningLossSettings, PinningLossSpec

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from torch import Tensor

    from lofi.nn.loss._ports import (
        ConstrainedItems,
        LossComponentBindings,
        PublishedOutputs,
    )
    from lofi.nn.loss._values import SpanReadout
    from lofi.terms import Term

__all__ = ["SpanConstraintObjective"]


def _declared(readouts: Iterable[ReadoutId]) -> tuple[ReadoutId, ...]:
    """Validate declared mass readouts.

    Parameters
    ----------
    readouts : Iterable[ReadoutId]
        Published target mass identifiers.

    Raises
    ------
    TypeError
        If an id is not a string.
    ValueError
        If an id is empty or repeated.
    """
    declared: list[ReadoutId] = []
    seen: set[ReadoutId] = set()

    for key in readouts:
        checked = ReadoutId(ensure_name("a readout id", key))

        if checked in seen:
            msg = f"the mass readout {checked!r} is declared more than once"
            raise ValueError(msg)

        seen.add(checked)
        declared.append(checked)

    return tuple(declared)


class _Instance(NamedTuple):
    """One constrained cell of one item, and which mass it is scored on."""

    item: int
    span: tuple[int, int]
    negate: bool
    weight: float


class SpanConstraintObjective[T, ScopeT: Hashable]:
    """Pin spans to target mass or exclude them from the constituent chart.

    Parameters
    ----------
    readouts : Iterable[ReadoutId]
        Published target mass identifiers accepted from localized constraints.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If an id or the name is not a string.
    ValueError
        If an id or the name is empty, or an id is repeated.
    """

    __slots__ = ("_name", "_readouts")

    def __init__(
        self,
        *,
        readouts: Iterable[ReadoutId],
        name: str,
    ) -> None:
        self._readouts = _declared(readouts)
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset(
            {"batch", "readouts"} if self._readouts else {"batch", "marginals"}
        )

    @property
    def readouts(self) -> tuple[ReadoutId, ...]:
        """Return the declared target mass identifiers."""
        return self._readouts

    def _published(
        self, readouts: Mapping[ReadoutId, SpanReadout]
    ) -> Mapping[ReadoutId, SpanReadout]:
        """Return the mass pair published under every declared id.

        Parameters
        ----------
        readouts : Mapping[ReadoutId, SpanReadout]
            What every readout published, keyed by id.

        Raises
        ------
        ValueError
            If nothing was published under a declared id, or if what was published
            there is not a mass pair.
        """
        published: dict[ReadoutId, SpanReadout] = {}

        for key in self._readouts:
            pair = readouts.get(key)

            if pair is None:
                offered = ", ".join(sorted(readouts)) or "nothing"
                msg = (
                    f"{self._name} reads the mass {key!r}, and the step published "
                    f"{offered}"
                )
                raise ValueError(msg)

            if pair.axes != MASS_AXES:
                msg = (
                    f"{self._name} reads a mass pair over {MASS_AXES}, and {key!r} "
                    f"is over {pair.axes}"
                )
                raise ValueError(msg)

            published[key] = pair

        return published

    def _instances(
        self, constraints: Iterable[BatchLocalConstraint[T, ScopeT]]
    ) -> dict[ReadoutId, list[_Instance]]:
        """Return the batch's constraint instances, grouped by the mass they read.

        Parameters
        ----------
        constraints : Iterable[BatchLocalConstraint[T, ScopeT]]
            Target-bearing constraints localized to this batch.

        Returns
        -------
        dict[ReadoutId, list[_Instance]]
            The instances under each declared id, every id present, in the order the
            batch lists its constraints and each constraint lists its items and then
            its spans.

        Raises
        ------
        ValueError
            If a constraint names an undeclared readout.
        """
        grouped: dict[ReadoutId, list[_Instance]] = {key: [] for key in self._readouts}

        for constraint in constraints:
            instances = grouped.get(constraint.readout)

            if instances is None:
                declared = ", ".join(sorted(self._readouts))
                msg = (
                    f"{self._name} is handed a constraint reading "
                    f"{constraint.readout!r}, which is not declared; "
                    f"declared: {declared}"
                )
                raise ValueError(msg)

            instances.extend(
                _Instance(
                    item,
                    span,
                    constraint.negate,
                    1.0 if constraint.penalty is None else constraint.penalty,
                )
                for item in constraint.item_positions
                for span in constraint.spans
            )

        return grouped

    @staticmethod
    def _bracket_instances(
        constraints: Iterable[BatchLocalNegativeBracket[ScopeT]],
    ) -> list[tuple[int, int, int]]:
        """Return negative bracket cells in batch and source order."""
        return [
            (item, *span)
            for constraint in constraints
            for item in constraint.item_positions
            for span in constraint.spans
        ]

    @staticmethod
    def _read(pair: SpanReadout, instances: list[_Instance]) -> Tensor:
        """Return the log mass each instance is scored on, in one gather.

        Parameters
        ----------
        pair : SpanReadout
            The mass pair published for the instances' target set.
        instances : list[_Instance]
            The cells to read, and whether each is scored on the mass off the set.

        Returns
        -------
        Tensor
            One log mass per instance, attached to the published tensor's graph.

        Raises
        ------
        ValueError
            If a span is empty or reaches outside the input.
        IndexError
            If an instance names an item the batch does not hold.
        UncomputedCellError
            If the readout never computed one of the cells.
        """
        cells = pair.gather([(held.item, *held.span) for held in instances])
        negated = torch.tensor(
            [held.negate for held in instances], device=cells.tensor.device
        )

        return torch.where(negated, log_complement(cells), log_mass(cells))

    def evaluate(
        self, outputs: PublishedOutputs[ConstrainedItems[T, ScopeT]], /
    ) -> LossTerm:
        """Return the mean negative log mass over the batch's constraint instances.

        Parameters
        ----------
        outputs : PublishedOutputs[ConstrainedItems[T, ScopeT]]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the instances, counted in them, reporting the mean
            probability that an instance is satisfied as ``satisfied``.

        Raises
        ------
        ValueError
            If a declared id was not published or is not a mass pair, if a constraint
            names an undeclared id, or if a constrained span
            reaches outside the input.
        IndexError
            If a constraint names an item the batch does not hold.
        UncomputedCellError
            If a constrained cell was never computed.
        ImproperMassError
            If a published log mass exceeds zero or is not a number.
        ImpossibleNegativeBracketError
            If a negative bracket excludes a span whose constituent probability is
            exactly one.
        """
        targets: list[BatchLocalConstraint[T, ScopeT]] = []
        brackets: list[BatchLocalNegativeBracket[ScopeT]] = []

        for constraint in outputs.batch.constraints:
            if isinstance(constraint, BatchLocalConstraint):
                targets.append(constraint)
            elif isinstance(constraint, BatchLocalNegativeBracket):
                brackets.append(constraint)
            else:
                msg = f"{self._name} was handed an unknown constraint {constraint!r}"
                raise TypeError(msg)

        if targets and not self._readouts:
            msg = (
                f"{self._name} has no readout and cannot score a target-bearing "
                "constraint"
            )
            raise ValueError(msg)

        if brackets and self._readouts:
            msg = (
                f"{self._name} reads target mass and cannot score a negative bracket; "
                "declare a pinning objective with no readout"
            )
            raise ValueError(msg)

        published = self._published(outputs.readouts) if self._readouts else {}
        grouped = self._instances(targets)
        keyed = [(key, held) for key, held in grouped.items() if held]

        if self._readouts:
            like = next(iter(published.values())).distribution.tensor
        else:
            marginals = outputs.marginals

            if marginals is None:
                msg = f"{self._name} excludes brackets but no marginals were published"
                raise ValueError(msg)

            like = marginals.constituent.tensor

        if not keyed and not brackets:
            return LossTerm.inapplicable(like)

        target_read = [self._read(published[key], held) for key, held in keyed]
        target_weights = [instance.weight for _, group in keyed for instance in group]
        bracket_cells = self._bracket_instances(brackets)

        if bracket_cells:
            marginals = outputs.marginals

            if marginals is None:
                msg = f"{self._name} excludes brackets but no marginals were published"
                raise ValueError(msg)

            bracket_read = marginals.log_nonconstituent(bracket_cells)
            certain = bracket_read == -torch.inf

            if bool(certain.any()):
                item, i, j = bracket_cells[int(certain.nonzero()[0])]
                raise ImpossibleNegativeBracketError(
                    objective=self._name, item=item, span=(i, j)
                )

            read = torch.cat((*target_read, bracket_read))
            weights = read.new_tensor((*target_weights, *(1.0 for _ in bracket_cells)))
        else:
            read = torch.cat(target_read)
            weights = read.new_tensor(target_weights)

        proper = read <= 0.0

        if not bool(proper.all()):
            offending = int(proper.logical_not().nonzero()[0])
            target_instances = [(key, held) for key, group in keyed for held in group]

            if offending >= len(target_instances):
                item, i, j = bracket_cells[offending - len(target_instances)]
                msg = (
                    f"{self._name} read an invalid negative-bracket probability at "
                    f"span [{i}, {j}) of item {item}: {float(read[offending])!r}"
                )
                raise ValueError(msg)

            key, instance = target_instances[offending]
            raise ImproperMassError(
                readout=key,
                item=instance.item,
                span=instance.span,
                value=float(read[offending]),
            )

        return LossTerm(
            -(read * weights).mean(),
            int(read.shape[0]),
            diagnostics={
                "gradient_scale": weights.detach().mean(),
                "satisfied": read.detach().exp().mean(),
            },
        )

    @override
    def __repr__(self) -> str:
        """Return the name and the declared ids."""
        return (
            f"SpanConstraintObjective(name={self._name!r}, readouts={self._readouts!r})"
        )


def _build(
    settings: PinningLossSettings, context: LossComponentBindings, /
) -> SpanConstraintObjective[Term, Hashable]:
    """Build span pinning against the target sets of the declared term readout.

    Parameters
    ----------
    settings : PinningLossSettings
        The term-readout role.
    context : LossComponentBindings
        The declared name and the built components the role names.

    Raises
    ------
    TypeError
        If the role holds a component of another family.
    ValueError
        If the role is not built.
    """
    if settings.readout is None:
        return SpanConstraintObjective(readouts=(), name=context.name)

    readout = context.terms_readout(settings.readout)

    if not readout.targets:
        msg = "a target-pinning objective reads at least one mass readout"
        raise ValueError(msg)

    return SpanConstraintObjective(readouts=readout.targets, name=context.name)


PLUGIN: Final = ComponentPlugin(spec=PinningLossSpec, build=_build)
"""The span-pinning objective's plugin."""

"""Weighted collections of training objectives."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Self, override

from lofi._checks import CHECKS, ensure_finite
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._ports import BoundObjective, LossTerm, Objective, PublishedOutputs
from lofi.nn.loss._values import OUTPUT_NAMES, RecordingOutputs

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = ["ObjectiveMeans", "ObjectiveSuite", "SuiteResult"]

type _Entry[ContextT, BatchT] = (
    Objective[PublishedOutputs[BatchT]]
    | BoundObjective[ContextT, PublishedOutputs[BatchT]]
)


def _checked_weight(name: str, weight: float) -> float:
    """Return a weight.

    Parameters
    ----------
    name : str
        The objective the weight is attached to, for the message.
    weight : float
        The weight to check.

    Returns
    -------
    float
        The weight, widened to float.

    Raises
    ------
    TypeError
        If the weight is not a number.
    ValueError
        If the weight is not finite or is negative.
    """
    checked = ensure_finite(f"the weight on {name}", weight)

    if checked < 0.0:
        msg = f"the weight on {name} is negative: {weight}"
        raise ValueError(msg)

    return checked


class SuiteResult:
    """Stores a suite's result for one training step.

    Parameters
    ----------
    total : Tensor
        The weighted sum of the terms that applied, as a scalar tensor.
    terms : Mapping[str, LossTerm]
        Each objective's unweighted term, keyed by the objective's name.
    diagnostics : Mapping[str, Tensor]
        Every scalar the step reported, keyed by the objective's name and the quantity,
        joined by a slash.
    """

    __slots__ = ("_diagnostics", "_terms", "_total")

    def __init__(
        self,
        total: Tensor,
        terms: Mapping[str, LossTerm],
        diagnostics: Mapping[str, Tensor],
    ) -> None:
        self._total = total
        self._terms: Mapping[str, LossTerm] = MappingProxyType(dict(terms))
        self._diagnostics: Mapping[str, Tensor] = MappingProxyType(dict(diagnostics))

    @property
    def total(self) -> Tensor:
        """Return the weighted sum of the terms that applied, as a scalar tensor."""
        return self._total

    @property
    def terms(self) -> Mapping[str, LossTerm]:
        """Return each objective's unweighted term, keyed by the objective's name."""
        return self._terms

    @property
    def diagnostics(self) -> Mapping[str, Tensor]:
        """Return every scalar the step reported, namespaced by the objective."""
        return self._diagnostics

    @property
    def applied(self) -> frozenset[str]:
        """Return the objectives whose term applied to this step."""
        return frozenset(name for name, term in self._terms.items() if term.count > 0)

    @override
    def __repr__(self) -> str:
        """Return the applied and evaluated term counts."""
        return (
            f"SuiteResult(applied={len(self.applied)} of {len(self._terms)}, "
            f"diagnostics={len(self._diagnostics)})"
        )


class ObjectiveSuite[BatchT]:
    """Computes a weighted collection of objectives.

    Parameters
    ----------
    entries : Sequence[tuple[float, Objective[PublishedOutputs[BatchT]]]]
        The weight and objective of each term, in the order they are reported.
    published : frozenset[str] | None
        The names the configured graph publishes, or None to accept any objective whose
        declared names are in the published vocabulary.

    Raises
    ------
    TypeError
        If an entry is not an objective.
    ValueError
        If the suite has no objectives, if two of them share a name, if a weight is
        negative or not finite, or if one declares a name outside the published
        vocabulary.
    MissingOutputError
        If an objective declares a name the configured graph does not publish.
    """

    __slots__ = ("_entries",)

    def __init__(
        self,
        entries: Sequence[tuple[float, Objective[PublishedOutputs[BatchT]]]],
        *,
        published: frozenset[str] | None = None,
    ) -> None:
        ordered = tuple(
            (_checked_weight(objective.name, weight), objective)
            for weight, objective in _checked_objectives(entries)
        )

        _reject_repeated_names(ordered)

        for _, objective in ordered:
            _reject_unreadable(objective, published)

        self._entries = ordered

    @classmethod
    def bound[ContextT](
        cls,
        entries: Sequence[tuple[float, _Entry[ContextT, BatchT]]],
        context: ContextT,
        *,
        published: frozenset[str] | None = None,
    ) -> Self:
        """Return the suite these entries make once each has been bound.

        Parameters
        ----------
        entries : Sequence[tuple[float, _Entry[ContextT, BatchT]]]
            The weight and objective of each term, bound or not.
        context : ContextT
            The built component context for the unbound objective.
        published : frozenset[str] | None
            The names the configured graph publishes, or None to accept any objective
            whose declared names are in the published vocabulary.

        Returns
        -------
        Self
            The suite of the bound objectives, in the order given.

        Raises
        ------
        TypeError
            If an entry is neither an objective nor a bound objective, or if binding one
            returned something that is not an objective.
        """
        return cls(
            [(weight, _bind(entry, context)) for weight, entry in entries],
            published=published,
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Return the objectives' names, in the order they are reported."""
        return tuple(objective.name for _, objective in self._entries)

    @property
    def weights(self) -> tuple[float, ...]:
        """Return the weight on each objective, in the same order."""
        return tuple(weight for weight, _ in self._entries)

    @property
    def requires(self) -> frozenset[str]:
        """Return every published name the suite reads, which the graph must publish."""
        return frozenset(
            name for _, objective in self._entries for name in objective.requires
        )

    def __len__(self) -> int:
        """Return the number of objectives."""
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[float, Objective[PublishedOutputs[BatchT]]]]:
        """Yield each weight and objective, in the order they are reported."""
        return iter(self._entries)

    def evaluate(self, outputs: ObjectiveOutputs[BatchT]) -> SuiteResult:
        """Return the weighted total, every term and the merged diagnostics.

        Parameters
        ----------
        outputs : ObjectiveOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        SuiteResult
            The weighted total, the unweighted term of each objective, and the
            diagnostics of all of them under one namespaced mapping.
        """
        terms: dict[str, LossTerm] = {}
        diagnostics: dict[str, Tensor] = {}
        contributions: list[Tensor] = []

        for weight, objective in self._entries:
            term = objective.evaluate(_view(outputs, objective))
            terms[objective.name] = term
            _report(diagnostics, objective.name, term)

            if term.count > 0:
                contributions.append(weight * term.value)

        return SuiteResult(_summed(contributions, terms), terms, diagnostics)

    @override
    def __repr__(self) -> str:
        """Return each objective's weight and name, in order."""
        weighted = ", ".join(
            f"{weight:g} * {objective.name}" for weight, objective in self._entries
        )

        return f"ObjectiveSuite({weighted})"


class ObjectiveMeans:
    """Tracks each objective's mean over applicable steps."""

    __slots__ = ("_counts", "_steps", "_sums", "_total")

    def __init__(self) -> None:
        self._sums: dict[str, Tensor] = {}
        self._counts: dict[str, int] = {}
        self._total: Tensor | None = None
        self._steps = 0

    def update(self, result: SuiteResult) -> None:
        """Fold one step's result into the running means.

        Parameters
        ----------
        result : SuiteResult
            What the suite produced for the step.
        """
        detached = result.total.detach()
        self._total = detached if self._total is None else self._total + detached
        self._steps += 1

        for name, term in result.terms.items():
            if term.count == 0:
                continue

            weighted = term.value.detach() * term.count
            held = self._sums.get(name)
            self._sums[name] = weighted if held is None else held + weighted
            self._counts[name] = self._counts.get(name, 0) + term.count

    def means(self) -> Mapping[str, Tensor]:
        """Return each objective's mean so far, over the items it scored.

        Returns
        -------
        Mapping[str, Tensor]
            One scalar tensor per objective that has applied at least once.
        """
        return MappingProxyType(
            {name: total / self._counts[name] for name, total in self._sums.items()}
        )

    @property
    def total(self) -> Tensor | None:
        """Return the mean weighted total, or None before any step is folded in."""
        if self._total is None:
            return None

        return self._total / self._steps

    @property
    def steps(self) -> int:
        """Return the number of accumulated steps."""
        return self._steps

    def counts(self) -> Mapping[str, int]:
        """Return each objective's contributing item count.

        Returns
        -------
        Mapping[str, int]
            One count per objective that has applied at least once.
        """
        return MappingProxyType(dict(self._counts))

    def reset(self) -> None:
        """Clear the accumulated steps at the start of an epoch."""
        self._sums.clear()
        self._counts.clear()
        self._total = None
        self._steps = 0

    @override
    def __repr__(self) -> str:
        """Return the step and objective counts."""
        return f"ObjectiveMeans(steps={self._steps}, objectives={len(self._sums)})"


def _checked_objectives[BatchT](
    entries: Sequence[tuple[float, Objective[PublishedOutputs[BatchT]]]],
) -> tuple[tuple[float, Objective[PublishedOutputs[BatchT]]], ...]:
    """Return the entries.

    Parameters
    ----------
    entries : Sequence[tuple[float, Objective[PublishedOutputs[BatchT]]]]
        The weight and objective of each term.

    Returns
    -------
    tuple[tuple[float, Objective[PublishedOutputs[BatchT]]], ...]
        The entries, unchanged.

    Raises
    ------
    TypeError
        If an entry is not an objective.
    ValueError
        If there are no entries.
    """
    ordered = tuple(entries)

    if not ordered:
        msg = "a suite with no objectives trains nothing"
        raise ValueError(msg)

    for _, objective in ordered:
        if not isinstance(objective, Objective):
            msg = f"{objective!r} is not an objective, so it cannot reach a step"
            raise TypeError(msg)

    return ordered


def _reject_repeated_names[BatchT](
    entries: tuple[tuple[float, Objective[PublishedOutputs[BatchT]]], ...],
) -> None:
    """Raise if two objectives share the name their reports are namespaced by.

    Parameters
    ----------
    entries : tuple[tuple[float, Objective[PublishedOutputs[BatchT]]], ...]
        The weight and objective of each term.

    Raises
    ------
    ValueError
        If two objectives have the same name.
    """
    seen: set[str] = set()

    for _, objective in entries:
        if objective.name in seen:
            msg = f"two objectives are named {objective.name!r}"
            raise ValueError(msg)

        seen.add(objective.name)


def _reject_unreadable[BatchT](
    objective: Objective[PublishedOutputs[BatchT]],
    published: frozenset[str] | None,
) -> None:
    """Raise unless the graph publishes everything one objective declared.

    Parameters
    ----------
    objective : Objective[PublishedOutputs[BatchT]]
        The objective to check.
    published : frozenset[str] | None
        The names the configured graph publishes, or None to check only that the
        declared names exist.

    Raises
    ------
    ValueError
        If the objective declares a name outside the published vocabulary.
    MissingOutputError
        If it declares a name the configured graph does not publish.
    """
    unknown = objective.requires - OUTPUT_NAMES

    if unknown:
        wanted = ", ".join(sorted(unknown))
        offered = ", ".join(sorted(OUTPUT_NAMES))
        msg = f"{objective.name} reads {wanted}, and a step publishes {offered}"
        raise ValueError(msg)

    if published is None:
        return

    missing = objective.requires - published

    if missing:
        raise MissingOutputError(
            objective=objective.name,
            missing=missing,
            published=published,
        )


def _bind[ContextT, BatchT](
    entry: _Entry[ContextT, BatchT],
    context: ContextT,
) -> Objective[PublishedOutputs[BatchT]]:
    """Return one entry as an objective, binding it against the context first.

    Parameters
    ----------
    entry : _Entry[ContextT, BatchT]
        An objective, or something that becomes one when it is handed the context.
    context : ContextT
        The built component context for the unbound objective.

    Returns
    -------
    Objective[PublishedOutputs[BatchT]]
        The entry itself where it was already an objective, and what binding returned
        otherwise.

    Raises
    ------
    TypeError
        If the entry is neither, or if binding returned something that is still not an
        objective.
    """
    if isinstance(entry, Objective):
        return entry

    bound = entry.bind(context)

    if not isinstance(bound, Objective):
        msg = f"binding {entry.name} returned {bound!r}, which is not an objective"
        raise TypeError(msg)

    return bound


def _view[BatchT](
    outputs: ObjectiveOutputs[BatchT],
    objective: Objective[PublishedOutputs[BatchT]],
) -> PublishedOutputs[BatchT]:
    """Return a restricted view of the step outputs.

    Parameters
    ----------
    outputs : ObjectiveOutputs[BatchT]
        What the graph published for the step.
    objective : Objective[PublishedOutputs[BatchT]]
        The objective about to read them.

    Returns
    -------
    PublishedOutputs[BatchT]
        The bundle itself, or a proxy admitting what the objective declared where the
        debug checks are on.
    """
    if not CHECKS:
        return outputs

    return RecordingOutputs(
        outputs,
        objective=objective.name,
        requires=objective.requires,
    )


def _report(diagnostics: dict[str, Tensor], name: str, term: LossTerm) -> None:
    """Add one term's reportable scalars to the merged mapping.

    Parameters
    ----------
    diagnostics : dict[str, Tensor]
        The mapping being merged into.
    name : str
        The objective's name, which every key is namespaced by.
    term : LossTerm
        The term the objective returned.

    Raises
    ------
    ValueError
        When debug checks are enabled and a diagnostic name collides with the term value
        or residual.
    """
    diagnostics[f"{name}/loss"] = term.value

    if term.residual is not None:
        diagnostics[f"{name}/residual"] = term.residual

    for key, value in term.diagnostics.items():
        merged = f"{name}/{key}"

        if CHECKS and merged in diagnostics:
            msg = f"the diagnostic {key!r} of {name} lands on the term's own {key}"
            raise ValueError(msg)

        diagnostics[merged] = value


def _summed(contributions: list[Tensor], terms: Mapping[str, LossTerm]) -> Tensor:
    """Return the summed contributions, or a graph-connected zero if none apply.

    Parameters
    ----------
    contributions : list[Tensor]
        The weighted value of each term that applied.
    terms : Mapping[str, LossTerm]
        Every term of the step, applied or not.

    Returns
    -------
    Tensor
        A scalar.
    """
    if not contributions:
        first = next(iter(terms.values()))

        return LossTerm.inapplicable(first.value).value

    total = contributions[0]

    for contribution in contributions[1:]:
        total = total + contribution

    return total

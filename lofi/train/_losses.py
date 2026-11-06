"""Weighted multi-term loss assembly and its telemetry keys."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
    refuse_reassignment,
)
from lofi._frozen import FrozenValue, Unpicklable
from lofi.lf.provers import ProofService
from lofi.nn.loss._ports import (
    LossComponentBindings,
    Objective,
    ProofRewardBindings,
)
from lofi.nn.loss._readout import TermsReadout
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.spaces import Space, SpaceRelationBinding
from lofi.spec.language import LanguageAssembly

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.loss._ports import (
        LossTerm,
        Objective,
        PublishedOutputs,
        SpanTokenDecoder,
    )
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.registry import ComponentPlugin

__all__ = [
    "InactiveLossError",
    "LossAssembly",
    "LossBindingContext",
    "LossPlugin",
    "LossResult",
    "MisnamedObjectiveError",
    "ProofRewardContext",
    "WeightSchedule",
    "WeightedObjective",
    "WeightedTerm",
    "metric_key",
]

type LossRole = TermsReadout | Space
"""A built component a loss term may resolve by role."""


@final
class ProofRewardContext(FrozenValue, Unpicklable):
    """The declared language and shared service used for shaped proof credit."""

    _noun = "proof-reward binding"
    _because = "holds an interned language and a live proof service"
    __slots__ = ("_language", "_service")

    def __init__(self, language: LanguageAssembly, service: ProofService, /) -> None:
        if not isinstance(language, LanguageAssembly):
            msg = f"a proof-reward language is a LanguageAssembly, not {language!r}"
            raise TypeError(msg)

        if not isinstance(service, ProofService):
            msg = f"a proof-reward service is a ProofService, not {service!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_service", service)

    @property
    def language(self) -> LanguageAssembly:
        """Return the logical form language whose forms are scored."""
        return self._language

    @property
    def service(self) -> ProofService:
        """Return the shared proof service that decides equivalence."""
        return self._service


class LossBindingContext(FrozenValue):
    """Built model components a loss binding may resolve.

    Parameters
    ----------
    name : str
        The declared name of the term whose objective is being built.
    readouts : Mapping[str, LossRole]
        Built components keyed by declared readout role.
    spaces : Mapping[str, LossRole]
        Built spaces keyed by declared space role.
    readout_spaces : Mapping[str, Space]
        The represented space each readout reads, keyed by readout role.
    relations : Mapping[str, SpaceRelationBinding]
        Cross-space relations keyed by source-space role.
    span_decoders : Mapping[str, SpanTokenDecoder]
        Built span decoders keyed by declared span-decoder role.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = (
        "_name",
        "_proof_reward",
        "_readout_spaces",
        "_readouts",
        "_relations",
        "_spaces",
        "_span_decoders",
    )

    _noun = "loss binding context"

    def __init__(
        self,
        *,
        name: str,
        readouts: Mapping[str, LossRole],
        spaces: Mapping[str, LossRole],
        readout_spaces: Mapping[str, Space],
        relations: Mapping[str, SpaceRelationBinding],
        span_decoders: Mapping[str, SpanTokenDecoder],
        proof_reward: ProofRewardBindings | None,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("name", name))
        object.__setattr__(self, "_readouts", MappingProxyType(dict(readouts)))
        object.__setattr__(
            self,
            "_spaces",
            MappingProxyType(dict(spaces)),
        )
        object.__setattr__(
            self, "_readout_spaces", MappingProxyType(dict(readout_spaces))
        )
        object.__setattr__(self, "_relations", MappingProxyType(dict(relations)))
        object.__setattr__(self, "_proof_reward", proof_reward)
        object.__setattr__(
            self, "_span_decoders", MappingProxyType(dict(span_decoders))
        )

    @property
    def name(self) -> str:
        """Return the declared name of the term whose objective is being built."""
        return self._name

    @property
    def readouts(self) -> Mapping[str, LossRole]:
        """Return the built components keyed by readout role."""
        return self._readouts

    @property
    def spaces(self) -> Mapping[str, LossRole]:
        """Return represented spaces keyed by their declared roles."""
        return self._spaces

    def terms_readout(self, role: str, /) -> TermsReadout:
        """Return the term readout installed under a role.

        Parameters
        ----------
        role : str
            Role named by the loss declaration.

        Raises
        ------
        TypeError
            If the role does not hold a term readout.
        ValueError
            If the role is missing.
        """
        return self._component(
            self._readouts,
            role,
            expected=TermsReadout,
            family="term-readout",
        )

    def space(self, role: str, /) -> Space:
        """Return the represented space installed under a role.

        Parameters
        ----------
        role : str
            Role named by the loss declaration.

        Raises
        ------
        TypeError
            If the role does not hold a represented space.
        ValueError
            If the role is missing.
        """
        return self._component(
            self._spaces,
            role,
            expected=Space,
            family="space",
        )

    def readout_space(self, role: str, /) -> Space:
        """Return the represented space read by one term-readout role.

        Raises
        ------
        ValueError
            If the readout role is not built.
        """
        try:
            return self._readout_spaces[role]

        except KeyError:
            known = ", ".join(sorted(self._readout_spaces)) or "no term readout"
            msg = f"term-readout role {role!r} has no represented space; known: {known}"
            raise ValueError(msg) from None

    def relation(self, role: str, /) -> SpaceRelationBinding:
        """Return the relation declared by one source-space role."""
        try:
            return self._relations[role]

        except KeyError:
            known = ", ".join(sorted(self._relations)) or "no space relation"
            msg = f"space role {role!r} has no relation; known: {known}"
            raise ValueError(msg) from None

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        """Return the span decoder installed under a role.

        Parameters
        ----------
        role : str
            Role named by the loss declaration.

        Raises
        ------
        ValueError
            If the role is missing.
        """
        try:
            return self._span_decoders[role]

        except KeyError:
            known = ", ".join(sorted(self._span_decoders)) or "no span-decoder role"
            msg = f"span-decoder role {role!r} is not built; known: {known}"
            raise ValueError(msg) from None

    def proof_reward(self) -> ProofRewardBindings:
        """Return the declared language and shared proof service.

        Raises
        ------
        ValueError
            If the objective's dataset declares no logical form interpretation.
        """
        binding = self._proof_reward

        if binding is None:
            msg = (
                f"loss term {self._name!r} requires gold-form proof capabilities, "
                "and its dataset declares no logical-form interpretation"
            )
            raise ValueError(msg)

        return binding

    @staticmethod
    def _component[ComponentT: LossRole](
        components: Mapping[str, LossRole],
        role: str,
        *,
        expected: type[ComponentT],
        family: str,
    ) -> ComponentT:
        """Return a role after checking its component family.

        Parameters
        ----------
        components : Mapping[str, LossRole]
            Built components keyed by role.
        role : str
            Role named by the loss declaration.
        expected : type[ComponentT]
            Concrete component family required by this loss role.
        family : str
            Component-family name used in errors.

        Raises
        ------
        TypeError
            If the role belongs to another component family.
        ValueError
            If the role is missing.
        """
        try:
            component = components[role]

        except KeyError:
            known = ", ".join(sorted(components)) or f"no {family} role"
            msg = f"{family} role {role!r} is not built; known: {known}"
            raise ValueError(msg) from None

        if not isinstance(component, expected):
            msg = (
                f"{family} role {role!r} holds {type(component).__name__}, "
                f"not {expected.__name__}"
            )
            raise TypeError(msg)

        return component


type LossPlugin[BatchT] = ComponentPlugin[
    object, LossComponentBindings, Objective[PublishedOutputs[BatchT]]
]
"""A loss variant's plugin, listed under the loss tag of its declaration."""


class InactiveLossError(Exception):
    """Reports a step on which no loss term applied.

    Parameters
    ----------
    names : tuple[str, ...]
        The name of every term, all of which were inactive.

    Raises
    ------
    ValueError
        If no name is given.
    """

    __slots__ = ("names",)

    def __init__(self, names: tuple[str, ...]) -> None:
        if not names:
            msg = "an inactive loss names at least one term"
            raise ValueError(msg)

        self.names = names
        listed = ", ".join(names)
        msg = f"no loss term applied to this step, nothing to optimize: {listed}"

        super().__init__(msg)


class MisnamedObjectiveError(Exception):
    """Reports an objective whose name is not the name its term declares.

    Parameters
    ----------
    declared : str
        The name the term declares.
    name : str
        The name the built objective reports.
    """

    __slots__ = ("declared", "name")

    def __init__(self, *, declared: str, name: str) -> None:
        self.declared = declared
        self.name = name
        msg = f"the objective of the term {declared!r} names itself {name!r}"

        super().__init__(msg)


def metric_key(name: str, diagnostic: str | None = None) -> str:
    """Return the telemetry key of a term, or of one of its diagnostics.

    Parameters
    ----------
    name : str
        The name the term is reported under.
    diagnostic : str | None
        The diagnostic's name, or None for the term's own value.

    Returns
    -------
    str
        ``loss/{name}`` for the value and ``loss/{name}/{diagnostic}`` otherwise.

    Raises
    ------
    TypeError
        If the name or the diagnostic is not a string.
    ValueError
        If the name or the diagnostic is empty.
    """
    base = f"loss/{ensure_name('name', name)}"

    if diagnostic is None:
        return base

    return f"{base}/{ensure_name('diagnostic', diagnostic)}"


@final
class WeightSchedule(FrozenValue):
    """A linear ramp of a term's weight over the first optimizer steps.

    Before the first optimizer step the term takes the starting weight. Over the
    ramp's steps the weight moves linearly to the term's declared weight, which
    it keeps from then on.

    Parameters
    ----------
    start : float
        The finite, nonnegative weight before the first optimizer step.
    steps : int
        The positive number of optimizer steps the ramp takes.

    Raises
    ------
    ValueError
        If the starting weight is negative or not finite, or the step count is
        not positive.
    """

    _noun = "weight schedule"

    __slots__ = ("_start", "_steps")

    def __init__(self, start: float, steps: int) -> None:
        held = ensure_finite("a schedule's starting weight", start)

        if held < 0.0:
            msg = f"a schedule's starting weight is negative: {held!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_start", held)
        object.__setattr__(self, "_steps", ensure_positive("a schedule's steps", steps))

    @property
    def start(self) -> float:
        """Return the weight before the first optimizer step."""
        return self._start

    @property
    def steps(self) -> int:
        """Return how many optimizer steps the ramp takes."""
        return self._steps

    def at(self, weight: float, step: int, /) -> float:
        """Return the weight a term takes at one optimizer step.

        Parameters
        ----------
        weight : float
            The term's declared weight, which the ramp reaches.
        step : int
            The count of optimizer steps the run has taken.

        Returns
        -------
        float
            The starting weight moved linearly toward the declared weight by the
            fraction of the ramp completed, and the declared weight once it is.
        """
        if step >= self._steps:
            return weight

        return self._start + (weight - self._start) * step / self._steps

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another schedule ramps from the same start as long."""
        if type(other) is not WeightSchedule:
            return NotImplemented

        return (self._start, self._steps) == (other._start, other._steps)

    @override
    def __hash__(self) -> int:
        """Return a hash over the start and the step count."""
        return hash((WeightSchedule, self._start, self._steps))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this schedule."""
        return f"WeightSchedule({self._start!r}, {self._steps!r})"


@final
class WeightedObjective[BatchT]:
    """Pairs an objective with the name and weight it is assembled under.

    Parameters
    ----------
    name : str
        The name the term declares.
    weight : float
        The multiplier applied to the objective's value in the total.
    objective : Objective[PublishedOutputs[BatchT]]
        The built objective, which must report the declared name.
    schedule : WeightSchedule | None
        How the weight ramps over the first optimizer steps, or None for a weight
        that holds throughout.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    MisnamedObjectiveError
        If the objective's name is not the declared name.
    """

    __slots__ = ("_name", "_objective", "_schedule", "_weight")

    def __init__(
        self,
        name: str,
        weight: float,
        objective: Objective[PublishedOutputs[BatchT]],
        *,
        schedule: WeightSchedule | None = None,
    ) -> None:
        named = ensure_name("name", name)

        if objective.name != named:
            raise MisnamedObjectiveError(declared=named, name=objective.name)

        object.__setattr__(self, "_name", named)
        object.__setattr__(self, "_weight", weight)
        object.__setattr__(self, "_objective", objective)
        object.__setattr__(self, "_schedule", schedule)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("weighted objective")

    @property
    def name(self) -> str:
        """Return the name the term is reported under."""
        return self._name

    @property
    def weight(self) -> float:
        """Return the weight."""
        return self._weight

    @property
    def objective(self) -> Objective[PublishedOutputs[BatchT]]:
        """Return the objective."""
        return self._objective

    @property
    def schedule(self) -> WeightSchedule | None:
        """Return how the weight ramps, or None for a weight that holds throughout."""
        return self._schedule

    def weight_at(self, step: int, /) -> float:
        """Return the weight the term takes at one optimizer step.

        Parameters
        ----------
        step : int
            The count of optimizer steps the run has taken.
        """
        if self._schedule is None:
            return self._weight

        return self._schedule.at(self._weight, step)

    @override
    def __repr__(self) -> str:
        """Return the name, the weight and the schedule."""
        return (
            f"WeightedObjective({self._name!r}, weight={self._weight!r}, "
            f"schedule={self._schedule!r})"
        )


@final
class WeightedTerm:
    """Records one term of a step's loss.

    Parameters
    ----------
    name : str
        The name the term is reported under.
    weight : float
        The multiplier its value entered the total under.
    term : LossTerm
        What the objective returned for the step.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_name", "_term", "_weight")

    def __init__(self, name: str, weight: float, term: LossTerm) -> None:
        object.__setattr__(self, "_name", ensure_name("name", name))
        object.__setattr__(self, "_weight", weight)
        object.__setattr__(self, "_term", term)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("weighted term")

    @property
    def name(self) -> str:
        """Return the name the term is reported under."""
        return self._name

    @property
    def key(self) -> str:
        """Return the telemetry key the term's value is reported under."""
        return metric_key(self._name)

    @property
    def weight(self) -> float:
        """Return the weight."""
        return self._weight

    @property
    def term(self) -> LossTerm:
        """Return the unweighted term."""
        return self._term

    @property
    def inactive(self) -> int:
        """Return one where no item contributed to the term and zero otherwise."""
        return int(self._term.count == 0)

    @override
    def __repr__(self) -> str:
        """Return the name, the weight and the inactive marker."""
        return (
            f"WeightedTerm({self._name!r}, weight={self._weight!r}, "
            f"inactive={self.inactive})"
        )


@final
class LossResult:
    """Records a step's assembled loss.

    Parameters
    ----------
    total : Tensor
        The weighted sum of the active terms, as a scalar on the graph.
    terms : Iterable[WeightedTerm]
        One record per term, in assembly order.
    inactive : int
        How many of the terms were inactive.
    metrics : Mapping[str, float]
        The scalars the step reports, keyed by telemetry key.

    Raises
    ------
    TypeError
        If the inactive count is not an integer.
    ValueError
        If the inactive count is negative or is not the number of inactive records.
    """

    __slots__ = ("_inactive", "_metrics", "_terms", "_total")

    def __init__(
        self,
        total: Tensor,
        terms: Iterable[WeightedTerm],
        inactive: int,
        metrics: Mapping[str, float],
    ) -> None:
        records = tuple(terms)
        counted = ensure_non_negative("inactive", inactive)
        marked = sum(record.inactive for record in records)

        if counted != marked:
            msg = f"inactive is {counted} and {marked} records are marked inactive"
            raise ValueError(msg)

        object.__setattr__(self, "_total", total)
        object.__setattr__(self, "_terms", records)
        object.__setattr__(self, "_inactive", counted)
        object.__setattr__(self, "_metrics", MappingProxyType(dict(metrics)))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("loss result")

    @property
    def total(self) -> Tensor:
        """Return the weighted sum of the active terms."""
        return self._total

    @property
    def terms(self) -> tuple[WeightedTerm, ...]:
        """Return one record per term, in assembly order."""
        return self._terms

    @property
    def inactive(self) -> int:
        """Return how many terms were inactive."""
        return self._inactive

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return the reported scalars, keyed by telemetry key."""
        return self._metrics

    @override
    def __repr__(self) -> str:
        """Return the term and inactive counts."""
        return f"LossResult(terms={len(self._terms)}, inactive={self._inactive})"


def _scalars(records: Iterable[WeightedTerm]) -> dict[str, float]:
    """Return the telemetry of every active term.

    Parameters
    ----------
    records : Iterable[WeightedTerm]
        The step's terms.

    Returns
    -------
    dict[str, float]
        The value of each active term under its key, its residual where it truncated
        a sum, and each of its diagnostics, as plain floats.
    """
    scalars: dict[str, float] = {}

    for record in records:
        if record.inactive:
            continue

        term = record.term
        scalars[record.key] = term.value.item()

        if term.residual is not None:
            scalars[metric_key(record.name, "residual")] = term.residual.item()

        for name, diagnostic in term.diagnostics.items():
            scalars[metric_key(record.name, name)] = diagnostic.item()

    return scalars


class LossAssembly[BatchT]:
    """Evaluates weighted loss terms into one total and its telemetry.

    Parameters
    ----------
    entries : Sequence[WeightedObjective[BatchT]]
        The terms, in the order they are reported.
    published : frozenset[str] | None
        The names the configured graph publishes, or None to accept any objective
        whose declared names are in the published vocabulary.

    Raises
    ------
    ValueError
        If there is no term, if two terms share a name, or if a weight is negative or
        not finite.
    MissingOutputError
        If an objective declares a name the configured graph does not publish.
    """

    __slots__ = ("_entries", "_suite")

    def __init__(
        self,
        entries: Sequence[WeightedObjective[BatchT]],
        *,
        published: frozenset[str] | None = None,
    ) -> None:
        self._entries = tuple(entries)
        self._suite: ObjectiveSuite[BatchT] = ObjectiveSuite(
            [(entry.weight, entry.objective) for entry in self._entries],
            published=published,
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Return the term names, in the order the terms are reported."""
        return tuple(entry.name for entry in self._entries)

    @property
    def residual_keys(self) -> Mapping[str, str]:
        """Return the telemetry key a term's residual is logged under, by its name.

        A term reports a residual on a step where it truncated a sum.
        """
        return MappingProxyType(
            {entry.name: metric_key(entry.name, "residual") for entry in self._entries}
        )

    @property
    def weights(self) -> tuple[float, ...]:
        """Return the weight on each term, in the same order."""
        return self._suite.weights

    @property
    def requires(self) -> frozenset[str]:
        """Return every published name the terms read, which the graph must publish."""
        return self._suite.requires

    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self._entries)

    def __iter__(self) -> Iterator[WeightedObjective[BatchT]]:
        """Yield each term, in the order it is reported."""
        return iter(self._entries)

    def evaluate(
        self, outputs: ObjectiveOutputs[BatchT], /, *, step: int
    ) -> LossResult:
        """Return the assembled loss for one step.

        Each term enters the total under the weight it takes at the step, so a
        scheduled term's weight follows its ramp, and an active scheduled term
        reports that weight under ``loss/{tag}/weight``.

        Parameters
        ----------
        outputs : ObjectiveOutputs[BatchT]
            What the graph published for the step.
        step : int
            The count of optimizer steps the run has taken.

        Returns
        -------
        LossResult
            The weighted total of the active terms, every term's record, how many were
            inactive, and the scalars of the active ones.

        Raises
        ------
        InactiveLossError
            If every term is inactive.
        """
        evaluated = self._suite.evaluate(outputs)
        records = tuple(
            WeightedTerm(entry.name, entry.weight_at(step), evaluated.terms[entry.name])
            for entry in self._entries
        )
        inactive = sum(record.inactive for record in records)

        if inactive == len(records):
            raise InactiveLossError(self.names)

        first, *rest = (
            record.weight * record.term.value
            for record in records
            if not record.inactive
        )
        total = first

        for contribution in rest:
            total = total + contribution

        metrics = _scalars(records)
        metrics.update(
            (metric_key(record.name, "weight"), record.weight)
            for entry, record in zip(self._entries, records, strict=True)
            if entry.schedule is not None and not record.inactive
        )

        return LossResult(total, records, inactive, metrics)

    @override
    def __repr__(self) -> str:
        """Return each term's weight and name, in order."""
        listed = ", ".join(
            f"{entry.weight:g} * {entry.name}" for entry in self._entries
        )

        return f"LossAssembly({listed})"

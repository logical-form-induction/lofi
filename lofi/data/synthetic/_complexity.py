"""Registered complexity measures over closed logical forms and target bands."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data._sampling import SamplingExhaustedError
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi.data.synthetic._values import Histogram, Target
    from lofi.lf import LogicalRole, LogicProfile
    from lofi.terms import Term, TermStore

__all__ = [
    "COMPLEXITY_CONTROLS",
    "ComplexityContext",
    "ComplexityControl",
    "ComplexityMeter",
    "ComplexityPlugin",
    "ComplexityReport",
    "ComplexityTarget",
    "ComplexityUnreachableError",
    "FormComplexity",
    "PairComplexityTarget",
    "complexity_controls",
]


class ComplexityUnreachableError(SamplingExhaustedError):
    """Raised when the draws do not reach a complexity target within budget."""


@final
class ComplexityContext(FrozenValue):
    """The language-owned resources one complexity plugin reads."""

    _noun = "logical-form complexity context"
    __slots__ = ("_abstraction", "_application", "_profile", "_store")

    def __init__(
        self, store: TermStore, profile: LogicProfile, /, *, abstraction: str
    ) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_application", profile.application)
        object.__setattr__(self, "_abstraction", abstraction)

    @property
    def store(self) -> TermStore:
        """Return the store holding the measured form."""
        return self._store

    @property
    def profile(self) -> LogicProfile:
        """Return the language's declared logical roles and constant classes."""
        return self._profile

    @property
    def application(self) -> str:
        """Return the language's application operator name."""
        return self._application

    @property
    def abstraction(self) -> str:
        """Return the calculus abstraction operator name."""
        return self._abstraction


class ComplexityPlugin(FrozenValue):
    """One named scalar measure installed under ``lofi.lf_complexity``."""

    _noun = "logical-form complexity plugin"
    __slots__ = ("_measure", "_name")

    def __init__(
        self, name: str, measure: Callable[[ComplexityContext, Term], int], /
    ) -> None:
        if not name:
            msg = "a logical-form complexity plugin needs a name"
            raise ValueError(msg)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_measure", measure)

    @property
    def name(self) -> str:
        """Return the control name the plugin contributes."""
        return self._name

    def measure(self, context: ComplexityContext, form: Term, /) -> int:
        """Return this control's nonnegative value on one form."""
        return ensure_non_negative(
            f"a {self._name.replace('_', ' ')}", self._measure(context, form)
        )


def _plugins() -> tuple[ComplexityPlugin, ...]:
    """Return the installed complexity plugins in entry-point order."""
    found: list[ComplexityPlugin] = []

    for tag, member in Registry[object]("lf_complexity").members():
        if not isinstance(member, ComplexityPlugin) or member.name != tag:
            msg = (
                f"logical-form complexity entry {tag!r} does not name "
                "its matching ComplexityPlugin"
            )
            raise TypeError(msg)

        found.append(member)

    return tuple(found)


def complexity_controls() -> tuple[str, ...]:
    """Return the installed complexity-control names in registry order."""
    return tuple(plugin.name for plugin in _plugins())


def _spine(context: ComplexityContext, term: Term, /) -> tuple[Term, tuple[Term, ...]]:
    """Unwind the language's declared application operator."""
    head = term
    arguments: list[Term] = []

    while head.op.name == context.application and len(head.args) == 2:
        head, argument = head.args
        arguments.append(argument)

    arguments.reverse()
    return head, tuple(arguments)


def _logical_node(
    context: ComplexityContext, node: Term, /
) -> tuple[str | None, LogicalRole | None, tuple[Term, ...]]:
    """Return a node's constant, logical role and semantic children."""
    head, arguments = _spine(context, node)
    constant = context.store.payload_name(head) if head.op.is_constant else None
    return (
        constant,
        context.profile.logical_roles.get(constant),
        arguments if constant is not None else node.args,
    )


def _quantifier_depth(context: ComplexityContext, form: Term, /) -> int:
    """Return the greatest quantifier nesting depth."""
    deepest = 0
    pending: list[tuple[Term, int]] = [(form, 0)]

    while pending:
        node, above = pending.pop()
        _, role, children = _logical_node(context, node)
        here = above + int(role is not None and role.quantifier)
        deepest = max(deepest, here)
        pending.extend((child, here) for child in children)

    return deepest


def _connective_count(context: ComplexityContext, form: Term, /) -> int:
    """Return how many language-declared connective occurrences a form holds."""
    count = 0
    pending = [form]

    while pending:
        node = pending.pop()
        _, role, children = _logical_node(context, node)
        count += int(role is not None and role.connective)
        pending.extend(children)

    return count


def _predicate_count(context: ComplexityContext, form: Term, /) -> int:
    """Return how many declared predicate occurrences a form holds."""
    predicates = {
        name
        for name, classes in context.profile.constant_classes.items()
        if "predicate" in classes
    }
    count = 0
    pending = [form]

    while pending:
        node = pending.pop()
        constant, _, children = _logical_node(context, node)
        count += int(constant in predicates)
        pending.extend(children)

    return count


def _variable_count(context: ComplexityContext, form: Term, /) -> int:
    """Return binder occurrences plus distinct free-variable names."""
    binders = 0
    free: set[str] = set()
    pending = [form]

    while pending:
        node = pending.pop()
        _, _, children = _logical_node(context, node)

        if node.op.name == context.abstraction:
            binders += 1
        elif node.op.is_free_variable and node.payload is not None:
            free.add(context.store.atom_name(node.payload))

        pending.extend(children)

    return binders + len(free)


def _meaning_depth(context: ComplexityContext, form: Term, /) -> int:
    """Return the greatest edge depth of the complete meaning term."""
    del context
    deepest = 0
    pending: list[tuple[Term, int]] = [(form, 0)]

    while pending:
        node, depth = pending.pop()
        deepest = max(deepest, depth)
        pending.extend((child, depth + 1) for child in node.args)

    return deepest


def _operator_nesting(context: ComplexityContext, form: Term, /) -> int:
    """Return the greatest nesting of language-declared connective roles."""
    deepest = 0
    pending: list[tuple[Term, int]] = [(form, 0)]

    while pending:
        node, above = pending.pop()
        _, role, children = _logical_node(context, node)
        here = above + int(role is not None and role.connective)
        deepest = max(deepest, here)
        pending.extend((child, here) for child in children)

    return deepest


QUANTIFIER_DEPTH = ComplexityPlugin("quantifier_depth", _quantifier_depth)
CONNECTIVE_COUNT = ComplexityPlugin("connective_count", _connective_count)
PREDICATE_COUNT = ComplexityPlugin("predicate_count", _predicate_count)
VARIABLE_COUNT = ComplexityPlugin("variable_count", _variable_count)
MEANING_DEPTH = ComplexityPlugin("meaning_depth", _meaning_depth)
OPERATOR_NESTING = ComplexityPlugin("operator_nesting", _operator_nesting)

type ComplexityControl = str
"""The registry name of one installed logical-form complexity meter."""

COMPLEXITY_CONTROLS: Final[tuple[str, ...]] = complexity_controls()
"""The installed complexity controls in registry order."""


@final
class FormComplexity:
    """The complexity one closed logical form achieves.

    Parameters
    ----------
    controls : Mapping[str, int]
        Every installed control's achieved scalar value.
    predicates : Iterable[str]
        The distinct predicate symbols occurring in the form.
    quantifiers : Iterable[str]
        The distinct binder names that quantify in the form.

    Raises
    ------
    TypeError
        If a measurement is not an integer.
    ValueError
        If a measurement is negative.
    """

    __slots__ = ("_controls", "_predicates", "_quantifiers")

    def __init__(
        self,
        controls: Mapping[str, int],
        *,
        predicates: Iterable[str] = (),
        quantifiers: Iterable[str] = (),
    ) -> None:
        self._controls = MappingProxyType(
            {
                control: ensure_non_negative(f"a {control.replace('_', ' ')}", value)
                for control, value in controls.items()
            }
        )
        self._predicates = frozenset(predicates)
        self._quantifiers = frozenset(quantifiers)

    @property
    def controls(self) -> Mapping[str, int]:
        """Return every installed control's achieved value."""
        return self._controls

    @property
    def predicates(self) -> frozenset[str]:
        """Return the distinct predicate symbols occurring in the form."""
        return self._predicates

    @property
    def quantifiers(self) -> frozenset[str]:
        """Return the distinct binder names that quantify in the form."""
        return self._quantifiers

    def value(self, control: str, /) -> int:
        """Return the measurement one control names.

        Parameters
        ----------
        control : str
            The dimension to read.

        Returns
        -------
        int
            The measurement.

        Raises
        ------
        KeyError
            If the control is not one of the declared dimensions.
        """
        return self._controls[control]

    def overlap(self, other: FormComplexity, /) -> int:
        """Return how many predicate symbols this form shares with another.

        Parameters
        ----------
        other : FormComplexity
            The other form's complexity.

        Returns
        -------
        int
            The size of the intersection of the two predicate sets.
        """
        return len(self._predicates & other._predicates)

    def _fields(
        self,
    ) -> tuple[tuple[tuple[str, int], ...], frozenset[str], frozenset[str]]:
        """Return the fields the measurement is equal on."""
        return (
            tuple(self._controls.items()),
            self._predicates,
            self._quantifiers,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another measurement holds the same numbers and symbols."""
        if type(other) is not FormComplexity:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the measurements and the two symbol sets."""
        return hash((FormComplexity, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this measurement."""
        return (
            f"FormComplexity({dict(self._controls)!r}, "
            f"predicates={sorted(self._predicates)!r}, "
            f"quantifiers={sorted(self._quantifiers)!r})"
        )


@final
class ComplexityMeter:
    """Measures complexity over forms interned in one store.

    A binder that is not the calculus abstraction is a quantifier, so a form
    holding a quantifier the fragment does not declare is counted rather than
    overlooked. A connective is a node the logic's connective vocabulary
    names, which leaves application and equality uncounted.

    Parameters
    ----------
    store : TermStore
        The store the forms are interned in, whose atom names resolve symbols.
    profile : LogicProfile
        The declared classes of typed constants.
    abstraction : str
        The name of the calculus abstraction operator, the one binder that is
        not a quantifier.
    """

    __slots__ = ("_context", "_plugins")

    def __init__(
        self, store: TermStore, profile: LogicProfile, /, *, abstraction: str
    ) -> None:
        self._context = ComplexityContext(store, profile, abstraction=abstraction)
        self._plugins = _plugins()

    @property
    def controls(self) -> tuple[str, ...]:
        """Return the installed controls in measurement order."""
        return tuple(plugin.name for plugin in self._plugins)

    @property
    def predicates(self) -> frozenset[str]:
        """Return the predicate symbols this meter counts."""
        return frozenset(
            name
            for name, classes in self._context.profile.constant_classes.items()
            if "predicate" in classes
        )

    def measure(self, form: Term, /) -> FormComplexity:
        """Return the complexity of one closed logical form.

        Parameters
        ----------
        form : Term
            The form to measure.

        Returns
        -------
        FormComplexity
            Every registered measurement, beside the predicate symbols occurring.
        """
        context = self._context
        predicates = self.predicates
        used: set[str] = set()
        binding: set[str] = set()
        pending = [form]

        while pending:
            node = pending.pop()
            constant, role, children = _logical_node(context, node)

            if role is not None and role.quantifier and constant is not None:
                binding.add(constant)
            elif constant in predicates:
                used.add(constant)
            pending.extend(children)

        return FormComplexity(
            {plugin.name: plugin.measure(context, form) for plugin in self._plugins},
            predicates=used,
            quantifiers=binding,
        )


@final
class ComplexityTarget:
    """The band each complexity control of one side of a pair must fall in.

    A control left unset is uncontrolled: any measurement is admitted on it.

    Parameters
    ----------
    controls : Mapping[str, Target]
        The installed controls to aim at, keyed by their registry names.
    """

    __slots__ = ("_controls",)

    def __init__(self, controls: Mapping[str, Target], /) -> None:
        unknown = tuple(set(controls).difference(complexity_controls()))

        if unknown:
            msg = f"unknown logical-form complexity controls: {sorted(unknown)!r}"
            raise ValueError(msg)

        self._controls = MappingProxyType(dict(controls))

    @property
    def controls(self) -> Mapping[str, Target]:
        """Return the targeted controls in declaration order."""
        return self._controls

    def band(self, control: str, /) -> Target | None:
        """Return the band one control is aimed at, when one is set.

        Parameters
        ----------
        control : str
            The dimension to read.

        Returns
        -------
        Target | None
            The band, or nothing when the control is uncontrolled.

        Raises
        ------
        A control omitted from the target is uncontrolled.
        """
        return self._controls.get(control)

    def admits(self, measured: FormComplexity, /) -> bool:
        """Return whether a measurement falls in every band this target sets.

        Parameters
        ----------
        measured : FormComplexity
            The measurement to test.

        Returns
        -------
        bool
            Whether every controlled dimension admits its measurement.
        """
        for control, band in self._controls.items():
            if not band.admits(measured.value(control)):
                return False

        return True

    def _fields(self) -> tuple[tuple[str, Target], ...]:
        """Return the fields the target is equal on."""
        return tuple(self._controls.items())

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another target sets the same bands."""
        if type(other) is not ComplexityTarget:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the targeted controls."""
        return hash((ComplexityTarget, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return f"ComplexityTarget({dict(self._controls)!r})"


@final
class PairComplexityTarget:
    """What a premise-hypothesis draw aims at on each side and across the pair.

    Parameters
    ----------
    premise : ComplexityTarget | None
        The bands the premise's form must fall in, or nothing to leave the
        premise uncontrolled.
    hypothesis : ComplexityTarget | None
        The bands the hypothesis's form must fall in, or nothing to leave the
        hypothesis uncontrolled.
    overlap : Target | None
        The number of predicate symbols the two sides share, or nothing to
        leave the overlap uncontrolled.
    """

    __slots__ = ("_hypothesis", "_overlap", "_premise")

    def __init__(
        self,
        *,
        premise: ComplexityTarget | None = None,
        hypothesis: ComplexityTarget | None = None,
        overlap: Target | None = None,
    ) -> None:
        self._premise = premise
        self._hypothesis = hypothesis
        self._overlap = overlap

    @property
    def premise(self) -> ComplexityTarget | None:
        """Return the premise's bands, when they are set."""
        return self._premise

    @property
    def hypothesis(self) -> ComplexityTarget | None:
        """Return the hypothesis's bands, when they are set."""
        return self._hypothesis

    @property
    def overlap(self) -> Target | None:
        """Return the predicate-overlap band, when one is set."""
        return self._overlap

    def admits(self, premise: FormComplexity, hypothesis: FormComplexity, /) -> bool:
        """Return whether a candidate pair falls in every band this target sets.

        Parameters
        ----------
        premise : FormComplexity
            The premise form's measurement.
        hypothesis : FormComplexity
            The hypothesis form's measurement.

        Returns
        -------
        bool
            Whether both sides and the overlap are admitted.
        """
        if self._premise is not None and not self._premise.admits(premise):
            return False

        if self._hypothesis is not None and not self._hypothesis.admits(hypothesis):
            return False

        overlap = self._overlap

        return overlap is None or overlap.admits(premise.overlap(hypothesis))

    def _fields(
        self,
    ) -> tuple[ComplexityTarget | None, ComplexityTarget | None, Target | None]:
        """Return the fields the target is equal on."""
        return (self._premise, self._hypothesis, self._overlap)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another target sets the same bands."""
        if type(other) is not PairComplexityTarget:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the two sides and the overlap."""
        return hash((PairComplexityTarget, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return (
            f"PairComplexityTarget(premise={self._premise!r}, "
            f"hypothesis={self._hypothesis!r}, overlap={self._overlap!r})"
        )


@final
class ComplexityReport:
    """The complexity a set of pairs achieved against what it aimed at.

    Parameters
    ----------
    premise : Mapping[str, Histogram]
        Over the accepted premises: the distribution of each control.
    hypothesis : Mapping[str, Histogram]
        Over the accepted hypotheses: the distribution of each control.
    overlap : Histogram
        Over the accepted pairs: how many predicate symbols the two sides
        share.
    rejected : int
        How many candidate pairs fell outside a band and were drawn again.
    target : PairComplexityTarget | None
        The target the draw aimed at, or nothing when it aimed at none.

    Raises
    ------
    TypeError
        If the rejection count is not an integer.
    ValueError
        If the rejection count is negative, or a mapping does not cover every
        control.
    """

    __slots__ = ("_hypothesis", "_overlap", "_premise", "_rejected", "_target")

    def __init__(
        self,
        premise: Mapping[str, Histogram],
        hypothesis: Mapping[str, Histogram],
        *,
        overlap: Histogram,
        rejected: int,
        target: PairComplexityTarget | None,
    ) -> None:
        expected = complexity_controls()

        if tuple(premise) != expected or tuple(hypothesis) != expected:
            msg = (
                "the premise and hypothesis histograms do not cover every control: "
                f"expected {expected!r}, got {tuple(premise)!r} and "
                f"{tuple(hypothesis)!r}"
            )
            raise ValueError(msg)

        self._premise = MappingProxyType(dict(premise))
        self._hypothesis = MappingProxyType(dict(hypothesis))
        self._overlap = overlap
        self._rejected = ensure_non_negative("rejected", rejected)
        self._target = target

    @property
    def premise(self) -> Mapping[str, Histogram]:
        """Return each control's distribution over the accepted premises."""
        return self._premise

    @property
    def hypothesis(self) -> Mapping[str, Histogram]:
        """Return each control's distribution over the accepted hypotheses."""
        return self._hypothesis

    @property
    def overlap(self) -> Histogram:
        """Return the shared-predicate counts over the accepted pairs."""
        return self._overlap

    @property
    def rejected(self) -> int:
        """Return how many candidate pairs were drawn again."""
        return self._rejected

    @property
    def target(self) -> PairComplexityTarget | None:
        """Return the target the draw aimed at, when it aimed at one."""
        return self._target

    def satisfied(self) -> bool:
        """Return whether every achieved bucket lies in the band it was aimed at.

        Returns
        -------
        bool
            True when no target was set, and otherwise whether every observed
            value of every controlled dimension, and every observed overlap,
            falls within its declared tolerance.
        """
        target = self._target

        if target is None:
            return True

        for side, achieved in (
            (target.premise, self._premise),
            (target.hypothesis, self._hypothesis),
        ):
            if side is None:
                continue

            for control, band in side.controls.items():
                if not all(band.admits(value) for value in achieved[control].counts):
                    return False

        if target.overlap is None:
            return True

        return all(target.overlap.admits(value) for value in self._overlap.counts)

    def _fields(
        self,
    ) -> tuple[
        tuple[tuple[str, Histogram], ...],
        tuple[tuple[str, Histogram], ...],
        Histogram,
        int,
        PairComplexityTarget | None,
    ]:
        """Return the fields the report is equal on."""
        return (
            tuple(self._premise.items()),
            tuple(self._hypothesis.items()),
            self._overlap,
            self._rejected,
            self._target,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report measured the same distributions."""
        if type(other) is not ComplexityReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the distributions, the rejections and the target."""
        return hash((ComplexityReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"ComplexityReport({dict(self._premise)!r}, {dict(self._hypothesis)!r}, "
            f"overlap={self._overlap!r}, rejected={self._rejected!r}, "
            f"target={self._target!r})"
        )

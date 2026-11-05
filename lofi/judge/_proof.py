"""Derive proof terms from structural rule schemata and from step witnesses.

A rule with a finite deterministic presentation publishes proof schemata, and
each schema erases to a sequent whose inhabitants the search below enumerates.
A rule standing for a family of schemata has no such presentation: it matches a
step and returns witnesses of what the step did, and those witnesses elaborate
the proof terms that step denotes.
"""

from itertools import product
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Self,
    final,
    override,
)

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue, Unpicklable
from lofi.terms import MVar, Node, Normalization, PVar, Spine, Term, normalize

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge._ports import (
        ElaboratedSchema,
        ProofCalculus,
        ProofSchema,
        StructuralErasure,
    )
    from lofi.terms import Functors, OpName, Path, Pattern, Sort, TermStore

__all__ = [
    "PROOF_LOGICS",
    "CompositionWitness",
    "DerivedAction",
    "Exhaustion",
    "Inhabitants",
    "ProofLogic",
    "ProofSearch",
    "Sequent",
    "StructuralRules",
    "derive_action",
    "elaborate_step",
    "inhabitants",
]

type Exhaustion = Literal["fuel", "inhabitants", "uses"]
"""A search bound that prevented complete enumeration."""

type ProofLogic = Literal["relevant", "linear", "affine", "intuitionistic"]
"""A name a declaration may give a pair of structural rules."""


class StructuralRules(NamedTuple):
    """The structural rules a proof search admits.

    Attributes
    ----------
    contraction : bool
        Whether one assumption may be used more than once.
    weakening : bool
        Whether an assumption may go unused.
    """

    contraction: bool
    weakening: bool


PROOF_LOGICS: Final[Mapping[ProofLogic, StructuralRules]] = MappingProxyType(
    {
        "relevant": StructuralRules(contraction=True, weakening=False),
        "linear": StructuralRules(contraction=False, weakening=False),
        "affine": StructuralRules(contraction=False, weakening=True),
        "intuitionistic": StructuralRules(contraction=True, weakening=True),
    }
)
"""The structural rules each named logic admits, in declaration order."""

type Sequent = tuple[tuple[Term, ...], Term]
"""Erased premises and conclusion."""


_EXHAUSTIONS: frozenset[Exhaustion] = frozenset(("fuel", "inhabitants", "uses"))


@final
class ProofSearch(FrozenValue):
    """Validated structural-logic and enumeration bounds for proof search."""

    _noun = "proof-search policy"

    __slots__ = ("_contraction", "_inhabitants", "_uses", "_weakening")

    def __init__(
        self,
        *,
        contraction: bool,
        weakening: bool,
        max_uses: int,
        max_inhabitants: int,
    ) -> None:
        uses = ensure_non_negative("max_uses", max_uses)
        inhabitants = ensure_non_negative("max_inhabitants", max_inhabitants)

        if uses == 0:
            msg = "max_uses must be positive"
            raise ValueError(msg)

        if inhabitants == 0:
            msg = "max_inhabitants must be positive"
            raise ValueError(msg)

        if not contraction and uses != 1:
            msg = "proof search without contraction uses each assumption at most once"
            raise ValueError(msg)

        object.__setattr__(self, "_contraction", contraction)
        object.__setattr__(self, "_weakening", weakening)
        object.__setattr__(self, "_uses", uses)
        object.__setattr__(self, "_inhabitants", inhabitants)

    @property
    def contraction(self) -> bool:
        """Return whether one assumption may be used more than once."""
        return self._contraction

    @property
    def weakening(self) -> bool:
        """Return whether an assumption may go unused."""
        return self._weakening

    @property
    def max_uses(self) -> int:
        """Return the maximum uses of one assumption."""
        return self._uses

    @property
    def max_inhabitants(self) -> int:
        """Return the maximum distinct normal inhabitants retained."""
        return self._inhabitants

    @property
    def requires_use(self) -> bool:
        """Return whether every assumption must occur in a complete proof."""
        return not self._weakening


@final
class Inhabitants(Unpicklable, FrozenValue):
    """Interned proof terms and genuine search exhaustion."""

    _noun = "proof-search result"
    _because = "contains live interned terms"

    __slots__ = ("_exhausted", "_terms")

    def __init__(
        self,
        terms: Sequence[Term],
        exhausted: Iterable[Exhaustion] = (),
    ) -> None:
        held_exhausted = frozenset(exhausted)

        if not held_exhausted <= _EXHAUSTIONS:
            msg = f"unknown proof-search exhaustion: {held_exhausted - _EXHAUSTIONS!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_terms", tuple(dict.fromkeys(terms)))
        object.__setattr__(self, "_exhausted", held_exhausted)

    @property
    def terms(self) -> tuple[Term, ...]:
        """Return the alpha-distinct proof terms."""
        return self._terms

    @property
    def exhausted(self) -> frozenset[Exhaustion]:
        """Return the bounds reached during search."""
        return self._exhausted

    @override
    def __repr__(self) -> str:
        """Render the result fields."""
        return f"Inhabitants(terms={self._terms!r}, exhausted={self._exhausted!r})"


@final
class DerivedAction(Unpicklable, FrozenValue):
    """One erased sequent and its proof inhabitants."""

    _noun = "proof-search result"
    _because = "contains live interned terms"

    __slots__ = ("_inhabitants", "_sequent")

    def __init__(
        self,
        sequent: Sequent,
        inhabitants: Inhabitants,
    ) -> None:
        object.__setattr__(self, "_sequent", sequent)
        object.__setattr__(self, "_inhabitants", inhabitants)

    @property
    def sequent(self) -> Sequent:
        """Return the erased sequent."""
        return self._sequent

    @property
    def inhabitants(self) -> Inhabitants:
        """Return the proof-search result."""
        return self._inhabitants

    @override
    def __repr__(self) -> str:
        """Render the result fields."""
        return (
            f"DerivedAction(sequent={self._sequent!r}, "
            f"inhabitants={self._inhabitants!r})"
        )


type _TypedHead = tuple[Term, Term]
type _Candidate = tuple[Term, tuple[int, ...]]

_ARROW_ARITY = 2


def _owned(store: TermStore, term: Term, channel: str) -> Term:
    """Reject a proof constructor that returns a foreign term."""
    if not store.contains(term):
        msg = f"the proof calculus {channel} channel returned a foreign term"
        raise ValueError(msg)

    return term


def _hypothesis(
    calculus: ProofCalculus,
    index: int,
) -> Term:
    """Validate one caller-constructed open hypothesis."""
    store = calculus.store
    term = _owned(store, calculus.hypothesis(index), "hypothesis")

    if term.payload is None:
        msg = "the proof calculus hypothesis channel did not return a free variable"
        raise ValueError(msg)

    try:
        name = store.atom_name(term.payload)
        canonical = store.fvar(term.op.sort, name)
    except ValueError as error:
        msg = "the proof calculus hypothesis has an invalid atom payload"
        raise ValueError(msg) from error

    if canonical is not term:
        msg = "the proof calculus hypothesis channel did not return a free variable"
        raise ValueError(msg)

    return term


def _variable_sort(calculus: ProofCalculus) -> Sort:
    """Return the sort of the variables a proof calculus abstracts over.

    Parameters
    ----------
    calculus : ProofCalculus
        The proof calculus.
    """
    abstraction = calculus.store.signature.by_name(calculus.operators.lam)

    return abstraction.binding[0].sort


def _free_variable(
    calculus: ProofCalculus,
    name: str,
) -> Term:
    """Validate one caller-constructed free variable."""
    store = calculus.store
    expected = store.fvar(_variable_sort(calculus), name)
    received = _owned(store, calculus.free_variable(name), "free-variable")

    if received is not expected:
        msg = "the proof calculus free-variable channel returned a malformed term"
        raise ValueError(msg)

    return received


def _bound_variable(
    calculus: ProofCalculus,
    index: int,
) -> Term:
    """Validate one caller-constructed bound variable."""
    store = calculus.store
    expected = store.bvar(_variable_sort(calculus), index)
    received = _owned(store, calculus.bound_variable(index), "bound-variable")

    if received is not expected:
        msg = "the proof calculus bound-variable channel returned a malformed term"
        raise ValueError(msg)

    return received


def _application(
    calculus: ProofCalculus,
    function: Term,
    argument: Term,
) -> Term:
    """Validate one caller-constructed application."""
    store = calculus.store
    expected = store.op(
        store.signature.by_name(calculus.operators.app),
        function,
        argument,
    )
    received = _owned(
        store,
        calculus.application(function, argument),
        "application",
    )

    if received is not expected:
        msg = "the proof calculus application channel returned a malformed term"
        raise ValueError(msg)

    return received


def _abstraction(
    calculus: ProofCalculus,
    variable: str,
    body: Term,
) -> Term:
    """Validate one caller-constructed abstraction."""
    store = calculus.store
    expected = store.bind(calculus.operators.lam, variable, body)
    received = _owned(
        store,
        calculus.abstraction(variable, body),
        "abstraction",
    )

    if received is not expected:
        msg = "the proof calculus abstraction channel returned a malformed term"
        raise ValueError(msg)

    return received


def _validate_calculus(calculus: ProofCalculus) -> Term:
    """Validate every proof-construction channel."""
    try:
        hypothesis = _hypothesis(calculus, 0)

        if _hypothesis(calculus, 0) is not hypothesis:
            msg = "the proof calculus hypothesis channel is not stable"
            raise ValueError(msg)

        if _hypothesis(calculus, 1) is hypothesis:
            msg = "the proof calculus hypothesis channel is not injective"
            raise ValueError(msg)

        probe_name = "__proof_probe"
        free = _free_variable(calculus, probe_name)
        _bound_variable(calculus, 0)
        _application(calculus, hypothesis, hypothesis)
        _abstraction(calculus, probe_name, free)
    except AttributeError as error:
        msg = "the proof calculus lacks a required construction channel"
        raise TypeError(msg) from error

    return hypothesis


def _fresh_variable(
    occupied: frozenset[Term],
    calculus: ProofCalculus,
    depth: int,
) -> tuple[str, Term]:
    """Construct a deterministic variable distinct from every occupied term."""
    nonce = 0

    while True:
        name = f"__proof_bound_{depth}_{nonce}"
        variable = _free_variable(calculus, name)

        if variable not in occupied:
            return name, variable

        nonce += 1


def _uncurry(
    term: Term,
    arrow: OpName,
) -> tuple[tuple[Term, ...], Term]:
    """Split a type into its curried arguments and atomic target."""
    arguments: list[Term] = []
    current = term

    while current.op.name == arrow and len(current.args) == _ARROW_ARITY:
        argument, current = current.args
        arguments.append(argument)

    return tuple(arguments), current


@final
class CompositionWitness(FrozenValue):
    """The orientation and argument spine of one generalized-composition step.

    The principal premise is the functor whose argument the container premise
    delivers, and the frames are the operators of the container's result spine,
    shallowest first. The step composes at the order those frames count, and it
    crosses when the deepest frame opposes the principal operator.

    Parameters
    ----------
    principal : int
        The premise slot holding the principal functor.
    container : int
        The premise slot holding the functor whose result spine is carried
        through to the conclusion.
    slash : OpName
        The principal functor's operator.
    frames : tuple[OpName, ...]
        The container's result-spine operators, shallowest first.
    """

    _noun = "composition witness"

    __slots__ = ("_container", "_frames", "_principal", "_slash")

    def __init__(
        self,
        principal: int,
        container: int,
        slash: OpName,
        frames: tuple[OpName, ...],
    ) -> None:
        held_principal = ensure_non_negative("principal", principal)
        held_container = ensure_non_negative("container", container)

        if held_principal == held_container:
            msg = f"premise slot {held_principal} is both principal and container"
            raise ValueError(msg)

        object.__setattr__(self, "_principal", held_principal)
        object.__setattr__(self, "_container", held_container)
        object.__setattr__(self, "_slash", ensure_name("slash", slash))
        object.__setattr__(
            self,
            "_frames",
            tuple(
                ensure_name(f"frames[{at}]", frame) for at, frame in enumerate(frames)
            ),
        )

    @property
    def principal(self) -> int:
        """Return the premise slot holding the principal functor."""
        return self._principal

    @property
    def container(self) -> int:
        """Return the premise slot whose result spine reaches the conclusion."""
        return self._container

    @property
    def slash(self) -> OpName:
        """Return the principal functor's operator."""
        return self._slash

    @property
    def frames(self) -> tuple[OpName, ...]:
        """Return the container's result-spine operators, shallowest first."""
        return self._frames

    @property
    def order(self) -> int:
        """Return the composition order the step used."""
        return len(self._frames)

    @property
    def crossed(self) -> bool:
        """Report whether the deepest frame opposes the principal operator."""
        return bool(self._frames) and self._frames[-1] != self._slash

    def elaborate(self, calculus: ProofCalculus, /) -> Term:
        """Return the lambda term the step denotes.

        The term abstracts one variable per spine frame, applies the container
        to them in spine order, and applies the principal functor to the
        result. At order zero it is the principal applied to the container,
        which is the term of application.

        Parameters
        ----------
        calculus : ProofCalculus
            The caller's proof-term construction channels.

        Returns
        -------
        Term
            The interned proof term, built through the calculus.

        Raises
        ------
        TypeError
            If the calculus lacks a required construction channel.
        ValueError
            If a construction channel returns a foreign or malformed term.
        """
        _validate_calculus(calculus)
        principal = _hypothesis(calculus, self._principal)
        container = _hypothesis(calculus, self._container)
        occupied = frozenset({principal, container})
        bound: list[str] = []
        applied = container

        for depth in range(len(self._frames)):
            name, variable = _fresh_variable(occupied, calculus, depth)
            applied = _application(calculus, applied, variable)
            bound.append(name)

        term = _application(calculus, principal, applied)

        for name in reversed(bound):
            term = _abstraction(calculus, name, term)

        return term

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the witness declaration."""
        return {
            "principal": self._principal,
            "container": self._container,
            "slash": self._slash,
            "frames": list(self._frames),
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, JsonValue], /) -> Self:
        """Return the witness a declaration describes.

        Parameters
        ----------
        spec : Mapping[str, JsonValue]
            A declaration of the shape :meth:`to_spec` returns.

        Returns
        -------
        Self
            The witness the declaration describes.

        Raises
        ------
        ValueError
            If a field is absent or holds a value of the wrong kind.
        """
        principal = spec.get("principal")
        container = spec.get("container")
        slash = spec.get("slash")
        frames = spec.get("frames")
        declared = frames if isinstance(frames, list) else []
        held = tuple(frame for frame in declared if isinstance(frame, str))

        if (
            isinstance(principal, bool)
            or not isinstance(principal, int)
            or isinstance(container, bool)
            or not isinstance(container, int)
            or not isinstance(slash, str)
            or not isinstance(frames, list)
            or len(held) != len(frames)
        ):
            msg = f"malformed composition witness declaration: {dict(spec)!r}"
            raise ValueError(msg)

        return cls(principal, container, slash, held)

    def _identity(self) -> tuple[int, int, OpName, tuple[OpName, ...]]:
        """Return the fields in comparison order."""
        return (self._principal, self._container, self._slash, self._frames)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another witness has the same fields."""
        if type(other) is not CompositionWitness:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((CompositionWitness, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"CompositionWitness({self._principal!r}, {self._container!r}, "
            f"{self._slash!r}, {self._frames!r})"
        )


def inhabitants(
    ctx: Sequence[Term],
    goal: Term,
    *,
    calculus: ProofCalculus,
    arrow: OpName,
    normalization: Normalization,
    search: ProofSearch,
) -> Inhabitants:
    """Enumerate inhabitants under the declared structural rules and normal form.

    Raises
    ------
    TypeError
        If a calculus channel is missing.
    ValueError
        If a bound is invalid or the calculus channels violate their contract.
    """
    held_fuel = normalization.fuel
    held_uses = search.max_uses
    held_inhabitants = search.max_inhabitants
    requires_use = search.requires_use

    proof_store = calculus.store
    hypothesis_probe = _validate_calculus(calculus)

    heads: tuple[_TypedHead, ...] = tuple(
        (held, _hypothesis(calculus, at)) for at, held in enumerate(ctx)
    )

    if heads and heads[0][1] is not hypothesis_probe:
        msg = "the proof calculus hypothesis channel is not stable"
        raise ValueError(msg)

    if len({head for _, head in heads}) != len(heads):
        msg = "the proof calculus hypothesis channel is not injective"
        raise ValueError(msg)

    if any(_hypothesis(calculus, at) is not head for at, (_, head) in enumerate(heads)):
        msg = "the proof calculus hypothesis channel is not stable"
        raise ValueError(msg)

    exhausted: set[Exhaustion] = set()

    def can_start(
        available: tuple[_TypedHead, ...],
        target: Term,
    ) -> bool:
        """Check whether one type-directed step exists."""
        if target.op.name == arrow and len(target.args) == _ARROW_ARITY:
            return True

        return any(_uncurry(held, arrow)[1] is target for held, _ in available)

    def enumerate_candidates(
        available: tuple[_TypedHead, ...],
        target: Term,
        counts: tuple[int, ...],
        remaining: int,
        depth: int,
    ) -> Iterator[_Candidate]:
        """Yield distinct candidates from one bounded focused-search state."""
        if remaining == 0:
            if can_start(available, target):
                exhausted.add("fuel")

            return

        if target.op.name == arrow and len(target.args) == _ARROW_ARITY:
            domain, codomain = target.args
            variable, proof_variable = _fresh_variable(
                frozenset(head for _, head in available), calculus, depth
            )
            extended = (*available, (domain, proof_variable))
            seen: set[_Candidate] = set()

            for body, used in enumerate_candidates(
                extended,
                codomain,
                (*counts, 0),
                remaining - 1,
                depth + 1,
            ):
                if requires_use and used[-1] == 0:
                    continue

                abstraction = _abstraction(calculus, variable, body)
                candidate = (abstraction, used[:-1])

                if candidate in seen:
                    continue

                seen.add(candidate)
                yield candidate

            return

        seen = set()

        for at, (held_type, head) in enumerate(available):
            arguments, result = _uncurry(held_type, arrow)

            if result is not target:
                continue

            if counts[at] >= held_uses:
                if search.contraction:
                    exhausted.add("uses")

                continue

            initial = list(counts)
            initial[at] += 1
            initial_candidate = (head, tuple(initial))

            def apply_arguments(
                candidate: _Candidate,
                remaining_arguments: tuple[Term, ...],
            ) -> Iterator[_Candidate]:
                """Yield applications of the remaining argument types."""
                if not remaining_arguments:
                    yield candidate
                    return

                function, used = candidate
                argument_type, *rest = remaining_arguments

                for argument, updated in enumerate_candidates(
                    available,
                    argument_type,
                    used,
                    remaining - 1,
                    depth,
                ):
                    application = _application(calculus, function, argument)
                    yield from apply_arguments(
                        (application, updated),
                        tuple(rest),
                    )

            for candidate in apply_arguments(initial_candidate, arguments):
                if candidate in seen:
                    continue

                seen.add(candidate)
                yield candidate

    initial_counts = (0,) * len(heads)
    terms: list[Term] = []
    retained: set[Term] = set()

    for proof, counts in enumerate_candidates(
        heads,
        goal,
        initial_counts,
        held_fuel,
        0,
    ):
        if requires_use and any(count == 0 for count in counts):
            continue

        normalized = normalize(
            proof_store,
            proof,
            calculus.operators,
            normalization=normalization,
        )

        if normalized in retained:
            continue

        if len(terms) == held_inhabitants:
            exhausted.add("inhabitants")
            break

        terms.append(normalized)
        retained.add(normalized)

    return Inhabitants(terms, tuple(exhausted))


def _deterministic(pattern: Pattern) -> bool:
    """Check whether a pattern has only term metavariables and fixed nodes."""
    if isinstance(pattern, MVar):
        return True

    if isinstance(pattern, Spine):
        return False

    return (
        not isinstance(pattern.op, PVar)
        and not isinstance(pattern.payload, PVar)
        and all(_deterministic(child) for child in pattern.args)
    )


def _erase_pattern(
    pattern: Pattern,
    erasure: StructuralErasure,
) -> Term:
    """Erase a deterministic category pattern structurally."""
    store = erasure.store

    if isinstance(pattern, MVar):
        return store.fvar(erasure.target_sort, pattern.name)

    if (
        isinstance(pattern, Spine)
        or isinstance(pattern.op, PVar)
        or isinstance(pattern.payload, PVar)
    ):
        msg = "the proof schema is not structurally deterministic"
        raise TypeError(msg)

    signature = store.signature
    category_sort = erasure.category_sort
    source_operator = signature.by_name(pattern.op)

    if source_operator.sort != category_sort:
        msg = (
            f"schema operator {pattern.op!r} has sort {source_operator.sort!r}, "
            f"not category sort {category_sort!r}"
        )
        raise ValueError(msg)

    if pattern.payload is not None:
        if source_operator is not signature.constant(category_sort) or pattern.args:
            msg = f"schema node {pattern!r} is not a category constant"
            raise ValueError(msg)

        name = signature.vocabulary(category_sort).name(pattern.payload)
        core_name = signature.constant_core(category_sort, name)
        image = erasure.atoms[core_name]

        if not store.contains(image):
            msg = f"erasure atom image {core_name!r} is outside its store lineage"
            raise ValueError(msg)

        return image

    if not pattern.args:
        msg = f"schema node {pattern.op!r} is not a category constant"
        raise ValueError(msg)

    if len(pattern.args) != source_operator.arity:
        msg = f"schema operator {pattern.op!r} has malformed arity"
        raise ValueError(msg)

    erased = tuple(_erase_pattern(child, erasure) for child in pattern.args)
    core_name = signature.erase(source_operator).name

    return erasure.image(core_name, erased)


def derive_action(
    schema: ProofSchema,
    erasure: StructuralErasure,
    /,
    *,
    calculus: ProofCalculus,
    normalization: Normalization,
    search: ProofSearch,
) -> tuple[DerivedAction, ...]:
    """Erase oriented schemata and derive their proof inhabitants.

    Raises
    ------
    TypeError
        If a proof-calculus channel is missing.
    UnknownOperatorError
        If an erasure operator is undeclared.
    ValueError
        If the erasure or proof-calculus channel is invalid.
    """
    store = erasure.store
    arrow_operator = store.signature.by_name(erasure.function_type)
    free_probe = store.fvar(arrow_operator.sort, "__proof_type_probe")

    if (
        not store.contains(free_probe)
        or arrow_operator.arg_sorts != (arrow_operator.sort, arrow_operator.sort)
        or arrow_operator.has_payload
        or arrow_operator.is_binder
    ):
        msg = "the erasure target lacks a valid variable or arrow channel"
        raise ValueError(msg)

    actions: list[DerivedAction] = []

    for premise_patterns, conclusion_pattern in schema.proof_schemata:
        patterns = (*premise_patterns, conclusion_pattern)

        if not all(_deterministic(pattern) for pattern in patterns):
            continue

        premises = tuple(
            _erase_pattern(pattern, erasure) for pattern in premise_patterns
        )
        conclusion = _erase_pattern(conclusion_pattern, erasure)
        sequent: Sequent = (premises, conclusion)
        found = inhabitants(
            premises,
            conclusion,
            calculus=calculus,
            arrow=erasure.function_type,
            normalization=normalization,
            search=search,
        )
        actions.append(DerivedAction(sequent, found))

    return tuple(actions)


def elaborate_step(
    rule: ElaboratedSchema[Term],
    inputs: tuple[Term, ...],
    output: Term,
    erasure: StructuralErasure,
    /,
    *,
    calculus: ProofCalculus,
) -> DerivedAction | None:
    """Erase one derivation step and elaborate the proof term it denotes.

    The rule matches the step and returns its witnesses, so the order the step
    used is read off the premises rather than declared, and the term and the
    erased sequent come from the same match.

    Parameters
    ----------
    rule : ElaboratedSchema[Term]
        The rule that licenses the step.
    inputs : tuple[Term, ...]
        Premise categories in rule order.
    output : Term
        The conclusion category drawn from them.
    erasure : StructuralErasure
        The category-to-type erasure.
    calculus : ProofCalculus
        The caller's proof-term construction channels.

    Returns
    -------
    DerivedAction | None
        The erased sequent with every distinct term the step denotes, or
        ``None`` when the rule licenses no such step.

    Raises
    ------
    TypeError
        If a proof-calculus channel is missing.
    ValueError
        If a proof-calculus channel returns a foreign or malformed term.
    """
    witnesses = rule.witnesses(inputs, output)

    if not witnesses:
        return None

    sequent: Sequent = (tuple(erasure(term) for term in inputs), erasure(output))

    _validate_calculus(calculus)
    hypotheses = tuple(_hypothesis(calculus, at) for at in range(len(inputs)))

    if len(set(hypotheses)) != len(hypotheses):
        msg = "the proof calculus hypothesis channel is not injective"
        raise ValueError(msg)

    if any(
        _hypothesis(calculus, at) is not hypothesis
        for at, hypothesis in enumerate(hypotheses)
    ):
        msg = "the proof calculus hypothesis channel is not stable"
        raise ValueError(msg)

    expected = 0

    for hypothesis in hypotheses:
        if hypothesis.payload is None:
            msg = "the proof calculus hypothesis channel returned a malformed term"
            raise ValueError(msg)

        expected |= 1 << hypothesis.payload

    terms: list[Term] = []

    for witness in witnesses:
        term = _owned(calculus.store, witness.elaborate(calculus), "witness")

        if term.free != expected:
            msg = (
                "the proof witness returned hypothesis mask "
                f"{term.free}, not {expected} for arity {len(inputs)}"
            )
            raise ValueError(msg)

        terms.append(term)

    return DerivedAction(sequent, Inhabitants(terms))


def _functor_pattern[T](
    functors: Functors[T],
    slash: OpName,
    result: Pattern,
    argument: Pattern,
) -> Node | None:
    """Build a binary functor pattern from declared one-step paths."""
    result_path: Path = functors.result_path
    argument_path: Path = functors.argument_path

    if {result_path, argument_path} != {(0,), (1,)}:
        return None

    children = (result, argument) if result_path == (0,) else (argument, result)

    return Node(slash, children)


def _spine_patterns[T](
    functors: Functors[T],
    order: int,
    deepest: OpName,
    foot: Pattern,
    arguments: Sequence[Pattern],
    prefix_slashes: Iterable[OpName] | None = None,
) -> Iterator[Node]:
    """Build every deterministic slash spine of a fixed order."""
    if order == 0:
        return

    available = functors.slashes if prefix_slashes is None else prefix_slashes
    ordered_slashes = tuple(sorted(available))

    for prefix in product(ordered_slashes, repeat=order - 1):
        held: Pattern = foot

        for slash, argument in reversed(
            tuple(zip((*prefix, deepest), arguments, strict=True))
        ):
            made = _functor_pattern(functors, slash, held, argument)

            if made is None:
                break

            held = made
        else:
            if isinstance(held, Node):
                yield held


def _term_pattern(term: Term) -> Node:
    """Convert an interned term into a fixed structural pattern."""
    return Node(
        term.op.name,
        tuple(_term_pattern(child) for child in term.args),
        term.payload,
    )

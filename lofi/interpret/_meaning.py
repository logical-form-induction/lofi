"""Meaning-term interpretations of derivations."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast, final, override

from lofi._frozen import FrozenValue
from lofi.interpret._errors import (
    AmbiguousMeaningError,
    CategoryWitnessRequiredError,
)
from lofi.judge._ports import ElaboratedSchema, ProofSchema
from lofi.judge._proof import derive_action, elaborate_step
from lofi.terms import Normalization, Term, normalize, substitute

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from lofi.interpret._derive import StepContext
    from lofi.judge import (
        Exhaustion,
        ProofCalculus,
        ProofSearch,
        Rule,
        Sequent,
        StructuralErasure,
    )
    from lofi.lexicon import Entry

__all__ = [
    "INHABITANT_SELECTIONS",
    "NORMALIZATION_POLICIES",
    "NORMALIZES_STEPS",
    "InhabitantSelection",
    "MeaningActions",
    "MeaningReading",
    "MeaningSelection",
    "NormalizationPolicy",
    "compile_meaning_actions",
    "run_action",
]

type NormalizationPolicy = Literal["eager", "lazy"]
"""When a meaning reading beta-normalizes: at every rule application, or never."""

NORMALIZATION_POLICIES: Final[tuple[NormalizationPolicy, ...]] = ("eager", "lazy")
"""The normalization policies a caller may declare, in declaration order."""

NORMALIZES_STEPS: Final[Mapping[NormalizationPolicy, bool]] = MappingProxyType(
    {"eager": True, "lazy": False}
)
"""Whether each declared policy normalizes the meaning of every rule application."""

type InhabitantSelection = Literal["unique", "first", "smallest", "all"]
"""How proof-search inhabitants become semantic rule actions."""

INHABITANT_SELECTIONS: Final[tuple[InhabitantSelection, ...]] = (
    "unique",
    "first",
    "smallest",
    "all",
)
"""The supplied inhabitant-selection policies, in declaration order."""


def _unique(terms: tuple[Term, ...], /) -> tuple[Term, ...]:
    """Keep the inhabitants only when exactly one exists."""
    return terms if len(terms) == 1 else ()


def _first(terms: tuple[Term, ...], /) -> tuple[Term, ...]:
    """Keep the first inhabitant in proof-search order."""
    return terms[:1]


def _smallest(terms: tuple[Term, ...], /) -> tuple[Term, ...]:
    """Keep the first inhabitant of least size."""
    if not terms:
        return ()

    size = min(term.size for term in terms)

    return (next(term for term in terms if term.size == size),)


def _all(terms: tuple[Term, ...], /) -> tuple[Term, ...]:
    """Keep every inhabitant."""
    return terms


_SELECTORS: Final[
    Mapping[InhabitantSelection, Callable[[tuple[Term, ...]], tuple[Term, ...]]]
] = MappingProxyType(
    {"unique": _unique, "first": _first, "smallest": _smallest, "all": _all}
)
"""The selection each declared inhabitant-selection name applies."""


@final
class MeaningSelection(FrozenValue):
    """Validated policy for selecting semantic actions from proof inhabitants."""

    _noun = "meaning-selection policy"

    __slots__ = ("_kind", "_select")

    def __init__(self, kind: InhabitantSelection, /) -> None:
        if kind not in INHABITANT_SELECTIONS:
            msg = (
                "an inhabitant selection is one of "
                f"{list(INHABITANT_SELECTIONS)!r}: {kind!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_select", _SELECTORS[kind])

    @property
    def kind(self) -> InhabitantSelection:
        """Return the declared selection name."""
        return self._kind

    def select(self, terms: Sequence[Term], /) -> tuple[Term, ...]:
        """Return selected terms in deterministic proof-search order."""
        return self._select(tuple(terms))


@final
class _SelectedCase(FrozenValue):
    """Selected proof inhabitants for one erased sequent or supplied rule."""

    _noun = "meaning-action case"
    __slots__ = ("_sequent", "_terms")

    def __init__(
        self,
        sequent: Sequent | None,
        terms: Sequence[Term],
        /,
    ) -> None:
        object.__setattr__(self, "_sequent", sequent)
        object.__setattr__(self, "_terms", tuple(terms))

    @property
    def sequent(self) -> Sequent | None:
        """Return the erased sequent, or no sequent for a supplied override."""
        return self._sequent

    @property
    def terms(self) -> tuple[Term, ...]:
        """Return selected inhabitants in declared search order."""
        return self._terms

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SelectedCase):
            return NotImplemented

        return self._sequent == other._sequent and self._terms == other._terms

    @override
    def __hash__(self) -> int:
        return hash((_SelectedCase, self._sequent, self._terms))

    @override
    def __repr__(self) -> str:
        return f"_SelectedCase(sequent={self._sequent!r}, terms={self._terms!r})"


@final
class _SelectedAction(FrozenValue):
    """A finite selected action program indexed by erased sequents."""

    _noun = "selected meaning-action program"
    __slots__ = ("_cases",)

    def __init__(self, cases: Sequence[_SelectedCase], /) -> None:
        object.__setattr__(self, "_cases", tuple(cases))

    @property
    def scalar(self) -> bool:
        """Return whether every selected case has exactly one term."""
        return all(len(case.terms) == 1 for case in self._cases)

    def resolve(
        self,
        rule_name: str,
        step: StepContext[Term, object] | None,
        /,
        *,
        erasure: StructuralErasure,
        calculus: ProofCalculus,
    ) -> tuple[Term, ...]:
        """Return terms selected for one step, or invariant terms without one."""
        del calculus

        if step is None:
            alternatives = tuple(dict.fromkeys(case.terms for case in self._cases))

            if len(alternatives) != 1:
                raise CategoryWitnessRequiredError(rule_name)

            return alternatives[0]

        sequent: Sequent = (
            tuple(erasure(premise) for premise in step.premises),
            erasure(step.conclusion),
        )
        matched: list[Term] = []

        for case in self._cases:
            if case.sequent is None or _sequent_instance(case.sequent, sequent):
                matched.extend(case.terms)

        return tuple(dict.fromkeys(matched))

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SelectedAction):
            return NotImplemented

        return self._cases == other._cases

    @override
    def __hash__(self) -> int:
        return hash((_SelectedAction, self._cases))


@final
class _WitnessAction(FrozenValue):
    """An action program elaborated from each concrete derivation step."""

    _noun = "witness meaning-action program"
    __slots__ = ("_rule", "_selection")

    def __init__(
        self,
        rule: ElaboratedSchema[Term],
        selection: MeaningSelection,
        /,
    ) -> None:
        object.__setattr__(self, "_rule", rule)
        object.__setattr__(self, "_selection", selection)

    @property
    def scalar(self) -> bool:
        """Return whether selection retains at most one witness per step."""
        return self._selection.kind != "all" or self._rule.witness_is_unique

    def resolve(
        self,
        rule_name: str,
        step: StepContext[Term, object] | None,
        /,
        *,
        erasure: StructuralErasure,
        calculus: ProofCalculus,
    ) -> tuple[Term, ...]:
        """Elaborate the exact witnessed step."""
        if step is None:
            raise CategoryWitnessRequiredError(rule_name)

        elaborated = elaborate_step(
            self._rule,
            step.premises,
            step.conclusion,
            erasure,
            calculus=calculus,
        )

        return (
            ()
            if elaborated is None
            else self._selection.select(elaborated.inhabitants.terms)
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _WitnessAction):
            return NotImplemented

        return self._rule is other._rule and self._selection == other._selection

    @override
    def __hash__(self) -> int:
        return hash((_WitnessAction, id(self._rule), self._selection))


type _ActionProgram = _SelectedAction | _WitnessAction


@final
class MeaningActions(FrozenValue):
    """Frozen semantic-action programs and derivation diagnostics."""

    _noun = "meaning interpretation value"

    __slots__ = (
        "_ambiguous",
        "_calculus",
        "_erasure",
        "_exhausted",
        "_programs",
        "_uncovered",
    )

    def __init__(
        self,
        programs: Mapping[str, _ActionProgram],
        erasure: StructuralErasure,
        calculus: ProofCalculus,
        uncovered: Iterable[str] = (),
        ambiguous: Iterable[str] = (),
        exhausted: Mapping[str, frozenset[Exhaustion]] | None = None,
    ) -> None:
        object.__setattr__(self, "_programs", MappingProxyType(dict(programs)))
        object.__setattr__(self, "_erasure", erasure)
        object.__setattr__(self, "_calculus", calculus)
        object.__setattr__(self, "_uncovered", frozenset(uncovered))
        object.__setattr__(self, "_ambiguous", frozenset(ambiguous))
        object.__setattr__(
            self,
            "_exhausted",
            MappingProxyType({} if exhausted is None else dict(exhausted)),
        )

    @property
    def rules(self) -> tuple[str, ...]:
        """Return rule names with a semantic-action program."""
        return tuple(self._programs)

    @property
    def calculus(self) -> ProofCalculus:
        """Return the sole proof calculus that owns every selected term."""
        return self._calculus

    @property
    def uncovered(self) -> frozenset[str]:
        """Return rules with no derived action."""
        return self._uncovered

    @property
    def ambiguous(self) -> frozenset[str]:
        """Return rules without a unique complete derivation."""
        return self._ambiguous

    @property
    def exhausted(self) -> Mapping[str, frozenset[Exhaustion]]:
        """Return genuinely exhausted search bounds by rule name."""
        return self._exhausted

    def is_scalar(self, rule_name: str, /) -> bool:
        """Return whether a rule program selects at most one term per step."""
        program = self._programs.get(rule_name)
        return program is not None and program.scalar

    def resolve[Weight](
        self,
        rule_name: str,
        step: StepContext[Term, Weight] | None,
        /,
    ) -> tuple[Term, ...]:
        """Resolve one rule through its sole compiled action program."""
        program = self._programs.get(rule_name)

        if program is None:
            return ()

        held_step = cast("StepContext[Term, object] | None", step)
        return program.resolve(
            rule_name,
            held_step,
            erasure=self._erasure,
            calculus=self._calculus,
        )

    def __len__(self) -> int:
        """Return the number of rules with selected action cases."""
        return len(self._programs)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether actions and diagnostics agree."""
        if type(other) is not MeaningActions:
            return NotImplemented

        return (
            dict(self._programs) == dict(other._programs)
            and self._erasure is other._erasure
            and self._calculus is other._calculus
            and self._uncovered == other._uncovered
            and self._ambiguous == other._ambiguous
            and self._exhausted == other._exhausted
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over actions and diagnostics."""
        return hash(
            (
                MeaningActions,
                tuple(sorted(self._programs.items())),
                id(self._erasure),
                id(self._calculus),
                self._uncovered,
                self._ambiguous,
                tuple(sorted(self._exhausted.items())),
            )
        )

    @override
    def __repr__(self) -> str:
        """Return a representation with action and diagnostic counts."""
        return (
            f"MeaningActions(rules={len(self._programs)}, "
            f"uncovered={sorted(self._uncovered)!r}, "
            f"ambiguous={sorted(self._ambiguous)!r}, "
            f"exhausted={dict(sorted(self._exhausted.items()))!r})"
        )


def compile_meaning_actions(
    rules: Iterable[Rule[Term]],
    supplied: Mapping[str, Term] | None = None,
    *,
    erasure: StructuralErasure,
    calculus: ProofCalculus,
    normalization: Normalization,
    search: ProofSearch,
    selection: MeaningSelection,
) -> MeaningActions:
    """Compile one semantic-action program for every covered rule."""
    held_rules = tuple(rules)
    overrides = {} if supplied is None else dict(supplied)
    by_name: dict[str, Rule[Term]] = {}

    for rule in held_rules:
        if rule.name in by_name:
            msg = f"semantic-action compilation received rule {rule.name!r} twice"
            raise ValueError(msg)

        by_name[rule.name] = rule

    unknown = sorted(set(overrides) - set(by_name))

    if unknown:
        msg = f"semantic actions name rules that are absent: {unknown!r}"
        raise ValueError(msg)

    for name, action in overrides.items():
        if not isinstance(action, Term) or not calculus.store.contains(action):
            msg = f"the supplied action of rule {name!r} is foreign to the calculus"
            raise ValueError(msg)

        hypotheses = _hypotheses(calculus, by_name[name].arity)
        expected = 0

        for hypothesis in hypotheses:
            if hypothesis.payload is None:
                msg = "the proof calculus returned a malformed hypothesis"
                raise ValueError(msg)

            expected |= 1 << hypothesis.payload

        if action.free != expected:
            msg = (
                f"the supplied action of rule {name!r} has hypothesis mask "
                f"{action.free}, not {expected} for arity {by_name[name].arity}"
            )
            raise ValueError(msg)

    programs: dict[str, _ActionProgram] = {}
    uncovered: set[str] = set()
    ambiguous: set[str] = set()
    exhausted_rules: dict[str, frozenset[Exhaustion]] = {}

    for rule in held_rules:
        name = rule.name

        if name in overrides:
            programs[name] = _SelectedAction((_SelectedCase(None, (overrides[name],)),))
            continue

        if isinstance(rule, ProofSchema) and rule.proof_schemata:
            orientations = derive_action(
                rule,
                erasure,
                calculus=calculus,
                normalization=normalization,
                search=search,
            )
        elif isinstance(rule, ElaboratedSchema):
            programs[name] = _WitnessAction(rule, selection)

            if not rule.witness_is_unique:
                ambiguous.add(name)

            continue
        else:
            uncovered.add(name)
            continue

        grouped: dict[Sequent, list[Term]] = {}
        grouped_exhausted: dict[Sequent, set[Exhaustion]] = {}

        for orientation in orientations:
            grouped.setdefault(orientation.sequent, []).extend(
                orientation.inhabitants.terms
            )
            grouped_exhausted.setdefault(orientation.sequent, set()).update(
                orientation.inhabitants.exhausted
            )

        exhausted = frozenset(
            bound
            for orientation in orientations
            for bound in orientation.inhabitants.exhausted
        )

        if exhausted:
            exhausted_rules[name] = exhausted

        selected_cases: list[_SelectedCase] = []

        if not orientations:
            uncovered.add(name)

        for sequent, terms in grouped.items():
            found = tuple(dict.fromkeys(terms))
            bounded = frozenset(grouped_exhausted[sequent])

            if len(found) > 1:
                ambiguous.add(name)

            if bounded and selection.kind != "first":
                uncovered.add(name)
                continue

            selected = selection.select(found)

            if not selected:
                uncovered.add(name)
                continue

            selected_cases.append(_SelectedCase(sequent, selected))

        overlaps = any(
            _sequents_overlap(left.sequent, right.sequent)
            for at, left in enumerate(selected_cases)
            for right in selected_cases[at + 1 :]
            if left.sequent is not None and right.sequent is not None
        )

        if overlaps:
            ambiguous.add(name)
            uncovered.add(name)
        elif selected_cases:
            programs[name] = _SelectedAction(selected_cases)

    return MeaningActions(
        programs,
        erasure,
        calculus,
        uncovered,
        ambiguous,
        exhausted_rules,
    )


def _term_instance(pattern: Term, concrete: Term, bindings: dict[int, Term]) -> bool:
    """Return whether a concrete type term instantiates one schematic type term."""
    if pattern.op.is_free_variable and pattern.payload is not None:
        bound = bindings.get(pattern.payload)

        if bound is None:
            bindings[pattern.payload] = concrete
            return True

        return bound is concrete

    return (
        pattern.op is concrete.op
        and pattern.payload == concrete.payload
        and len(pattern.args) == len(concrete.args)
        and all(
            _term_instance(left, right, bindings)
            for left, right in zip(pattern.args, concrete.args, strict=True)
        )
    )


def _sequent_instance(pattern: Sequent, concrete: Sequent) -> bool:
    """Return whether one concrete erased step instantiates a schematic sequent."""
    pattern_premises, pattern_conclusion = pattern
    concrete_premises, concrete_conclusion = concrete

    if len(pattern_premises) != len(concrete_premises):
        return False

    bindings: dict[int, Term] = {}
    return all(
        _term_instance(left, right, bindings)
        for left, right in zip(
            (*pattern_premises, pattern_conclusion),
            (*concrete_premises, concrete_conclusion),
            strict=True,
        )
    )


type _SidedTerm = tuple[Term, int]
type _Variable = tuple[int, int]


def _variable(value: _SidedTerm, /) -> _Variable | None:
    """Return a side-qualified schematic-variable identity, if present."""
    term, side = value
    return (
        (side, term.payload)
        if term.op.is_free_variable and term.payload is not None
        else None
    )


def _walk(value: _SidedTerm, bindings: dict[_Variable, _SidedTerm], /) -> _SidedTerm:
    """Follow unifier bindings to one current term."""
    current = value
    variable = _variable(current)

    while variable is not None and variable in bindings:
        current = bindings[variable]
        variable = _variable(current)

    return current


def _occurs(
    variable: _Variable,
    value: _SidedTerm,
    bindings: dict[_Variable, _SidedTerm],
    /,
) -> bool:
    """Return whether a variable occurs in a partly unified term."""
    term, side = _walk(value, bindings)
    found = _variable((term, side))

    if found is not None:
        return found == variable

    return any(_occurs(variable, (child, side), bindings) for child in term.args)


def _unify(
    left: _SidedTerm,
    right: _SidedTerm,
    bindings: dict[_Variable, _SidedTerm],
    /,
) -> bool:
    """Unify two side-qualified schematic terms in place."""
    held_left = _walk(left, bindings)
    held_right = _walk(right, bindings)
    left_variable = _variable(held_left)
    right_variable = _variable(held_right)

    if left_variable is not None:
        if left_variable == right_variable:
            return True

        if _occurs(left_variable, held_right, bindings):
            return False

        bindings[left_variable] = held_right
        return True

    if right_variable is not None:
        if _occurs(right_variable, held_left, bindings):
            return False

        bindings[right_variable] = held_left
        return True

    left_term, left_side = held_left
    right_term, right_side = held_right
    return (
        left_term.op is right_term.op
        and left_term.payload == right_term.payload
        and len(left_term.args) == len(right_term.args)
        and all(
            _unify((one, left_side), (other, right_side), bindings)
            for one, other in zip(left_term.args, right_term.args, strict=True)
        )
    )


def _sequents_overlap(left: Sequent, right: Sequent, /) -> bool:
    """Return whether two schematic sequents admit a common instance."""
    left_premises, left_conclusion = left
    right_premises, right_conclusion = right

    if len(left_premises) != len(right_premises):
        return False

    bindings: dict[_Variable, _SidedTerm] = {}
    return all(
        _unify((one, 0), (other, 1), bindings)
        for one, other in zip(
            (*left_premises, left_conclusion),
            (*right_premises, right_conclusion),
            strict=True,
        )
    )


def _hypotheses(
    calculus: ProofCalculus,
    arity: int,
) -> tuple[Term, ...]:
    """Validate the calculus hypotheses for an action arity."""
    store = calculus.store
    hypotheses: list[Term] = []

    for position in range(max(arity, 2)):
        hypothesis = calculus.hypothesis(position)

        if not store.contains(hypothesis) or hypothesis.payload is None:
            msg = "the proof calculus returned a foreign or malformed hypothesis"
            raise ValueError(msg)

        try:
            name = store.atom_name(hypothesis.payload)
        except ValueError as error:
            msg = "the proof calculus returned a malformed hypothesis"
            raise ValueError(msg) from error

        if (
            hypothesis is not store.fvar(hypothesis.op.sort, name)
            or calculus.hypothesis(position) is not hypothesis
        ):
            msg = "the proof calculus hypothesis channel is not stable and canonical"
            raise ValueError(msg)

        hypotheses.append(hypothesis)

    if len(set(hypotheses)) != len(hypotheses):
        msg = "the proof calculus hypothesis channel is not injective"
        raise ValueError(msg)

    return tuple(hypotheses[:arity])


def _temporary_atoms(
    calculus: ProofCalculus,
    count: int,
    occupied: int,
) -> tuple[Term, ...]:
    """Create temporary atoms absent from the action and child meanings."""
    store = calculus.store
    sort = store.signature.by_name(calculus.operators.lam).binding[0].sort
    temporary: list[Term] = []
    nonce = 0

    while len(temporary) < count:
        candidate = store.fvar(sort, f"__meaning_argument_{nonce}")
        nonce += 1

        if candidate.payload is None or occupied & (1 << candidate.payload):
            continue

        temporary.append(candidate)
        occupied |= 1 << candidate.payload

    return tuple(temporary)


def _checked_policy(policy: NormalizationPolicy) -> NormalizationPolicy:
    """Return a declared policy or refuse an undeclared one.

    Parameters
    ----------
    policy : NormalizationPolicy
        The policy to check.
    """
    if policy not in NORMALIZATION_POLICIES:
        msg = (
            f"a normalization policy is one of {list(NORMALIZATION_POLICIES)}: "
            f"{policy!r}"
        )
        raise ValueError(msg)

    return policy


def _substituted(
    action: Term,
    children: Sequence[Term],
    /,
    *,
    calculus: ProofCalculus,
) -> Term:
    """Substitute child meanings for an action's hypotheses, leaving redexes.

    Raises
    ------
    ValueError
        If a term is foreign, the hypothesis channel is invalid, or the action's
        hypotheses do not match its child arity.
    """
    store = calculus.store
    held_children = tuple(children)

    if not store.contains(action) or any(
        not store.contains(child) for child in held_children
    ):
        msg = "an action and its child meanings must share the calculus store lineage"
        raise ValueError(msg)

    hypotheses = _hypotheses(calculus, len(held_children))
    expected = 0

    for hypothesis in hypotheses:
        if hypothesis.payload is None:
            msg = "the proof calculus returned a malformed hypothesis"
            raise ValueError(msg)

        expected |= 1 << hypothesis.payload

    if action.free != expected:
        msg = "the action's open hypotheses do not match its child arity"
        raise ValueError(msg)

    occupied = action.free

    for child in held_children:
        occupied |= child.free

    temporary = _temporary_atoms(calculus, len(hypotheses), occupied)
    substituted = action

    for hypothesis, replacement in zip(hypotheses, temporary, strict=True):
        if hypothesis.payload is None:
            msg = "the proof calculus returned a malformed hypothesis"
            raise ValueError(msg)

        substituted = substitute(store, substituted, hypothesis.payload, replacement)

    for placeholder, child in zip(temporary, held_children, strict=True):
        if placeholder.payload is None:
            msg = "a temporary action atom is malformed"
            raise ValueError(msg)

        substituted = substitute(store, substituted, placeholder.payload, child)

    return substituted


def run_action(
    action: Term,
    children: Sequence[Term],
    /,
    *,
    calculus: ProofCalculus,
    normalization: Normalization,
) -> Term:
    """Substitute child meanings and beta-normalize an action.

    Raises
    ------
    ValueError
        If a term is foreign, the hypothesis channel is invalid, or the action's
        hypotheses do not match its child arity.
    """
    substituted = _substituted(action, children, calculus=calculus)

    return normalize(
        calculus.store,
        substituted,
        calculus.operators,
        normalization=normalization,
    )


@final
class MeaningReading[Category, Weight = float](FrozenValue):
    """A derivation interpretation based on semantic actions.

    Under the eager policy every rule application is beta-normalized as it is
    read, so derivations of one span that reach one meaning pack. Under the lazy
    policy an application substitutes its child meanings and keeps the redexes,
    and normalization is left to whatever consumes the meaning.

    Parameters
    ----------
    actions : MeaningActions
        The compiled action set and its proof calculus.
    normalization : Normalization
        The declared normal form and grammar-owned reduction budget.
    policy : NormalizationPolicy
        When rule applications are beta-normalized.

    Raises
    ------
    ValueError
        If the policy is undeclared.
    """

    _noun = "meaning interpretation value"

    __slots__ = ("_actions", "_normalization", "_normalizes", "_policy")

    def __init__(
        self,
        actions: MeaningActions,
        *,
        normalization: Normalization,
        policy: NormalizationPolicy,
    ) -> None:
        object.__setattr__(self, "_actions", actions)
        object.__setattr__(self, "_normalization", normalization)
        object.__setattr__(self, "_policy", _checked_policy(policy))
        object.__setattr__(self, "_normalizes", NORMALIZES_STEPS[policy])

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "meaning"

    @property
    def refines_items(self) -> bool:
        """Return whether meanings refine packed items."""
        return True

    @property
    def actions(self) -> MeaningActions:
        """Return the compiled action set."""
        return self._actions

    @property
    def policy(self) -> NormalizationPolicy:
        """Return when rule applications are beta-normalized."""
        return self._policy

    @property
    def normalization(self) -> Normalization:
        """Return the declared normal form and reduction budget."""
        return self._normalization

    def supports(self, rule: Rule[Category], /) -> bool:
        """Return whether the action set covers a rule."""
        return self._actions.is_scalar(rule.name)

    def axiom(
        self,
        entry: Entry[Category, Term],
        /,
    ) -> Term | None:
        """Return an entry's meaning."""
        meaning = entry.meaning

        if meaning is not None and not self._actions.calculus.store.contains(meaning):
            msg = "an entry meaning must share the calculus store lineage"
            raise ValueError(msg)

        return meaning

    def step(
        self,
        context: StepContext[Category, Weight],
        children: tuple[Term, ...],
        /,
    ) -> Term | None:
        """Run the action for a rule application under the declared policy."""
        rule = context.rule

        if len(children) != rule.arity:
            msg = (
                f"rule {rule.name!r} takes {rule.arity} child meanings, "
                f"not {len(children)}"
            )
            raise ValueError(msg)

        if len(context.premises) != rule.arity:
            msg = (
                f"rule {rule.name!r} takes {rule.arity} premise categories, "
                f"not {len(context.premises)}"
            )
            raise ValueError(msg)

        selected = self._actions.resolve(
            rule.name,
            cast("StepContext[Term, Weight]", context),
        )

        if not selected:
            return None

        if len(selected) != 1:
            raise AmbiguousMeaningError(rule.name, len(selected))

        (action,) = selected
        calculus = self._actions.calculus

        if not self._normalizes:
            return _substituted(action, children, calculus=calculus)

        return run_action(
            action,
            children,
            calculus=calculus,
            normalization=self._normalization,
        )

    @override
    def __repr__(self) -> str:
        """Return a representation with the action count and the policy."""
        return (
            f"MeaningReading(actions={len(self._actions)}, "
            f"policy={self._policy!r}, normalization={self._normalization!r})"
        )

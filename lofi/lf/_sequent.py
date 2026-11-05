"""Declared signed-tableau proof search over typed logical forms."""

from itertools import product
from typing import TYPE_CHECKING, Protocol, final, override, runtime_checkable

from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue
from lofi.lf._constants import ChoiceFunction
from lofi.terms import Term, instantiate, normalize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf._profile import LogicProfile
    from lofi.lf._roles import LogicalRole
    from lofi.lf._semantics import LogicalConstantSemantics
    from lofi.terms import Calculus, Normalization, TermStore

__all__ = [
    "PostulatedRule",
    "PostulatedRuleProvider",
    "SequentCalculus",
    "SequentLimits",
    "SequentProof",
    "SequentSearch",
    "SequentStep",
    "TruthTableRule",
    "finite_frame_rule_sound",
]


class _Language(Protocol):
    """The lowered-language capabilities used by guided proof search."""

    @property
    def store(self) -> TermStore: ...

    @property
    def profile(self) -> LogicProfile: ...

    @property
    def calculus(self) -> Calculus: ...

    @property
    def function_type(self) -> str: ...

    @property
    def term_sort(self) -> str: ...

    def spine(self, term: Term, /) -> tuple[Term, tuple[Term, ...]]: ...

    def type_of(self, term: Term, /) -> Term: ...

    def constant(self, name: str, /) -> Term: ...

    def apply(self, function: Term, *arguments: Term) -> Term: ...

    def print_type(self, term: Term, /) -> str: ...


@final
class TruthTableRule(FrozenValue):
    """One signed tableau expansion derived from a declared truth table."""

    _noun = "truth-table proof rule"
    __slots__ = ("_assignments", "_constant", "_sign")

    def __init__(
        self,
        constant: str,
        assignments: Sequence[tuple[bool, ...]],
        /,
        *,
        sign: bool,
    ) -> None:
        held = tuple(tuple(row) for row in assignments)

        if not isinstance(sign, bool):
            msg = f"a signed proof rule has a bool sign, not {sign!r}"
            raise TypeError(msg)

        if held and len({len(row) for row in held}) != 1:
            msg = "a truth-table proof rule has one common argument arity"
            raise ValueError(msg)

        object.__setattr__(
            self, "_constant", ensure_name("a logical constant", constant)
        )
        object.__setattr__(self, "_sign", sign)
        object.__setattr__(self, "_assignments", held)

    @classmethod
    def from_role(
        cls,
        constant: str,
        role: LogicalRole,
        /,
        *,
        sign: bool,
    ) -> TruthTableRule:
        """Derive the exact signed expansion from one logical role."""
        if not role.truth_functional:
            msg = f"logical role {role.name!r} has no finite truth table"
            raise ValueError(msg)

        assignments = tuple(
            values
            for values in product((False, True), repeat=role.arity)
            if role.truth(values) is sign
        )
        return cls(constant, assignments, sign=sign)

    @property
    def constant(self) -> str:
        return self._constant

    @property
    def sign(self) -> bool:
        return self._sign

    @property
    def assignments(self) -> tuple[tuple[bool, ...], ...]:
        return self._assignments


@final
class PostulatedRule(FrozenValue):
    """A plugin-supplied finite-frame tableau expansion."""

    _noun = "postulated proof rule"
    __slots__ = ("_assignments", "_name", "_sign")

    def __init__(
        self,
        name: str,
        assignments: Sequence[tuple[bool, ...]],
        /,
        *,
        sign: bool,
    ) -> None:
        if not isinstance(sign, bool):
            msg = f"a postulated rule has a bool sign, not {sign!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_name", ensure_name("a postulated rule", name))
        object.__setattr__(self, "_sign", sign)
        object.__setattr__(
            self, "_assignments", tuple(tuple(row) for row in assignments)
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def sign(self) -> bool:
        return self._sign

    @property
    def assignments(self) -> tuple[tuple[bool, ...], ...]:
        return self._assignments


@runtime_checkable
class PostulatedRuleProvider(Protocol):
    """Supply executable rules for one postulated logical semantics."""

    def sequent_rules(self) -> tuple[PostulatedRule, ...]:
        """Return finite-frame rules whose names are declared by ``rules``."""
        ...


class _BooleanAlgebra:
    """Lift crisp finite-frame truth values for rule validation."""

    __slots__ = ()

    def lift(self, *, value: bool) -> float:
        return float(value)


def _validated_postulated_rules(
    semantics: LogicalConstantSemantics,
    /,
) -> tuple[PostulatedRule, ...]:
    """Validate a plugin's executable rules against every Boolean frame."""
    if not isinstance(semantics, PostulatedRuleProvider):
        return ()

    rules = semantics.sequent_rules()

    if tuple(rule.name for rule in rules) != semantics.rules:
        msg = (
            f"semantics {semantics.name!r} declares rules {semantics.rules!r}, "
            f"not {tuple(rule.name for rule in rules)!r}"
        )
        raise ValueError(msg)

    for rule in rules:
        if any(len(row) != semantics.arity for row in rule.assignments):
            msg = f"postulated rule {rule.name!r} does not have arity {semantics.arity}"
            raise ValueError(msg)

        if not finite_frame_rule_sound(semantics, rule):
            msg = f"postulated rule {rule.name!r} is not sound on every Boolean frame"
            raise ValueError(msg)

    return rules


def finite_frame_rule_sound(
    semantics: LogicalConstantSemantics,
    rule: PostulatedRule,
    /,
) -> bool:
    """Check one executable postulate against the complete Boolean frame."""
    if any(len(row) != semantics.arity for row in rule.assignments):
        return False

    expected = {
        values
        for values in product((False, True), repeat=semantics.arity)
        if bool(semantics.denote(values, _BooleanAlgebra(), ChoiceFunction({})))
        is rule.sign
    }
    return set(rule.assignments) == expected


@final
class SequentLimits(FrozenValue):
    """Resource bounds for one guided proof search."""

    _noun = "sequent search limits"
    __slots__ = ("_depth", "_instantiations", "_steps", "_top_k")

    def __init__(
        self,
        *,
        steps: int,
        depth: int,
        top_k: int,
        instantiations: int,
    ) -> None:
        object.__setattr__(
            self, "_steps", ensure_positive("sequent search steps", steps)
        )
        object.__setattr__(
            self, "_depth", ensure_positive("sequent search depth", depth)
        )
        object.__setattr__(
            self, "_top_k", ensure_positive("sequent proof top-k", top_k)
        )
        object.__setattr__(
            self,
            "_instantiations",
            ensure_positive("sequent instantiation bound", instantiations),
        )

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def instantiations(self) -> int:
        return self._instantiations

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is SequentLimits and (
            self._steps,
            self._depth,
            self._top_k,
            self._instantiations,
        ) == (
            other._steps,
            other._depth,
            other._top_k,
            other._instantiations,
        )

    @override
    def __hash__(self) -> int:
        return hash(
            (
                SequentLimits,
                self._steps,
                self._depth,
                self._top_k,
                self._instantiations,
            )
        )


@final
class SequentStep(FrozenValue):
    """One declared rule application and the formula it focused."""

    _noun = "sequent proof step"
    __slots__ = ("_focus", "_rule")

    def __init__(self, rule: str, focus: Term, /) -> None:
        object.__setattr__(self, "_rule", ensure_name("a sequent rule", rule))
        object.__setattr__(self, "_focus", focus)

    @property
    def rule(self) -> str:
        return self._rule

    @property
    def focus(self) -> Term:
        return self._focus


@final
class SequentProof(FrozenValue):
    """One closed tableau, in rule-application order."""

    _noun = "sequent proof"
    __slots__ = ("_steps",)

    def __init__(self, steps: Sequence[SequentStep], /) -> None:
        object.__setattr__(self, "_steps", tuple(steps))

    @property
    def steps(self) -> tuple[SequentStep, ...]:
        return self._steps


@final
class SequentSearch(FrozenValue):
    """Bounded proof findings and an honest fragment decision."""

    _noun = "sequent search"
    __slots__ = ("_exhausted", "_proofs", "_reason", "_supported")

    def __init__(
        self,
        proofs: Sequence[SequentProof],
        /,
        *,
        supported: bool,
        exhausted: bool,
        reason: str | None,
    ) -> None:
        if not isinstance(supported, bool) or not isinstance(exhausted, bool):
            msg = "sequent supported and exhausted flags are bools"
            raise TypeError(msg)

        object.__setattr__(self, "_proofs", tuple(proofs))
        object.__setattr__(self, "_supported", supported)
        object.__setattr__(self, "_exhausted", exhausted)
        object.__setattr__(self, "_reason", reason)

    @property
    def proofs(self) -> tuple[SequentProof, ...]:
        return self._proofs

    @property
    def supported(self) -> bool:
        return self._supported

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def reason(self) -> str | None:
        return self._reason


type _Signed = tuple[Term, bool]


def _subterms(term: Term, /) -> tuple[Term, ...]:
    return (term, *(nested for argument in term.args for nested in _subterms(argument)))


def _type_order(term: Term, function_type: str, /) -> int:
    if term.op.name != function_type or len(term.args) != 2:
        return 0

    return max(
        _type_order(term.args[0], function_type) + 1,
        _type_order(term.args[1], function_type),
    )


class SequentCalculus:
    """Generate and execute signed-tableau rules from one language declaration."""

    __slots__ = (
        "_language",
        "_normalization",
        "_order",
        "_postulated",
        "_truth_rules",
    )

    def __init__(
        self,
        language: _Language,
        normalization: Normalization,
        /,
        *,
        order: str,
    ) -> None:
        if order not in {"first", "second", "higher"}:
            msg = f"a guided proof order is first, second or higher, not {order!r}"
            raise ValueError(msg)

        truth_rules: dict[tuple[str, bool], TruthTableRule] = {}
        for constant, role in language.profile.logical_roles.items():
            if role.truth_functional:
                for sign in (False, True):
                    truth_rules[constant, sign] = TruthTableRule.from_role(
                        constant, role, sign=sign
                    )

        postulated: dict[tuple[str, bool], PostulatedRule] = {}
        for constant, semantics in language.profile.logical_semantics.items():
            for rule in _validated_postulated_rules(semantics):
                postulated[constant, rule.sign] = rule

        self._language = language
        self._normalization = normalization
        self._order = order
        self._truth_rules = truth_rules
        self._postulated = postulated

    @property
    def rules(self) -> tuple[str, ...]:
        """Return the executable rule inventory generated from the language."""
        names = ["identity", "beta"]
        names.extend(f"delta:{name}" for name in self._language.profile.definitions)
        names.extend(
            f"truth:{constant}:{'true' if sign else 'false'}"
            for constant, sign in self._truth_rules
        )
        names.extend(rule.name for rule in self._postulated.values())
        names.extend(
            f"instantiate:{constant}"
            for constant, role in self._language.profile.logical_roles.items()
            if role.quantifier
        )
        return tuple(dict.fromkeys(names))

    def _normalize(self, term: Term, /) -> Term:
        return normalize(
            self._language.store,
            term,
            self._language.calculus,
            normalization=self._normalization,
        )

    def _logical(self, term: Term, /) -> tuple[str, tuple[Term, ...]] | None:
        head, arguments = self._language.spine(term)
        if not head.op.is_constant:
            return None
        name = self._language.store.payload_name(head)
        if name not in self._language.profile.logical_constants:
            return None
        return name, arguments

    def _abstraction(self, term: Term, /) -> tuple[Term, Term, Term] | None:
        logical = self._logical(term)
        if logical is None:
            return None
        name, arguments = logical
        role = self._language.profile.logical_roles.get(name)
        if role is None or not role.quantifier or len(arguments) != 1:
            return None
        abstraction = arguments[0]
        if not abstraction.op.is_binder or len(abstraction.op.binding) != 1:
            return None
        site = abstraction.op.binding[0]
        if len(site.arguments) != 1:
            return None
        body = abstraction.args[site.arguments[0]]
        abstraction_type = self._language.type_of(abstraction)
        if (
            abstraction_type.op.name != self._language.function_type
            or len(abstraction_type.args) != 2
        ):
            return None
        return abstraction, body, abstraction_type.args[0]

    def _contains_bound(self, term: Term, index: int, /) -> bool:
        if term.op.is_bound_variable and term.payload == index:
            return True
        return any(self._contains_bound(argument, index) for argument in term.args)

    def _pattern_body(self, body: Term, /) -> bool:
        def visit(node: Term) -> bool:
            head, arguments = self._language.spine(node)
            if head.op.is_bound_variable and head.payload == 0:
                return bool(arguments) and all(
                    not self._contains_bound(argument, 0) for argument in arguments
                )
            if node.op.is_bound_variable and node.payload == 0:
                return False
            return all(visit(argument) for argument in node.args)

        return visit(body)

    def _supported(self, term: Term, /) -> str | None:
        for node in _subterms(term):
            logical = self._logical(node)
            if logical is None:
                continue
            name, arguments = logical
            semantics = self._language.profile.logical_semantics.get(name)
            if semantics is not None and (name, True) not in self._postulated:
                return f"postulated constant {name!r} supplies no executable rule"
            role = self._language.profile.logical_roles.get(name)
            binder = self._abstraction(node)
            if role is not None and role.quantifier and arguments and binder is None:
                return f"quantifier {name!r} needs one declared abstraction body"
            if binder is None:
                continue
            _, body, quantified_type = binder
            rank = _type_order(quantified_type, self._language.function_type)
            if rank > 0 and self._order == "first":
                return (
                    "first-order guided proof cannot serve a second-order "
                    f"binder of type {self._language.print_type(quantified_type)!r}"
                )
            if rank > 1 and self._order == "second":
                return (
                    "second-order guided proof cannot serve a higher-order "
                    f"binder of type {self._language.print_type(quantified_type)!r}"
                )
            if rank > 0 and not self._pattern_body(body):
                return (
                    "the quantified functional variable is not a higher-order pattern"
                )
        return None

    def _candidates(
        self,
        quantified_type: Term,
        state: tuple[_Signed, ...],
        /,
        *,
        limit: int,
    ) -> tuple[Term, ...]:
        candidates: list[Term] = []

        def offer(term: Term) -> None:
            if (
                term.is_closed
                and term not in candidates
                and self._language.type_of(term) == quantified_type
            ):
                candidates.append(term)

        for formula, _ in state:
            for term in _subterms(formula):
                offer(term)

        vocabulary = self._language.store.signature.vocabulary(self._language.term_sort)
        for name in vocabulary.names:
            try:
                offer(self._language.constant(name))
            except ValueError:
                continue

        return tuple(candidates[:limit])

    def _delta(self, term: Term, /) -> tuple[str, Term] | None:
        logical = self._logical(term)
        if logical is None:
            return None
        name, arguments = logical
        definition = self._language.profile.definitions.get(name)
        if definition is None:
            return None
        return name, self._normalize(self._language.apply(definition, *arguments))

    def search(
        self,
        premise: Term,
        goal: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        limits: SequentLimits,
    ) -> SequentSearch:
        """Search for closed signed tableaux within the declared bounds."""
        normalized_theory = tuple(self._normalize(term) for term in theory)
        normalized_premise = self._normalize(premise)
        normalized_goal = self._normalize(goal)
        beta_steps = tuple(
            SequentStep("beta", original)
            for original, reduced in zip(
                (*theory, premise, goal),
                (*normalized_theory, normalized_premise, normalized_goal),
                strict=True,
            )
            if original != reduced
        )
        state = (
            *((term, True) for term in normalized_theory),
            (normalized_premise, True),
            (normalized_goal, False),
        )
        reason = next(
            (held for term, _ in state if (held := self._supported(term)) is not None),
            None,
        )
        if reason is not None:
            return SequentSearch((), supported=False, exhausted=False, reason=reason)

        if len(beta_steps) > limits.steps:
            return SequentSearch((), supported=True, exhausted=True, reason=None)

        steps_used = len(beta_steps)
        exhausted = False

        def close_all(
            rows: tuple[_Signed, ...],
            focus: int,
            branches: tuple[tuple[_Signed, ...], ...],
            depth: int,
            step: SequentStep,
        ) -> tuple[tuple[SequentStep, ...], ...]:
            nonlocal exhausted, steps_used

            if steps_used >= limits.steps:
                exhausted = True
                return ()
            steps_used += 1

            combinations: list[tuple[SequentStep, ...]] = [()]
            for added in branches:
                following = (*rows[:focus], *added, *rows[focus + 1 :])
                child = apply(following, depth + 1)
                if not child:
                    return ()

                combined: list[tuple[SequentStep, ...]] = []
                for prefix in combinations:
                    for proof in child:
                        combined.append((*prefix, *proof))
                        if len(combined) >= limits.top_k:
                            break
                    if len(combined) >= limits.top_k:
                        break
                combinations = combined

            return tuple((step, *proof) for proof in combinations)

        def apply(
            rows: tuple[_Signed, ...],
            depth: int,
        ) -> tuple[tuple[SequentStep, ...], ...]:
            nonlocal exhausted, steps_used

            seen: dict[Term, bool] = {}
            for formula, sign in rows:
                opposite = seen.get(formula)
                if opposite is not None and opposite is not sign:
                    return ((SequentStep("identity", formula),),)
                seen[formula] = sign

            if depth >= limits.depth:
                exhausted = True
                return ()

            focus = -1
            branches: tuple[tuple[_Signed, ...], ...] = ()
            rule_name: str | None = None
            require_all = False

            for position, (formula, sign) in enumerate(rows):
                delta = self._delta(formula)
                if delta is not None:
                    name, unfolded = delta
                    focus = position
                    branches = (((unfolded, sign),),)
                    rule_name = f"delta:{name}"
                    require_all = True
                    break

                logical = self._logical(formula)
                if logical is None:
                    continue
                name, arguments = logical
                rule = self._truth_rules.get((name, sign))
                if rule is not None:
                    focus = position
                    branches = tuple(
                        tuple(zip(arguments, assignment, strict=True))
                        for assignment in rule.assignments
                    )
                    rule_name = f"truth:{name}:{'true' if sign else 'false'}"
                    require_all = True
                    break

                postulated = self._postulated.get((name, sign))
                if postulated is not None:
                    focus = position
                    branches = tuple(
                        tuple(zip(arguments, assignment, strict=True))
                        for assignment in postulated.assignments
                    )
                    rule_name = postulated.name
                    require_all = True
                    break

                binder = self._abstraction(formula)
                role = self._language.profile.logical_roles.get(name)
                if binder is None or role is None:
                    continue
                operation = role.model_operation
                if (operation == "universal" and sign) or (
                    operation == "existential" and not sign
                ):
                    _, body, quantified_type = binder
                    candidates = self._candidates(
                        quantified_type,
                        rows,
                        limit=limits.instantiations,
                    )
                    focus = position
                    branches = tuple(
                        ((instantiate(self._language.store, body, candidate), sign),)
                        for candidate in candidates
                    )
                    rule_name = f"instantiate:{name}"
                    break

            if focus < 0 or rule_name is None or (not branches and not require_all):
                return ()

            original = rows[focus][0]
            step = SequentStep(rule_name, original)

            if require_all:
                return close_all(rows, focus, branches, depth, step)

            proofs: list[tuple[SequentStep, ...]] = []

            for added in branches:
                if steps_used >= limits.steps:
                    exhausted = True
                    break
                steps_used += 1
                following = (*rows[:focus], *added, *rows[focus + 1 :])
                child = apply(following, depth + 1)
                for proof in child:
                    proofs.append((step, *proof))
                    if len(proofs) >= limits.top_k:
                        break
                if len(proofs) >= limits.top_k:
                    break

            return tuple(proofs)

        proofs = tuple(SequentProof((*beta_steps, *steps)) for steps in apply(state, 0))
        return SequentSearch(
            proofs,
            supported=True,
            exhausted=exhausted,
            reason=None,
        )

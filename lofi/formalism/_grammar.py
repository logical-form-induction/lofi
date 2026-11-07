"""Runtime grammar compilation and fingerprints."""

from hashlib import sha256
from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.deduce import NamedRoots
from lofi.formalism._profile import DeductionProfile
from lofi.judge import deduction_requirements

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import Goal
    from lofi.formalism._profile import ProfileBound
    from lofi.formalism._values import (
        Budget,
        FormalismDeclaration,
        TheoryArtifact,
    )
    from lofi.interpret import InterpretationDeclaration
    from lofi.judge import (
        CapacityReport,
        ClosureReport,
        DeductionRequirements,
        NormalForm,
        RuleSet,
    )
    from lofi.lexicon import Clamp
    from lofi.terms import Matching, Signature, TargetSet, Term, TermStore

__all__ = ["Formalism", "Grammar"]


@final
class _GrammarConstruction:
    """Compiler-owned grammar construction authority."""

    __slots__ = ()


_GRAMMAR_CONSTRUCTION = _GrammarConstruction()


def _term_spec(term: Term) -> JsonValue:
    """Return a term as a JSON-shaped declaration."""
    return [
        term.op.name,
        term.payload,
        [_term_spec(argument) for argument in term.args],
    ]


def _clamp_spec(clamp: Clamp[Term]) -> JsonValue:
    """Return a clamp as a JSON-shaped declaration."""
    categories = sorted(
        (_term_spec(category) for category in clamp.category),
        key=lambda value: dumps(value, separators=(",", ":"), sort_keys=True),
    )

    return {
        "span": list(clamp.span),
        "categories": categories,
        "meaning": None if clamp.meaning is None else _term_spec(clamp.meaning),
        "mode": clamp.mode.value,
        "penalty": clamp.penalty,
        "scope": clamp.scope.value,
        "require_constituent": clamp.require_constituent,
        "negate": clamp.negate,
    }


def _validate_targets(
    name: str,
    store: TermStore,
    rules: RuleSet[Term],
    targets: TargetSet[Term],
    closure: ClosureReport[Term],
) -> None:
    """Validate declared targets against the runtime grammar.

    Parameters
    ----------
    name : str
        Formalism name used in diagnostics.
    store : TermStore
        Store that must own each target.
    rules : RuleSet[Term]
        Rule inventory that supplies the matching.
    targets : TargetSet[Term]
        Nonempty target declaration to validate.
    closure : ClosureReport[Term]
        Reachability result that must contain each target class.

    Raises
    ------
    ValueError
        If the target declaration is invalid for the runtime grammar.
    """
    if not targets.categories:
        msg = f"formalism {name!r} declares an empty target set"
        raise ValueError(msg)

    if targets.matching is not rules.matching:
        msg = (
            f"formalism {name!r} target matching "
            f"{targets.matching_name!r} is not its rule matching "
            f"{rules.matching.name!r} by identity"
        )
        raise ValueError(msg)

    for target in targets.categories:
        if not store.contains(target):
            msg = f"target {target!r} is outside formalism {name!r}'s store lineage"
            raise ValueError(msg)

        if not any(
            rules.matching.equivalent(target, reached) for reached in closure.reached
        ):
            msg = f"target {target!r} is unreachable in closure {closure!r}"
            raise ValueError(msg)


def _fingerprint(formalism: Formalism) -> str:
    """Return the stable grammar fingerprint."""
    rules = formalism.rules
    closure = formalism.closure
    profile = formalism.profile
    budget = formalism.budget
    theory = formalism.theory
    rule_specs: list[JsonValue] = [dict(rule.to_spec()) for rule in rules]
    payload = {
        "name": formalism.name,
        "signature": formalism.signature.digest,
        "rules": {
            "name": rules.name,
            "matching": rules.matching.name,
            "normal_form": rules.normal_form.name,
            "normal_form_alphabet": sorted(rules.normal_form.alphabet),
            "restriction_mode": rules.restriction_mode,
            "inventory": rule_specs,
        },
        "closure": {
            "category_set": closure.category_set,
            "reached": [_term_spec(term) for term in closure.reached],
            "escaping": [_term_spec(term) for term in closure.escaping],
        },
        "profile": {
            "fanout": profile.fanout,
            "ranks": sorted(profile.ranks),
            "unary_closure": profile.unary_closure,
            "category_set": profile.category_set,
            "lexicalized": profile.lexicalized,
        },
        "budget": budget.to_spec(),
        "capacity": {
            "result": formalism.capacity.result,
            "theorem_applicable": formalism.capacity.theorem_applicable,
            "rule_family": formalism.capacity.rule_family,
            "restricted_rules": list(formalism.capacity.restricted_rules),
            "properly_includes_context_free": (
                formalism.capacity.properly_includes_context_free
            ),
            "parikh_context_free_sublanguage": (
                formalism.capacity.parikh_context_free_sublanguage
            ),
            "citation": formalism.capacity.citation,
            "reason": formalism.capacity.reason,
        },
        "clamps": [_clamp_spec(clamp) for clamp in formalism.clamps],
        "targets": {
            "categories": [_term_spec(term) for term in formalism.targets.categories],
            "matching": formalism.targets.matching_name,
        },
        "goal": dict(formalism.goal.settings),
        "interpretations": [
            dict(interpretation.settings)
            for interpretation in formalism.interpretations.values()
        ],
        "theory": {"version": theory.version, "digest": theory.digest},
    }
    encoded = dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    return sha256(encoded.encode()).hexdigest()


@final
class Grammar(FrozenValue):
    """A compiled run-scoped grammar."""

    _noun = "formalism value"

    __slots__ = (
        "_budget",
        "_capacity",
        "_clamps",
        "_closure",
        "_fingerprint",
        "_goal",
        "_interpretations",
        "_matching",
        "_name",
        "_normal_form",
        "_profile",
        "_rules",
        "_signature",
        "_store",
        "_targets",
        "_theory",
    )

    def __init__(
        self,
        source: Formalism,
        construction: _GrammarConstruction | None = None,
    ) -> None:
        if construction is not _GRAMMAR_CONSTRUCTION:
            msg = "construct a grammar with Formalism.compile()"
            raise TypeError(msg)

        rules = source.rules
        object.__setattr__(self, "_name", source.name)
        object.__setattr__(self, "_signature", source.signature)
        object.__setattr__(self, "_store", source.store)
        object.__setattr__(self, "_matching", rules.matching)
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "_normal_form", rules.normal_form)
        object.__setattr__(self, "_closure", source.closure)
        object.__setattr__(self, "_targets", source.targets)
        object.__setattr__(self, "_goal", source.goal)
        object.__setattr__(self, "_interpretations", source.interpretations)
        object.__setattr__(self, "_profile", source.profile)
        object.__setattr__(self, "_budget", source.budget)
        object.__setattr__(self, "_capacity", source.capacity)
        object.__setattr__(self, "_clamps", source.clamps)
        object.__setattr__(self, "_theory", source.theory)
        object.__setattr__(self, "_fingerprint", source.fingerprint)

    @property
    def name(self) -> str:
        """Return the formalism name."""
        return self._name

    @property
    def signature(self) -> Signature:
        """Return the native term signature."""
        return self._signature

    @property
    def store(self) -> TermStore:
        """Return the shared root term store."""
        return self._store

    @property
    def matching(self) -> Matching[Term]:
        """Return the shared term congruence."""
        return self._matching

    @property
    def rules(self) -> RuleSet[Term]:
        """Return the native rule inventory."""
        return self._rules

    @property
    def normal_form(self) -> NormalForm[Term]:
        """Return the derivation-step filter."""
        return self._normal_form

    @property
    def closure(self) -> ClosureReport[Term]:
        """Return the native structural closure report."""
        return self._closure

    @property
    def targets(self) -> TargetSet[Term]:
        """Return the declared parse targets."""
        return self._targets

    @property
    def goal(self) -> Goal[Term]:
        """Return the parse goal."""
        return self._goal

    @property
    def interpretations(self) -> Mapping[str, InterpretationDeclaration]:
        """Return the grammar's named derivation interpretations."""
        return self._interpretations

    @property
    def profile(self) -> DeductionProfile:
        """Return the derived deduction profile."""
        return self._profile

    @property
    def budget(self) -> Budget:
        """Return the approximation budget."""
        return self._budget

    @property
    def capacity(self) -> CapacityReport:
        """Return the checked language-capacity report."""
        return self._capacity

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        """Return the run-scoped clamps in declaration order."""
        return self._clamps

    @property
    def theory_version(self) -> str:
        """Return the theory format version."""
        return self._theory.version

    @property
    def theory_specification(self) -> bytes:
        """Return the canonical theory specification."""
        return self._theory.specification

    @property
    def theory_digest(self) -> str:
        """Return the canonical theory digest."""
        return self._theory.digest

    @property
    def fingerprint(self) -> str:
        """Return the stable grammar identity."""
        return self._fingerprint


@final
class Formalism(FrozenValue):
    """A native grammar declaration and compiler."""

    _noun = "formalism value"

    __slots__ = (
        "_compiled",
        "_declaration",
        "_fingerprint",
        "_goal",
        "_interpretations",
        "_name",
        "_profile",
        "_requirements",
        "_rules",
        "_store",
        "_targets",
    )

    def __init__(
        self,
        name: str,
        store: TermStore,
        rules: RuleSet[Term],
        targets: TargetSet[Term],
        goal: Goal[Term],
        declaration: FormalismDeclaration,
        *,
        interpretations: Sequence[InterpretationDeclaration] = (),
    ) -> None:
        held_name = ensure_name("a formalism's name", name)
        closure = declaration.closure

        for term in (*closure.reached, *closure.escaping):
            if not store.contains(term):
                msg = "a closure term is outside the formalism's store lineage"
                raise ValueError(msg)

        for clamp in declaration.clamps:
            for term in clamp.category:
                if not store.contains(term):
                    msg = "a clamp category is outside the formalism's store lineage"
                    raise ValueError(msg)

            if clamp.meaning is not None and not store.contains(clamp.meaning):
                msg = "a clamp meaning is outside the formalism's store lineage"
                raise ValueError(msg)

        _validate_targets(held_name, store, rules, targets, closure)

        if isinstance(goal, NamedRoots) and (
            goal.matching is not targets.matching
            or goal.categories != targets.categories
        ):
            msg = (
                f"formalism {held_name!r}'s named-root goal does not retain its "
                "declared target set"
            )
            raise ValueError(msg)

        requirements = deduction_requirements(rules, closure)
        profile = DeductionProfile.derive(
            requirements,
            fanout=declaration.fanout,
            lexicalized=declaration.lexicalized,
        )

        if profile.unary_closure and declaration.budget.unary_rounds == 0:
            msg = "a unary rule inventory requires at least one unary-closure round"
            raise ValueError(msg)

        named_interpretations = {
            interpretation.name: interpretation for interpretation in interpretations
        }

        if len(named_interpretations) != len(interpretations):
            msg = "a formalism declares one interpretation per name"
            raise ValueError(msg)

        object.__setattr__(self, "_name", held_name)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "_goal", goal)
        object.__setattr__(
            self,
            "_interpretations",
            MappingProxyType(named_interpretations),
        )
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_requirements", requirements)
        object.__setattr__(self, "_fingerprint", _fingerprint(self))
        object.__setattr__(self, "_compiled", None)

    @property
    def name(self) -> str:
        """Return the formalism name."""
        return self._name

    @property
    def signature(self) -> Signature:
        """Return the native term signature."""
        return self._store.signature

    @property
    def store(self) -> TermStore:
        """Return the shared root term store."""
        return self._store

    @property
    def rules(self) -> RuleSet[Term]:
        """Return the native rule inventory."""
        return self._rules

    @property
    def targets(self) -> TargetSet[Term]:
        """Return the declared parse targets."""
        return self._targets

    @property
    def goal(self) -> Goal[Term]:
        """Return the parse goal."""
        return self._goal

    @property
    def interpretations(self) -> Mapping[str, InterpretationDeclaration]:
        """Return the named derivation-interpretation declarations."""
        return self._interpretations

    @property
    def closure(self) -> ClosureReport[Term]:
        """Return the native structural closure report."""
        return self._declaration.closure

    @property
    def theory(self) -> TheoryArtifact:
        """Return the checked theory artifact."""
        return self._declaration.theory

    @property
    def budget(self) -> Budget:
        """Return the approximation budget."""
        return self._declaration.budget

    @property
    def capacity(self) -> CapacityReport:
        """Return the checked language-capacity report."""
        return self._declaration.capacity

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        """Return clamps in declaration order."""
        return self._declaration.clamps

    @property
    def profile(self) -> DeductionProfile:
        """Return the derived deduction profile."""
        return self._profile

    @property
    def requirements(self) -> DeductionRequirements:
        """Return the judge-owned deduction requirements."""
        return self._requirements

    @property
    def fingerprint(self) -> str:
        """Return the prospective grammar fingerprint."""
        return self._fingerprint

    def compile(
        self,
        bounds: Sequence[ProfileBound] = (),
        /,
    ) -> Grammar:
        """Check each supplied consumer bound and return the shared grammar."""
        for bound in bounds:
            bound.check(self._profile, self._name)

        held = self._compiled

        if held is None:
            held = Grammar(self, _GRAMMAR_CONSTRUCTION)
            object.__setattr__(self, "_compiled", held)

        return held

"""Judge-native deduction and CCG-capacity evidence."""

from typing import TYPE_CHECKING, Final, Never, Protocol, override, runtime_checkable

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.judge._values import (
    CAPACITY_RESULTS,
    CATEGORY_SETS,
    RESTRICTION_MODES,
)
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lofi.judge._ports import Rule
    from lofi.judge._ruleset import RuleSet
    from lofi.judge._values import (
        CapacityResult,
        CategorySet,
        ClosureReport,
        RestrictionMode,
    )

__all__ = [
    "PREFIX_CLOSED_VW_CCG",
    "PURE_CCG",
    "CapacityEvidence",
    "CapacityReport",
    "CapacityTheorem",
    "DeductionRequirements",
    "capacity_report",
    "deduction_requirements",
]

type _RequirementFields = tuple[
    tuple[int, ...],
    bool,
    CategorySet,
    RestrictionMode,
]
"""Fields needed to reconstruct deduction requirements."""

type _RequirementReduction = tuple[
    Callable[[_RequirementFields], DeductionRequirements],
    tuple[_RequirementFields],
]
"""Deduction-requirement pickle reduction."""


class DeductionRequirements:
    """Judge-owned deduction requirements."""

    __slots__ = ("_category_set", "_ranks", "_restriction_mode", "_unary_closure")

    def __init__(
        self,
        ranks: Iterable[int],
        *,
        unary_closure: bool,
        category_set: CategorySet,
        restriction_mode: RestrictionMode,
    ) -> None:
        held_ranks = frozenset(
            ensure_non_negative("a deduction rule rank", rank) for rank in ranks
        )

        if not isinstance(unary_closure, bool):
            msg = f"unary closure is not a boolean: {unary_closure!r}"
            raise TypeError(msg)

        if unary_closure != (1 in held_ranks):
            msg = "unary closure must agree with the presence of rank-one rules"
            raise ValueError(msg)

        if category_set not in CATEGORY_SETS:
            declared = ", ".join(CATEGORY_SETS)
            msg = f"category-set verdict {category_set!r} is not one of {declared}"
            raise ValueError(msg)

        if restriction_mode not in RESTRICTION_MODES:
            declared = ", ".join(RESTRICTION_MODES)
            msg = f"restriction mode {restriction_mode!r} is not one of {declared}"
            raise ValueError(msg)

        object.__setattr__(self, "_ranks", held_ranks)
        object.__setattr__(self, "_unary_closure", unary_closure)
        object.__setattr__(self, "_category_set", category_set)
        object.__setattr__(self, "_restriction_mode", restriction_mode)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "deduction requirements are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "deduction requirements are immutable"

        raise AttributeError(msg)

    @property
    def ranks(self) -> frozenset[int]:
        """Return the declared rule ranks."""
        return self._ranks

    @property
    def unary_closure(self) -> bool:
        """Return whether rank-one rules require local closure."""
        return self._unary_closure

    @property
    def category_set(self) -> CategorySet:
        """Return the structural closure verdict."""
        return self._category_set

    @property
    def restriction_mode(self) -> RestrictionMode:
        """Return the rule instance restriction mode."""
        return self._restriction_mode

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every requirement field."""
        if type(other) is not DeductionRequirements:
            return NotImplemented

        return (
            self._ranks == other._ranks
            and self._unary_closure == other._unary_closure
            and self._category_set == other._category_set
            and self._restriction_mode == other._restriction_mode
        )

    @override
    def __hash__(self) -> int:
        """Hash every requirement field."""
        return hash(
            (
                DeductionRequirements,
                self._ranks,
                self._unary_closure,
                self._category_set,
                self._restriction_mode,
            )
        )

    @override
    def __reduce__(self) -> _RequirementReduction:
        """Return the reconstruction call."""
        fields: _RequirementFields = (
            tuple(sorted(self._ranks)),
            self._unary_closure,
            self._category_set,
            self._restriction_mode,
        )

        return (_restore_requirements, (fields,))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"DeductionRequirements({self._ranks!r}, "
            f"unary_closure={self._unary_closure!r}, "
            f"category_set={self._category_set!r}, "
            f"restriction_mode={self._restriction_mode!r})"
        )


def _restore_requirements(fields: _RequirementFields) -> DeductionRequirements:
    """Reconstruct deduction requirements from frozen fields."""
    ranks, unary_closure, category_set, restriction_mode = fields

    return DeductionRequirements(
        ranks,
        unary_closure=unary_closure,
        category_set=category_set,
        restriction_mode=restriction_mode,
    )


def deduction_requirements[T](
    ruleset: RuleSet[T], closure: ClosureReport[T]
) -> DeductionRequirements:
    """Derive judge-owned requirements without inferring lexicalization."""
    ranks = frozenset(
        ensure_non_negative(f"rule {rule.name!r} rank", rule.arity) for rule in ruleset
    )

    return DeductionRequirements(
        ranks,
        unary_closure=1 in ranks,
        category_set=closure.category_set,
        restriction_mode=ruleset.restriction_mode,
    )


type _EvidenceFields = tuple[str, bool, bool, bool]
"""Fields needed to reconstruct capacity evidence."""

type _EvidenceReduction = tuple[
    Callable[[_EvidenceFields], CapacityEvidence],
    tuple[_EvidenceFields],
]
"""Capacity-evidence pickle reduction."""


class CapacityEvidence:
    """CCG-family theorem hypotheses from a grammar boundary."""

    __slots__ = (
        "_application_target_restrictions",
        "_bounded_degree",
        "_prefix_closed",
        "_rule_family",
    )

    def __init__(
        self,
        rule_family: str,
        *,
        bounded_degree: bool,
        prefix_closed: bool,
        application_target_restrictions: bool,
    ) -> None:
        rule_family = ensure_name("a CCG rule family", rule_family)

        for field, value in (
            ("bounded degree", bounded_degree),
            ("prefix closure", prefix_closed),
            ("application-target restrictions", application_target_restrictions),
        ):
            if not isinstance(value, bool):
                msg = f"{field} evidence is not a boolean: {value!r}"
                raise TypeError(msg)

        object.__setattr__(self, "_rule_family", rule_family)
        object.__setattr__(self, "_bounded_degree", bounded_degree)
        object.__setattr__(self, "_prefix_closed", prefix_closed)
        object.__setattr__(
            self,
            "_application_target_restrictions",
            application_target_restrictions,
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "capacity evidence is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "capacity evidence is immutable"

        raise AttributeError(msg)

    @property
    def rule_family(self) -> str:
        """Return the attested CCG rule family."""
        return self._rule_family

    @property
    def bounded_degree(self) -> bool:
        """Return whether composition degree has a fixed bound."""
        return self._bounded_degree

    @property
    def prefix_closed(self) -> bool:
        """Return whether the valid rule instances are prefix-closed."""
        return self._prefix_closed

    @property
    def application_target_restrictions(self) -> bool:
        """Return whether application admits primary-target restrictions."""
        return self._application_target_restrictions

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every evidence field."""
        if type(other) is not CapacityEvidence:
            return NotImplemented

        return (
            self._rule_family == other._rule_family
            and self._bounded_degree == other._bounded_degree
            and self._prefix_closed == other._prefix_closed
            and self._application_target_restrictions
            == other._application_target_restrictions
        )

    @override
    def __hash__(self) -> int:
        """Hash every evidence field."""
        return hash(
            (
                CapacityEvidence,
                self._rule_family,
                self._bounded_degree,
                self._prefix_closed,
                self._application_target_restrictions,
            )
        )

    @override
    def __reduce__(self) -> _EvidenceReduction:
        """Return the reconstruction call."""
        fields: _EvidenceFields = (
            self._rule_family,
            self._bounded_degree,
            self._prefix_closed,
            self._application_target_restrictions,
        )

        return (_restore_capacity_evidence, (fields,))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"CapacityEvidence({self._rule_family!r}, "
            f"bounded_degree={self._bounded_degree!r}, "
            f"prefix_closed={self._prefix_closed!r}, "
            "application_target_restrictions="
            f"{self._application_target_restrictions!r})"
        )


def _restore_capacity_evidence(fields: _EvidenceFields) -> CapacityEvidence:
    """Reconstruct capacity evidence from frozen fields."""
    rule_family, bounded_degree, prefix_closed, target_restrictions = fields

    return CapacityEvidence(
        rule_family,
        bounded_degree=bounded_degree,
        prefix_closed=prefix_closed,
        application_target_restrictions=target_restrictions,
    )


type _ReportFields = tuple[
    CapacityResult,
    bool,
    str | None,
    tuple[str, ...],
    bool | None,
    bool | None,
    str | None,
    str,
]
"""Fields needed to reconstruct a capacity report."""

type _ReportReduction = tuple[
    Callable[[_ReportFields], CapacityReport],
    tuple[_ReportFields],
]
"""Capacity-report pickle reduction."""


class CapacityReport:
    """Capacity theorem and applicability report."""

    __slots__ = (
        "_citation",
        "_parikh_context_free_sublanguage",
        "_properly_includes_context_free",
        "_reason",
        "_restricted_rules",
        "_result",
        "_rule_family",
        "_theorem_applicable",
    )

    def __init__(
        self,
        result: CapacityResult,
        *,
        theorem_applicable: bool,
        rule_family: str | None,
        restricted_rules: Iterable[str],
        properly_includes_context_free: bool | None,
        parikh_context_free_sublanguage: bool | None,
        citation: str | None,
        reason: str,
    ) -> None:
        if result not in CAPACITY_RESULTS:
            declared = ", ".join(CAPACITY_RESULTS)
            msg = f"capacity result {result!r} is not one of {declared}"
            raise ValueError(msg)

        if not isinstance(theorem_applicable, bool):
            msg = f"theorem applicability is not a boolean: {theorem_applicable!r}"
            raise TypeError(msg)

        if rule_family is not None:
            rule_family = ensure_name("a CCG rule family", rule_family)

        for field, value in (
            ("context-free inclusion", properly_includes_context_free),
            ("Parikh sublanguage", parikh_context_free_sublanguage),
        ):
            if value is not None and not isinstance(value, bool):
                msg = f"{field} result is not a boolean: {value!r}"
                raise TypeError(msg)

        if theorem_applicable != (result != "not_characterized"):
            msg = "theorem applicability must agree with the capacity result"
            raise ValueError(msg)

        if theorem_applicable and citation is None:
            msg = "an applicable capacity theorem requires a citation"
            raise ValueError(msg)

        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_theorem_applicable", theorem_applicable)
        object.__setattr__(self, "_rule_family", rule_family)
        object.__setattr__(
            self,
            "_restricted_rules",
            tuple(ensure_name("a restricted rule", name) for name in restricted_rules),
        )
        object.__setattr__(
            self,
            "_properly_includes_context_free",
            properly_includes_context_free,
        )
        object.__setattr__(
            self,
            "_parikh_context_free_sublanguage",
            parikh_context_free_sublanguage,
        )
        object.__setattr__(
            self,
            "_citation",
            None if citation is None else ensure_name("a capacity citation", citation),
        )
        object.__setattr__(self, "_reason", ensure_name("a capacity reason", reason))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a capacity report is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a capacity report is immutable"

        raise AttributeError(msg)

    @property
    def result(self) -> CapacityResult:
        """Return the proved language-family relationship."""
        return self._result

    @property
    def theorem_applicable(self) -> bool:
        """Return whether every cited theorem hypothesis holds."""
        return self._theorem_applicable

    @property
    def rule_family(self) -> str | None:
        """Return the attested CCG rule family, if supplied."""
        return self._rule_family

    @property
    def restricted_rules(self) -> tuple[str, ...]:
        """Return restricted rules in declaration order."""
        return self._restricted_rules

    @property
    def properly_includes_context_free(self) -> bool | None:
        """Return the proved CFL-inclusion result, if characterized."""
        return self._properly_includes_context_free

    @property
    def parikh_context_free_sublanguage(self) -> bool | None:
        """Return the proved Parikh-sublanguage result, if characterized."""
        return self._parikh_context_free_sublanguage

    @property
    def citation(self) -> str | None:
        """Return the theorem citation, if applicable."""
        return self._citation

    @property
    def reason(self) -> str:
        """Return the applicability explanation."""
        return self._reason

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every report field."""
        if type(other) is not CapacityReport:
            return NotImplemented

        return (
            self._result == other._result
            and self._theorem_applicable == other._theorem_applicable
            and self._rule_family == other._rule_family
            and self._restricted_rules == other._restricted_rules
            and self._properly_includes_context_free
            == other._properly_includes_context_free
            and self._parikh_context_free_sublanguage
            == other._parikh_context_free_sublanguage
            and self._citation == other._citation
            and self._reason == other._reason
        )

    @override
    def __hash__(self) -> int:
        """Hash every report field."""
        return hash(
            (
                CapacityReport,
                self._result,
                self._theorem_applicable,
                self._rule_family,
                self._restricted_rules,
                self._properly_includes_context_free,
                self._parikh_context_free_sublanguage,
                self._citation,
                self._reason,
            )
        )

    @override
    def __reduce__(self) -> _ReportReduction:
        """Return the reconstruction call."""
        fields: _ReportFields = (
            self._result,
            self._theorem_applicable,
            self._rule_family,
            self._restricted_rules,
            self._properly_includes_context_free,
            self._parikh_context_free_sublanguage,
            self._citation,
            self._reason,
        )

        return (_restore_capacity_report, (fields,))

    @override
    def __repr__(self) -> str:
        """Return a stable report summary."""
        return (
            f"CapacityReport(result={self._result!r}, "
            f"theorem_applicable={self._theorem_applicable!r}, "
            f"rule_family={self._rule_family!r}, "
            f"restricted_rules={self._restricted_rules!r})"
        )


def _restore_capacity_report(fields: _ReportFields) -> CapacityReport:
    """Reconstruct a capacity report from frozen fields."""
    (
        result,
        applicable,
        rule_family,
        restricted_rules,
        includes_context_free,
        parikh_sublanguage,
        citation,
        reason,
    ) = fields

    return CapacityReport(
        result,
        theorem_applicable=applicable,
        rule_family=rule_family,
        restricted_rules=restricted_rules,
        properly_includes_context_free=includes_context_free,
        parikh_context_free_sublanguage=parikh_sublanguage,
        citation=citation,
        reason=reason,
    )


def _uncharacterized[T](
    ruleset: RuleSet[T],
    evidence: CapacityEvidence | None,
    reason: str,
) -> CapacityReport:
    """Build an uncharacterized report with structural rule names."""
    return CapacityReport(
        "not_characterized",
        theorem_applicable=False,
        rule_family=None if evidence is None else evidence.rule_family,
        restricted_rules=(
            rule.name for rule in ruleset if rule.restriction is not None
        ),
        properly_includes_context_free=None,
        parikh_context_free_sublanguage=None,
        citation=None,
        reason=reason,
    )


def _composition_order[T](rule: Rule[T]) -> int | None:
    """Return the declared order of a composition rule, or None for another rule."""
    role = rule.normal_form_role.composition

    return None if role is None else role.order


def _application_target_restrictions[T](ruleset: RuleSet[T]) -> tuple[str, ...]:
    """Return application rules with primary-target restrictions."""
    return tuple(
        rule.name
        for rule in ruleset
        if _composition_order(rule) == 0
        and rule.restriction is not None
        and rule.restriction.primary_targets
    )


def _evidence_problem[T](ruleset: RuleSet[T], evidence: CapacityEvidence) -> str | None:
    """Return the first unmet common theorem hypothesis."""
    if not evidence.bounded_degree:
        return "bounded composition degree was not established"

    if any(
        (role := rule.normal_form_role.composition) is not None and role.order is None
        for rule in ruleset
    ):
        return "a schema admitting every finite order has no bounded degree"

    if not evidence.prefix_closed:
        return "prefix closure was not established"

    if not ruleset.rules or not all(
        rule.normal_form_role.composition is not None for rule in ruleset
    ):
        return "the inventory is not exclusively generalized CCG composition"

    return None


@runtime_checkable
class CapacityTheorem(Protocol):
    """Report the cited capacity result for one declared CCG rule family."""

    def report[T](
        self,
        ruleset: RuleSet[T],
        evidence: CapacityEvidence,
        restricted_rules: tuple[str, ...],
        /,
    ) -> CapacityReport:
        """Return the family's result once the common hypotheses hold."""
        ...


def _pure_ccg_report[T](
    ruleset: RuleSet[T],
    evidence: CapacityEvidence,
    restricted_rules: tuple[str, ...],
) -> CapacityReport:
    """Report the bounded-degree pure-CCG result."""
    if (
        ruleset.restriction_mode != "unrestricted"
        or restricted_rules
        or evidence.application_target_restrictions
    ):
        return _uncharacterized(
            ruleset,
            evidence,
            "pure CCG requires globally unrestricted rule instances",
        )

    return CapacityReport(
        "proper_superclass_of_context_free",
        theorem_applicable=True,
        rule_family=evidence.rule_family,
        restricted_rules=(),
        properly_includes_context_free=True,
        parikh_context_free_sublanguage=True,
        citation="Kuhlmann, Koller & Satta 2010, Theorem 1 and Proposition 1",
        reason=(
            "the report characterizes the bounded-degree pure-CCG family, "
            "not one concrete grammar"
        ),
    )


def _prefix_closed_vw_report[T](
    ruleset: RuleSet[T],
    evidence: CapacityEvidence,
    restricted_rules: tuple[str, ...],
) -> CapacityReport:
    """Report the applicable prefix-closed VW-CCG result."""
    target_rules = _application_target_restrictions(ruleset)

    if evidence.application_target_restrictions:
        if ruleset.restriction_mode != "grammar_specific" or not target_rules:
            return _uncharacterized(
                ruleset,
                evidence,
                "application-target restriction evidence does not match the inventory",
            )

        return CapacityReport(
            "tag_equivalent",
            theorem_applicable=True,
            rule_family=evidence.rule_family,
            restricted_rules=restricted_rules,
            properly_includes_context_free=True,
            parikh_context_free_sublanguage=None,
            citation="Kuhlmann, Koller & Satta 2015, Theorem 2 and Lemma 13",
            reason=(
                "the report characterizes prefix-closed VW-CCG with target "
                "restrictions on application, not one concrete grammar"
            ),
        )

    if target_rules:
        return _uncharacterized(
            ruleset,
            evidence,
            "the inventory contains an undeclared application-target restriction",
        )

    return CapacityReport(
        "proper_subclass_of_tag",
        theorem_applicable=True,
        rule_family=evidence.rule_family,
        restricted_rules=restricted_rules,
        properly_includes_context_free=True,
        parikh_context_free_sublanguage=True,
        citation="Kuhlmann, Koller & Satta 2015, Theorem 3 and Lemmas 4 and 6",
        reason=(
            "the report characterizes prefix-closed VW-CCG without target "
            "restrictions, not one concrete grammar"
        ),
    )


def capacity_report[T](
    ruleset: RuleSet[T], evidence: CapacityEvidence | None = None
) -> CapacityReport:
    """Report a capacity theorem only when its CCG hypotheses are supplied."""
    if evidence is None:
        return _uncharacterized(ruleset, None, "no CCG-family evidence was supplied")

    if (problem := _evidence_problem(ruleset, evidence)) is not None:
        return _uncharacterized(ruleset, evidence, problem)

    restricted_rules = tuple(
        rule.name for rule in ruleset if rule.restriction is not None
    )

    family = evidence.rule_family
    theorem = Registry[object]("rule_family").get(family)

    if not isinstance(theorem, CapacityTheorem):
        msg = f"the rule_family entry {family!r} names {theorem!r}, not a theorem"
        raise RegistryError(msg, family="rule_family")

    return theorem.report(ruleset, evidence, restricted_rules)


class _PureCcg(FrozenValue):
    """The bounded-degree pure-CCG capacity theorem."""

    _noun = "pure-CCG capacity theorem"
    __slots__ = ()

    def report[T](
        self,
        ruleset: RuleSet[T],
        evidence: CapacityEvidence,
        restricted_rules: tuple[str, ...],
        /,
    ) -> CapacityReport:
        """Report the bounded-degree pure-CCG result."""
        return _pure_ccg_report(ruleset, evidence, restricted_rules)


class _PrefixClosedVwCcg(FrozenValue):
    """The prefix-closed VW-CCG capacity theorems."""

    _noun = "prefix-closed VW-CCG capacity theorem"
    __slots__ = ()

    def report[T](
        self,
        ruleset: RuleSet[T],
        evidence: CapacityEvidence,
        restricted_rules: tuple[str, ...],
        /,
    ) -> CapacityReport:
        """Report the applicable prefix-closed VW-CCG result."""
        return _prefix_closed_vw_report(ruleset, evidence, restricted_rules)


PURE_CCG: Final[CapacityTheorem] = _PureCcg()
"""The installed bounded-degree pure-CCG rule family."""

PREFIX_CLOSED_VW_CCG: Final[CapacityTheorem] = _PrefixClosedVwCcg()
"""The installed prefix-closed VW-CCG rule family."""

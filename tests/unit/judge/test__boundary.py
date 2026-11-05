import copy
import pickle
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import (
    PREFIX_CLOSED_VW_CCG,
    PURE_CCG,
    CapacityEvidence,
    CapacityTheorem,
    DeductionRequirements,
    NoNormalForm,
    Rule,
    RuleRestriction,
    RuleSet,
    capacity_report,
    deduction_requirements,
)
from lofi.judge._values import ClosureReport
from lofi.registry import Registry, UnknownTagError
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    NormalFormRoleSpec,
    PatternRuleSpec,
    RuleRestrictionSpec,
    RuleSetSpec,
    SubstitutionSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.terms import Identity, Term

if TYPE_CHECKING:
    from lofi.judge import RestrictionMode
    from lofi.spec import RuleSpec


SIGNATURE: Final = ccg_signature("S", "NP", name="capacity-boundary")
MATCHING: Final = Identity[Term]()


def _rule(declaration: RuleSpec) -> Rule[Term]:
    """Compile one rule over the boundary signature."""
    (rule,) = declared_rules(
        SIGNATURE, RuleSetSpec(name="capacity", rules=(declaration,))
    ).rule_set

    return rule


CONTEXT: Final = declared_rules(
    SIGNATURE, RuleSetSpec(name="capacity", rules=(ApplicationSpec(view="/"),))
).context
S: Final = CONTEXT.syntax.parse(CONTEXT.store, "S")
NP: Final = CONTEXT.syntax.parse(CONTEXT.store, "NP")
X: Final = TermPatternSpec(name="X", sort="Cat")


def _composition(
    *,
    restriction: RuleRestrictionSpec | None = None,
    order: int | None = 0,
) -> Rule[Term]:
    return _rule(
        ApplicationSpec(view="/", restriction=restriction)
        if order == 0
        else CompositionSpec(
            view="/", order=order, direction="kuhlmann", restriction=restriction
        )
    )


def _ruleset(
    *rules: Rule[Term],
    restriction_mode: str = "unrestricted",
) -> RuleSet[Term]:
    return RuleSet(
        "capacity",
        rules,
        MATCHING,
        NoNormalForm(),
        cast("RestrictionMode", restriction_mode),
    )


def _evidence(
    family: str,
    *,
    bounded: bool = True,
    prefix_closed: bool = True,
    target_restrictions: bool = False,
) -> CapacityEvidence:
    return CapacityEvidence(
        family,
        bounded_degree=bounded,
        prefix_closed=prefix_closed,
        application_target_restrictions=target_restrictions,
    )


def test_deduction_requirements_derive_only_judge_owned_facts() -> None:
    rules = _ruleset(_composition(order=1))
    closure = ClosureReport(
        (S, NP),
        saturated=False,
        steps=3,
        budget=8,
        escaping=(CONTEXT.syntax.parse(CONTEXT.store, "S/NP"),),
    )

    requirements = deduction_requirements(rules, closure)

    assert requirements.ranks == frozenset({2})
    assert not requirements.unary_closure
    assert requirements.category_set == "unknown"
    assert requirements.restriction_mode == "unrestricted"
    assert not hasattr(requirements, "lexicalized")


def test_deduction_requirements_are_frozen_and_reconstructable() -> None:
    requirements = DeductionRequirements(
        (2, 1, 2),
        unary_closure=True,
        category_set="finite",
        restriction_mode="grammar_specific",
    )

    for rebuilt in (
        copy.copy(requirements),
        copy.deepcopy(requirements),
        pickle.loads(pickle.dumps(requirements)),
    ):
        assert rebuilt == requirements
        assert hash(rebuilt) == hash(requirements)

    assert requirements != DeductionRequirements(
        (2,),
        unary_closure=False,
        category_set="finite",
        restriction_mode="grammar_specific",
    )

    with pytest.raises(AttributeError, match="immutable"):
        requirements.__delattr__("_ranks")


def test_mode_alone_never_characterizes_capacity() -> None:
    report = capacity_report(_ruleset(_composition()))

    assert report.result == "not_characterized"
    assert not report.theorem_applicable
    assert report.citation is None
    assert report.properly_includes_context_free is None
    assert report.parikh_context_free_sublanguage is None


def test_each_rule_family_is_an_installed_capacity_theorem() -> None:
    registry = Registry[object]("rule_family")

    assert registry.get("pure_ccg") is PURE_CCG
    assert registry.get("prefix_closed_vw_ccg") is PREFIX_CLOSED_VW_CCG
    assert isinstance(PURE_CCG, CapacityTheorem)

    with pytest.raises(UnknownTagError, match="mildly_ccg"):
        capacity_report(_ruleset(_composition()), _evidence("mildly_ccg"))


def test_capacity_evidence_has_structural_value_semantics() -> None:
    evidence = _evidence("pure_ccg")

    for rebuilt in (
        copy.copy(evidence),
        copy.deepcopy(evidence),
        pickle.loads(pickle.dumps(evidence)),
    ):
        assert rebuilt == evidence
        assert hash(rebuilt) == hash(evidence)

    assert evidence != _evidence("pure_ccg", prefix_closed=False)


@pytest.mark.parametrize(
    "evidence",
    [
        _evidence("pure_ccg", bounded=False),
        _evidence("pure_ccg", prefix_closed=False),
        _evidence("pure_ccg", target_restrictions=True),
    ],
)
def test_missing_or_inconsistent_pure_ccg_evidence_is_uncharacterized(
    evidence: CapacityEvidence,
) -> None:
    assert capacity_report(_ruleset(_composition()), evidence).result == (
        "not_characterized"
    )


def test_bounded_pure_ccg_properly_includes_context_free() -> None:
    report = capacity_report(_ruleset(_composition()), _evidence("pure_ccg"))

    assert report.result == "proper_superclass_of_context_free"
    assert report.theorem_applicable
    assert report.properly_includes_context_free is True
    assert report.parikh_context_free_sublanguage is True
    assert report.restricted_rules == ()
    assert report.citation is not None
    assert "2010" in report.citation
    assert "not one concrete grammar" in report.reason


def test_a_schema_over_every_finite_order_refuses_the_bounded_degree_claim() -> None:
    report = capacity_report(
        _ruleset(_composition(), _composition(order=None)),
        _evidence("pure_ccg"),
    )

    assert report.result == "not_characterized"
    assert not report.theorem_applicable
    assert (
        report.reason == "a schema admitting every finite order has no bounded degree"
    )


def test_capacity_evidence_reads_the_declared_composition_role() -> None:
    unbounded = _rule(
        PatternRuleSpec(
            name="declared-composition",
            premises=(X, TermPatternSpec(name="Y", sort="Cat")),
            conclusion=X,
            normal_form_role=NormalFormRoleSpec(
                kind="composition", direction="forward", principal_slot=0
            ),
        )
    )
    report = capacity_report(_ruleset(_composition(), unbounded), _evidence("pure_ccg"))

    assert (
        report.reason == "a schema admitting every finite order has no bounded degree"
    )


def test_prefix_closed_vw_ccg_with_application_targets_is_tag_equivalent() -> None:
    restriction = RuleRestrictionSpec(primary_targets=("S",))
    rules = _ruleset(
        _composition(restriction=restriction),
        restriction_mode="grammar_specific",
    )
    report = capacity_report(
        rules,
        _evidence("prefix_closed_vw_ccg", target_restrictions=True),
    )

    assert report.result == "tag_equivalent"
    assert report.restricted_rules == ("application:/:only(targets=S)",)
    assert report.properly_includes_context_free is True
    assert report.parikh_context_free_sublanguage is None
    assert report.citation is not None
    assert "Lemma 13" in report.citation


def test_prefix_closed_vw_ccg_without_targets_is_a_proper_tag_subclass() -> None:
    restriction = RuleRestrictionSpec(secondary_patterns=("NP",))
    rules = _ruleset(
        _composition(restriction=restriction),
        restriction_mode="grammar_specific",
    )
    report = capacity_report(rules, _evidence("prefix_closed_vw_ccg"))

    assert report.result == "proper_subclass_of_tag"
    assert report.properly_includes_context_free is True
    assert report.parikh_context_free_sublanguage is True
    assert report.restricted_rules == ("application:/:only(patterns=NP)",)
    assert report.citation is not None
    assert "Theorem 3" in report.citation


@pytest.mark.parametrize("kind", ["raising", "substitution", "pattern"])
def test_nonclassical_rule_families_remain_uncharacterized(kind: str) -> None:
    declaration: RuleSpec = (
        TypeRaisingSpec(view="/", targets=("S",), arguments=None)
        if kind == "raising"
        else SubstitutionSpec(view="/", order=1)
        if kind == "substitution"
        else PatternRuleSpec(name="identity", premises=(X,), conclusion=X)
    )
    rule = _rule(declaration)

    report = capacity_report(_ruleset(rule), _evidence("pure_ccg"))

    assert report.result == "not_characterized"
    assert "not exclusively" in report.reason


def test_every_production_rule_retains_and_declares_the_exact_restriction() -> None:
    binary = RuleRestrictionSpec(primary_targets=("S",), secondary_patterns=("NP",))
    unary = RuleRestrictionSpec(primary_targets=("S",))
    restricted = (
        (
            ApplicationSpec(view="/", restriction=binary),
            RuleRestriction(("S",), ("NP",)),
        ),
        (
            SubstitutionSpec(view="/", order=1, restriction=binary),
            RuleRestriction(("S",), ("NP",)),
        ),
        (
            TypeRaisingSpec(
                view="/", targets=("S",), restriction=unary, arguments=None
            ),
            RuleRestriction(("S",)),
        ),
        (
            PatternRuleSpec(
                name="identity", premises=(X,), conclusion=X, restriction=unary
            ),
            RuleRestriction(("S",)),
        ),
    )

    for declaration, expected in restricted:
        rule = _rule(declaration)

        assert isinstance(rule, Rule)
        assert rule.restriction is not None
        assert rule.restriction.to_spec() == expected.to_spec()
        assert rule.to_spec()["restriction"] == expected.to_spec()

        with pytest.raises(AttributeError, match="immutable"):
            rule.__delattr__("_restriction")


def test_capacity_reports_are_frozen_and_reconstructable() -> None:
    report = capacity_report(_ruleset(_composition()), _evidence("pure_ccg"))

    for rebuilt in (
        copy.copy(report),
        copy.deepcopy(report),
        pickle.loads(pickle.dumps(report)),
    ):
        assert rebuilt == report
        assert hash(rebuilt) == hash(report)

    assert report != capacity_report(_ruleset(_composition()))

    with pytest.raises(AttributeError, match="immutable"):
        report.__setattr__("_result", "not_characterized")

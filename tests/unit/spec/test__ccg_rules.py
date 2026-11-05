"""Tests for the installed CCG rule plugins."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import RuleProgram
from lofi.deduce import NativeTermRule
from lofi.judge import HOLE
from lofi.spec import (
    AbsorptionSpec,
    ApplicationSpec,
    ClausalAdjunctSpec,
    CoercionSpec,
    CompositionSpec,
    CoordinationSpec,
    DeclarationError,
    PatternRuleSpec,
    RuleRestrictionSpec,
    RuleSetSpec,
    SubstitutionSpec,
    TypeRaisingSpec,
)
from lofi.spec import _ccg_rules as rules

if TYPE_CHECKING:
    from lofi.spec import RuleSpec
    from lofi.spec.grammar import AdoptedRules
    from lofi.terms import Term


def test_every_shipped_rule_tag_has_one_plugin_for_its_exact_declaration() -> None:
    installed = {
        "absorption": (rules.ABSORPTION_PLUGIN, AbsorptionSpec),
        "application": (rules.APPLICATION_PLUGIN, ApplicationSpec),
        "clausal-adjunct": (rules.CLAUSAL_ADJUNCT_PLUGIN, ClausalAdjunctSpec),
        "coercion": (rules.COERCION_PLUGIN, CoercionSpec),
        "composition": (rules.COMPOSITION_PLUGIN, CompositionSpec),
        "coordination": (rules.COORDINATION_PLUGIN, CoordinationSpec),
        "pattern": (rules.PATTERN_PLUGIN, PatternRuleSpec),
        "substitution": (rules.SUBSTITUTION_PLUGIN, SubstitutionSpec),
        "type-raising": (rules.TYPE_RAISING_PLUGIN, TypeRaisingSpec),
    }

    assert {
        tag: (plugin.spec, repr(plugin)) for tag, (plugin, _) in installed.items()
    } == {
        tag: (spec, f"RulePlugin(spec={spec.__name__})")
        for tag, (_, spec) in installed.items()
    }


def test_shipped_plugins_expose_only_compiled_recipe_adoption() -> None:
    plugins = (
        rules.ABSORPTION_PLUGIN,
        rules.APPLICATION_PLUGIN,
        rules.CLAUSAL_ADJUNCT_PLUGIN,
        rules.COERCION_PLUGIN,
        rules.COMPOSITION_PLUGIN,
        rules.COORDINATION_PLUGIN,
        rules.PATTERN_PLUGIN,
        rules.SUBSTITUTION_PLUGIN,
        rules.TYPE_RAISING_PLUGIN,
    )

    for plugin in plugins:
        assert callable(plugin.adopt_compiled)
        assert not hasattr(plugin, "build")
        assert not hasattr(plugin, "construct")
        assert not hasattr(plugin, "selection")


def _restricted(rule: RuleSpec) -> AdoptedRules:
    """Compile one restricted rule over the S, NP, N and PP atoms."""
    return declared_rules(
        ccg_signature("S", "NP", "N", "PP"),
        RuleSetSpec(name="restricted", rules=(rule,)),
    )


def _category(adopted: AdoptedRules, text: str) -> Term:
    """Parse one category in the adopted rules' store."""
    return adopted.context.syntax.parse(adopted.context.store, text)


@pytest.mark.parametrize("view", ["/", "\\"])
def test_a_primary_target_restriction_admits_only_functors_of_that_target(
    view: str,
) -> None:
    adopted = _restricted(
        ApplicationSpec(
            view=view, restriction=RuleRestrictionSpec(primary_targets=("S",))
        )
    )
    (rule,) = adopted.rule_set
    allowed, refused, argument = (
        _category(adopted, text) for text in (f"(S{view}NP){view}N", f"NP{view}N", "N")
    )
    pair = (
        (lambda functor: (functor, argument))
        if view == "/"
        else (lambda functor: (argument, functor))
    )
    allowed_output = _category(adopted, f"S{view}NP")

    assert tuple(rule.apply(pair(allowed))) == (allowed_output,)
    assert rule.admits(pair(allowed), allowed_output)
    assert tuple(rule.apply(pair(refused))) == ()
    assert not rule.admits(pair(refused), _category(adopted, "NP"))
    hole = (HOLE, argument) if view == "/" else (argument, HOLE)
    assert tuple(rule.residuate(_category(adopted, "NP"), hole)) == ()
    assert tuple(rule.residuate(allowed_output, hole)) == (allowed,)


def test_a_secondary_restriction_admits_only_the_declared_secondary_categories() -> (
    None
):
    adopted = _restricted(
        CompositionSpec(
            view="/",
            order=1,
            direction="unrestricted",
            restriction=RuleRestrictionSpec(secondary_patterns=("NP/N",)),
        )
    )
    (rule,) = adopted.rule_set
    functor = _category(adopted, "S/NP")
    allowed, refused = (_category(adopted, text) for text in ("NP/N", "NP/PP"))

    assert tuple(rule.apply((functor, allowed))) == (_category(adopted, "S/N"),)
    assert tuple(rule.apply((functor, refused))) == ()
    assert not rule.admits((functor, refused), _category(adopted, "S/PP"))
    assert tuple(rule.residuate(_category(adopted, "S/PP"), (functor, HOLE))) == ()


def test_a_restricted_unary_rule_raises_only_categories_of_its_target() -> None:
    adopted = _restricted(
        TypeRaisingSpec(
            view="/",
            targets=("S",),
            arguments=None,
            restriction=RuleRestrictionSpec(primary_targets=("NP",)),
        )
    )
    (rule,) = adopted.rule_set

    assert tuple(rule.apply((_category(adopted, "NP/N"),))) == (
        _category(adopted, "S/(S\\(NP/N))"),
    )
    assert tuple(rule.apply((_category(adopted, "N"),))) == ()


def test_a_restricted_chart_parses_only_through_the_declared_targets() -> None:
    restricted = RuleSetSpec(
        name="restricted",
        rules=(
            ApplicationSpec(
                view="/", restriction=RuleRestrictionSpec(primary_targets=("NP",))
            ),
        ),
    )
    adopted = declared_rules(ccg_signature("S", "NP", "N"), restricted)
    program = RuleProgram(
        tuple(
            rule.lowering.native
            for rule in adopted.rule_set
            if isinstance(rule, NativeTermRule)
        )
    )
    determiner, noun, verb = (
        _category(adopted, text) for text in ("NP/N", "N", "S/NP")
    )
    (rule,) = adopted.rule_set

    assert tuple(rule.apply((determiner, noun))) == (_category(adopted, "NP"),)
    assert tuple(rule.apply((verb, _category(adopted, "NP")))) == ()
    assert [
        output
        for _, outputs in program.apply([0], (determiner, noun))
        for output in outputs
    ] == [_category(adopted, "NP")]
    assert [
        output
        for _, outputs in program.apply([0], (verb, _category(adopted, "NP")))
        for output in outputs
    ] == []


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        (
            ClausalAdjunctSpec(
                name="adjunct",
                source_view="\\",
                target_view="\\",
                restriction=RuleRestrictionSpec(primary_targets=("S",)),
            ),
            "no primary premise",
        ),
        (
            TypeRaisingSpec(
                view="/",
                targets=("S",),
                arguments=None,
                restriction=RuleRestrictionSpec(secondary_patterns=("NP",)),
            ),
            "no secondary premise",
        ),
        (
            ApplicationSpec(
                view="/", restriction=RuleRestrictionSpec(primary_targets=("Q",))
            ),
            "Q",
        ),
    ],
    ids=["clausal-adjunct", "unary-secondary", "unknown-category"],
)
def test_a_restriction_the_chart_cannot_enforce_is_refused(
    rule: RuleSpec, match: str
) -> None:
    with pytest.raises(DeclarationError, match=match):
        _restricted(rule)


def test_a_restriction_declares_no_lexical_anchor() -> None:
    with pytest.raises(dx.ValidationError):
        RuleRestrictionSpec.model_validate({"anchors": ["that"]})

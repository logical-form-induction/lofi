import copy
import pickle
from typing import TYPE_CHECKING, cast

import pytest

from lofi.judge._values import (
    Action,
    ActionSet,
    ClosureReport,
    Diagnostic,
    RuleRestriction,
)

if TYPE_CHECKING:
    from lofi.judge._values import Severity


def _apply() -> Action:
    return Action("apply", "forward-application")


def _compose() -> Action:
    return Action("compose", "forward-composition")


def _none() -> Action:
    return Action("none")


class TestRuleRestriction:
    """Rule-restriction tests."""

    def test_freezes_the_two_inventories(self) -> None:
        """Freeze the target and pattern inventories."""
        restriction = RuleRestriction(["NP", "S", "NP"], ["N/NP", "N/NP"])

        assert restriction.primary_targets == frozenset({"NP", "S"})
        assert restriction.secondary_patterns == frozenset({"N/NP"})

    def test_each_inventory_is_independently_sufficient(self) -> None:
        """Accept a restriction naming either premise."""
        assert RuleRestriction(["NP"]).secondary_patterns == frozenset()
        assert RuleRestriction(secondary_patterns=["NP/N"]).primary_targets == (
            frozenset()
        )

    def test_an_empty_restriction_is_refused(self) -> None:
        """Reject an empty restriction."""
        with pytest.raises(ValueError, match="no primary target or pattern"):
            RuleRestriction()

    def test_an_empty_target_is_refused(self) -> None:
        """Reject an empty primary-target name."""
        with pytest.raises(ValueError, match="restricted primary target is empty"):
            RuleRestriction([""])

    def test_a_pattern_that_is_not_a_name_is_refused(self) -> None:
        """Reject a non-string secondary pattern."""
        with pytest.raises(TypeError, match="restricted secondary pattern is not"):
            RuleRestriction(secondary_patterns=cast("list[str]", [3]))

    def test_two_restrictions_over_the_same_sets_agree(self) -> None:
        """Compare and hash both sets."""
        one = RuleRestriction(["NP"], ["NP/N"])
        other = RuleRestriction(("NP",), ("NP/N",))

        assert one == other
        assert hash(one) == hash(other)
        assert one != RuleRestriction(["NP"])

    def test_a_qualified_name_lists_each_nonempty_set_in_sorted_order(self) -> None:
        """Distinguish rule names by every restriction field."""
        full = RuleRestriction(["S", "NP"], ["S/NP"])

        assert full.qualify("rule") == "rule:only(targets=NP,S;patterns=S/NP)"
        assert RuleRestriction(secondary_patterns=["S"]).qualify("rule") == (
            "rule:only(patterns=S)"
        )
        assert RuleRestriction(["S"]).qualify("rule") != RuleRestriction(
            secondary_patterns=["S"]
        ).qualify("rule")

    def test_a_restriction_is_unequal_to_another_type(self) -> None:
        """Return NotImplemented for another type."""
        assert RuleRestriction(["NP"]).__eq__(object()) is NotImplemented

    def test_the_spec_and_repr_sort_every_set(self) -> None:
        """Render each declaration in canonical order."""
        restriction = RuleRestriction(["S", "NP"], ["S/NP", "NP/N"])

        assert restriction.to_spec() == {
            "primary_targets": ["NP", "S"],
            "secondary_patterns": ["NP/N", "S/NP"],
        }
        assert repr(restriction) == (
            "RuleRestriction(primary_targets=['NP', 'S'], "
            "secondary_patterns=['NP/N', 'S/NP'])"
        )

    def test_copy_deepcopy_pickle_and_fields_are_frozen(self) -> None:
        """Preserve a frozen restriction through standard reconstruction."""
        restriction = RuleRestriction(["S"], ["NP"])

        for rebuilt in (
            copy.copy(restriction),
            copy.deepcopy(restriction),
            pickle.loads(pickle.dumps(restriction)),
        ):
            assert rebuilt == restriction

        with pytest.raises(AttributeError, match="immutable"):
            restriction.__setattr__("_secondary_patterns", frozenset({"S"}))

        with pytest.raises(AttributeError, match="immutable"):
            restriction.__delattr__("_secondary_patterns")


class TestDiagnostic:
    """Rule-inventory-diagnostic tests."""

    def test_carries_its_structured_fields(self) -> None:
        """Return every diagnostic field."""
        finding = Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "rule 'Phi' lacks complete forward enumeration",
            severity="warning",
            rules=["Phi", "Psi"],
        )

        assert finding.code == "forward_enumeration_unavailable"
        assert finding.requirement == "forward_enumeration"
        assert finding.severity == "warning"
        assert finding.message == "rule 'Phi' lacks complete forward enumeration"
        assert finding.rules == ("Phi", "Psi")

    def test_a_finding_may_concern_the_whole_inventory(self) -> None:
        """Accept a finding naming no rule at all."""
        finding = Diagnostic(
            "generation_unavailable",
            "generative_parsing",
            "the inventory cannot generate conclusions",
        )

        assert finding.rules == ()
        assert finding.severity == "error"

    @pytest.mark.parametrize(
        ("code", "requirement", "field"),
        [("", "forward_enumeration", "code"), ("missing", "", "requirement")],
    )
    def test_an_empty_code_or_requirement_is_refused(
        self, code: str, requirement: str, field: str
    ) -> None:
        """Reject a finding without a code or requirement name."""
        with pytest.raises(ValueError, match=f"a diagnostic's {field} is empty"):
            Diagnostic(code, requirement, "something")

    def test_an_undeclared_severity_is_refused(self) -> None:
        """Reject an unknown severity."""
        with pytest.raises(ValueError, match="error, warning"):
            Diagnostic(
                "generation_unavailable",
                "generative_parsing",
                "something",
                severity=cast("Severity", "fatal"),
            )

    def test_an_empty_message_is_refused(self) -> None:
        """Reject an empty diagnostic message."""
        with pytest.raises(ValueError, match="a diagnostic's message is empty"):
            Diagnostic("generation_unavailable", "generative_parsing", "")

    def test_two_findings_recording_the_same_thing_agree(self) -> None:
        """Compare and hash every structured field."""
        one = Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "missing",
            rules=["Phi"],
        )
        other = Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "missing",
            rules=("Phi",),
        )

        assert one == other
        assert hash(one) == hash(other)
        assert one != Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "missing",
            severity="warning",
            rules=["Phi"],
        )

    def test_a_finding_is_unequal_to_another_type(self) -> None:
        """Return NotImplemented for another type."""
        finding = Diagnostic("generation_unavailable", "generative_parsing", "missing")

        assert finding.__eq__("error") is NotImplemented

    def test_the_repr_names_every_field(self) -> None:
        """Render all structured fields in stable order."""
        finding = Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "missing",
            severity="warning",
            rules=["Phi"],
        )

        assert repr(finding) == (
            "Diagnostic('forward_enumeration_unavailable', 'forward_enumeration', "
            "'missing', severity='warning', rules=['Phi'])"
        )


class TestAction:
    """Controller-action tests."""

    def test_carries_a_name_and_a_rule(self) -> None:
        """Report the column's name and the rule it applies."""
        action = _apply()

        assert action.name == "apply"
        assert action.rule == "forward-application"

    def test_the_identity_applies_no_rule(self) -> None:
        """Accept an action standing for no rule."""
        assert _none().rule is None

    def test_an_empty_name_is_refused(self) -> None:
        """Reject an empty action name."""
        with pytest.raises(ValueError, match="an action's name is empty"):
            Action("")

    def test_an_empty_rule_is_refused(self) -> None:
        """Reject an empty rule name."""
        with pytest.raises(ValueError, match="an action's rule is empty"):
            Action("apply", "")

    def test_two_actions_over_the_same_pair_agree(self) -> None:
        """Compare and hash on the name and the rule."""
        assert _apply() == _apply()
        assert hash(_apply()) == hash(_apply())
        assert _apply() != Action("apply")

    def test_an_action_is_unequal_to_another_type(self) -> None:
        """Return NotImplemented for another type."""
        assert _apply().__eq__("apply") is NotImplemented

    def test_the_repr_names_the_pair(self) -> None:
        """Print the name and the rule."""
        assert repr(_none()) == "Action('none', None)"


class TestActionSet:
    """Ordered-action-set tests."""

    def test_keeps_the_order_it_was_given(self) -> None:
        """Report the actions in the order the columns take."""
        actions = ActionSet([_apply(), _compose(), _none()], _none())

        assert actions.actions == (_apply(), _compose(), _none())
        assert actions.identity == _none()
        assert len(actions.actions) == 3

    def test_an_index_lookup_returns_the_column(self) -> None:
        """Return the position each action occupies."""
        actions = ActionSet([_apply(), _compose(), _none()], _none())

        assert actions.index("apply") == 0
        assert actions.index("compose") == 1
        assert actions.index("none") == 2

    def test_an_unknown_action_has_no_column(self) -> None:
        """Reject an unknown action name."""
        actions = ActionSet([_apply(), _none()], _none())

        with pytest.raises(KeyError, match="apply, none"):
            actions.index("substitute")

    def test_an_empty_set_is_refused(self) -> None:
        """Reject an empty action set."""
        with pytest.raises(ValueError, match="no actions"):
            ActionSet([], _none())

    def test_a_repeated_name_is_refused_with_both_columns_named(self) -> None:
        """Reject duplicate action names."""
        with pytest.raises(ValueError, match="columns 0 and 2"):
            ActionSet([_apply(), _compose(), _apply()], _compose())

    def test_an_identity_outside_the_columns_is_refused(self) -> None:
        """Reject an identity absent from the action set."""
        with pytest.raises(ValueError, match="'none' is not one of the columns"):
            ActionSet([_apply(), _compose()], _none())

    def test_the_digest_changes_under_permutation(self) -> None:
        """Distinguish permutations of the same actions."""
        one = ActionSet([_apply(), _compose(), _none()], _none())
        swapped = ActionSet([_compose(), _apply(), _none()], _none())

        assert one.digest != swapped.digest

    def test_the_digest_is_a_function_of_the_ordering(self) -> None:
        """Agree for independently built equal orderings."""
        one = ActionSet([_apply(), _none()], _none())
        other = ActionSet([Action("apply", "forward-application"), _none()], _none())

        assert one.digest == other.digest
        assert len(one.digest) == 64

    def test_the_digest_changes_when_a_column_repoints(self) -> None:
        """Distinguish columns that name different rules."""
        one = ActionSet([_apply(), _none()], _none())
        repointed = ActionSet(
            [Action("apply", "backward-application"), _none()], _none()
        )

        assert one.digest != repointed.digest

    def test_two_sets_with_the_same_ordering_agree(self) -> None:
        """Compare and hash on the ordering and the identity."""
        one = ActionSet([_apply(), _none()], _none())
        other = ActionSet([_apply(), _none()], _none())

        assert one == other
        assert hash(one) == hash(other)
        assert one != ActionSet([_none(), _apply()], _none())

    def test_a_set_is_unequal_to_another_type(self) -> None:
        """Return NotImplemented for another type."""
        assert ActionSet([_none()], _none()).__eq__(()) is NotImplemented

    def test_the_repr_names_the_ordering(self) -> None:
        """Print the columns and the identity."""
        assert repr(ActionSet([_none()], _none())) == (
            "ActionSet([Action('none', None)], Action('none', None))"
        )

    def test_a_derived_set_puts_the_identity_first_and_one_action_per_rule(
        self,
    ) -> None:
        """Derive the identity column and one column per rule, in rule order."""
        actions = ActionSet.derive(("compose", "apply"), identity="keep", null=None)

        assert actions.actions == (
            Action("keep"),
            Action("compose", "compose"),
            Action("apply", "apply"),
        )
        assert actions.identity == Action("keep")
        assert actions.null is None
        assert actions.composing == actions.actions[1:]

    def test_a_derived_null_action_is_the_last_column(self) -> None:
        """Append a declared null action after the rules."""
        actions = ActionSet.derive(("apply",), identity="keep", null="none-applies")

        assert actions.actions[-1] == Action("none-applies")
        assert actions.null == Action("none-applies")
        assert actions.index("none-applies") == 2
        assert actions.composing == (Action("apply", "apply"),)

    def test_a_derived_name_collision_is_refused(self) -> None:
        """Reject a rule named like the identity or the null action."""
        with pytest.raises(ValueError, match="columns 0 and 1"):
            ActionSet.derive(("keep",), identity="keep", null=None)

        with pytest.raises(ValueError, match="columns 1 and 2"):
            ActionSet.derive(("none-applies",), identity="keep", null="none-applies")

    def test_a_null_action_outside_the_columns_is_refused(self) -> None:
        """Reject a null action absent from the action set."""
        with pytest.raises(ValueError, match="null action 'void' is not one"):
            ActionSet([_apply(), _none()], _none(), null=Action("void"))

    def test_the_identity_cannot_also_be_the_null_action(self) -> None:
        """Reject one action declared as both identity and null."""
        with pytest.raises(ValueError, match="both identity and null"):
            ActionSet([_apply(), _none()], _none(), null=_none())

    def test_a_null_action_applying_a_rule_is_refused(self) -> None:
        """Reject a null action that names a rule."""
        with pytest.raises(ValueError, match="applies the rule 'forward-application'"):
            ActionSet([_apply(), _none()], _none(), null=_apply())

    def test_declaring_a_null_action_changes_the_digest_and_the_equality(
        self,
    ) -> None:
        """Distinguish a set declaring its null action from one that does not."""
        void = Action("void")
        plain = ActionSet([_apply(), _none(), void], _none())
        declared = ActionSet([_apply(), _none(), void], _none(), null=void)

        assert plain.digest != declared.digest
        assert plain != declared
        assert declared == ActionSet([_apply(), _none(), void], _none(), null=void)
        assert hash(declared) == hash(
            ActionSet([_apply(), _none(), void], _none(), null=void)
        )
        assert plain.composing == (_apply(), void)
        assert declared.composing == (_apply(),)

    def test_the_repr_names_a_declared_null_action(self) -> None:
        """Print the null action as the keyword the constructor takes."""
        void = Action("void")

        assert repr(ActionSet([_none(), void], _none(), null=void)) == (
            "ActionSet([Action('none', None), Action('void', None)], "
            "Action('none', None), null=Action('void', None))"
        )


class TestClosureReport:
    """Forward-closure-report tests."""

    def test_carries_what_the_closure_found_and_spent(self) -> None:
        """Report closure results and costs."""
        report = ClosureReport(
            [1, 2, 4],
            saturated=True,
            steps=7,
            budget=64,
            never_applicable=["Phi"],
        )

        assert report.reached == (1, 2, 4)
        assert report.saturated is True
        assert report.steps == 7
        assert report.budget == 64
        assert report.never_applicable == ("Phi",)
        assert report.escaping == ()

    def test_refuses_private_slot_mutation_and_deletion(self) -> None:
        """Freeze closure-report fields after construction."""
        report = ClosureReport([1], saturated=True, steps=0, budget=1)

        with pytest.raises(AttributeError, match="immutable"):
            report.__setattr__("_reached", (2,))

        with pytest.raises(AttributeError, match="immutable"):
            report.__delattr__("_budget")

    def test_copy_deepcopy_and_pickle_use_public_reconstruction(self) -> None:
        """Reconstruct frozen reports through standard protocols."""
        report = ClosureReport(
            [1, 2],
            saturated=False,
            steps=3,
            budget=4,
            never_applicable=["unused"],
            escaping=[5],
        )
        restored = cast(
            "ClosureReport[int]",
            pickle.loads(pickle.dumps(report)),
        )

        for rebuilt in (copy.copy(report), copy.deepcopy(report), restored):
            assert rebuilt.reached == report.reached
            assert rebuilt.saturated == report.saturated
            assert rebuilt.steps == report.steps
            assert rebuilt.budget == report.budget
            assert rebuilt.never_applicable == report.never_applicable
            assert rebuilt.escaping == report.escaping

    def test_a_fixpoint_with_nothing_escaping_is_finite(self) -> None:
        """Read a saturated closure over a closed inventory as finite."""
        report = ClosureReport([1, 2], saturated=True, steps=2, budget=8)

        assert report.category_set == "finite"

    def test_a_fixpoint_with_an_escaping_category_is_infinite(self) -> None:
        """Read a named witness of unbounded growth as infinite."""
        report = ClosureReport([1, 2], saturated=True, steps=2, budget=8, escaping=[2])

        assert report.category_set == "infinite"

    def test_exhaustion_is_unknown_rather_than_finite(self) -> None:
        """Return unknown after budget exhaustion."""
        report = ClosureReport([1, 2], saturated=False, steps=8, budget=8)

        assert report.category_set == "unknown"

    def test_exhaustion_stays_unknown_with_escapees_named(self) -> None:
        """Name escapees without upgrading an exhausted run."""
        report = ClosureReport([1, 2], saturated=False, steps=8, budget=8, escaping=[2])

        assert report.escaping == (2,)
        assert report.category_set == "unknown"

    def test_a_non_boolean_saturation_is_refused(self) -> None:
        """Reject a non-boolean saturation value."""
        with pytest.raises(TypeError, match="saturation is not a boolean"):
            ClosureReport([], saturated=cast("bool", 1), steps=0, budget=0)

    def test_a_negative_step_count_is_refused(self) -> None:
        """Reject negative counts."""
        with pytest.raises(ValueError, match="step count is negative"):
            ClosureReport([], saturated=True, steps=-1, budget=8)

    def test_an_empty_rule_name_is_refused(self) -> None:
        """Reject an empty inapplicable-rule name."""
        with pytest.raises(ValueError, match="an inapplicable rule is empty"):
            ClosureReport([], saturated=True, steps=0, budget=8, never_applicable=[""])

    def test_the_repr_counts_rather_than_lists_the_categories(self) -> None:
        """Print category counts rather than categories."""
        report = ClosureReport(
            range(1000), saturated=False, steps=99, budget=99, escaping=[1, 2]
        )

        assert repr(report) == (
            "ClosureReport(reached=1000, saturated=False, steps=99, budget=99, "
            "never_applicable=[], escaping=2, category_set='unknown')"
        )

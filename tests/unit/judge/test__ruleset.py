import copy
import pickle
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Final, cast, override

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import NormalFormRole
from lofi.judge._errors import RuleSetValidationError
from lofi.judge._index import RuleIndex, RuleProbe
from lofi.judge._normal import NoNormalForm
from lofi.judge._ports import HOLE, Direction
from lofi.judge._recover import recover
from lofi.judge._ruleset import (
    FORWARD_ENUMERATION,
    GenerativeParsing,
    RuleSet,
    refuse_coinciding_rules,
)
from lofi.judge._values import Diagnostic, RuleRestriction
from lofi.spec import ApplicationSpec, CompositionSpec, RuleSetSpec
from lofi.terms import Identity, Term

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge._ports import NormalFormDecisionTable, Rule, Slot
    from lofi.judge._values import RestrictionMode

_ALL: Final[frozenset[Direction]] = frozenset({"forward", "check", "residual"})
_FORWARD: Final[frozenset[Direction]] = frozenset({"forward"})


class Exact:
    """Integer-identity congruence."""

    @property
    def name(self) -> str:
        """Name the congruence."""
        return "exact"

    @property
    def has_key(self) -> bool:
        """Report that classes have representatives."""
        return True

    def key(self, term: int, /) -> int:
        """Return the term, which is its own representative."""
        return term

    def equivalent(self, left: int, right: int, /) -> bool:
        """Compare integers."""
        return left == right


MATCHING: Final = Exact()


class IntegerMeasure:
    """Two independent structural dimensions for integer closure fixtures."""

    @staticmethod
    def size(term: int, /) -> int:
        """Treat each successor as one additional node."""
        return term + 1

    @staticmethod
    def arity(term: int, /) -> int:
        """Treat the integer as a functor-spine arity."""
        return term


class Flat:
    """Permissive one-tag normal form."""

    @property
    def name(self) -> str:
        """Name the filter."""
        return "flat"

    @property
    def alphabet(self) -> frozenset[str]:
        """Use one tag."""
        return frozenset({"any"})

    def lexical_tag(self) -> str:
        """Tag a lexical item."""
        return "any"

    def decisions(self, rules: Sequence[Rule[int]], /) -> NormalFormDecisionTable:
        """Defer every finite tag state to exact step admission."""
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "premises")
            for rule in rules
        )

    def tag(self, rule: Rule[int], children: tuple[str, ...], /) -> str:
        """Tag a derived item."""
        del rule, children

        return "any"

    def admits(
        self,
        rule: Rule[int],
        children: tuple[str, ...],
        inputs: tuple[int, ...],
        output: int,
        /,
    ) -> bool:
        """Keep every step."""
        del rule, children, inputs, output

        return True


class OneHead:
    """Single head code for integers."""

    @property
    def codes(self) -> frozenset[int]:
        """Return the sole head code."""
        return frozenset({0})

    def term(self, term: int, /) -> int:
        """Map every term to the sole code."""
        del term
        return 0

    def named(self, name: str, /) -> int:
        """Map every name to the sole code."""
        del name
        return 0


class ParityHeads:
    """Parity head codes for integers."""

    @property
    def codes(self) -> frozenset[int]:
        """Return both parity codes."""
        return frozenset({0, 1})

    def term(self, term: int, /) -> int:
        """Map an integer to its parity."""
        return term % 2

    def named(self, name: str, /) -> int:
        """Map a parity name to its code."""
        return {"even": 0, "odd": 1}[name]


class Absolute(Exact):
    """Absolute-value congruence."""

    @property
    @override
    def name(self) -> str:
        """Name the congruence."""
        return "absolute"

    @override
    def key(self, term: int, /) -> int:
        """Return the nonnegative representative."""
        return abs(term)

    @override
    def equivalent(self, left: int, right: int, /) -> bool:
        """Compare absolute values."""
        return abs(left) == abs(right)


class Sum:
    """Integer-sum rule."""

    def __init__(
        self,
        name: str,
        arity: int,
        directions: frozenset[Direction] = _ALL,
        *,
        residual_slots: frozenset[int] | None = None,
        forward_is_complete: bool = True,
        restriction: RuleRestriction | None = None,
    ) -> None:
        self._name = name
        self._arity = arity
        self._directions = directions
        self._residual_slots = residual_slots
        self._forward_is_complete = forward_is_complete
        self._restriction = restriction

    @property
    def name(self) -> str:
        """Name the rule."""
        return self._name

    @property
    def arity(self) -> int:
        """Report how many premises the rule takes."""
        return self._arity

    @property
    def directions(self) -> frozenset[Direction]:
        """Report the directions this instance computes."""
        return self._directions

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return every premise slot when residuation is available."""
        if self._residual_slots is not None:
            return self._residual_slots

        if "residual" not in self._directions:
            return frozenset()

        return frozenset(range(self._arity))

    @property
    def forward_is_complete(self) -> bool:
        """Enumerate every conclusion the rule admits."""
        return self._forward_is_complete

    @property
    def matching(self) -> Exact:
        """Read premises under equality."""
        return MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the rule-instance restriction, if declared."""
        return self._restriction

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Sum the premises."""
        if len(inputs) != self._arity:
            return

        yield sum(inputs)

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide whether the conclusion is the sum of the premises."""
        return len(inputs) == self._arity and output == sum(inputs)

    def residuate(self, output: int, known: tuple[Slot[int], ...], /) -> Iterator[int]:
        """Fill one hole from the conclusion and known premises."""
        if len(known) != self._arity or known.count(HOLE) != 1:
            return

        yield output - sum(slot for slot in known if isinstance(slot, int))

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self._name, "arity": self._arity}


class Successor(Sum):
    """Unary integer-successor rule."""

    def __init__(self) -> None:
        super().__init__("successor", 1)

    @override
    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Add one to a single premise."""
        if len(inputs) != 1:
            return

        yield inputs[0] + 1

    @override
    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide whether the output succeeds the premise."""
        return len(inputs) == 1 and output == inputs[0] + 1


class ProbedSum(Sum):
    """Integer-sum rule with declared head constraints."""

    def __init__(
        self,
        name: str,
        probe: RuleProbe,
        restriction: RuleRestriction | None = None,
    ) -> None:
        super().__init__(name, 2, restriction=restriction)
        self._declared_probe = probe

    @property
    @override
    def probe(self) -> RuleProbe:
        """Return the declared premise-head constraints."""
        return self._declared_probe


class StoredSum(ProbedSum):
    """Probed sum storing its matching object."""

    def __init__(
        self,
        name: str,
        matching: Exact,
        probe: RuleProbe,
        restriction: RuleRestriction | None = None,
    ) -> None:
        super().__init__(name, probe, restriction)
        self._stored_matching = matching

    @property
    @override
    def matching(self) -> Exact:
        """Return the stored matching object."""
        return self._stored_matching


class ConstantOne(Sum):
    """Binary rule returning one while recording its premises."""

    def __init__(self) -> None:
        super().__init__("constant-one", 2)
        self.calls: list[tuple[int, ...]] = []

    @override
    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Record a valid pair and return one."""
        if len(inputs) != 2:
            return

        self.calls.append(inputs)
        yield 1

    @override
    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide whether a pair concludes one."""
        return len(inputs) == 2 and output == 1


def _ccg_inventory(
    order: int,
) -> tuple[RuleSet[Term], tuple[Term, ...]]:
    """Build a two-slash CCG inventory over a chain of atoms."""
    atom_names = tuple(f"a{index}" for index in range(6))
    views = ("/", "\\")
    adopted = declared_rules(
        ccg_signature(*atom_names, name="closure-ccg"),
        RuleSetSpec(
            name=f"ccg-composition-{order}",
            rules=tuple(
                ApplicationSpec(view=view)
                if order == 0
                else CompositionSpec(view=view, order=order, direction="kuhlmann")
                for view in views
            ),
        ),
    )
    context = adopted.context
    atoms = tuple(context.syntax.parse(context.store, name) for name in atom_names)
    functors_in_both_directions = tuple(
        context.syntax.parse(
            context.store, f"{atom_names[index]}{view}{atom_names[index + 1]}"
        )
        for view in views
        for index in range(len(atoms) - 1)
    )

    return (adopted.rule_set, (*atoms, *functors_in_both_directions))


def _set(
    *rules: Rule[int], restriction_mode: RestrictionMode = "unrestricted"
) -> RuleSet[int]:
    return RuleSet("sums", rules, MATCHING, Flat(), restriction_mode)


class TestConstruction:
    """Rule-set construction tests."""

    def test_carries_its_declared_properties(self) -> None:
        """Report all declared properties."""
        unary = Sum("unary", 1)
        binary = Sum("binary", 2)
        inventory = _set(unary, binary, restriction_mode="grammar_specific")

        assert inventory.name == "sums"
        assert inventory.rules == (unary, binary)
        assert inventory.matching.name == "exact"
        assert inventory.normal_form.name == "flat"
        assert inventory.restriction_mode == "grammar_specific"
        assert inventory.introduces == (None, None)
        assert inventory.consumes == (None, None)
        assert inventory.index is None

    def test_replacing_the_normal_form_preserves_the_inventory(self) -> None:
        """Change only the derivation-step filter."""
        unary = Sum("unary", 1)
        inventory = RuleSet(
            "sums",
            (unary,),
            MATCHING,
            Flat(),
            "grammar_specific",
            introduces=((1, 2),),
            consumes=((3,),),
        )

        replaced = inventory.with_normal_form(NoNormalForm())

        assert replaced is not inventory
        assert replaced.name == inventory.name
        assert replaced.rules[0] is unary
        assert replaced.matching is inventory.matching
        assert replaced.normal_form.name == "none"
        assert replaced.restriction_mode == inventory.restriction_mode
        assert replaced.introduces == inventory.introduces
        assert replaced.consumes == inventory.consumes

    def test_category_declarations_are_copied_and_aligned_with_rules(self) -> None:
        """Keep finite effects aligned with their runtime rule identities."""
        unary = Sum("unary", 1)
        binary = Sum("binary", 2)
        introduced = [1, 2]
        consumed = [3]
        inventory = RuleSet(
            "sums",
            (unary, binary),
            MATCHING,
            Flat(),
            introduces=(introduced, None),
            consumes=(None, consumed),
        )
        introduced.append(4)
        consumed.append(5)

        assert inventory.introduces == ((1, 2), None)
        assert inventory.consumes == (None, (3,))

    def test_introduced_categories_cover_every_rule(self) -> None:
        """Reject a category declaration that loses rule alignment."""
        with pytest.raises(ValueError, match="not the rule count"):
            RuleSet(
                "sums",
                (Sum("unary", 1), Sum("binary", 2)),
                MATCHING,
                Flat(),
                introduces=((1,),),
            )

    def test_consumed_categories_cover_every_rule(self) -> None:
        """Reject a category declaration that loses rule alignment."""
        with pytest.raises(ValueError, match="not the rule count"):
            RuleSet(
                "sums",
                (Sum("unary", 1), Sum("binary", 2)),
                MATCHING,
                Flat(),
                consumes=((1,),),
            )

    def test_carries_an_optional_matching_binary_index(self) -> None:
        """Return an injected index over the same binary rules."""
        unary = Sum("unary", 1)
        binary = Sum("binary", 2)
        index = RuleIndex((unary, binary), OneHead())
        inventory = RuleSet(
            "sums",
            (unary, binary),
            MATCHING,
            Flat(),
            index=index,
        )

        assert inventory.index is index

    def test_refuses_an_index_over_a_different_binary_inventory(self) -> None:
        """Reject an index whose binary rule identities differ."""
        binary = Sum("binary", 2)
        other = Sum("other", 2)
        index = RuleIndex((other,), OneHead())

        with pytest.raises(ValueError, match="binary rules in declaration order"):
            RuleSet("sums", (binary,), MATCHING, Flat(), index=index)

    def test_refuses_an_index_that_leaves_out_the_unary_rules(self) -> None:
        """Reject an index built over the binary rules alone."""
        unary = Sum("unary", 1)
        binary = Sum("binary", 2)
        index = RuleIndex((binary,), OneHead())

        with pytest.raises(ValueError, match="unary rules in declaration order"):
            RuleSet("sums", (unary, binary), MATCHING, Flat(), index=index)

    def test_refuses_an_index_over_a_different_unary_inventory(self) -> None:
        """Reject a unary index whose rule identities differ."""
        unary = Sum("unary", 1)
        other = Sum("other", 1)
        index = RuleIndex((other,), OneHead())

        with pytest.raises(ValueError, match="unary rules in declaration order"):
            RuleSet("sums", (unary,), MATCHING, Flat(), index=index)

    def test_packs_under_the_grammar_matching_when_every_rule_shares_it(
        self,
    ) -> None:
        """Pack conclusions under the congruence every rule matches under."""
        assert _set(Sum("unary", 1), Sum("binary", 2)).packing is MATCHING
        assert _set().packing is MATCHING

    def test_packs_under_identity_once_a_rule_matches_under_its_own_congruence(
        self,
    ) -> None:
        """Keep the grammar congruence and pack every distinct term apart."""
        grammar = Absolute()
        inventory = RuleSet("sums", (Sum("binary", 2),), grammar, Flat())

        assert inventory.matching is grammar
        assert type(inventory.packing) is Identity
        assert not inventory.packing.equivalent(-2, 2)

    def test_the_mode_defaults_to_unrestricted(self) -> None:
        """Default to unrestricted rules."""
        assert _set(Sum("binary", 2)).restriction_mode == "unrestricted"

    def test_unrestricted_mode_rejects_a_restricted_rule_by_name(self) -> None:
        """Reject a rule-instance restriction in unrestricted mode."""
        rule = Sum("binary", 2, restriction=RuleRestriction(["S"]))

        with pytest.raises(ValueError, match="restricted rule 'binary'"):
            _set(rule)

    def test_grammar_specific_mode_accepts_empty_restricted_and_mixed_rules(
        self,
    ) -> None:
        """Accept each grammar-specific restriction arrangement."""
        restricted = Sum("restricted", 2, restriction=RuleRestriction(["S"]))
        open_rule = Sum("open", 2)

        assert _set(restriction_mode="grammar_specific").rules == ()
        assert _set(restricted, restriction_mode="grammar_specific").rules == (
            restricted,
        )
        assert _set(
            restricted,
            open_rule,
            restriction_mode="grammar_specific",
        ).rules == (restricted, open_rule)

    def test_a_repeated_rule_name_is_refused_with_both_positions_named(self) -> None:
        """Reject duplicate rule names."""
        with pytest.raises(ValueError, match="'binary' twice, at positions 0 and 2"):
            _set(Sum("binary", 2), Sum("unary", 1), Sum("binary", 2))

    def test_the_refusal_names_the_inventory(self) -> None:
        """Name the rule set and the colliding rule."""
        with pytest.raises(ValueError, match="the rule set 'sums' names"):
            _set(Sum("binary", 2), Sum("binary", 2))

    def test_an_undeclared_restriction_mode_is_refused(self) -> None:
        """Reject an unknown restriction mode."""
        with pytest.raises(ValueError, match="unrestricted, grammar_specific"):
            _set(
                Sum("binary", 2),
                restriction_mode=cast("RestrictionMode", "local"),
            )

    def test_an_empty_name_is_refused(self) -> None:
        """Reject an empty inventory name."""
        with pytest.raises(ValueError, match="a rule set's name is empty"):
            RuleSet("", (Sum("binary", 2),), MATCHING, Flat())

    def test_an_inventory_with_no_rules_constructs(self) -> None:
        """Accept an empty inventory."""
        assert _set().rules == ()

    def test_a_rule_set_refuses_private_slot_mutation_and_deletion(self) -> None:
        """Freeze the rule-set fields after construction."""
        inventory = _set(Sum("binary", 2))

        with pytest.raises(AttributeError, match="immutable"):
            inventory.__setattr__("_name", "changed")

        with pytest.raises(AttributeError, match="immutable"):
            inventory.__delattr__("_rules")

    def test_copy_deepcopy_and_pickle_use_public_reconstruction(self) -> None:
        """Reconstruct a frozen indexed rule set through standard protocols."""
        matching = Exact()
        restriction = RuleRestriction(("S",), ("NP",))
        rule = StoredSum(
            "binary",
            matching,
            RuleProbe(2, ((None, None),)),
            restriction,
        )
        inventory = RuleSet(
            "reconstructable",
            (rule,),
            matching,
            Flat(),
            "grammar_specific",
            index=RuleIndex((rule,), OneHead()),
        )
        restored = cast(
            "RuleSet[int]",
            pickle.loads(pickle.dumps(inventory)),
        )

        for rebuilt in (copy.copy(inventory), copy.deepcopy(inventory), restored):
            assert rebuilt.name == inventory.name
            assert tuple(each.name for each in rebuilt.rules) == ("binary",)
            assert rebuilt.rules[0].matching is rebuilt.matching
            assert rebuilt.normal_form.name == "flat"
            assert rebuilt.restriction_mode == "grammar_specific"
            assert rebuilt.rules[0].restriction == restriction
            assert rebuilt.index is not None
            assert rebuilt.index.rules[0] is rebuilt.rules[0]


class TestSelection:
    """Rule-selection tests."""

    def test_of_arity_selects_in_declaration_order(self) -> None:
        """Return one arity in declaration order."""
        first = Sum("first", 2)
        unary = Sum("unary", 1)
        second = Sum("second", 2)
        inventory = _set(first, unary, second)

        assert inventory.of_arity(2) == (first, second)
        assert inventory.of_arity(1) == (unary,)
        assert inventory.of_arity(3) == ()

    def test_a_negative_arity_is_refused(self) -> None:
        """Reject a negative arity."""
        with pytest.raises(ValueError, match="a selected arity is negative"):
            _set(Sum("binary", 2)).of_arity(-1)

    def test_supporting_selects_by_published_direction(self) -> None:
        """Return rules that publish a direction."""
        complete = Sum("complete", 2)
        forward_only = Sum("forward-only", 2, _FORWARD)
        inventory = _set(complete, forward_only)

        assert inventory.supporting("forward") == (complete, forward_only)
        assert inventory.supporting("check") == (complete,)
        assert inventory.supporting("residual") == (complete,)


WITH_PROPOSER: Final = GenerativeParsing(proposer_available=True)
WITHOUT_PROPOSER: Final = GenerativeParsing(proposer_available=False)


class Unary:
    """A requirement that every rule take one premise."""

    @property
    def name(self) -> str:
        """Return the requirement's name."""
        return "unary-only"

    def diagnose[T](self, rule: Rule[T], /) -> Diagnostic | None:
        """Report a rule with more than one premise."""
        if rule.arity == 1:
            return None
        return Diagnostic("not-unary", self.name, "not unary", rules=(rule.name,))


class TestValidation:
    """Algorithm-scoped capability-validation tests."""

    def test_complete_forward_rules_support_forward_enumeration(self) -> None:
        """Accept rules that publish complete forward enumeration."""
        assert _set(Sum("binary", 2)).validate(FORWARD_ENUMERATION) == ()

    def test_forward_enumeration_reports_every_incomplete_rule(self) -> None:
        """Report incomplete forward capabilities in order."""
        check_only = Sum("check-only", 1, frozenset({"check"}))
        partial = Sum("partial", 2, _ALL, forward_is_complete=False)
        findings = _set(check_only, partial).validate(FORWARD_ENUMERATION, strict=False)

        assert tuple(finding.code for finding in findings) == (
            "forward_enumeration_unavailable",
            "forward_enumeration_unavailable",
        )
        assert tuple(finding.rules for finding in findings) == (
            ("check-only",),
            ("partial",),
        )
        assert all(finding.requirement == "forward_enumeration" for finding in findings)

    def test_generative_parsing_accepts_complete_forward_rules(self) -> None:
        """Accept complete forward enumeration without a proposer."""
        assert _set(Sum("binary", 2)).validate(WITHOUT_PROPOSER) == ()

    def test_checkable_rules_require_a_proposer_for_generation(self) -> None:
        """Require proposals for incomplete forward rules."""
        check_only = Sum("check-only", 1, frozenset({"check"}))
        partial = Sum("partial", 2, _ALL, forward_is_complete=False)
        inventory = _set(check_only, partial)
        findings = inventory.validate(WITHOUT_PROPOSER, strict=False)

        assert tuple(finding.code for finding in findings) == (
            "proposer_required",
            "proposer_required",
        )
        assert inventory.validate(WITH_PROPOSER) == ()

    def test_uncheckable_rules_cannot_generate_even_with_a_proposer(self) -> None:
        """Reject rules without forward output or checking."""
        residual_only = Sum("residual-only", 1, frozenset({"residual"}))
        finding = _set(residual_only).validate(WITH_PROPOSER, strict=False)[0]

        assert finding.code == "generation_unavailable"
        assert finding.rules == ("residual-only",)
        assert finding.requirement == "generative_parsing"

    def test_strict_validation_raises_with_the_full_diagnostic_tuple(self) -> None:
        """Raise one error with all diagnostics in order."""
        inventory = _set(
            Sum("none", 2, frozenset({"check"})),
            Sum("partial", 2, frozenset({"residual"})),
        )
        expected = inventory.validate(WITHOUT_PROPOSER, strict=False)

        with pytest.raises(RuleSetValidationError) as caught:
            inventory.validate(WITHOUT_PROPOSER)

        assert caught.value.ruleset == "sums"
        assert caught.value.requirement == "generative_parsing"
        assert caught.value.diagnostics == expected

    def test_a_declared_requirement_names_its_findings(self) -> None:
        """Run a requirement the rule set does not know by name."""
        findings = _set(Sum("binary", 2), Sum("unary", 1)).validate(
            Unary(), strict=False
        )

        assert tuple(finding.rules for finding in findings) == (("binary",),)
        assert findings[0].requirement == "unary-only"

    def test_strictness_must_be_a_boolean(self) -> None:
        """Reject a non-boolean strictness switch."""
        with pytest.raises(TypeError, match="not a boolean"):
            _set().validate(FORWARD_ENUMERATION, strict=cast("bool", 1))

    def test_proposer_availability_must_be_a_boolean(self) -> None:
        """Reject a non-boolean proposer switch."""
        with pytest.raises(TypeError, match="not a boolean"):
            GenerativeParsing(proposer_available=cast("bool", 1))

    def test_requirements_render_their_constructor_calls(self) -> None:
        """Render each built-in requirement as its rebuilding call."""
        assert repr(FORWARD_ENUMERATION) == "ForwardEnumeration()"
        assert repr(WITH_PROPOSER) == "GenerativeParsing(proposer_available=True)"


class TestClosure:
    """Bounded forward-closure tests."""

    def test_erases_and_deduplicates_the_seed_inventory(self) -> None:
        """Return one matching representative per seed class."""
        matching = Absolute()
        inventory = RuleSet("absolute", (), matching, Flat())

        report = inventory.closure((-3, 3, -2, 2), max_depth=0, budget=4)

        assert report.reached == (3, 2)
        assert report.saturated
        assert report.category_set == "finite"
        assert report.escaping == ()

    def test_keeps_every_term_apart_once_a_rule_matches_more_finely(self) -> None:
        """Reach exact terms when a rule tells apart what the grammar merges."""
        inventory = RuleSet("absolute", (Successor(),), Absolute(), Flat())

        report = inventory.closure((-3, 3, -2, 2), max_depth=0, budget=4)

        assert report.reached == (-3, 3, -2, 2)

    def test_application_only_inventory_saturates_without_escapees(self) -> None:
        """Saturate a two-slash application inventory."""
        inventory, seeds = _ccg_inventory(0)

        report = inventory.closure(seeds, max_depth=4, budget=64)

        assert report.reached == seeds
        assert report.saturated
        assert report.steps == 0
        assert report.category_set == "finite"
        assert report.never_applicable == ()
        assert report.escaping == ()

    def test_first_order_composition_exhausts_its_node_budget(self) -> None:
        """Name an escapee from bounded two-slash composition."""
        inventory, seeds = _ccg_inventory(1)

        report = inventory.closure(seeds, max_depth=8, budget=20)

        assert len(report.reached) == report.budget
        assert not report.saturated
        assert report.category_set == "unknown"
        assert len(report.escaping) == 1
        assert report.escaping[0] not in report.reached
        assert report.never_applicable == ()

    def test_depth_exhaustion_names_the_first_excluded_category(self) -> None:
        """Report the deterministic category beyond the depth bound."""
        inventory = _set(Successor())

        report = inventory.closure((0,), max_depth=2, budget=8)

        assert report.reached == (0, 1, 2)
        assert report.steps == 2
        assert not report.saturated
        assert report.category_set == "unknown"
        assert report.escaping == (3,)

    def test_later_steps_skip_tuples_without_a_frontier_member(self) -> None:
        """Apply each old premise tuple only on its first eligible step."""
        rule = ConstantOne()

        report = _set(rule).closure((0,), max_depth=4, budget=4)

        assert report.saturated
        assert report.reached == (0, 1)
        assert rule.calls == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_budget_exhaustion_never_exceeds_the_node_cap(self) -> None:
        """Stop before adding a category beyond the node budget."""
        inventory = _set(Successor())

        report = inventory.closure((0,), max_depth=8, budget=3)

        assert report.reached == (0, 1, 2)
        assert len(report.reached) == report.budget
        assert report.steps == 2
        assert not report.saturated
        assert report.category_set == "unknown"
        assert report.escaping == (3,)

    def test_a_seed_beyond_the_budget_is_the_first_escapee(self) -> None:
        """Apply the node cap to erased seeds in their given order."""
        report = _set().closure((4, 2, 6), max_depth=3, budget=2)

        assert report.reached == (4, 2)
        assert report.steps == 0
        assert report.escaping == (6,)
        assert not report.saturated

    @pytest.mark.parametrize(
        ("max_arity", "max_size"),
        [(2, None), (None, 3), (2, 3)],
    )
    def test_category_bounds_keep_every_in_bound_closure_step(
        self,
        max_arity: int | None,
        max_size: int | None,
    ) -> None:
        """Stop exactly before the first conclusion outside either dimension."""
        report = _set(Successor()).closure(
            (0,),
            max_depth=8,
            budget=10,
            category_measure=IntegerMeasure(),
            max_category_arity=max_arity,
            max_category_size=max_size,
        )

        assert report.reached == (0, 1, 2)
        assert report.escaping == (3,)
        assert not report.saturated

    def test_category_bounds_need_the_owner_measure(self) -> None:
        """Refuse a bound whose formalism supplied no structural measure."""
        with pytest.raises(ValueError, match="declared category measure"):
            _set().closure(
                (),
                max_depth=0,
                budget=1,
                max_category_size=2,
            )

    def test_a_category_bound_applies_to_closure_seeds(self) -> None:
        """Report the first lexical category outside a declared bound."""
        report = _set().closure(
            (4, 1),
            max_depth=0,
            budget=8,
            category_measure=IntegerMeasure(),
            max_category_arity=3,
        )

        assert report.reached == ()
        assert report.escaping == (4,)
        assert not report.saturated

    def test_binary_closure_uses_the_injected_probe_table(self) -> None:
        """Exercise no indexed rule when the declaration requests no rounds."""
        even = ProbedSum(
            "even",
            RuleProbe(
                2,
                ((frozenset({"even"}), frozenset({"even"})),),
            ),
        )
        odd = ProbedSum(
            "odd",
            RuleProbe(
                2,
                ((frozenset({"odd"}), frozenset({"odd"})),),
            ),
        )
        index = RuleIndex((even, odd), ParityHeads())
        inventory = RuleSet("parity", (even, odd), MATCHING, Flat(), index=index)

        report = inventory.closure((2, 4), max_depth=0, budget=8)

        assert report.reached == (2, 4)
        assert not report.saturated
        assert report.escaping == ()
        assert report.never_applicable == ("even", "odd")

    def test_zero_rounds_need_no_forward_enumeration_capability(self) -> None:
        """Return seed-only closure for a check-only relational inventory."""
        inventory = _set(Sum("check-only", 1, frozenset({"check"})))

        report = inventory.closure((0,), max_depth=0, budget=8)

        assert report.reached == (0,)
        assert not report.saturated
        assert report.never_applicable == ("check-only",)

    def test_never_applicable_follows_rule_declaration_order(self) -> None:
        """List every unfired indexed rule in declaration order."""
        odd_first = ProbedSum(
            "odd-first",
            RuleProbe(
                2,
                ((frozenset({"odd"}), frozenset({"odd"})),),
            ),
        )
        odd_second = ProbedSum(
            "odd-second",
            RuleProbe(
                2,
                ((frozenset({"odd"}), frozenset({"odd"})),),
            ),
        )
        index = RuleIndex((odd_first, odd_second), ParityHeads())
        inventory = RuleSet(
            "parity",
            (odd_first, odd_second),
            MATCHING,
            Flat(),
            index=index,
        )

        report = inventory.closure((2,), max_depth=2, budget=8)

        assert report.saturated
        assert report.never_applicable == ("odd-first", "odd-second")

    @pytest.mark.parametrize("field", ["max_depth", "budget"])
    def test_closure_bounds_reject_invalid_integers(self, field: str) -> None:
        """Reject negative and non-integer closure bounds."""
        if field == "max_depth":
            negative = partial(_set().closure, (), max_depth=-1, budget=1)
            malformed = partial(
                _set().closure, (), max_depth=cast("int", 1.5), budget=1
            )
        else:
            negative = partial(_set().closure, (), max_depth=1, budget=-1)
            malformed = partial(
                _set().closure, (), max_depth=1, budget=cast("int", 1.5)
            )

        with pytest.raises(ValueError, match="negative"):
            negative()

        with pytest.raises(TypeError, match="not an integer"):
            malformed()

    def test_closure_requires_complete_forward_enumeration(self) -> None:
        """Reject closure over a rule without complete forward output."""
        inventory = _set(Sum("check-only", 1, frozenset({"check"})))

        with pytest.raises(RuleSetValidationError) as caught:
            inventory.closure((0,), max_depth=2, budget=8)

        assert caught.value.requirement == "forward_enumeration"


class TestAsAnInventory:
    """Rule-set collection tests."""

    def test_iterating_yields_the_rules_in_order(self) -> None:
        """Yield rules in declaration order."""
        unary = Sum("unary", 1)
        binary = Sum("binary", 2)

        assert tuple(_set(unary, binary)) == (unary, binary)

    def test_the_length_is_the_rule_count(self) -> None:
        """Report how many rules the inventory holds."""
        assert len(_set(Sum("unary", 1), Sum("binary", 2))) == 2

    def test_recovery_reads_a_label_off_the_inventory(self) -> None:
        """Recover a rule label through the inventory."""
        binary = Sum("binary", 2)
        inventory = _set(Sum("unary", 1), binary, Sum("forward-only", 2, _FORWARD))

        assert inventory.recover((2, 3), 5) == (binary,)
        assert inventory.recover((2, 3), 6) == ()
        assert recover(inventory, (2, 3), 5) == (binary,)
        assert recover(inventory, (2, 3), 6) == ()

    def test_the_repr_names_the_declared_parameters(self) -> None:
        """Include declared parameters in the representation."""
        assert repr(_set(Sum("binary", 2))) == (
            "RuleSet(name='sums', rules=1, matching='exact', "
            "normal_form='flat', restriction_mode='unrestricted')"
        )


class Declared(Sum):
    """Integer-sum rule declaring the relation it denotes."""

    def __init__(self, name: str, arity: int, offset: int = 0) -> None:
        super().__init__(name, arity)
        self._offset = offset

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the arity and offset, which fix the relation."""
        return {"arity": self.arity, "offset": self._offset}


class TestCoincidence:
    """Refusal of inventories declaring one relation twice."""

    def test_two_rules_declaring_one_relation_are_refused_by_name(self) -> None:
        """Name both rules and the parameters they differ in."""
        with pytest.raises(
            ValueError,
            match=r"rule set 'sums' declares the same relation twice: "
            r"'first' and 'second' differ only in 'rule'",
        ):
            refuse_coinciding_rules(
                "sums", (Declared("first", 2), Declared("second", 2))
            )

    def test_rules_declaring_different_relations_are_accepted(self) -> None:
        """Accept rules whose declared relations differ."""
        refuse_coinciding_rules(
            "sums", (Declared("first", 2), Declared("second", 2, offset=1))
        )
        refuse_coinciding_rules("sums", (Declared("first", 2), Declared("unary", 1)))

    def test_a_rule_declaring_no_relation_coincides_with_none(self) -> None:
        """Leave rules without a declared relation out of the comparison."""
        refuse_coinciding_rules("sums", (Sum("first", 2), Sum("second", 2)))
        refuse_coinciding_rules("sums", (Sum("first", 2), Declared("second", 2)))

    def test_every_coinciding_pair_is_named(self) -> None:
        """Report each later rule against the first that declared its relation."""
        rules = (
            Declared("a", 2),
            Declared("b", 2),
            Declared("c", 1),
            Declared("d", 1),
        )

        with pytest.raises(ValueError, match=r"'a' and 'b'.*'c' and 'd'"):
            refuse_coinciding_rules("sums", rules)

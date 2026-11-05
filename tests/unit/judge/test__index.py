"""Tests for conservative unary- and binary-rule indexing."""

from typing import TYPE_CHECKING, Final, cast, override

import pytest

from lofi.judge import HeadIndex, NormalFormRole, Rule, RuleIndex, RuleProbe

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge import Direction, RuleRestriction, Slot

_DIRECTIONS: Final[frozenset[Direction]] = frozenset({"forward", "check"})


class Exact:
    """String-identity congruence."""

    @property
    def name(self) -> str:
        """Return the congruence name."""
        return "exact"

    @property
    def has_key(self) -> bool:
        """Return whether classes have representatives."""
        return True

    def key(self, term: str, /) -> str:
        """Return a class representative."""
        return term

    def equivalent(self, left: str, right: str, /) -> bool:
        """Compare two strings."""
        return left == right


class NamedHeads:
    """Finite head codes for strings."""

    def __init__(
        self,
        mapping: Mapping[str, int],
        codes: frozenset[int] | None = None,
    ) -> None:
        self._mapping = dict(mapping)
        self._codes = frozenset(mapping.values()) if codes is None else codes

    @property
    def codes(self) -> frozenset[int]:
        """Return every declared code."""
        return self._codes

    def term(self, term: str, /) -> int:
        """Map a term to its head code."""
        return self._mapping[term]

    def named(self, name: str, /) -> int | None:
        """Map an operator name to its head code when declared."""
        return self._mapping.get(name)


class IndexedRule:
    """Test rule with configurable head constraints."""

    def __init__(self, name: str, arity: int, probe: RuleProbe | None) -> None:
        self._name = name
        self._arity = arity
        self._probe = probe

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return self._arity

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the computable directions."""
        return _DIRECTIONS

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return no residuable premise slots."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Return whether forward enumeration is complete."""
        return True

    @property
    def matching(self) -> Exact:
        """Return exact matching."""
        return Exact()

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Return conservative head constraints or decline indexing."""
        return self._probe

    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        """Enumerate no conclusions."""
        del inputs
        return iter(())

    def admits(self, inputs: tuple[str, ...], output: str, /) -> bool:
        """Reject every relation instance."""
        del inputs, output
        return False

    def residuate(self, output: str, known: tuple[Slot[str], ...], /) -> Iterator[str]:
        """Enumerate no residuals."""
        del output, known
        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {"rule": self._name}


HEADS: Final = NamedHeads({"a": 0, "b": 1, "c": 2})


def test_the_head_index_protocol_is_structural() -> None:
    assert isinstance(HEADS, HeadIndex)


def test_the_rule_protocol_requires_a_probe() -> None:
    assert isinstance(IndexedRule("binary", 2, None), Rule)


def test_a_probe_copies_and_deduplicates_alternatives() -> None:
    alternatives = [
        (frozenset({"a"}), None),
        (frozenset({"a"}), None),
    ]
    probe = RuleProbe(2, alternatives)
    alternatives.append((frozenset({"b"}), None))

    assert probe.arity == 2
    assert probe.alternatives == ((frozenset({"a"}), None),)
    assert probe == RuleProbe(2, ((frozenset({"a"}), None),))
    assert hash(probe) == hash(RuleProbe(2, ((frozenset({"a"}), None),)))
    assert repr(probe) == "RuleProbe(2, ((frozenset({'a'}), None),))"


@pytest.mark.parametrize("arity", [True, 1.5, "2", -1])
def test_a_probe_refuses_an_invalid_arity(arity: float | str) -> None:
    with pytest.raises((TypeError, ValueError), match="probe arity"):
        RuleProbe(cast("int", arity), ())


def test_a_probe_refuses_an_alternative_of_the_wrong_arity() -> None:
    with pytest.raises(ValueError, match="alternative 0 has arity 1, not 2"):
        RuleProbe(2, ((None,),))


@pytest.mark.parametrize("name", ["", 3])
def test_a_probe_refuses_an_invalid_operator_name(name: str | int) -> None:
    with pytest.raises((TypeError, ValueError), match="alternative 0 head"):
        RuleProbe(1, ((frozenset({cast("str", name)}),),))


def test_a_probe_is_frozen() -> None:
    probe = RuleProbe(2, ((None, None),))

    with pytest.raises(AttributeError, match="immutable"):
        probe.__setattr__("arity", 3)

    with pytest.raises(AttributeError, match="immutable"):
        probe.__delattr__("_arity")


def test_the_index_preserves_order_and_keeps_declined_rules() -> None:
    left = IndexedRule("left", 2, RuleProbe(2, ((frozenset({"a"}), None),)))
    always = IndexedRule("always", 2, None)
    right = IndexedRule("right", 2, RuleProbe(2, ((None, frozenset({"b"})),)))
    index = RuleIndex((left, always, right), HEADS)

    assert index.rules == (left, always, right)
    assert index.always == (always,)
    assert index.candidates("a", "b") == (left, always, right)
    assert index.candidates("c", "c") == (always,)


def test_disjunctive_alternatives_select_a_rule_once() -> None:
    disjunctive = IndexedRule(
        "either",
        2,
        RuleProbe(
            2,
            (
                (frozenset({"a"}), None),
                (None, frozenset({"b"})),
            ),
        ),
    )
    index = RuleIndex((disjunctive,), HEADS)

    assert index.candidates("a", "c") == (disjunctive,)
    assert index.candidates("c", "b") == (disjunctive,)
    assert index.candidates("a", "b") == (disjunctive,)
    assert index.candidates("c", "c") == ()


def test_empty_constraints_and_empty_disjunctions_select_nothing() -> None:
    impossible = IndexedRule(
        "impossible",
        2,
        RuleProbe(2, ((frozenset(), None),)),
    )
    absent = IndexedRule("absent", 2, RuleProbe(2, ()))
    index = RuleIndex((impossible, absent), HEADS)

    assert index.candidates("a", "b") == ()


def test_unary_rules_are_indexed_separately_and_keep_declined_rules() -> None:
    unary = IndexedRule("unary", 1, RuleProbe(1, ((frozenset({"a"}),),)))
    always = IndexedRule("unary-always", 1, None)
    binary = IndexedRule("binary", 2, None)
    index = RuleIndex((unary, always, binary), HEADS)

    assert index.rules == (binary,)
    assert index.unary_rules == (unary, always)
    assert index.unary_always == (always,)
    assert index.unary_candidates("a") == (unary, always)
    assert index.unary_candidates("c") == (always,)


def test_a_rule_probe_must_match_its_rule_arity() -> None:
    malformed = IndexedRule("malformed", 2, RuleProbe(1, ((None,),)))

    with pytest.raises(ValueError, match="arity 2 but its probe has arity 1"):
        RuleIndex((malformed,), HEADS)


def test_a_named_head_must_map_to_a_declared_code() -> None:
    heads = NamedHeads({"a": 0, "outside": 4}, frozenset({0}))
    rule = IndexedRule(
        "outside",
        2,
        RuleProbe(2, ((frozenset({"outside"}), None),)),
    )

    with pytest.raises(ValueError, match="outside its codes"):
        RuleIndex((rule,), heads)


def test_an_undeclared_named_head_makes_the_constraint_unsatisfiable() -> None:
    rule = IndexedRule(
        "missing",
        1,
        RuleProbe(1, ((frozenset({"missing"}),),)),
    )

    index = RuleIndex((rule,), HEADS)

    assert index.unary_candidates("a") == ()
    assert index.unary_candidates("b") == ()
    assert index.unary_candidates("c") == ()


@pytest.mark.parametrize("codes", [frozenset(), frozenset({-1}), frozenset({True})])
def test_a_head_index_must_declare_valid_codes(codes: frozenset[int]) -> None:
    heads = NamedHeads({"a": 0}, codes)

    with pytest.raises((TypeError, ValueError), match=r"codes|head code"):
        RuleIndex((), heads)


def test_the_index_copies_its_rule_inventory_and_is_frozen() -> None:
    rule = IndexedRule("binary", 2, None)
    rules = [rule]
    index = RuleIndex(rules, HEADS)
    rules.clear()

    assert index.rules == (rule,)
    assert index.heads is HEADS
    assert index.codes == frozenset({0, 1, 2})

    with pytest.raises(AttributeError, match="immutable"):
        index.__setattr__("rules", ())

    with pytest.raises(AttributeError, match="immutable"):
        index.__delattr__("_rules")


class JoinedRule(IndexedRule):
    """Binary test rule declaring join keys."""

    def join_keys(self, term: str, slot: int, /) -> tuple[str, ...]:
        """Offer the term itself in either slot."""
        del slot
        return (term,)


class GroundRule(IndexedRule):
    """Unary test rule whose premise admits one representative."""

    def __init__(self, name: str, ground: str | None) -> None:
        super().__init__(name, 1, None)
        self._ground = ground

    @property
    def ground_premises(self) -> tuple[str | None, ...]:
        """Return the one premise the rule admits, if any."""
        return (self._ground,)


def test_join_rules_are_listed_apart_from_the_head_rows_they_leave() -> None:
    joined = JoinedRule("joined", 2, None)
    plain = IndexedRule("plain", 2, None)
    index = RuleIndex[str]((plain, joined), cast("HeadIndex[str]", HEADS))

    assert index.joined == (joined,)
    assert index.unjoined(0, 1) == (plain,)
    assert index.candidates("a", "b") == (plain, joined)


def test_a_ground_unary_rule_meets_only_its_representative() -> None:
    first = GroundRule("first", "a")
    open_rule = GroundRule("open", None)
    last = GroundRule("last", "a")
    other = GroundRule("other", "b")
    index = RuleIndex[str](
        (first, open_rule, last, other), cast("HeadIndex[str]", HEADS)
    )

    assert index.unary_candidates("a") == (first, open_rule, last)
    assert index.unary_candidates("b") == (open_rule, other)
    assert index.unary_candidates("c") == (open_rule,)
    assert index.unary_always == (open_rule,)


class Folded(Exact):
    """Case-insensitive congruence over strings."""

    @property
    @override
    def name(self) -> str:
        """Return the congruence name."""
        return "folded"

    @override
    def key(self, term: str, /) -> str:
        """Return the lowercase representative."""
        return term.lower()

    @override
    def equivalent(self, left: str, right: str, /) -> bool:
        """Compare two strings without case."""
        return left.lower() == right.lower()


FOLDED: Final = Folded()


class FoldedGroundRule(GroundRule):
    """Ground unary rule under case-insensitive matching."""

    @property
    @override
    def matching(self) -> Folded:
        """Return case-insensitive matching."""
        return FOLDED


def test_a_ground_unary_rule_meets_every_term_its_representative_covers() -> None:
    folded = FoldedGroundRule("folded", "a")
    heads = NamedHeads({"a": 0, "A": 0, "b": 1})
    index = RuleIndex[str]((folded,), cast("HeadIndex[str]", heads))

    assert index.unary_candidates("A") == (folded,)
    assert index.unary_candidates("a") == (folded,)
    assert index.unary_candidates("b") == ()

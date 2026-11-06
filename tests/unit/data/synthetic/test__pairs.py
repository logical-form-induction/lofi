"""Tests for synthetic labeled-pair sampling."""

from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    overload,
    override,
)

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.data import (
    PairDistribution,
    TypeDistribution,
    TypeSampleReport,
    term_depth,
)
from lofi.data.synthetic import (
    RecursiveTypeSampler,
    RulePairSampler,
    RulePairSamplerContext,
)
from lofi.data.synthetic._pairs import PLUGIN
from lofi.judge import Action, ActionSet
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    RuleSetSpec,
    SubstitutionSpec,
)
from lofi.spec.pair_samplers.rule import RulePairSamplerSpec

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from lofi.judge import Direction
    from lofi.spec import RuleSpec
    from lofi.terms import OpName, Signature, Term, TermStore


CATEGORIES: Final = ccg_signature("x", "y", "z0", "z1", name="synthetic-pairs")


def _rule(declaration: RuleSpec, /) -> SamplingRule:
    """Compile one rule over the shared category signature."""
    (rule,) = declared_rules(
        CATEGORIES, RuleSetSpec(name="pairs", rules=(declaration,))
    ).rule_set

    return rule


CONTEXT: Final = declared_rules(
    CATEGORIES, RuleSetSpec(name="pairs", rules=(ApplicationSpec(view="/"),))
).context
STORE: Final = CONTEXT.store
SIGNATURE: Final = STORE.signature
X: Final = CONTEXT.syntax.parse(STORE, "x")
Y: Final = CONTEXT.syntax.parse(STORE, "y")
Z: Final = (CONTEXT.syntax.parse(STORE, "z0"), CONTEXT.syntax.parse(STORE, "z1"))
FUNCTORS: Final = CONTEXT.functors["/"][0]


class SamplingRule(Protocol):
    """Rule operations used by pair sampling."""

    @property
    def name(self) -> str: ...

    @property
    def arity(self) -> int: ...

    @property
    def directions(self) -> frozenset[Direction]: ...

    @property
    def forward_is_complete(self) -> bool: ...

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]: ...

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool: ...


class FixedTypeSampler:
    """A type sampler that repeats one ordered daughter pair."""

    def __init__(self, inputs: Sequence[Term]) -> None:
        self._inputs = tuple(inputs)
        self._inventory = FixedTypeInventory(inputs)

    @property
    def inventory(self) -> FixedTypeInventory:
        """Return the finite terms from which admissible pairs are built."""
        return self._inventory

    @property
    def signature(self) -> Signature:
        """Return the category signature."""
        return SIGNATURE

    def sample(
        self,
        count: int,
        distribution: TypeDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[Term, ...], TypeSampleReport]:
        """Repeat the fixed daughters without consulting the seed."""
        del distribution, seed
        terms = tuple(self._inputs[at % len(self._inputs)] for at in range(count))
        depths: dict[int, int] = {}

        for term in terms:
            depth = term_depth(term)
            depths[depth] = depths.get(depth, 0) + 1

        return terms, TypeSampleReport(count, depths, {})


class FixedTypeInventory:
    """The stable finite inventory of a fixed sampler fixture."""

    def __init__(self, terms: Sequence[Term]) -> None:
        self._terms = tuple(dict.fromkeys(terms))

    @property
    def store(self) -> TermStore:
        """Return the store owning every candidate."""
        return STORE

    @property
    def signature(self) -> Signature:
        """Return the shared candidate signature."""
        return SIGNATURE

    @property
    def max_size(self) -> int:
        """Return the largest candidate node count."""
        return max((_term_size(term) for term in self._terms), default=0)

    @property
    def candidates(self) -> tuple[Term, ...]:
        """Return all candidates in stable order."""
        return self._terms

    def at_depth(self, depth: int, /) -> tuple[Term, ...]:
        """Return candidates at one exact depth."""
        return tuple(term for term in self._terms if term_depth(term) == depth)

    def through_depth(self, max_depth: int, /) -> tuple[Term, ...]:
        """Return candidates no deeper than the bound."""
        return tuple(term for term in self._terms if term_depth(term) <= max_depth)

    def __iter__(self) -> Iterator[Term]:
        """Iterate over candidates in stable order."""
        return iter(self._terms)

    def __len__(self) -> int:
        """Return the candidate count."""
        return len(self._terms)

    @overload
    def __getitem__(self, index: int, /) -> Term: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[Term, ...]: ...

    def __getitem__(self, index: int | slice, /) -> Term | tuple[Term, ...]:
        """Return one candidate or a candidate slice."""
        return self._terms[index]


def _term_size(term: Term, /) -> int:
    """Return one term's recursive node count."""
    return 1 + sum(_term_size(child) for child in term.args)


class Inventory:
    """Rules indexed by arity."""

    def __init__(self, rules: Sequence[SamplingRule]) -> None:
        self.rules = tuple(rules)

    def of_arity(self, k: int) -> tuple[SamplingRule, ...]:
        """Select rules with the requested premise count."""
        return tuple(rule for rule in self.rules if rule.arity == k)

    def __iter__(self) -> Iterator[SamplingRule]:
        """Return rules in declaration order."""
        return iter(self.rules)


class RejectRule:
    """A binary rule that refuses every instance."""

    name = "reject"
    arity = 2
    directions: frozenset[Direction] = frozenset({"forward"})
    forward_is_complete = True

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate no conclusions."""
        del inputs
        return iter(())

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Refuse the candidate."""
        del inputs, output
        return False


class UnaryRule:
    """A unary rule that maps x to y."""

    name = "unary"
    arity = 1
    directions: frozenset[Direction] = frozenset({"forward"})
    forward_is_complete = True

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Yield y from x."""
        if inputs == (X,):
            yield Y

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Accept exactly x to y."""
        return inputs == (X,) and output is Y


class NegativeArityRule(UnaryRule):
    """A malformed rule whose premise count is negative."""

    name = "negative"
    arity = -1


class SeedSelectiveTypeSampler(FixedTypeSampler):
    """A type sampler that emits valid daughters for half of derived seeds."""

    @override
    def sample(
        self,
        count: int,
        distribution: TypeDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[Term, ...], TypeSampleReport]:
        """Select valid or invalid daughters from the derived seed."""
        selected = self._inputs if seed % 2 else (X, X)
        terms = tuple(selected[at % 2] for at in range(count))
        depths: dict[int, int] = {}

        for term in terms:
            depth = term_depth(term)
            depths[depth] = depths.get(depth, 0) + 1

        del distribution
        return terms, TypeSampleReport(count, depths, {})


class DeepConclusionRule:
    """A rule whose only conclusion exceeds the configured test bound."""

    name = "deep"
    arity = 2
    directions: frozenset[Direction] = frozenset({"forward"})
    forward_is_complete = True

    def __init__(self) -> None:
        self.output = FUNCTORS.make(FUNCTORS.forward_slash, X, Y)

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Yield the deep conclusion."""
        if len(inputs) == 2:
            yield self.output

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Accept only the declared deep conclusion."""
        return len(inputs) == 2 and output is self.output


class RuleCase(NamedTuple):
    """One sampler-to-rule equation."""

    name: str
    rule: SamplingRule
    inputs: tuple[Term, Term]
    output: Term


def _spine(foot: Term, slashes: Sequence[OpName]) -> Term:
    """Build a result spine around a foot."""
    term = foot

    for slash, argument in reversed(tuple(zip(slashes, Z, strict=False))):
        term = FUNCTORS.make(slash, term, argument)

    return term


def _composition_case(
    role: Literal["forward", "backward"], order: int, *, crossed: bool
) -> RuleCase:
    """Build a valid generalized-composition instance."""
    slash = FUNCTORS.forward_slash if role == "forward" else FUNCTORS.backward_slash
    deepest = FUNCTORS.opposite(slash) if crossed else slash
    slashes = (deepest,) * order
    principal = FUNCTORS.make(slash, X, Y)
    container = _spine(Y, slashes)
    output = _spine(X, slashes)
    inputs = (principal, container) if role == "forward" else (container, principal)
    view = "/" if role == "forward" else "\\"
    rule = _rule(
        ApplicationSpec(view=view)
        if order == 0
        else CompositionSpec(
            view=view, order=order, direction="unrestricted", crossed=crossed
        )
    )
    variant = "crossed" if crossed else "harmonic"

    return RuleCase(f"{role}-composition-{order}-{variant}", rule, inputs, output)


def _substitution_case(
    role: Literal["forward", "backward"], *, crossed: bool
) -> RuleCase:
    """Build a valid first-order substitution instance."""
    slash = FUNCTORS.forward_slash if role == "forward" else FUNCTORS.backward_slash
    deepest = FUNCTORS.opposite(slash) if crossed else slash
    slashes = (deepest,)
    principal = FUNCTORS.make(slash, X, Y)
    functor = _spine(principal, slashes)
    container = _spine(Y, slashes)
    output = _spine(X, slashes)
    inputs = (functor, container) if role == "forward" else (container, functor)
    rule = _rule(
        SubstitutionSpec(
            view="/" if role == "forward" else "\\", order=1, crossed=crossed
        )
    )
    variant = "crossed" if crossed else "harmonic"

    return RuleCase(f"{role}-substitution-{variant}", rule, inputs, output)


BASE_CASES: Final = tuple(
    _composition_case(role, order, crossed=crossed)
    for order, crossed in ((0, False), (1, False), (1, True), (2, False), (2, True))
    for role in ("forward", "backward")
) + tuple(
    _substitution_case(role, crossed=crossed)
    for crossed in (False, True)
    for role in ("forward", "backward")
)
RULE_CASES: Final = BASE_CASES
MAX_DEPTH: Final = 5
DEPTH_MATRIX: Final = tuple(
    (case, bound)
    for bound in range(1, MAX_DEPTH + 1)
    for case in RULE_CASES
    if max(term_depth(term) for term in (*case.inputs, case.output)) <= bound
)


@pytest.mark.parametrize("case", RULE_CASES, ids=lambda case: case.name)
def test_sampler_to_rule_equations(case: RuleCase) -> None:
    rule = case.rule
    identity = Action("identity")
    actions = ActionSet((identity, Action(rule.name, rule.name)), identity)
    inventory = Inventory((rule,))
    sampler = RulePairSampler(
        FixedTypeSampler(case.inputs),
        inventory,
        actions,
    )
    pair_distribution = PairDistribution(
        max_depth=max(term_depth(term) for term in (*case.inputs, case.output)),
    )
    pairs, report = sampler.sample(30, pair_distribution, seed=13)

    assert len(pairs) == 30
    assert report.emitted == 30
    assert all(pair.output is case.output for pair in pairs)
    assert all(pair.actions == (actions.actions[1],) for pair in pairs)
    assert all(rule.admits(pair.inputs, pair.output) for pair in pairs)


@pytest.mark.parametrize(
    ("case", "bound"),
    DEPTH_MATRIX,
    ids=lambda value: value.name if isinstance(value, RuleCase) else f"depth-{value}",
)
def test_every_applicable_rule_instance_at_each_depth_emits_thirty_bounded_pairs(
    case: RuleCase, bound: int
) -> None:
    identity = Action("identity")
    action = Action(case.rule.name, case.rule.name)
    rejecting_rule = RejectRule()
    rejecting_action = Action(rejecting_rule.name, rejecting_rule.name)
    inventory = Inventory((case.rule, rejecting_rule))
    sampler = RulePairSampler(
        FixedTypeSampler(case.inputs),
        inventory,
        ActionSet((identity, action, rejecting_action), identity),
    )
    pairs, _ = sampler.sample(
        30,
        PairDistribution(max_depth=bound),
        seed=bound,
    )

    assert len(pairs) == 30
    assert all(
        term_depth(term) <= bound
        for pair in pairs
        for term in (*pair.inputs, pair.output)
    )
    assert all(pair.actions == (action,) for pair in pairs)

    for pair in pairs:
        labeled = {label.rule for label in pair.actions}
        unlabeled_rules = tuple(
            rule for rule in inventory.rules if rule.name not in labeled
        )

        assert labeled
        assert unlabeled_rules
        assert all(
            rule.admits(pair.inputs, pair.output)
            for rule in inventory.rules
            if rule.name in labeled
        )
        assert all(
            not rule.admits(pair.inputs, pair.output) for rule in unlabeled_rules
        )


@pytest.mark.parametrize(
    ("role", "crossed"),
    [
        ("forward", False),
        ("backward", False),
    ],
)
def test_order_zero_variants_remain_application_rules(
    role: Literal["forward", "backward"], *, crossed: bool
) -> None:
    case = _composition_case(role, 0, crossed=crossed)

    assert tuple(case.rule.apply(case.inputs)) == (case.output,)
    assert case.rule.admits(case.inputs, case.output)


def test_overlapping_rules_receive_both_exact_labels() -> None:
    # Order one, not zero: at order zero there is no spine to cross, so the two
    # variants would be one rule under two names rather than two rules that
    # happen to derive the same conclusion. Under an unrestricted direction the
    # order-one variants each admit the other's output, which is a real overlap.
    harmonic = _composition_case("forward", 1, crossed=False)
    crossed = _composition_case("forward", 1, crossed=True)
    identity = Action("identity")
    labels = (
        identity,
        Action(harmonic.rule.name, harmonic.rule.name),
        Action(crossed.rule.name, crossed.rule.name),
    )
    sampler = RulePairSampler(
        FixedTypeSampler(harmonic.inputs),
        Inventory((harmonic.rule, crossed.rule)),
        ActionSet(labels, identity),
    )
    pairs, _ = sampler.sample(1, PairDistribution(max_depth=2), seed=3)

    assert pairs[0].output is harmonic.output
    assert pairs[0].actions == labels[1:]


def test_unary_and_binary_rules_are_sampled_and_reported_by_arity() -> None:
    case = _composition_case("forward", 1, crossed=False)
    unary = UnaryRule()
    identity = Action("identity")
    unary_action = Action(unary.name, unary.name)
    binary_action = Action(case.rule.name, case.rule.name)
    sampler = RulePairSampler(
        FixedTypeSampler((X, Y, *case.inputs)),
        Inventory((unary, case.rule)),
        ActionSet((identity, unary_action, binary_action), identity),
    )
    pairs, report = sampler.sample(1_000, PairDistribution(max_depth=3), seed=607)
    observed = {
        arity: sum(len(pair.inputs) == arity for pair in pairs) for arity in (1, 2)
    }

    assert dict(report.arities) == observed
    assert all(count > 0 for count in observed.values())
    assert report.arity_distribution[1] == observed[1] / len(pairs)
    assert report.arity_distribution[2] == observed[2] / len(pairs)


def test_the_rule_sampler_plugin_builds_from_its_declared_context() -> None:
    unary = UnaryRule()
    identity = Action("identity")
    action = Action(unary.name, unary.name)
    context = RulePairSamplerContext(
        FixedTypeSampler((X, Y)),
        Inventory((unary,)),
        ActionSet((identity, action), identity),
    )
    sampler = PLUGIN.construct(RulePairSamplerSpec(), context)

    pairs, report = sampler.sample(12, PairDistribution(max_depth=1), seed=41)

    assert len(pairs) == report.emitted == 12
    assert report.arities == {1: 12}
    assert all(pair.inputs == (X,) and pair.output is Y for pair in pairs)


def test_a_rule_sampler_refuses_a_negative_declared_arity() -> None:
    rule = NegativeArityRule()
    identity = Action("identity")

    with pytest.raises(ValueError, match=r"arity of rule 'negative'.*negative"):
        RulePairSampler(
            FixedTypeSampler((X,)),
            Inventory((rule,)),
            ActionSet((identity, Action(rule.name, rule.name)), identity),
        )


def test_admissible_inventory_sampling_emits_the_requested_count() -> None:
    case = _composition_case("forward", 0, crossed=False)
    identity = Action("identity")
    action = Action(case.rule.name, case.rule.name)
    sampler = RulePairSampler(
        SeedSelectiveTypeSampler(case.inputs),
        Inventory((case.rule,)),
        ActionSet((identity, action), identity),
    )
    pairs, report = sampler.sample(80, PairDistribution(max_depth=2), seed=12)

    assert len(pairs) == report.emitted == 80
    assert report.emitted == 80


def test_conclusions_above_the_depth_bound_are_rejected() -> None:
    rule = DeepConclusionRule()
    identity = Action("identity")
    sampler = RulePairSampler(
        FixedTypeSampler((X, Y)),
        Inventory((rule,)),
        ActionSet((identity, Action(rule.name, rule.name)), identity),
    )

    with pytest.raises(ValueError, match="no admissible rule instance"):
        sampler.sample(1, PairDistribution(max_depth=1), seed=5)


def test_a_bounded_generated_inventory_samples_only_admissible_pairs() -> None:
    case = _composition_case("forward", 0, crossed=False)
    identity = Action("identity")
    action = Action(case.rule.name, case.rule.name)
    sampler = RulePairSampler(
        RecursiveTypeSampler(STORE, max_size=3, sort="Cat"),
        Inventory((case.rule,)),
        ActionSet((identity, action), identity),
    )

    pairs, report = sampler.sample(24, PairDistribution(max_depth=2), seed=31)

    assert len(pairs) == report.emitted == 24
    assert all(case.rule.admits(pair.inputs, pair.output) for pair in pairs)
    assert all(
        term_depth(term) <= 2 for pair in pairs for term in (*pair.inputs, pair.output)
    )

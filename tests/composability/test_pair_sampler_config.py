"""Composition tests for installed rule-pair sampler choices."""

import json
from typing import TYPE_CHECKING, Final, cast

from didactic.settings import load_document

from lofi.data import PairDistribution, PairSampler
from lofi.data.synthetic import RecursiveTypeSampler, RulePairSamplerContext
from lofi.judge import Action, ActionSet
from lofi.registry import Registry, plugin_for
from lofi.run import packaged_options, section_root
from lofi.spec import PairSamplerSpec
from lofi.spec._plugins import load_plugins
from lofi.terms import Operator, Signature, Term, TermStore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lofi.spec import ConfiguredSpec


SIGNATURE: Final = Signature(
    "configured-pair-sampler",
    ("Cat",),
    (Operator("n", "Cat"), Operator("np", "Cat")),
)


class UnaryCopyRule:
    """A complete unary rule preserving every premise."""

    name = "unary-copy"
    arity = 1
    directions = frozenset({"forward"})
    forward_is_complete = True

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Yield the sole premise."""
        yield inputs[0]

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        """Return whether the conclusion preserves the sole premise."""
        return len(inputs) == 1 and output is inputs[0]


class UnaryRules:
    """An inventory containing one unary rule."""

    def __init__(self) -> None:
        self.rule = UnaryCopyRule()

    def of_arity(self, k: int) -> tuple[UnaryCopyRule, ...]:
        """Return the unary rule only for its declared arity."""
        return (self.rule,) if k == 1 else ()

    def __iter__(self) -> Iterator[UnaryCopyRule]:
        """Iterate in declaration order."""
        return iter((self.rule,))


def test_packaged_pair_sampler_validates_discovers_constructs_and_samples_unary() -> (
    None
):
    loaded = load_plugins()
    document = load_document(section_root("pair_sampler") / "rule.yaml")
    spec = PairSamplerSpec.model_validate_json(json.dumps(document))
    declaration = cast("ConfiguredSpec[object]", spec)
    store = TermStore(SIGNATURE)
    types = RecursiveTypeSampler(store, max_size=3, sort="Cat")
    rules = UnaryRules()
    identity = Action("identity")
    actions = ActionSet(
        (identity, Action("unary-copy", rules.rule.name)),
        identity,
    )
    sampler = cast(
        "PairSampler[Term, Action]",
        plugin_for(declaration).construct(
            declaration,
            RulePairSamplerContext(types, rules, actions),
        ),
    )
    pairs, report = sampler.sample(20, PairDistribution(max_depth=1), seed=73)
    registry = Registry[object]("pair_sampler")

    assert "rule" in loaded
    assert packaged_options("pair_sampler") == ("rule",)
    assert registry.available() == ("rule",)
    assert isinstance(sampler, PairSampler)
    assert len(pairs) == report.emitted == 20
    assert dict(report.arities) == {1: 20}
    assert all(len(pair.inputs) == 1 for pair in pairs)
    assert all(pair.actions == (actions.actions[1],) for pair in pairs)

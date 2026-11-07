from typing import Final, cast

import pytest
import torch
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import ActionSet, Rule, RuleSet, TypeRaising
from lofi.judge._raising import soft_action_set
from lofi.nn._values import Encoded
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.controllers import MlpController
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.raising import (
    SoftControllerRaising,
    mix_raised,
    reencode_unary,
    reencode_unary_domain,
)
from lofi.nn.sites import ComponentId
from lofi.spec import ApplicationSpec, RuleSetSpec, TypeRaisingSpec
from lofi.spec.controllers import MlpControllerSpec
from lofi.spec.layers import FeedForwardLayerSpec
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    EncodingOverflowError,
    OperatorIndex,
    Term,
    TermCodec,
)

WIDTH: Final = 8
FORWARD: Final = "/"
BACKWARD: Final = "\\"
CATEGORIES: Final = ccg_signature("x", "t", "u", name="neural-raising")
CONTEXT: Final = declared_rules(
    CATEGORIES, RuleSetSpec(name="neural-raising", rules=(ApplicationSpec(view="/"),))
).context
STORE: Final = CONTEXT.store
SIGNATURE: Final = STORE.signature
CATEGORY_OPERATORS: Final = tuple(
    operator for operator in SIGNATURE.operators if operator.sort == "Cat"
)
X: Final = CONTEXT.syntax.parse(STORE, "x")
T: Final = CONTEXT.syntax.parse(STORE, "t")
U: Final = CONTEXT.syntax.parse(STORE, "u")
INDEX: Final = OperatorIndex(SIGNATURE)
CODEC: Final = TermCodec(
    INDEX,
    CompleteTree(SIGNATURE, CATEGORY_OPERATORS, order="df", depth=3),
)
FUNCTORS: Final = CONTEXT.functors[FORWARD][0]


def raising(slash: str, targets: tuple[Term, ...] | None) -> TypeRaising:
    """Compile type raising over the shared categories."""
    (rule,) = declared_rules(
        CATEGORIES,
        RuleSetSpec(
            name="neural-raising",
            rules=(
                TypeRaisingSpec(
                    view=slash,
                    targets=None
                    if targets is None
                    else tuple(
                        CONTEXT.syntax.render(STORE, target) for target in targets
                    ),
                    arguments=None,
                ),
            ),
        ),
    ).rule_set

    return cast("TypeRaising", rule)


def encoder(
    *, layers: int = 2, cell: str = "gru", codec: TermCodec = CODEC
) -> TreeRnnEncoder:
    torch.manual_seed(17)

    return TreeRnnEncoder(
        codec=NeuralCodecView(codec),
        choices=ChoiceIndex(
            SIGNATURE,
            depth_bounds=dict.fromkeys(SIGNATURE.sorts, codec.capacity),
        ),
        component=ComponentId("raising-encoder"),
        scheme=ArityHeadScheme(width=WIDTH, reached=CATEGORY_OPERATORS),
        cell=cell,
        layers=layers,
        norm_cap=None,
        inventory="uncached",
    )


def parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_type_raising_reencodes_the_explicitly_constructed_tree() -> None:
    subject = encoder()
    rule = raising(FORWARD, (T,))
    raised = (
        FUNCTORS.make(FORWARD, T, FUNCTORS.make(BACKWARD, T, X)),
        FUNCTORS.make(FORWARD, T, FUNCTORS.make(BACKWARD, T, U)),
    )

    found = reencode_unary(rule, (X, U), codec=CODEC, encoder=subject)
    expected = subject.encode(CODEC.encode_batch(raised))

    assert found.axes == ("term", "width")
    assert torch.equal(found.tensor, expected.tensor)


def test_one_hot_raise_with_equals_the_full_explicit_raised_encoding() -> None:
    subject = encoder(layers=1)
    rule = raising(FORWARD, (T, U))
    source = subject.encode(CODEC.encode_batch((X,)))
    probabilities = torch.tensor([[1.0, 0.0]])

    found = mix_raised(
        rule,
        (T, U),
        source,
        probabilities,
        codec=CODEC,
        encoder=subject,
    )
    explicit = (
        FUNCTORS.make(FORWARD, T, FUNCTORS.make(BACKWARD, T, X)),
        FUNCTORS.make(FORWARD, U, FUNCTORS.make(BACKWARD, U, X)),
    )

    expected = subject.encode(CODEC.encode_batch(explicit)).tensor[:1]

    assert torch.equal(found.tensor, expected)


@pytest.mark.parametrize(("layers", "cell"), [(2, "gru"), (1, "lstm")])
def test_soft_raising_refuses_an_encoder_without_complete_live_state(
    layers: int, cell: str
) -> None:
    subject = encoder(layers=layers, cell=cell)
    rules = raising_rules()
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )
    actions = soft_action_set(base, rules)

    with pytest.raises(ValueError, match="one-layer TreeRNN whose cell carries no"):
        SoftControllerRaising(
            actions,
            rules,
            (X, T),
            codec=CODEC,
            encoder=subject,
        )


@pytest.mark.parametrize("cell", ["elman-tanh", "elman-relu", "gru"])
def test_soft_raising_accepts_every_one_layer_cell_without_separate_memory(
    cell: str,
) -> None:
    subject = encoder(layers=1, cell=cell)
    rules = raising_rules()
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )
    raising = SoftControllerRaising(
        soft_action_set(base, rules), rules, (X, T), codec=CODEC, encoder=subject
    )

    assert raising.base_actions == base


def raising_rules(
    *,
    composition_slash: str = FORWARD,
    raising_slash: str = FORWARD,
    targets: tuple[str, ...] | None = ("x", "t", "u"),
    arguments: tuple[str, ...] | None = None,
) -> RuleSet[Term]:
    return declared_rules(
        CATEGORIES,
        RuleSetSpec(
            name="raising-controller",
            rules=(
                ApplicationSpec(view=composition_slash),
                TypeRaisingSpec(
                    view=raising_slash, targets=targets, arguments=arguments
                ),
            ),
        ),
    ).rule_set


def test_soft_actions_are_the_combinator_by_raise_configuration_product() -> None:
    rules = raising_rules()
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null="null"
    )
    found = soft_action_set(base, rules)
    application = rules.rules[0].name

    assert [action.name for action in found.actions if action.rule == application] == [
        f"{application}:raise=none",
        f"{application}:raise=functor",
        f"{application}:raise=argument",
        f"{application}:raise=both",
    ]
    assert found.identity is base.identity
    assert found.null is base.null


def test_soft_raising_requires_a_rule_for_each_composition_direction() -> None:
    subject = encoder(layers=1)
    rules = raising_rules(composition_slash=BACKWARD)
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )

    with pytest.raises(ValueError, match="exactly one rule for 'backward'"):
        SoftControllerRaising(
            soft_action_set(base, rules),
            rules,
            (X, T, U),
            codec=CODEC,
            encoder=subject,
        )


def test_a_needed_soft_raise_step_trains_both_target_heads() -> None:
    subject = encoder(layers=1)
    rules = raising_rules()
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )
    actions = soft_action_set(base, rules)
    raising = SoftControllerRaising(
        actions,
        rules,
        (X, T, U),
        codec=CODEC,
        encoder=subject,
    )
    settings = MlpControllerSpec.model_validate(
        {
            "kind": "mlp",
            "raise": "soft",
            "hidden": (FeedForwardLayerSpec(width=5, activation="tanh", dropout=0.0),),
        }
    ).settings()
    controller = MlpController(
        actions,
        width=WIDTH,
        component=ComponentId("soft-controller"),
        settings=settings,
        raising=raising,
    )
    left = Encoded(
        subject.encode(CODEC.encode_batch((X, U))).tensor,
        ("cell", "width"),
    )
    right = Encoded(
        subject.encode(CODEC.encode_batch((U, X))).tensor,
        ("cell", "width"),
    )
    weights = controller.weigh(left, right=right)
    names = tuple(action.name for action in actions.actions)
    selected = torch.tensor(
        (
            names.index(f"{rules.rules[0].name}:raise=functor"),
            names.index(f"{rules.rules[0].name}:raise=argument"),
        )
    )
    loss = -weights.tensor[:, selected].diagonal().sum()
    loss.backward()

    for name in ("_functor_targets.weight", "_argument_targets.weight"):
        gradient = raising.get_parameter(name).grad

        assert gradient is not None
        assert bool(gradient.abs().sum() > 0.0)


def test_soft_raising_does_not_reown_its_term_encoder() -> None:
    subject = encoder(layers=1)
    rules = raising_rules()
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )
    raising = SoftControllerRaising(
        soft_action_set(base, rules),
        rules,
        (X, T, U),
        codec=CODEC,
        encoder=subject,
    )

    assert all(
        not name.startswith("_encoder.") for name, _ in raising.named_parameters()
    )


def test_none_raising_is_exactly_the_existing_controller_path() -> None:
    actions = ActionSet.derive(("application",), identity="identity", null=None)
    explicit = MlpControllerSpec.model_validate(
        {
            "kind": "mlp",
            "raise": "none",
            "hidden": (FeedForwardLayerSpec(width=5, activation="tanh"),),
        }
    )
    implicit = MlpControllerSpec(
        hidden=(FeedForwardLayerSpec(width=5, activation="tanh"),)
    )
    torch.manual_seed(31)
    left_controller = MlpController(
        actions,
        width=WIDTH,
        component=ComponentId("explicit-none"),
        settings=explicit.settings(),
        raising=None,
    )
    torch.manual_seed(31)
    right_controller = MlpController(
        actions,
        width=WIDTH,
        component=ComponentId("implicit-none"),
        settings=implicit.settings(),
        raising=None,
    )
    left = Encoded(torch.randn(3, WIDTH), ("cell", "width"))
    right = Encoded(torch.randn(3, WIDTH), ("cell", "width"))

    assert torch.equal(
        left_controller.weigh(left, right=right).tensor,
        right_controller.weigh(left, right=right).tensor,
    )


def test_the_raising_path_adds_no_parameter() -> None:
    subject = encoder()
    before = parameters(subject)
    rule = raising(BACKWARD, (T,))

    reencode_unary(rule, (X, U), codec=CODEC, encoder=subject)

    assert parameters(subject) == before
    assert tuple(subject.named_parameters())


class PartialUnary:
    """A complete unary relation whose domain contains only ``x``."""

    name = "partial-unary"
    arity = 1
    directions = frozenset({"forward"})
    forward_is_complete = True

    def apply(self, premises: tuple[Term, ...], /) -> tuple[Term, ...]:
        return (T,) if premises == (X,) else ()


def test_partial_unary_reencoding_reports_its_source_rows() -> None:
    subject = encoder()
    found = reencode_unary_domain(
        cast("Rule[Term]", PartialUnary()),
        (X, U),
        codec=CODEC,
        encoder=subject,
    )
    expected = subject.encode(CODEC.encode_batch((T,)))

    assert found.source_rows == (0,)
    assert torch.equal(found.encoded.tensor, expected.tensor)


def test_multivalued_unary_reencoding_repeats_its_source_row() -> None:
    subject = encoder()
    rule = raising(FORWARD, (T, U))
    found = reencode_unary_domain(rule, (X,), codec=CODEC, encoder=subject)
    conclusions = (
        FUNCTORS.make(FORWARD, T, FUNCTORS.make(BACKWARD, T, X)),
        FUNCTORS.make(FORWARD, U, FUNCTORS.make(BACKWARD, U, X)),
    )
    expected = subject.encode(CODEC.encode_batch(conclusions))

    assert found.source_rows == (0, 0)
    assert torch.equal(found.encoded.tensor, expected.tensor)


def test_unary_domain_reencoding_omits_only_overflowing_conclusions() -> None:
    subject = encoder()
    deep = FUNCTORS.make(FORWARD, X, FUNCTORS.make(BACKWARD, T, U))
    rule = raising(FORWARD, (T, deep, U))
    found = reencode_unary_domain(
        rule,
        (X,),
        codec=CODEC,
        encoder=subject,
    )
    retained = (
        FUNCTORS.make(FORWARD, T, FUNCTORS.make(BACKWARD, T, X)),
        FUNCTORS.make(FORWARD, U, FUNCTORS.make(BACKWARD, U, X)),
    )
    expected = subject.encode(CODEC.encode_batch(retained))

    assert found.source_rows == (0, 0)
    assert torch.equal(found.encoded.tensor, expected.tensor)


def test_strict_unary_reencoding_reports_a_sole_overflowing_conclusion() -> None:
    rule = raising(FORWARD, (T,))
    deep = FUNCTORS.make(FORWARD, X, FUNCTORS.make(BACKWARD, T, U))

    with pytest.raises(EncodingOverflowError):
        reencode_unary(rule, (deep,), codec=CODEC, encoder=encoder())


def test_the_raising_path_keeps_the_encoder_graph() -> None:
    subject = encoder()
    rule = raising(FORWARD, (T,))
    encoded = reencode_unary(rule, (X,), codec=CODEC, encoder=subject)

    encoded.tensor.square().sum().backward()

    reached = {
        name
        for name, parameter in subject.named_parameters()
        if parameter.grad is not None
    }

    assert reached == {
        "_start",
        "_cells.0.bias_hh",
        "_cells.0.bias_ih",
        "_cells.0.weight_hh",
        "_cells.0.weight_ih",
        "_cells.1.bias_hh",
        "_cells.1.bias_ih",
        "_cells.1.weight_hh",
        "_cells.1.weight_ih",
        "_scheme._up.0.bias",
        "_scheme._up.0.weight",
        "_embedding._table.weight",
    }


def test_input_order_is_the_encoded_row_order() -> None:
    subject = encoder()
    rule = raising(FORWARD, (T,))
    forward = reencode_unary(rule, (X, U), codec=CODEC, encoder=subject)
    reverse = reencode_unary(rule, (U, X), codec=CODEC, encoder=subject)

    # A batched matrix product may round a row differently at another position.
    assert not torch.allclose(forward.tensor[0], forward.tensor[1])
    torch.testing.assert_close(forward.tensor, reverse.tensor.flip(0))


def test_an_untargeted_rule_is_refused_before_encoding() -> None:
    rule = raising(FORWARD, None)

    with pytest.raises(ValueError, match="completely enumerate"):
        reencode_unary(rule, (X,), codec=CODEC, encoder=encoder())


def test_a_rule_with_two_targets_is_refused_as_not_single_valued() -> None:
    rule = raising(FORWARD, (T, U))

    with pytest.raises(ValueError, match=r"produced 2 conclusions.*row 0"):
        reencode_unary(rule, (X,), codec=CODEC, encoder=encoder())


class Binary:
    name = "binary"
    arity = 2


def test_a_nonunary_rule_is_refused_before_encoding() -> None:
    rule = cast("Rule[Term]", Binary())

    with pytest.raises(ValueError, match=r"arity 2.*unary arity 1"):
        reencode_unary(rule, (X,), codec=CODEC, encoder=encoder())


def test_an_empty_batch_remains_an_empty_batch() -> None:
    subject = encoder()
    rule = raising(FORWARD, (T,))

    found = reencode_unary(rule, (), codec=CODEC, encoder=subject)

    assert found.axes == ("term", "width")
    assert found.tensor.shape == (0, WIDTH)


@pytest.mark.parametrize("target", ["t", "u", "t/u"])
def test_soft_raising_uses_only_the_rules_declared_target(target: str) -> None:
    codec = TermCodec(
        INDEX, CompleteTree(SIGNATURE, CATEGORY_OPERATORS, order="df", depth=4)
    )
    subject = encoder(layers=1, codec=codec)
    rules = raising_rules(targets=(target,))
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )
    actions = soft_action_set(base, rules)
    raising = SoftControllerRaising(
        actions, rules, (X, T, U), codec=codec, encoder=subject
    )
    source = subject.encode(codec.encode_batch((X,)))

    def score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left + right).sum(dim=-1, keepdim=True).expand(-1, len(base.actions))

    found = raising.scores(score, source, source)
    raised = reencode_unary(rules.rules[1], (X,), codec=codec, encoder=subject)
    expected = score(raised.tensor, source.tensor)[:, 0]
    column = tuple(action.name for action in actions.actions).index(
        f"{rules.rules[0].name}:raise=functor"
    )

    torch.testing.assert_close(found[:, column], expected)


def test_soft_raising_refuses_restricted_arguments_before_scoring() -> None:
    rules = raising_rules(arguments=("x",))
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )

    with pytest.raises(ValueError, match=r"soft controller raising.*arguments"):
        SoftControllerRaising(
            soft_action_set(base, rules),
            rules,
            (X, T, U),
            codec=CODEC,
            encoder=encoder(layers=1),
        )

    assert tuple(rules.rules[1].apply((X,)))
    assert not tuple(rules.rules[1].apply((U,)))


def test_soft_raising_keeps_each_directions_target_support_separate() -> None:
    subject = encoder(layers=1)
    rules = declared_rules(
        CATEGORIES,
        RuleSetSpec(
            name="directional-raising",
            rules=(
                ApplicationSpec(view=FORWARD),
                ApplicationSpec(view=BACKWARD),
                TypeRaisingSpec(view=FORWARD, targets=("t",), arguments=None),
                TypeRaisingSpec(view=BACKWARD, targets=("u",), arguments=None),
            ),
        ),
    ).rule_set
    base = ActionSet.derive(
        (rule.name for rule in rules), identity="identity", null=None
    )
    actions = soft_action_set(base, rules)
    raising = SoftControllerRaising(actions, rules, (), codec=CODEC, encoder=subject)
    source = subject.encode(CODEC.encode_batch((X,)))

    def score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left + right).sum(dim=-1, keepdim=True).expand(-1, len(base.actions))

    found = raising.scores(score, source, source)
    names = tuple(action.name for action in actions.actions)

    for composition, rule in zip(rules.rules[:2], rules.rules[2:], strict=True):
        raised = reencode_unary(rule, (X,), codec=CODEC, encoder=subject)
        column = names.index(f"{composition.name}:raise=functor")
        expected = score(raised.tensor, source.tensor)[:, 0]
        torch.testing.assert_close(found[:, column], expected)

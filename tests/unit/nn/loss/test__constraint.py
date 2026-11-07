from copy import copy, deepcopy
from typing import TYPE_CHECKING, Final

import pytest
from tests.language import language

import lofi.nn.loss._constraint as constraint_module
from lofi._types import ItemId, ReadoutId
from lofi.data import (
    SpanConstraint,
    SpanConstraints,
    Token,
    TokenScope,
    TokenSpan,
    TypeScope,
)
from lofi.lexicon import ClampMode, ClampScope
from lofi.nn.loss._constraint import (
    BatchLocalConstraint,
    BatchLocalNegativeBracket,
    Constraint,
    ConstraintOccurrence,
    LogicalFormConstraint,
    NegativeBracketConstraint,
    resolve_spans,
)
from lofi.nn.loss._ports import ResolvableConstraint
from lofi.terms import Identity, TargetSet, Term

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from lofi.terms import Matching


READOUT = ReadoutId("nominals")


def targets() -> TargetSet[str]:
    return TargetSet(("S", "NP"), Identity[str]())


def occurrence(item: str, start: int, end: int) -> ConstraintOccurrence:
    return ConstraintOccurrence(ItemId(item), (start, end))


def test_a_constraint_freezes_its_dataset_occurrences_and_provenance() -> None:
    scope = TypeScope(("sea", "turtles"))
    source = [occurrence("sentence/1", 1, 3)]
    constraint = Constraint(targets(), source, scope)
    source.append(occurrence("sentence/2", 0, 2))

    assert constraint.occurrences == (occurrence("sentence/1", 1, 3),)
    assert constraint.scope is scope
    assert not constraint.negate


def test_type_scope_expands_once_in_data_before_objective_resolution() -> None:
    scope = TypeScope(("sea", "turtles"))
    tokens = tuple(
        Token(position, surface)
        for position, surface in enumerate(("sea", "turtles", "and", "sea", "turtles"))
    )
    declared = SpanConstraint(TokenSpan(0, 2), frozenset({"NP"}), scope)
    expanded = SpanConstraints((declared,), len(tokens)).instances(tokens)
    constraint = Constraint(
        targets(),
        tuple(
            ConstraintOccurrence(
                ItemId("sentence/1"),
                (instance.span.start, instance.span.end),
            )
            for instance in expanded
        ),
        scope,
    )

    resolved = constraint.to_spans((ItemId("sentence/1"),), readout=READOUT)

    assert len(resolved) == 1
    assert resolved[0].spans == ((0, 2), (3, 5))
    assert resolved[0].scope is scope


def test_to_spans_remaps_items_and_merges_equal_span_lists() -> None:
    scope = TokenScope()
    declared_targets = targets()
    constraint = Constraint(
        declared_targets,
        (
            occurrence("sentence/1", 2, 3),
            occurrence("sentence/2", 2, 3),
            occurrence("sentence/3", 4, 5),
            occurrence("sentence/3", 0, 1),
            occurrence("sentence/3", 4, 5),
            occurrence("outside-batch", 7, 8),
        ),
        scope,
        negate=True,
    )

    resolved = constraint.to_spans(
        (
            ItemId("sentence/3"),
            ItemId("sentence/2"),
            ItemId("sentence/1"),
        ),
        readout=READOUT,
    )

    assert len(resolved) == 2
    assert resolved[0].spans == ((4, 5), (0, 1), (4, 5))
    assert resolved[0].item_positions == (0,)
    assert resolved[1].spans == ((2, 3),)
    assert resolved[1].item_positions == (1, 2)
    assert all(local.targets is declared_targets for local in resolved)
    assert all(local.readout == READOUT for local in resolved)
    assert all(local.scope is scope for local in resolved)
    assert all(local.negate for local in resolved)


def test_to_spans_retains_repeated_batch_positions() -> None:
    constraint = Constraint(
        targets(),
        (occurrence("sentence/1", 0, 1),),
        TokenScope(),
    )

    (resolved,) = constraint.to_spans(
        (
            ItemId("sentence/1"),
            ItemId("sentence/2"),
            ItemId("sentence/1"),
        ),
        readout=READOUT,
    )

    assert resolved.item_positions == (0, 2)


def test_a_batch_local_constraint_is_a_distinct_frozen_value() -> None:
    scope = TokenScope()
    declared_targets = targets()
    global_constraint = Constraint(
        declared_targets,
        (occurrence("sentence/1", 0, 1),),
        scope,
    )
    (local,) = global_constraint.to_spans((ItemId("sentence/1"),), readout=READOUT)

    assert type(local) is BatchLocalConstraint
    assert local.targets is declared_targets
    assert local.spans == ((0, 1),)
    assert local.item_positions == (0,)
    assert local.scope is scope
    assert local.readout == READOUT
    assert copy(local) is local
    assert deepcopy(local) is local

    with pytest.raises(AttributeError, match="immutable"):
        local.__setattr__("_spans", ())


def test_a_soft_constraint_retains_its_penalty_when_localized() -> None:
    declared = Constraint(
        targets(),
        (occurrence("sentence/1", 0, 1),),
        TokenScope(),
        mode=ClampMode.SOFT,
        penalty=0.25,
    )

    (local,) = declared.to_spans((ItemId("sentence/1"),), readout=READOUT)

    assert declared.mode is local.mode is ClampMode.SOFT
    assert declared.penalty == local.penalty == 0.25


@pytest.mark.parametrize(
    ("mode", "penalty"),
    [
        (ClampMode.HARD, 1.0),
        (ClampMode.SOFT, None),
        (ClampMode.SOFT, 0.0),
        (ClampMode.SOFT, float("inf")),
    ],
)
def test_a_constraint_refuses_a_penalty_incoherent_with_its_mode(
    mode: ClampMode, penalty: float | None
) -> None:
    with pytest.raises(ValueError, match="penalty"):
        Constraint(
            targets(),
            (occurrence("sentence/1", 0, 1),),
            TokenScope(),
            mode=mode,
            penalty=penalty,
        )


def test_negation_participates_in_equality_and_hashing() -> None:
    declared_targets = targets()
    occurrences = (occurrence("sentence/1", 0, 1),)
    scope = TokenScope()
    constraint = Constraint(declared_targets, occurrences, scope)
    equal = Constraint(declared_targets, occurrences, scope)
    negated = Constraint(declared_targets, occurrences, scope, negate=True)
    original_hash = hash(constraint)

    constraint.to_spans((ItemId("sentence/1"),), readout=READOUT)

    assert constraint == equal
    assert hash(constraint) == hash(equal) == original_hash
    assert constraint != negated
    assert hash(constraint) != hash(negated)


@pytest.mark.parametrize(
    ("declared_targets", "occurrences", "message"),
    [
        (TargetSet((), Identity[str]()), (occurrence("sentence/1", 0, 1),), "target"),
        (targets(), (), "occurrence"),
    ],
)
def test_empty_targets_and_occurrences_are_construction_errors(
    declared_targets: TargetSet[str],
    occurrences: tuple[ConstraintOccurrence, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Constraint(declared_targets, occurrences, TokenScope())


@pytest.mark.parametrize("span", [(-1, 1), (2, 1), (1, 1)])
def test_an_occurrence_requires_a_nonempty_forward_span(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="span"):
        ConstraintOccurrence(ItemId("sentence/1"), span)


def test_a_constraint_returns_nothing_when_the_batch_has_no_occurrence() -> None:
    constraint = Constraint(
        targets(),
        (occurrence("sentence/1", 0, 1),),
        TokenScope(),
    )

    assert constraint.to_spans((ItemId("sentence/2"),), readout=READOUT) == ()


def test_a_readout_id_is_required_even_when_no_occurrence_is_in_the_batch() -> None:
    constraint = Constraint(
        targets(),
        (occurrence("sentence/1", 0, 1),),
        TokenScope(),
    )

    with pytest.raises(ValueError, match="readout id"):
        constraint.to_spans((ItemId("sentence/2"),), readout=ReadoutId(""))


def test_a_negative_bracket_localizes_without_a_target_or_readout() -> None:
    scope = TypeScope(("sea", "turtles"))
    declared = NegativeBracketConstraint(
        (
            occurrence("sentence/1", 0, 2),
            occurrence("sentence/2", 1, 3),
            occurrence("sentence/1", 2, 4),
        ),
        scope,
    )

    localized = declared.to_spans(
        (ItemId("sentence/2"), ItemId("sentence/1"), ItemId("outside"))
    )

    assert localized == (
        BatchLocalNegativeBracket(((1, 3),), (0,), scope),
        BatchLocalNegativeBracket(((0, 2), (2, 4)), (1,), scope),
    )
    assert all(not hasattr(held, "targets") for held in localized)
    assert all(not hasattr(held, "readout") for held in localized)
    assert not hasattr(declared, "targets")
    assert not hasattr(declared, "readout")
    assert copy(declared) is declared
    assert deepcopy(declared) is declared
    assert copy(localized[0]) is localized[0]
    assert deepcopy(localized[0]) is localized[0]


def test_a_negative_bracket_is_frozen_hashable_and_requires_an_occurrence() -> None:
    source = [occurrence("sentence/1", 0, 2)]
    scope = TokenScope()
    declared = NegativeBracketConstraint(source, scope)
    same = NegativeBracketConstraint(tuple(source), scope)
    source.append(occurrence("sentence/2", 1, 3))

    assert declared.occurrences == (occurrence("sentence/1", 0, 2),)
    assert declared.scope is scope
    assert declared == same
    assert hash(declared) == hash(same)
    assert "NegativeBracketConstraint" in repr(declared)

    with pytest.raises(AttributeError, match="immutable"):
        declared.__setattr__("_occurrences", ())

    with pytest.raises(ValueError, match="occurrence"):
        NegativeBracketConstraint((), scope)

    with pytest.raises(ValueError, match="span occurrence"):
        BatchLocalNegativeBracket((), (0,), scope)

    with pytest.raises(ValueError, match="item position"):
        BatchLocalNegativeBracket(((0, 2),), (), scope)


LANGUAGE: Final = language(
    ("bo", "likes", "mary"),
    types={"bo": "e", "likes": "e -> t", "mary": "e"},
)
STORE: Final = LANGUAGE.store
FORM_MATCHING: Final[Matching[Term]] = Identity()
LIKES: Final = LANGUAGE.constant("likes")
BO: Final = LANGUAGE.constant("bo")
MARY: Final = LANGUAGE.constant("mary")


def apply(function: Term, argument: Term) -> Term:
    return LANGUAGE.apply(function, argument)


def abstraction(variable: str) -> Term:
    return STORE.bind(
        "lam", variable, apply(LIKES, STORE.fvar(LANGUAGE.term_sort, variable))
    )


LIKES_BO: Final = apply(LIKES, BO)
LIKES_MARY: Final = apply(LIKES, MARY)


def form_constraint(
    form: Term = LIKES_BO,
    occurrences: tuple[ConstraintOccurrence, ...] = (occurrence("sentence/1", 1, 3),),
    scope: TokenScope | TypeScope | None = None,
    *,
    matching: Matching[Term] = FORM_MATCHING,
    negate: bool = False,
) -> LogicalFormConstraint[TokenScope | TypeScope]:
    return LogicalFormConstraint(
        form,
        occurrences,
        TokenScope() if scope is None else scope,
        matching=matching,
        negate=negate,
    )


def test_a_logical_form_constraint_pins_one_meaning_as_a_singleton_target() -> None:
    scope = TypeScope(("sea", "turtles"))
    source = [occurrence("sentence/1", 1, 3)]
    constraint = LogicalFormConstraint(LIKES_BO, source, scope, matching=FORM_MATCHING)
    source.append(occurrence("sentence/2", 0, 2))

    assert constraint.form is LIKES_BO
    assert constraint.targets == TargetSet((LIKES_BO,), FORM_MATCHING)
    assert constraint.targets.matching is FORM_MATCHING
    assert constraint.occurrences == (occurrence("sentence/1", 1, 3),)
    assert constraint.scope is scope
    assert not constraint.negate
    assert isinstance(constraint, ResolvableConstraint)
    assert copy(constraint) is constraint
    assert deepcopy(constraint) is constraint

    with pytest.raises(AttributeError, match="immutable"):
        constraint.__setattr__("_form", LIKES_MARY)


def test_both_constraint_kinds_resolve_through_the_one_shared_function() -> None:
    calls: list[tuple[tuple[ItemId, ...], ReadoutId]] = []
    shared = resolve_spans

    def counting[T, ScopeT: Hashable](
        targets: TargetSet[T],
        occurrences: tuple[ConstraintOccurrence, ...],
        scope: ScopeT,
        batch_item_ids: Sequence[ItemId],
        /,
        *,
        readout: ReadoutId,
        clamp_scope: ClampScope,
        mode: ClampMode,
        penalty: float | None,
        negate: bool,
    ) -> tuple[BatchLocalConstraint[T, ScopeT], ...]:
        calls.append((tuple(batch_item_ids), readout))

        return shared(
            targets,
            occurrences,
            scope,
            batch_item_ids,
            readout=readout,
            clamp_scope=clamp_scope,
            mode=mode,
            penalty=penalty,
            negate=negate,
        )

    occurrences = (occurrence("sentence/1", 1, 3), occurrence("sentence/2", 0, 2))
    typed = Constraint(targets(), occurrences, TokenScope(), negate=True)
    formed = form_constraint(occurrences=occurrences, negate=True)
    batch = (ItemId("sentence/2"), ItemId("sentence/1"))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(constraint_module, "resolve_spans", counting)
        typed_local = typed.to_spans(batch, readout=READOUT)
        formed_local = formed.to_spans(batch, readout=READOUT)

    assert calls == [(batch, READOUT), (batch, READOUT)]
    assert typed_local == typed.to_spans(batch, readout=READOUT)
    assert formed_local == formed.to_spans(batch, readout=READOUT)
    assert formed_local == shared(
        formed.targets,
        occurrences,
        formed.scope,
        batch,
        readout=READOUT,
        clamp_scope=formed.clamp_scope,
        mode=formed.mode,
        penalty=formed.penalty,
        negate=True,
    )


def test_a_logical_form_constraint_localizes_exactly_as_a_type_constraint() -> None:
    scope = TokenScope()
    occurrences = (
        occurrence("sentence/1", 2, 3),
        occurrence("sentence/2", 2, 3),
        occurrence("sentence/3", 4, 5),
        occurrence("sentence/3", 0, 1),
        occurrence("outside-batch", 7, 8),
    )
    batch = (ItemId("sentence/3"), ItemId("sentence/2"), ItemId("sentence/1"))
    typed = Constraint(targets(), occurrences, scope, negate=True)
    formed = form_constraint(occurrences=occurrences, scope=scope, negate=True)

    typed_local = typed.to_spans(batch, readout=READOUT)
    formed_local = formed.to_spans(batch, readout=READOUT)

    assert len(formed_local) == len(typed_local) == 2
    assert all(type(local) is BatchLocalConstraint for local in formed_local)

    for from_form, from_type in zip(formed_local, typed_local, strict=True):
        assert from_form.spans == from_type.spans
        assert from_form.item_positions == from_type.item_positions
        assert from_form.scope is from_type.scope
        assert from_form.readout == from_type.readout
        assert from_form.negate == from_type.negate
        assert from_form.targets is formed.targets
        assert from_form.targets.categories == (LIKES_BO,)


def test_a_logical_form_constraint_has_one_equality_over_its_meaning() -> None:
    occurrences = (occurrence("sentence/1", 0, 1),)
    constraint = form_constraint(occurrences=occurrences)
    same = form_constraint(occurrences=occurrences)
    other_form = form_constraint(LIKES_MARY, occurrences)
    negated = form_constraint(occurrences=occurrences, negate=True)
    other_matching = form_constraint(occurrences=occurrences, matching=Identity())
    as_type = Constraint(
        TargetSet((LIKES_BO,), FORM_MATCHING), occurrences, TokenScope()
    )
    original_hash = hash(constraint)

    constraint.to_spans((ItemId("sentence/1"),), readout=READOUT)

    assert constraint == same
    assert hash(constraint) == hash(same) == original_hash
    assert constraint != other_form
    assert constraint != negated
    assert hash(constraint) != hash(negated)
    assert constraint != other_matching
    assert constraint != as_type
    assert as_type != constraint


def test_alpha_variants_of_a_pinned_meaning_are_one_constraint() -> None:
    occurrences = (occurrence("sentence/1", 0, 2),)

    over_x = form_constraint(abstraction("x"), occurrences)
    over_y = form_constraint(abstraction("y"), occurrences)

    assert over_x.form is over_y.form
    assert over_x == over_y
    assert hash(over_x) == hash(over_y)


def test_a_logical_form_constraint_with_no_occurrence_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="occurrence"):
        form_constraint(occurrences=())


def test_a_logical_form_constraint_renders_its_meaning_and_direction() -> None:
    rendered = repr(form_constraint(negate=True))

    assert rendered.startswith("LogicalFormConstraint(form=")
    assert "negate=True" in rendered

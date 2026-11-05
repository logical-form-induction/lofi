from typing import TYPE_CHECKING, Final, Literal, override

import pytest
from tests.language import language

from lofi._types import ItemId
from lofi.data import TokenScope
from lofi.lf import term_codec
from lofi.nn.loss._constraint import (
    Constraint,
    ConstraintOccurrence,
    LogicalFormConstraint,
)
from lofi.nn.loss._linearize import (
    LinearizedConstraint,
    LinearizedTargets,
    linearize,
    linearize_targets,
)
from lofi.terms import (
    CompleteTree,
    EncodingOverflowError,
    FeatureErasure,
    Identity,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TargetSet,
    Term,
    TermCodec,
    TermStore,
    UnknownOperatorError,
    beta_normal,
)

if TYPE_CHECKING:
    from lofi.terms import Matching


def signature(atoms: tuple[str, ...]) -> Signature:
    return Signature(
        "constraints",
        ("Cat",),
        (
            *(Operator(name, "Cat", family="atom") for name in atoms),
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="constructor"),
        ),
        features={"S": ("dcl",)},
    )


DECLARED: Final = signature(("S", "NP"))
SWAPPED: Final = signature(("NP", "S"))


def codec_over(
    declaration: Signature,
    *,
    order: Literal["df", "bf"] = "df",
    depth: int = 3,
) -> TermCodec:
    index = OperatorIndex(declaration)
    layout = CompleteTree(declaration, declaration.operators, order=order, depth=depth)

    return TermCodec(index, layout)


def atom(declaration: Signature, name: str) -> Term:
    return TermStore(declaration).op(declaration.by_name(name))


def slashed(declaration: Signature, left: str, right: str) -> Term:
    store = TermStore(declaration)

    return store.op(
        declaration.by_name("fwd"),
        store.op(declaration.by_name(left)),
        store.op(declaration.by_name(right)),
    )


def nested(declaration: Signature) -> Term:
    store = TermStore(declaration)
    fwd = declaration.by_name("fwd")
    inner = store.op(
        fwd, store.op(declaration.by_name("S")), store.op(declaration.by_name("NP"))
    )

    return store.op(fwd, inner, store.op(declaration.by_name("NP")))


def codes(codec: TermCodec, declaration: Signature) -> dict[str, int]:
    return {
        name: codec.index.index(declaration.by_name(name))
        for name in ("S", "NP", "fwd")
    }


def constraint_over(
    targets: TargetSet[Term], *, negate: bool = False
) -> Constraint[Term, TokenScope]:
    return Constraint(
        targets,
        (ConstraintOccurrence(ItemId("sentence/1"), (0, 2)),),
        TokenScope(),
        negate=negate,
    )


def test_each_category_is_encoded_in_the_target_sets_declaration_order() -> None:
    codec = codec_over(DECLARED)
    complex_category = slashed(DECLARED, "S", "NP")
    primitive = atom(DECLARED, "NP")
    targets = TargetSet((complex_category, primitive), Identity[Term]())

    linearized = linearize_targets(targets, codec)

    assert linearized.encoded.rows == (
        codec.encode(complex_category),
        codec.encode(primitive),
    )
    assert linearized.encoded.lengths == (3, 1)


def test_the_padding_code_is_the_one_code_no_operator_holds() -> None:
    codec = codec_over(DECLARED)
    targets = TargetSet((atom(DECLARED, "NP"),), Identity[Term]())

    linearized = linearize_targets(targets, codec)

    assigned = {codec.index.index(op) for op in DECLARED.operators}
    (row,) = linearized.encoded.rows
    vacant = {
        symbol
        for symbol, occupied in zip(row.symbols, row.occupied, strict=True)
        if not occupied
    }

    assert linearized.pad not in assigned
    assert vacant == {linearized.pad}


def test_two_operator_indexes_with_different_codes_linearize_differently() -> None:
    here = codec_over(DECLARED)
    there = codec_over(SWAPPED)
    declared = linearize_targets(
        TargetSet((slashed(DECLARED, "S", "NP"),), Identity[Term]()), here
    )
    before = declared.encoded.symbols
    swapped = linearize_targets(
        TargetSet((slashed(SWAPPED, "S", "NP"),), Identity[Term]()), there
    )
    (result,) = (row[1] for row in declared.encoded.symbols)
    (elsewhere,) = (row[1] for row in swapped.encoded.symbols)

    assert result == codes(here, DECLARED)["S"]
    assert elsewhere == codes(there, SWAPPED)["S"]
    assert result != elsewhere
    assert declared.encoded.symbols != swapped.encoded.symbols
    assert declared.encoded.symbols == before


def test_two_layout_orders_linearize_one_category_differently() -> None:
    depth_first = codec_over(DECLARED, order="df")
    breadth_first = codec_over(DECLARED, order="bf")
    targets = TargetSet((nested(DECLARED),), Identity[Term]())
    code = codes(depth_first, DECLARED)

    preorder = linearize_targets(targets, depth_first)
    before = preorder.encoded.symbols
    level_order = linearize_targets(targets, breadth_first)
    pad = preorder.pad

    assert preorder.encoded.symbols == (
        (code["fwd"], code["fwd"], code["S"], code["NP"], code["NP"], pad, pad),
    )
    assert level_order.encoded.symbols == (
        (code["fwd"], code["fwd"], code["NP"], code["S"], code["NP"], pad, pad),
    )
    assert preorder.encoded.symbols == before


def test_a_matching_reduces_each_category_before_it_is_encoded() -> None:
    codec = codec_over(DECLARED)
    featured = atom(DECLARED, "S[dcl]")
    plain = atom(DECLARED, "S")

    erased = linearize_targets(TargetSet((featured,), FeatureErasure()), codec)
    kept = linearize_targets((TargetSet((featured,), Identity[Term]())), codec)

    assert erased.encoded.rows == (codec.encode(plain),)
    assert erased.matching == "feature-erasure"
    assert kept.encoded.rows == (codec.encode(featured),)
    assert kept.matching == "identity"
    assert erased.encoded.symbols != kept.encoded.symbols


def test_a_category_the_index_does_not_declare_is_refused() -> None:
    elsewhere = Signature(
        "elsewhere", ("Cat",), (Operator("PP", "Cat", family="atom"),)
    )
    targets = TargetSet((atom(elsewhere, "PP"),), Identity[Term]())
    codec = codec_over(DECLARED)

    with pytest.raises(UnknownOperatorError):
        linearize_targets(targets, codec)

    with pytest.raises(UnknownOperatorError):
        linearize(constraint_over(targets), codec)


def test_a_category_wider_than_the_layout_is_refused() -> None:
    targets = TargetSet((slashed(DECLARED, "S", "NP"),), Identity[Term]())
    narrow = TermCodec(OperatorIndex(DECLARED), PreorderArityLayout(2))

    with pytest.raises(EncodingOverflowError):
        linearize_targets(targets, narrow)

    with pytest.raises(EncodingOverflowError):
        linearize(constraint_over(targets), narrow)


def test_a_constraint_linearizes_its_targets_and_keeps_its_direction() -> None:
    codec = codec_over(DECLARED)
    targets = TargetSet((atom(DECLARED, "NP"),), Identity[Term]())

    permitted = linearize(constraint_over(targets), codec)
    excluded = linearize(constraint_over(targets, negate=True), codec)

    assert permitted == LinearizedConstraint(
        linearize_targets(targets, codec), negate=False
    )
    assert not permitted.negate
    assert excluded.negate
    assert excluded.targets == permitted.targets


def test_linearizing_over_two_codecs_leaves_one_constraint_list_untouched() -> None:
    targets = TargetSet((atom(DECLARED, "NP"),), Identity[Term]())
    constraints = [constraint_over(targets), constraint_over(targets, negate=True)]
    rendered = [repr(each) for each in constraints]
    hashed = [hash(each) for each in constraints]

    wide = [linearize(each, codec_over(DECLARED, depth=3)) for each in constraints]
    narrow = [linearize(each, codec_over(DECLARED, depth=2)) for each in constraints]

    assert [repr(each) for each in constraints] == rendered
    assert [hash(each) for each in constraints] == hashed
    assert all(each.targets is targets for each in constraints)
    assert wide[0].targets.encoded.width != narrow[0].targets.encoded.width
    assert wide[1].negate
    assert narrow[1].negate


def test_a_linearized_value_is_frozen_and_compares_by_its_channels() -> None:
    codec = codec_over(DECLARED)
    category = atom(DECLARED, "NP")
    targets = TargetSet((category,), Identity[Term]())
    linearized = linearize_targets(targets, codec)
    again = linearize_targets(targets, codec)
    under_erasure = linearize_targets(TargetSet((category,), FeatureErasure()), codec)
    over_another = linearize_targets(
        TargetSet((atom(DECLARED, "S"),), Identity[Term]()), codec
    )

    assert isinstance(linearized, LinearizedTargets)
    assert linearized == again
    assert hash(linearized) == hash(again)
    assert under_erasure.encoded == linearized.encoded
    assert under_erasure.matching != linearized.matching
    assert under_erasure != linearized
    assert over_another.encoded != linearized.encoded
    assert over_another != linearized

    with pytest.raises(AttributeError):
        linearized.__setattr__("pad", 7)


FORM_LANGUAGE: Final = language(("bo", "likes"), types={"bo": "e", "likes": "e -> t"})
FORMS: Final = FORM_LANGUAGE.store
CALCULUS: Final = FORM_LANGUAGE.calculus
LIKES: Final = FORM_LANGUAGE.constant("likes")
BO: Final = FORM_LANGUAGE.constant("bo")


def applied(function: Term, argument: Term) -> Term:
    return FORM_LANGUAGE.apply(function, argument)


LIKES_BO: Final = applied(LIKES, BO)
DETOUR: Final = applied(
    FORMS.bind("lam", "x", applied(LIKES, FORMS.fvar(FORM_LANGUAGE.term_sort, "x"))),
    BO,
)


class Normalizing:
    """Matches logical forms up to beta reduction."""

    @property
    def name(self) -> str:
        return "beta-normal"

    @property
    def has_key(self) -> bool:
        return True

    def key(self, t: Term, /) -> Term:
        return beta_normal(FORMS, t, CALCULUS, fuel=40)

    def equivalent(self, a: Term, b: Term, /) -> bool:
        return self.key(a) is self.key(b)

    @override
    def __repr__(self) -> str:
        return "Normalizing()"


def form_codec(depth: int = 4) -> TermCodec:
    return term_codec(
        FORM_LANGUAGE,
        CompleteTree(
            FORMS.signature, FORMS.signature.operators, order="df", depth=depth
        ),
    )


def form_constraint_over(
    form: Term, *, matching: Matching[Term], negate: bool = False
) -> LogicalFormConstraint[TokenScope]:
    return LogicalFormConstraint(
        form,
        (ConstraintOccurrence(ItemId("sentence/1"), (0, 2)),),
        TokenScope(),
        matching=matching,
        negate=negate,
    )


def test_a_logical_form_constraint_linearizes_its_form_through_the_lf_codec() -> None:
    codec = form_codec()
    matching = Identity[Term]()

    required = linearize(form_constraint_over(LIKES_BO, matching=matching), codec)
    forbidden = linearize(
        form_constraint_over(LIKES_BO, matching=matching, negate=True), codec
    )

    assert required == LinearizedConstraint(
        linearize_targets(TargetSet((LIKES_BO,), matching), codec), negate=False
    )
    assert required.targets.encoded.rows == (codec.encode(LIKES_BO),)
    assert required.targets.pad == codec.index.PAD
    assert required.targets.matching == "identity"
    assert not required.negate
    assert forbidden.negate
    assert forbidden.targets == required.targets


def test_a_normalizing_matching_linearizes_the_forms_normal_form() -> None:
    codec = form_codec()

    reduced = linearize(form_constraint_over(DETOUR, matching=Normalizing()), codec)
    literal = linearize(form_constraint_over(DETOUR, matching=Identity[Term]()), codec)

    assert beta_normal(FORMS, DETOUR, CALCULUS, fuel=40) is LIKES_BO
    assert reduced.targets.encoded.rows == (codec.encode(LIKES_BO),)
    assert reduced.targets.matching == "beta-normal"
    assert literal.targets.encoded.rows == (codec.encode(DETOUR),)
    assert reduced.targets.encoded.symbols != literal.targets.encoded.symbols


def test_a_form_wider_than_the_lf_layout_is_refused() -> None:
    with pytest.raises(EncodingOverflowError):
        linearize(
            form_constraint_over(DETOUR, matching=Identity[Term]()), form_codec(1)
        )

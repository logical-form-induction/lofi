import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.support.term_reference import categories, formulas

from lofi._checks import LawViolationError
from lofi.terms import (
    BindingShapeError,
    BindingSite,
    DuplicateNameError,
    ErasureError,
    Operator,
    ReservedDeclarationError,
    Signature,
    UndeclaredSortError,
    UnknownConstantError,
    UnknownOperatorError,
    Vocabulary,
)
from lofi.terms.laws import (
    SIGNATURE_INVARIANTS,
    AlphabetFixture,
    check_signature_invariants,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TREE: Final = Path(__file__).resolve().parents[3]

TABLES: Final = """
import json

from tests.support.term_reference import categories, formulas

print(
    json.dumps(
        {
            signature.name: {
                "digest": signature.digest,
                "codes": {op.name: signature.code(op) for op in signature.operators},
            }
            for signature in (categories(), formulas())
        }
    )
)
"""


def two_sorted() -> Signature:
    """Return a Church-style two-sorted signature of types and terms."""
    return Signature(
        "church",
        ("Ty", "Tm"),
        (
            Operator("arrow", "Ty", arg_sorts=("Ty", "Ty"), family="constructor"),
            Operator(
                "lam",
                "Tm",
                arg_sorts=("Ty", "Tm"),
                binding=(BindingSite((1,), "Tm", 0),),
                family="binder",
            ),
            Operator("app", "Tm", arg_sorts=("Tm", "Tm"), family="application"),
            Operator(
                "most",
                "Tm",
                arg_sorts=("Tm", "Tm"),
                binding=(BindingSite((0, 1), "Tm"),),
                family="binder",
            ),
            Operator(
                "all",
                "Ty",
                arg_sorts=("Ty",),
                binding=(BindingSite((0,), "Ty"),),
                family="binder",
            ),
        ),
        vocabularies=(
            Vocabulary("Ty", ("e", "t")),
            Vocabulary("Tm", ("jo", "sleep")),
        ),
        open_sorts=("Ty",),
        classifiers={"Tm": "Ty"},
    )


BUILDERS: Final = (categories, formulas, two_sorted)

NAMES: Final = tuple(build().name for build in BUILDERS)


def alphabet(
    build: Callable[[], Signature],
    rebuild: Callable[[], Signature] | None = None,
) -> AlphabetFixture[Signature, Operator]:
    return AlphabetFixture(
        table=build(),
        rebuild=build if rebuild is None else rebuild,
        operators=lambda signature: signature.operators,
        code=lambda signature, op: signature.code(op),
        operator=lambda signature, code: signature.operator(code),
        arity=lambda _signature, op: op.arity,
        sort=lambda _signature, op: op.sort,
        family=lambda _signature, op: op.family,
        digest=lambda signature: signature.digest,
        pad=Signature.PAD,
    )


def tables_from_a_fresh_interpreter(
    seed: str,
) -> dict[str, dict[str, str | dict[str, int]]]:
    finished = subprocess.run(
        [sys.executable, "-"],
        input=TABLES,
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=TREE,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )

    return json.loads(finished.stdout)


def test_a_binding_site_reads_back_its_scope_sort_and_annotation() -> None:
    site = BindingSite((1, 2), "Tm", 0)

    assert (site.arguments, site.sort, site.annotation) == ((1, 2), "Tm", 0)
    assert BindingSite([0], "Tm").annotation is None
    assert repr(site) == "BindingSite((1, 2), 'Tm', annotation=0)"


def test_two_binding_sites_are_equal_exactly_when_every_field_is() -> None:
    site = BindingSite((0,), "Tm")

    assert site == BindingSite((0,), "Tm")
    assert hash(site) == hash(BindingSite((0,), "Tm"))
    assert site != BindingSite((1,), "Tm")
    assert site != BindingSite((0,), "Ty")
    assert site != BindingSite((0,), "Tm", 1)
    assert site != (0,)


@pytest.mark.parametrize(
    ("arguments", "annotation", "match"),
    [
        ((), None, "at least one argument"),
        ((1, 0), None, "strictly ascending"),
        ((0, 0), None, "strictly ascending"),
        ((-1,), None, "negative"),
        ((0, 1), 1, "inside the variable's scope"),
        ((0,), -1, "negative"),
    ],
    ids=["empty", "descending", "repeated", "negative", "annotated", "annotation"],
)
def test_a_binding_site_refuses_a_malformed_scope_or_annotation(
    arguments: tuple[int, ...], annotation: int | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BindingSite(arguments, "Tm", annotation)


def test_an_operator_reads_back_every_field_it_was_declared_with() -> None:
    site = BindingSite((0,), "Lf")
    op = Operator(
        "lam",
        "Lf",
        arg_sorts=("Lf",),
        binding=(site,),
        family="binder",
        erases_to="app",
    )

    assert (op.name, op.sort, op.arg_sorts, op.binding) == (
        "lam",
        "Lf",
        ("Lf",),
        (site,),
    )
    assert (op.payload_sort, op.family, op.erases_to) == ("none", "binder", "app")


def test_arity_is_the_length_of_the_argument_sorts() -> None:
    unary = Operator("tc", "Cat", arg_sorts=("Cat",))
    ternary = Operator("cond", "Lf", arg_sorts=("Lf", "Lf", "Lf"))

    assert (Operator("np", "Cat").arity, unary.arity, ternary.arity) == (0, 1, 3)


def test_an_operator_binds_when_it_declares_a_binding_site() -> None:
    binder = Operator(
        "lam", "Lf", arg_sorts=("Lf",), binding=(BindingSite((0,), "Lf"),)
    )
    application = Operator("app", "Lf", arg_sorts=("Lf", "Lf"))

    assert binder.is_binder
    assert not application.is_binder
    assert application.binding == ()


def test_the_scope_of_an_argument_lists_the_sites_over_it_in_order() -> None:
    restrictor = BindingSite((0, 1), "Tm")
    inner = BindingSite((1,), "Tm")
    typed = BindingSite((1,), "Ty")
    op = Operator(
        "q",
        "Tm",
        arg_sorts=("Tm", "Tm", "Ty"),
        binding=(restrictor, inner, typed),
    )

    assert op.scope(0) == (restrictor,)
    assert op.scope(1) == (restrictor, inner, typed)
    assert op.scope(2) == ()
    assert (op.shift(0, "Tm"), op.shift(1, "Tm"), op.shift(1, "Ty")) == (1, 2, 1)
    assert (op.shift(2, "Tm"), op.shift(0, "Ty")) == (0, 0)


@pytest.mark.parametrize(
    "site",
    [BindingSite((1,), "Lf"), BindingSite((0,), "Lf", 1)],
    ids=["scope", "annotation"],
)
def test_a_binding_site_naming_a_position_outside_the_arity_is_refused(
    site: BindingSite,
) -> None:
    with pytest.raises(BindingShapeError) as caught:
        Operator("lam", "Lf", arg_sorts=("Lf",), binding=(site,))

    assert caught.value.operator == "lam"
    assert "outside the arity 1" in str(caught.value)


def test_a_binding_that_holds_something_other_than_a_site_is_refused() -> None:
    with pytest.raises(TypeError, match="non-site"):
        Operator(
            "lam",
            "Lf",
            arg_sorts=("Lf",),
            binding=cast("tuple[BindingSite, ...]", ((0,),)),
        )


def test_an_operator_refuses_an_empty_name_or_sort() -> None:
    with pytest.raises(ValueError, match="name is empty"):
        Operator("", "Cat")

    with pytest.raises(ValueError, match="sort is empty"):
        Operator("np", "")

    with pytest.raises(TypeError, match="name is not a string"):
        Operator(cast("str", 7), "Cat")


def test_two_operators_are_equal_when_every_field_is_and_hash_by_name() -> None:
    here = Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash")
    there = Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash")

    assert here is not there
    assert here == there
    assert hash(here) == hash(there)
    assert here != "fwd"


@pytest.mark.parametrize(
    "differing",
    [
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("fwd", "Lf", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("fwd", "Cat", arg_sorts=("Cat",), family="slash"),
        Operator(
            "fwd",
            "Cat",
            arg_sorts=("Cat", "Cat"),
            binding=(BindingSite((0,), "Cat"),),
            family="slash",
        ),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="primitive"),
        Operator(
            "fwd",
            "Cat",
            arg_sorts=("Cat", "Cat"),
            payload_sort="constant",
            family="slash",
        ),
        Operator(
            "fwd",
            "Cat",
            arg_sorts=("Cat", "Cat"),
            family="slash",
            erases_to="bwd",
        ),
    ],
    ids=["name", "sort", "arg_sorts", "binding", "family", "payload_sort", "erases_to"],
)
def test_operators_differing_in_any_one_field_are_not_equal(
    differing: Operator,
) -> None:
    assert differing != Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash")


def test_an_operator_renders_as_the_call_that_rebuilds_it() -> None:
    op = Operator("s[dcl]", "Cat", erases_to="s")

    assert repr(op) == (
        "Operator('s[dcl]', 'Cat', arg_sorts=(), binding=(), payload_sort='none', "
        "family='primitive', erases_to='s')"
    )


def test_a_vocabulary_identifies_each_constant_by_its_position() -> None:
    vocabulary = Vocabulary("Tm", ("jo", "bo", "sleep"))

    assert vocabulary.sort == "Tm"
    assert vocabulary.names == ("jo", "bo", "sleep")
    assert [vocabulary.id(name) for name in vocabulary.names] == [0, 1, 2]
    assert [vocabulary.name(at) for at in range(3)] == ["jo", "bo", "sleep"]
    assert len(vocabulary) == 3
    assert "bo" in vocabulary
    assert "snore" not in vocabulary


def test_a_vocabulary_digest_is_the_sha256_of_its_names_in_order() -> None:
    names = ("jo", "bo")
    written = json.dumps(list(names), separators=(",", ":")).encode()

    assert Vocabulary("Tm", names).digest == sha256(written).hexdigest()
    assert Vocabulary("Ty", names).digest == Vocabulary("Tm", names).digest
    assert Vocabulary("Tm", ("bo", "jo")).digest != Vocabulary("Tm", names).digest


def test_two_vocabularies_are_equal_when_sort_and_names_are() -> None:
    here = Vocabulary("Tm", ("jo",))

    assert here == Vocabulary("Tm", ["jo"])
    assert hash(here) == hash(Vocabulary("Tm", ("jo",)))
    assert here != Vocabulary("Ty", ("jo",))
    assert here != Vocabulary("Tm", ("jo", "bo"))
    assert repr(here) == "Vocabulary('Tm', ('jo',))"


def test_a_vocabulary_refuses_a_repeated_or_empty_name() -> None:
    with pytest.raises(ValueError, match="repeats the constant 'jo'"):
        Vocabulary("Tm", ("jo", "bo", "jo"))

    with pytest.raises(ValueError, match=r"names\[0\] is empty"):
        Vocabulary("Tm", ("",))

    with pytest.raises(ValueError, match="sort is empty"):
        Vocabulary("", ("jo",))


def test_a_vocabulary_refuses_a_name_or_identifier_it_does_not_hold() -> None:
    vocabulary = Vocabulary("Tm", ("jo",))

    with pytest.raises(UnknownConstantError) as caught:
        vocabulary.id("bo")

    assert (caught.value.sort, caught.value.name) == ("Tm", "bo")

    for identifier in (-1, 1):
        with pytest.raises(ValueError, match="no constant with identifier"):
            vocabulary.name(identifier)

    boolean = False
    boolean_identifier = cast("int", boolean)

    with pytest.raises(TypeError, match="identifier is not an integer"):
        vocabulary.name(boolean_identifier)


def test_codes_run_from_one_and_leave_the_padding_code_to_no_operator() -> None:
    signature = categories()
    codes = [signature.code(op) for op in signature.operators]

    assert Signature.PAD == 0
    assert codes == list(range(1, len(signature.operators) + 1))


def test_codes_order_sort_major_then_family_then_arity() -> None:
    signature = two_sorted()

    assert [op.name for op in signature.operators] == [
        "#const[Ty]",
        "#bvar[Ty]",
        "#fvar[Ty]",
        "arrow",
        "all",
        "#const[Tm]",
        "#bvar[Tm]",
        "#fvar[Tm]",
        "lam",
        "most",
        "app",
    ]


def test_a_two_place_binder_beside_one_place_binders_keeps_its_family_whole() -> None:
    signature = Signature(
        "quantifiers",
        ("Lf",),
        (
            Operator("not", "Lf", arg_sorts=("Lf",), family="connective"),
            Operator(
                "exists",
                "Lf",
                arg_sorts=("Lf",),
                binding=(BindingSite((0,), "Lf"),),
                family="binder",
            ),
            Operator("and", "Lf", arg_sorts=("Lf", "Lf"), family="connective"),
            Operator(
                "most",
                "Lf",
                arg_sorts=("Lf", "Lf"),
                binding=(BindingSite((0, 1), "Lf"),),
                family="binder",
            ),
            Operator(
                "forall",
                "Lf",
                arg_sorts=("Lf",),
                binding=(BindingSite((0,), "Lf"),),
                family="binder",
            ),
        ),
    )
    families = [op.family for op in signature.operators]

    assert families == [
        "variable",
        "variable",
        "connective",
        "connective",
        "binder",
        "binder",
        "binder",
    ]
    assert [op.name for op in signature.operators if op.family == "binder"] == [
        "exists",
        "forall",
        "most",
    ]


def test_declaration_order_within_one_family_and_arity_is_what_orders_the_codes() -> (
    None
):
    sorts = ("Cat",)
    operators = (
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("np", "Cat"),
        Operator("tc", "Cat", arg_sorts=("Cat",)),
        Operator("n", "Cat"),
    )
    signature = Signature("mixed", sorts, operators)

    assert [op.name for op in signature.operators] == ["np", "n", "tc", "fwd"]
    assert signature.max_arity == 2


def test_reordering_a_declaration_moves_the_codes_and_the_digest() -> None:
    sorts = ("Cat",)
    here = (Operator("np", "Cat"), Operator("n", "Cat"))
    signature, reordered = (
        Signature("mixed", sorts, here),
        Signature("mixed", sorts, tuple(reversed(here))),
    )

    assert signature.code("np") == 1
    assert reordered.code("np") == 2
    assert signature.digest != reordered.digest


def test_one_declaration_gives_one_table_and_one_digest() -> None:
    here, there = two_sorted(), two_sorted()

    assert here is not there
    assert here.operators == there.operators
    assert [here.code(op) for op in here.operators] == [
        there.code(op) for op in there.operators
    ]
    assert here.digest == there.digest


def test_the_digest_is_the_same_in_an_interpreter_that_hashes_differently() -> None:
    here = {
        signature.name: {
            "digest": signature.digest,
            "codes": {op.name: signature.code(op) for op in signature.operators},
        }
        for signature in (categories(), formulas())
    }
    first = tables_from_a_fresh_interpreter("1")
    second = tables_from_a_fresh_interpreter("1024")

    assert first == here
    assert second == here


def test_the_digest_moves_when_a_code_moves() -> None:
    sorts = ("Cat",)
    two = (Operator("np", "Cat"), Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")))
    grown = Signature("mixed", sorts, (*two[:1], Operator("n", "Cat"), *two[1:]))

    assert Signature("mixed", sorts, two).code("fwd") == 2
    assert grown.code("fwd") == 3
    assert Signature("mixed", sorts, two).digest != grown.digest


def test_the_digest_reads_an_erasure_target_resolved_rather_than_as_written() -> None:
    sorts = ("Cat",)
    implicit = Signature("mixed", sorts, (Operator("np", "Cat"),))
    explicit = Signature("mixed", sorts, (Operator("np", "Cat", erases_to="np"),))

    assert implicit.digest == explicit.digest


def test_the_digest_moves_when_an_erasure_target_moves() -> None:
    sorts = ("Cat",)
    bare = (Operator("s", "Cat"), Operator("s[dcl]", "Cat"))
    erasing = (bare[0], Operator("s[dcl]", "Cat", erases_to="s"))

    assert (
        Signature("mixed", sorts, bare).digest
        != Signature("mixed", sorts, erasing).digest
    )


def test_the_digest_moves_when_a_sort_moves() -> None:
    sorts = ("Cat", "Lf")
    here = Signature("mixed", sorts, (Operator("np", "Cat"),))
    there = Signature("mixed", sorts, (Operator("np", "Lf"),))

    assert here.code("np") == there.code("np")
    assert here.digest != there.digest


def test_the_digest_reads_the_whole_sort_inventory_in_order() -> None:
    here = Signature("empty", ("A", "B"), ())
    reordered = Signature("empty", ("B", "A"), ())

    assert here.digest != reordered.digest


def test_the_digest_moves_when_an_arity_moves() -> None:
    sorts = ("Cat",)
    here = Signature("mixed", sorts, (Operator("tc", "Cat", arg_sorts=("Cat",)),))
    there = Signature(
        "mixed", sorts, (Operator("tc", "Cat", arg_sorts=("Cat", "Cat")),)
    )

    assert here.code("tc") == there.code("tc")
    assert here.digest != there.digest


def test_the_digest_moves_when_an_argument_sort_or_family_moves() -> None:
    here = Signature(
        "mixed",
        ("A", "B"),
        (Operator("f", "A", arg_sorts=("A",), family="left"),),
    )
    resorted = Signature(
        "mixed",
        ("A", "B"),
        (Operator("f", "A", arg_sorts=("B",), family="left"),),
    )
    refamilied = Signature(
        "mixed",
        ("A", "B"),
        (Operator("f", "A", arg_sorts=("A",), family="right"),),
    )

    assert len({here.digest, resorted.digest, refamilied.digest}) == 3


def test_the_digest_moves_when_an_operator_is_renamed() -> None:
    sorts = ("Cat",)
    bare = Operator("s", "Cat")
    here = Signature("mixed", sorts, (bare, Operator("s[dcl]", "Cat", erases_to="s")))
    there = Signature("mixed", sorts, (bare, Operator("s[b]", "Cat", erases_to="s")))

    assert here.code("s") == there.code("s")
    assert here.digest != there.digest


def scoped(sites: tuple[BindingSite, ...]) -> Signature:
    """Return a two-sorted signature whose one binder declares the sites."""
    return Signature(
        "scoped",
        ("Ty", "Tm"),
        (Operator("q", "Tm", arg_sorts=("Ty", "Tm", "Tm"), binding=sites),),
        classifiers={"Tm": "Ty"},
    )


def test_the_digest_moves_when_a_binding_site_moves() -> None:
    base = scoped((BindingSite((1,), "Tm"),))
    variants = (
        scoped((BindingSite((2,), "Tm"),)),
        scoped((BindingSite((1, 2), "Tm"),)),
        scoped((BindingSite((1,), "Tm", 0),)),
        scoped((BindingSite((1,), "Tm"), BindingSite((2,), "Tm"))),
    )

    assert len({base.digest, *(variant.digest for variant in variants)}) == 5


def test_the_digest_moves_with_a_vocabulary_an_open_sort_and_a_classifier() -> None:
    bare = Signature("plain", ("Ty", "Tm"), ())
    variants = (
        Signature("plain", ("Ty", "Tm"), (), vocabularies=(Vocabulary("Tm", ("a",)),)),
        Signature("plain", ("Ty", "Tm"), (), vocabularies=(Vocabulary("Tm", ("b",)),)),
        Signature("plain", ("Ty", "Tm"), (), open_sorts=("Ty",)),
        Signature("plain", ("Ty", "Tm"), (), classifiers={"Tm": "Ty"}),
    )

    assert len({bare.digest, *(variant.digest for variant in variants)}) == 5


def test_constant_feature_cores_are_direct_and_part_of_the_signature() -> None:
    vocabulary = Vocabulary("Cat", ("S", "S[dcl]", "S[q]"))
    bare = Signature("categories", ("Cat",), (), vocabularies=(vocabulary,))
    featured = Signature(
        "categories",
        ("Cat",),
        (),
        vocabularies=(vocabulary,),
        constant_cores={"Cat": {"S[q]": "S", "S[dcl]": "S"}},
    )

    assert featured.constant_cores == {"Cat": {"S[dcl]": "S", "S[q]": "S"}}
    assert featured.constant_core("Cat", "S") == "S"
    assert featured.constant_core("Cat", "S[dcl]") == "S"
    assert featured.with_holes().constant_cores == featured.constant_cores
    assert featured.digest != bare.digest


@pytest.mark.parametrize(
    ("cores", "reason"),
    [
        ({"Other": {"S[dcl]": "S"}}, "has no vocabulary"),
        ({"Cat": {"missing": "S"}}, "source 'missing'"),
        ({"Cat": {"S[dcl]": "missing"}}, "target 'missing'"),
        ({"Cat": {"S": "S"}}, "names itself"),
        (
            {"Cat": {"S[q]": "S[dcl]", "S[dcl]": "S"}},
            "must be direct",
        ),
    ],
)
def test_a_malformed_constant_core_map_is_refused(
    cores: dict[str, dict[str, str]], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        Signature(
            "categories",
            ("Cat",),
            (),
            vocabularies=(Vocabulary("Cat", ("S", "S[dcl]", "S[q]")),),
            constant_cores=cores,
        )


def test_a_signature_declaring_nothing_still_answers_for_its_table() -> None:
    empty = Signature("empty", (), ())

    assert (empty.operators, empty.max_arity) == ((), 0)
    assert not empty.has_binders
    assert not empty.has_payloads
    assert not empty.has_holes
    assert len(empty.digest) == 64
    assert empty.digest != empty.with_holes().digest

    with pytest.raises(UnknownOperatorError):
        empty.operator(1)


def test_a_signature_reports_a_payload_when_the_kernel_derives_one() -> None:
    sorts = ("Lf",)
    bare = Operator("top", "Lf")

    assert not Signature("plain", sorts, (bare,)).has_payloads
    assert Signature("open", sorts, (bare,), open_sorts=sorts).has_payloads
    assert Signature(
        "named", sorts, (bare,), vocabularies=(Vocabulary("Lf", ()),)
    ).has_payloads
    assert formulas().has_payloads
    assert not categories().has_payloads


def test_a_vocabulary_derives_one_constant_operator_for_its_sort() -> None:
    signature = two_sorted()
    constant = signature.constant("Tm")

    assert constant is signature.by_name("#const[Tm]")
    assert (constant.sort, constant.arity) == ("Tm", 0)
    assert (constant.payload_sort, constant.family) == ("constant", "constant")
    assert signature.vocabulary("Tm") == Vocabulary("Tm", ("jo", "sleep"))
    assert dict(signature.vocabularies) == {
        "Ty": Vocabulary("Ty", ("e", "t")),
        "Tm": Vocabulary("Tm", ("jo", "sleep")),
    }


def test_each_bound_sort_gets_a_bound_and_a_free_variable_operator() -> None:
    signature = two_sorted()

    assert signature.bound_sorts == ("Ty", "Tm")
    assert signature.open_sorts == ("Ty",)

    for sort in ("Ty", "Tm"):
        bound, free = signature.bound(sort), signature.free(sort)

        assert (bound.name, free.name) == (f"#bvar[{sort}]", f"#fvar[{sort}]")
        assert (bound.payload_sort, free.payload_sort) == ("index", "free")
        assert (bound.family, free.family) == ("variable", "variable")
        assert (bound.sort, free.sort) == (sort, sort)


def test_an_open_sort_no_binder_binds_has_free_variables_only() -> None:
    signature = Signature(
        "types",
        ("Ty",),
        (Operator("arrow", "Ty", arg_sorts=("Ty", "Ty")),),
        open_sorts=("Ty",),
    )

    assert signature.free("Ty").name == "#fvar[Ty]"
    assert signature.bound_sorts == ()

    with pytest.raises(UnknownOperatorError) as caught:
        signature.bound("Ty")

    assert caught.value.operator == "#bvar[Ty]"


def test_a_sort_declaring_nothing_has_no_derived_operator() -> None:
    signature = categories()

    for lookup, name in (
        (signature.constant, "#const[Cat]"),
        (signature.bound, "#bvar[Cat]"),
        (signature.free, "#fvar[Cat]"),
        (signature.hole, "#hole[Cat]"),
        (signature.vocabulary, "#const[Cat]"),
    ):
        with pytest.raises(UnknownOperatorError) as caught:
            lookup("Cat")

        assert caught.value.operator == name


def test_holes_give_every_sort_its_own_hole_last_in_its_block() -> None:
    signature = two_sorted().with_holes()

    assert signature.has_holes

    for sort in ("Ty", "Tm"):
        hole = signature.hole(sort)
        block = [op for op in signature.operators if op.sort == sort]

        assert (hole.name, hole.sort, hole.arity) == (f"#hole[{sort}]", sort, 0)
        assert (hole.family, hole.payload_sort) == ("hole", "none")
        assert block[-1] is hole


def test_with_holes_and_with_vocabulary_preserve_the_rest_of_the_declaration() -> None:
    base = Signature(
        "logic",
        ("Ty", "Lf"),
        (Operator("s", "Lf", family="formula"),),
        vocabularies=(Vocabulary("Ty", ("e",)),),
        open_sorts=("Ty",),
        classifiers={"Lf": "Ty"},
        allows_vacuous_binding=False,
        features={"s": ("past",)},
    )
    vocabulary = Vocabulary("Lf", ("jo",))

    for derived in (base.with_holes(), base.with_vocabulary(vocabulary)):
        assert not derived.allows_vacuous_binding
        assert derived.features == base.features
        assert derived.open_sorts == base.open_sorts
        assert derived.classifiers == base.classifiers
        assert derived.declarations == base.declarations
        assert derived.vocabulary("Ty") == base.vocabulary("Ty")
        assert derived.erase("s[past]") is derived.by_name("s")
        assert derived.digest != base.digest

    extended = base.with_vocabulary(vocabulary)

    assert extended.vocabulary("Lf") is vocabulary
    assert not extended.has_holes
    assert base.with_holes().with_vocabulary(vocabulary).has_holes
    assert extended.with_vocabulary(Vocabulary("Lf", ("bo",))).vocabulary("Lf") == (
        Vocabulary("Lf", ("bo",))
    )


@pytest.mark.parametrize(
    ("operator", "reason"),
    [
        (Operator("#const[Lf]", "Lf"), "takes a name starting with '#'"),
        (Operator("x", "Lf", family="variable"), "declares the family 'variable'"),
        (Operator("x", "Lf", family="constant"), "declares the family 'constant'"),
        (Operator("x", "Lf", family="hole"), "declares the family 'hole'"),
        (Operator("x", "Lf", payload_sort="index"), "declares a 'index' payload"),
    ],
    ids=["name", "variable", "constant", "hole", "payload"],
)
def test_a_declared_operator_claiming_what_the_kernel_derives_is_refused(
    operator: Operator, reason: str
) -> None:
    with pytest.raises(ReservedDeclarationError) as caught:
        Signature("logic", ("Lf",), (operator,))

    assert caught.value.operator == operator.name
    assert reason in str(caught.value)


def test_a_vocabulary_or_open_sort_declared_twice_is_refused() -> None:
    with pytest.raises(DuplicateNameError) as vocabulary:
        Signature(
            "logic",
            ("Lf",),
            (),
            vocabularies=(Vocabulary("Lf", ("a",)), Vocabulary("Lf", ("b",))),
        )

    with pytest.raises(DuplicateNameError) as open_sort:
        Signature("logic", ("Lf",), (), open_sorts=("Lf", "Lf"))

    assert (vocabulary.value.name, vocabulary.value.kind) == ("Lf", "vocabulary")
    assert (open_sort.value.name, open_sort.value.kind) == ("Lf", "open sort")


@pytest.mark.parametrize(
    "build",
    [
        lambda: Signature(
            "logic", ("Lf",), (), vocabularies=(Vocabulary("Ty", ("e",)),)
        ),
        lambda: Signature("logic", ("Lf",), (), open_sorts=("Ty",)),
        lambda: Signature("logic", ("Lf",), (), classifiers={"Ty": "Lf"}),
        lambda: Signature("logic", ("Lf",), (), classifiers={"Lf": "Ty"}),
        lambda: Signature(
            "logic",
            ("Lf",),
            (
                Operator(
                    "q",
                    "Lf",
                    arg_sorts=("Lf",),
                    binding=(BindingSite((0,), "Ty"),),
                ),
            ),
        ),
    ],
    ids=["vocabulary", "open", "classified", "classifier", "bound"],
)
def test_a_sort_named_anywhere_but_the_inventory_is_refused(
    build: Callable[[], Signature],
) -> None:
    with pytest.raises(UndeclaredSortError) as caught:
        build()

    assert caught.value.sort == "Ty"


def test_an_annotation_must_hold_the_classifier_sort_of_its_variable() -> None:
    church = scoped((BindingSite((1,), "Tm", 0),))

    assert church.classifiers == {"Tm": "Ty"}

    with pytest.raises(BindingShapeError, match="not its classifier sort 'Ty'"):
        scoped((BindingSite((1,), "Tm", 2),))

    with pytest.raises(BindingShapeError, match="has no declared classifier"):
        Signature(
            "unclassified",
            ("Ty", "Tm"),
            (
                Operator(
                    "q",
                    "Tm",
                    arg_sorts=("Ty", "Tm"),
                    binding=(BindingSite((1,), "Tm", 0),),
                ),
            ),
        )


def test_the_declarations_are_what_was_declared_in_declaration_order() -> None:
    signature = two_sorted()

    assert [op.name for op in signature.declarations] == [
        "arrow",
        "lam",
        "app",
        "most",
        "all",
    ]


def test_holes_and_the_binding_policy_refuse_a_non_boolean_value() -> None:
    with pytest.raises(TypeError, match="holes must be a bool"):
        Signature("logic", ("Lf",), (), holes=cast("bool", 1))

    with pytest.raises(TypeError, match="allows_vacuous_binding must be a bool"):
        Signature(
            "logic",
            ("Lf",),
            (Operator("top", "Lf"),),
            allows_vacuous_binding=cast("bool", 1),
        )


def test_binding_policy_is_declared_preserved_and_fingerprinted() -> None:
    operator = Operator("top", "Lf")
    allowing = Signature("logic", ("Lf",), (operator,))
    requiring = Signature(
        "logic",
        ("Lf",),
        (operator,),
        allows_vacuous_binding=False,
    )

    assert allowing.allows_vacuous_binding
    assert not requiring.allows_vacuous_binding
    assert not requiring.with_holes().allows_vacuous_binding
    assert allowing.digest != requiring.digest


def test_a_name_a_code_and_an_operator_are_three_views_of_one_row() -> None:
    signature = categories()
    op = signature.by_name("fwd")

    assert signature.operator(signature.code(op)) is op
    assert signature.code(op.name) == signature.code(op)


def test_a_code_outside_the_alphabet_is_refused_by_the_code_it_was_asked_for() -> None:
    signature = categories()
    width = len(signature.operators)

    for code in (Signature.PAD, -1, width + 1):
        with pytest.raises(UnknownOperatorError) as caught:
            signature.operator(code)

        assert caught.value.operator == code

    assert signature.operator(width) is signature.operators[-1]


def test_an_operator_of_another_declaration_cannot_borrow_a_code() -> None:
    signature = categories()
    twin = Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash")
    impostor = Operator("fwd", "Cat", arg_sorts=("Cat",), family="slash")

    assert signature.code(twin) == signature.code("fwd")

    with pytest.raises(UnknownOperatorError) as caught:
        signature.code(impostor)

    assert caught.value.operator == "fwd"


def test_a_lookup_by_a_name_no_operator_carries_is_refused() -> None:
    signature = categories()

    with pytest.raises(UnknownOperatorError):
        signature.by_name("tc")

    with pytest.raises(UnknownOperatorError):
        signature.erase("tc")


def test_an_operator_declaring_no_target_erases_to_itself() -> None:
    signature = categories()

    assert signature.erase("np") is signature.by_name("np")
    assert signature.erase("s[dcl]") is signature.by_name("s")


def test_an_operator_naming_itself_as_its_target_erases_to_itself() -> None:
    signature = Signature("mixed", ("Cat",), (Operator("np", "Cat", erases_to="np"),))

    assert signature.erase("np") is signature.by_name("np")


def test_a_repeated_operator_name_and_a_repeated_sort_are_both_refused() -> None:
    with pytest.raises(DuplicateNameError) as repeated_operator:
        Signature("mixed", ("Cat",), (Operator("np", "Cat"), Operator("np", "Cat")))

    assert (repeated_operator.value.name, repeated_operator.value.kind) == (
        "np",
        "operator",
    )

    with pytest.raises(DuplicateNameError) as repeated_sort:
        Signature("mixed", ("Cat", "Cat"), ())

    assert (repeated_sort.value.name, repeated_sort.value.kind) == ("Cat", "sort")


def test_an_operator_naming_a_sort_the_inventory_does_not_hold_is_refused() -> None:
    with pytest.raises(UndeclaredSortError) as built:
        Signature("mixed", ("Cat",), (Operator("np", "Lf"),))

    assert (built.value.sort, built.value.operator) == ("Lf", "np")

    with pytest.raises(UndeclaredSortError) as taken:
        Signature("mixed", ("Cat",), (Operator("fwd", "Cat", arg_sorts=("Lf",)),))

    assert (taken.value.sort, taken.value.operator) == ("Lf", "fwd")


def test_an_erasure_to_an_operator_the_signature_does_not_declare_is_refused() -> None:
    with pytest.raises(UnknownOperatorError) as caught:
        Signature("mixed", ("Cat",), (Operator("s[dcl]", "Cat", erases_to="s"),))

    assert caught.value.operator == "s"


@pytest.mark.parametrize(
    ("operators", "requirement"),
    [
        (
            (
                Operator("s", "Cat"),
                Operator("s[b]", "Cat", erases_to="s"),
                Operator("s[dcl]", "Cat", erases_to="s[b]"),
            ),
            "idempotent",
        ),
        (
            (
                Operator("np", "Cat"),
                Operator("tc", "Cat", arg_sorts=("Cat",), erases_to="np"),
            ),
            "arity-preserving",
        ),
        (
            (
                Operator("e", "Lf"),
                Operator("np", "Cat", erases_to="e"),
            ),
            "sort-preserving",
        ),
    ],
    ids=["idempotent", "arity-preserving", "sort-preserving"],
)
def test_an_erasure_breaking_any_one_requirement_is_refused(
    operators: tuple[Operator, ...],
    requirement: str,
) -> None:
    with pytest.raises(ErasureError) as caught:
        Signature("mixed", ("Cat", "Lf"), operators)

    assert caught.value.requirement == requirement
    assert requirement in str(caught.value)


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
@pytest.mark.parametrize(
    "law",
    SIGNATURE_INVARIANTS,
    ids=[law.__name__ for law in SIGNATURE_INVARIANTS],
)
def test_every_signature_invariant_holds_of_each_reference_signature(
    law: Callable[[AlphabetFixture[Signature, Operator]], None],
    build: Callable[[], Signature],
) -> None:
    law(alphabet(build))


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
def test_running_the_family_in_one_call_runs_it_against_a_signature(
    build: Callable[[], Signature],
) -> None:
    check_signature_invariants(alphabet(build))


def test_the_determinism_invariant_sees_a_code_that_moved() -> None:
    sorts = ("Cat",)
    here = (Operator("np", "Cat"), Operator("n", "Cat"))
    fixture = alphabet(
        lambda: Signature("mixed", sorts, here),
        lambda: Signature("mixed", sorts, tuple(reversed(here))),
    )

    check_signature_invariants(alphabet(lambda: Signature("mixed", sorts, here)))

    with pytest.raises(LawViolationError, match="coded 1 once and 2 again"):
        check_signature_invariants(fixture)


def erased_inventory(signature: Signature) -> set[str]:
    return {signature.erase(op).name for op in signature.operators}


def test_a_declared_feature_becomes_an_operator_erasing_to_its_primitive() -> None:
    signature = Signature(
        "mixed", ("Cat",), (Operator("s", "Cat"),), features={"s": ("dcl",)}
    )
    featured = signature.featured("s", "dcl")

    assert featured.name == "s[dcl]"
    assert featured is signature.by_name("s[dcl]")
    assert signature.erase(featured) is signature.by_name("s")
    assert signature.features == {"s": ("dcl",)}


def test_three_features_on_one_primitive_are_three_codes_over_one_core() -> None:
    values = ("dcl", "b", "ng")
    signature = Signature(
        "mixed", ("Cat",), (Operator("s", "Cat"),), features={"s": values}
    )
    featured = [signature.featured("s", each) for each in values]
    codes = {signature.code(each) for each in featured}

    assert len(codes) == len(values)
    assert {signature.erase(each) for each in featured} == {signature.by_name("s")}
    assert erased_inventory(signature) == {"s"}


def test_a_feature_leaves_the_primitive_inventory_where_it_was() -> None:
    sorts = ("Cat",)
    primitives = (Operator("np", "Cat"), Operator("s", "Cat"))
    bare = Signature("mixed", sorts, primitives)
    featured = Signature("mixed", sorts, primitives, features={"s": ("dcl", "b")})

    assert erased_inventory(featured) == erased_inventory(bare)
    assert len(erased_inventory(featured)) < len(featured.operators)
    assert len(featured.operators) == len(bare.operators) + 2


def test_a_featured_operator_copies_every_field_of_its_primitive() -> None:
    tc = Operator(
        "tc",
        "Cat",
        arg_sorts=("Cat",),
        binding=(BindingSite((0,), "Cat"),),
        family="unary",
    )
    signature = Signature("mixed", ("Cat",), (tc,), features={"tc": ("nb",)})
    featured = signature.featured("tc", "nb")

    assert (featured.arity, featured.sort) == (tc.arity, tc.sort)
    assert (featured.binding, featured.family) == (tc.binding, tc.family)
    assert featured.payload_sort == tc.payload_sort
    assert featured.erases_to == "tc"


def test_a_feature_on_a_primitive_the_signature_does_not_declare_is_refused() -> None:
    with pytest.raises(UnknownOperatorError) as caught:
        Signature("mixed", ("Cat",), (Operator("np", "Cat"),), features={"s": ("dcl",)})

    assert caught.value.operator == "s"


def test_a_feature_admitted_twice_on_one_primitive_is_refused() -> None:
    with pytest.raises(DuplicateNameError) as caught:
        Signature(
            "mixed", ("Cat",), (Operator("s", "Cat"),), features={"s": ("dcl", "dcl")}
        )

    assert caught.value.name == "s[dcl]"


def test_a_pair_an_operator_already_carries_the_name_of_is_refused() -> None:
    declared = (Operator("s", "Cat"), Operator("s[dcl]", "Cat", erases_to="s"))

    with pytest.raises(DuplicateNameError) as caught:
        Signature("mixed", ("Cat",), declared, features={"s": ("dcl",)})

    assert caught.value.name == "s[dcl]"


def test_a_feature_on_an_operator_that_is_itself_featured_is_refused() -> None:
    declared = (Operator("s", "Cat"), Operator("s[dcl]", "Cat", erases_to="s"))

    with pytest.raises(ErasureError) as caught:
        Signature("mixed", ("Cat",), declared, features={"s[dcl]": ("q",)})

    assert caught.value.requirement == "idempotent"


def test_a_feature_that_is_not_a_non_empty_string_is_refused() -> None:
    sorts, primitive = ("Cat",), (Operator("s", "Cat"),)

    with pytest.raises(ValueError, match=r"features\['s'\]\[0\] is empty"):
        Signature("mixed", sorts, primitive, features={"s": ("",)})

    with pytest.raises(TypeError, match="features is not a string"):
        Signature("mixed", sorts, primitive, features={cast("str", 7): ("dcl",)})


def test_the_featured_lookup_refuses_a_pair_no_declaration_generated() -> None:
    signature = Signature(
        "mixed", ("Cat",), (Operator("s", "Cat"),), features={"s": ("dcl",)}
    )

    with pytest.raises(UnknownOperatorError) as caught:
        signature.featured("s", "b")

    assert caught.value.operator == "s[b]"

    with pytest.raises(ValueError, match="feature is empty"):
        signature.featured("s", "")


def test_the_digest_moves_when_a_feature_is_admitted() -> None:
    sorts, primitive = ("Cat",), (Operator("s", "Cat"),)
    bare = Signature("mixed", sorts, primitive)
    featured = Signature("mixed", sorts, primitive, features={"s": ("dcl",)})

    assert bare.code("s") == featured.code("s")
    assert bare.digest != featured.digest


def test_the_digest_distinguishes_a_feature_table_from_an_explicit_twin() -> None:
    sorts = ("Cat",)
    primitive = Operator("s", "Cat")
    explicit = Signature(
        "mixed",
        sorts,
        (primitive, Operator("s[dcl]", "Cat", erases_to="s")),
    )
    generated = Signature("mixed", sorts, (primitive,), features={"s": ("dcl",)})

    assert explicit.operators == generated.operators
    assert explicit.digest != generated.digest


def test_a_few_hundred_featured_primitives_add_no_primitive_and_no_slash() -> None:
    values = tuple(f"f{at}" for at in range(8))
    primitives = tuple(Operator(f"p{at}", "Cat") for at in range(40))
    slashes = tuple(
        Operator(name, "Cat", arg_sorts=("Cat", "Cat"), family="slash")
        for name in ("fwd", "bwd")
    )
    inventory = Signature(
        "wide",
        ("Cat",),
        (*primitives, *slashes),
        features={op.name: values for op in primitives},
    )
    featured = len(primitives) * len(values)

    assert featured > 100
    assert len(inventory.operators) == len(primitives) + len(slashes) + featured
    assert erased_inventory(inventory) == {op.name for op in (*primitives, *slashes)}
    assert len(erased_inventory(inventory)) < len(inventory.operators)


def test_a_signature_declaring_no_feature_carries_an_empty_inventory() -> None:
    assert Signature("empty", (), ()).features == {}
    assert formulas().features == {}


def test_the_feature_inventory_is_a_read_only_view_of_the_declaration() -> None:
    declared = {"s": ["dcl"]}
    signature = Signature("mixed", ("Cat",), (Operator("s", "Cat"),), features=declared)

    declared["s"].append("b")

    assert signature.features["s"] == ("dcl",)

    with pytest.raises(TypeError):
        cast("dict[str, tuple[str, ...]]", signature.features)["np"] = ()


def test_every_arity_declared_answers_with_its_own_operators_in_code_order() -> None:
    signature = formulas()
    blocks = {
        arity: [op.name for op in signature.by_arity(arity)]
        for arity in range(signature.max_arity + 1)
    }

    assert blocks == {
        0: ["#const[Lf]", "#bvar[Lf]", "#fvar[Lf]"],
        1: ["lam"],
        2: ["app"],
        3: ["ite"],
    }


def test_an_arity_no_operator_takes_answers_with_nothing() -> None:
    signature = categories()

    assert signature.by_arity(1) == ()
    assert signature.by_arity(signature.max_arity + 1) == ()


def test_the_arity_blocks_partition_the_alphabet() -> None:
    signature = formulas()
    blocks = [signature.by_arity(k) for k in range(signature.max_arity + 1)]
    codes = [signature.code(op) for block in blocks for op in block]

    assert sorted(codes) == [signature.code(op) for op in signature.operators]
    assert all(
        [signature.code(op) for op in block]
        == sorted(signature.code(op) for op in block)
        for block in blocks
    )


def test_by_arity_refuses_a_negative_arity_and_an_arity_that_is_not_whole() -> None:
    signature = categories()

    with pytest.raises(ValueError, match="arity is negative"):
        signature.by_arity(-1)

    with pytest.raises(TypeError, match="arity is not an integer"):
        signature.by_arity(cast("int", 2.0))


def test_a_unary_and_a_ternary_operator_are_ordered_and_found_by_arity() -> None:
    sorts = ("Lf",)
    operators = (
        Operator("cond", "Lf", arg_sorts=("Lf", "Lf", "Lf")),
        Operator("app", "Lf", arg_sorts=("Lf", "Lf")),
        Operator("not", "Lf", arg_sorts=("Lf",)),
        Operator("top", "Lf"),
    )
    signature = Signature("mixed", sorts, operators)

    assert [op.name for op in signature.operators] == ["top", "not", "app", "cond"]
    assert signature.by_arity(1) == (signature.by_name("not"),)
    assert signature.by_arity(3) == (signature.by_name("cond"),)
    assert signature.max_arity == 3


def test_a_featured_pair_takes_a_code_after_the_operators_declared_outright() -> None:
    declared = (Operator("s", "Cat"), Operator("np", "Cat"))
    signature = Signature("mixed", ("Cat",), declared, features={"s": ("dcl",)})

    assert [op.name for op in signature.operators] == ["s", "np", "s[dcl]"]

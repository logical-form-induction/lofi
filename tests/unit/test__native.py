import ast
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import version
from pathlib import Path

import pytest

from lofi import _native
from lofi._native import (
    BuildInfo,
    Profile,
    SpanDependent,
    SpanFree,
    UnitEdges,
    build_info,
)
from lofi.terms import (
    BindingSite,
    ForeignTermError,
    NotLocallyClosedError,
    Operator,
    RegionStateError,
    Signature,
    Term,
    TermError,
    TermStore,
    Vocabulary,
    abstract,
    close_binder,
    instantiate,
    open_binder,
)

STUB = ast.parse(
    Path(_native.__file__).with_name("_native.pyi").read_text(encoding="utf-8")
)


def _declared(body: list[ast.stmt]) -> list[str]:
    names: list[str] = []
    for statement in body:
        match statement:
            case ast.FunctionDef(name=name) | ast.ClassDef(name=name):
                names.append(name)
            case ast.AnnAssign(target=ast.Name(id=name)):
                names.append(name)
            case _:
                pass
    return names


def test_the_extension_reports_the_distribution_version() -> None:
    assert build_info().version == version("lofi")


def test_an_installed_extension_is_an_optimized_build() -> None:
    assert build_info().profile == Profile.RELEASE


def test_the_extension_is_the_packages_one_stable_abi_module() -> None:
    compiled = sorted(
        path.name
        for path in Path(_native.__file__).parent.iterdir()
        if path.name.endswith(tuple(EXTENSION_SUFFIXES))
    )

    expected = "_native.pyd" if sys.platform == "win32" else "_native.abi3.so"

    assert compiled == [expected]
    assert Path(_native.__file__).name == expected


@pytest.mark.parametrize("name", _declared(STUB.body))
def test_every_name_the_stub_declares_is_in_the_module(name: str) -> None:
    assert hasattr(_native, name)


@pytest.mark.parametrize(
    ("owner", "member"),
    [
        (node.name, member)
        for node in STUB.body
        if isinstance(node, ast.ClassDef)
        for member in _declared(node.body)
    ],
)
def test_every_member_the_stub_declares_is_on_its_class(
    owner: str, member: str
) -> None:
    assert hasattr(getattr(_native, owner), member)


def test_the_two_profiles_are_distinct_values() -> None:
    assert Profile.DEBUG != Profile.RELEASE
    assert Profile.RELEASE == Profile.RELEASE
    assert len({Profile.DEBUG, Profile.RELEASE, Profile.RELEASE}) == 2


def test_build_information_is_one_hashable_value() -> None:
    first, second = build_info(), build_info()

    assert first == second
    assert hash(first) == hash(second)
    assert first != Profile.RELEASE


@pytest.mark.parametrize("attribute", ["version", "profile"])
def test_build_information_is_frozen(attribute: str) -> None:
    info = build_info()

    with pytest.raises(AttributeError):
        setattr(info, attribute, None)


def test_build_information_is_constructed_only_by_the_extension() -> None:
    with pytest.raises(TypeError):
        BuildInfo()


def test_the_representation_names_the_version_and_profile() -> None:
    info = build_info()

    assert (
        repr(info) == f"BuildInfo(version='{info.version}', profile={info.profile!r})"
    )
    assert repr(Profile.RELEASE) == "Profile.RELEASE"


def test_scorer_lowerings_are_distinct_frozen_values() -> None:
    assert SpanFree() == SpanFree()
    assert SpanFree() != SpanDependent()
    assert UnitEdges() == UnitEdges()
    assert UnitEdges() != SpanFree()
    assert repr(SpanFree()) == "SpanFree()"
    assert repr(SpanDependent()) == "SpanDependent()"
    assert repr(UnitEdges()) == "UnitEdges()"


def _term_signature() -> Signature:
    sort = "Tm"

    return Signature(
        "native-terms",
        (sort,),
        (
            Operator("pair", sort, arg_sorts=(sort, sort)),
            Operator(
                "scope",
                sort,
                arg_sorts=(sort,),
                binding=(BindingSite((0,), sort),),
            ),
        ),
        vocabularies=(Vocabulary(sort, ("a", "b")),),
        open_sorts=(sort,),
    )


def test_native_terms_have_one_structural_identity_across_root_stores() -> None:
    signature = _term_signature()
    here = TermStore(signature)
    there = TermStore(signature)
    pair = signature.by_name("pair")
    near = here.op(pair, here.const("Tm", "a"), here.fvar("Tm", "x"))
    far = there.op(pair, there.const("Tm", "a"), there.fvar("Tm", "x"))

    assert near is here.op(pair, *near.args)
    assert near is not far
    assert near == far
    assert hash(near) == hash(far)
    assert near.key() == ("pair", None, (("#const[Tm]", 0, ()), ("#fvar[Tm]", "x", ())))


def test_native_store_serials_and_region_generations_are_dense() -> None:
    signature = _term_signature()
    root = TermStore(signature)
    held_nested: TermStore | None = None
    held_made: Term | None = None

    assert [root.const("Tm", name).serial for name in ("a", "b")] == [0, 1]

    with (
        root.region("first", max_nodes=8) as first,
        first.region("nested", max_nodes=8) as nested,
    ):
        made = nested.fvar("Tm", "x")
        held_nested = nested
        held_made = made
    with root.region("second", max_nodes=8) as second:
        pass

    if held_nested is None or held_made is None:
        pytest.fail("the nested region did not open")

    assert (
        root.generation,
        first.generation,
        held_nested.generation,
        second.generation,
    ) == (
        0,
        1,
        2,
        3,
    )
    assert held_made.serial == held_nested.generation << Term.GENERATION_SHIFT


def test_native_region_promotion_preserves_sharing_and_releases_storage() -> None:
    signature = _term_signature()
    root = TermStore(signature)
    pair = signature.by_name("pair")
    held_local: Term | None = None
    held_promoted: Term | None = None

    with root.region("sentence", max_nodes=8) as region:
        leaf = region.const("Tm", "a")
        local = region.op(pair, leaf, leaf)
        promoted = region.promote(local)
        held_local = local
        held_promoted = promoted

        assert promoted is root.op(pair, root.const("Tm", "a"), root.const("Tm", "a"))
        assert promoted.args[0] is promoted.args[1]

    if held_local is None or held_promoted is None:
        pytest.fail("the sentence region did not publish its terms")

    assert not region.active
    assert region.local_nodes == 0
    assert held_local.key() == held_promoted.key()

    with pytest.raises(RegionStateError, match="released"):
        region.const("Tm", "b")


def test_native_store_refuses_a_foreign_node_until_it_is_promoted() -> None:
    signature = _term_signature()
    root = TermStore(signature)
    pair = signature.by_name("pair")
    left = root.const("Tm", "a")

    with root.region("sentence", max_nodes=8) as region:
        child = region.const("Tm", "b")

        with pytest.raises(ForeignTermError):
            root.op(pair, left, child)

        promoted = region.promote(child)
        assert root.op(pair, left, promoted).args[1] is promoted


def test_native_interning_keeps_python_hash_collision_payloads_distinct() -> None:
    signature = _term_signature()
    store = TermStore(signature)
    index = signature.bound("Tm")
    first = store.op(index, payload=0)
    second = store.op(index, payload=sys.hash_info.modulus)

    assert first is store.op(index, payload=0)
    assert second is store.op(index, payload=sys.hash_info.modulus)
    assert first is not second
    assert first != second


def test_native_binding_round_trip_preserves_the_interned_binder() -> None:
    signature = _term_signature()
    store = TermStore(signature)
    pair = signature.by_name("pair")
    scope = signature.by_name("scope")
    atom = store.fvar("Tm", "x")
    body = store.op(pair, atom, store.const("Tm", "a"))
    abstracted = abstract(store, body, store.atom("Tm", "x"))
    binder = store.op(scope, abstracted)
    opened, atoms = open_binder(store, binder)

    assert instantiate(store, abstracted, atom) is body
    assert close_binder(store, opened, atoms) is binder

    with pytest.raises(NotLocallyClosedError):
        instantiate(store, store.bvar("Tm", 0), store.bvar("Tm", 0))


def test_native_term_exceptions_share_the_one_public_base() -> None:
    error = ForeignTermError(operator="pair", position=1, expected=0, received=1)

    assert isinstance(error, TermError)
    assert (error.operator, error.position, error.expected, error.received) == (
        "pair",
        1,
        0,
        1,
    )

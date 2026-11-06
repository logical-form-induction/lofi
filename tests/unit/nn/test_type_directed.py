from collections import deque
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from tests.language import language

from lofi.nn.type_directed import (
    UNRESTRICTED_ADMISSION,
    TypeDirectedAdmission,
    term_admission,
)
from lofi.spec.decoders.rnn import TypeDirectedSettings
from lofi.spec.language import (
    ConstantSpec,
    LanguageAssembly,
    LanguageIso,
    registered_language,
)
from lofi.terms import Frontier, HoleContext, MVar, TermStore, replace_at
from lofi.typecheck import LexicalSignature, Scheme, TypeChecker, TypingRule, Unifier

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import ChoiceIndex, Path, Term


class _DirectOracle(NamedTuple):
    """Independent partial-term checker used by the admission law."""

    store: TermStore
    checker: TypeChecker
    lexicon: LexicalSignature
    rules: Mapping[str, TypingRule]
    truth: Term


def _small_open_language() -> LanguageAssembly:
    declaration = registered_language("propositional")
    return LanguageIso().forward(
        declaration.with_(
            bounds={
                **declaration.bounds,
                "Tm": declaration.bounds["Tm"].with_(free_ranks=2),
            },
            constants=(
                *declaration.constants,
                ConstantSpec(name="p", type="t"),
                ConstantSpec(name="john", type="e"),
                ConstantSpec(name="sleep", type="e -> t"),
            ),
        )
    )


def _rehome(source: TermStore, target: TermStore, term: Term, /) -> Term:
    """Re-intern one type term for the independent hole-aware checker."""
    if source.signature.is_free_variable(term.op):
        if term.payload is None:
            msg = "a free type variable has no atom"
            raise ValueError(msg)

        return target.fvar(term.op.sort, source.atom_name(term.payload))

    if source.signature.is_bound_variable(term.op):
        if term.payload is None:
            msg = "a bound type variable has no index"
            raise ValueError(msg)

        return target.bvar(term.op.sort, term.payload)

    if source.signature.is_constant(term.op):
        return target.const(term.op.sort, source.payload_name(term))

    return target.op(
        target.signature.by_name(term.op.name),
        *(_rehome(source, target, child) for child in term.args),
        payload=term.payload,
    )


def _oracle(language: LanguageAssembly) -> _DirectOracle:
    """Build an independent hole-aware type-checking oracle."""
    store = TermStore(language.store.signature.with_holes())
    lexicon = LexicalSignature(
        {
            symbol: Scheme(
                _rehome(language.store, store, scheme.body),
                tuple(
                    store.atom(
                        language.store.atom_sort(atom),
                        language.store.atom_name(atom),
                    )
                    for atom in scheme.quantified
                ),
            )
            for symbol, scheme in language.lexicon.schemes.items()
        }
    )
    hole = store.signature.hole(language.term_sort)
    rules = MappingProxyType(
        {
            **language.rules,
            hole.name: TypingRule(
                hole.name,
                (),
                (),
                MVar("OracleHole", language.type_sort),
            ),
        }
    )
    return _DirectOracle(
        store,
        TypeChecker(store),
        lexicon,
        rules,
        _rehome(language.store, store, language.type_of(language.constant("p"))),
    )


def _partial(
    store: TermStore,
    root_sort: str,
    decisions: tuple[tuple[Path, int], ...],
    index: ChoiceIndex,
    /,
) -> Term:
    """Rebuild one partial frontier through public term operations."""
    skeleton = store.hole(root_sort)

    for path, code in decisions:
        selected = index.choice(code)
        replacement = index.instantiate(
            store,
            code,
            *(store.hole(sort) for sort in selected.operator.arg_sorts),
        )
        skeleton = replace_at(store, skeleton, path, replacement)

    return skeleton


def _direct_mask(
    oracle: _DirectOracle,
    decisions: tuple[tuple[Path, int], ...],
    context: HoleContext,
    index: ChoiceIndex,
    /,
) -> tuple[bool, ...]:
    skeleton = _partial(
        oracle.store, context.sort if not decisions else "Tm", decisions, index
    )
    found = []

    for choice, legal in enumerate(index.legal(context)):
        if not legal:
            found.append(False)
            continue

        selected = index.choice(choice)
        replacement = index.instantiate(
            oracle.store,
            choice,
            *(oracle.store.hole(sort) for sort in selected.operator.arg_sorts),
        )
        candidate = replace_at(oracle.store, skeleton, context.path, replacement)

        try:
            result = oracle.checker.check(
                candidate,
                oracle.lexicon,
                oracle.rules,
            )
        except ValueError:
            found.append(False)
            continue

        if result.residual or result.scheme is None:
            found.append(False)
            continue

        found.append(
            Unifier(oracle.store).unify(result.scheme.body, oracle.truth) is not None
        )

    return tuple(found)


def _hole_paths(term: Term, path: Path = ()) -> tuple[Path, ...]:
    if term.op.is_hole:
        return (path,)

    return tuple(
        found
        for side, child in enumerate(term.args)
        for found in _hole_paths(child, (*path, side))
    )


def test_known_types_and_the_type_grammar_admit_only_formulas() -> None:
    declared = language(("p", "john", "sleep"), types={"john": "e", "sleep": "e -> t"})
    admission = TypeDirectedAdmission(
        declared,
        TypeDirectedSettings(prefix="type-grammar", mask="known"),
    )

    assert admission.admits(declared.read("p"))
    assert admission.admits(declared.read("sleep(john)"))
    assert not admission.admits(declared.constant("john"))
    assert not admission.admits(declared.constant("sleep"))
    assert not admission.admits(
        declared.apply(declared.constant("sleep"), declared.constant("sleep"))
    )


def test_a_known_mask_without_a_type_prefix_admits_every_well_typed_result() -> None:
    declared = language(("john", "sleep"), types={"john": "e", "sleep": "e -> t"})
    admission = TypeDirectedAdmission(
        declared,
        TypeDirectedSettings(prefix="none", mask="known"),
    )

    assert admission.admits(declared.constant("john"))
    assert admission.admits(declared.constant("sleep"))
    assert not admission.admits(
        declared.apply(declared.constant("sleep"), declared.constant("sleep"))
    )


def test_known_types_mask_each_partial_choice_before_sampling() -> None:
    declared = language(
        ("p", "john", "sleep"),
        types={
            "john": "e",
            "sleep": "e -> t",
        },
    )
    admission = TypeDirectedAdmission(
        declared,
        TypeDirectedSettings(prefix="type-grammar", mask="known"),
    )
    index = declared.codec.choices
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    constant = declared.store.signature.constant(declared.term_sort)
    app = index.index(declared.calculus.app)
    p = index.index(constant, vocabulary.id("p"))
    john = index.index(constant, vocabulary.id("john"))
    sleep = index.index(constant, vocabulary.id("sleep"))
    bound = index.index(declared.store.signature.bound(declared.term_sort), 0)
    empty = MappingProxyType({})

    root = admission.admitted_choices(
        (),
        HoleContext((), declared.term_sort, empty, empty, 30),
        index,
    )
    function = admission.admitted_choices(
        (((), app),),
        HoleContext((0,), declared.term_sort, empty, empty, 4),
        index,
    )
    argument = admission.admitted_choices(
        (((), app), ((0,), sleep)),
        HoleContext((1,), declared.term_sort, empty, empty, 3),
        index,
    )

    assert root[p]
    assert root[app]
    assert not root[john]
    assert not root[sleep]
    assert not root[bound]
    assert function[sleep]
    assert not function[p]
    assert not function[john]
    assert argument[john]
    assert not argument[p]
    assert not argument[sleep]


def test_partial_masks_equal_direct_typing_on_every_bounded_frontier() -> None:
    declared = _small_open_language()
    admission = TypeDirectedAdmission(
        declared,
        TypeDirectedSettings(prefix="type-grammar", mask="known"),
    )
    index = declared.codec.choices
    oracle = _oracle(declared)
    pending: deque[tuple[tuple[Path, int], ...]] = deque([()])
    visited = set()
    bound = 5
    saw_nested_bound = False
    saw_fresh_free = False
    saw_reused_free = False

    while pending:
        decisions = pending.popleft()
        skeleton = _partial(oracle.store, declared.term_sort, decisions, index)

        if skeleton in visited:
            continue

        visited.add(skeleton)
        holes = _hole_paths(skeleton)

        if not holes:
            continue

        context = Frontier(
            skeleton,
            dict.fromkeys(holes),
            0.0,
        ).context(bound)
        admitted = admission.admitted_choices(decisions, context, index)
        direct = _direct_mask(oracle, decisions, context, index)

        assert admitted == direct, (decisions, context)

        for choice, allowed in enumerate(admitted):
            if allowed:
                selected = index.choice(choice)
                payload = selected.payload
                saw_nested_bound |= (
                    selected.operator.is_bound_variable
                    and payload is not None
                    and payload > 0
                )
                free_used = context.free_used.get(selected.operator.sort, 0)
                saw_fresh_free |= (
                    selected.operator.is_free_variable and payload == free_used
                )
                saw_reused_free |= (
                    selected.operator.is_free_variable
                    and payload is not None
                    and payload < free_used
                )
                pending.append((*decisions, (context.path, choice)))

    assert saw_nested_bound
    assert saw_fresh_free
    assert saw_reused_free


def test_type_admission_checks_free_variables_by_their_inferred_type() -> None:
    declared = _small_open_language()
    admission = TypeDirectedAdmission(
        declared,
        TypeDirectedSettings(prefix="type-grammar", mask="known"),
    )
    variable = declared.store.fvar(declared.term_sort, "x")

    assert admission.admits(variable)


def test_the_explicit_untyped_policy_leaves_typing_to_the_objective() -> None:
    declared = language(("sleep",), types={"sleep": "e -> t"})
    ill_typed = declared.apply(declared.constant("sleep"), declared.constant("sleep"))
    admission = term_admission(
        declared,
        TypeDirectedSettings(prefix="none", mask="none"),
    )

    assert admission is UNRESTRICTED_ADMISSION
    assert admission.admits(ill_typed)

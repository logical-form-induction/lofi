"""Term-layer laws and conformance fixtures."""

from typing import TYPE_CHECKING, Final, NamedTuple

from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._checks import LawViolationError, fail_law
from lofi._native import (
    EncodingOverflowError,
    ForeignTermError,
    NotLocallyClosedError,
)
from lofi.terms._ports import Encoding
from lofi.terms._reach import substrate_reaches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from hypothesis.strategies import SearchStrategy

    from lofi.terms._ports import Layout, Matching, Reifiable

__all__ = [
    "LAYOUT_LAWS",
    "MATCHING_LAWS",
    "PRESENTATION_LAWS",
    "SIGNATURE_INVARIANTS",
    "TERM_LAWS",
    "AlphabetFixture",
    "BindingFixture",
    "Context",
    "LayoutFixture",
    "MatchingFixture",
    "PresentationFixture",
    "TermFixture",
    "a_foreign_argument_is_refused",
    "alpha_variants_are_one_node",
    "check_layout_laws",
    "check_matching_laws",
    "check_presentation_laws",
    "check_signature_invariants",
    "check_term_laws",
    "closing_undoes_opening",
    "codes_are_a_bijection",
    "codes_are_determined_by_the_declaration",
    "codes_group_by_sort_then_family",
    "derived_fields_agree_with_their_definitions",
    "encoding_refuses_to_truncate",
    "equivalence_coarsens_identity",
    "equivalence_is_an_equivalence",
    "equivalence_survives_construction",
    "erasure_is_a_congruence",
    "erasure_is_idempotent",
    "hashing_is_stable",
    "key_is_a_section",
    "key_is_idempotent",
    "legal_next_admits_the_next_symbol",
    "mask_is_complete",
    "mask_is_sound",
    "nothing_refused_is_reachable",
    "one_component_by_identity",
    "padding_belongs_to_no_operator",
    "path_addressing_is_honest",
    "round_trip_is_identity",
    "spec_round_trip",
    "structure_is_identity",
    "structure_is_store_independent",
    "substitution_cancels_abstraction",
]

type Context[T] = Callable[[T], T]
"""A one-hole term context."""


_POOL: Final = 3
"""The term-pool size for relational examples."""


def _budget(max_examples: int) -> settings:
    """Return the Hypothesis settings for a law.

    Parameters
    ----------
    max_examples : int
        The maximum generated examples.
    """
    # disable per-example deadlines because runtime varies by machine.
    return settings(max_examples=max_examples, deadline=None)


class AlphabetFixture[A, O](NamedTuple):
    """Inputs for signature-invariant checks.

    Attributes
    ----------
    table : A
        The operator table under test.
    rebuild : Callable[[], A]
        A factory for an equivalent fresh table.
    operators : Callable[[A], tuple[O, ...]]
        A function returning the table's operators.
    code : Callable[[A, O], int]
        A function returning an operator code.
    operator : Callable[[A, int], O]
        A function returning the operator for a code.
    arity : Callable[[A, O], int]
        A function returning an operator arity.
    sort : Callable[[A, O], str]
        A function returning an operator's result sort.
    family : Callable[[A, O], str]
        A function returning an operator's family.
    digest : Callable[[A], str]
        A function returning the declaration digest.
    pad : int
        The reserved padding code.
    """

    table: A
    rebuild: Callable[[], A]
    operators: Callable[[A], tuple[O, ...]]
    code: Callable[[A, O], int]
    operator: Callable[[A, int], O]
    arity: Callable[[A, O], int]
    sort: Callable[[A, O], str]
    family: Callable[[A, O], str]
    digest: Callable[[A], str]
    pad: int = 0


def padding_belongs_to_no_operator[A, O](fixture: AlphabetFixture[A, O]) -> None:
    """Check that no operator maps to the padding code.

    Parameters
    ----------
    fixture : AlphabetFixture[A, O]
        The alphabet fixture.

    Raises
    ------
    LawViolationError
        If an operator uses the padding code.
    """
    table = fixture.table

    for op in fixture.operators(table):
        if fixture.code(table, op) != fixture.pad:
            continue

        detail = f"{op!r} is coded {fixture.pad}, which is the padding code"

        fail_law(detail, law="padding")


def codes_are_a_bijection[A, O](fixture: AlphabetFixture[A, O]) -> None:
    """Check that operator codes form a reversible bijection.

    Parameters
    ----------
    fixture : AlphabetFixture[A, O]
        The alphabet fixture.

    Raises
    ------
    LawViolationError
        If codes are missing, repeated, or do not read back.
    """
    table = fixture.table
    operators = fixture.operators(table)
    codes = sorted(fixture.code(table, op) for op in operators)
    alphabet = [code for code in range(len(operators) + 1) if code != fixture.pad]

    if codes != alphabet:
        detail = f"{len(operators)} operators are coded {codes}, and not {alphabet}"

        fail_law(detail, law="bijection")

    for op in operators:
        back = fixture.operator(table, fixture.code(table, op))

        if back == op:
            continue

        detail = f"{op!r} is coded {fixture.code(table, op)}, which reads back {back!r}"

        fail_law(detail, law="bijection")


def codes_group_by_sort_then_family[A, O](fixture: AlphabetFixture[A, O]) -> None:
    """Check that codes order sort-major, then by family, then by arity.

    Parameters
    ----------
    fixture : AlphabetFixture[A, O]
        The alphabet fixture.

    Raises
    ------
    LawViolationError
        If a sort's codes, or a family's codes within a sort, are not one
        contiguous block, or arities descend within a family's block.
    """
    table = fixture.table
    ordered = sorted(fixture.operators(table), key=lambda op: fixture.code(table, op))
    closed_sorts: set[str] = set()
    closed_blocks: set[tuple[str, str]] = set()
    previous: tuple[str, str, int] | None = None

    for op in ordered:
        sort, family = fixture.sort(table, op), fixture.family(table, op)
        arity = fixture.arity(table, op)

        if previous is not None and previous[0] != sort:
            closed_sorts.add(previous[0])

        if previous is not None and previous[:2] != (sort, family):
            closed_blocks.add(previous[:2])

        if sort in closed_sorts:
            detail = f"{op!r} returns to the sort {sort!r} after another sort's codes"

            fail_law(detail, law="segregation")

        if (sort, family) in closed_blocks:
            detail = (
                f"{op!r} returns to the family {family!r} of sort {sort!r} after "
                "another family's codes"
            )

            fail_law(detail, law="segregation")

        if (
            previous is not None
            and previous[:2] == (sort, family)
            and (arity < previous[2])
        ):
            detail = (
                f"{op!r} has arity {arity} after arity {previous[2]} within the "
                f"family {family!r} of sort {sort!r}"
            )

            fail_law(detail, law="segregation")

        previous = (sort, family, arity)


def codes_are_determined_by_the_declaration[A, O](
    fixture: AlphabetFixture[A, O],
) -> None:
    """Check that one declaration always gives one code table.

    Parameters
    ----------
    fixture : AlphabetFixture[A, O]
        The alphabet fixture.

    Raises
    ------
    LawViolationError
        If rebuilding changes operators, codes, or the digest.
    """
    table = fixture.table
    again = fixture.rebuild()
    here = fixture.operators(table)
    there = fixture.operators(again)

    if set(here) != set(there):
        detail = f"one declaration names {here!r} and then {there!r}"

        fail_law(detail, law="determinism")

    for op in here:
        first, second = fixture.code(table, op), fixture.code(again, op)

        if first == second:
            continue

        detail = f"{op!r} is coded {first} once and {second} again"

        fail_law(detail, law="determinism")

    if fixture.digest(table) != fixture.digest(again):
        detail = (
            f"one declaration digests to {fixture.digest(table)!r} and then to "
            f"{fixture.digest(again)!r}"
        )

        fail_law(detail, law="determinism")


SIGNATURE_INVARIANTS: Final = (
    padding_belongs_to_no_operator,
    codes_are_a_bijection,
    codes_group_by_sort_then_family,
    codes_are_determined_by_the_declaration,
)
"""Signature invariants in conformance order."""


def check_signature_invariants[A, O](fixture: AlphabetFixture[A, O]) -> None:
    """Run every signature invariant against one operator table.

    Parameters
    ----------
    fixture : AlphabetFixture[A, O]
        The alphabet fixture.

    Raises
    ------
    LawViolationError
        If the table violates an invariant.
    """
    for law in SIGNATURE_INVARIANTS:
        law(fixture)


class BindingFixture[T, A](NamedTuple):
    """Inputs for binding-law checks.

    Attributes
    ----------
    binders : SearchStrategy[T]
        A strategy over binder terms.
    open_binder : Callable[[T], tuple[T, tuple[A, ...]]]
        A function opening a binder with fresh atoms.
    close_binder : Callable[[T, tuple[A, ...]], T]
        A function closing a body over atoms.
    rename : Callable[[T, tuple[A, ...], tuple[A, ...]], T]
        A function renaming atoms in a term.
    fresh : Callable[[], A]
        A function returning a fresh atom.
    variable : Callable[[A], T]
        A function returning a variable term.
    abstract : Callable[[T, A], T]
        A function abstracting an atom.
    instantiate : Callable[[T, T], T]
        A function instantiating an abstraction.
    open_terms : SearchStrategy[T]
        A strategy over terms with open indices.
    """

    binders: SearchStrategy[T]
    open_binder: Callable[[T], tuple[T, tuple[A, ...]]]
    close_binder: Callable[[T, tuple[A, ...]], T]
    rename: Callable[[T, tuple[A, ...], tuple[A, ...]], T]
    fresh: Callable[[], A]
    variable: Callable[[A], T]
    abstract: Callable[[T, A], T]
    instantiate: Callable[[T, T], T]
    open_terms: SearchStrategy[T]


class TermFixture[T, O, A](NamedTuple):
    """Inputs for term-law checks.

    Attributes
    ----------
    terms : SearchStrategy[T]
        A strategy over terms.
    root : Callable[[T], O]
        A function returning a term's root operator.
    arguments : Callable[[T], tuple[T, ...]]
        A function returning a term's arguments.
    rebuild : Callable[[T, tuple[T, ...]], T]
        A function rebuilding a term over arguments.
    core : Callable[[T], T]
        A function returning the erased core.
    size : Callable[[T], int]
        A function returning the node count.
    open_indices : Callable[[T], Mapping[str, int]]
        A function returning, per bound sort, one past the greatest open
        bound index of that sort, omitting a sort with none open.
    free : Callable[[T], int]
        A function returning the free variable mask.
    shift : Callable[[T, int], Mapping[str, int]]
        A function returning, per bound sort, how many binding sites of the
        root scope over one argument position.
    is_closed : Callable[[T], bool]
        A function testing local closure.
    binding : BindingFixture[T, A] | None
        Optional binding-law operations.
    adopt : Callable[[], T] | None
        An optional foreign-term construction.
    transplant : Callable[[T], T] | None
        An optional reconstruction of a term in a second store, built from
        the term's constant and variable names in a different interning
        order.
    max_examples : int
        The maximum generated examples per law.
    """

    terms: SearchStrategy[T]
    root: Callable[[T], O]
    arguments: Callable[[T], tuple[T, ...]]
    rebuild: Callable[[T, tuple[T, ...]], T]
    core: Callable[[T], T]
    size: Callable[[T], int]
    open_indices: Callable[[T], Mapping[str, int]]
    free: Callable[[T], int]
    shift: Callable[[T, int], Mapping[str, int]]
    is_closed: Callable[[T], bool]
    binding: BindingFixture[T, A] | None = None
    adopt: Callable[[], T] | None = None
    transplant: Callable[[T], T] | None = None
    max_examples: int = 200


def a_foreign_argument_is_refused[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that construction refuses a foreign argument.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If a foreign argument is accepted.
    """
    if fixture.adopt is None:
        return

    try:
        adopted = fixture.adopt()

    except ForeignTermError:
        return

    detail = f"an argument interned outside the store's ancestry built {adopted!r}"

    fail_law(detail, law="foreign argument")


def structure_is_identity[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that identical structure produces one node.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If rebuilding a generated term changes its identity.
    """

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        again = fixture.rebuild(term, fixture.arguments(term))

        if again is not term:
            detail = f"{term!r} rebuilt from its own parts is {again!r}"

            fail_law(detail, law="interning")

    law()


def structure_is_store_independent[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that a term rebuilt in a second store equals it and hashes alike.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If a term and its reconstruction from names in a second store are
        unequal or hash apart.
    """
    transplant = fixture.transplant

    if transplant is None:
        return

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        there = transplant(term)

        if there != term or hash(there) != hash(term):
            detail = (
                f"{term!r} rebuilt in a second store is {there!r}, which is not "
                "equal to it with an equal hash"
            )

            fail_law(detail, law="store independence")

    law()


def hashing_is_stable[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that term hashes remain stable.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If repeated or rebuilt hashes differ.
    """

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        once = hash(term)
        twice = hash(term)

        if twice != once:
            detail = f"{term!r} hashes to {once} and then to {twice}"

            fail_law(detail, law="hashing")

        again = hash(fixture.rebuild(term, fixture.arguments(term)))

        if again != once:
            detail = (
                f"{term!r} hashes to {once}, and the node rebuilt from its parts "
                f"to {again}"
            )

            fail_law(detail, law="hashing")

    law()


def erasure_is_idempotent[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that term erasure is idempotent.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If erasing a generated core changes its identity.
    """

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        once = fixture.core(term)
        twice = fixture.core(once)

        if twice is not once:
            detail = f"{term!r} erases to {once!r} and then to {twice!r}"

            fail_law(detail, law="erasure idempotence")

    law()


def erasure_is_a_congruence[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that erasure is a term congruence.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If erasure disagrees across roots or arguments.
    """

    @given(fixture.terms, fixture.terms)
    @_budget(fixture.max_examples)
    def law(left: T, right: T) -> None:
        erased = fixture.core(left)

        if fixture.root(left) == fixture.root(right):
            here, there = fixture.root(erased), fixture.root(fixture.core(right))

            if here != there:
                detail = (
                    f"{left!r} and {right!r} share a root operator, and erase to "
                    f"the root operators {here!r} and {there!r}"
                )

                fail_law(detail, law="erasure congruence")

        arguments = fixture.arguments(left)

        if not arguments:
            return

        rebuilt = fixture.rebuild(erased, tuple(fixture.core(k) for k in arguments))

        if rebuilt is erased:
            return

        detail = (
            f"{left!r} erases to {erased!r}, and its erased root over its erased "
            f"arguments is {rebuilt!r}"
        )

        fail_law(detail, law="erasure congruence")

    law()


def derived_fields_agree_with_their_definitions[T, O, A](
    fixture: TermFixture[T, O, A],
) -> None:
    """Check cached term fields against their definitions.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If a generated field disagrees with the term structure.
    """

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        arguments = fixture.arguments(term)
        counted = 1 + sum(fixture.size(k) for k in arguments)

        if fixture.size(term) != counted:
            detail = f"{term!r} holds {fixture.size(term)} nodes, and covers {counted}"

            fail_law(detail, law="derived fields")

        if fixture.is_closed(term) != (not any(fixture.open_indices(term).values())):
            detail = (
                f"{term!r} leaves {dict(fixture.open_indices(term))} indices open, "
                f"and reports closedness {fixture.is_closed(term)}"
            )

            fail_law(detail, law="derived fields")

        if not arguments:
            return

        _summaries_agree(fixture, term, arguments)

    law()


def _summaries_agree[T, O, A](
    fixture: TermFixture[T, O, A], term: T, arguments: tuple[T, ...]
) -> None:
    """Check a node's index and free variable summaries.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.
    term : T
        The term under test.
    arguments : tuple[T, ...]
        The term's arguments.

    Raises
    ------
    LawViolationError
        If a summary disagrees with the arguments.
    """
    expected: dict[str, int] = {}

    for position, argument in enumerate(arguments):
        shift = fixture.shift(term, position)

        for sort, bound in fixture.open_indices(argument).items():
            left = bound - shift.get(sort, 0)

            if left > expected.get(sort, 0):
                expected[sort] = left

    held = {sort: bound for sort, bound in fixture.open_indices(term).items() if bound}

    if held != expected:
        detail = (
            f"{term!r} leaves {held} indices open, and its arguments under its "
            f"binding sites leave {expected}"
        )

        fail_law(detail, law="derived fields")

    mask = 0

    for argument in arguments:
        mask |= fixture.free(argument)

    if fixture.free(term) != mask:
        detail = (
            f"{term!r} names the free variables {fixture.free(term):b}, and its "
            f"arguments name {mask:b}"
        )

        fail_law(detail, law="derived fields")


def alpha_variants_are_one_node[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that alpha variants share node identity.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If renaming bound atoms changes a generated node.
    """
    binding = fixture.binding

    if binding is None:
        return

    make_atom = binding.fresh

    @given(binding.binders)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        body, atoms = binding.open_binder(term)
        fresh = tuple(make_atom() for _ in atoms)
        renamed = binding.rename(body, atoms, fresh)
        back = binding.close_binder(renamed, fresh)

        if back is not term:
            detail = f"{term!r} reopened under fresh atoms closes to {back!r}"

            fail_law(detail, law="alpha")

    law()


def closing_undoes_opening[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that closing a binder undoes opening it.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If closing a generated opened binder changes it.
    """
    binding = fixture.binding

    if binding is None:
        return

    @given(binding.binders)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        body, atoms = binding.open_binder(term)
        back = binding.close_binder(body, atoms)

        if back is not term:
            detail = f"{term!r} opens to {body!r} and closes to {back!r}"

            fail_law(detail, law="open and close")

    law()


def substitution_cancels_abstraction[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Check that substitution cancels abstraction.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If cancellation fails or generated cases do not exercise it.
    """
    binding = fixture.binding

    if binding is None:
        return

    canceled: list[T] = []

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def cancellation(body: T) -> None:
        if not fixture.is_closed(body):
            return

        atom = binding.fresh()
        abstracted = binding.abstract(body, atom)
        back = binding.instantiate(abstracted, binding.variable(atom))
        canceled.append(body)

        if back is not body:
            detail = f"{body!r} abstracted and instantiated again is {back!r}"

            fail_law(detail, law="substitution")

    cancellation()

    if not canceled:
        detail = (
            "every generated term left an index open, so the cancellation was "
            "never exercised"
        )

        fail_law(detail, law="substitution")

    _substitution_guards(fixture, binding)


def _substitution_guards[T, O, A](
    fixture: TermFixture[T, O, A], binding: BindingFixture[T, A]
) -> None:
    """Check that substitution refuses terms that are not locally closed.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.
    binding : BindingFixture[T, A]
        The binding-law operations.

    Raises
    ------
    LawViolationError
        If an open term is accepted or no generated case exercises the guard.
    """
    refused: list[T] = []

    @given(fixture.terms, binding.open_terms)
    @_budget(fixture.max_examples)
    def guard(body: T, value: T) -> None:
        if fixture.is_closed(value):
            return

        abstracted = binding.abstract(body, binding.fresh())

        try:
            substituted = binding.instantiate(abstracted, value)

        except NotLocallyClosedError:
            refused.append(value)

            return

        detail = (
            f"{value!r} leaves {dict(fixture.open_indices(value))} indices open, "
            f"and substituting it into {abstracted!r} gave {substituted!r}"
        )

        fail_law(detail, law="substitution")

    guard()

    if not refused:
        detail = (
            "every generated term was locally closed, so the substitution guard "
            "was never exercised"
        )

        fail_law(detail, law="substitution")


TERM_LAWS: Final = (
    a_foreign_argument_is_refused,
    structure_is_identity,
    structure_is_store_independent,
    hashing_is_stable,
    erasure_is_idempotent,
    erasure_is_a_congruence,
    derived_fields_agree_with_their_definitions,
    alpha_variants_are_one_node,
    closing_undoes_opening,
    substitution_cancels_abstraction,
)
"""Term laws in conformance order."""


def check_term_laws[T, O, A](fixture: TermFixture[T, O, A]) -> None:
    """Run every term law against one algebra.

    Parameters
    ----------
    fixture : TermFixture[T, O, A]
        The term fixture.

    Raises
    ------
    LawViolationError
        If the algebra violates a term law.
    """
    for law in TERM_LAWS:
        law(fixture)


class MatchingFixture[T](NamedTuple):
    """Inputs for matching-law checks.

    Attributes
    ----------
    matching : Matching[T]
        The congruence under test.
    terms : SearchStrategy[T]
        A strategy over terms.
    contexts : SearchStrategy[Context[T]]
        A strategy over one-hole contexts.
    max_examples : int
        The maximum generated examples per law.
    """

    matching: Matching[T]
    terms: SearchStrategy[T]
    contexts: SearchStrategy[Context[T]]
    max_examples: int = 200


def _pool[T](matching: Matching[T], drawn: list[T]) -> list[T]:
    """Return drawn terms followed by their representatives.

    Parameters
    ----------
    matching : Matching[T]
        The congruence under test.
    drawn : list[T]
        The sampled terms.
    """
    if not matching.has_key:
        return list(drawn)

    return [*drawn, *(matching.key(term) for term in drawn)]


def equivalence_is_an_equivalence[T](fixture: MatchingFixture[T]) -> None:
    """Check that the relation is an equivalence.

    Parameters
    ----------
    fixture : MatchingFixture[T]
        The matching fixture.

    Raises
    ------
    LawViolationError
        If reflexivity, symmetry, or transitivity fails.
    """
    matching = fixture.matching
    name = matching.name

    @given(st.lists(fixture.terms, min_size=_POOL, max_size=_POOL))
    @_budget(fixture.max_examples)
    def law(drawn: list[T]) -> None:
        pool = _pool(matching, drawn)

        for left in pool:
            if not matching.equivalent(left, left):
                fail_law(
                    f"{name} makes {left!r} inequivalent to itself", law="equivalence"
                )

            for right in pool:
                if matching.equivalent(left, right) != matching.equivalent(right, left):
                    detail = f"{name} relates {left!r} to {right!r} in one direction"

                    fail_law(detail, law="equivalence")

                if not matching.equivalent(left, right):
                    continue

                for third in pool:
                    if matching.equivalent(right, third) and not matching.equivalent(
                        left, third
                    ):
                        detail = (
                            f"{name} relates {left!r} to {right!r} and {right!r} "
                            f"to {third!r}, and not {left!r} to {third!r}"
                        )

                        fail_law(detail, law="equivalence")

    law()


def key_is_a_section[T](fixture: MatchingFixture[T]) -> None:
    """Check that representative identity characterizes equivalence.

    Parameters
    ----------
    fixture : MatchingFixture[T]
        The matching fixture.

    Raises
    ------
    LawViolationError
        If representative identity and equivalence disagree.
    """
    matching = fixture.matching

    if not matching.has_key:
        return

    name = matching.name

    @given(st.lists(fixture.terms, min_size=_POOL, max_size=_POOL))
    @_budget(fixture.max_examples)
    def law(drawn: list[T]) -> None:
        pool = _pool(matching, drawn)

        for left in pool:
            for right in pool:
                shared = matching.key(left) is matching.key(right)

                if shared == matching.equivalent(left, right):
                    continue

                detail = (
                    f"{name} relates {left!r} to {right!r} "
                    f"{matching.equivalent(left, right)}, and gives them "
                    f"{'one' if shared else 'two'} representatives"
                )

                fail_law(detail, law="section")

    law()


def equivalence_survives_construction[T](fixture: MatchingFixture[T]) -> None:
    """Check that construction preserves equivalence.

    Parameters
    ----------
    fixture : MatchingFixture[T]
        The matching fixture.

    Raises
    ------
    LawViolationError
        If a generated context separates equivalent terms.
    """
    matching = fixture.matching
    name = matching.name

    @given(fixture.terms, fixture.terms, fixture.contexts)
    @_budget(fixture.max_examples)
    def law(left: T, right: T, context: Context[T]) -> None:
        related = [(left, matching.key(left))] if matching.has_key else []

        if matching.equivalent(left, right):
            related.append((left, right))

        for one, other in related:
            if not matching.equivalent(context(one), context(other)):
                detail = (
                    f"{name} relates {one!r} to {other!r}, and not "
                    f"{context(one)!r} to {context(other)!r}"
                )

                fail_law(detail, law="congruence")

    law()


def equivalence_coarsens_identity[T](fixture: MatchingFixture[T]) -> None:
    """Check that equivalence contains identity and its representatives.

    Parameters
    ----------
    fixture : MatchingFixture[T]
        The matching fixture.

    Raises
    ------
    LawViolationError
        If a term is not equivalent to itself or its representative.
    """
    matching = fixture.matching
    name = matching.name

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not matching.equivalent(term, term):
            fail_law(f"{name} makes {term!r} inequivalent to itself", law="coarsening")

        if matching.has_key and not matching.equivalent(matching.key(term), term):
            detail = (
                f"{name} stands {matching.key(term)!r} in for {term!r}, "
                f"and relates neither to the other"
            )

            fail_law(detail, law="coarsening")

    law()


def key_is_idempotent[T](fixture: MatchingFixture[T]) -> None:
    """Check that representative selection is idempotent.

    Parameters
    ----------
    fixture : MatchingFixture[T]
        The matching fixture.

    Raises
    ------
    LawViolationError
        If selecting a representative twice changes its identity.
    """
    matching = fixture.matching

    if not matching.has_key:
        return

    name = matching.name

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        once = matching.key(term)
        twice = matching.key(once)

        if twice is not once:
            detail = f"{name} keys {term!r} to {once!r} and then to {twice!r}"

            fail_law(detail, law="idempotence")

    law()


MATCHING_LAWS: Final = (
    equivalence_is_an_equivalence,
    key_is_a_section,
    equivalence_survives_construction,
    equivalence_coarsens_identity,
    key_is_idempotent,
)
"""Matching laws in conformance order."""


def check_matching_laws[T](fixture: MatchingFixture[T]) -> None:
    """Run every matching law against one relation.

    Parameters
    ----------
    fixture : MatchingFixture[T]
        The matching fixture.

    Raises
    ------
    LawViolationError
        If the relation violates a matching law.
    """
    for law in MATCHING_LAWS:
        law(fixture)


class LayoutFixture[T, S, I](NamedTuple):
    """Inputs for layout-law checks.

    Attributes
    ----------
    layout : Layout[T, S, I]
        The term layout under test.
    terms : SearchStrategy[T]
        A strategy over terms.
    store : S
        The decoder store.
    index : I
        The operator index.
    size : Callable[[T], int]
        A function returning a term's node count.
    fits : Callable[[T], bool]
        A function testing whether a term fits the layout.
    max_examples : int
        The maximum generated examples per law.
    """

    layout: Layout[T, S, I]
    terms: SearchStrategy[T]
    store: S
    index: I
    size: Callable[[T], int]
    fits: Callable[[T], bool]
    max_examples: int = 200


def _encode[T, S, I](fixture: LayoutFixture[T, S, I], term: T, *, law: str) -> Encoding:
    """Encode a term and report unexpected rejection.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.
    term : T
        The term to encode.
    law : str
        The calling law's name.

    Raises
    ------
    LawViolationError
        If an expected-to-fit term overflows the layout.
    """
    layout = fixture.layout

    try:
        return layout.encode(term, fixture.index)

    except EncodingOverflowError as exc:
        message = (
            f"{law}: {layout.name} refused {term!r}, of {fixture.size(term)} "
            f"nodes, though it holds {layout.capacity} slots"
        )

        raise LawViolationError(message) from exc


def round_trip_is_identity[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Check that decoding an encoding preserves term identity.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If a generated round trip changes identity or is not exercised.
    """
    layout = fixture.layout
    tried: list[T] = []

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not fixture.fits(term):
            return

        encoding = _encode(fixture, term, law="round trip")
        back = layout.decode(encoding, fixture.store, fixture.index)
        tried.append(term)

        if back is not term:
            detail = f"{layout.name} encodes {term!r} and decodes {back!r}"

            fail_law(detail, law="round trip")

    law()

    if not tried:
        detail = (
            f"every generated term was too big for {layout.name}, so the round "
            f"trip was never exercised"
        )

        fail_law(detail, law="round trip")


def encoding_refuses_to_truncate[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Check that encoding refuses to truncate oversized terms.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If capacity or overflow behavior is incorrect or untested.
    """
    layout = fixture.layout
    oversized: list[T] = []

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not fixture.fits(term):
            try:
                layout.encode(term, fixture.index)

            except EncodingOverflowError:
                oversized.append(term)

                return

            detail = (
                f"{layout.name} holds {layout.capacity} slots and encoded "
                f"{term!r}, of {fixture.size(term)} nodes, without complaint"
            )

            fail_law(detail, law="no truncation")

        encoding = _encode(fixture, term, law="no truncation")

        if encoding.width != layout.capacity:
            detail = (
                f"{layout.name} declares {layout.capacity} slots and encoded "
                f"{term!r} into {encoding.width}"
            )

            fail_law(detail, law="no truncation")

    law()

    if not oversized:
        detail = (
            f"no generated term was too big for {layout.name}, so refusing to "
            f"truncate was never exercised"
        )

        fail_law(detail, law="no truncation")


def mask_is_sound[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Check that occupied slots form a valid rooted tree.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If a generated occupancy mask is invalid or untested.
    """
    layout = fixture.layout
    tried: list[T] = []

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not fixture.fits(term):
            return

        encoding = _encode(fixture, term, law="mask soundness")
        tried.append(term)
        roots = [
            slot
            for slot in range(encoding.width)
            if encoding.occupied[slot] and encoding.parents[slot] == Encoding.NO_PARENT
        ]

        if len(roots) != 1:
            detail = f"{layout.name} encodes {term!r} with {len(roots)} roots"

            fail_law(detail, law="mask soundness")

        if encoding.depths[roots[0]]:
            detail = (
                f"{layout.name} puts the root of {term!r} at depth "
                f"{encoding.depths[roots[0]]}"
            )

            fail_law(detail, law="mask soundness")

        _parents_are_occupied(encoding, layout.name, term, root=roots[0])

    law()

    if not tried:
        detail = (
            f"every generated term was too big for {layout.name}, so mask "
            f"soundness was never exercised"
        )

        fail_law(detail, law="mask soundness")


def _slot_phrase[T](name: str, term: T, slot: int) -> str:
    """Return a phrase naming one encoding slot.

    Parameters
    ----------
    name : str
        The layout name.
    term : T
        The encoded term.
    slot : int
        The slot index.
    """
    return f"slot {slot} of {name}'s encoding of {term!r}"


def _parents_are_occupied[T](
    encoding: Encoding, name: str, term: T, *, root: int
) -> None:
    """Check that each occupied nonroot slot has an occupied parent.

    Parameters
    ----------
    encoding : Encoding
        The encoding under test.
    name : str
        The layout name.
    term : T
        The encoded term.
    root : int
        The root slot.

    Raises
    ------
    LawViolationError
        If a parent or depth relation is invalid.
    """
    for slot in range(encoding.width):
        if not encoding.occupied[slot] or slot == root:
            continue

        parent = encoding.parents[slot]

        if not 0 <= parent < encoding.width or not encoding.occupied[parent]:
            detail = (
                f"{_slot_phrase(name, term, slot)} hangs off slot {parent}, "
                f"which holds no node"
            )

            fail_law(detail, law="mask soundness")

        if encoding.depths[slot] != encoding.depths[parent] + 1:
            detail = (
                f"{_slot_phrase(name, term, slot)} is at depth "
                f"{encoding.depths[slot]} under a parent at depth "
                f"{encoding.depths[parent]}"
            )

            fail_law(detail, law="mask soundness")


def mask_is_complete[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Check that occupied slots match the term's node count.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If a generated mask has the wrong size or is untested.
    """
    layout = fixture.layout
    tried: list[T] = []

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not fixture.fits(term):
            return

        size = fixture.size(term)

        encoding = _encode(fixture, term, law="mask completeness")
        tried.append(term)

        if encoding.length != size:
            detail = (
                f"{layout.name} encodes {term!r}, of {size} nodes, into "
                f"{encoding.length} occupied slots"
            )

            fail_law(detail, law="mask completeness")

    law()

    if not tried:
        detail = (
            f"every generated term was too big for {layout.name}, so mask "
            f"completeness was never exercised"
        )

        fail_law(detail, law="mask completeness")


def path_addressing_is_honest[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Check a layout's declared path addressing.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If a slot has inconsistent structure or no case is exercised.
    """
    layout = fixture.layout

    if not layout.path_addressed:
        return

    seen: dict[int, tuple[int, int, T]] = {}

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not fixture.fits(term):
            return

        encoding = _encode(fixture, term, law="path addressing")

        for slot in range(encoding.width):
            if not encoding.occupied[slot]:
                continue

            here = (encoding.depths[slot], encoding.parents[slot], term)
            there = seen.setdefault(slot, here)

            if here[:2] == there[:2]:
                continue

            detail = (
                f"{_slot_phrase(layout.name, term, slot)} is at depth "
                f"{here[0]} under slot {here[1]}, and "
                f"{_slot_phrase(layout.name, there[2], slot)} at depth "
                f"{there[0]} under slot {there[1]}"
            )

            fail_law(detail, law="path addressing")

    law()

    if not seen:
        detail = (
            f"every generated term was too big for {layout.name}, so path "
            f"addressing was never exercised"
        )

        fail_law(detail, law="path addressing")


def legal_next_admits_the_next_symbol[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Check that each encoded symbol is a legal continuation.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If a symbol is rejected or no generated case is exercised.
    """
    layout = fixture.layout
    tried: list[T] = []

    @given(fixture.terms)
    @_budget(fixture.max_examples)
    def law(term: T) -> None:
        if not fixture.fits(term):
            return

        encoding = _encode(fixture, term, law="legal continuation")
        tried.append(term)

        for at in range(encoding.width):
            allowed = layout.legal_next(encoding.symbols[:at], fixture.index)
            code = encoding.symbols[at]

            if code < len(allowed) and allowed[code]:
                continue

            detail = (
                f"{layout.name} encodes {term!r} with symbol {code} at slot "
                f"{at}, and admits {len(allowed)} symbols there, not that one"
            )

            fail_law(detail, law="legal continuation")

    law()

    if not tried:
        detail = (
            f"every generated term was too big for {layout.name}, so the legal "
            f"continuations were never exercised"
        )

        fail_law(detail, law="legal continuation")


LAYOUT_LAWS: Final = (
    round_trip_is_identity,
    encoding_refuses_to_truncate,
    mask_is_sound,
    mask_is_complete,
    path_addressing_is_honest,
    legal_next_admits_the_next_symbol,
)
"""Layout laws in conformance order."""


def check_layout_laws[T, S, I](fixture: LayoutFixture[T, S, I]) -> None:
    """Run every layout law against one layout.

    Parameters
    ----------
    fixture : LayoutFixture[T, S, I]
        The layout fixture.

    Raises
    ------
    LawViolationError
        If the layout violates a law.
    """
    for law in LAYOUT_LAWS:
        law(fixture)


class PresentationFixture[S, C](NamedTuple):
    """Inputs for presentation-law checks.

    Attributes
    ----------
    subject : Reifiable[S]
        The reifiable object under test, which is also the root of the built
        graph the reachability law walks.
    rebuild : Callable[[S], Reifiable[S]]
        A function rebuilding an object from its specification.
    components : tuple[C, ...]
        References expected to share identity.
    refused : tuple[type, ...]
        Classes the built graph must not reach, beside the substrate's own:
        a lowerer, a checked intermediate, or any other boundary transient.
    """

    subject: Reifiable[S]
    rebuild: Callable[[S], Reifiable[S]]
    components: tuple[C, ...] = ()
    refused: tuple[type, ...] = ()


def spec_round_trip[S, C](fixture: PresentationFixture[S, C]) -> None:
    """Check that a specification rebuilds an equivalent specification.

    Parameters
    ----------
    fixture : PresentationFixture[S, C]
        The presentation fixture.

    Raises
    ------
    LawViolationError
        If specification output changes or fails to round-trip.
    """
    subject = fixture.subject
    spec = subject.to_spec()
    again = subject.to_spec()

    if again != spec:
        detail = f"{subject!r} declares {spec!r}, and then {again!r}"

        fail_law(detail, law="specification round trip")

    rebuilt = fixture.rebuild(spec).to_spec()

    if rebuilt != spec:
        detail = f"{subject!r} declares {spec!r}, and its rebuild {rebuilt!r}"

        fail_law(detail, law="specification round trip")


def one_component_by_identity[S, C](fixture: PresentationFixture[S, C]) -> None:
    """Check that component references share identity.

    Parameters
    ----------
    fixture : PresentationFixture[S, C]
        The presentation fixture.

    Raises
    ------
    LawViolationError
        If component references name different objects.
    """
    components = fixture.components

    if not components:
        return

    first = components[0]

    for other in components[1:]:
        if other is first:
            continue

        detail = f"a built graph names {first!r} and {other!r}, which are two objects"

        fail_law(detail, law="one component")


def nothing_refused_is_reachable[S, C](fixture: PresentationFixture[S, C]) -> None:
    """Check that the built graph reaches no substrate or refused object.

    Parameters
    ----------
    fixture : PresentationFixture[S, C]
        The presentation fixture.

    Raises
    ------
    LawViolationError
        If the subject reaches, at any depth, an object whose class is
        defined by didactic or panproto or is one the fixture refuses.
    """
    reached = substrate_reaches(fixture.subject, refused=fixture.refused)

    if not reached:
        return

    named = ", ".join(
        f"{reach.held.__module__}.{reach.held.__qualname__}"
        f"{'' if reach.instance else ' itself'} at {reach.path or '<root>'}"
        for reach in reached
    )
    detail = f"{fixture.subject!r} reaches {named}, which is refused there"

    fail_law(detail, law="reachability")


PRESENTATION_LAWS: Final = (
    spec_round_trip,
    one_component_by_identity,
    nothing_refused_is_reachable,
)
"""Presentation laws in conformance order."""


def check_presentation_laws[S, C](fixture: PresentationFixture[S, C]) -> None:
    """Run every presentation law against one built object.

    Parameters
    ----------
    fixture : PresentationFixture[S, C]
        The presentation fixture.

    Raises
    ------
    LawViolationError
        If the object violates a presentation law.
    """
    for law in PRESENTATION_LAWS:
        law(fixture)

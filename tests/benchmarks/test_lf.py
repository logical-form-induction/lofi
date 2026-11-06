"""Logical-form term-operation benchmarks.

Hashing is timed with the harness loop subtracted, since one empty call
through ``pytest-benchmark`` costs more than the whole hash budget. The beta
step is timed through the fixture directly, where the wrapper is a small
fraction of the operation.
"""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.benchmarks.test_terms import _per_operation
from tests.language import language

from lofi.terms import NotLocallyClosedError, Term, TermStore, instantiate

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

#: The constants the measured forms name.
VOCABULARY: Final = (
    "actor",
    "agent",
    "delivery",
    "item",
    "move",
    "theme",
    *(f"p{index}" for index in range(600)),
)

LANGUAGE: Final = language(VOCABULARY)

#: Nanoseconds one guarded beta step may take.
BETA_STEP_NS: Final = 40_000

#: Nanoseconds one logical-form hash diagnostic may take.
HASH_DIAGNOSTIC_NS: Final = 80

#: The two binder names the one scope is built over.
VARIABLES: Final = ("x", "y")


class Forms(NamedTuple):
    """The logical forms every benchmark in this module reads.

    Attributes
    ----------
    store : TermStore
        The interner the forms were drawn from.
    abstraction : Term
        ``lam x. move(x) and agent(x, actor) and theme(x, item)``, built over
        the variable named ``x``.
    renamed : Term
        The same abstraction built over a variable named ``y``, which
        locally nameless abstraction makes the very same node.
    body : Term
        The abstraction's body, whose bound index a beta step replaces.
    value : Term
        The closed event the beta step substitutes in.
    open_value : Term
        A bare bound index, which the locally-closed guard refuses.
    large : Term
        A conjunction of six hundred distinct atoms, whose hash a stored
        field answers as quickly as the small form's.
    """

    store: TermStore
    abstraction: Term
    renamed: Term
    body: Term
    value: Term
    open_value: Term
    large: Term


def _chain(store: TermStore, width: int) -> Term:
    """Build a right-branching conjunction of distinct atoms.

    Parameters
    ----------
    store : TermStore
        The interner the conjunction is built in.
    width : int
        How many atoms the conjunction holds.
    """
    form = store.const(LANGUAGE.term_sort, "p0")

    for index in range(1, width):
        form = LANGUAGE.apply(
            LANGUAGE.constant("and"),
            store.const(LANGUAGE.term_sort, f"p{index}"),
            form,
        )

    return form


def _scope(store: TermStore, variable: str) -> Term:
    """Build one three-conjunct neo-Davidsonian scope over a named variable."""
    application = store.signature.by_name(LANGUAGE.calculus.app)
    event = store.fvar(LANGUAGE.term_sort, variable)

    def apply(*terms: Term) -> Term:
        head, *rest = terms

        for term in rest:
            head = store.op(application, head, term)

        return head

    return store.bind(
        LANGUAGE.calculus.lam,
        variable,
        LANGUAGE.apply(
            LANGUAGE.constant("and"),
            apply(store.const(LANGUAGE.term_sort, "move"), event),
            LANGUAGE.apply(
                LANGUAGE.constant("and"),
                apply(
                    store.const(LANGUAGE.term_sort, "agent"),
                    event,
                    store.const(LANGUAGE.term_sort, "actor"),
                ),
                apply(
                    store.const(LANGUAGE.term_sort, "theme"),
                    event,
                    store.const(LANGUAGE.term_sort, "item"),
                ),
            ),
        ),
    )


@pytest.fixture(scope="module")
def forms() -> Forms:
    """Return the abstraction, its renaming, and the beta step's operands."""
    store = LANGUAGE.store
    named, other = VARIABLES
    abstraction = _scope(store, named)
    renamed = _scope(store, other)

    return Forms(
        store,
        abstraction,
        renamed,
        abstraction.args[0],
        store.const(LANGUAGE.term_sort, "delivery"),
        store.bvar(LANGUAGE.term_sort, 0),
        _chain(store, 600),
    )


def test_the_forms_measured_are_alpha_variants_of_one_scope(forms: Forms) -> None:
    named, other = VARIABLES
    assert named != other
    assert forms.store.fvar(LANGUAGE.term_sort, named) is not forms.store.fvar(
        LANGUAGE.term_sort, other
    )
    assert forms.abstraction.size == 20
    assert forms.body is forms.abstraction.args[0]
    assert not forms.body.is_closed
    assert forms.value.is_closed
    assert not forms.open_value.is_closed
    assert forms.large.size == 2_397


def test_alpha_equality_is_pointer_identity(forms: Forms) -> None:
    assert forms.renamed is forms.abstraction


def test_logical_form_hashing_is_within_budget(
    benchmark: BenchmarkFixture, forms: Forms, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, "hash(form)", {"form": forms.abstraction})
    benchmark.extra_info["diagnostic_ceiling_ns"] = HASH_DIAGNOSTIC_NS

    if wall_clock_budget.enforced:
        assert nanoseconds <= HASH_DIAGNOSTIC_NS


def test_hashing_a_large_form_stays_inside_the_same_budget(
    benchmark: BenchmarkFixture, forms: Forms, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, "hash(form)", {"form": forms.large})
    benchmark.extra_info["diagnostic_ceiling_ns"] = HASH_DIAGNOSTIC_NS
    benchmark.extra_info["nodes"] = forms.large.size

    if wall_clock_budget.enforced:
        assert nanoseconds <= HASH_DIAGNOSTIC_NS


def test_the_beta_step_refuses_an_open_value(forms: Forms) -> None:
    with pytest.raises(NotLocallyClosedError):
        instantiate(forms.store, forms.body, forms.open_value)


def test_one_guarded_beta_step_returns_a_distinct_closed_form(forms: Forms) -> None:
    reduced = instantiate(forms.store, forms.body, forms.value)

    assert reduced.is_closed
    assert reduced is not forms.body


def test_one_guarded_beta_step_is_within_budget(
    benchmark: BenchmarkFixture, forms: Forms, wall_clock_budget: WallClockBudget
) -> None:
    benchmark(instantiate, forms.store, forms.body, forms.value)

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the beta step ran but pytest-benchmark timed nothing")

    nanoseconds = measured.stats.median * 1e9
    benchmark.extra_info["nodes"] = forms.body.size
    benchmark.extra_info["median_ns_per_step"] = nanoseconds
    benchmark.extra_info["mean_ns_per_step"] = measured.stats.mean * 1e9
    benchmark.extra_info["budget_ns"] = BETA_STEP_NS

    if wall_clock_budget.enforced:
        assert nanoseconds <= BETA_STEP_NS

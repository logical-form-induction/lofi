"""Term-operation benchmarks: what one operation on an interned term costs."""

import gc
import tracemalloc
from timeit import Timer
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.support.term_reference import categories

from lofi.terms import Operator, Signature, Term, TermStore, Vocabulary

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

# repetitions and baseline rounds for nanosecond-scale timing.
REPEATS: Final = 20_000
ROUNDS: Final = 7

# distinct misses per timed batch.
BATCH: Final = 500

# distinct terms the timed set holds.
MEMBERS: Final = 1_000

# distinct nodes the traced root store interns.
INTERNED_NODES: Final = 20_000

# constant names the memory and membership fixtures draw from.
INDEXED_NAMES: Final = tuple(f"leaf-{payload}" for payload in range(INTERNED_NODES + 1))

# depth and node count of the term compared across two store lineages.
EQUALITY_DEPTH: Final = 5
EQUALITY_NODES: Final = 63

# the comparisons timed within and across stores.
EQUALITY: Final = "here == there"


# the names a timed statement reads, handed to `timeit` as its globals.
type Context = dict[str, object]


class Nodes(NamedTuple):
    """Benchmark terms and their interner.

    Attributes
    ----------
    store : TermStore
        The interner every node below was drawn from.
    branching : Term
        A binary node over a featured sentence and a noun phrase.
    primitive : Term
        The noun phrase, which is a second node for an identity comparison.
    featured : Term
        The featured sentence category, whose core is a different node.
    """

    store: TermStore
    branching: Term
    primitive: Term
    featured: Term


class Lineages(NamedTuple):
    """One structure built in two root stores.

    Attributes
    ----------
    here : Term
        The structure as the first store interned it.
    there : Term
        The same structure as the second store interned it, a distinct node.
    """

    here: Term
    there: Term


class Members(NamedTuple):
    """A set of interned terms, one it holds and one it does not.

    Attributes
    ----------
    members : frozenset[Term]
        The distinct terms the timed lookup searches.
    present : Term
        A term the set holds, whose lookup is the hit that is timed.
    absent : Term
        A term of the same shape the set does not hold.
    """

    members: frozenset[Term]
    present: Term
    absent: Term


class Interned(NamedTuple):
    """What a root store's interned nodes hold alive.

    Attributes
    ----------
    nodes : int
        How many nodes the store owns.
    held : int
        The live traced bytes those nodes added.
    """

    nodes: int
    held: int


def _indexed() -> Signature:
    """Return a signature over one sort with a bare atom and a pair.

    Kernel-derived constants mint arbitrarily many distinct leaves at a fixed
    shape. Repeated ``at`` applications over one shared ``top`` mint branching
    nodes without declaring a payload-bearing operator.

    Returns
    -------
    Signature
        One declared nullary operator and one declared binary operator. The
        vocabulary lets the store derive its constant operator.
    """
    return Signature(
        "indexed",
        ("Lf",),
        (
            Operator("top", "Lf"),
            Operator("at", "Lf", arg_sorts=("Lf", "Lf")),
        ),
        vocabularies=(Vocabulary("Lf", INDEXED_NAMES),),
    )


def _leaves(store: TermStore, count: int) -> None:
    """Intern distinct payload leaves, the cheapest node the store can hold.

    Parameters
    ----------
    store : TermStore
        The interner the nodes are published into.
    count : int
        How many distinct leaves to mint.
    """
    for payload in range(count):
        store.const("Lf", INDEXED_NAMES[payload])


def _pairs(store: TermStore, count: int) -> None:
    """Intern distinct binary nodes over one shared argument.

    Parameters
    ----------
    store : TermStore
        The interner the nodes are published into.
    count : int
        How many distinct binary nodes to mint. The shared argument is one
        further node the store holds.
    """
    signature = store.signature
    at = signature.by_name("at")
    shared = store.op(signature.by_name("top"))
    prior = shared

    for _ in range(count):
        prior = store.op(at, shared, prior)


def _traced(publish: Callable[[TermStore, int], None], count: int) -> Interned:
    """Return what one shape of interned node holds alive, per node.

    The nodes go into a root store nothing releases, so the figure is what the
    interner pays to keep a node: the node object, the key it is stored under
    and the table slot holding it. Nothing collects the nodes into a
    container, since a container of pointers would be traced beside them.

    Parameters
    ----------
    publish : Callable[[TermStore, int], None]
        Interns the requested count of distinct nodes into the store it is
        handed.
    count : int
        How many distinct nodes to intern.

    Returns
    -------
    Interned
        The store's node count and the live traced bytes it added.
    """
    store = TermStore(_indexed())
    gc.collect()
    tracemalloc.start()

    try:
        start = tracemalloc.get_traced_memory()[0]
        publish(store, count)
        held = tracemalloc.get_traced_memory()[0] - start
        owned = store.local_nodes

    finally:
        tracemalloc.stop()

    return Interned(owned, held)


def _node_shapes(count: int) -> tuple[Interned, Interned]:
    """Return what both node shapes hold, at one node count.

    Parameters
    ----------
    count : int
        How many distinct nodes each shape interns.

    Returns
    -------
    tuple[Interned, Interned]
        The payload leaves first, then the binary nodes.
    """
    return _traced(_leaves, count), _traced(_pairs, count)


def _miss_batch(signature: Signature, count: int) -> TermStore:
    """Return a store after interning one shared leaf and distinct pairs.

    Parameters
    ----------
    signature : Signature
        The declaration used by the fresh store.
    count : int
        How many distinct pairs to intern after the shared leaf.

    Returns
    -------
    TermStore
        The store holding the fresh nodes.
    """
    store: TermStore = TermStore(signature)
    shared = store.op(signature.by_name("top"))
    at = signature.by_name("at")
    prior = shared

    for _ in range(count):
        prior = store.op(at, shared, prior)

    return store


def _empty_batch(count: int) -> None:
    """Run the loop shape used around a batch of misses.

    Parameters
    ----------
    count : int
        How many loop iterations to execute.
    """
    for _ in range(count):
        pass


def _balanced(store: TermStore, depth: int, position: int = 0) -> Term:
    """Return a complete binary category of a given depth.

    Parameters
    ----------
    store : TermStore
        The interner the category is built in.
    depth : int
        How many levels of slashes lie above the noun-phrase leaves.
    position : int
        The node's index among the nodes at its depth. An even index takes a
        forward slash and an odd one a backward slash, so the two subtrees
        below any slash differ.

    Returns
    -------
    Term
        The category, with ``2 ** (depth + 1) - 1`` nodes.
    """
    signature = store.signature

    if depth == 0:
        return store.op(signature.by_name("np"))

    slash = signature.by_name("bwd" if position % 2 else "fwd")
    left = _balanced(store, depth - 1, 2 * position)
    right = _balanced(store, depth - 1, 2 * position + 1)

    return store.op(slash, left, right)


@pytest.fixture(scope="module")
def lineages() -> Lineages:
    """Return the committed equality fixture, built in two root stores.

    Returns
    -------
    Lineages
        A complete binary category of 63 nodes, twice.
    """
    here = TermStore(categories())
    there = TermStore(categories())

    return Lineages(_balanced(here, EQUALITY_DEPTH), _balanced(there, EQUALITY_DEPTH))


@pytest.fixture(scope="module")
def members() -> Members:
    """Return the term set the membership benchmark searches.

    Returns
    -------
    Members
        A thousand distinct terms, one of them, and one outside the set.
    """
    store = TermStore(_indexed())
    minted = [
        store.const("Lf", INDEXED_NAMES[payload]) for payload in range(MEMBERS + 1)
    ]

    return Members(frozenset(minted[:MEMBERS]), minted[MEMBERS // 2], minted[MEMBERS])


@pytest.fixture(scope="module")
def nodes() -> Nodes:
    """Return the terms every benchmark in this module reads.

    Returns
    -------
    Nodes
        A branching node, a primitive and a featured category.
    """
    store = TermStore(categories())
    signature = store.signature
    primitive = store.op(signature.by_name("np"))
    featured = store.op(signature.featured("s", "dcl"))
    branching = store.op(signature.by_name("fwd"), featured, primitive)

    return Nodes(store, branching, primitive, featured)


def _per_operation(
    benchmark: BenchmarkFixture, statement: str, context: Context
) -> float:
    """Return what one execution of a statement costs, in nanoseconds.

    Parameters
    ----------
    benchmark : BenchmarkFixture
        The fixture timing the loop.
    statement : str
        The operation, as the loop body it is compiled into.
    context : Context
        The terms the statement names.

    Returns
    -------
    float
        The nanoseconds one execution takes, the loop subtracted.
    """
    return _per_unit(benchmark, statement, "pass", context, units=1)


def _per_unit(
    benchmark: BenchmarkFixture,
    statement: str,
    empty: str,
    context: Context,
    *,
    units: int,
) -> float:
    """Return what one unit of work inside a statement costs, in nanoseconds.

    Parameters
    ----------
    benchmark : BenchmarkFixture
        The fixture timing the loop.
    statement : str
        The code timed, run as many times as it takes to perform ``REPEATS`` units.
    empty : str
        A statement with the same loop shape and no work, whose time is subtracted.
    context : Context
        The names the two statements read.
    units : int
        How many units of work one execution of the statement performs.

    Returns
    -------
    float
        The nanoseconds one unit takes, the loop subtracted.
    """
    executions = REPEATS // units
    busy = Timer(statement, globals=context)
    idle = Timer(empty, globals=context)
    benchmark(busy.timeit, executions)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the operation ran but pytest-benchmark timed nothing")

    # subtract the minimum empty-loop time from the measured median.
    overhead = min(idle.repeat(ROUNDS, executions))
    nanoseconds = (measured.stats.median - overhead) / REPEATS * 1e9
    benchmark.extra_info["repeats"] = REPEATS
    benchmark.extra_info["ns_per_operation"] = nanoseconds
    benchmark.extra_info["loop_overhead_ns"] = overhead / REPEATS * 1e9
    benchmark.extra_info["mean_ns_per_operation"] = (
        (measured.stats.mean - overhead) / REPEATS * 1e9
    )

    return nanoseconds


def test_the_nodes_measured_are_three_distinct_terms(nodes: Nodes) -> None:
    assert nodes.branching is not nodes.primitive
    assert nodes.branching.args[1] is nodes.primitive
    assert nodes.featured.core is not nodes.featured
    assert nodes.branching.core is not nodes.branching
    assert nodes.branching.core.args[0] is nodes.featured.core


def _within_one_lineage(nodes: Nodes) -> Context:
    """Return the two distinct nodes compared within one store.

    Parameters
    ----------
    nodes : Nodes
        The module's terms, all from one store.

    Returns
    -------
    Context
        The branching node as ``here`` and its second argument as ``there``.
    """
    return {"here": nodes.branching, "there": nodes.primitive}


def _across_lineages(lineages: Lineages) -> Context:
    """Return one structure from two stores for an equality comparison.

    Parameters
    ----------
    lineages : Lineages
        The equality fixture.

    Returns
    -------
    Context
        The first store's node as ``here`` and the second's as ``there``.
    """
    return {"here": lineages.here, "there": lineages.there}


def test_the_timed_comparison_is_structural_and_not_identity(
    lineages: Lineages,
) -> None:
    compared = _across_lineages(lineages)

    assert compared["here"] is not compared["there"]
    assert (compared["here"] == compared["there"]) is True


def test_the_nodes_compared_within_one_lineage_are_unequal(nodes: Nodes) -> None:
    compared = _within_one_lineage(nodes)
    swapped = {"here": compared["there"], "there": compared["here"]}

    assert compared["here"] is not compared["there"]
    assert (compared["here"] == compared["there"]) is False
    assert (swapped["here"] == swapped["there"]) is False


def test_term_equality_is_within_budget(
    benchmark: BenchmarkFixture, nodes: Nodes, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, EQUALITY, _within_one_lineage(nodes))
    benchmark.extra_info["budget"] = "term-equality"

    # enforce the 250 ns budget for equality within one store lineage
    if wall_clock_budget.enforced:
        assert nanoseconds <= 250


def test_the_terms_compared_across_lineages_are_one_structure(
    lineages: Lineages,
) -> None:
    compared = _across_lineages(lineages)
    here, there = compared["here"], compared["there"]

    assert here is not there
    assert lineages.here.size == EQUALITY_NODES
    assert lineages.here.args[0] != lineages.here.args[1]
    assert (here == there) is True
    assert hash(here) == hash(there)


def test_the_miss_fixture_executes_only_fresh_constructions() -> None:
    count = 17
    batch = _miss_batch(_indexed(), count)

    assert batch.local_nodes == count + 1
    assert batch.misses == count + 1
    assert batch.hits == 0


def test_the_membership_fixture_discriminates_present_and_absent_terms(
    members: Members,
) -> None:
    assert len(members.members) == MEMBERS
    assert members.present in members.members
    assert members.absent not in members.members


def test_the_memory_fixture_counts_both_shapes() -> None:
    count = 31
    leaves, pairs = _node_shapes(count)
    per_leaf = leaves.held / leaves.nodes
    per_pair = pairs.held / pairs.nodes

    assert leaves.nodes == count
    assert pairs.nodes == count + 1
    assert per_leaf > 0
    assert per_leaf < per_pair


def test_the_interning_hit_fixture_returns_the_held_branching_node(
    nodes: Nodes,
) -> None:
    forward = nodes.store.signature.by_name("fwd")
    before = (nodes.store.hits, nodes.store.misses)

    held = nodes.store.op(forward, nodes.featured, nodes.primitive)

    assert held is nodes.branching
    assert (nodes.store.hits, nodes.store.misses) == (before[0] + 1, before[1])


def test_term_equality_across_lineages_is_within_budget(
    benchmark: BenchmarkFixture, lineages: Lineages, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, EQUALITY, _across_lineages(lineages))
    benchmark.extra_info["budget"] = "cross-store-term-equality"
    benchmark.extra_info["nodes"] = EQUALITY_NODES

    if wall_clock_budget.enforced:
        assert nanoseconds <= 100_000


def test_term_hashing_is_within_budget(
    benchmark: BenchmarkFixture, nodes: Nodes, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, "hash(node)", {"node": nodes.branching})
    benchmark.extra_info["budget"] = "term-hashing"

    if wall_clock_budget.enforced:
        assert nanoseconds <= 80


def test_an_interning_hit_is_within_budget(
    benchmark: BenchmarkFixture, nodes: Nodes, wall_clock_budget: WallClockBudget
) -> None:
    forward = nodes.store.signature.by_name("fwd")
    nanoseconds = _per_operation(
        benchmark,
        "store.op(fwd, featured, primitive)",
        {
            "store": nodes.store,
            "fwd": forward,
            "featured": nodes.featured,
            "primitive": nodes.primitive,
        },
    )
    benchmark.extra_info["budget"] = "interning-hit"

    if wall_clock_budget.enforced:
        assert nanoseconds <= 250


def test_an_interning_miss_is_within_budget(
    benchmark: BenchmarkFixture, wall_clock_budget: WallClockBudget
) -> None:
    signature = _indexed()
    nanoseconds = _per_unit(
        benchmark,
        "miss_batch(signature, count)",
        "empty_batch(count)",
        {
            "empty_batch": _empty_batch,
            "miss_batch": _miss_batch,
            "signature": signature,
            "count": BATCH,
        },
        units=BATCH,
    )
    benchmark.extra_info["budget"] = "interning-miss"

    if wall_clock_budget.enforced:
        assert nanoseconds <= 10_000


def test_subterm_access_is_within_budget(
    benchmark: BenchmarkFixture, nodes: Nodes, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, "node.args[1]", {"node": nodes.branching})
    benchmark.extra_info["budget"] = "subterm-access"

    if wall_clock_budget.enforced:
        assert nanoseconds <= 100


def test_the_erased_core_is_within_budget(
    benchmark: BenchmarkFixture, nodes: Nodes, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(benchmark, "node.core", {"node": nodes.featured})
    benchmark.extra_info["budget"] = "term-erasure"

    if wall_clock_budget.enforced:
        assert nanoseconds <= 200


def test_set_membership_is_within_budget(
    benchmark: BenchmarkFixture, members: Members, wall_clock_budget: WallClockBudget
) -> None:
    nanoseconds = _per_operation(
        benchmark,
        "term in held",
        {"term": members.present, "held": members.members},
    )
    benchmark.extra_info["budget"] = "term-set-membership"
    benchmark.extra_info["members"] = MEMBERS

    if wall_clock_budget.enforced:
        assert nanoseconds <= 100


def test_interned_node_memory_is_within_budget(benchmark: BenchmarkFixture) -> None:
    leaves, pairs = benchmark.pedantic(
        _node_shapes,
        args=(INTERNED_NODES,),
        rounds=1,
        iterations=1,
    )
    per_leaf = leaves.held / leaves.nodes
    per_pair = pairs.held / pairs.nodes
    benchmark.extra_info["budget"] = "interned-node-memory"
    benchmark.extra_info["nodes"] = INTERNED_NODES
    benchmark.extra_info["leaf_bytes"] = leaves.held
    benchmark.extra_info["leaf_bytes_per_node"] = per_leaf
    benchmark.extra_info["pair_bytes"] = pairs.held
    benchmark.extra_info["pair_bytes_per_node"] = per_pair

    assert max(per_leaf, per_pair) <= 300

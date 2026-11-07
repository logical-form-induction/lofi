"""Prices what the interned term store holds alive, and for how long.

The store's lifetime discipline is three tiers, and each tier has a budget.
This module measures all three as live traced bytes under ``tracemalloc``:

- **The permanent tier.** The closure of CCGbank's distinct categories,
  interned once into a root store and never released. It needs the licensed
  corpus, so it is an integration benchmark, and it commits the counts and the
  bytes rather than anything the corpus holds.
- **The sentence tier.** Exactly 30,000 fresh nodes in one region,
  measured at the two shapes a sentence region holds: payload leaves, which is
  the least a published node can cost, and logical forms over shared
  structure, which is what a beam publishes.
- **The process.** The peak across repeated batches that are released,
  against a retained control that is not.

The report extrapolates measured bytes per fresh node to larger batches and
longer training horizons. Those workloads are not allocated here: the
measurements come from bounded sequences of eight-sentence batches.

What a released region does **not** release is its atom names: they live in a
table shared with the whole lineage, so a decoder that mints a fresh symbol per
candidate keeps paying for it after the batch is gone. The functional test
below checks that lifetime directly. Python's allocation tracer cannot price
the Rust-owned atom table.
"""

import gc
import tracemalloc
from itertools import pairwise
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.support.term_reference import CALCULUS, formulas, infix_syntax

from lofi.data.corpora.ccgbank import AutoLeaf, parse_auto
from lofi.terms import Operator, Signature, TermStore, Vocabulary

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

    from lofi.data.corpora.ccgbank import AutoNode

#: How many sentences of the corpus the category closure is read over, at
#: least. The file that reaches the count is read to its end.
SENTENCES: Final = 5_000

#: How many fresh nodes one sentence region is measured over.
REGION_NODES: Final = 30_000

#: The beam width and the fresh nodes one generated beam entry contributes.
BEAM: Final = 8
FRESH: Final = 6

#: The measured batch: how many sentences it holds, at which lengths it is
#: measured, and how many times each length is repeated.
BATCH: Final = 8
LENGTHS: Final = (5, 10, 15, 20)
REPEATS: Final = 3

#: How many batches the retained control publishes without releasing any.
RETAINED: Final = 4

#: The working point and training horizon used for extrapolation.
#: Neither workload is allocated here.
WORKING_BATCH: Final = 32
WORKING_LENGTH: Final = 20
STEPS: Final = 1_000

#: How many fresh symbols the atom-table measurement mints.
NAMES: Final = 2_000

#: The surface tokens CCGbank writes its two slashes as.
SLASHES: Final = ("/", "\\")

#: What ends an atom name in a category string.
DELIMITERS: Final = frozenset("()[/\\")


class Held(NamedTuple):
    """What one region held while its nodes were live.

    Attributes
    ----------
    nodes : int
        How many nodes the region owned, erased twins included.
    held : int
        The live traced bytes the region's nodes added.
    residual : int
        The live traced bytes still held after the region was released and the
        collector ran.
    """

    nodes: int
    held: int
    residual: int


class Corpus(NamedTuple):
    """The distinct categories read off a prefix of a corpus.

    Attributes
    ----------
    categories : tuple[str, ...]
        The distinct category strings, in sorted order.
    sentences : int
        How many derivations were read.
    files : int
        How many derivation files were read.
    """

    categories: tuple[str, ...]
    sentences: int
    files: int


class Inventory(NamedTuple):
    """The operator inventory a set of category strings is written over.

    Attributes
    ----------
    atoms : tuple[str, ...]
        The distinct atomic category names, in sorted order.
    features : tuple[str, ...]
        The distinct bracketed feature values, in sorted order.
    """

    atoms: tuple[str, ...]
    features: tuple[str, ...]


class Closure(NamedTuple):
    """What interning a set of categories into one root store costs.

    Attributes
    ----------
    categories : int
        How many distinct category strings were interned.
    nodes : int
        How many nodes the root store owns, their subterms and the erased
        twins of their featured operators included.
    held : int
        The live traced bytes those nodes added.
    """

    categories: int
    nodes: int
    held: int


class Round(NamedTuple):
    """One measured batch.

    Attributes
    ----------
    length : int
        The sentence length the batch was built at, in tokens.
    nodes : int
        How many fresh nodes the batch published.
    held : int
        The live traced bytes held while the batch was live.
    """

    length: int
    nodes: int
    held: int


class Lifetime(NamedTuple):
    """What a released sequence of batches costs, against a retained control.

    Attributes
    ----------
    released : tuple[Round, ...]
        The measured batches, each released before the next was built.
    peak : int
        The peak traced bytes over the whole released sequence.
    retained : tuple[int, ...]
        The live traced bytes after each batch of the control, which releases
        nothing.
    """

    released: tuple[Round, ...]
    peak: int
    retained: tuple[int, ...]


def cells(length: int) -> int:
    """Return how many chart cells a sentence of this length has.

    Parameters
    ----------
    length : int
        The sentence length in tokens.
    """
    return length * (length + 1) // 2


class Beams:
    """Publishes logical forms in the shape a beam generates them.

    One beam entry introduces its own event variable and applies the permanent
    predicates to it, producing six fresh nodes over already-interned
    structure. The predicates are published in the root store before anything is
    measured, so a batch pays for its own nodes and not for the lexicon.
    """

    def __init__(self) -> None:
        self._signature = formulas().with_vocabulary(
            Vocabulary("Lf", ("agent", "see", "theme"))
        )
        self._root = TermStore(self._signature)
        self._var = self._signature.bound("Lf")
        self._app = self._signature.by_name(CALCULUS.app)
        self._agent = self._root.const("Lf", "agent")
        self._theme = self._root.const("Lf", "theme")
        self._verb = self._root.const("Lf", "see")
        self._serial = 0

    @property
    def root(self) -> TermStore:
        """Return the root store the permanent predicates live in."""
        return self._root

    def entry(self, store: TermStore, /) -> None:
        """Publish the fresh nodes of one beam entry.

        Parameters
        ----------
        store : TermStore
            The region the entry is published into.
        """
        event = store.op(self._var, payload=self._serial)
        self._serial += 1
        agent = store.op(self._app, self._agent, event)
        theme = store.op(self._app, self._theme, event)
        joined = store.op(self._app, agent, theme)
        verbal = store.op(self._app, joined, self._verb)
        store.op(self._app, verbal, event)

    def batch(self, store: TermStore, /, *, sentences: int, length: int) -> None:
        """Publish one batch of beam entries.

        Parameters
        ----------
        store : TermStore
            The region the batch is published into.
        sentences : int
            How many sentences the batch holds.
        length : int
            The length of each sentence, in tokens.
        """
        for _ in range(sentences * cells(length) * BEAM):
            self.entry(store)


def traced(
    store: TermStore,
    build: Callable[[TermStore], None],
    /,
    *,
    max_nodes: int,
) -> Held:
    """Return what one region holds alive, and what survives its release.

    The region is measured while its nodes are live, then released and
    measured again with the collector run, so the second figure is what
    releasing the region did not reclaim.

    Parameters
    ----------
    store : TermStore
        The store the measured region is opened under.
    build : Callable[[TermStore], None]
        Publishes the nodes into the region it is handed.
    max_nodes : int
        The region's local node cap.
    """
    gc.collect()
    tracemalloc.start()
    nodes = 0
    held = 0

    try:
        start = tracemalloc.get_traced_memory()[0]

        with store.region("sentence", max_nodes=max_nodes) as region:
            build(region)
            nodes = region.local_nodes
            held = tracemalloc.get_traced_memory()[0] - start

        gc.collect()
        residual = tracemalloc.get_traced_memory()[0] - start

    finally:
        tracemalloc.stop()

    return Held(nodes, held, residual)


def leaf_bytes(nodes: int) -> Held:
    """Return what a region of fresh payload leaves holds.

    A leaf carries a payload and no arguments, which is the least a published
    node can hold, so this is the cheapest shape a sentence region can have.

    Parameters
    ----------
    nodes : int
        How many fresh leaves to publish.
    """
    signature = Signature(
        "region",
        ("T",),
        (),
        vocabularies=(Vocabulary("T", tuple(f"n{at}" for at in range(nodes))),),
    )
    root = TermStore(signature)
    op = signature.constant("T")

    def build(region: TermStore) -> None:
        for payload in range(nodes):
            region.op(op, payload=payload)

    return traced(root, build, max_nodes=nodes)


def logical_form_bytes(nodes: int) -> Held:
    """Return what a region of beam logical forms holds.

    Parameters
    ----------
    nodes : int
        How many fresh nodes to publish, a whole number of beam entries.

    Raises
    ------
    ValueError
        If the count is not a whole number of beam entries.
    """
    if nodes % FRESH:
        msg = f"{nodes} fresh nodes is not a whole number of {FRESH}-node entries"

        raise ValueError(msg)

    beams = Beams()

    def build(region: TermStore) -> None:
        for _ in range(nodes // FRESH):
            beams.entry(region)

    return traced(beams.root, build, max_nodes=nodes)


def region_shapes(nodes: int) -> tuple[Held, Held]:
    """Return what both sentence-region shapes hold at one node count.

    Parameters
    ----------
    nodes : int
        How many fresh nodes each shape publishes.
    """
    return leaf_bytes(nodes), logical_form_bytes(nodes)


def lifetime() -> Lifetime:
    """Return what repeated batches cost when released, and when not.

    The released sequence publishes one batch per region at each length, in
    ascending order, and releases it before the next is built, so its peak is
    one batch's worth plus whatever release failed to reclaim. The retained
    control publishes its batches into the root store, which nothing releases.
    The control runs after the peak is read, so it cannot inflate it.
    """
    beams = Beams()
    rounds: list[Round] = []
    retained: list[int] = []
    gc.collect()
    tracemalloc.start()

    try:
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()

        for length in LENGTHS:
            for _ in range(REPEATS):
                with beams.root.region("batch", max_nodes=2_000_000) as region:
                    beams.batch(region, sentences=BATCH, length=length)
                    rounds.append(
                        Round(
                            length,
                            region.local_nodes,
                            tracemalloc.get_traced_memory()[0] - start,
                        )
                    )

                gc.collect()

        peak = tracemalloc.get_traced_memory()[1] - start

        for _ in range(RETAINED):
            beams.batch(beams.root, sentences=BATCH, length=LENGTHS[0])
            retained.append(tracemalloc.get_traced_memory()[0] - start)

    finally:
        tracemalloc.stop()

    return Lifetime(tuple(rounds), peak, tuple(retained))


def node_categories(node: AutoNode, /) -> Iterator[str]:
    """Yield the category of a derivation node and of every node below it.

    Parameters
    ----------
    node : AutoNode
        The derivation node to read.
    """
    yield node.category

    if isinstance(node, AutoLeaf):
        return

    for child in node.children:
        yield from node_categories(child)


def conjunct_base(category: str, /) -> str:
    """Return the category a conjunct marks, without the marker's parentheses.

    Parameters
    ----------
    category : str
        A node category, conjunct-marked or not.
    """
    base = category.removesuffix("[conj]")

    if base == category or not base.startswith("("):
        return base

    depth = 0

    for position, character in enumerate(base):
        depth += {"(": 1, ")": -1}.get(character, 0)

        if depth == 0:
            return base[1:-1] if position == len(base) - 1 else base

    return base


def corpus_categories(root: Path, /, *, at_least: int) -> Corpus:
    """Return the distinct categories of a prefix of a corpus, in file order.

    Files are read in path order and whole, so the categories are those of a
    prefix of the corpus rather than of a subsample drawn from all of it. A
    trailing ``[conj]`` marks a node as a coordination conjunct rather than
    adding a feature, so it is read off as derivation markup, as the corpus
    round trip reads it, and the conjunct interns as the category it marks.

    Parameters
    ----------
    root : Path
        The corpus root to walk, whose ``.auto`` files hold the derivations.
    at_least : int
        How many derivations to read before stopping. The file that reaches
        the count is read to its end.
    """
    seen: set[str] = set()
    sentences = 0
    files = 0

    for path in sorted(root.rglob("*.auto")):
        derivations = parse_auto(path.read_bytes())
        files += 1

        for derivation in derivations:
            sentences += 1
            seen.update(map(conjunct_base, node_categories(derivation.tree)))

        if sentences >= at_least:
            break

    return Corpus(tuple(sorted(seen)), sentences, files)


def inventory(categories: Iterable[str], /) -> Inventory:
    """Return the atoms and features a set of category strings is written over.

    A category string is a tree of slashes over atoms, each atom optionally
    carrying one bracketed feature. Reading the inventory off the strings is
    what lets a licensed corpus be interned without a declared atom list that
    a different release could outgrow.

    Parameters
    ----------
    categories : Iterable[str]
        The category strings to read.
    """
    atoms: set[str] = set()
    features: set[str] = set()

    for text in categories:
        at = 0

        while at < len(text):
            character = text[at]

            if character == "[":
                close = text.index("]", at)
                features.add(text[at + 1 : close])
                at = close + 1

            elif character.isspace() or character in "()" or character in SLASHES:
                at += 1

            else:
                # an atom runs to the next delimiter, and is at least one
                # character long, so the scan always advances
                end = at + 1

                while end < len(text) and not (
                    text[end] in DELIMITERS or text[end].isspace()
                ):
                    end += 1

                atoms.add(text[at:end])
                at = end

    return Inventory(tuple(sorted(atoms)), tuple(sorted(features)))


def closure_signature(found: Inventory, /) -> Signature:
    """Return a signature over a corpus's own atoms, features and slashes.

    Every feature is declared on every atom and on both slashes. Which
    operator carries which feature is not recoverable from a category string
    without parsing it, and parsing is what the signature is built for, so the
    declaration is the inventory's product rather than the corpus's own
    pairing.

    Parameters
    ----------
    found : Inventory
        The atoms and features read off the categories.
    """
    slashes = tuple(
        Operator(token, "Cat", arg_sorts=("Cat", "Cat"), family="slash")
        for token in SLASHES
    )
    atoms = tuple(Operator(name, "Cat") for name in found.atoms)

    return Signature(
        "ccgbank",
        ("Cat",),
        (*atoms, *slashes),
        features={op.name: found.features for op in (*atoms, *slashes)},
    )


def closure_bytes(categories: Sequence[str], /) -> Closure:
    """Return what interning every distinct category into one root store costs.

    The signature and the syntax are built before anything is traced, so the
    figure is the closure the categories intern to and not the declaration
    they are read under.

    Parameters
    ----------
    categories : Sequence[str]
        The distinct category strings to intern.
    """
    signature = closure_signature(inventory(categories))
    syntax = infix_syntax(signature, *zip(SLASHES, SLASHES, strict=True))
    store = TermStore(signature)
    gc.collect()
    tracemalloc.start()

    try:
        start = tracemalloc.get_traced_memory()[0]

        for text in categories:
            syntax.parse(store, text)

        held = tracemalloc.get_traced_memory()[0] - start
        nodes = store.local_nodes

    finally:
        tracemalloc.stop()

    return Closure(len(categories), nodes, held)


#: Two derivations in the AUTO format, holding the category shapes a CCGbank
#: closure has to admit: features on atoms, a conjunct-marked functor, a
#: punctuation atom and the coordinator.
SAMPLE_AUTO: Final = (
    "ID=sample.1\n"
    r"(<T S[dcl] 1 2> (<L NP NNP NNP Vinken NP>) (<T S[dcl]\NP 0 2> "
    r"(<L (S[dcl]\NP)/NP VBD VBD joined (S[dcl]\NP)/NP>) (<T NP 1 2> "
    r"(<L NP[nb]/N DT DT the NP[nb]/N>) (<L N NN NN board N>) ) ) )"
    "\nID=sample.2\n"
    r"(<T S[dcl] 0 2> (<T S[dcl] 0 2> (<L S[dcl] VBD VBD rose S[dcl]>) "
    r"(<L , , , , ,>) ) (<T (S[dcl]\NP)[conj] 0 2> (<L conj CC CC and conj>) "
    r"(<L S[dcl]\NP VBD VBD fell S[dcl]\NP>) ) )"
    "\n"
)


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Return a corpus root holding one unlicensed derivation file.

    Returns
    -------
    Path
        The directory the sample file was written under.
    """
    (tmp_path / "sample.auto").write_text(SAMPLE_AUTO, encoding="utf-8", newline="\n")

    return tmp_path


def test_the_sample_corpus_reads_as_two_derivations_over_eight_categories(
    sample_corpus: Path,
) -> None:
    read = corpus_categories(sample_corpus, at_least=SENTENCES)

    assert (read.files, read.sentences) == (1, 2)
    assert read.categories == (
        r"(S[dcl]\NP)/NP",
        ",",
        "N",
        "NP",
        "NP[nb]/N",
        "S[dcl]",
        r"S[dcl]\NP",
        "conj",
    )


def test_the_read_stops_at_the_file_that_reaches_the_count(
    sample_corpus: Path,
) -> None:
    (sample_corpus / "second.auto").write_text(
        SAMPLE_AUTO, encoding="utf-8", newline="\n"
    )
    bounded = corpus_categories(sample_corpus, at_least=1)
    whole = corpus_categories(sample_corpus, at_least=SENTENCES)

    assert (bounded.files, bounded.sentences) == (1, 2)
    assert (whole.files, whole.sentences) == (2, 4)
    assert bounded.categories == whole.categories


def test_the_inventory_reads_every_atom_and_feature_off_the_strings(
    sample_corpus: Path,
) -> None:
    found = inventory(corpus_categories(sample_corpus, at_least=SENTENCES).categories)

    assert found.atoms == (",", "N", "NP", "S", "conj")
    assert found.features == ("dcl", "nb")


def test_the_closure_interns_every_category_and_shares_their_subterms(
    sample_corpus: Path,
) -> None:
    categories = corpus_categories(sample_corpus, at_least=SENTENCES).categories
    closure = closure_bytes(categories)

    # eight categories over five atoms: the subterms they share and the erased
    # twin of each featured node are interned once each
    assert closure.categories == len(categories)
    assert closure.nodes > closure.categories
    assert closure.held > 0

    signature = closure_signature(inventory(categories))
    syntax = infix_syntax(signature, *zip(SLASHES, SLASHES, strict=True))
    store = TermStore(signature)
    transitive = syntax.parse(store, r"(S[dcl]\NP)/NP")

    assert store.local_nodes < closure.nodes
    assert syntax.parse(store, r"S[dcl]\NP") is transitive.args[0]


def test_a_beam_entry_publishes_six_fresh_nodes_over_the_permanent_ones() -> None:
    beams = Beams()
    permanent = beams.root.local_nodes
    one = 0

    with beams.root.region("batch", max_nodes=100) as region:
        beams.entry(region)
        one = region.local_nodes
        beams.entry(region)

        assert region.local_nodes == 2 * FRESH
        # the predicates the entries apply stay in the root store
        assert beams.root.local_nodes == permanent

    assert one == FRESH
    assert permanent == 3


def test_a_sentence_has_one_chart_cell_per_span() -> None:
    # the ten spans of a four-token sentence, and the 210 that §4.5 multiplies
    # the beam by at the working point
    assert (cells(1), cells(4), cells(WORKING_LENGTH)) == (1, 10, 210)


def test_a_batch_publishes_one_entry_per_beam_slot_of_every_cell() -> None:
    beams = Beams()

    with beams.root.region("batch", max_nodes=100_000) as region:
        beams.batch(region, sentences=2, length=4)

        # two sentences of four tokens, so ten cells each
        assert region.local_nodes == 2 * 10 * BEAM * FRESH


def test_a_released_region_does_not_release_the_atom_names_it_minted() -> None:
    root = Beams().root
    names = tuple(f"candidate-{serial:06d}" for serial in range(NAMES))
    atoms: tuple[int, ...] = ()

    with root.region("sentence", max_nodes=1) as region:
        atoms = tuple(region.atom("Lf", name) for name in names)

        assert region.local_nodes == 0
        assert len(set(atoms)) == NAMES

    assert tuple(root.atom("Lf", name) for name in names) == atoms
    assert tuple(root.atom_name(atom) for atom in atoms) == names


@pytest.mark.corpus
@pytest.mark.integration
def test_the_ccgbank_category_closure_contains_categories_and_subterms(
    ccgbank_root: Path,
) -> None:
    read = corpus_categories(ccgbank_root, at_least=SENTENCES)

    closure = closure_bytes(read.categories)

    assert closure.categories > 0
    assert closure.nodes > closure.categories


@pytest.mark.corpus
@pytest.mark.integration
def test_the_ccgbank_category_closure_is_within_budget(
    benchmark: BenchmarkFixture, ccgbank_root: Path
) -> None:
    read = corpus_categories(ccgbank_root, at_least=SENTENCES)
    closure = benchmark.pedantic(
        closure_bytes,
        args=(read.categories,),
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["budget"] = "permanent-term-storage"
    benchmark.extra_info["sentences_read"] = read.sentences
    benchmark.extra_info["files_read"] = read.files
    benchmark.extra_info["unique_categories"] = closure.categories
    benchmark.extra_info["closure_nodes"] = closure.nodes
    benchmark.extra_info["closure_bytes"] = closure.held
    benchmark.extra_info["bytes_per_node"] = closure.held / closure.nodes

    # enforce the 2 MiB budget on the permanent tier
    assert closure.held <= 2 * 2**20


def test_thirty_thousand_fresh_nodes_report_their_shape_and_release() -> None:
    control_leaves, control_forms = region_shapes(REGION_NODES // 10)
    leaves, forms = region_shapes(REGION_NODES)

    assert leaves.nodes == REGION_NODES
    assert forms.nodes == REGION_NODES
    assert min(leaves.held, forms.held) > 0
    assert leaves.residual < control_leaves.held
    assert forms.residual < control_forms.held


def test_thirty_thousand_fresh_nodes_are_within_budget(
    benchmark: BenchmarkFixture,
) -> None:
    leaves, forms = benchmark.pedantic(
        region_shapes,
        args=(REGION_NODES,),
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["budget"] = "sentence-term-storage"
    benchmark.extra_info["nodes"] = REGION_NODES
    benchmark.extra_info["leaf_bytes"] = leaves.held
    benchmark.extra_info["leaf_bytes_per_node"] = leaves.held / leaves.nodes
    benchmark.extra_info["logical_form_bytes"] = forms.held
    benchmark.extra_info["logical_form_bytes_per_node"] = forms.held / forms.nodes

    # enforce the 8 MiB budget on one sentence region, against the cheaper of
    # the two shapes, so the budget is read against the store's best case
    assert min(leaves.held, forms.held) <= 8 * 2**20


def test_released_batches_hold_one_batch_and_retained_batches_accumulate() -> None:
    measured = lifetime()
    largest = max(measured.released, key=lambda round_: round_.held)
    fresh = sum(round_.nodes for round_ in measured.released)
    per_node = sum(round_.held for round_ in measured.released) / fresh
    working = WORKING_BATCH * cells(WORKING_LENGTH) * BEAM * FRESH

    assert [round_.nodes for round_ in measured.released] == [
        BATCH * cells(length) * BEAM * FRESH
        for length in LENGTHS
        for _ in range(REPEATS)
    ]
    assert all(before < after for before, after in pairwise(measured.retained))
    assert measured.retained[0] > 0
    assert measured.retained[-1] > measured.retained[0] * (RETAINED - 1)
    assert largest.held <= measured.peak <= largest.held * 1.5
    assert measured.peak < working * per_node


def test_released_batches_hold_one_batch_and_retained_batches_hold_all_of_them(
    benchmark: BenchmarkFixture,
) -> None:
    measured: Lifetime = benchmark.pedantic(lifetime, rounds=1, iterations=1)
    largest = max(measured.released, key=lambda round_: round_.held)
    fresh = sum(round_.nodes for round_ in measured.released)
    per_node = sum(round_.held for round_ in measured.released) / fresh
    working = WORKING_BATCH * cells(WORKING_LENGTH) * BEAM * FRESH
    benchmark.extra_info["budget"] = "released-term-batches"
    benchmark.extra_info["measured_batches"] = len(measured.released)
    benchmark.extra_info["measured_sentences_per_batch"] = BATCH
    benchmark.extra_info["measured_fresh_nodes"] = fresh
    benchmark.extra_info["measured_peak_bytes"] = measured.peak
    benchmark.extra_info["measured_largest_batch_bytes"] = largest.held
    benchmark.extra_info["measured_bytes_per_fresh_node"] = per_node
    benchmark.extra_info["measured_retained_bytes"] = measured.retained
    benchmark.extra_info["extrapolated_working_batch_nodes"] = working
    benchmark.extra_info["extrapolated_working_batch_bytes"] = working * per_node
    benchmark.extra_info["extrapolated_unfreed_bytes_over_steps"] = (
        working * per_node * STEPS
    )
    benchmark.extra_info["extrapolated_steps"] = STEPS

    # enforce the 512 MiB budget on peak store memory across released batches
    assert measured.peak <= 512 * 2**20

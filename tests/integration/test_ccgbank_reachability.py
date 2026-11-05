"""Gold-derivation reachability of CCGbank under three declared rule inventories.

The derivations of CCGbank's Section 00 are read in file order, whole files at
a time, until at least a thousand have been read. They are decoded by the AUTO
codec into Layers records and loaded once per inventory by the Layers reader,
which classifies every decoded derivation against the inventory and returns
the reachability report the rows are built from. Each file is parsed once.

The three inventories share one category algebra read off the sample's own
categories. The full inventory is generalized composition up to order two in
both directions, harmonic and crossed, with application, first-order
substitution and type raising, together with CCGbank's declared exceptional
rules. The application inventory keeps the exceptional rules and drops every
combinator except application. The combinatory inventory keeps every
combinator and drops the exceptional rules. Both are strict subinventories of
the full one.

The committed report beside this module holds the rows a run produced. Running
this module with ``LOFI_REGENERATE_REACHABILITY`` set rewrites it from the
corpus and skips the comparison; a run without it compares a fresh computation
against it. The committed report is a record of a previous run, so the
comparison detects drift; it is not an independently sourced expectation. Each test is
marked ``integration`` and ``corpus`` and reads its corpus through
``ccgbank_root``, which skips unless ``LOFI_CCGBANK`` names a directory. The
report holds categories and counts and no corpus text.
"""

import json
import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from lairs import Corpus
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import expression as expression_records
from lairs.records import segmentation as segmentation_records
from tests.support.term_reference import infix_syntax

from lofi.data.corpora import SignatureLabels
from lofi.data.corpora.ccgbank import AutoLeaf, parse_auto
from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec
from lofi.data.corpora.layers import ComparisonInventory, LayersCorpusReader
from lofi.eval import ReachabilityRow, reachability_table
from lofi.grammars.ccg import CCGAssembly, FunctorView
from lofi.judge import Conjunctions, NoNormalForm
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    RuleSetSpec,
    RuleSpec,
    SubstitutionSpec,
    TypeRaisingSpec,
)
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.language import registered_language
from lofi.terms import (
    FeatureErasure,
    Operator,
    Signature,
    TermStore,
    Vocabulary,
    _compile_grammar,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Sequence

    import didactic.api as dx

    from lofi._types import JsonValue
    from lofi.data.corpora.ccgbank import AutoDerivation, AutoNode
    from lofi.judge import Rule, RuleSet
    from lofi.terms import Term

pytestmark = [pytest.mark.integration, pytest.mark.corpus]

#: The committed report a fresh run is compared against.
REPORT: Final = Path(__file__).with_name("ccgbank_reachability.json")

#: The environment variable whose presence rewrites the committed report.
REGENERATE: Final = "LOFI_REGENERATE_REACHABILITY"

#: How many derivations the evaluated sample holds at least.
SAMPLE: Final = 1_000

#: The section the evaluated sample is read from.
EVALUATED: Final = "00"

#: The sections of the conventional CCGbank training split.
TRAINING: Final = tuple(f"{section:02d}" for section in range(2, 22))

#: The corpus name and language every row is reported under.
CORPUS: Final = "ccgbank-1.1"
LANGUAGE: Final = "en"

#: The atoms the exceptional rules and the conjunction markers name.
DECLARED_ATOMS: Final = ("N", "NP", "S", "conj", ",", ".", ":", ";", "LRB", "RRB")
MARKERS: Final = ("conj", ",", ".", ":", ";", "LRB", "RRB")
CONJ: Final = "[conj]"
SLASHES: Final = ("/", "\\")
CREATED: Final = datetime(1970, 1, 1, tzinfo=UTC)
CORPUS_URI: Final = "at://local/pub.layers.corpus.corpus/ccgbank"

#: The record model each collection the AUTO codec emits is validated as.
MODELS: Final[dict[str, type[dx.Model]]] = {
    "pub.layers.expression.expression": expression_records.Expression,
    "pub.layers.segmentation.segmentation": segmentation_records.Segmentation,
    "pub.layers.annotation.annotationLayer": annotation_records.AnnotationLayer,
}


def sample(root: Path, /) -> tuple[AutoDerivation, ...]:
    """Return the evaluated derivations, whole files at a time, in path order.

    Parameters
    ----------
    root : Path
        The CCGbank root.
    """
    derivations: list[AutoDerivation] = []
    files = (
        path for path in sorted(root.rglob("*.auto")) if path.parent.name == EVALUATED
    )

    for path in files:
        derivations.extend(parse_auto(path.read_bytes()))

        if len(derivations) >= SAMPLE:
            break

    return tuple(derivations)


def training_size(root: Path, /) -> int:
    """Return how many derivations the training sections hold.

    Parameters
    ----------
    root : Path
        The CCGbank root.
    """
    return sum(
        len(parse_auto(path.read_bytes()))
        for path in sorted(root.rglob("*.auto"))
        if path.parent.name in TRAINING
    )


def node_categories(node: AutoNode, /) -> Iterator[str]:
    """Yield the category of a node and of every node below it.

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


def written_inventory(
    categories: Sequence[str], /
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the atoms and bracketed feature runs the categories are written over.

    A run of adjacent brackets, as in ``S[dcl][conj]``, is one feature.

    Parameters
    ----------
    categories : Sequence[str]
        The category strings to read.
    """
    atoms: set[str] = set(DECLARED_ATOMS)
    features: set[str] = set()

    for text in categories:
        at = 0

        while at < len(text):
            if text[at] == "[":
                end = at

                while end < len(text) and text[end] == "[":
                    end = text.index("]", end) + 1

                features.add(text[at + 1 : end - 1])
                at = end
            elif text[at] in "()" or text[at] in SLASHES:
                at += 1
            else:
                end = at + 1

                while (
                    end < len(text)
                    and text[end] not in "()["
                    and text[end] not in SLASHES
                ):
                    end += 1

                atoms.add(text[at:end])
                at = end

    return tuple(sorted(atoms)), tuple(sorted(features))


class Algebra:
    """The category algebra, rule inventories and labels one sample is read under.

    Atoms carry every feature the sample writes. Slashes carry only the
    conjunction mark, so that a trailing ``[conj]`` marks the whole category,
    which is CCGbank's convention.

    Parameters
    ----------
    categories : Sequence[str]
        Every distinct category the sample's derivations hold.
    """

    def __init__(self, categories: Sequence[str]) -> None:
        atoms, features = written_inventory(categories)
        featured_atoms = tuple(
            f"{atom}[{feature}]" for atom in atoms for feature in features
        )
        atom_names = (*atoms, *featured_atoms)
        slash_operators = tuple(
            Operator(token, "Cat", arg_sorts=("Cat", "Cat"), family="slash")
            for token in SLASHES
        )
        featured_slashes = tuple(
            Operator(
                f"{token}{CONJ}",
                "Cat",
                arg_sorts=("Cat", "Cat"),
                family="slash",
                erases_to=token,
            )
            for token in SLASHES
        )
        signature = Signature(
            "ccgbank-categories",
            ("Cat",),
            (*slash_operators, *featured_slashes),
            vocabularies=(Vocabulary("Cat", atom_names),),
            constant_cores={
                "Cat": {
                    f"{atom}[{feature}]": atom for atom in atoms for feature in features
                }
            },
        )
        self.store = TermStore(signature)
        syntax = infix_syntax(signature, *zip(SLASHES, SLASHES, strict=True))
        self.terms = {text: syntax.parse(self.store, text) for text in categories}
        marked = tuple(term for text, term in self.terms.items() if text.endswith(CONJ))
        matching = FeatureErasure()
        pairs = tuple((matching.key(term), term) for term in marked)
        conjunctions = Conjunctions(
            "ccgbank",
            tuple(syntax.parse(self.store, marker) for marker in MARKERS),
            pairs,
        )
        functors = FunctorView(self.store, "/", "\\", result=0, argument=1)
        self.assembly = CCGAssembly(
            self.store,
            syntax,
            partial(syntax.parse, self.store),
            {"/": (functors, "/"), "\\": (functors, "\\")},
            matching,
            conjunctions,
            NoNormalForm(),
            category_sort="Cat",
        )
        self.labels = Signature(
            "ccgbank-labels",
            ("Cat",),
            tuple(Operator(text, "Cat") for text in categories),
        )

    def rules(self, name: str, specs: Sequence[RuleSpec], /) -> RuleSet[Term]:
        """Return a declared inventory built over the algebra.

        Parameters
        ----------
        name : str
            The inventory name.
        specs : Sequence[RuleSpec]
            The declared rules.
        """
        declaration = RuleSetSpec(name=name, rules=tuple(specs))
        grammar = ccgbank_grammar().with_(rules=declaration)
        language = registered_language("propositional")
        plan = _compile_grammar(
            language.compile_envelope.canonical_declaration,
            grammar.compile_envelope.canonical_declaration,
        )

        return grammar.prepared_plugins.rules.adopt(
            tuple(plan.rules),
            self.assembly,
        ).rule_set


class TermRecovery:
    """Step recovery over category strings, through a rule set over their terms.

    Parameters
    ----------
    terms : dict[str, Term]
        The term each category string parses to.
    rules : RuleSet[Term]
        The inventory that recovers a step.
    """

    __slots__ = ("_rules", "_terms")

    def __init__(self, terms: dict[str, Term], rules: RuleSet[Term]) -> None:
        self._terms = terms
        self._rules = rules

    def recover(self, inputs: Sequence[str], output: str, /) -> tuple[Rule[Term], ...]:
        """Return the rules that relate the premises' terms to the conclusion's.

        Parameters
        ----------
        inputs : Sequence[str]
            The premise categories in child order.
        output : str
            The conclusion category.
        """
        return self._rules.recover(
            tuple(self._terms[label] for label in inputs), self._terms[output]
        )


APPLICATION: Final[tuple[RuleSpec, ...]] = (
    ApplicationSpec(view="/"),
    ApplicationSpec(view="\\"),
)
COMBINATORS: Final[tuple[RuleSpec, ...]] = (
    *APPLICATION,
    *(
        CompositionSpec(
            direction="unrestricted",
            view=slash,
            order=order,
            crossed=crossed,
        )
        for order in (1, 2)
        for slash in ("/", "\\")
        for crossed in (False, True)
    ),
    *(
        SubstitutionSpec(view=slash, order=1, crossed=crossed)
        for slash in ("/", "\\")
        for crossed in (False, True)
    ),
    TypeRaisingSpec(view="/", arguments=None),
    TypeRaisingSpec(view="\\", arguments=None),
)
PACKAGED_RULES: Final = ccgbank_grammar().rules.rules

#: Each inventory's name and declared rules, the full inventory first.
INVENTORIES: Final[tuple[tuple[str, tuple[RuleSpec, ...]], ...]] = (
    ("ccgbank-full", (*COMBINATORS, *PACKAGED_RULES)),
    ("ccgbank-application", (*APPLICATION, *PACKAGED_RULES)),
    ("ccgbank-combinatory", COMBINATORS),
)


def layers_corpus(derivations: Sequence[AutoDerivation], /) -> Corpus:
    """Return the evaluated derivations as one Layers corpus with one split.

    Parameters
    ----------
    derivations : Sequence[AutoDerivation]
        The parsed derivations, in order.
    """
    fragment = CcgbankAutoCodec().iso.forward(tuple(derivations))
    corpus = Corpus(uri=CORPUS_URI)
    corpus.add_record(CORPUS_URI, corpus_records.Corpus(name=CORPUS, createdAt=CREATED))
    members = 0

    for record in fragment.records:
        corpus.add_record(
            record.local_id, MODELS[record.nsid].model_validate_json(record.value_json)
        )

        if record.nsid == "pub.layers.expression.expression":
            corpus.add_record(
                f"at://local/pub.layers.corpus.membership/member-{members}",
                corpus_records.Membership(
                    corpusRef=CORPUS_URI,
                    expressionRef=record.local_id,
                    createdAt=CREATED,
                    split=EVALUATED,
                    ordinal=members,
                ),
            )
            members += 1

    return corpus


def measured_rows(
    root: Path, derivations: Sequence[AutoDerivation], /
) -> tuple[ReachabilityRow[Hashable], ...]:
    """Return one reachability row per inventory over the evaluated sample.

    Parameters
    ----------
    root : Path
        The CCGbank root, whose training sections are counted.
    derivations : Sequence[AutoDerivation]
        The evaluated derivations.
    """
    categories = tuple(
        sorted(
            {
                text
                for derivation in derivations
                for text in node_categories(derivation.tree)
            }
        )
    )
    algebra = Algebra(categories)
    corpus = layers_corpus(derivations)
    trained = training_size(root)
    rows: list[ReachabilityRow[Hashable]] = []

    for name, specs in INVENTORIES:
        comparison = ComparisonInventory(
            SignatureLabels(algebra.labels),
            TermRecovery(algebra.terms, algebra.rules(name, specs)),
        )
        loaded = LayersCorpusReader(
            corpus, SignatureLabels(algebra.labels), comparison=comparison
        ).report()
        report = loaded.reachability_report

        assert report is not None

        rows.append(
            ReachabilityRow(
                report,
                inventory=name,
                corpus=CORPUS,
                language=LANGUAGE,
                training_size=trained,
            )
        )

    return reachability_table(rows)


@pytest.fixture(scope="module")
def sampled(ccgbank_root: Path) -> tuple[AutoDerivation, ...]:
    """Return the evaluated derivations of the configured corpus."""
    return sample(ccgbank_root)


@pytest.fixture(scope="module")
def rows(
    ccgbank_root: Path, sampled: tuple[AutoDerivation, ...]
) -> tuple[ReachabilityRow[Hashable], ...]:
    """Return the rows of a fresh run over the configured corpus."""
    return measured_rows(ccgbank_root, sampled)


def specification(
    rows: Sequence[ReachabilityRow[Hashable]], /
) -> list[dict[str, JsonValue]]:
    """Return the rows as the committed report writes them."""
    return [row.to_spec(str) for row in rows]


@pytest.mark.skipif(
    REGENERATE not in os.environ, reason=f"set {REGENERATE} to rewrite the report"
)
def test_regenerate_the_committed_report(
    rows: tuple[ReachabilityRow[Hashable], ...],
) -> None:
    REPORT.write_text(
        json.dumps(specification(rows), indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    assert json.loads(REPORT.read_text(encoding="utf-8")) == specification(rows)


@pytest.mark.skipif(
    REGENERATE in os.environ, reason="the report is being rewritten from this run"
)
def test_a_fresh_run_reproduces_the_committed_report(
    rows: tuple[ReachabilityRow[Hashable], ...],
) -> None:
    committed = json.loads(REPORT.read_text(encoding="utf-8"))

    assert specification(rows) == committed


def test_the_report_covers_three_inventories_over_a_thousand_derivations(
    rows: tuple[ReachabilityRow[Hashable], ...],
    sampled: tuple[AutoDerivation, ...],
) -> None:
    assert len(sampled) >= SAMPLE
    assert [row.inventory for row in rows] == [name for name, _ in INVENTORIES]
    assert all(row.checked_derivations == len(sampled) for row in rows)
    assert all(row.training_size > row.checked_derivations for row in rows)


def test_each_strict_subinventory_reaches_strictly_fewer_derivations(
    rows: tuple[ReachabilityRow[Hashable], ...],
) -> None:
    full, *reduced = rows

    for row in reduced:
        assert row.reachable_derivations < full.reachable_derivations
        assert row.reachable_steps < full.reachable_steps


def test_every_blocking_histogram_sums_to_its_unreachable_derivations(
    rows: tuple[ReachabilityRow[Hashable], ...],
) -> None:
    for row in rows:
        assert sum(count for _, count in row.histogram) == row.unreachable_derivations
        assert (
            row.report.checked_steps - row.report.reachable_steps
            >= row.unreachable_derivations
        )

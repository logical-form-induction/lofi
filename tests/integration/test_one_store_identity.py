"""Constant identity across stores, read from the packaged documents.

The packaged synthetic-inference document is composed, its interpretation is
lowered twice, and every meaning the corpus and its pairs carry is read into
both lowered stores. The second store interns free-variable names before it
reads anything, so a constant whose identity depended on the order a store met
its names would key differently in the two.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

from hypothesis import strategies as st
from tests.support.term_strategies import terms

from lofi.data.synthetic.spec.inference import (
    SyntheticInferenceDecl,
    SyntheticInferenceLowerer,
)
from lofi.data.synthetic.spec.interpretation import lower_interpretation
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.run import MANIFEST_NAME, ManifestStatus
from lofi.run._entrypoint import main
from lofi.spec import SpecLoader
from lofi.spec.grammar import GrammarIso
from lofi.spec.language import LanguageIso
from lofi.terms import Signature, Term, TermStore
from lofi.terms.laws import TermFixture, check_term_laws

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.synthetic import InterpretationFunction
    from lofi.terms import Operator

CONFIGS: Final = Path(__file__).resolve().parents[2] / "lofi/configs"
INFERENCE: Final = CONFIGS / "data/datasets/train/dataset/synthetic-inference.yaml"
PARSER: Final = CONFIGS / "run/synthetic-parser.yaml"


def _declaration() -> SyntheticInferenceDecl:
    """Compose the packaged synthetic-inference document."""
    return SpecLoader().compose_traced(INFERENCE, schema=SyntheticInferenceDecl).spec


def _lowered(declaration: SyntheticInferenceDecl) -> InterpretationFunction:
    """Lower the document's interpretation into a store of its own."""
    _, interpretation = lower_interpretation(
        declaration.grammar,
        declaration.source.language,
        declaration.source.interpretation,
        GrammarIso(),
        LanguageIso(),
    )

    return interpretation


def _meanings(
    declaration: SyntheticInferenceDecl, interpretation: InterpretationFunction
) -> tuple[str, ...]:
    """Return every meaning the document's lexicon and drawn pairs carry."""
    run = SyntheticInferenceLowerer().lower(
        declaration, ParsePolicy(Retain.FOREST, Reachability.ALL)
    )
    drawn = (
        form.text for pair in run.pairs for form in (pair.premise, pair.hypothesis)
    )

    return (
        *(
            interpretation.render(meaning)
            for meanings in interpretation.entries.values()
            for meaning in meanings
        ),
        *drawn,
    )


def _transplant(here: TermStore, there: TermStore) -> Callable[[Term], Term]:
    """Return the rebuilding of a term of one store in another, by name.

    Constants and free variables are rebuilt from their names, so the
    rebuilt term shares nothing with the original but its structure and the
    names its leaves carry.
    """

    def rebuild(term: Term) -> Term:
        kind = term.op.payload_sort

        if kind == "constant":
            return there.const(term.op.sort, here.payload_name(term))

        if kind == "free":
            return there.fvar(term.op.sort, here.payload_name(term))

        args = tuple(rebuild(arg) for arg in term.args)

        return there.op(term.op, *args, payload=term.payload)

    return rebuild


def _seeded(signature: Signature, sort: str, names: tuple[str, ...]) -> TermStore:
    """Return a store that has interned free-variable names before anything else."""
    store = TermStore(signature)

    for name in reversed(names):
        store.atom(sort, name)

    return store


def _shift(term: Term, position: int) -> dict[str, int]:
    """Return how many binding sites of each sort scope over one argument."""
    op: Operator = term.op

    return {site.sort: op.shift(position, site.sort) for site in op.scope(position)}


def test_every_lowered_meaning_has_one_key_in_two_lowered_stores() -> None:
    declaration = _declaration()
    first = _lowered(declaration)
    second = _lowered(declaration)
    sort = "Tm"

    for name in ("z", "y", "x", "p1", "p0"):
        second.store.atom(sort, name)

    meanings = _meanings(declaration, first)

    assert first.store is not second.store
    assert first.store.signature.digest == second.store.signature.digest
    assert meanings

    for text in meanings:
        here = first.read_text(text)
        there = second.read_text(text)

        assert here.key() == there.key(), text
        assert here == there
        assert hash(here) == hash(there)

    vocabulary = first.store.signature.vocabulary(sort)

    for name in vocabulary.names:
        here = first.store.const(sort, name)
        there = second.store.const(sort, name)

        assert here.payload == there.payload == vocabulary.id(name)


def test_the_term_laws_hold_on_the_lowered_unified_signature() -> None:
    lowered = _lowered(_declaration())
    signature = lowered.store.signature
    names = tuple(f"x{at}" for at in range(8))
    store = TermStore(signature)
    there = _seeded(signature, "Tm", names)
    drawn = st.one_of(
        [
            terms(store, sort=sort, max_size=7, closed=closed)
            for sort in signature.sorts
            for closed in (True, False)
        ]
    )
    fixture: TermFixture[Term, Operator, int] = TermFixture(
        terms=drawn,
        root=lambda term: term.op,
        arguments=lambda term: term.args,
        rebuild=lambda term, args: store.op(term.op, *args, payload=term.payload),
        core=lambda term: term.core,
        size=lambda term: term.size,
        open_indices=lambda term: dict(term.unbound),
        free=lambda term: term.free,
        shift=_shift,
        is_closed=lambda term: term.is_closed,
        transplant=_transplant(store, there),
        max_examples=100,
    )

    assert signature.sorts == ("Ty", "Tm", "Cat")
    assert signature.vocabulary("Tm") == lowered.store.signature.vocabulary("Tm")

    check_term_laws(fixture)


def test_the_packaged_synthetic_parser_trains_to_completed(tmp_path: Path) -> None:
    argv = [
        "run",
        "--config",
        str(PARSER),
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path),
    ]

    assert main(argv) == 0

    (directory,) = (tmp_path / "runs").glob("[0-9a-f]*")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["status"] == ManifestStatus.COMPLETED.value

"""Semantic types written in angle brackets, and gold derivations as term records.

The packaged synthetic-inference document is rewritten so that noun phrases
denote generalized quantifiers, once with its category-to-type map in angle
brackets and once in the type signature's s-expressions. Both documents lower
to the same interpretation and draw the same labeled pairs. A gold derivation
the generator draws is then published with every category as a term record,
read back into a grammar lowered from a document that declares its primitive
categories in another order, and compared with the original.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from lofi.data import Derivation, DerivationNode, TokenSpan
from lofi.data.synthetic.spec.inference import (
    SyntheticInferenceDecl,
    SyntheticInferenceLowerer,
)
from lofi.data.synthetic.spec.interpretation import lower_interpretation
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import DeclarationError, SpecLoader
from lofi.spec.grammar import GrammarIso, SignatureMorphismErasureSpec
from lofi.spec.language import LanguageIso
from lofi.terms import from_record, record_from_json, record_to_json, to_record

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data.synthetic import InterpretationFunction, SyntheticInference
    from lofi.spec.grammar import GrammarAssembly

INFERENCE: Final = (
    Path(__file__).resolve().parents[2]
    / "lofi/configs/data/datasets/train/dataset/synthetic-inference.yaml"
)
ENTITY_MEANINGS: Final = """\
      - {form: john, category: NP, meaning: john}
      - {form: mary, category: NP, meaning: mary}
      - {form: sleeps, category: S\\NP, meaning: '\\x. sleep(x)'}
      - {form: snores, category: S\\NP, meaning: '\\x. (sleep(x) & snore(x))'}
      - {form: wakes, category: S\\NP, meaning: '\\x. not(sleep(x))'}
      - {form: loves, category: (S\\NP)/NP, meaning: '\\y. \\x. love(x)(y)'}
"""
QUANTIFIER_MEANINGS: Final = """\
      - {form: john, category: NP, meaning: '\\P. P(john)'}
      - {form: mary, category: NP, meaning: '\\P. P(mary)'}
      - {form: sleeps, category: S\\NP, meaning: '\\Q. Q(\\x. sleep(x))'}
      - {form: snores, category: S\\NP, meaning: '\\Q. Q(\\x. (sleep(x) & snore(x)))'}
      - {form: wakes, category: S\\NP, meaning: '\\Q. Q(\\x. not(sleep(x)))'}
      - form: loves
        category: (S\\NP)/NP
        meaning: '\\Y. \\X. Y(\\y. X(\\x. love(x)(y)))'
"""
ENTITY_TYPES: Final = """\
  erasure:
    kind: signature-morphism
    atoms:
      S: t
      NP: e
      conj: e
"""
ANGLE_TYPES: Final = """\
  erasure:
    kind: signature-morphism
    atoms:
      S: t
      NP: ⟨⟨e,t⟩,t⟩
      conj: e
"""
PRIMITIVES: Final = "primitives: [S, NP, conj]"
REORDERED_PRIMITIVES: Final = "primitives: [conj, NP, S]"
SEXP_TYPES: Final = """\
  type_syntax: s-expression
  erasure:
    kind: signature-morphism
    atoms:
      S: '("#const[Ty]" "t")'
      NP: '("->" ("->" ("#const[Ty]" "e") ("#const[Ty]" "t")) ("#const[Ty]" "t"))'
      conj: '("#const[Ty]" "e")'
"""


def _document(
    directory: Path, name: str, types: str, primitives: str = PRIMITIVES
) -> SyntheticInferenceDecl:
    """Write the packaged document with quantifier meanings, types and primitives."""
    packaged = INFERENCE.read_text(encoding="utf-8")

    assert ENTITY_TYPES in packaged
    assert ENTITY_MEANINGS in packaged
    assert PRIMITIVES in packaged

    path = directory / name
    path.write_text(
        packaged.replace(ENTITY_TYPES, types)
        .replace(ENTITY_MEANINGS, QUANTIFIER_MEANINGS)
        .replace(PRIMITIVES, primitives),
        encoding="utf-8",
        newline="\n",
    )

    return SpecLoader().compose_traced(path, schema=SyntheticInferenceDecl).spec


def _interpretation(declaration: SyntheticInferenceDecl) -> InterpretationFunction:
    """Lower a document's interpretation over its compiled grammar."""
    _, interpretation = lower_interpretation(
        declaration.grammar,
        declaration.source.language,
        declaration.source.interpretation,
        GrammarIso(),
        LanguageIso(),
    )

    return interpretation


def _drawn(run: SyntheticInference) -> tuple[tuple[str, str, str], ...]:
    """Return every drawn pair as its premise, hypothesis and label."""
    return tuple(
        (pair.premise.text, pair.hypothesis.text, str(label))
        for pair, label in zip(run.pairs, run.labels, strict=True)
    )


def test_angle_bracket_types_lower_as_their_s_expression_spelling(
    tmp_path: Path,
) -> None:
    angle = _document(tmp_path, "angle.yaml", ANGLE_TYPES)
    sexp = _document(tmp_path, "sexp.yaml", SEXP_TYPES)
    here = _interpretation(angle)
    there = _interpretation(sexp)
    entries = {
        (form, category.key()): tuple(meaning.key() for meaning in meanings)
        for (form, category), meanings in here.entries.items()
    }

    assert isinstance(angle.source.interpretation.erasure, SignatureMorphismErasureSpec)
    assert angle.source.interpretation.erasure.atoms["NP"] == "⟨⟨e,t⟩,t⟩"
    assert entries == {
        (form, category.key()): tuple(meaning.key() for meaning in meanings)
        for (form, category), meanings in there.entries.items()
    }
    assert set(here.actions.rules) == set(there.actions.rules)

    lowerer = SyntheticInferenceLowerer()
    policy = ParsePolicy(Retain.FOREST, Reachability.ALL)
    drawn = _drawn(lowerer.lower(angle, policy))

    assert drawn
    assert drawn == _drawn(lowerer.lower(sexp, policy))


def test_a_type_that_does_not_read_is_refused_naming_the_grammar(
    tmp_path: Path,
) -> None:
    misread = _document(
        tmp_path, "misread.yaml", ANGLE_TYPES.replace("⟨⟨e,t⟩", "⟨⟨e,w⟩")
    )
    reversed_pair = _document(
        tmp_path, "reversed.yaml", ANGLE_TYPES.replace("⟨⟨e,t⟩,t⟩", "⟨t,⟨e,t⟩⟩")
    )

    with pytest.raises(DeclarationError, match="'w'"):
        _interpretation(misread)

    with pytest.raises(DeclarationError, match="neo-davidsonian-inference"):
        _interpretation(reversed_pair)


def _published(derivation: Derivation[str], grammar: GrammarAssembly) -> JsonValue:
    """Return a gold derivation as JSON with every category as a term record."""
    store, syntax = grammar.store, grammar.syntax

    return [
        {
            "id": node_id,
            "span": [node.span.start, node.span.end],
            "rule": node.rule,
            "children": list(node.children),
            "category": record_to_json(
                to_record(store, syntax.parse(store, node.label))
            ),
        }
        for node_id, node in derivation.nodes.items()
    ] + [{"root": derivation.root}]


def _read(published: JsonValue, grammar: GrammarAssembly) -> Derivation[str]:
    """Return the gold derivation a published form describes, over another grammar."""
    assert isinstance(published, list)

    store, syntax = grammar.store, grammar.syntax
    nodes: dict[int, DerivationNode[str]] = {}
    root = -1

    for entry in published:
        assert isinstance(entry, dict)

        if "root" in entry:
            assert isinstance(entry["root"], int)
            root = entry["root"]
            continue

        node_id, span, rule, children = (
            entry["id"],
            entry["span"],
            entry["rule"],
            entry["children"],
        )

        assert isinstance(node_id, int)
        assert isinstance(span, list)
        assert isinstance(children, list)
        assert rule is None or isinstance(rule, str)

        start, end = (int(str(bound)) for bound in span)
        category = from_record(store, record_from_json(entry["category"]))
        nodes[node_id] = DerivationNode(
            TokenSpan(start, end),
            syntax.render(store, category),
            rule=rule,
            children=tuple(int(str(child)) for child in children),
        )

    return Derivation(root, nodes)


def test_a_gold_derivation_round_trips_through_term_records(tmp_path: Path) -> None:
    declaration = _document(tmp_path, "angle.yaml", ANGLE_TYPES)
    run = SyntheticInferenceLowerer().lower(
        declaration, ParsePolicy(Retain.FOREST, Reachability.ALL)
    )
    reordered = _document(tmp_path, "reordered.yaml", ANGLE_TYPES, REORDERED_PRIMITIVES)
    publisher = GrammarIso().forward(declaration.grammar, declaration.source.language)
    reader = GrammarIso().forward(reordered.grammar, reordered.source.language)
    derivations = [
        example.supervision.derivation
        for corpus in (run.premises, run.hypotheses)
        for example in corpus.examples
    ]

    assert publisher.store is not reader.store
    assert publisher.store.signature.vocabulary("Cat").id(
        "S"
    ) != reader.store.signature.vocabulary("Cat").id("S")
    assert any(len(derivation) > 1 for derivation in derivations)

    for derivation in derivations:
        published = json.loads(json.dumps(_published(derivation, publisher)))
        read = _read(published, reader)

        assert read == derivation
        assert [read.node(node).rule for node in read.preorder] == [
            derivation.node(node).rule for node in derivation.preorder
        ]

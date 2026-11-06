"""A treebank grammar's observed unary coercions declared as one coercion set.

The grammar directory is the parse-cost reproduction structural-probes ships:
``grammar.json`` holds a CCGbank-derived calculus with its ground unary
coercions as ``[premise, conclusion]`` pairs in surface category syntax,
``lexicon.json`` the categories of every form the timed sentences use, and
``sentences.json`` the sentences. ``LOFI_SURFACE_COERCIONS`` names the
directory. Its contents derive from CCGbank, so they are read where they lie
rather than committed.

The test declares the grammar as a document: its primitives, the slash pair,
its conjunction inventory, application, the CCGbank exceptional rules that
enumerate forward, which grammar lowering requires, and every coercion as one
``"premise => conclusion"`` entry of one coercion set. Every sentence then has
more derivations than it has under the same grammar without the set.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lofi.data.synthetic import SyntheticGenerator
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import ClausalAdjunctSpec, CoercionSetSpec, CoordinationSpec
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.grammar import GrammarDecl, GrammarIso
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from lofi._types import JsonValue


@pytest.fixture(scope="module")
def grammar_root() -> Path:
    """Return the reproduction directory or skip the module."""
    named = os.environ.get("LOFI_SURFACE_COERCIONS")

    if named is None:
        pytest.skip("set LOFI_SURFACE_COERCIONS to a parse-cost grammar directory")

    root = Path(named)

    if not (root / "grammar.json").is_file():
        pytest.skip(f"LOFI_SURFACE_COERCIONS names {root}, which has no grammar.json")

    return root


def _read(root: Path, name: str) -> JsonValue:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _forward_rules() -> list[JsonValue]:
    """Return the exceptional rules whose runtime enumerates conclusions."""
    return [
        json.loads(rule.model_dump_json())
        for rule in ccgbank_grammar().rules.rules
        if not isinstance(rule, ClausalAdjunctSpec)
        and not (
            isinstance(rule, CoordinationSpec) and rule.core_relation == "different"
        )
    ]


def _document(root: Path) -> str:
    grammar = _read(root, "grammar.json")
    lexicon = _read(root, "lexicon.json")
    assert isinstance(grammar, dict)
    assert isinstance(lexicon, dict)
    calculus = grammar["calculus"]
    adequacy = grammar["adequacy"]
    primitives = grammar["primitives"]
    assert isinstance(calculus, dict)
    assert isinstance(adequacy, dict)
    assert isinstance(primitives, list)
    coercions = calculus["coercions"]
    pairs = calculus["coordination_pairs"]
    assert isinstance(coercions, list)
    assert isinstance(pairs, list)
    featured = [str(name) for name in primitives if "[" in str(name)]

    return json.dumps(
        {
            "name": "surface-coercions",
            "signature": {
                "primitives": [name for name in primitives if "[" not in str(name)],
                "operators": [
                    {"name": name, "sort": "Cat", "erases_to": name.split("[")[0]}
                    for name in featured
                ],
                "constructors": {
                    "/": {"functor": {"direction": "forward"}},
                    "\\": {"functor": {"direction": "backward"}},
                },
                "conjunctions": {
                    "markers": calculus["coordination_markers"],
                    "pairs": [
                        {"unmarked": unmarked, "marked": marked}
                        for unmarked, marked in (
                            pair for pair in pairs if isinstance(pair, list)
                        )
                    ],
                },
            },
            "rules": {
                "name": "surface-coercions",
                "rules": [
                    {"rule": "application", "view": "/"},
                    {"rule": "application", "view": "\\"},
                    *_forward_rules(),
                    {
                        "rule": "coercions",
                        "label": "observed-unary",
                        "entries": [
                            f"{premise} => {conclusion}"
                            for premise, conclusion in (
                                pair for pair in coercions if isinstance(pair, list)
                            )
                        ],
                    },
                ],
            },
            "lexicon": {
                "kind": "inline",
                "entries": [
                    {"form": form, "category": category}
                    for form, categories in lexicon.items()
                    if isinstance(categories, list)
                    for category in categories
                ],
            },
            "goal": {"kind": "root", "categories": adequacy["roots"]},
            "matching": {"kind": "feature-erasure"},
            "budget": {"unary_rounds": adequacy["unary_depth"]},
        }
    )


@pytest.mark.corpus
@pytest.mark.integration
def test_every_observed_coercion_is_one_entry_and_every_sentence_parses(
    grammar_root: Path,
) -> None:
    grammar = _read(grammar_root, "grammar.json")
    sentences = _read(grammar_root, "sentences.json")
    assert isinstance(grammar, dict)
    assert isinstance(sentences, list)
    calculus = grammar["calculus"]
    assert isinstance(calculus, dict)
    observed = calculus["coercions"]
    assert isinstance(observed, list)

    declaration = GrammarDecl.model_validate_json(_document(grammar_root))
    (coercions,) = (
        rule for rule in declaration.rules.rules if isinstance(rule, CoercionSetSpec)
    )
    uncoerced = declaration.with_(
        rules=declaration.rules.with_(
            rules=tuple(rule for rule in declaration.rules.rules if rule != coercions)
        )
    )
    language = registered_language("propositional")
    assembly = GrammarIso().forward(declaration, language)
    names = [rule.name for rule in assembly.grammar.rules.rules]
    coerced = [name for name in names if name.startswith(f"{coercions.label}:")]
    tokens = [
        tuple(str(token) for token in sentence["tokens"])
        for sentence in sentences
        if isinstance(sentence, dict) and isinstance(sentence["tokens"], list)
    ]
    policy = ParsePolicy(Retain.FOREST, Reachability.ALL)
    counts = [
        SyntheticGenerator(assembly, policy=policy).derivation_count(forms)
        for forms in tokens
    ]
    baseline = SyntheticGenerator(
        GrammarIso().forward(uncoerced, language), policy=policy
    )

    assert len(coercions.entries) == len(observed)
    assert len(coerced) == len(observed)
    assert len(tokens) == len(sentences)
    assert all(
        count > baseline.derivation_count(forms)
        for count, forms in zip(counts, tokens, strict=True)
    )

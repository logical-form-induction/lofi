"""Third-party rule plugins compose through the published rule context."""

import json
import os
import subprocess
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.data.synthetic import SyntheticGenerator
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.judge import ProofSchema
from lofi.run import Selection, compose, plan_training
from lofi.spec import (
    CombinatorFamilySpec,
    DeclarationError,
    RunSpec,
    config_search_path,
)
from lofi.spec.grammar import GrammarDecl, GrammarIso, declare_grammar
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl


FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _composed(root: Path, text: str) -> RunSpec:
    document = root / "run.yaml"
    document.write_text(text, encoding="utf-8", newline="\n")

    return compose(
        Selection(
            config=document,
            profile=None,
            overrides=(),
            overlays=(),
            search_path=config_search_path(root),
        )
    )


def _write_plugin(root: Path, module: str) -> None:
    source = """from typing import Literal

import didactic.api as dx

from lofi._types import JsonValue
from lofi.spec import NormalFormRoleSpec, PatternSpec, RulePlugin, RuleSpec
from lofi.spec._ccg_rules import _adopt_pattern


class TheoryTerm(dx.Model, extra="forbid"):
    pattern: Literal["term"] = "term"
    name: str
    sort: str


class TheorySchema(dx.Model, extra="forbid"):
    premises: tuple[TheoryTerm, ...]
    conclusion: TheoryTerm


class TheorySpec(dx.Model, extra="forbid"):
    kind: Literal["schemata"] = "schemata"
    name: str
    forward_complete: bool
    schemata: tuple[TheorySchema, ...]


class ThirdPartyIdentitySpec(RuleSpec, extra="forbid"):
    rule: Literal["third-party-identity"] = "third-party-identity"
    head_policy: Literal["pass-through"] = "pass-through"
    name: str = "third-party-identity"
    premises: tuple[PatternSpec, ...]
    conclusion: PatternSpec
    targets: tuple[str, ...] | None = None
    normal_form_role: NormalFormRoleSpec = dx.field(
        default_factory=NormalFormRoleSpec
    )
    runtime: dict[str, JsonValue]
    theory: TheorySpec


PLUGIN = RulePlugin(
    spec=ThirdPartyIdentitySpec,
    adopt=_adopt_pattern,
)
"""
    (root / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")
    metadata = root / "lofi_rule_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-rule-plugin\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.rule]\nthird-party-identity = {module}:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )


def test_a_third_party_rule_plugin_loads_from_a_document_and_builds(
    tmp_path: Path,
) -> None:
    module = f"lofi_rule_plugin_{tmp_path.name.replace('-', '_')}"
    _write_plugin(tmp_path, module)
    script = r"""from lofi.spec import (
    ConstructorSpec,
    InfixConstructorNotationSpec,
    RuleSetSpec,
    load_plugins,
)
from lofi.spec.grammar import (
    FunctorSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

loaded = load_plugins()
assert "third-party-identity" in loaded

signature_spec = SignatureSpec(
    primitives=("N",),
    constructors=(
        ConstructorSpec(
            name="/",
            kind=("Cat", "Cat", "Cat"),
            notation=InfixConstructorNotationSpec(
                spelling="/", precedence=50, associativity="left"
            ),
        ),
        ConstructorSpec(
            name="\\",
            kind=("Cat", "Cat", "Cat"),
            notation=InfixConstructorNotationSpec(
                spelling="\\", precedence=50, associativity="left"
            ),
        ),
    ),
    functors=(
        FunctorSpec(
            name="/", constructor="/", argument_side="right", opposite="\\"
        ),
        FunctorSpec(
            name="\\", constructor="\\", argument_side="left", opposite="/"
        ),
    ),
)
declaration = RuleSetSpec.model_validate({
    "name": "external rules",
    "rules": [{
        "rule": "third-party-identity",
        "name": "external identity",
        "premises": [{"pattern": "term", "name": "X", "sort": "Cat"}],
        "conclusion": {"pattern": "term", "name": "X", "sort": "Cat"},
        "runtime": {"kind": "pattern", "head_slot": 0},
        "theory": {
            "kind": "schemata",
            "name": "external identity",
            "forward_complete": True,
            "schemata": [{
                "premises": [{"pattern": "term", "name": "X", "sort": "Cat"}],
                "conclusion": {"pattern": "term", "name": "X", "sort": "Cat"},
            }],
        },
    }],
})
grammar = GrammarIso().forward(
    GrammarDecl(
        name="external grammar",
        signature=signature_spec,
        rules=declaration,
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="fixture", category="N"),)
        ),
        goal=RootGoalSpec(categories=("N",)),
    ),
    registered_language("propositional"),
)
external = grammar.grammar.rules.rules[0]
n = grammar.syntax.parse(grammar.store, "N")

assert type(declaration.rules[0]).__name__ == "ThirdPartyIdentitySpec"
assert type(external).__name__ == "PatternRule"
assert external.name == "external identity"
assert tuple(external.apply((n,))) == (n,)
print("third-party plugin built")
"""
    python_path = os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-"],
        input=script,
        check=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": python_path},
    )

    assert completed.stdout == "third-party plugin built\n"
    assert module not in sys.modules


def _write_combinator_plugin(root: Path, module: str) -> None:
    source = """from typing import Literal

import didactic.api as dx

from lofi._types import JsonValue
from lofi.spec import NormalFormRoleSpec, PatternSpec, RulePlugin, RuleSpec
from lofi.spec._ccg_rules import _adopt_pattern


class TheorySchema(dx.Model, extra="forbid"):
    premises: tuple[PatternSpec, ...]
    conclusion: PatternSpec


class TheorySpec(dx.Model, extra="forbid"):
    kind: Literal["schemata"] = "schemata"
    name: str
    forward_complete: bool
    schemata: tuple[TheorySchema, ...]


class ThirdPartyApplicationSpec(RuleSpec, extra="forbid"):
    rule: Literal["third-party-application"] = "third-party-application"
    head_policy: Literal["functor"] = "functor"
    name: str
    view: str
    premises: tuple[PatternSpec, ...]
    conclusion: PatternSpec
    targets: tuple[str, ...] | None = None
    normal_form_role: NormalFormRoleSpec = dx.field(
        default_factory=NormalFormRoleSpec
    )
    runtime: dict[str, JsonValue]
    theory: TheorySpec


PLUGIN = RulePlugin(
    spec=ThirdPartyApplicationSpec,
    adopt=_adopt_pattern,
)
"""
    (root / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")
    metadata = root / "lofi_combinator_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-combinator-plugin\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.rule]\nthird-party-application = {module}:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )


def test_a_third_party_combinator_kind_selected_by_a_run_document_draws_data(
    tmp_path: Path,
) -> None:
    module = f"lofi_combinator_plugin_{tmp_path.name.replace('-', '_')}"
    _write_combinator_plugin(tmp_path, module)
    (tmp_path / "run.yaml").write_text(
        dedent(
            r"""
            defaults:
              - run/synthetic-parser
            data:
              datasets:
                train:
                  dataset:
                    grammar:
                      rules:
                        name: third-party
                        rules:
                          - rule: third-party-application
                            name: 'application:/'
                            view: /
                            premises:
                              - pattern: node
                                operator: {kind: fixed, name: /}
                                arguments:
                                  - {pattern: term, name: X, sort: Cat}
                                  - {pattern: term, name: Y, sort: Cat}
                              - {pattern: term, name: Y, sort: Cat}
                            conclusion: {pattern: term, name: X, sort: Cat}
                            runtime: {kind: pattern, head_slot: 0}
                            theory:
                              kind: schemata
                              name: 'application:/'
                              forward_complete: true
                              schemata:
                                - premises:
                                    - pattern: node
                                      operator: {kind: fixed, name: /}
                                      arguments:
                                        - {pattern: term, name: X, sort: Cat}
                                        - {pattern: term, name: Y, sort: Cat}
                                    - {pattern: term, name: Y, sort: Cat}
                                  conclusion: {pattern: term, name: X, sort: Cat}
                          - rule: third-party-application
                            name: 'application:\'
                            view: '\'
                            premises:
                              - {pattern: term, name: Y, sort: Cat}
                              - pattern: node
                                operator: {kind: fixed, name: '\'}
                                arguments:
                                  - {pattern: term, name: X, sort: Cat}
                                  - {pattern: term, name: Y, sort: Cat}
                            conclusion: {pattern: term, name: X, sort: Cat}
                            runtime: {kind: pattern, head_slot: 1}
                            theory:
                              kind: schemata
                              name: 'application:\'
                              forward_complete: true
                              schemata:
                                - premises:
                                    - {pattern: term, name: Y, sort: Cat}
                                    - pattern: node
                                      operator: {kind: fixed, name: '\'}
                                      arguments:
                                        - {pattern: term, name: X, sort: Cat}
                                        - {pattern: term, name: Y, sort: Cat}
                                  conclusion: {pattern: term, name: X, sort: Cat}
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    script = r"""import sys
from pathlib import Path
from typing import cast

from lofi.data.synthetic import SyntheticDataset
from lofi.run import Selection, compose, plan_training
from lofi.spec import config_search_path, load_plugins

root = Path(sys.argv[1])
loaded = load_plugins()
assert "third-party-application" in loaded

run_spec = compose(
    Selection(
        config=root / "run.yaml",
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(root),
    )
)
declared = run_spec.data.datasets["train"].dataset.grammar.rules.rules
assert {type(rule).__name__ for rule in declared} == {"ThirdPartyApplicationSpec"}

dataset = cast("SyntheticDataset", plan_training(run_spec).lower_dataset("train"))
built = {rule.name for rule in dataset.source_grammar.grammar.rules}
used = {
    node.rule
    for example in dataset.corpus.examples
    for node in example.supervision.derivation.nodes.values()
    if node.rule is not None
}

assert len(dataset.corpus.examples) == run_spec.data.datasets["train"].dataset.size
assert built == {"application:/", "application:\\"}
assert used and used <= built
print("third-party combinator drew data")
"""
    python_path = os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=script,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": python_path},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "third-party combinator drew data\n"
    assert module not in sys.modules


def test_composition_restrictions_give_distinct_rule_names_and_decoder_keys(
    tmp_path: Path,
) -> None:
    steedman = "composition:/:2:harmonic:steedman"
    kuhlmann = "composition:/:2:harmonic:kuhlmann"
    run_spec = _composed(
        tmp_path,
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "model:\n"
        "  spaces:\n"
        "    categories:\n"
        "      combinators:\n"
        "        decoders:\n"
        f'          "{steedman}": application\n'
        f'          "{kuhlmann}": application\n'
        "data:\n"
        "  datasets:\n"
        "    train:\n"
        "      dataset:\n"
        "        grammar:\n"
        "          rules:\n"
        "            rules+:\n"
        "              - {rule: composition, direction: steedman, order: 2, view: /}\n"
        "              - {rule: composition, direction: kuhlmann, order: 2, view: /}\n",
    )
    blueprint = plan_training(run_spec)
    names = tuple(action.name for action in blueprint.actions.actions)
    family = run_spec.model.spaces["categories"].combinators

    assert names[-2:] == (steedman, kuhlmann)
    assert isinstance(family, CombinatorFamilySpec)
    assert {steedman, kuhlmann} <= set(family.decoders)


def test_a_pattern_variable_of_an_undeclared_sort_is_refused_naming_the_sort(
    tmp_path: Path,
) -> None:
    run = _composed(
        tmp_path,
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "data:\n"
        "  datasets:\n"
        "    train:\n"
        "      dataset:\n"
        "        grammar:\n"
        "          rules:\n"
        "            rules+:\n"
        "              - rule: pattern\n"
        "                name: unsorted\n"
        "                premises: [{pattern: term, name: X, sort: Nonexistent}]\n"
        "                conclusion: {pattern: term, name: X, sort: Nonexistent}\n",
    )
    dataset = cast("SyntheticCorpusDecl", run.data.datasets["train"].dataset)

    with pytest.raises(DeclarationError, match='unknown sort "Nonexistent"'):
        GrammarIso().forward(declare_grammar(dataset.grammar), dataset.language)


def test_a_document_declaring_bounded_and_unbounded_spine_rules_plans_both(
    tmp_path: Path,
) -> None:
    def document(bound: str) -> str:
        spine = (
            "{pattern: spine, name: C, operators: [/, '\\\\'], "
            f"core: {{pattern: term, name: CORE}}, min_order: 1, max_order: {bound}}}"
        )
        result = spine.replace("name: CORE", "name: X")
        argument = spine.replace("name: CORE", "name: Y")

        return (
            "defaults:\n"
            "  - run/synthetic-parser\n"
            "model:\n"
            "  spaces:\n"
            "    categories:\n"
            "      combinators:\n"
            "        decoders:\n"
            "          lifted-application: application\n"
            "data:\n"
            "  datasets:\n"
            "    train:\n"
            "      dataset:\n"
            "        grammar:\n"
            "          rules:\n"
            "            rules+:\n"
            "              - rule: pattern\n"
            "                name: lifted-application\n"
            "                premises:\n"
            "                  - {pattern: node, operator: {kind: fixed, name: /}, "
            "arguments: [{pattern: term, name: X}, {pattern: term, name: Y}]}\n"
            f"                  - {argument}\n"
            f"                conclusion: {result}\n"
        )

    bounded = plan_training(_composed(tmp_path, document("4")))
    unbounded = plan_training(_composed(tmp_path, document("null")))

    for blueprint in (bounded, unbounded):
        assert "lifted-application" in {
            action.name for action in blueprint.actions.actions
        }


def _lifting_grammar(bound: int) -> GrammarDecl:
    """Return a grammar whose one sentence needs backward lifting at order four."""
    context = {
        "pattern": "spine",
        "name": "C",
        "operators": ["/"],
        "min_order": 1,
        "max_order": bound,
    }

    return GrammarDecl.model_validate_json(
        json.dumps(
            {
                "name": "deep-lifting",
                "signature": {
                    "primitives": ["S", "NP", "A"],
                    "constructors": [
                        {
                            "name": "/",
                            "kind": ["Cat", "Cat", "Cat"],
                            "notation": {
                                "style": "infix",
                                "spelling": "/",
                                "precedence": 50,
                                "associativity": "left",
                            },
                        },
                        {
                            "name": "\\",
                            "kind": ["Cat", "Cat", "Cat"],
                            "notation": {
                                "style": "infix",
                                "spelling": "\\",
                                "precedence": 50,
                                "associativity": "left",
                            },
                        },
                    ],
                    "functors": [
                        {
                            "name": "/",
                            "constructor": "/",
                            "argument_side": "right",
                            "opposite": "\\",
                        },
                        {
                            "name": "\\",
                            "constructor": "\\",
                            "argument_side": "left",
                            "opposite": "/",
                        },
                    ],
                },
                "rules": {
                    "name": "lifted-application",
                    "rules": [
                        {"rule": "application", "view": "/"},
                        {"rule": "application", "view": "\\"},
                        {
                            "rule": "pattern",
                            "name": "lifted-backward-application",
                            "head_policy": "non-absorbing",
                            "premises": [
                                {**context, "core": {"pattern": "term", "name": "Y"}},
                                {
                                    "pattern": "node",
                                    "operator": {"kind": "fixed", "name": "\\"},
                                    "arguments": [
                                        {"pattern": "term", "name": "X"},
                                        {"pattern": "term", "name": "Y"},
                                    ],
                                },
                            ],
                            "conclusion": {
                                **context,
                                "core": {"pattern": "term", "name": "X"},
                            },
                        },
                    ],
                },
                "lexicon": {
                    "kind": "inline",
                    "entries": [
                        {"form": "deep", "category": "(((NP/A)/A)/A)/A"},
                        {"form": "verb", "category": "S\\NP"},
                        {"form": "arg", "category": "A"},
                    ],
                },
                "goal": {"kind": "root", "categories": ["S"]},
                "matching": {"kind": "identity"},
                "budget": {"closure_rounds": 6},
            }
        )
    )


def test_a_spine_rule_bounded_above_order_three_derives_an_order_four_sentence() -> (
    None
):
    forms = ("deep", "verb", "arg", "arg", "arg", "arg")
    base_language = registered_language("propositional")
    language = base_language.with_(
        bounds={
            **base_language.bounds,
            base_language.category_sort: base_language.bounds[
                base_language.category_sort
            ].with_(depth=5),
        }
    )
    deep = GrammarIso().forward(_lifting_grammar(4), language)
    shallow = GrammarIso().forward(_lifting_grammar(3), language)
    (lifting,) = (
        rule
        for rule in deep.grammar.rules.rules
        if rule.name == "lifted-backward-application"
    )

    assert SyntheticGenerator(deep, policy=FOREST_POLICY).derivation_count(forms) == 1
    assert (
        SyntheticGenerator(shallow, policy=FOREST_POLICY).derivation_count(forms) == 0
    )
    assert isinstance(lifting, ProofSchema)
    assert len(lifting.proof_schemata) == 4

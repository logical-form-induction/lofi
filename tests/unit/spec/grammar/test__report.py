import json
from pathlib import Path

import pytest
from tests.unit.spec.grammar.test__lowering import _ccg

from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
from lofi.run import plan_training
from lofi.spec import (
    ConstructorSpec,
    DeclarationError,
    PrefixConstructorNotationSpec,
    RunSpecCodec,
    config_search_path,
)
from lofi.spec.grammar import GrammarIso, RootGoalSpec
from lofi.spec.language import registered_language

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "lofi/configs/run/synthetic-parser.yaml"


def test_the_compile_report_names_the_compiled_grammar_structure() -> None:
    run = RunSpecCodec(search_path=config_search_path(ROOT))(RUN)
    blueprint = plan_training(run)
    assembly = blueprint.learner.grammar
    report = assembly.compile_report
    measurements = report.measurements()

    assert measurements["congruence"] == assembly.grammar.matching.name
    assert (
        measurements["codec-slots"]
        == assembly.language.codec.at(assembly.language.category_sort).capacity
    )
    assert measurements["closure-category-set"] in {"finite", "infinite", "unknown"}
    assert json.loads(str(measurements["operator-codes"]))["#const[Cat]"] > 0
    assert json.loads(str(measurements["rule-directions"])) == {
        rule.name: sorted(rule.directions) for rule in assembly.grammar.rules
    }
    assert isinstance(json.loads(str(measurements["closure-never-applicable"])), list)
    assert json.loads(str(measurements["capacity-hypotheses"]))["result"] == (
        assembly.grammar.capacity.result
    )


def test_the_compile_report_exposes_unreached_arity_diagnostics() -> None:
    source = _ccg()
    signature = source.signature.with_(
        constructors=(
            *source.signature.constructors,
            ConstructorSpec(
                name="ternary",
                kind=("Cat", "Cat", "Cat", "Cat"),
                notation=PrefixConstructorNotationSpec(spelling="ternary"),
            ),
        )
    )
    assembly = GrammarIso().forward(
        source.with_(signature=signature),
        registered_language("propositional"),
    )
    diagnostics = json.loads(str(assembly.compile_report.measurements()["diagnostics"]))

    assert diagnostics == ["E310: operator 'ternary' declares unreached arity 3"]
    assert assembly.compile_report.measurements()["codec-branching"] == 2


def test_a_run_refusal_names_the_full_dataset_configuration_path() -> None:
    run = RunSpecCodec(search_path=config_search_path(ROOT))(RUN)
    datasets = dict(run.data.datasets)
    section = datasets["train"]
    dataset = section.dataset
    assert isinstance(dataset, SyntheticCorpusDecl)
    grammar = dataset.grammar.with_(goal=RootGoalSpec(categories=("Q",)))
    datasets["train"] = section.with_(dataset=dataset.with_(grammar=grammar))
    invalid = run.with_(data=run.data.with_(datasets=datasets))

    with pytest.raises(DeclarationError) as caught:
        plan_training(invalid)

    assert caught.value.path == ("data.datasets.train.dataset.grammar.goal.categories")

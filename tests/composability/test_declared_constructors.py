import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from didactic.settings import compose, load_document

from lofi.data.synthetic import SyntheticGenerator, UniformLength
from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.run import (
    CHECKPOINTS_NAME,
    MetricsRecorder,
    RandomStreams,
    SeedPolicy,
    SweepPoint,
    operations,
    plan_training,
    read_metrics,
    resolve_device_policy,
    section_root,
)
from lofi.spec import DataSection, RunSpec, RunSpecCodec, config_search_path
from lofi.spec._plugins import load_plugins
from lofi.spec.grammar import GrammarDecl, GrammarIso
from lofi.spec.language import LanguageDecl, registered_language
from lofi.train import DirectoryStore

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.terms import Term

ROOT: Final = Path(__file__).resolve().parents[2]
RUN: Final = ROOT / "lofi/configs/run/synthetic-parser.yaml"
GRAMMARS: Final = Path(__file__).parent / "fixtures" / "constructors"
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


@pytest.fixture(autouse=True)
def _declarations() -> None:
    """Make every packaged component declaration available to validation."""
    load_plugins()


def _document(path: Path) -> dict[str, JsonValue]:
    """Return a configuration document as JSON data."""
    return cast("dict[str, JsonValue]", load_document(path))


def _grammar(name: str) -> GrammarDecl:
    """Return the grammar a constructor document declares."""
    return GrammarDecl.model_validate(_document(GRAMMARS / f"{name}.yaml"))


def _run_document() -> dict[str, JsonValue]:
    """Return the composed packaged parser run as mutable JSON data."""
    run = RunSpecCodec(search_path=config_search_path(ROOT))(RUN)

    return cast("dict[str, JsonValue]", json.loads(run.model_dump_json()))


def _packaged_directed() -> dict[str, JsonValue]:
    """Return the packaged directed grammar composed from its group fragments."""
    section = compose(
        None,
        schema=DataSection,
        groups={
            "datasets.train.dataset": "synthetic",
            "datasets.train.featurizer": "unlabeled",
            "datasets.train.dataset.grammar": "directed",
        },
        search_path=(section_root("data"),),
    )
    dataset = section.datasets["train"].dataset

    assert isinstance(dataset, SyntheticCorpusDecl)

    return cast("dict[str, JsonValue]", json.loads(dataset.grammar.model_dump_json()))


def _run(name: str) -> RunSpec:
    """Return the packaged parser run over a constructor document's grammar."""
    return _run_over(_document(GRAMMARS / f"{name}.yaml"))


def _run_over(grammar: dict[str, JsonValue]) -> RunSpec:
    """Return the packaged parser run over a declared grammar.

    Every combinator the grammar builds is decoded by the run's one decoder.
    """
    document = _run_document()
    data = document["data"]
    model = document["model"]

    assert isinstance(data, dict)
    assert isinstance(model, dict)

    datasets = data["datasets"]
    spaces = model["spaces"]

    assert isinstance(datasets, dict)
    assert isinstance(spaces, dict)

    training = datasets["train"]

    assert isinstance(training, dict)

    dataset = training["dataset"]

    assert isinstance(dataset, dict)

    declaration = GrammarDecl.model_validate(grammar)
    language_document = dataset["language"]

    assert isinstance(language_document, dict)

    language = LanguageDecl.model_validate(language_document)
    rules = GrammarIso().forward(declaration, language).grammar.rules
    dataset["grammar"] = {"formalism": "declared", **grammar}
    categories = spaces["categories"]

    assert isinstance(categories, dict)

    combinators = categories["combinators"]

    assert isinstance(combinators, dict)

    combinators["decoders"] = {rule.name: "application" for rule in rules}

    return RunSpec.model_validate(document)


@pytest.mark.parametrize(
    ("name", "constructors", "rules"),
    [
        (
            "c01",
            ("/", "\\"),
            ("application:/", "application:\\"),
        ),
        ("c03", ("arrow",), ("application:arrow",)),
    ],
)
def test_a_constructor_document_composes_its_functor_rules(
    name: str, constructors: tuple[str, ...], rules: tuple[str, ...]
) -> None:
    declaration = _grammar(name)
    grammar = (
        GrammarIso().forward(declaration, registered_language("propositional")).grammar
    )

    assert (
        tuple(constructor.name for constructor in declaration.signature.constructors)
        == constructors
    )
    assert tuple(rule.name for rule in grammar.rules) == rules


def test_the_undirected_document_draws_functors_on_either_side() -> None:
    directed = SyntheticGenerator(
        GrammarIso().forward(_grammar("c01"), registered_language("propositional")),
        policy=FOREST_POLICY,
    )
    undirected = SyntheticGenerator(
        GrammarIso().forward(_grammar("c03"), registered_language("propositional")),
        policy=FOREST_POLICY,
    )
    length = UniformLength(2, 3)

    def orders(
        generator: SyntheticGenerator,
    ) -> set[tuple[str, ...]]:
        corpus = generator.sample(40, seed=4, length=length)

        assert corpus.examples

        return {
            tuple(token.surface for token in example.item.tokens)
            for example in corpus.examples
            if len(example.item.tokens) == 2
        }

    assert orders(directed) == {("w0", "w1")}
    assert orders(undirected) == {("w0", "w1"), ("w1", "w0")}


def _train(run_spec: RunSpec, directory: Path) -> None:
    """Train one tiny point of a run into a directory."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()

    operations.TRAINING_OPERATIONS.execute(
        blueprint,
        directory,
        streams,
        resolve_device_policy(run_spec),
        scoring=None,
        sinks=(),
        recorder=MetricsRecorder(
            directory, SweepPoint(json.loads(blueprint.configuration))
        ),
    )


@pytest.mark.parametrize("name", ["c01", "c03"])
def test_a_constructor_document_trains_a_tiny_point(name: str, tmp_path: Path) -> None:
    _train(_run(name), tmp_path)

    assert DirectoryStore(tmp_path / CHECKPOINTS_NAME).list()


def test_the_packaged_directed_grammar_keeps_its_marks_and_trains(
    tmp_path: Path,
) -> None:
    run_spec = _run_over(_packaged_directed())
    dataset = cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset)
    grammar = dataset.grammar

    assert isinstance(grammar, GrammarDecl)

    built = GrammarIso().forward(grammar, dataset.language)
    matching = built.grammar.rules.matching

    def category(text: str) -> Term:
        return built.syntax.parse(built.store, text)

    assert not matching.equivalent(category("NP[conj]"), category("NP"))
    assert matching.equivalent(category("NP"), category("NP"))

    _train(run_spec, tmp_path)

    assert DirectoryStore(tmp_path / CHECKPOINTS_NAME).list()
    recorded = read_metrics(tmp_path)

    assert recorded is not None
    assert recorded.stage == "closed"

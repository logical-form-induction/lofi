"""Exercise the L2 source-logic comparison through its public boundaries."""

from __future__ import annotations

import shutil
from collections import Counter
from json import loads
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pytest

from lofi.data.synthetic.spec.inference import (
    SyntheticInferenceDecl,
    inference_fingerprint,
)
from lofi.lf import (
    CLASSICAL,
    Decided,
    InferenceLabel,
    Proof,
    Proved,
    SequentCalculus,
    SequentLimits,
)
from lofi.run import expand_experiment, read_experiment
from lofi.run.results import ingest_results
from lofi.spec import RunSpecCodec, config_search_path
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language
from lofi.terms import Normalization

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.lf import SequentSearch
    from lofi.spec import RunSpec
    from lofi.spec.language import LanguageAssembly


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
FRAGMENTS = (
    "quantifier-free",
    "universal-only",
    "existential-only",
    "second-order",
)


def _at(document: Mapping[str, JsonValue], path: str) -> JsonValue:
    value: JsonValue = dict(document)
    for part in path.split("."):
        if not isinstance(value, dict):
            msg = f"{path!r} passes through a scalar at {part!r}"
            raise TypeError(msg)
        value = value[part]
    return value


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _text_list_at(document: Mapping[str, JsonValue], path: str) -> tuple[str, ...]:
    value = _at(document, path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{path!r} is not a list of text"
        raise TypeError(msg)
    return tuple(item for item in value if isinstance(item, str))


def _document(run: RunSpec) -> Mapping[str, JsonValue]:
    return cast("Mapping[str, JsonValue]", loads(run.model_dump_json()))


def _quantifier_names_at(
    document: Mapping[str, JsonValue], path: str
) -> tuple[str, ...]:
    value = _at(document, path)
    if not isinstance(value, list):
        msg = f"{path!r} is not a list of quantifier declarations"
        raise TypeError(msg)
    names: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            msg = f"{path!r} contains an unnamed quantifier declaration"
            raise TypeError(msg)
        name = row.get("name")
        if not isinstance(name, str):
            msg = f"{path!r} contains an unnamed quantifier declaration"
            raise TypeError(msg)
        names.append(name)
    return tuple(names)


def _logic_run(fragment: str, *overrides: str) -> RunSpec:
    return RunSpecCodec(search_path=config_search_path(EXPERIMENTS))(
        EXPERIMENTS / "l2-run.yaml",
        overrides=(
            "data.datasets.train.dataset.seed=0",
            f"data/datasets/train/dataset/grammar={fragment}",
            (
                "data/datasets/train/dataset/source/interpretation/meanings/"
                f"source/structure={fragment}"
            ),
            f"model/language/grammar={fragment}",
            f"model/language/interpretation/meanings/source/structure={fragment}",
            f"comparison.source={fragment}",
            *overrides,
        ),
    )


@pytest.fixture(scope="module")
def logic_runs() -> dict[str, RunSpec]:
    """Return one fully composed run for every source/candidate fragment."""
    return {fragment: _logic_run(fragment) for fragment in FRAGMENTS}


def test_every_document_loads_and_the_small_factor_preflights_expand() -> None:
    """The one minimal comparison and both preflights declare their full grids."""
    comparison = read_experiment(EXPERIMENTS / "source-factors.yaml")
    vocabularies = expand_experiment(
        read_experiment(EXPERIMENTS / "vocabulary-size.yaml"), ROOT
    )
    decompositions = expand_experiment(
        read_experiment(EXPERIMENTS / "decompositions.yaml"), ROOT
    )
    structure = (
        "data.datasets.train.dataset.source.interpretation.meanings.source.structure"
    )

    _check(
        comparison.preregistration.grid
        == {
            "source-fragment": FRAGMENTS,
            "candidate-logic": FRAGMENTS,
        },
        "the source and candidate grids do not cross all four logic fragments",
    )
    _check(
        comparison.sweep.expected_points == 72,
        "the minimal source comparison does not declare its 72 augmented points",
    )
    twins = comparison.signature_twins
    if twins is None:
        msg = "the source comparison has no signature twins"
        raise AssertionError(msg)
    _check(
        (twins.count, twins.primary_condition, twins.observed_twin)
        == (3, "primary", "observed"),
        "the source comparison does not augment only its primary observations",
    )
    _check(
        comparison.freeze_id == "freeze/1",
        "the canonical reduced comparison does not own freeze 1",
    )
    [reducer] = comparison.reducers
    _check(
        (
            reducer.source_path,
            reducer.candidate_path,
            reducer.data_size_path,
            reducer.description_length_metric,
        )
        == (
            "comparison.source",
            "model.language.grammar.name",
            "data.datasets.train.dataset.size",
            "description-length-nats",
        ),
        "the reducer does not read the source, candidate, size, and complexity owners",
    )
    _check(
        {(row.evaluator, row.metric) for row in comparison.rows}
        == {
            ("fit", "string-log-likelihood-per-sentence"),
            ("fit", "description-length-nats"),
            ("labels", "inference-label-3-way"),
        },
        "the comparison does not report held-out fit, description length, and labels",
    )
    _check(
        len(vocabularies) == 2,
        "the vocabulary-size preflight does not expand to two points",
    )
    _check(
        {_at(point.config, f"{structure}.constants") for point in vocabularies}
        == {8, 12},
        "the vocabulary preflight did not resolve both inventory sizes",
    )
    _check(
        len(decompositions) == 2,
        "the decomposition preflight does not expand to two points",
    )
    _check(
        {
            _text_list_at(point.config, f"{structure}.fragment.connectives")
            for point in decompositions
        }
        == {
            ("and", "if", "or", "not"),
            ("and", "if", "cause", "become"),
        },
        "the decomposition preflight did not resolve both connective inventories",
    )


def test_source_and_candidate_fragments_compose_independently(
    logic_runs: Mapping[str, RunSpec],
) -> None:
    """Every source and candidate fragment reaches its independent run owner."""
    found: set[tuple[str, str, tuple[str, ...]]] = set()

    for name, run in logic_runs.items():
        document = _document(run)
        source = "data.datasets.train.dataset.source.interpretation.meanings.source"
        candidate = "model.language.interpretation.meanings.source"
        source_quantifiers = _quantifier_names_at(
            document, f"{source}.structure.fragment.quantifiers"
        )
        candidate_quantifiers = _quantifier_names_at(
            document, f"{candidate}.structure.fragment.quantifiers"
        )
        source_order = _at(document, f"{source}.structure.fragment.order")
        candidate_order = _at(document, f"{candidate}.structure.fragment.order")

        _check(
            _at(document, "data.datasets.train.dataset.grammar.name") == name,
            f"{name} did not reach the source grammar",
        )
        _check(
            _at(document, "model.language.grammar.name") == name,
            f"{name} did not reach the candidate grammar",
        )
        _check(
            (source_order, source_quantifiers)
            == (candidate_order, candidate_quantifiers),
            f"{name} resolved different source and candidate fragment declarations",
        )
        _check(
            _at(document, "model.capacity.target") == 65_536,
            f"{name} did not retain the matched parameter capacity",
        )
        if not isinstance(source_order, str):
            msg = f"{name} resolved a non-text source order"
            raise TypeError(msg)
        found.add((name, source_order, source_quantifiers))

    _check(
        found
        == {
            ("quantifier-free", "first", ()),
            ("universal-only", "first", ("forall",)),
            ("existential-only", "first", ("exists",)),
            ("second-order", "second", ("forall",)),
        },
        "the composed runs do not realize all four declared logic fragments",
    )


def test_validity_controls_change_the_owned_treatment() -> None:
    """Controls alter labels or the learner lexicon while holding the source fixed."""
    primary = _logic_run("quantifier-free")
    reversed_run = _logic_run(
        "quantifier-free", "data/datasets/train/dataset=l2-pairs-reversed"
    )
    scrambled = _logic_run(
        "quantifier-free",
        "model.language.interpretation.meanings.source.structure.seed=991",
    )
    primary_data = primary.data.datasets["train"].dataset
    reversed_data = reversed_run.data.datasets["train"].dataset
    scrambled_data = scrambled.data.datasets["train"].dataset
    if not isinstance(primary_data, SyntheticInferenceDecl):
        msg = "the primary L2 dataset is not synthetic inference"
        raise TypeError(msg)
    if not isinstance(reversed_data, SyntheticInferenceDecl):
        msg = "the reversed L2 dataset is not synthetic inference"
        raise TypeError(msg)
    if not isinstance(scrambled_data, SyntheticInferenceDecl):
        msg = "the scrambled L2 dataset is not synthetic inference"
        raise TypeError(msg)
    noise_spec = reversed_data.label_noise

    if noise_spec is None:
        msg = "the reversed control declares no label noise"
        raise AssertionError(msg)
    noise = noise_spec.build()
    for label in (
        InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS,
        InferenceLabel.NEUTRAL,
    ):
        _check(noise.rate(label) == 1.0, f"{label.value} is not always reversed")
        _check(
            noise.confusion.lands(label, 0.0) is not label,
            f"{label.value} can survive the reversed control",
        )

    _check(
        inference_fingerprint(primary_data) != inference_fingerprint(reversed_data),
        "the reversed control did not change the data-generating declaration",
    )
    _check(
        inference_fingerprint(primary_data) == inference_fingerprint(scrambled_data),
        "the learner-lexicon control changed the source corpus",
    )
    _check(
        (
            _at(
                _document(primary),
                "model.language.interpretation.meanings.source.structure.seed",
            ),
            _at(
                _document(scrambled),
                "model.language.interpretation.meanings.source.structure.seed",
            ),
        )
        == (0, 991),
        "the lexicon control did not permute the learner's semantic assignment",
    )
    _check(
        primary_data.source == scrambled_data.source,
        "the lexicon control changed the source language or interpretation",
    )


def _proof_language() -> LanguageAssembly:
    source = registered_language("neo-davidsonian")
    declaration = source.with_(
        name="l2-reduced-freeze",
        constants=(
            *source.constants,
            ConstantSpec(name="rain", type="t"),
            ConstantSpec(name="dog", type="e -> t"),
            ConstantSpec(name="rex", type="e"),
            ConstantSpec(name="cause", type="t -> t -> t", classes=("sentential",)),
            ConstantSpec(name="become", type="t -> t", classes=("sentential",)),
        ),
        logic=source.logic.with_(order="second"),
    )
    return LanguageIso().forward(declaration)


def _proved(search: SequentSearch, name: str) -> Proved:
    _check(search.supported, f"{name} lies outside guided proof's fragment")
    _check(bool(search.proofs), f"{name} produced no proof")
    rules = ",".join(step.rule for step in search.proofs[0].steps)
    return Proved(Proof("sequent-calculus", f"{name}:{rules}"))


@pytest.mark.parametrize(
    ("name", "order", "goal"),
    [
        ("quantifier-free", "first", "cause(rain)(become(rain))"),
        ("universal-only", "first", "forall x. dog(x)"),
        ("existential-only", "first", "exists x. dog(x)"),
        ("second-order", "second", "forall P:(e -> t). P(rex)"),
    ],
)
def test_reduced_proof_freeze_decides_every_source_fragment(
    name: str,
    order: Literal["first", "second"],
    goal: str,
) -> None:
    """Bounded theorem proving yields a decided label in every fragment."""
    language = _proof_language()
    calculus = SequentCalculus(
        language,
        Normalization(
            fuel=32,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
        order=order,
    )
    limits = SequentLimits(steps=128, depth=16, top_k=2, instantiations=8)
    premise = language.read(f"({goal}) & ~({goal})")
    positive = calculus.search(premise, language.read(goal), limits=limits)
    negative = calculus.search(premise, language.read(f"~({goal})"), limits=limits)
    decided = Decided.from_verdicts(
        _proved(positive, f"{name}-positive"),
        _proved(negative, f"{name}-negative"),
    )

    _check(decided.complete, f"{name} did not select one inference cell")
    _check(
        CLASSICAL.label(decided) is InferenceLabel.ENTAILS,
        f"{name} did not emit the expected decided label",
    )


@pytest.mark.integration
def test_source_comparison_expands_actual_signature_twins() -> None:
    """The canonical 72-point freeze emits observed points, twins, and controls."""
    if shutil.which("vampire") is None:
        pytest.skip("the L2 source declaration requires the Vampire theorem prover")

    points = expand_experiment(
        read_experiment(EXPERIMENTS / "source-factors.yaml"), ROOT
    )
    conditions = Counter(_at(point.config, "comparison.condition") for point in points)
    twins = Counter(_at(point.config, "comparison.twin") for point in points)

    _check(len(points) == 72, "signature-twin augmentation changed the point count")
    _check(
        conditions
        == {
            "primary": 64,
            "reversed-ground-truth": 4,
            "lexicon-scrambled": 4,
        },
        "the augmenter did not confine twins to the primary condition",
    )
    _check(
        twins["observed"] == 24
        and sum(count for twin, count in twins.items() if twin != "observed") == 48,
        "the augmented sweep has the wrong observed-to-twin allocation",
    )


@pytest.mark.integration
def test_completed_minimal_freeze_yields_labels_verdicts_and_source_by_winner() -> None:
    """The completed 72-point freeze emits labels and all comparison reports."""
    freeze = ROOT / "freezes/1/FREEZE.json"
    results = ROOT / "results/1/source-factors"
    if not freeze.is_file() or not results.is_dir():
        pytest.skip("the L2 reduced freeze has not been run in this checkout")
    if shutil.which("vampire") is None:
        pytest.skip("ingesting the L2 freeze requires expanding its Vampire points")

    ingested = ingest_results(
        results,
        experiment=read_experiment(EXPERIMENTS / "source-factors.yaml"),
        freeze=freeze,
        root=ROOT,
    )
    labels = tuple(
        row
        for row in ingested.rows
        if row.get("evaluator") == "labels"
        and row.get("metric") == "inference-label-3-way"
    )
    verdicts = tuple(
        row
        for row in ingested.rows
        if row.get("reducer") == "source-comparison-verdict"
    )
    matrix = tuple(
        row for row in ingested.rows if row.get("reducer") == "source-by-winner"
    )

    _check(labels, "the completed freeze emitted no decided-label metric")
    _check(
        len(verdicts) == len(FRAGMENTS)
        and {row["source"] for row in verdicts} == set(FRAGMENTS),
        "the completed freeze omitted a source-logic verdict",
    )
    _check(
        len(matrix) == len(FRAGMENTS)
        and {row["source"] for row in matrix} == set(FRAGMENTS),
        "the completed freeze omitted a source-by-winner matrix row",
    )

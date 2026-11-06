import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.acceptance.driver import composed

from lofi import train
from lofi._types import ItemId
from lofi.data import (
    Derivation,
    DerivationNode,
    Example,
    GoldDerivation,
    Item,
    Token,
    TokenSpan,
)
from lofi.eval import (
    Approximation,
    ConventionVector,
    EvaluationArtifact,
    ReportIdentity,
)
from lofi.eval._boundary import decode_root_mass
from lofi.eval._bracketing_evaluator import BracketingEvaluator
from lofi.eval._held_out import surface_derivation_dataset
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    plan_training,
    resolve_device_policy,
)
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run._sinks import PointReportSink, write_artifacts
from lofi.run.held_out import (
    _lexical_training_evidence,
    _ordinary_split,
    _unique_derivations,
    evaluate_held_out,
    held_out_predictions,
)
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec import ConfigError, RunSpecCodec
from lofi.spec.evaluators.bracketing import BracketingEvaluatorSpec

if TYPE_CHECKING:
    from lofi.eval import Report
    from lofi.spec import RunSpec

CONFIG: Final = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)
PACKAGED_CODEC: Final = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))
FREEZE: Final = "freeze/3"
SEED: Final = 17
CONVENTIONS: Final = (
    "{punctuation: keep, unary_chains: keep, trivial_spans: keep, "
    "minimum_length: 0, aggregation: sentence, coverage: zero, "
    "self_arcs: allow, gold_derivation: canonical}"
)
BRACKETING: Final = f"""
  brackets:
    kind: bracketing
    interval: {{replicates: 50, level: 0.95}}
    punctuation: {{source: none, forms: []}}
    decoder: minimum-risk
    conventions: {CONVENTIONS}
"""
LEXICAL_TYPES: Final = f"""
  lexicon:
    kind: lexical-types
    interval: {{replicates: 50, level: 0.95}}
    readout: category-terms
    alignment:
      kind: identity
      fit_on: dev
      score_on: test
      random_floor: {{replicates: 50, level: 0.95}}
      stability: {{replicates: 50, level: 0.95}}
    conventions: {CONVENTIONS}
"""
EXPERT_CATEGORIES: Final = f"""
  experts:
    kind: expert-categories
    interval: {{replicates: 50, level: 0.95}}
    readout: category-terms
    taxonomy: [wrong-arity, wrong-result]
    alignment:
      kind: many-to-one
      tie_break: largest-gold-marginal
      fit_on: dev
      score_on: test
      random_floor: {{replicates: 50, level: 0.95}}
      stability: {{replicates: 50, level: 0.95}}
    conventions: {CONVENTIONS}
"""
DECLARED: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
BRACKETING_CONVENTIONS: Final = DECLARED.extended(
    {
        "punctuation_source": "none",
        "punctuation_forms": "[]",
        "decoder": "minimum-risk",
    }
)
LEXICAL_CONVENTIONS: Final = DECLARED.extended(
    {
        "alignment": "identity",
        "alignment_tie_break": None,
        "alignment_fit_on": "dev",
        "alignment_score_on": "test",
        "random_floor_replicates": 50,
        "random_floor_level": 0.95,
        "stability_replicates": 50,
        "stability_level": 0.95,
    }
)


class _DatasetWithHeldOutDefault:
    """Dataset declaration with multiple retained partitions."""

    def retained_splits(self) -> frozenset[str]:
        """Return the declared retained partitions."""
        return frozenset({"development", "test"})

    def held_out_split(self) -> str:
        """Return the partition selected by the dataset."""
        return "test"


def test_a_held_out_dataset_selects_the_scoring_partition_without_an_override() -> None:
    run_spec = cast("RunSpec", SimpleNamespace(evaluators={}))

    assert _ordinary_split(run_spec, _DatasetWithHeldOutDefault()) == "test"


class RecordingSink:
    def __init__(self) -> None:
        self.reports: list[Report] = []
        self.artifacts: list[tuple[str, EvaluationArtifact]] = []

    def emit(self, report: Report, /) -> None:
        self.reports.append(report)

    def emit_artifact(self, role: str, artifact: EvaluationArtifact, /) -> None:
        self.artifacts.append((role, artifact))


def declared(
    tmp_path: Path,
    evaluators: str,
    *,
    size: int = 4,
    accumulation_steps: int = 1,
) -> RunSpec:
    document = tmp_path / "run.yaml"
    retained = ""
    if "kind: lexical-types" in evaluators or "kind: logical-form" in evaluators:
        retained = "        retained: {dev: 0.25, test: 0.25}\n"
    document.write_text(
        "defaults: [run/synthetic-parser]\n"
        "data:\n"
        "  datasets:\n"
        "    train:\n"
        "      dataset:\n"
        f"{retained}"
        f"        size: {size}\n"
        "trainer:\n"
        f"  accumulation_steps: {accumulation_steps}\n"
        f"evaluators:{evaluators}",
        encoding="utf-8",
        newline="\n",
    )

    return PACKAGED_CODEC(document)


def trained(run_spec: RunSpec, tmp_path: Path) -> train.PreparedTraining:
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(SEED, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = train.prepare_training(
        blueprint, streams, resolve_device_policy(run_spec)
    )
    template = _training_checkpoint_manifest(
        blueprint,
        prepared,
        streams,
        CHECKPOINT_RECORDS,
        step=0,
        data_position=0,
    )
    train.fit_training(
        prepared,
        train.DirectoryStore(tmp_path / "checkpoints"),
        template,
        streams,
        train.ResumePosition(0, 0),
        sinks=(),
    )

    return prepared


def held_out_gold(prepared: train.PreparedTraining, split: str) -> int:
    plan = cast("train.ParserFittingPlan", prepared.plan)
    corpus = plan.corpus

    if corpus is None:
        msg = "the held-out parser fixture has no sentence corpus"
        raise TypeError(msg)

    gold = {example.item.item_id for example in corpus.examples}

    return sum(
        item_id in gold
        for batch in plan.retained_batches[split]
        for item_id in batch.item_ids
    )


def test_surface_derivation_data_retains_category_text_without_translation(
    tmp_path: Path,
) -> None:
    run_spec = declared(tmp_path, BRACKETING)
    prepared = trained(run_spec, tmp_path)
    plan = cast("train.ParserFittingPlan", prepared.plan)
    corpus = plan.corpus

    if corpus is None:
        msg = "the held-out parser fixture has no sentence corpus"
        raise TypeError(msg)

    retained = tuple(corpus.examples)
    data = surface_derivation_dataset(retained)

    assert data.ids == tuple(example.item.item_id for example in retained)
    assert tuple(annotation.value for annotation in data.annotations) == tuple(
        example.supervision.derivation for example in retained
    )


def test_repeated_inference_exposure_scores_one_consistent_gold_derivation() -> None:
    item = Item(ItemId("repeated"), (Token(0, "word"),))
    analysis = Derivation(0, {0: DerivationNode(TokenSpan(0, 1), "N")})
    example = Example(item, GoldDerivation(analysis))

    assert _unique_derivations((example, example)) == (example,)


def test_repeated_identifier_refuses_conflicting_gold_derivations() -> None:
    item = Item(ItemId("conflict"), (Token(0, "word"),))
    first = Example(
        item,
        GoldDerivation(Derivation(0, {0: DerivationNode(TokenSpan(0, 1), "N")})),
    )
    second = Example(
        item,
        GoldDerivation(Derivation(0, {0: DerivationNode(TokenSpan(0, 1), "NP")})),
    )

    with pytest.raises(ValueError, match="conflicting held-out gold derivations"):
        _unique_derivations((first, second))


def test_lexical_evidence_reads_gold_categories_under_the_source_grammar(
    tmp_path: Path,
) -> None:
    prepared = trained(declared(tmp_path, BRACKETING), tmp_path)
    plan = cast("train.ParserFittingPlan", prepared.plan)
    candidate = cast(
        "train.ParserFittingPlan",
        SimpleNamespace(
            corpus=plan.corpus,
            dataset=plan.dataset,
            grammar=SimpleNamespace(),
            batches=plan.batches,
        ),
    )

    evidence = _lexical_training_evidence(candidate)

    assert evidence
    assert all(value.frequency >= value.ambiguity for value in evidence.values())


@pytest.mark.parametrize(
    ("evaluators", "role", "metric", "split", "conventions"),
    [
        (
            BRACKETING,
            "brackets",
            "unlabeled-bracketing-f1",
            "validation",
            BRACKETING_CONVENTIONS,
        ),
        (LEXICAL_TYPES, "lexicon", "lexical-type-f1", "test", LEXICAL_CONVENTIONS),
    ],
    ids=["bracketing", "lexical-types"],
)
def test_a_declared_evaluator_reports_held_out_rows_with_full_identity(
    tmp_path: Path,
    evaluators: str,
    role: str,
    metric: str,
    split: str,
    conventions: ConventionVector,
) -> None:
    run_spec = declared(tmp_path, evaluators)
    prepared = trained(run_spec, tmp_path)
    sink = RecordingSink()

    reports = evaluate_held_out(
        prepared, run_spec, freeze_id=FREEZE, seed=SEED, sink=sink
    )

    grammar = cast("train.ParserFittingPlan", prepared.plan).grammar.grammar

    assert tuple(sink.reports) == reports
    assert reports[0].rows[0].approximation == Approximation(
        grammar.budget, residual=None
    )
    assert len(reports) == 1
    row = next(row for row in reports[0].rows if row.metric == metric)
    expected = ReportIdentity(run_spec, grammar, conventions, freeze_id=FREEZE)

    assert row.identity == expected
    assert row.identity.freeze_id == FREEZE
    assert (
        row.normal_form
        == cast(
            "train.ParserFittingPlan", prepared.plan
        ).grammar.grammar.normal_form.name
    )
    assert (row.task, row.metric, row.split, row.seed) == (
        role,
        metric,
        split,
        SEED,
    )
    assert row.n == held_out_gold(prepared, split) > 0
    assert row.value is not None
    assert 0.0 <= row.value <= 1.0
    assert reports[0].unpredicted == ()
    assert reports[0].unmatched == ()


@pytest.mark.parametrize("level", [0.8, 0.95])
def test_a_held_out_row_carries_the_interval_its_evaluator_declares(
    tmp_path: Path, level: float
) -> None:
    run_spec = declared(
        tmp_path, BRACKETING.replace("level: 0.95", f"level: {level}"), size=16
    )

    reports = evaluate_held_out(
        trained(run_spec, tmp_path),
        run_spec,
        freeze_id=FREEZE,
        seed=SEED,
        sink=RecordingSink(),
    )

    (row,) = reports[0].rows

    assert row.n > 1
    assert row.ci_level == level
    assert row.ci_lo is not None
    assert row.ci_hi is not None
    assert row.ci_lo <= row.ci_hi


def test_every_declared_evaluator_emits_its_own_report_in_role_order(
    tmp_path: Path,
) -> None:
    run_spec = declared(tmp_path, LEXICAL_TYPES + BRACKETING)
    sink = RecordingSink()

    reports = evaluate_held_out(
        trained(run_spec, tmp_path), run_spec, freeze_id=FREEZE, seed=SEED, sink=sink
    )

    assert [report.rows[0].task for report in reports] == ["brackets", "lexicon"]
    assert tuple(sink.reports) == reports


def test_expert_workflow_exports_then_ingests_digest_bound_ratings(
    tmp_path: Path,
) -> None:
    export_spec = declared(tmp_path, EXPERT_CATEGORIES, size=8, accumulation_steps=1)
    prepared = trained(export_spec, tmp_path)
    export_sink = PointReportSink("expert-point")

    exported = evaluate_held_out(
        prepared,
        export_spec,
        freeze_id=FREEZE,
        seed=SEED,
        sink=export_sink,
    )
    (artifact_path,) = write_artifacts(tmp_path / "run", export_sink.artifacts)
    package = json.loads(artifact_path.read_bytes())

    assert all(row.value is None for row in exported[0].rows)
    assert package["kind"] == "expert-categories"
    assert len(package["digest"]) == 64
    assert package["categories"]

    ratings_path = tmp_path / "expert-ratings.json"
    ratings_path.write_text(
        json.dumps(
            {
                "kind": "expert-ratings",
                "package_digest": package["digest"],
                "ratings": [
                    {
                        "category_id": category["category_id"],
                        "annotator": annotator,
                        "rating": "correct",
                        "errors": [],
                    }
                    for category in package["categories"]
                    for annotator in ("rater-a", "rater-b")
                ],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    scored_spec = declared(
        tmp_path,
        EXPERT_CATEGORIES.replace(
            "    taxonomy:",
            "    ratings: {kind: expert-ratings, source: "
            f"{ratings_path.as_posix()}" + "}\n    taxonomy:",
        ),
        size=8,
        accumulation_steps=1,
    )
    scored_sink = RecordingSink()

    (scored,) = evaluate_held_out(
        prepared,
        scored_spec,
        freeze_id=FREEZE,
        seed=SEED,
        sink=scored_sink,
    )

    assert scored_sink.artifacts == []
    assert tuple(row.metric for row in scored.rows) == (
        "expert-rating-alpha",
        "expert-error-wrong-arity",
        "expert-error-wrong-result",
    )
    assert tuple(row.value for row in scored.rows) == (1.0, 0.0, 0.0)
    assert all(row.ci_lo is not None and row.ci_hi is not None for row in scored.rows)


def test_grammaticality_reports_score_corrupted_validation_items(
    tmp_path: Path,
) -> None:
    run_spec = composed(
        "regimes-point.yaml",
        "/evaluators.grammaticality=root-mass",
        "evaluators.grammaticality.interval.replicates=20",
        "evaluators.grammaticality.flip_rate=0.1",
        "evaluators.grammaticality.bins=2",
        "data.datasets.train.dataset.size=8",
        "data.datasets.train.dataset.corruption.proportion.value=0.5",
        "data.datasets.train.dataset.retained.validation=0.5",
    )
    prepared = trained(run_spec, tmp_path)
    plan = cast("train.ParserFittingPlan", prepared.plan)
    labels = tuple(label for batch in plan.fit_batches for label in batch.labels)
    sink = RecordingSink()

    reports = evaluate_held_out(
        prepared, run_spec, freeze_id=FREEZE, seed=SEED, sink=sink
    )
    report = next(
        report for report in reports if report.rows[0].task == "grammaticality"
    )

    assert set(labels) == {False, True}
    assert tuple(labels.count(label) for label in (False, True)) == (1, 3)
    assert len(report.rows) == 9
    assert all(row.n == len(labels) for row in report.rows)
    assert all(row.value is not None for row in report.rows)
    assert all(
        row.ci_lo is None and row.ci_hi is None and row.ci_level is None
        for row in report.rows
    )
    assert report.unpredicted == ()
    assert report.unmatched == ()
    [(role, artifact)] = [
        emitted for emitted in sink.artifacts if emitted[0] == "grammaticality"
    ]
    published = decode_root_mass(artifact.content)
    assert role == "grammaticality"
    assert artifact.name == "root-mass.json"
    assert published.readout == "sentences"
    assert tuple(item.item_id for item in published.items) == tuple(
        item_id for batch in plan.fit_batches for item_id in batch.item_ids
    )


def test_an_undeclared_evaluator_kind_is_refused_at_the_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="kind"):
        declared(tmp_path, "\n  unknown:\n    kind: exact-match\n")


def _no_metrics(declaration: BracketingEvaluatorSpec) -> tuple[str, ...]:
    del declaration

    return ()


def test_an_evaluator_whose_metric_its_declaration_does_not_list_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_spec = declared(tmp_path, BRACKETING)
    prepared = trained(run_spec, tmp_path)
    monkeypatch.setattr(BracketingEvaluatorSpec, "metrics", _no_metrics)

    with pytest.raises(ValueError, match="'unlabeled-bracketing-f1'"):
        evaluate_held_out(
            prepared, run_spec, freeze_id=FREEZE, seed=SEED, sink=RecordingSink()
        )


def test_a_malformed_freeze_is_refused_before_any_report(tmp_path: Path) -> None:
    run_spec = declared(tmp_path, BRACKETING)
    prepared = trained(run_spec, tmp_path)
    sink = RecordingSink()

    with pytest.raises(ValueError, match="freeze"):
        evaluate_held_out(
            prepared, run_spec, freeze_id="induction/3", seed=SEED, sink=sink
        )

    assert sink.reports == []


def test_held_out_predictions_rerun_one_role_over_one_retained_split(
    tmp_path: Path,
) -> None:
    run_spec = declared(tmp_path, BRACKETING, size=16)
    prepared = trained(run_spec, tmp_path)
    plan = cast("train.ParserFittingPlan", prepared.plan)
    corpus = plan.corpus

    assert corpus is not None

    gold = {example.item.item_id for example in corpus.examples}
    expected = tuple(
        item_id
        for batch in plan.retained_batches["validation"]
        for item_id in batch.item_ids
        if item_id in gold
    )

    rerun = held_out_predictions(prepared, run_spec, "brackets", "validation")

    assert expected
    assert tuple(item.item_id for item in rerun.items) == expected
    assert sorted(rerun.predictions) == sorted(expected)
    assert type(rerun.evaluator) is BracketingEvaluator


@pytest.mark.parametrize(
    ("role", "split", "named"),
    [("missing", "validation", "'missing'"), ("brackets", "test", "'test'")],
)
def test_held_out_predictions_refuse_an_undeclared_role_or_split(
    tmp_path: Path, role: str, split: str, named: str
) -> None:
    run_spec = declared(tmp_path, BRACKETING)
    prepared = trained(run_spec, tmp_path)

    with pytest.raises(ValueError, match=named):
        held_out_predictions(prepared, run_spec, role, split)

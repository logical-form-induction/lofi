"""Configuration documents compile end to end into synthesizers, charts and trainers.

Each case starts from documents alone: a run document composes a grammar
option, an interpretation option and the shared model, the synthesizer draws
data from the compiled grammar, the chart and trainer fit a tiny point, the
evaluators write report rows, and an experiment document compares declared
candidate grammars. The cases also cover declared data regimes, a grammar
given only by its primitives, a sweep over normal forms, and the inference,
negative-data and logical-form routes a document must reach.
"""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from didactic.settings import load_document
from tests.acceptance.driver import DOCUMENTS, composed, leaf, selection, train
from tests.language import language

from lofi.eval import (
    GROUNDING_FIELD,
    PRIMARY_LEFT,
    PRIMARY_RIGHT,
    READOUT_FIELD,
    SIGNAL_FIELD,
    ArmSpecification,
    BudgetSetting,
    EstimatorProtocol,
    Outcome,
    RegimeRunDirectory,
    RunKind,
    estimator_arms,
    prove_inference_pairs,
    reduce_grounding_runs,
)
from lofi.lf import InferenceCell, ProverBound, TruthTableOracle
from lofi.lf.provers import ProofBudget, ProofService
from lofi.run import (
    GridSweepExpander,
    Sweep,
    composer,
    expand_experiment,
    plan_training,
    read_experiment,
    read_metrics,
)
from lofi.spec import ConfigDocumentError, ConfigError
from lofi.terms import Normalization
from lofi.train import ControllerStep, EvalResult, TermsStep

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tests.acceptance.driver import Row

    from lofi._types import JsonValue
    from lofi.run._boundary.metrics import ContributedValue

REFUSED = (ConfigError, ConfigDocumentError)
"""The refusals a document meets when the schema has no field for its choice."""

PRIMARY = "unlabeled-bracketing-f1"
"""The metric every tiny point's held-out evaluator reports."""

NORMAL_FORMS = ("none", "eisner", "hockenmaier-bisk")
"""The normal forms the composition grammar declares in turn."""

GROUNDING_PROTOCOL = EstimatorProtocol(
    PRIMARY_LEFT,
    PRIMARY_RIGHT,
    outcome=Outcome.SYMBOLIC_RATE,
    margin=0.1,
    falsifier="prover reward is no better than expected embedding",
    family="grounding-regimes",
    registered_at=datetime(2026, 9, 1, tzinfo=UTC),
    alpha=0.05,
    level=0.95,
    replicates=199,
    seed=7,
)
"""The preregistered primary contrast exercised by the document fixture."""


def _primary(rows: tuple[Row, ...]) -> Row:
    """Return the one held-out row reporting the primary metric."""
    (row,) = (row for row in rows if row["metric"] == PRIMARY)

    return row


def _contributed_number(report: Mapping[str, ContributedValue], name: str, /) -> float:
    """Return one scalar number from a contributed report."""
    value = report[name].value

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"contributed metric {name!r} is not numeric: {value!r}"
        raise TypeError(msg)

    return float(value)


def _grounding_sweep() -> Sweep:
    """Return the validated four-regime document."""
    document = cast(
        "dict[str, JsonValue]", load_document(DOCUMENTS / "grounding-regimes.yaml")
    )
    return Sweep.model_validate(document)


@pytest.mark.parametrize(
    ("sweep_name", "point_name", "expected"),
    [
        ("s3-related-factors.yaml", "related-s3-point.yaml", 64),
        ("l3-related-factors.yaml", "related-l3-point.yaml", 32),
    ],
)
def test_related_factor_sweeps_load_and_expand(
    sweep_name: str, point_name: str, expected: int
) -> None:
    """The S3 and L3 documents expose every approved factor as an axis."""
    document = cast("dict[str, JsonValue]", load_document(DOCUMENTS / sweep_name))
    sweep = Sweep.model_validate(document)
    points = GridSweepExpander(composer(selection(point_name))).expand(sweep)

    assert len(points) == expected
    assert len({point.point_id for point in points}) == expected


def _grounding_arms(sweep: Sweep) -> tuple[ArmSpecification, ...]:
    """Translate the one yoked document axis into estimator arms."""
    (group,) = sweep.yoked
    grounding, readout, loss = group
    regimes: dict[str, dict[str, JsonValue]] = {}

    for position, fragment in enumerate(grounding.points):
        if not isinstance(fragment, str):
            msg = f"a grounding fragment is named, not {fragment!r}"
            raise TypeError(msg)
        regimes[fragment] = {
            GROUNDING_FIELD: fragment,
            READOUT_FIELD: readout.points[position],
            SIGNAL_FIELD: loss.points[position],
        }

    return estimator_arms({"dataset": "quantifier-free"}, regimes)


def _write_regime_run(
    directory: Path,
    *,
    seed: int,
    symbolic_rate: float,
    label_likelihood: float,
    label_accuracy: float,
) -> None:
    """Write one completed run boundary consumed by the reducer."""
    directory.mkdir(parents=True)
    (directory / "point.json").write_text(
        json.dumps({"freeze_id": "fixture", "seed_derivation": {}}),
        encoding="utf-8",
        newline="\n",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8", newline="\n"
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "stage": "closed",
                "point": {"seed": seed},
                "fitting": {
                    "optimizer_steps": 4,
                    "epochs": [
                        {
                            "steps": [
                                {
                                    "metrics": {
                                        "grounded/cpu_seconds": 0.25,
                                        "grounded/wall_seconds": 0.5,
                                    }
                                }
                            ]
                        }
                    ],
                    "steps": [],
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    (directory / "rows.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "metric": metric.value,
                        "value": value,
                        "seed": seed,
                        "dataset": RunKind.SYNTHETIC.value,
                    }
                    for metric, value in (
                        (Outcome.SYMBOLIC_RATE, symbolic_rate),
                        (Outcome.LABEL_LIKELIHOOD, label_likelihood),
                        (Outcome.LABEL_ACCURACY, label_accuracy),
                    )
                ]
            }
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_a_document_composes_synthesizes_trains_and_evaluates(
    tmp_path: Path,
) -> None:
    point = train("slash-point.yaml", tmp_path)
    row = _primary(point.rows)

    assert any((point.directory / "checkpoints").iterdir())
    assert row["suite"] == "held-out"
    assert row["dataset"] == "slash"
    value = row["value"]

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_the_declared_flat_parser_control_trains_to_completion(
    tmp_path: Path,
) -> None:
    point = train("flat-point.yaml", tmp_path)
    parser = cast("dict[str, JsonValue]", point.resolved["model"])["parser"]

    assert parser == {
        "kind": "flat",
        "width": 4,
        "features": ["left", "right", "bilinear"],
        "projection_initialization": "identity",
        "declared_output": None,
    }
    assert any((point.directory / "checkpoints").iterdir())


def test_the_declared_span_crf_trains_without_vector_chart_components(
    tmp_path: Path,
) -> None:
    point = train("span-crf.yaml", tmp_path)
    model = cast("dict[str, JsonValue]", point.resolved["model"])

    assert model["parser"] == {
        "kind": "span-crf",
        "hidden": 4,
        "declared_output": None,
    }
    assert model["controller"] is None
    assert model["encoders"] == {}
    assert any((point.directory / "checkpoints").iterdir())


def test_a_flat_parser_refuses_a_recursive_marginal_readout_by_name() -> None:
    with pytest.raises(ConfigDocumentError, match=r"model\.readouts\.types") as refused:
        composed("flat-marginal-refused.yaml")

    message = str(refused.value)

    assert "'flat' parser" in message
    assert "['alpha', 'mu']" in message


def test_an_arrow_product_document_emits_every_constructor_diagnostic(
    tmp_path: Path,
) -> None:
    point = train("arrow-product-diagnostics.yaml", tmp_path)
    expected = {
        "beam-accuracy@1",
        "beam-accuracy@2",
        "beam-accuracy@4",
        "typeless-accuracy@1",
        "typeless-accuracy@2",
        "typeless-accuracy@4",
        "truncated-accuracy@1",
        "truncated-accuracy@2",
        "truncated-accuracy@3",
        "mean-reciprocal-rank",
        "primitive-induced-count",
        "primitive-gold-count",
        "primitive-inventory-exact",
        "primitive-f1",
        "constructor-induced-count",
        "constructor-gold-count",
        "constructor-inventory-exact",
        "constructor-role-information",
        "lexical-type-f1",
        "lexical-token-accuracy",
        "depth-gap",
        "depth-collapse",
    }
    reported = {
        row["metric"]: row["value"] for row in point.rows if row["metric"] in expected
    }

    assert set(reported) == expected
    assert all(isinstance(value, float) for value in reported.values())
    assert reported["depth-collapse"] in {0.0, 1.0}


def test_an_experiment_compares_declared_candidate_grammars(tmp_path: Path) -> None:
    experiment = read_experiment(DOCUMENTS / "experiments" / "candidates.yaml")
    points = expand_experiment(experiment, DOCUMENTS)
    rows = {
        point.point_id: _primary(
            train(
                "tagged-combinator-point.yaml",
                tmp_path / point.point_id,
                *point.overrides,
            ).rows
        )
        for point in points
    }

    assert len(points) == 2
    assert set(rows) == {point.point_id for point in points}
    assert {row["dataset"] for row in rows.values()} == {"slash", "arrow-product"}
    assert len({row["grammar_fingerprint"] for row in rows.values()}) == 2


def test_declared_noise_and_corruption_axes_sweep_independently(
    tmp_path: Path,
) -> None:
    point = train("regimes-point.yaml", tmp_path / "base")
    proportion = train(
        "regimes-point.yaml",
        tmp_path / "proportion",
        "data/datasets/train/dataset/corruption/proportion=half",
    )
    operators = train(
        "regimes-point.yaml",
        tmp_path / "operators",
        "data/datasets/train/dataset/corruption/operators=standard",
    )

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.noise.grammatical.rate")
        == 0.1
    )
    assert (
        leaf(point.resolved, "data.datasets.train.dataset.corruption.proportion.value")
        == 0.25
    )

    reports = []
    for trained in (point, proportion, operators):
        metrics = read_metrics(trained.directory)
        assert metrics is not None
        assert metrics.corpus is not None
        assert metrics.corpus.corruption is not None
        reports.append(metrics.corpus.corruption)

    base_report, proportion_report, operator_report = reports
    assert base_report.sample_size == proportion_report.sample_size
    assert (
        set(base_report.accepted)
        == set(proportion_report.accepted)
        == {
            "swap",
            "deletion",
        }
    )
    assert base_report.achieved_proportion == 0.25
    assert proportion_report.achieved_proportion == 0.5
    assert operator_report.achieved_proportion == base_report.achieved_proportion
    assert set(operator_report.accepted) == {
        "swap",
        "deletion",
        "insertion",
        "substitution",
    }


def test_a_grammar_declared_by_its_primitives_alone_trains(tmp_path: Path) -> None:
    point = train("generated-point.yaml", tmp_path)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.grammar.lexicon.kind")
        == "generated"
    )
    assert point.rows


def test_a_sweep_over_declared_normal_forms_trains_each_form(
    tmp_path: Path,
) -> None:
    points = {
        form: train(
            "composition-point.yaml",
            tmp_path / form,
            f"data/datasets/train/dataset/grammar/normal_form={form}",
        )
        for form in NORMAL_FORMS
    }

    assert len({point.point_id for point in points.values()}) == len(NORMAL_FORMS)

    for form, point in points.items():
        assert (
            leaf(point.resolved, "data.datasets.train.dataset.grammar.normal_form.kind")
            == form
        )


def test_report_rows_record_the_declared_normal_form(tmp_path: Path) -> None:
    point = train(
        "composition-point.yaml",
        tmp_path,
        "data/datasets/train/dataset/grammar/normal_form=eisner",
    )

    assert _primary(point.rows)["normal_form"] == "eisner"


def test_synthesized_inference_pairs_train_a_tiny_point(tmp_path: Path) -> None:
    point = train(
        "inference-point.yaml",
        tmp_path,
        "model/grounding=prover-enumerated",
        "model.readouts.forms.decoding.cap=64",
    )
    recorded = read_metrics(point.directory)

    assert recorded is not None
    grounding = recorded.contributed["grounding"]
    assert grounding["variant"].value == "pairwise"
    assert grounding["scorer"].value == "prover"
    assert grounding["source"].value == "truth-table"
    assert _contributed_number(grounding, "calls") >= 0.0
    assert _contributed_number(grounding, "questions") > 0
    assert _contributed_number(grounding, "questions_per_second") >= 0.0
    assert _contributed_number(grounding, "memo_hit_rate") >= 0.0
    assert _contributed_number(grounding, "cpu_seconds") >= 0.0
    assert _contributed_number(grounding, "wall_seconds") >= 0.0
    assert "process_starts" not in grounding
    assert 0.0 <= _contributed_number(grounding, "label_agreement") <= 1.0


def test_a_collected_nli_release_trains_a_tiny_grounded_point(tmp_path: Path) -> None:
    point = train("collected-inference-point.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "data.datasets.train.dataset.kind") == "snli"
    assert leaf(point.resolved, "data.datasets.train.dataset.training_split") == (
        "train"
    )
    assert recorded is not None
    assert recorded.fitting is not None
    assert recorded.inference is None
    assert recorded.contributed["grounding"]["variant"].value == "expected-embedding"


def test_rte_labeling_trains_a_binary_inference_point(tmp_path: Path) -> None:
    point = train("rte-inference.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "data.datasets.train.dataset.labeling") == "rte"
    assert recorded is not None
    assert recorded.fitting is not None


def test_neural_proof_trains_binary_forms_and_persists_proof_quality(
    tmp_path: Path,
) -> None:
    point = train("neural-proof.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "data.datasets.train.dataset.labeling") == "rte"
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.source.language.logic.prover.kind",
        )
        == "truth-table"
    )
    assert leaf(point.resolved, "model.grounding.scorer.kind") == "neural-proof"
    assert recorded is not None
    grounding = recorded.contributed["grounding"]
    assert 0.0 <= _contributed_number(grounding, "label_agreement") <= 1.0
    assert 0.0 <= _contributed_number(grounding, "verified_share") <= 1.0
    assert (
        0.0
        <= _contributed_number(
            grounding,
            "out_of_fragment_share",
        )
        <= 1.0
    )


def test_guided_proof_trains_binary_forms_and_persists_proof_quality(
    tmp_path: Path,
) -> None:
    point = train("guided-proof.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "data.datasets.train.dataset.labeling") == "rte"
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.source.language.logic.prover.kind",
        )
        == "truth-table"
    )
    assert leaf(point.resolved, "model.grounding.scorer.kind") == "guided-proof"
    assert recorded is not None
    grounding = recorded.contributed["grounding"]
    assert 0.0 <= _contributed_number(grounding, "label_agreement") <= 1.0
    assert 0.0 <= _contributed_number(grounding, "verified_share") <= 1.0
    assert 0.0 <= _contributed_number(grounding, "out_of_fragment_share") <= 1.0


@pytest.mark.parametrize(
    ("document", "scorer"),
    [
        ("second-order-neural-proof.yaml", "neural-proof"),
        ("second-order-guided-proof.yaml", "guided-proof"),
    ],
)
def test_second_order_proof_scorers_train_and_persist_thf_agreement(
    document: str,
    scorer: str,
    tmp_path: Path,
) -> None:
    point = train(document, tmp_path)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "data.datasets.train.dataset.labeling") == "rte"
    assert (
        leaf(point.resolved, "data.datasets.train.dataset.source.language.logic.order")
        == "second"
    )
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.source.language.logic.prover.dialect",
        )
        == "thf"
    )
    assert leaf(point.resolved, "model.grounding.scorer.kind") == scorer
    assert leaf(point.resolved, "model.grounding.scorer.order") == "second"
    assert recorded is not None
    grounding = recorded.contributed["grounding"]
    assert 0.0 <= _contributed_number(grounding, "label_agreement") <= 1.0
    assert 0.0 <= _contributed_number(grounding, "verified_share") <= 1.0
    assert 0.0 <= _contributed_number(grounding, "out_of_fragment_share") <= 1.0


def test_the_configured_grounding_regime_sweep_trains_all_four_points(
    tmp_path: Path,
) -> None:
    sweep = _grounding_sweep()
    points = GridSweepExpander(composer(selection("inference-point.yaml"))).expand(
        sweep
    )
    trained = tuple(
        train(
            "inference-point.yaml",
            tmp_path / point.point_id,
            *point.overrides,
        )
        for point in points
    )

    assert len(points) == len(trained) == 4
    assert len({point.point_id for point in points}) == 4
    assert [
        leaf(point.resolved, "model.readouts.forms.decoding.strategy")
        for point in trained
    ] == ["sample", "beam", "beam", "beam"]
    assert [
        leaf(point.resolved, "loss.terms.0.objective.signal") for point in trained
    ] == ["reward", "likelihood", "likelihood", "likelihood"]


def test_the_grounding_regime_reducer_scores_known_forms(tmp_path: Path) -> None:
    """Reduce the declared arms over rates computed from actual form pairs."""
    assembly = language(("rain", "wet"))
    bound = ProverBound("truth-table", assembly.profile.fragment)
    service = ProofService(
        lambda: TruthTableOracle(
            assembly.store,
            assembly.profile,
            bound=bound,
            atoms=8,
        ),
        profile=assembly.profile,
        workers=1,
        batch_size=8,
        timeout=30.0,
    )
    hit = (assembly.read("rain ∧ wet"), assembly.read("rain"))
    miss = (assembly.read("rain | wet"), assembly.read("rain"))
    patterns = {
        "prover-reward": (hit,) * 10,
        "prover-enumerated": (hit, miss) * 5,
        "expected-embedding": (miss,) * 10,
        "nli-baseline": (miss, hit) * 5,
    }
    directories: list[RegimeRunDirectory] = []

    try:
        for arm, candidates in patterns.items():
            for seed, candidate in enumerate(candidates):
                proof = prove_inference_pairs(
                    assembly,
                    service,
                    ProofBudget(),
                    (candidate,),
                    normalization=Normalization(
                        fuel=40,
                        order="leftmost-outermost",
                        eta="contract",
                        associative_commutative=(),
                    ),
                )
                if proof.cells[0] is None:
                    msg = "the truth-table prover settled no known form pair"
                    raise AssertionError(msg)
                rate = float(proof.cells[0] is InferenceCell.ENTAILS)

                directory = tmp_path / arm / str(seed)
                _write_regime_run(
                    directory,
                    seed=seed,
                    symbolic_rate=rate,
                    label_likelihood=rate - 1.0,
                    label_accuracy=rate,
                )
                directories.append(
                    RegimeRunDirectory(arm, BudgetSetting.FULL, directory)
                )
    finally:
        service.close()

    report = reduce_grounding_runs(
        _grounding_arms(_grounding_sweep()),
        directories,
        protocol=GROUNDING_PROTOCOL,
    )
    rates = {run.arm: run.value(Outcome.SYMBOLIC_RATE) for run in report.runs}

    assert rates == {
        "prover-reward": 1.0,
        "prover-enumerated": 0.5,
        "expected-embedding": 0.0,
        "nli-baseline": 0.5,
    }
    assert report.primary[0].left.cost.optimizer_steps == 40
    assert report.primary[0].right.cost.optimizer_steps == 40
    assert report.primary[0].equivalence.verdict.value == "separated"
    assert json.loads(json.dumps(report.to_spec())) == report.to_spec()


def test_prover_reward_with_countermodel_filter_trains_and_persists_calls(
    tmp_path: Path,
) -> None:
    sampling = (
        'model.readouts.forms.decoding={"strategy":"sample",'
        '"restriction":{"kind":"full"},"draws":2,'
        '"rescore_tolerance":1e-5,"clamp":null}'
    )
    reward = (
        'loss.terms=[{"objective":{"kind":"inference","signal":"reward",'
        '"response":{"kind":"label"},"auxiliary_weight":0.0}}]'
    )
    filtered = train(
        "inference-point.yaml",
        tmp_path / "filtered",
        "model/grounding=prover-reward",
        sampling,
        reward,
    )
    filtered_metrics = read_metrics(filtered.directory)

    assert filtered_metrics is not None
    filtered_grounding = filtered_metrics.contributed["grounding"]
    assert leaf(filtered.resolved, "model.grounding.scorer.countermodels.method") == (
        "exhaustive"
    )
    assert _contributed_number(filtered_grounding, "calls") >= 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param((), id="nli-baseline"),
        pytest.param(("model/grounding=expected-embedding",), id="expected-embedding"),
        pytest.param(
            ("model/grounding=model-checking-sampled",),
            id="model-checking-sampled",
        ),
        pytest.param(
            ("model/grounding=model-checking-given",),
            id="model-checking-given",
        ),
        pytest.param(
            ("model/grounding=model-checking-learned",),
            id="model-checking-learned",
        ),
        pytest.param(("model/grounding=prover-enumerated",), id="prover-enumerated"),
        pytest.param(
            (
                "model/grounding=prover-reward",
                (
                    'model.readouts.forms.decoding={"strategy":"sample",'
                    '"restriction":{"kind":"full"},"draws":2,'
                    '"rescore_tolerance":1e-5,"clamp":null}'
                ),
                (
                    'loss.terms=[{"objective":{"kind":"inference",'
                    '"signal":"reward","response":{"kind":"label"},'
                    '"auxiliary_weight":0.0}}]'
                ),
            ),
            id="prover-reward",
        ),
    ],
)
def test_each_grounding_fragment_composes(overrides: tuple[str, ...]) -> None:
    run = composed("inference-point.yaml", *overrides)

    assert run.model.grounding is not None
    assert str(run.model.grounding.kind) in {"expected-embedding", "pairwise"}


def test_neural_proof_fragment_composes_with_binary_labels() -> None:
    run = composed("rte-inference.yaml", "model/grounding=neural-proof")

    assert run.model.grounding is not None
    assert str(run.model.grounding.kind) == "pairwise"


def test_an_arrow_product_grammar_with_curry_meanings_is_compared_to_slash(
    tmp_path: Path,
) -> None:
    curried = train("curry-inference.yaml", tmp_path / "curried")
    slash = train("inference-point.yaml", tmp_path / "slash")
    curried_metrics = read_metrics(curried.directory)
    slash_metrics = read_metrics(slash.directory)

    assert curried_metrics is not None
    assert slash_metrics is not None
    assert curried_metrics.inference is not None
    assert slash_metrics.inference is not None
    assert (
        leaf(curried.resolved, "data.datasets.train.dataset.grammar.name")
        == "arrow-product"
    )
    assert leaf(slash.resolved, "data.datasets.train.dataset.grammar.name") == "slash"


@pytest.mark.parametrize(
    ("regime", "overrides"),
    [
        ("absent", ("data.datasets.train.dataset.ill_typing=null",)),
        ("labeled-negatives", ()),
        (
            "positive-only",
            ("data.datasets.train.dataset.ill_typing.regime=positive-only",),
        ),
        (
            "label-flips",
            (
                "data.datasets.train.dataset.ill_typing.regime=label-flips",
                "data.datasets.train.dataset.ill_typing.noise=1.0",
            ),
        ),
    ],
)
def test_ill_typed_forms_reach_training_under_each_negative_data_regime(
    tmp_path: Path, regime: str, overrides: tuple[str, ...]
) -> None:
    point = train("ill-typed-negatives.yaml", tmp_path, *overrides)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert recorded.bucketing is not None
    assert recorded.corpus is not None
    split_reports = (
        recorded.bucketing.training.form_typing,
        recorded.bucketing.validation.form_typing,
    )
    assert all(report is not None for report in split_reports)

    if regime == "absent":
        assert recorded.corpus.ill_typing is None
        assert (
            sum(0 if report is None else report.ill_typed for report in split_reports)
            == 0
        )
        return

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.ill_typing.regime") == regime
    )
    report = recorded.corpus.ill_typing
    assert report is not None
    assert abs(report.injection.achieved_proportion - 0.25) <= (
        1.0 / report.injection.sample_size
    )
    assert (
        sum(0 if split is None else split.ill_typed for split in split_reports)
        == report.injection.corrupted
    )
    assert {
        label: sum(
            0 if split is None else split.observed[label] for split in split_reports
        )
        for label in ("well-typed", "ill-typed")
    } == {
        "well-typed": report.injection.sample_size + report.observed["well-typed"],
        "ill-typed": report.observed["ill-typed"],
    }

    observed_negative = "ill-typed" if regime == "labeled-negatives" else "well-typed"
    assert report.observed[observed_negative] == report.injection.corrupted

    if regime == "labeled-negatives":
        assert recorded.fitting is not None
        steps = tuple(step for epoch in recorded.fitting.epochs for step in epoch.steps)
        probabilities = tuple(
            value
            for step in steps
            if (value := step.metrics.get("loss/form-typing/ill_typed_probability"))
            is not None
        )

        assert probabilities
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_a_term_decoder_trains_on_synthesized_meanings(tmp_path: Path) -> None:
    point = train("meaning-objective.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "trainer.spaces") == ["forms"]
    assert leaf(point.resolved, "model.spaces.forms.sort") == "Tm"
    assert recorded is not None
    assert recorded.fitting is not None
    assert recorded.fitting.unit == "steps"
    assert any(
        step.metrics.get("pairs/sample/emitted", 0.0) > 0.0
        for step in recorded.fitting.steps
    )


def test_rule_pair_pretraining_trains_a_declared_unary_rule(tmp_path: Path) -> None:
    run = composed("rule-pair-unary.yaml")
    blueprint = plan_training(run)
    rules = tuple(blueprint.learner.grammar.grammar.rules)
    point = train("rule-pair-unary.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert run.trainer.kind == "terms"
    assert run.pair_sampler is not None
    assert run.pair_sampler.kind == "rule"
    assert "data.pair_sampler" in blueprint.component_fingerprints
    assert [(rule.name, rule.arity) for rule in rules] == [("bare-plural:0:N=>NP", 1)]
    assert [action.rule for action in blueprint.actions.composing] == [rules[0].name]
    assert leaf(point.resolved, "model.encoders.terms.inventory") == "shared-dag"
    assert leaf(point.resolved, "model.spaces.categories.combinators.decoders") == {}
    assert recorded is not None
    assert recorded.fitting is not None
    assert recorded.fitting.unit == "steps"
    assert any(
        step.metrics.get("loss/categories/combinators", 0.0) > 0.0
        for step in recorded.fitting.steps
    )


def test_soft_raise_with_trains_a_point_that_requires_type_raising(
    tmp_path: Path,
) -> None:
    run = composed("raising-soft.yaml")
    blueprint = plan_training(run)
    point = train("raising-soft.yaml", tmp_path)
    recorded = read_metrics(point.directory)
    application = "application:/"

    assert leaf(point.resolved, "model.controller.raise") == "soft"
    assert [
        action.name
        for action in blueprint.actions.actions
        if action.rule == application
    ] == [
        f"{application}:raise=none",
        f"{application}:raise=functor",
        f"{application}:raise=argument",
        f"{application}:raise=both",
    ]
    assert recorded is not None
    assert recorded.fitting is not None
    assert any(epoch.steps for epoch in recorded.fitting.epochs)


@pytest.mark.parametrize(
    ("accuracy", "expected_steps", "expected_stop"),
    [(0.95, 1, "target-met"), (0.5, 2, "target-not-met")],
)
def test_combinator_pretraining_persists_its_held_out_target_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accuracy: float,
    expected_steps: int,
    expected_stop: str,
) -> None:
    def evaluate(self: TermsStep, draw: object) -> EvalResult:
        del self, draw
        return EvalResult(
            2,
            metrics={"loss/categories/combinators/exact_match": accuracy},
        )

    monkeypatch.setattr(TermsStep, "evaluate", evaluate)
    point = train("accuracy-target.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert recorded.fitting is not None
    assert recorded.fitting.optimizer_steps == expected_steps
    assert recorded.fitting.stop == expected_stop
    assert recorded.fitting.achieved_accuracy == accuracy


@pytest.mark.parametrize(
    ("accuracy", "expected_steps", "expected_stop"),
    [(0.95, 1, "target-met"), (0.5, 2, "target-not-met")],
)
def test_controller_pretraining_persists_its_held_out_target_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accuracy: float,
    expected_steps: int,
    expected_stop: str,
) -> None:
    def evaluate(self: ControllerStep, draw: object) -> EvalResult:
        del self, draw
        return EvalResult(
            2,
            metrics={"loss/controller/accuracy": accuracy},
        )

    monkeypatch.setattr(ControllerStep, "evaluate", evaluate)
    point = train("controller-target.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert recorded.fitting is not None
    assert recorded.fitting.optimizer_steps == expected_steps
    assert recorded.fitting.stop == expected_stop
    assert recorded.fitting.achieved_accuracy == accuracy

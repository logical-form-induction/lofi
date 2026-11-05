import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import lofi.data
import lofi.eval
import lofi.eval._artifact as artifact
import lofi.eval._bisk_hockenmaier as bisk_hockenmaier
import lofi.eval._case_study as case_study
import lofi.eval._declared_alignment as declared_alignment
import lofi.eval._equivalence as equivalence
import lofi.eval._estimator as estimator
import lofi.eval._inference_proof as inference_proof
import lofi.eval._interannotator as interannotator
import lofi.eval._lexicon_mass as lexicon_mass
import lofi.eval._logical_form as logical_form
import lofi.eval._ports as ports
import lofi.eval._publish as publish
import lofi.eval._reconstruction as reconstruction
import lofi.eval._scrambled as scrambled
import lofi.eval._supertagging as supertagging
import lofi.eval._tree_agreement as tree_agreement
import lofi.stats
from lofi.data import ItemId
from lofi.eval import grammaticality
from lofi.registry import Registry
from lofi.spec.evaluators.alignment import AlignmentSettings
from lofi.spec.reducers import ConventionGridSettings

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.formalism import Grammar
    from lofi.spec import RunSpec


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "eval-boundary-test"}


class _Grammar:
    fingerprint = "grammar:eval-boundary-test"


def _identity(
    conventions: lofi.eval.ConventionVector | None = None,
) -> lofi.eval.ReportIdentity:
    return lofi.eval.ReportIdentity(
        _Config(),
        _Grammar(),
        lofi.eval.convention_crossing()[0] if conventions is None else conventions,
        freeze_id="freeze/1",
    )


def test_the_base_eval_package_imports_without_corpus_dependencies() -> None:
    script = """
import sys

for name in ("lairs", "pyarrow", "pyarrow.parquet"):
    sys.modules[name] = None

import lofi.eval

optional = {
    "DependencyScorer",
    "TransferReport",
    "CrosslingualAlignmentMatrix",
}
assert optional.isdisjoint(lofi.eval.__all__)
assert all(not hasattr(lofi.eval, name) for name in optional)
"""
    finished = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=Path(__file__).parents[3],
    )

    assert finished.returncode == 0, finished.stderr


def test_bisk_hockenmaier_names_are_published_at_the_package_boundary() -> None:
    for name in bisk_hockenmaier.__all__:
        assert name in lofi.eval.__all__
        assert getattr(lofi.eval, name) is getattr(bisk_hockenmaier, name)


def test_case_study_names_are_published_at_the_package_boundary() -> None:
    names = (
        "COMPLEMENT_CONTRAST",
        "FINITENESS_CONTRAST",
        "MissingByteSpanError",
        "MissingPosteriorError",
        "MissingTextError",
        "SpanContrast",
        "SpanGroup",
        "SpanGroupReport",
        "SpanHit",
        "TokenSurfaceMismatchError",
        "induced_type_report",
        "locate_span_groups",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(getattr(lofi.eval, name) is getattr(case_study, name) for name in names)


def test_entailment_preserving_names_are_published_at_the_package_boundary() -> None:
    names = (
        "InferenceProof",
        "InferenceProofBatch",
        "prove_inference_beam",
        "prove_inference_pairs",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(inference_proof, name) for name in names
    )


def test_lexicon_mass_names_are_published_at_the_package_boundary() -> None:
    names = (
        "LexiconMassRow",
        "SupertagConfusionTable",
        "lexicon_mass_report",
        "supertag_confusion_table",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(lexicon_mass, name) for name in names
    )


def test_interannotator_names_are_published_at_the_package_boundary() -> None:
    names = ("InterannotatorCeilingRow", "interannotator_ceiling")

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(interannotator, name) for name in names
    )


def test_logical_form_names_are_published_at_the_package_boundary() -> None:
    names = (
        "LOGICAL_FORM_MEASURES",
        "LogicalFormMetric",
        "LogicalFormScores",
        "SmatchScore",
        "alpha_match",
        "exact_match",
        "logical_form_scores",
        "normal_form_match",
        "normalized_form",
        "smatch_score",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(logical_form, name) for name in names
    )


def test_equivalence_names_are_published_at_the_package_boundary() -> None:
    names = (
        "EQUIVALENCE_MEASURES",
        "EquivalenceAccountingMetric",
        "EquivalenceLedger",
        "EquivalenceMetric",
        "EquivalenceScores",
        "equivalence_scores",
        "equivalence_status",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(getattr(lofi.eval, name) is getattr(equivalence, name) for name in names)
    assert lofi.eval.SZS_STATUSES is ports.SZS_STATUSES


def test_the_held_out_evaluator_protocols_are_published() -> None:
    assert lofi.eval.HeldOutEvaluationContext is ports.HeldOutEvaluationContext
    assert lofi.eval.InferenceEvaluationContext is ports.InferenceEvaluationContext
    assert lofi.eval.HeldOutEvaluator is ports.HeldOutEvaluator
    assert lofi.eval.DerivationHeldOutEvaluator is ports.DerivationHeldOutEvaluator
    assert lofi.eval.InferenceHeldOutEvaluator is ports.InferenceHeldOutEvaluator
    assert lofi.eval.HeldOutItems is ports.HeldOutItems
    assert lofi.eval.InferenceItems is ports.InferenceItems
    assert lofi.eval.StratifiedMetric is ports.StratifiedMetric
    assert {
        "DerivationHeldOutEvaluator",
        "HeldOutEvaluationContext",
        "HeldOutEvaluator",
        "HeldOutItems",
        "InferenceEvaluationContext",
        "InferenceHeldOutEvaluator",
        "InferenceItems",
    } <= set(lofi.eval.__all__)


def test_grammaticality_names_are_published_at_the_package_boundary() -> None:
    names = (
        "GrammaticalityReport",
        "grammaticality_report",
        "noise_adjusted_auc",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(grammaticality, name) for name in names
    )


def test_scrambled_signature_names_are_published_at_the_package_boundary() -> None:
    names = (
        "EmpiricalTwinNull",
        "SignatureAssignments",
        "TwinFit",
        "empirical_twin_null",
        "sample_signature_twins",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(getattr(lofi.eval, name) is getattr(scrambled, name) for name in names)


def test_tree_agreement_names_are_published_at_the_package_boundary() -> None:
    names = ("TypeTreeAgreement", "normalized_tree_distance", "type_tree_agreement")

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(tree_agreement, name) for name in names
    )


def test_reconstruction_names_are_published_at_the_package_boundary() -> None:
    names = (
        "ConstructorSetRecovery",
        "LexicalTypeRecovery",
        "PrimitiveSetRecovery",
        "TypeSystemReconstruction",
        "constructor_set_recovery",
        "lexical_type_recovery",
        "primitive_set_recovery",
        "type_system_reconstruction",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(reconstruction, name) for name in names
    )


def test_declared_alignment_names_are_published_at_the_package_boundary() -> None:
    names = (
        "DeclaredAlignmentScore",
        "FittedDeclaredAlignment",
        "RelabelingBasePrediction",
        "fit_declared_alignment",
        "predict_relabeling_base",
        "score_declared_alignment",
        "stable_declared_alignments",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(declared_alignment, name) for name in names
    )


def test_evaluator_specific_choices_change_the_convention_hash() -> None:
    base = lofi.eval.convention_crossing()[0]
    identity = _identity(base.extended({"alignment": "identity"}))
    aligned = _identity(base.extended({"alignment": "one-to-one"}))
    minimum_risk = _identity(base.extended({"decoder": "minimum-risk"}))
    viterbi = _identity(base.extended({"decoder": "viterbi"}))

    assert identity.convention_hash != aligned.convention_hash
    assert minimum_risk.convention_hash != viterbi.convention_hash


def test_supertagging_names_are_published_at_the_package_boundary() -> None:
    names = (
        "SUPERTAG_ROWS",
        "CategoryProbability",
        "SupertagAccuracy",
        "SupertagConfusionRow",
        "SupertagLengthError",
        "SupertagPrediction",
        "TokenSupertags",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(
        getattr(lofi.eval, name) is getattr(supertagging, name) for name in names
    )


def test_a_report_is_built_from_the_names_the_package_publishes() -> None:
    scheme = lofi.eval.Resampling("sentence", 11)
    result = lofi.eval.MetricResult(
        0.82,
        250,
        scheme,
        components={"tp": 205.0, "fp": 45.0},
        ci=lofi.stats.Interval(0.78, 0.86, 0.95),
    )

    assert result.value == 0.82
    assert result.ci is not None
    assert result.ci.covers(0.82)
    assert result.resampling.unit == "sentence"
    assert lofi.eval.Sample(ItemId("s-1"), 1.0, stratum="short").weight == 1.0


def test_a_scored_number_carries_the_conventions_it_was_scored_under() -> None:
    conventions = lofi.eval.ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )

    assert conventions in lofi.eval.convention_crossing()
    assert tuple(conventions.factors) == tuple(lofi.eval.CONVENTION_LEVELS)


def test_an_alignment_is_fitted_from_counts_and_carries_them_onward() -> None:
    table = lofi.eval.Contingency(
        {("u0", "NP"): 9, ("u0", "VP"): 1, ("u1", "VP"): 4}, split="dev"
    )
    alignment = lofi.eval.Alignment({"u0": "NP", "u1": "VP"}, table)

    assert alignment.split == "dev"
    assert alignment.apply("u0") == "NP"
    assert alignment.contingency.total == 14
    assert alignment.is_injective

    published = lofi.eval.Alignment(
        {"u0": "NP"}, lofi.eval.Contingency(table.counts, split="test")
    )
    assert published.split == "test"


def test_a_bracketing_score_is_built_from_the_names_the_package_publishes() -> None:
    conventions = lofi.eval.ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )
    gold = lofi.eval.GoldBracketing(
        4,
        [
            lofi.eval.Constituent(0, 4, label="S", depth=0),
            lofi.eval.Constituent(1, 4, label="VP", depth=1),
            lofi.eval.Constituent(2, 4, label="NP", depth=2),
        ],
        punctuation=(),
    )
    metric = lofi.eval.BracketingF1(conventions, seed=3)
    metric.update(lofi.eval.right_branching(4), gold, key=ItemId("s-1"))
    scored = metric.compute()

    assert scored.value == 1.0
    assert scored.resampling.seed == 3
    assert set(metric.per_label_recall()) == {"NP", "VP"}


def test_a_comparison_carries_the_falsifier_a_report_has_to_print() -> None:
    p_value = lofi.stats.p_value_against_replicates(
        2.4,
        [0.2, 0.1, 0.5],
        tail="greater",
        null="permuted-arms",
        family="permutation",
        exact=False,
        seed=0,
    )
    comparison = lofi.eval.Comparison(
        "ccg",
        "cfg",
        statistic=2.4,
        p_value=p_value,
        effect=0.3,
        falsifier="the ccg arm brackets no better than the cfg arm",
        smoke_test=False,
        family="grammar-comparison",
    )

    assert comparison.falsifier
    assert not comparison.smoke_test
    assert comparison.p_value.provenance.family == "permutation"

    with pytest.raises(ValueError, match="falsifier is empty"):
        lofi.eval.Comparison(
            "ccg",
            "cfg",
            statistic=2.4,
            p_value=p_value,
            effect=0.3,
            falsifier="",
            smoke_test=True,
            family="grammar-comparison",
        )


def test_a_label_metric_is_aligned_through_the_policies_the_package_publishes() -> None:
    dev = lofi.eval.Contingency(
        {("u0", "NP"): 9, ("u0", "VP"): 1, ("u1", "VP"): 4, ("u1", "NP"): 2},
        split="dev",
    )
    test = lofi.eval.Contingency(
        {("u0", "NP"): 7, ("u0", "VP"): 2, ("u1", "VP"): 5}, split="test"
    )
    many = lofi.eval.ManyToOne(tie_break="least-gold-label")
    one = lofi.eval.OneToOne(tie_break="largest-gold-marginal")
    fitted = one.fit(dev)
    scored = lofi.eval.apply_alignment(fitted, test, unit="token")

    assert many.tie_break in lofi.eval.TIE_BREAKS
    assert not many.injective
    assert one.injective
    assert dict(fitted.mapping) == {"u0": "NP", "u1": "VP"}
    assert scored.value == 12 / 14
    assert scored.components["fitted_induced_labels"] == 2.0


def test_the_clustering_scores_declare_whether_they_are_chance_corrected() -> None:
    table = lofi.eval.Contingency(
        {("u0", "NP"): 9, ("u0", "VP"): 1, ("u1", "VP"): 4}, split="dev"
    )
    scores = lofi.eval.clustering_scores(table, unit="token", beta=1.0)

    assert set(scores) == {
        "homogeneity",
        "completeness",
        "v-measure",
        "adjusted-mutual-information",
        "adjusted-rand",
    }
    assert scores["adjusted-rand"].chance_corrected
    assert not scores["homogeneity"].chance_corrected
    assert isinstance(scores["homogeneity"], lofi.eval.ClusteringScore)
    assert scores["homogeneity"].result.components["induced_labels"] == 2.0


def test_every_row_carries_its_budget_and_a_residual_that_may_be_absent() -> None:
    class Budget:
        """A three-bound report budget."""

        depth = 4
        beam = 16
        mass = 0.01

    measured = lofi.eval.Residual(dict.fromkeys(lofi.eval.RESIDUAL_SOURCES, 0.0))
    rows = [
        lofi.eval.Approximation(Budget(), residual=measured),
        lofi.eval.Approximation(Budget(), residual=None),
    ]
    summary = lofi.eval.residual_summary(rows)

    assert isinstance(rows[0].budget, lofi.eval.Budget)
    assert isinstance(summary, lofi.eval.ResidualSummary)
    assert summary.rows == 2
    assert summary.measured == 1
    assert summary.unmeasured == 1
    assert summary.by_source["beam-truncation"] == 1
    assert rows[0].to_spec()["residual_beam-truncation"] == 0.0
    assert rows[1].to_spec()["residual_beam-truncation"] is None
    assert rows[1].to_spec()["residual_measured"] is False


def test_an_inference_judgment_is_scored_through_the_names_the_package_publishes() -> (
    None
):
    labels = frozenset({"entails", "neutral", "contradicts"})
    gold = [
        lofi.data.Categorical("entails", labels),
        lofi.data.Categorical("neutral", labels),
        lofi.data.Categorical("contradicts", labels),
        lofi.data.Categorical("contradicts", labels),
    ]
    metric = lofi.eval.InferenceLabelMetric(
        "entailment-f1",
        lofi.eval.declared_labels(gold),
        measure="f1",
        target="non-entailment",
        reduction=lofi.eval.LabelReduction(
            {
                "entails": "entails",
                "neutral": "non-entailment",
                "contradicts": "non-entailment",
            }
        ),
        conventions=lofi.eval.ConventionVector(
            punctuation="keep",
            unary_chains="keep",
            trivial_spans="keep",
            minimum_length=0,
            aggregation="corpus",
            coverage="zero",
            self_arcs="forbid",
            gold_derivation="canonical",
        ),
    )

    for index, response in enumerate(gold):
        label = response.label
        assert label is not None
        metric.update(label, response, key=ItemId(f"i-{index}"))

    assert metric.name == "entailment-f1"
    assert metric.compute().value == 1.0

    with pytest.raises(lofi.eval.UndeclaredLabelError):
        metric.update("yes", gold[0], key=ItemId("i-9"))


def test_reported_scores_partition_into_the_sources_the_package_publishes() -> None:
    vectors = lofi.eval.convention_crossing()
    scores = [
        lofi.eval.ReportedScore(
            method=f"method-{m}",
            conventions=vectors[i],
            seed=j,
            score=0.5 + 0.1 * i + 0.03 * j + 0.01 * m + 0.002 * i * j,
        )
        for i in range(3)
        for j in range(3)
        for m in range(3)
    ]
    decomposition = lofi.eval.variance_decomposition(scores, seed=4, replicates=19)

    assert [component.source for component in decomposition.components] == list(
        lofi.eval.VARIANCE_SOURCES
    )
    assert decomposition.crossing == lofi.eval.Crossing(
        scores=27, methods=3, conventions=3, seeds=3, cells=9
    )
    assert (
        decomposition.component("convention").variance
        > decomposition.component("seed").variance
    )
    assert decomposition.component("convention").ci is not None
    assert sum(
        component.variance for component in decomposition.components
    ) == pytest.approx(decomposition.total)

    with pytest.raises(lofi.eval.UnidentifiedDesignError):
        lofi.eval.variance_decomposition(scores[:3], seed=4, replicates=19)


def test_a_multi_seed_row_prices_the_seeds_its_claim_needs() -> None:
    results = [
        lofi.eval.MetricResult(
            0.80 + 0.004 * index, 250, lofi.eval.Resampling("s", index)
        )
        for index in range(6)
    ]
    priced = lofi.eval.seed_budget(
        results, target="variance", precision=0.2, level=0.95
    )
    row = lofi.eval.multi_seed_summary(
        results, target="variance", precision=0.2, level=0.95
    )

    assert priced.target in lofi.eval.SEED_TARGETS
    assert priced.achieved == 6
    assert priced.required > priced.achieved
    assert not priced.is_sufficient
    assert row.components["seeds_required"] == float(priced.required)
    assert row.components["seeds"] == 6.0
    assert (
        lofi.eval.seeds_required(
            [entry.value for entry in results if entry.value is not None],
            target="variance",
            precision=0.2,
            level=0.95,
            minimum=lofi.eval.MINIMUM_SEEDS,
        )
        == priced.required
    )


def test_a_report_row_carries_the_budget_and_the_residual_it_was_computed_under() -> (
    None
):
    class _Budget:
        @property
        def depth(self) -> int:
            return 5

        @property
        def beam(self) -> int:
            return 16

        @property
        def mass(self) -> float | None:
            return 0.05

    scheme = lofi.eval.Resampling("sentence", 2)
    measured = lofi.eval.ReportRow(
        identity=_identity(),
        method="ccg",
        metric="unlabeled-bracketing-f1",
        result=lofi.eval.MetricResult(0.43, 100, scheme),
        approximation=lofi.eval.Approximation(
            _Budget(), residual=lofi.eval.Residual({"beam-truncation": 0.01})
        ),
    )
    absent = lofi.eval.ReportRow(
        identity=_identity(),
        method="cfg",
        metric="unlabeled-bracketing-f1",
        result=lofi.eval.MetricResult(0.39, 100, scheme),
        approximation=lofi.eval.Approximation(_Budget(), residual=None),
    )
    built = lofi.eval.report([measured, absent])

    assert built.residuals.measured == 0
    assert built.residuals.unmeasured == 2
    assert absent.to_spec()["residual_beam-truncation"] is None
    assert measured.to_spec()["residual_beam-truncation"] == 0.01
    assert measured.to_spec()["residual_prover-timeout"] is None
    assert set(lofi.eval.RESIDUAL_SOURCES) == {
        source
        for source in lofi.eval.RESIDUAL_SOURCES
        if f"residual_{source}" in measured.to_spec()
    }


def test_recall_homogeneity_scores_from_the_names_the_package_publishes() -> None:
    conventions = lofi.eval.ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )
    gold = lofi.eval.GoldBracketing(
        4,
        [
            lofi.eval.Constituent(0, 4, label="S", depth=0),
            lofi.eval.Constituent(2, 4, label="NP", depth=1),
        ],
        punctuation=(),
    )
    metric = lofi.eval.RecallHomogeneity(conventions, split="test", seed=1)
    metric.update(
        lofi.eval.InducedBracketing(4, {(0, 4): "c0", (2, 4): "c1"}),
        gold,
        key=ItemId("s-1"),
    )
    rows = metric.rows()

    assert tuple(rows) == lofi.eval.MATCHED_ROWS
    assert rows["unlabeled-recall"].value == 1.0
    assert rows["homogeneity"].value == 1.0
    assert rows["recall-homogeneity"].value == 1.0
    assert isinstance(metric, lofi.eval.Metric)


def test_alignment_stability_reads_several_seeds_at_once() -> None:
    table = lofi.eval.Contingency(
        {("u0", "NP"): 6, ("u0", "VP"): 1, ("u1", "VP"): 5}, split="dev"
    )
    settled = [
        lofi.eval.Alignment({"u0": "NP", "u1": "VP"}, table)
        for _ in range(lofi.eval.MINIMUM_SEEDS)
    ]
    stability = lofi.eval.alignment_stability(
        settled,
        bootstrap=lofi.stats.Bootstrap(unit="seed", replicates=99, seed=3, level=0.95),
    )

    assert stability.agreement.value == 1.0
    assert stability.labels == ("u0", "u1")
    assert stability.entropy == {"u0": 0.0, "u1": 0.0}
    assert stability.to_spec()["seeds"] == lofi.eval.MINIMUM_SEEDS


def test_released_predictions_are_re_scored_across_the_published_crossing() -> None:
    text = json.dumps(
        {
            "system": "released-parser",
            "budget": {"depth": 3, "beam": 5, "mass": None},
            "items": [
                {
                    "item_id": "d1",
                    "split": "dev",
                    "length": 4,
                    "punctuation": [],
                    "gold": [
                        {"start": 0, "end": 4, "label": "S", "depth": 0},
                        {"start": 2, "end": 4, "label": "VP", "depth": 1},
                    ],
                    "spans": [
                        {"start": 0, "end": 4, "label": "c0"},
                        {"start": 2, "end": 4, "label": "c1"},
                    ],
                },
                {
                    "item_id": "t1",
                    "split": "test",
                    "length": 4,
                    "punctuation": [],
                    "gold": [
                        {"start": 0, "end": 4, "label": "S", "depth": 0},
                        {"start": 2, "end": 4, "label": "VP", "depth": 1},
                    ],
                    "spans": [
                        {"start": 0, "end": 4, "label": "c0"},
                        {"start": 2, "end": 4, "label": "c1"},
                    ],
                },
            ],
        }
    )
    predictions = lofi.eval.read_predictions(text)
    vectors = lofi.eval.declared_conventions(
        ConventionGridSettings(
            punctuation=("keep",),
            unary_chains=("keep",),
            trivial_spans=("keep",),
            minimum_length=(0,),
            aggregation=("corpus",),
            coverage=("ignore",),
            self_arcs=("allow",),
            gold_derivation=("canonical",),
        ),
        (),
        registry=Registry("convention"),
    )
    rows = lofi.eval.rescore_declared(
        predictions,
        run_spec=cast("RunSpec", _Config()),
        grammar=cast("Grammar", _Grammar()),
        freeze_id="freeze/1",
        alignments=(
            AlignmentSettings(
                kind="many-to-one",
                tie_break="least-gold-label",
                fit_on="dev",
                score_on="test",
                floor_replicates=19,
                floor_level=0.9,
                stability_replicates=19,
                stability_level=0.9,
            ),
        ),
        metric=lofi.eval.RecallHomogeneityMetric(),
        conventions=vectors,
    )

    assert isinstance(predictions.budget, lofi.eval.Budget)
    assert len(rows) == 1
    assert {row.source for row in rows} == {predictions.digest}
    assert all(isinstance(row.row, lofi.eval.ReportRow) for row in rows)

    with pytest.raises(lofi.eval.PredictionError):
        lofi.eval.read_predictions("[]")


def test_rank_churn_is_published_as_the_only_churn_reducer() -> None:
    result = lofi.eval.rank_churn(
        (
            lofi.eval.RankedScore("a", "constrained", 0.9),
            lofi.eval.RankedScore("b", "constrained", 0.7),
            lofi.eval.RankedScore("a", "unconstrained", 0.6),
            lofi.eval.RankedScore("b", "unconstrained", 0.8),
        ),
        left="constrained",
        right="unconstrained",
    )

    assert isinstance(result, lofi.eval.RankChurnResult)
    assert result.changed == 2
    assert "churn_study" not in lofi.eval.__all__
    assert not hasattr(lofi.eval, "churn_study")


def test_a_report_is_read_through_the_plan_it_was_preregistered_under() -> None:
    plan = lofi.eval.read_preregistration(
        json.dumps(
            {
                "grid": {"grammar": ["ab", "ccg"], "inventory": [4, 8]},
                "hypotheses": [
                    {
                        "name": "ccg-over-ab",
                        "left": "ccg",
                        "right": "ab",
                        "family": "grammar-comparison",
                        "margin": 0.02,
                    },
                    {
                        "name": "wide-over-narrow",
                        "left": "wide",
                        "right": "narrow",
                        "family": "inventory-comparison",
                        "margin": None,
                    },
                ],
            }
        )
    )
    p_value = lofi.stats.p_value_against_replicates(
        2.4,
        [0.2, 0.1, 0.5],
        tail="greater",
        null="permuted-arms",
        family="permutation",
        exact=False,
        seed=0,
    )
    declared = lofi.eval.Comparison(
        "ccg",
        "ab",
        statistic=2.4,
        p_value=p_value,
        effect=0.3,
        falsifier="the ccg arm brackets no better than the ab arm",
        smoke_test=False,
        family="grammar-comparison",
    )
    audit = lofi.eval.check_preregistration(plan, [declared])

    assert isinstance(plan, lofi.eval.Preregistration)
    assert isinstance(audit, lofi.eval.PreregistrationAudit)
    assert plan.grid_size == 4
    assert plan.margin("ab", "ccg") == 0.02
    assert plan.hypothesis("ab", "ccg") == lofi.eval.DeclaredHypothesis(
        "ccg-over-ab", "ccg", "ab", family="grammar-comparison", margin=0.02
    )
    assert audit.tested == ("ccg-over-ab",)
    assert audit.untested == ("wide-over-narrow",)
    assert audit.ratio == pytest.approx(0.5)

    with pytest.raises(lofi.eval.PreregistrationError):
        plan.margin("wide", "narrow")

    with pytest.raises(lofi.eval.PreregistrationError):
        lofi.eval.check_preregistration(
            plan,
            [
                lofi.eval.Comparison(
                    "ccg",
                    "ab",
                    statistic=2.4,
                    p_value=p_value,
                    effect=0.3,
                    falsifier="the ccg arm brackets no better than the ab arm",
                    smoke_test=False,
                    family="a-family-invented-later",
                )
            ],
        )


def test_the_publish_module_is_not_part_of_the_package_boundary() -> None:
    assert all(name not in lofi.eval.__all__ for name in publish.__all__)
    assert all(not hasattr(lofi.eval, name) for name in publish.__all__)


def test_artifact_control_names_are_published_at_the_package_boundary() -> None:
    names = ("Hypothesis", "HypothesisFeaturizer", "LabelHead")

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(getattr(lofi.eval, name) is getattr(artifact, name) for name in names)


def test_the_report_control_capability_is_published_at_the_package_surface() -> None:
    assert "HYPOTHESIS_ONLY" in lofi.eval.__all__
    assert "ControlRequired" in lofi.eval.__all__
    assert "MissingControlError" in lofi.eval.__all__
    assert "UncontrolledInferenceError" not in lofi.eval.__all__
    assert "reports_inference_accuracy" not in lofi.eval.__all__


def test_regime_estimator_names_are_published_at_the_package_boundary() -> None:
    names = (
        "GROUNDING_FIELD",
        "PRIMARY_LEFT",
        "PRIMARY_RIGHT",
        "READOUT_FIELD",
        "REGIME_FIELDS",
        "RESAMPLING_UNIT",
        "SIGNAL_FIELD",
        "ArmCost",
        "ArmSpecification",
        "BudgetSetting",
        "CurvePoint",
        "EstimatorComparison",
        "EstimatorProtocol",
        "EstimatorReport",
        "EstimatorRow",
        "EstimatorVerdict",
        "GroundedRun",
        "Outcome",
        "RegimeRunDirectory",
        "RunKind",
        "SpecificationMismatchError",
        "UnmatchedBudgetError",
        "estimator_arms",
        "estimator_report",
        "estimator_verdict",
        "paired_contrast",
        "reduce_grounding_runs",
    )

    assert all(name in lofi.eval.__all__ for name in names)
    assert all(getattr(lofi.eval, name) is getattr(estimator, name) for name in names)

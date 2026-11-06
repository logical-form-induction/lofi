from typing import TYPE_CHECKING

import pytest

from lofi._types import ItemId
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._matched import InducedBracketing
from lofi.eval._ports import (
    SZS_STATUSES,
    TIE_BREAKS,
    AlignmentPolicy,
    ArtifactEvaluator,
    BaselineBuildContext,
    BaselineMeasurement,
    BaselineResult,
    EvaluationArtifact,
    InferenceFeaturizer,
    LabeledDerivationEvaluator,
    Metric,
    TieBreak,
)
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


_ITEMS = Resampling("item")


def test_a_baseline_runtime_boundary_is_immutable_and_exact(tmp_path: Path) -> None:
    context = BaselineBuildContext(tmp_path, 17)
    measurement = BaselineMeasurement(
        "induction",
        "bisk-hockenmaier-2012",
        "dependencies-length-10",
        "re-running",
        "Bisk and Hockenmaier (2012)",
        "paper reproduction",
        0.715,
        10,
        "proportion",
    )

    assert context == (tmp_path, 17)
    assert measurement.metric == "dependencies-length-10"
    assert measurement.arm == "re-running"
    result = BaselineResult((measurement,), {"ccgbank-auto": f"sha256:{'0' * 64}"})
    assert result.measurements == (measurement,)
    assert result.source_digests == {"ccgbank-auto": f"sha256:{'0' * 64}"}

    with pytest.raises(ValueError, match="SHA-256"):
        BaselineResult((measurement,), {"ccgbank-auto": "mutable-path-only"})


def test_an_evaluation_artifact_is_frozen_and_path_local() -> None:
    artifact = EvaluationArtifact("categories.json", b"{}\n")

    assert artifact.name == "categories.json"
    assert artifact.content == b"{}\n"
    assert repr(artifact) == "EvaluationArtifact('categories.json', 3 bytes)"

    attribute = "name"
    with pytest.raises(AttributeError):
        setattr(artifact, attribute, "other.json")

    with pytest.raises(ValueError, match="one path segment"):
        EvaluationArtifact("../categories.json", b"{}\n")


def test_the_artifact_evaluator_capability_is_structural() -> None:
    class Emits:
        def artifacts(self) -> tuple[EvaluationArtifact, ...]:
            return (EvaluationArtifact("categories.json", b"{}\n"),)

    assert isinstance(Emits(), ArtifactEvaluator)


def test_the_labeled_derivation_capability_is_structural() -> None:
    class Labels:
        def labeled_spans(self, prediction: str, /) -> InducedBracketing | None:
            return None if not prediction else InducedBracketing(1, {(0, 1): "NP"})

    labels = Labels()

    assert isinstance(labels, LabeledDerivationEvaluator)
    assert not isinstance(object(), LabeledDerivationEvaluator)
    assert labels.labeled_spans("") is None


class ExactMatch:
    """Exact string matching without conventions."""

    def __init__(self) -> None:
        self._hits: dict[ItemId, bool] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "exact-match"

    @property
    def conventions(self) -> None:
        """Return no scoring conventions."""
        return

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the result."""
        return True

    def reset(self) -> None:
        """Forget every item scored so far."""
        self._hits = {}

    def update(self, pred: str, gold: str, *, key: ItemId) -> None:
        """Record whether one prediction matches the gold string."""
        self._hits[key] = pred == gold

    def compute(self) -> MetricResult:
        """Return the proportion matched, absent over no items."""
        if not self._hits:
            return MetricResult(None, 0, _ITEMS)

        matched = sum(self._hits.values())

        return MetricResult(
            matched / len(self._hits),
            len(self._hits),
            _ITEMS,
            components={"tp": float(matched)},
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each item's own zero or one."""
        return {
            key: MetricResult(float(hit), 1, _ITEMS) for key, hit in self._hits.items()
        }


class OneGoldLabel:
    """Maps every induced label to one selected gold label."""

    def __init__(self, *, tie_break: TieBreak) -> None:
        self._tie_break = tie_break

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "one-gold-label"

    @property
    def tie_break(self) -> TieBreak:
        """Return the tie-breaking order."""
        return self._tie_break

    @property
    def injective(self) -> bool:
        """Return whether the fitted mapping is injective."""
        return False

    def _chosen(self, contingency: Contingency) -> str:
        if self._tie_break == "least-gold-label":
            return min(contingency.gold_labels)

        return max(
            contingency.gold_labels,
            key=lambda gold: sum(
                contingency.count(induced, gold)
                for induced in contingency.induced_labels
            ),
        )

    def fit(self, contingency: Contingency, /) -> Alignment:
        """Assign every induced label to one selected gold label."""
        return Alignment(
            dict.fromkeys(contingency.induced_labels, self._chosen(contingency)),
            contingency,
        )


def test_a_metric_is_driven_through_the_protocol_it_conforms_to() -> None:
    metric: Metric[str, str] = ExactMatch()

    assert isinstance(metric, Metric)
    assert metric.compute().value is None
    assert metric.compute().n == 0

    metric.update("the dog barked", "the dog barked", key=ItemId("i0"))
    metric.update("the dog barked", "the cat slept", key=ItemId("i1"))

    assert metric.name == "exact-match"
    assert metric.order_invariant
    assert metric.conventions is None
    assert metric.compute().value == 0.5
    assert metric.compute().n == 2
    assert set(metric.per_item()) == {ItemId("i0"), ItemId("i1")}
    assert metric.per_item()[ItemId("i0")].value == 1.0

    metric.reset()

    assert metric.compute().value is None


def test_an_object_missing_a_metric_member_is_not_a_metric() -> None:
    class HalfAMetric:
        """An incomplete metric."""

        @property
        def name(self) -> str:
            """Return the fixture name."""
            return "half"

    assert not isinstance(HalfAMetric(), Metric)


def test_a_policy_is_driven_through_the_protocol_it_conforms_to() -> None:
    policy: AlignmentPolicy = OneGoldLabel(tie_break="least-gold-label")
    table = Contingency({("u0", "VP"): 2, ("u1", "NP"): 3}, split="dev")
    fitted = policy.fit(table)

    assert isinstance(policy, AlignmentPolicy)
    assert policy.name == "one-gold-label"
    assert policy.tie_break == "least-gold-label"
    assert not policy.injective
    assert fitted.apply("u0") == "NP"
    assert fitted.apply("u1") == "NP"
    assert not fitted.is_injective


def test_an_object_missing_a_policy_member_is_not_a_policy() -> None:
    class HalfAPolicy:
        """An incomplete alignment policy."""

        @property
        def name(self) -> str:
            """Return the fixture name."""
            return "half"

    assert not isinstance(HalfAPolicy(), AlignmentPolicy)


def test_the_declared_tie_breaks_choose_differently_on_one_table() -> None:
    # NP is least alphabetically. VP has the largest marginal
    table = Contingency({("u0", "VP"): 9, ("u1", "NP"): 3}, split="dev")
    chosen = {
        tie_break: OneGoldLabel(tie_break=tie_break).fit(table).apply("u0")
        for tie_break in TIE_BREAKS
    }

    assert chosen == {"least-gold-label": "NP", "largest-gold-marginal": "VP"}
    assert set(chosen) == set(TIE_BREAKS)


def test_the_declared_statuses_put_the_two_answers_before_the_non_answers() -> None:
    answers = SZS_STATUSES[:2]
    non_answers = SZS_STATUSES[2:]

    assert set(answers) == {"Theorem", "CounterSatisfiable"}
    assert "Timeout" in non_answers
    assert len(set(SZS_STATUSES)) == len(SZS_STATUSES)


class _Tokens:
    """A featurization of a string as its counted words."""

    @property
    def name(self) -> str:
        """Return the featurization's name."""
        return "tokens"

    def __call__(self, item: str, /) -> Mapping[str, int]:
        """Return the item's words with their counts."""
        counts: dict[str, int] = {}

        for token in item.split():
            counts[token] = counts.get(token, 0) + 1

        return counts


class _Unnamed:
    """A featurization that never says what it is."""

    def __call__(self, item: str, /) -> Mapping[str, int]:
        """Return no features."""
        del item

        return {}


def test_a_featurization_is_held_by_the_protocol_that_names_it() -> None:
    held: InferenceFeaturizer[str] = _Tokens()

    assert isinstance(held, InferenceFeaturizer)
    assert held.name == "tokens"
    assert dict(held("a a b")) == {"a": 2, "b": 1}


def test_a_featurization_that_does_not_name_itself_is_not_one() -> None:
    assert not isinstance(_Unnamed(), InferenceFeaturizer)
    assert not isinstance(_ITEMS, InferenceFeaturizer)

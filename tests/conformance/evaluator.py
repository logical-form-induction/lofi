"""Reusable laws for held-out evaluators."""

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn, cast

import pytest
import torch

from lofi._checks import LawViolationError
from lofi.data.synthetic import SyntheticDataset
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.eval import HeldOutEvaluationContext
from lofi.nn.loss._ports import CoherenceEvaluationObjective
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.run.held_out import _lexical_training_evidence
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.train import ChartForward, ParserFittingPlan, prepare_training

if TYPE_CHECKING:
    from lofi._types import ItemId
    from lofi.data import Derivation, Item
    from lofi.eval import (
        DerivationHeldOutEvaluator,
        HeldOutItems,
        Metric,
        MetricResult,
    )
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec import RunSpec
    from lofi.spec.grammar import GrammarAssembly

__all__ = [
    "CONFIG",
    "HELD_OUT_EVALUATOR_LAWS",
    "HeldOutEvaluatorSuite",
    "HeldOutPass",
    "HeldOutSubject",
    "held_out_pass",
    "held_out_run",
    "metric_carries_the_declared_conventions",
    "packaged_held_out_pass",
    "predictions_cover_every_published_item",
    "scoring_ignores_update_order_where_declared",
    "scoring_is_repeatable_after_reset",
]

CONFIG: Final = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
)
CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))


class HeldOutPass(NamedTuple):
    """A grammar, the forward passes published over its items, and their gold."""

    grammar: GrammarAssembly
    published: tuple[ObjectiveOutputs[HeldOutItems], ...]
    gold: tuple[tuple[Item, Derivation[str]], ...]
    context: HeldOutEvaluationContext


class HeldOutSubject[Pred, Gold](NamedTuple):
    """An evaluator and one published pass for it to score."""

    evaluator: DerivationHeldOutEvaluator[Pred, Gold]
    held_out: HeldOutPass


@cache
def held_out_pass(document: Path, /) -> HeldOutPass:
    """Return an untrained synthetic parser's pass over every corpus batch.

    Parameters
    ----------
    document : Path
        A synthetic parser run document.

    Returns
    -------
    HeldOutPass
        The grammar, one published pass per training and validation batch, and
        the gold derivation of every item they cover.
    """
    return held_out_run(CODEC(document))


def held_out_run(run_spec: RunSpec, /) -> HeldOutPass:
    """Return an untrained parser pass for a validated run document."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()
    plan = prepare_training(blueprint, streams, resolve_device_policy(run_spec)).plan
    assert isinstance(plan, ParserFittingPlan)
    assert isinstance(plan.forward, ChartForward)
    assert isinstance(plan.dataset, SyntheticDataset)

    batches = tuple(
        batch
        for batch in (*plan.batches, *plan.fit_batches)
        if isinstance(batch, SyntheticTrainingBatch)
    )

    if len(batches) != len(plan.batches) + len(plan.fit_batches):
        msg = "the held-out parser fixture contains a non-sentence batch"
        raise TypeError(msg)

    with torch.no_grad():
        published = tuple(
            cast("ObjectiveOutputs[HeldOutItems]", plan.forward.publish(batch))
            for batch in batches
        )

    covered = {item_id for batch in batches for item_id in batch.item_ids}
    gold = tuple(
        (example.item, example.supervision.derivation)
        for example in plan.dataset.corpus.examples
        if example.item.item_id in covered
    )

    return HeldOutPass(
        plan.grammar,
        published,
        gold,
        HeldOutEvaluationContext(
            plan.grammar,
            interpretation=plan.interpretation,
            forms={
                item_id: form.text
                for item_id, form in plan.dataset.corpus.forms.items()
            },
            coherence={
                entry.name: entry.objective
                for entry in plan.step.assembly
                if isinstance(entry.objective, CoherenceEvaluationObjective)
            },
            target_sets=plan.forward.target_sets,
            lexical_training=_lexical_training_evidence(plan),
            parse=run_spec.parse.build(),
            training_items=len(
                {
                    item_id
                    for batch in plan.batches
                    if isinstance(batch, SyntheticTrainingBatch)
                    for item_id in batch.item_ids
                }
            ),
        ),
    )


def packaged_held_out_pass() -> HeldOutPass:
    """Return the packaged synthetic parser's pass over every corpus batch."""
    return held_out_pass(CONFIG)


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a held-out evaluator law violation.

    Parameters
    ----------
    detail : str
        Description of the failed condition.
    law : str
        Name of the failed law.

    Raises
    ------
    LawViolationError
        Always raised with the law name and detail.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _score[Pred, Gold](
    subject: HeldOutSubject[Pred, Gold],
    metric: Metric[Pred, Gold],
    *,
    reverse: bool,
) -> MetricResult:
    """Score every gold item of a subject's pass from a reset metric."""
    evaluator = subject.evaluator
    predictions: dict[ItemId, Pred] = {}

    for outputs in subject.held_out.published:
        predictions.update(evaluator.predict(outputs))

    pairs = [
        (item.item_id, evaluator.annotate(item, derivation))
        for item, derivation in subject.held_out.gold
    ]
    metric.reset()

    for item_id, gold in reversed(pairs) if reverse else pairs:
        metric.update(predictions[item_id], gold, key=item_id)

    return metric.compute()


def predictions_cover_every_published_item[Pred, Gold](
    subject: HeldOutSubject[Pred, Gold], /
) -> None:
    """Check that every published batch yields one prediction per item.

    Parameters
    ----------
    subject : HeldOutSubject[Pred, Gold]
        Evaluator and one published pass for it to score.

    Raises
    ------
    LawViolationError
        If the predictions of a batch omit one of its items or name another.
    """
    law = "predictions_cover_every_published_item"

    for outputs in subject.held_out.published:
        expected = set(outputs.batch.item_ids)
        found = set(subject.evaluator.predict(outputs))

        if found != expected:
            _fail(
                f"a batch of items {sorted(expected)} was predicted as {sorted(found)}",
                law=law,
            )


def metric_carries_the_declared_conventions[Pred, Gold](
    subject: HeldOutSubject[Pred, Gold], /
) -> None:
    """Check that the metric scores under the conventions the evaluator reports.

    Parameters
    ----------
    subject : HeldOutSubject[Pred, Gold]
        Evaluator and one published pass for it to score.

    Raises
    ------
    LawViolationError
        If the metric has no name or scores under other conventions.
    """
    law = "metric_carries_the_declared_conventions"
    for metric in subject.evaluator.metrics:
        if not metric.name:
            _fail("the metric has no name to report its rows under", law=law)

        if metric.conventions != subject.evaluator.conventions:
            _fail(
                f"the metric scores under {metric.conventions!r}, and the evaluator "
                f"reports {subject.evaluator.conventions!r}",
                law=law,
            )


def scoring_is_repeatable_after_reset[Pred, Gold](
    subject: HeldOutSubject[Pred, Gold], /
) -> None:
    """Check that a reset metric scores one pass the same way twice.

    Parameters
    ----------
    subject : HeldOutSubject[Pred, Gold]
        Evaluator and one published pass for it to score.

    Raises
    ------
    LawViolationError
        If a reset metric still reports data, or two scorings of one pass differ.
    """
    law = "scoring_is_repeatable_after_reset"
    for metric in subject.evaluator.metrics:
        first = _score(subject, metric, reverse=False)
        metric.reset()

        if metric.compute().n != 0:
            _fail("a reset metric still reports scored data", law=law)

        second = _score(subject, metric, reverse=False)

        if (first.value, first.n) != (second.value, second.n):
            _fail(
                f"two scorings of one pass gave {(first.value, first.n)} and "
                f"{(second.value, second.n)}",
                law=law,
            )

        if first.n == 0:
            _fail("the pass scored no item, so it exercises nothing", law=law)


def scoring_ignores_update_order_where_declared[Pred, Gold](
    subject: HeldOutSubject[Pred, Gold], /
) -> None:
    """Check that an order-invariant metric scores reversed updates the same.

    Parameters
    ----------
    subject : HeldOutSubject[Pred, Gold]
        Evaluator and one published pass for it to score.

    Raises
    ------
    LawViolationError
        If a metric declared order invariant scores reversed updates differently.
    """
    law = "scoring_ignores_update_order_where_declared"

    for metric in subject.evaluator.metrics:
        if not metric.order_invariant:
            continue

        forward = _score(subject, metric, reverse=False)
        backward = _score(subject, metric, reverse=True)

        if (forward.value, dict(forward.components)) != (
            backward.value,
            dict(backward.components),
        ):
            _fail(
                f"updates in order gave {forward.value!r} and reversed gave "
                f"{backward.value!r}",
                law=law,
            )


class HeldOutEvaluatorSuite:
    """Reusable held-out evaluator conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_predictions_cover_every_published_item(
        self, held_out_subject: HeldOutSubject[object, object]
    ) -> None:
        """Run the prediction-coverage law.

        Parameters
        ----------
        held_out_subject : HeldOutSubject[object, object]
            Evaluator and one published pass for it to score.
        """
        predictions_cover_every_published_item(held_out_subject)

    def test_metric_carries_the_declared_conventions(
        self, held_out_subject: HeldOutSubject[object, object]
    ) -> None:
        """Run the declared-conventions law.

        Parameters
        ----------
        held_out_subject : HeldOutSubject[object, object]
            Evaluator and one published pass for it to score.
        """
        metric_carries_the_declared_conventions(held_out_subject)

    def test_scoring_is_repeatable_after_reset(
        self, held_out_subject: HeldOutSubject[object, object]
    ) -> None:
        """Run the repeatability law.

        Parameters
        ----------
        held_out_subject : HeldOutSubject[object, object]
            Evaluator and one published pass for it to score.
        """
        scoring_is_repeatable_after_reset(held_out_subject)

    def test_scoring_ignores_update_order_where_declared(
        self, held_out_subject: HeldOutSubject[object, object]
    ) -> None:
        """Run the order-invariance law.

        Parameters
        ----------
        held_out_subject : HeldOutSubject[object, object]
            Evaluator and one published pass for it to score.
        """
        scoring_ignores_update_order_where_declared(held_out_subject)


HELD_OUT_EVALUATOR_LAWS: Final = (
    HeldOutEvaluatorSuite.test_predictions_cover_every_published_item,
    HeldOutEvaluatorSuite.test_metric_carries_the_declared_conventions,
    HeldOutEvaluatorSuite.test_scoring_is_repeatable_after_reset,
    HeldOutEvaluatorSuite.test_scoring_ignores_update_order_where_declared,
)

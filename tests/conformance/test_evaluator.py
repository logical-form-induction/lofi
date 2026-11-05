from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    metric_carries_the_declared_conventions,
    packaged_held_out_pass,
    predictions_cover_every_published_item,
    scoring_ignores_update_order_where_declared,
    scoring_is_repeatable_after_reset,
)

from lofi._checks import LawViolationError
from lofi.eval import (
    Bracketing,
    BracketingF1,
    Constituent,
    ConventionVector,
    GoldBracketing,
    MetricResult,
    Resampling,
    right_branching,
)
from lofi.eval._held_out import derivation_dataset

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval._ports import HeldOutData, HeldOutItems
    from lofi.nn.loss._values import ObjectiveOutputs

CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
OTHER: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="drop",
    minimum_length=0,
    aggregation="corpus",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)


class RightBranching:
    """Predicts the right-branching bracketing of every item."""

    def __init__(
        self, metric: BracketingF1 | None = None, *, omit: bool = False
    ) -> None:
        self._metric = BracketingF1(CONVENTIONS) if metric is None else metric
        self._omit = omit

    @property
    def conventions(self) -> ConventionVector:
        return CONVENTIONS

    @property
    def metrics(self) -> tuple[BracketingF1, ...]:
        """Return the metric in report order."""
        return (self._metric,)

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[GoldBracketing]:
        """Return a retained partition's derivations as gold bracketings."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> GoldBracketing:
        return GoldBracketing(
            len(item.tokens),
            (
                Constituent(node.span.start, node.span.end, label=node.label, depth=0)
                for node in derivation.nodes.values()
            ),
            punctuation=(),
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, Bracketing | None]:
        pairs = tuple(zip(outputs.batch.item_ids, outputs.batch.words, strict=True))
        kept = pairs[1:] if self._omit else pairs

        return MappingProxyType(
            {item_id: right_branching(len(words)) for item_id, words in kept}
        )


class Forgetless(BracketingF1):
    """Never forgets what it scored."""

    @override
    def reset(self) -> None:
        return None


class FirstKeyWins(BracketingF1):
    """Reports whether its first update came before its last, claiming invariance."""

    def __init__(self, conventions: ConventionVector) -> None:
        super().__init__(conventions)
        self.order: list[ItemId] = []

    @override
    def reset(self) -> None:
        super().reset()
        self.order = []

    @override
    def update(
        self, pred: Bracketing | None, gold: GoldBracketing, *, key: ItemId
    ) -> None:
        super().update(pred, gold, key=key)
        self.order.append(key)

    @override
    def compute(self) -> MetricResult:
        if not self.order:
            return MetricResult(None, 0, Resampling("sentence"))

        return MetricResult(
            float(self.order[0] < self.order[-1]),
            len(self.order),
            Resampling("sentence"),
        )


class Mislabeled(RightBranching):
    """Reports conventions its metric does not score under."""

    def __init__(self) -> None:
        super().__init__(BracketingF1(OTHER))


class TestAReferenceEvaluatorObeysTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(self) -> HeldOutSubject[Bracketing | None, GoldBracketing]:
        return HeldOutSubject(RightBranching(), packaged_held_out_pass())


def test_the_coverage_law_catches_an_omitted_item() -> None:
    subject = HeldOutSubject(RightBranching(omit=True), packaged_held_out_pass())

    with pytest.raises(LawViolationError, match="was predicted as"):
        predictions_cover_every_published_item(subject)


def test_the_conventions_law_catches_a_mislabeled_metric() -> None:
    subject = HeldOutSubject(Mislabeled(), packaged_held_out_pass())

    with pytest.raises(LawViolationError, match="the metric scores under"):
        metric_carries_the_declared_conventions(subject)


def test_the_repeatability_law_catches_a_metric_that_never_forgets() -> None:
    subject = HeldOutSubject(
        RightBranching(Forgetless(CONVENTIONS)), packaged_held_out_pass()
    )

    with pytest.raises(LawViolationError, match="still reports scored data"):
        scoring_is_repeatable_after_reset(subject)


def test_the_order_law_catches_an_order_dependent_invariant_metric() -> None:
    subject = HeldOutSubject(
        RightBranching(FirstKeyWins(CONVENTIONS)), packaged_held_out_pass()
    )

    with pytest.raises(LawViolationError, match="reversed gave"):
        scoring_ignores_update_order_where_declared(subject)

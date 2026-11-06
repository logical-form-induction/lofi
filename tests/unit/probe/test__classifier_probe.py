from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest
import torch
from tests.conformance.declared_probe import (
    CELLS,
    DeclaredProbeSuite,
    ToyChart,
    ToyPoint,
)

from lofi.deduce import SpanIndex
from lofi.nn.sites import Role
from lofi.probe import PassLayout, UnpublishedSiteError, published_sites
from lofi.probe._boundary import ClassifierProbeSpec, SpanLabels
from lofi.probe._classifier_probe import PLUGIN, SpanClassifierProbe, _labeled
from lofi.probe._description import VOITA_BLOCKS
from lofi.probe._probing import GradientMode, ProbeBuildContext
from lofi.probe._selectivity import ClassifierProbe, ControlTask, ProbeCapacity
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.ports import Reading
    from lofi.probe import HeldOutPass, Probed


def probe(
    labels: SpanLabels = SpanLabels.CONSTITUENT, *, fraction: float = 0.5
) -> SpanClassifierProbe:
    return SpanClassifierProbe(
        Role.CHART_CELL,
        labels=labels,
        held_out_fraction=fraction,
        split_seed=3,
        classifier=ClassifierProbe(
            ProbeCapacity(hidden=0, epochs=100, learning_rate=0.5, seed=0),
            ControlTask(seed=1),
        ),
        schedule=VOITA_BLOCKS,
    )


class TestSpanClassifierProbeConformance(DeclaredProbeSuite):
    @pytest.fixture
    def declared_probe(self) -> SpanClassifierProbe:
        return probe()


def values(labels: SpanLabels) -> dict[str, float]:
    return {value.measure: value.value for value in probe(labels).run(ToyPoint())}


def test_the_constituent_label_is_read_and_then_erased() -> None:
    measured = values(SpanLabels.CONSTITUENT)

    assert measured["accuracy"] == 1.0
    assert (
        measured["selectivity"] == measured["accuracy"] - measured["control-accuracy"]
    )
    assert measured["erased-accuracy"] < measured["accuracy"]
    assert measured["compression"] == pytest.approx(
        measured["uniform-length"] / measured["description-length"]
    )


def test_the_erased_classifier_reads_nothing_but_the_majority_label() -> None:
    lengths = {(0, 0): 3, (0, 1): 2, (1, 0): 4, (1, 1): 3, (2, 0): 2, (2, 1): 3}
    items = sorted(lengths)
    Random(3).shuffle(items)
    scored = items[:3]
    constituents = sum(2 * lengths[item] - 1 for item in scored)
    spans = sum(lengths[item] * (lengths[item] + 1) // 2 for item in scored)
    measured = values(SpanLabels.CONSTITUENT)

    assert measured["erased-accuracy"] == pytest.approx(constituents / spans)


@pytest.mark.parametrize(
    "scale", [(1.0, 1.0, 1.0), (-10.0, 1.0, 1.0), (-3.0, 0.5, 2.0)]
)
def test_the_scored_spans_are_erased_however_the_label_is_encoded(
    scale: tuple[float, float, float],
) -> None:
    chart = ToyChart()

    with torch.no_grad():
        chart.scale.copy_(torch.tensor(scale))

    measured = {
        value.measure: value.value
        for value in probe(SpanLabels.CONSTITUENT).run(ToyPoint(chart))
    }

    assert measured["accuracy"] == 1.0
    assert measured["erased-accuracy"] == pytest.approx(
        values(SpanLabels.CONSTITUENT)["erased-accuracy"]
    )


def test_the_category_label_is_read_over_gold_constituents_only() -> None:
    reported = probe(SpanLabels.CATEGORY).run(ToyPoint())
    counts = {value.measure: value.count for value in reported}
    gold = sum(2 * n - 1 for n in (3, 2, 4, 3, 2, 3))

    assert {value.measure: value.value for value in reported}["accuracy"] == 1.0
    assert counts["accuracy"] + counts["description-length"] == gold


def test_every_live_span_is_labeled_once_and_no_item_is_in_both_shares() -> None:
    reported = probe().run(ToyPoint())
    counts = {value.measure: value.count for value in reported}
    spans = sum(n * (n + 1) // 2 for n in (3, 2, 4, 3, 2, 3))

    assert counts["accuracy"] + counts["description-length"] == spans
    assert counts["accuracy"] == counts["erased-accuracy"] == counts["selectivity"]
    assert all(value.site == CELLS for value in reported)
    assert len(reported) == 8


def test_the_scored_share_follows_the_declared_fraction() -> None:
    small = {v.measure: v.count for v in probe(fraction=0.2).run(ToyPoint())}
    large = {v.measure: v.count for v in probe(fraction=0.8).run(ToyPoint())}

    assert small["accuracy"] < large["accuracy"]


def test_a_fraction_near_one_still_leaves_an_item_to_fit_on() -> None:
    counts = {v.measure: v.count for v in probe(fraction=0.99).run(ToyPoint())}
    longest = max(n * (n + 1) // 2 for n in (3, 2, 4, 3, 2, 3))

    assert 0 < counts["description-length"] <= longest
    assert counts["accuracy"] > 0


def test_a_span_is_keyed_by_the_words_it_covers() -> None:
    index = SpanIndex(2)
    layout = PassLayout(
        (("a", "b"), ("b", "c")),
        index,
        ({(0, 2): "X"}, {(0, 2): "X"}),
    )
    reading = Observation(
        CELLS,
        torch.zeros(2, index.num_spans, 1),
        axes=("item", "span", "width"),
    )
    spans = _labeled(reading, layout, SpanLabels.CONSTITUENT, 4)
    keys = {(span.item, span.key) for span in spans}

    assert keys == {
        ((4, 0), "a"),
        ((4, 0), "b"),
        ((4, 0), "a b"),
        ((4, 1), "b"),
        ((4, 1), "c"),
        ((4, 1), "b c"),
    }


def test_a_role_without_span_vectors_is_refused() -> None:
    point = ToyPoint()
    refusing = SpanClassifierProbe(
        Role.LOG_PARTITION,
        labels=SpanLabels.CONSTITUENT,
        held_out_fraction=0.5,
        split_seed=0,
        classifier=ClassifierProbe(
            ProbeCapacity(hidden=0, epochs=1, learning_rate=0.1, seed=0),
            ControlTask(seed=0),
        ),
        schedule=VOITA_BLOCKS,
    )

    with pytest.raises(UnpublishedSiteError, match="LOG_PARTITION"):
        refusing.admit(published_sites(point.components))

    probe().admit(published_sites(point.components))


class _OneItem:
    def __init__(self) -> None:
        self.chart = ToyChart()
        layout = PassLayout(
            (("a", "b"),), SpanIndex(2), ({(0, 2): "X", (0, 1): "W", (1, 2): "W"},)
        )

        def run(gradients: GradientMode, /) -> PassLayout:
            with torch.set_grad_enabled(gradients is GradientMode.ENABLED):
                self.chart(layout)
            return layout

        self._passes: tuple[HeldOutPass, ...] = (run,)

    @property
    def components(self) -> Mapping[str, Probed[Reading]]:
        return MappingProxyType({"chart": self.chart})

    @property
    def passes(self) -> tuple[HeldOutPass, ...]:
        return self._passes


def test_a_single_labeled_item_cannot_be_divided() -> None:
    with pytest.raises(ValueError, match="1 held-out items"):
        probe().run(_OneItem())


def test_the_plugin_builds_the_declared_classifier() -> None:
    built = PLUGIN.construct(
        ClassifierProbeSpec.model_validate(
            {"kind": "classifier", "labels": "category"}
        ),
        ProbeBuildContext("cls", ToyPoint()),
    )

    assert built.role is Role.CHART_CELL
    assert repr(built) == "SpanClassifierProbe(CHART_CELL, labels='category')"

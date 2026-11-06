"""The classifier probe: span classifiers over a chart role's vectors.

Every observable point of the declared role holds one vector per span, over an
``item`` axis and then a ``span`` axis that follows the pass's span addressing;
any further axes are flattened into the vector. Each live span of a held-out
item with a gold analysis is labeled from it, and keyed by the words it covers,
which is what the control task draws its labels by. The items are divided under
a seed into a fitting and a scored share, so no item contributes spans to both.

For each point the probe reports three things. The classifier's accuracy on the
real task and on the control task, and the selectivity between them. The online
codelength of the fitting labels under the same fit, with the uniform
codelength it is compared against. And the classifier's accuracies after LEACE,
fitted to the fitting spans' labels, has erased their linear information from
every vector.
"""

from math import ceil
from random import Random
from typing import TYPE_CHECKING, Final, NamedTuple, override

import torch
from torch.nn.functional import one_hot

from lofi.nn.sites import Access
from lofi.probe._boundary import (
    ClassifierProbeSettings,
    ClassifierProbeSpec,
    SpanLabels,
)
from lofi.probe._description import measure_complexity
from lofi.probe._erasure import fit_leace
from lofi.probe._probing import (
    GradientMode,
    ProbeBuildContext,
    ProbeValue,
    UnpublishedSiteError,
    observed_pass,
    published_sites,
)
from lofi.probe._selectivity import SpanInventory
from lofi.probe._values import Observation
from lofi.registry import ComponentPlugin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.nn.sites import Role, Site, SiteSpec
    from lofi.probe._description import BlockSchedule
    from lofi.probe._erasure import LinearErasure
    from lofi.probe._probing import PassLayout, TrainedPoint
    from lofi.probe._selectivity import ClassifierProbe

__all__ = ["SpanClassifierProbe"]


def _scalar(measure: str, site: Site, value: float, count: int, /) -> ProbeValue:
    """Return an ordinary scalar with explicitly absent inferential metadata."""
    return ProbeValue(
        measure,
        site,
        value,
        count,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_CHART_AXES: Final = ("item", "span")
"""The leading axes of a reading that holds one vector per span."""


class _Span(NamedTuple):
    """One labeled span vector and the item it came from."""

    item: tuple[int, int]
    vector: Tensor
    key: str
    label: str


def _labeled(
    reading: Observation, layout: PassLayout, labels: SpanLabels, held_out: int, /
) -> list[_Span]:
    """Return the labeled live spans of one pass's reading at one point.

    Raises
    ------
    ValueError
        If the pass built no chart, or the reading's span axis does not follow
        the pass's span addressing.
    """
    spans = layout.spans

    if spans is None:
        msg = f"the pass that read {reading.site!r} built no chart to label"
        raise ValueError(msg)

    if reading.extent("span") != spans.num_spans:
        msg = (
            f"{reading.site!r} holds {reading.extent('span')} spans per item, but "
            f"the pass addresses {spans.num_spans}"
        )
        raise ValueError(msg)

    vectors = reading.tensor.reshape(
        reading.extent("item"), spans.num_spans, -1
    ).detach()
    found: list[_Span] = []

    for position, (words, gold) in enumerate(
        zip(layout.words, layout.gold, strict=True)
    ):
        if gold is None:
            continue

        for span in range(spans.num_spans):
            start, end = spans.span_of(span)

            if end > len(words):
                continue

            match labels:
                case SpanLabels.CONSTITUENT:
                    label = str((start, end) in gold)

                case SpanLabels.CATEGORY:
                    if (start, end) not in gold:
                        continue

                    label = gold[start, end]

            found.append(
                _Span(
                    (held_out, position),
                    vectors[position, span],
                    " ".join(words[start:end]),
                    label,
                )
            )

    return found


def _inventory(
    spans: Sequence[_Span], classes: Mapping[str, int], vectors: Tensor, /
) -> SpanInventory:
    """Return the spans as an inventory over the given vectors."""
    return SpanInventory(
        vectors,
        [span.key for span in spans],
        [classes[span.label] for span in spans],
        classes=len(classes),
    )


def _erasure(inventory: SpanInventory, /) -> LinearErasure:
    """Return the LEACE map fitted to erase an inventory's labels from its vectors."""
    concepts = one_hot(
        torch.tensor(inventory.labels), num_classes=inventory.classes
    ).to(torch.float64)

    return fit_leace(inventory.vectors, concepts)


def _erased(erasure: LinearErasure, site: Site, vectors: Tensor, /) -> Tensor:
    """Return span vectors with an erasure applied to each."""
    return erasure.stands_in_for(Observation(site, vectors, axes=("span", "feature")))


class SpanClassifierProbe:
    """Classifies the spans at a chart role's points and scores the classifier.

    Parameters
    ----------
    role : Role
        The role of the points whose span vectors are classified.
    labels : SpanLabels
        What each span is labeled with.
    held_out_fraction : float
        The share of items scored rather than fitted on.
    split_seed : int
        The seed the items are divided under.
    classifier : ClassifierProbe
        The classifier, its capacity and its control task.
    schedule : BlockSchedule
        Where the online code's blocks end.
    """

    __slots__ = (
        "_classifier",
        "_fraction",
        "_labels",
        "_role",
        "_schedule",
        "_split_seed",
    )

    def __init__(
        self,
        role: Role,
        *,
        labels: SpanLabels,
        held_out_fraction: float,
        split_seed: int,
        classifier: ClassifierProbe,
        schedule: BlockSchedule,
    ) -> None:
        self._role = role
        self._labels = labels
        self._fraction = held_out_fraction
        self._split_seed = split_seed
        self._classifier = classifier
        self._schedule = schedule

    @property
    def role(self) -> Role:
        """Return the role of the points whose span vectors are classified."""
        return self._role

    def _sites(self, published: Mapping[Site, SiteSpec], /) -> frozenset[Site]:
        """Return the observable span-vector points of the declared role."""
        return frozenset(
            site
            for site, spec in published.items()
            if site.role is self._role
            and Access.OBSERVE in spec.access
            and spec.axes[: len(_CHART_AXES)] == _CHART_AXES
        )

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse a point that publishes no observable span vectors of the role.

        Parameters
        ----------
        published : Mapping[Site, SiteSpec]
            Every point the trained point publishes.

        Raises
        ------
        UnpublishedSiteError
            If no observable published point of the role is laid out over an
            ``item`` axis and then a ``span`` axis.
        """
        if not self._sites(published):
            msg = (
                f"no component publishes an observable {self._role.name} point "
                f"over the axes {_CHART_AXES}, so there are no span vectors to "
                "classify"
            )
            raise UnpublishedSiteError(msg)

    def _divided(self, spans: Sequence[_Span], /) -> tuple[list[_Span], list[_Span]]:
        """Return the spans divided by item into a fitting and a scored share.

        Raises
        ------
        ValueError
            If fewer than two items have labeled spans.
        """
        items = sorted({span.item for span in spans})

        if len(items) < 2:
            msg = (
                f"{len(items)} held-out items have labeled spans, and a classifier "
                "is fitted on one share of the items and scored on another"
            )
            raise ValueError(msg)

        Random(self._split_seed).shuffle(items)
        scored = frozenset(
            items[: min(len(items) - 1, max(1, ceil(self._fraction * len(items))))]
        )

        return (
            [span for span in spans if span.item not in scored],
            [span for span in spans if span.item in scored],
        )

    def _measured(
        self, site: Site, spans: Sequence[_Span], /
    ) -> tuple[ProbeValue, ...]:
        """Return what the classifier reports at one point."""
        fitting, scored = self._divided(spans)
        classes = {
            label: index for index, label in enumerate(sorted({s.label for s in spans}))
        }
        fit = _inventory(fitting, classes, torch.stack([s.vector for s in fitting]))
        held = _inventory(scored, classes, torch.stack([s.vector for s in scored]))
        complexity = measure_complexity(
            self._classifier, fit, held, schedule=self._schedule
        )
        erasure = _erasure(fit)
        erased = self._classifier.run(
            _inventory(fitting, classes, _erased(erasure, site, fit.vectors)),
            _inventory(scored, classes, _erased(erasure, site, held.vectors)),
        )
        codelength = complexity.codelength

        return (
            _scalar("accuracy", site, complexity.accuracy, len(held)),
            _scalar("control-accuracy", site, complexity.control_accuracy, len(held)),
            _scalar("selectivity", site, complexity.selectivity, len(held)),
            _scalar("description-length", site, codelength.bits, len(fit)),
            _scalar("uniform-length", site, codelength.uniform, len(fit)),
            _scalar("compression", site, codelength.compression, len(fit)),
            _scalar("erased-accuracy", site, erased.accuracy, len(held)),
            _scalar("erased-selectivity", site, erased.selectivity, len(held)),
        )

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return the classifier's scores at every point of the declared role.

        Parameters
        ----------
        point : TrainedPoint
            The trained point.

        Returns
        -------
        tuple[ProbeValue, ...]
            Eight values per point, in ascending point order.

        Raises
        ------
        ValueError
            If a pass built no chart, fewer than two held-out items have
            labeled spans, or the labels take a single value.
        """
        sites = self._sites(published_sites(point.components))
        collected: dict[Site, list[_Span]] = {site: [] for site in sites}

        for position, held_out in enumerate(point.passes):
            trace, layout = observed_pass(
                point,
                held_out,
                observe=sites,
                patch={},
                gradients=GradientMode.DISABLED,
            )

            for site in sites:
                if site in trace:
                    collected[site].extend(
                        _labeled(trace[site], layout, self._labels, position)
                    )

        return tuple(
            value
            for site in sorted(sites, key=repr)
            for value in self._measured(site, collected[site])
        )

    @override
    def __repr__(self) -> str:
        """Return the role classified and what the spans are labeled with."""
        return f"SpanClassifierProbe({self._role.name}, labels={self._labels.value!r})"


def _build(
    settings: ClassifierProbeSettings, context: ProbeBuildContext, /
) -> SpanClassifierProbe:
    """Build a span classifier probe from its settings.

    Parameters
    ----------
    settings : ClassifierProbeSettings
        The role, labels, division, classifier and block schedule.
    context : ProbeBuildContext
        The declared role and point capabilities.
    """
    del context

    return SpanClassifierProbe(
        settings.role,
        labels=settings.labels,
        held_out_fraction=settings.held_out_fraction,
        split_seed=settings.split_seed,
        classifier=settings.classifier,
        schedule=settings.schedule,
    )


PLUGIN: Final = ComponentPlugin(spec=ClassifierProbeSpec, build=_build)
"""The plugin the ``classifier`` probe kind registers."""

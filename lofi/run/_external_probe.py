"""The document-driven probe over one released external DIORA model."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, override

import torch

from lofi.deduce import SpanIndex
from lofi.nn._snapshots import artifact_fetcher
from lofi.probe import (
    DiagnosticCatalog,
    DiagnosticsProbe,
    DioraCheckpointLoader,
    GradientMode,
    PassLayout,
    ProbeBuildContext,
    SplitPosteriorKL,
    published_sites,
)
from lofi.registry import ComponentPlugin
from lofi.run._boundary.external_probe import ExternalProbeSettings, ExternalProbeSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site, SiteSpec
    from lofi.probe import (
        LoadedDiora,
        Probed,
        ProbeValue,
        TrainedPoint,
    )

__all__ = ["PLUGIN", "ExternalProbe", "ExternalProbeSettings", "ExternalProbeSpec"]


class _DioraPass:
    """One deterministic released-model pass at a declared sentence length."""

    __slots__ = ("_length", "_loaded")

    def __init__(self, loaded: LoadedDiora, length: int) -> None:
        self._loaded = loaded
        self._length = length

    def __call__(self, gradients: GradientMode, /) -> PassLayout:
        """Run the released chart and return the stimulus layout."""
        embeddings = torch.zeros(1, self._length, self._loaded.input_width)

        with torch.set_grad_enabled(gradients is GradientMode.ENABLED):
            self._loaded.model.run({"embeddings": embeddings})

        words = tuple(f"token-{position}" for position in range(self._length))

        return PassLayout((words,), SpanIndex(self._length), (None,))


class _ExternalPoint:
    """The released model offered to the ordinary declared-probe protocol."""

    __slots__ = ("_components", "_passes")

    def __init__(self, loaded: LoadedDiora, lengths: tuple[int, ...]) -> None:
        model = loaded.model
        self._components: Mapping[str, Probed[Reading]] = MappingProxyType(
            {str(model.declared.model): cast("Probed[Reading]", model)}
        )
        self._passes = tuple(_DioraPass(loaded, length) for length in lengths)

    @property
    def components(self) -> Mapping[str, Probed[Reading]]:
        """Return the released external model under its declared name."""
        return self._components

    @property
    def passes(self) -> tuple[_DioraPass, ...]:
        """Return one pass per declared stimulus length."""
        return self._passes


class ExternalProbe:
    """Run the split posterior diagnostic over a released external point."""

    __slots__ = ("_diagnostics", "_point")

    def __init__(self, loaded: LoadedDiora, lengths: tuple[int, ...]) -> None:
        self._point = _ExternalPoint(loaded, lengths)
        self._diagnostics = DiagnosticsProbe(
            None,
            DiagnosticCatalog({SplitPosteriorKL.tag: SplitPosteriorKL}),
        )

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Validate the released point independently of the trained host point."""
        del published
        self._diagnostics.admit(published_sites(self._point.components))

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return split posterior KL rows over the declared stimulus lengths."""
        del point

        return self._diagnostics.run(self._point)

    @override
    def __repr__(self) -> str:
        """Return how many released-model passes this probe runs."""
        return f"ExternalProbe(passes={len(self._point.passes)})"


def _build(
    settings: ExternalProbeSettings, context: ProbeBuildContext, /
) -> ExternalProbe:
    """Build the declared released-model probe."""
    del context
    fetcher = artifact_fetcher(settings.cache)
    loaded = DioraCheckpointLoader(fetcher, settings.factory).load(settings.model)

    return ExternalProbe(loaded, settings.lengths)


PLUGIN: Final = ComponentPlugin(spec=ExternalProbeSpec, build=_build)
"""The plugin registered for the external probe arm."""

"""Released DIORA models and their split-posterior diagnostic."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    NamedTuple,
    Protocol,
    override,
    runtime_checkable,
)

import torch
from torch import nn

from lofi._checks import ensure_positive
from lofi._frozen import FrozenValue
from lofi.nn.sites import Access, DTypeClass, Role
from lofi.probe._boundary import ExternalModelSpec, ExternalSiteDecl
from lofi.probe._chart import (
    _index,
    _observed,
    _span_support,
    _split_support,
    _validate_reading,
)
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult, Measurement
from lofi.probe._external import ExternalModel, ExternalSiteMap
from lofi.spec.artifacts import source_spec

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor
    from torch.nn import Module

    from lofi._artifacts import ArtifactFetcher
    from lofi.deduce import SpanIndex
    from lofi.nn.sites import Site
    from lofi.probe._values import Observation, Trace
    from lofi.spec.artifacts import PretrainedSourceSpec

__all__ = [
    "DioraCheckpointFactory",
    "DioraCheckpointLoader",
    "DioraForward",
    "DioraInput",
    "LoadedDiora",
    "SplitPosteriorKL",
    "diora_model_spec",
]

_SPAN_AXES: Final = ("item", "span")
_SPLIT_AXES: Final = ("item", "span", "split")


class DioraForward(NamedTuple):
    """Holds a DIORA backend's output and the three published chart values."""

    output: Tensor
    alpha: Tensor
    inside: Tensor
    outside: Tensor


@runtime_checkable
class DioraInput(Protocol):
    """Publishes the embedding width a DIORA backend owns."""

    @property
    def input_width(self) -> int:
        """Return how many features each input token carries."""


class LoadedDiora(FrozenValue):
    """A released DIORA model and the input width its backend declares."""

    __slots__ = ("_input_width", "_model")

    _noun = "loaded DIORA model"

    def __init__(self, model: ExternalModel, input_width: int) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(
            self,
            "_input_width",
            ensure_positive("a DIORA input width", input_width),
        )

    @property
    def model(self) -> ExternalModel:
        """Return the instrumentable released model."""
        return self._model

    @property
    def input_width(self) -> int:
        """Return how many features each input token carries."""
        return self._input_width


@runtime_checkable
class DioraCheckpointFactory(Protocol):
    """Builds a DIORA backend from one fetched released artifact."""

    def build(self, checkpoint: Path, /) -> Module:
        """Return a backend loaded from the verified local artifact.

        The backend routes its log attention, inside scores and outside scores
        through modules named ``alpha``, ``inside`` and ``outside`` respectively,
        which are the sites :func:`diora_model_spec` publishes.
        """


def diora_model_spec(
    model: str, checkpoint: PretrainedSourceSpec, /
) -> ExternalModelSpec:
    """Return the external-model declaration for a released DIORA checkpoint.

    Parameters
    ----------
    model : str
        What the released model is called.
    checkpoint : PretrainedSourceSpec
        Where the pinned released artifact is fetched from.

    Returns
    -------
    ExternalModelSpec
        Its checkpoint and the adapter's attention, inside and outside sites.
    """

    def site(
        path: str,
        role: Role,
        axes: tuple[str, ...],
        dtype_class: DTypeClass,
    ) -> ExternalSiteDecl:
        return ExternalSiteDecl(
            path=path,
            role=role.name,
            layer=0,
            axes=axes,
            dtype_class=str(dtype_class.value),
            differentiable=True,
            access=(Access.OBSERVE.name,),
        )

    return ExternalModelSpec(
        model=model,
        checkpoint=checkpoint,
        sites=(
            site(
                "alpha",
                Role.SPLIT_DISTRIBUTION,
                _SPLIT_AXES,
                DTypeClass.LOG_PROBABILITY,
            ),
            site("inside", Role.INSIDE_SCORE, _SPAN_AXES, DTypeClass.REAL),
            site("outside", Role.OUTSIDE_SCORE, _SPAN_AXES, DTypeClass.REAL),
        ),
    )


def _true_split_posterior(
    attention: Tensor, inside: Tensor, index: SpanIndex, /
) -> Tensor:
    """Return the exact posterior implied by local attention and child insides.

    Local log attention differs from the underlying local split score only by a
    rowwise normalizer. That constant cancels in this second normalization.
    """
    if index.max_splits == 0:
        return attention.clone()

    shape = (int(inside.shape[0]), index.num_spans, index.max_splits)
    left = inside.new_tensor(index.left_child_ids, dtype=torch.long).reshape(-1)
    right = inside.new_tensor(index.right_child_ids, dtype=torch.long).reshape(-1)
    support = attention.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
    joined = (
        inside.index_select(1, left).reshape(shape)
        + inside.index_select(1, right).reshape(shape)
        + attention
    ).masked_fill(~support, -torch.inf)

    return joined - torch.logsumexp(joined, dim=2, keepdim=True)


class DioraCheckpointLoader:
    """Fetches released DIORA checkpoints and adapts them to their sites.

    Parameters
    ----------
    fetcher : ArtifactFetcher
        The fetcher that resolves the declared checkpoint to a verified path.
    factory : DioraCheckpointFactory
        The declared backend built from the fetched artifact.
    """

    __slots__ = ("_factory", "_fetcher")

    def __init__(
        self, fetcher: ArtifactFetcher, factory: DioraCheckpointFactory
    ) -> None:
        if not isinstance(factory, DioraCheckpointFactory):
            msg = f"a DIORA checkpoint factory cannot build: {factory!r}"
            raise TypeError(msg)

        self._fetcher = fetcher
        self._factory = factory

    def load(self, declared: ExternalSiteMap, /) -> LoadedDiora:
        """Return the released model behind its lowered external declaration."""
        release = declared.checkpoint

        if release is None:
            msg = "a released DIORA external model declares its checkpoint"
            raise ValueError(msg)

        canonical = diora_model_spec(
            str(declared.model), source_spec(release)
        ).site_map()

        if declared != canonical:
            msg = (
                "a released DIORA model publishes the canonical alpha, inside and "
                "outside sites for its checkpoint"
            )
            raise ValueError(msg)

        backend = self._factory.build(self._fetcher.fetch(release))

        if not isinstance(backend, nn.Module):
            msg = f"a DIORA checkpoint factory built no torch module: {backend!r}"
            raise TypeError(msg)

        if not isinstance(backend, DioraInput):
            msg = f"a DIORA checkpoint backend publishes no input width: {backend!r}"
            raise TypeError(msg)

        return LoadedDiora(ExternalModel(backend, declared), backend.input_width)


def _one_by_key(
    observed: tuple[tuple[Site, Observation], ...], role: Role
) -> dict[tuple[int | str, ...], tuple[Site, Observation]]:
    """Return one observed point of a role per site key."""
    keyed: dict[tuple[int | str, ...], tuple[Site, Observation]] = {}

    for site, reading in observed:
        if site.key in keyed:
            msg = f"two {role.name} observations share site key {site.key!r}"
            raise ValueError(msg)

        keyed[site.key] = (site, reading)

    return keyed


class SplitPosteriorKL:
    """Measures how far DIORA attention lies from its exact split posterior."""

    __slots__ = ()

    tag: ClassVar[str] = "split-posterior-kl"
    requires: ClassVar[frozenset[Role]] = frozenset(
        {Role.SPLIT_DISTRIBUTION, Role.INSIDE_SCORE, Role.OUTSIDE_SCORE}
    )
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return mean attention-to-posterior KL rows by sentence length.

        DIORA attention is a locally normalized version of its learned split score.
        The omitted rowwise constant cancels when the child inside scores are added,
        so the exact conditional posterior is recovered by treating log attention as
        the local score. Outside scores are part of the released chart contract and
        are validated here; they cancel from the conditional split posterior.
        """
        taken = dict(context.readings(trace))
        attention = _one_by_key(
            _observed(context, taken, Role.SPLIT_DISTRIBUTION),
            Role.SPLIT_DISTRIBUTION,
        )
        inside = _one_by_key(
            _observed(context, taken, Role.INSIDE_SCORE), Role.INSIDE_SCORE
        )
        outside = _one_by_key(
            _observed(context, taken, Role.OUTSIDE_SCORE), Role.OUTSIDE_SCORE
        )

        if not attention:
            return DiagnosticResult(self.tag, ())

        if attention.keys() != inside.keys() or attention.keys() != outside.keys():
            msg = "attention, inside and outside chart sites must share the same keys"
            raise ValueError(msg)

        index = _index(context)
        measured: list[Measurement] = []

        for key in attention:
            alpha_site, alpha = attention[key]
            _, inside_reading = inside[key]
            _, outside_reading = outside[key]
            _validate_reading(
                context,
                alpha_site,
                alpha,
                axes=_SPLIT_AXES,
                dtype_class=DTypeClass.LOG_PROBABILITY,
            )
            _validate_reading(
                context,
                inside_reading.site,
                inside_reading,
                axes=_SPAN_AXES,
                dtype_class=DTypeClass.REAL,
            )
            _validate_reading(
                context,
                outside_reading.site,
                outside_reading,
                axes=_SPAN_AXES,
                dtype_class=DTypeClass.REAL,
            )
            support, lengths, widths = _split_support(alpha, index)
            inside_support, inside_lengths, _ = _span_support(inside_reading, index)
            outside_support, outside_lengths, _ = _span_support(outside_reading, index)

            if (
                not torch.equal(lengths, inside_lengths)
                or not torch.equal(lengths, outside_lengths)
                or not torch.equal(support.any(dim=2), inside_support & (widths > 1))
                or not torch.equal(inside_support, outside_support)
            ):
                msg = "attention, inside and outside masks describe different items"
                raise ValueError(msg)

            posterior = _true_split_posterior(
                alpha.tensor, inside_reading.tensor, index
            )
            positive = support & torch.isfinite(alpha.tensor)

            if bool((positive & ~torch.isfinite(posterior)).any()):
                msg = "positive DIORA attention has no finite exact posterior mass"
                raise ValueError(msg)

            terms = torch.where(
                positive,
                alpha.tensor.exp() * (alpha.tensor - posterior),
                torch.zeros_like(alpha.tensor),
            )
            divergence = terms.sum(dim=2).clamp_min(0.0)
            live_rows = support.any(dim=2)

            for length in sorted(set(lengths.tolist())):
                selected_items = lengths == length
                selected = live_rows & selected_items.unsqueeze(1)
                value = float(divergence[selected].mean())
                measured.append(Measurement(alpha_site, f"kl:length:{length}", value))

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "SplitPosteriorKL()"

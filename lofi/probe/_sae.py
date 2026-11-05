"""Sparse-autoencoder probing over declared causal variables.

The arm fits one sparse autoencoder to the activations at a published site.  Its
training population is the fitting share of the same model-ready causal pairs
used by distributed alignment search; its reported reconstruction, activation
sparsity, and feature alignments are measured on the disjoint held-out share.
The probe therefore compares SAE and DAS findings without inspecting a model's
module paths or private tensor layout.
"""

from math import sqrt
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

import torch
from torch.nn import functional

from lofi.nn.sites import Access, DTypeClass
from lofi.probe._boundary import SaeProbeSettings, SaeProbeSpec
from lofi.probe._causal import Side, variable_label
from lofi.probe._instrument import instrument
from lofi.probe._probing import (
    CausalHeldOutPass,
    ProbeBuildContext,
    ProbeValue,
    UnpublishedSiteError,
)
from lofi.registry import ComponentPlugin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site, SiteSpec
    from lofi.probe._causal import HighLevelVariable
    from lofi.probe._ports import ModelReadyCausalPair, Subject
    from lofi.probe._probing import TrainedPoint

__all__ = [
    "SaeExperimentPass",
    "SaeProbe",
    "SaeProbeSettings",
    "SaeProbeSpec",
]


@runtime_checkable
class SaeExperimentPass(CausalHeldOutPass, Protocol):
    """A causal pass exposing the populations and subject an SAE compares to DAS."""

    def causal_subject(self) -> Subject[Reading, ModelReadyCausalPair]:
        """Return the instrumentable subject whose published site is encoded."""
        ...

    def causal_fitting_pairs(self) -> tuple[ModelReadyCausalPair, ...]:
        """Return the model-ready pairs used to fit the sparse autoencoder."""
        ...

    def causal_held_out_pairs(self) -> tuple[ModelReadyCausalPair, ...]:
        """Return the disjoint pairs used only for measurement."""
        ...


class _Fitted(NamedTuple):
    """Detached parameters of one fitted sparse autoencoder."""

    encoder: Tensor
    encoder_bias: Tensor
    decoder: Tensor
    decoder_bias: Tensor


class _Examples(NamedTuple):
    """Activation vectors and declared decisions for one high-level variable."""

    vectors: Tensor
    labels: tuple[str | int, ...]


def _ordinary(measure: str, site: Site, value: float, count: int, /) -> ProbeValue:
    """Return one non-inferential SAE measurement."""
    return ProbeValue(measure, site, value, count, None, None, None, None, None, None)


def _pass(candidate: CausalHeldOutPass, position: int, /) -> SaeExperimentPass:
    """Return a causal pass that publishes the SAE populations and subject."""
    if not isinstance(candidate, SaeExperimentPass):
        msg = (
            f"causal held-out pass {position} publishes no SAE subject, fitting "
            "population, or held-out population"
        )
        raise TypeError(msg)
    return candidate


def _group(
    pairs: Sequence[ModelReadyCausalPair], /
) -> tuple[tuple[HighLevelVariable, tuple[ModelReadyCausalPair, ...]], ...]:
    """Return causal pairs grouped by variable in first-occurrence order."""
    grouped: dict[HighLevelVariable, list[ModelReadyCausalPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.varied, []).append(pair)
    return tuple((variable, tuple(members)) for variable, members in grouped.items())


def _vectors(
    subject: Subject[Reading, ModelReadyCausalPair],
    site: Site,
    pairs: tuple[ModelReadyCausalPair, ...],
    side: Side,
    /,
) -> Tensor:
    """Return one site vector per causal pair on one side.

    A rank-two reading already has one vector per item.  A rank-three reading
    has one row sequence per item; the pair's public tokenizer alignment selects
    the rows whose mean represents the varied source span.
    """
    if not pairs:
        msg = "an SAE reads at least one causal pair"
        raise ValueError(msg)
    with torch.no_grad(), instrument(subject, observe=frozenset({site})) as region:
        region(lambda: subject.run(pairs, side))
    reading = region.trace[site]
    tensor = reading.tensor
    if int(tensor.shape[0]) != len(pairs):
        msg = (
            f"{site!r} must hold one outer activation row per causal pair, not axes "
            f"{reading.axes!r} and shape {tuple(tensor.shape)!r} for {len(pairs)} pairs"
        )
        raise ValueError(msg)
    if tensor.ndim == 2:
        return tensor.detach().to(dtype=torch.float64, device="cpu")
    if tensor.ndim != 3:
        msg = (
            f"{site!r} must hold item-by-width or item-by-row-by-width "
            f"activations, not axes {reading.axes!r}"
        )
        raise ValueError(msg)
    selected: list[Tensor] = []
    for position, pair in enumerate(pairs):
        rows = pair.rows(side)
        if not rows or max(rows) >= int(tensor.shape[1]):
            msg = (
                f"causal pair {pair.id!r} selects rows {rows!r} outside the "
                f"{tensor.shape[1]} rows at {site!r}"
            )
            raise ValueError(msg)
        selected.append(tensor[position, rows].mean(dim=0))
    return torch.stack(selected).detach().to(dtype=torch.float64, device="cpu")


def _examples(
    subject: Subject[Reading, ModelReadyCausalPair],
    site: Site,
    pairs: tuple[ModelReadyCausalPair, ...],
    /,
) -> _Examples:
    """Return both sides' activation vectors and decisions for one variable."""
    variable = pairs[0].varied
    if any(pair.varied != variable for pair in pairs):
        msg = "one SAE activation batch must vary one high-level variable"
        raise ValueError(msg)
    base = _vectors(subject, site, pairs, Side.BASE)
    variant = _vectors(subject, site, pairs, Side.VARIANT)
    labels = tuple(pair.base[variable] for pair in pairs) + tuple(
        pair.variant[variable] for pair in pairs
    )
    return _Examples(torch.cat((base, variant)), labels)


def _fit(vectors: Tensor, settings: SaeProbeSettings, /) -> _Fitted:
    """Return a sparse autoencoder fitted to finite rank-two vectors."""
    if vectors.ndim != 2 or not vectors.numel():
        msg = f"an SAE fits a nonempty matrix of vectors, not {tuple(vectors.shape)!r}"
        raise ValueError(msg)
    if not bool(torch.isfinite(vectors).all().item()):
        msg = "an SAE fitting population contains a nonfinite activation"
        raise ValueError(msg)
    input_width = int(vectors.shape[1])
    draws = torch.Generator(device="cpu").manual_seed(settings.seed)
    radius = sqrt(6.0 / (input_width + settings.width))
    encoder = torch.empty(input_width, settings.width, dtype=vectors.dtype)
    encoder.uniform_(-radius, radius, generator=draws)
    decoder = encoder.T.clone()
    encoder = torch.nn.Parameter(encoder)
    decoder = torch.nn.Parameter(functional.normalize(decoder, dim=1))
    encoder_bias = torch.nn.Parameter(torch.zeros(settings.width, dtype=vectors.dtype))
    decoder_bias = torch.nn.Parameter(torch.zeros(input_width, dtype=vectors.dtype))
    optimizer = torch.optim.Adam(
        (encoder, encoder_bias, decoder, decoder_bias), lr=settings.learning_rate
    )
    for _ in range(settings.epochs):
        optimizer.zero_grad(set_to_none=True)
        code = functional.relu(vectors @ encoder + encoder_bias)
        reconstruction = code @ decoder + decoder_bias
        loss = (
            functional.mse_loss(reconstruction, vectors)
            + settings.sparsity_penalty * code.mean()
        )
        if not bool(torch.isfinite(loss).item()):
            msg = "an SAE fitting objective became nonfinite"
            raise FloatingPointError(msg)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            decoder.copy_(functional.normalize(decoder, dim=1))
    return _Fitted(
        encoder.detach(),
        encoder_bias.detach(),
        decoder.detach(),
        decoder_bias.detach(),
    )


def _encode(fitted: _Fitted, vectors: Tensor, /) -> Tensor:
    """Return nonnegative feature activations for a matrix of site vectors."""
    return functional.relu(vectors @ fitted.encoder + fitted.encoder_bias)


def _reconstruct(fitted: _Fitted, vectors: Tensor, /) -> tuple[Tensor, Tensor]:
    """Return feature activations and their decoded reconstruction."""
    code = _encode(fitted, vectors)
    return code, code @ fitted.decoder + fitted.decoder_bias


def _alignment(
    fitting: _Examples,
    held_out: _Examples,
    decisions: tuple[str | int, ...],
    fitted: _Fitted,
    /,
) -> tuple[int, float]:
    """Return the fitting-selected feature and its held-out centroid accuracy."""
    observed = tuple(decision for decision in decisions if decision in fitting.labels)
    if len(observed) < 2:
        msg = "feature alignment requires at least two observed decisions"
        raise ValueError(msg)
    if any(label not in observed for label in held_out.labels):
        msg = "an SAE held-out decision was absent from its fitting population"
        raise ValueError(msg)
    fitting_code = _encode(fitted, fitting.vectors)
    held_code = _encode(fitted, held_out.vectors)
    centroids = torch.stack(
        tuple(
            fitting_code[
                torch.tensor([label == decision for label in fitting.labels])
            ].mean(dim=0)
            for decision in observed
        )
    )
    fitting_targets = torch.tensor(
        tuple(observed.index(label) for label in fitting.labels)
    )
    fitting_predictions = (
        -(fitting_code[:, None, :] - centroids[None, :, :]).square()
    ).argmax(dim=1)
    feature_accuracy = (
        (fitting_predictions == fitting_targets[:, None]).to(torch.float64).mean(dim=0)
    )
    feature = int(feature_accuracy.argmax().item())
    held_targets = torch.tensor(
        tuple(observed.index(label) for label in held_out.labels)
    )
    held_predictions = (
        -(held_code[:, None, feature] - centroids[None, :, feature]).square()
    ).argmax(dim=1)
    accuracy = float((held_predictions == held_targets).to(torch.float64).mean().item())
    return feature, accuracy


class SaeProbe:
    """Fit and assess a sparse autoencoder at one published activation site."""

    __slots__ = ("_settings",)

    def __init__(self, settings: SaeProbeSettings) -> None:
        self._settings = settings

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse a site that is not an observable real activation matrix."""
        site = self._settings.site
        if site not in published:
            msg = f"an SAE probe names unpublished site {site!r}"
            raise UnpublishedSiteError(msg)
        specification = published[site]
        if (
            Access.OBSERVE not in specification.access
            or specification.dtype_class is not DTypeClass.REAL
            or specification.rank not in {2, 3}
        ):
            msg = (
                f"SAE site {site!r} is not an observable real outer-by-feature or "
                "outer-by-row-by-feature activation"
            )
            raise UnpublishedSiteError(msg)

    def _run_pass(
        self, held: SaeExperimentPass, position: int, /
    ) -> tuple[ProbeValue, ...]:
        """Return reconstruction, sparsity, and feature alignment for one pass."""
        fitting_pairs = held.causal_fitting_pairs()
        held_pairs = held.causal_held_out_pairs()
        if not fitting_pairs or not held_pairs:
            msg = "an SAE requires nonempty fitting and held-out causal populations"
            raise ValueError(msg)
        if {pair.id for pair in fitting_pairs} & {pair.id for pair in held_pairs}:
            msg = "SAE fitting and held-out causal identities overlap"
            raise ValueError(msg)
        fitting_groups = _group(fitting_pairs)
        held_groups = dict(_group(held_pairs))
        fitting_variables = tuple(variable for variable, _pairs in fitting_groups)
        if set(fitting_variables) != set(held_groups):
            msg = "SAE fitting and held-out populations vary different variables"
            raise ValueError(msg)
        subject = held.causal_subject()
        fitting = {
            variable: _examples(subject, self._settings.site, pairs)
            for variable, pairs in fitting_groups
        }
        held_out = {
            variable: _examples(subject, self._settings.site, held_groups[variable])
            for variable in fitting_variables
        }
        fitted = _fit(
            torch.cat(tuple(examples.vectors for examples in fitting.values())),
            self._settings,
        )
        scored = torch.cat(tuple(examples.vectors for examples in held_out.values()))
        code, reconstructed = _reconstruct(fitted, scored)
        rows: list[ProbeValue] = [
            _ordinary(
                f"pass-{position}/reconstruction-mse",
                self._settings.site,
                float(functional.mse_loss(reconstructed, scored).item()),
                int(scored.shape[0]),
            ),
            _ordinary(
                f"pass-{position}/activation-sparsity",
                self._settings.site,
                float((code == 0.0).to(torch.float64).mean().item()),
                int(code.numel()),
            ),
        ]
        for variable in fitting_variables:
            feature, accuracy = _alignment(
                fitting[variable],
                held_out[variable],
                subject.decisions(variable),
                fitted,
            )
            label = variable_label(variable)
            count = len(held_out[variable].labels)
            rows.extend(
                (
                    _ordinary(
                        f"pass-{position}/feature-alignment/{label}/feature",
                        self._settings.site,
                        float(feature),
                        count,
                    ),
                    _ordinary(
                        f"pass-{position}/feature-alignment/{label}/accuracy",
                        self._settings.site,
                        accuracy,
                        count,
                    ),
                )
            )
        return tuple(rows)

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return SAE rows over every causal held-out pass."""
        rows: list[ProbeValue] = []
        for position, candidate in enumerate(point.passes):
            if not isinstance(candidate, CausalHeldOutPass):
                msg = f"held-out pass {position} publishes no causal population"
                raise TypeError(msg)
            rows.extend(self._run_pass(_pass(candidate, position), position))
        return tuple(rows)


def _build(settings: SaeProbeSettings, context: ProbeBuildContext, /) -> SaeProbe:
    """Build an SAE probe after checking every held-out pass capability."""
    for position, candidate in enumerate(context.causal_passes()):
        _pass(candidate, position)
    return SaeProbe(settings)


PLUGIN: Final = ComponentPlugin(spec=SaeProbeSpec, build=_build)
"""The plugin the ``sae`` probe kind registers."""

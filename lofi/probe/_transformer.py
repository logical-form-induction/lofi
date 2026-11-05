"""Observation-only diagnostics for Transformer internals."""

from itertools import combinations
from typing import TYPE_CHECKING, ClassVar, Final, final, override

import torch

from lofi._frozen import FrozenValue
from lofi.nn.sites import DTypeClass, Role
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._scores import (
    maximum_probability,
    normalized_entropy,
    normalized_js_divergence,
    sparsity,
)
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.nn.sites import Site
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._observation_groups import ObservationGrouping
    from lofi.probe._values import Trace

__all__ = [
    "ATTENTION_POLICY",
    "AttentionPatterns",
    "AttentionPolicy",
    "FeedForwardSpecialization",
    "ResidualContributions",
]

_ATTENTION_AXES: Final = ("batch", "position", "attended-position")
_ACTIVATION_AXES: Final = ("batch", "position", "feature")
_INPUT: Final = "input"
_POST_ATTENTION: Final = "post-attention"
_OUTPUT: Final = "output"

_ENTROPY: Final = "normalized-entropy"
_COVERAGE: Final = "significant-mass-coverage"
_CONCENTRATION: Final = "maximum-concentration"
_CONSISTENCY: Final = "head-consistency"
_SELF_MASS: Final = "self-mass"
_LOCAL_MASS: Final = "local-mass"

_SPARSITY: Final = "sparsity"
_ETA_MEAN: Final = "eta-squared-mean"
_ETA_MAXIMUM: Final = "eta-squared-maximum"

_RESULTANT_SHARE: Final = "resultant-norm-share"
_SKIP_SHARE: Final = "skip-norm-share"

type _ResidualLayers = tuple[
    tuple[
        tuple[Site, Observation],
        tuple[Site, Observation],
        tuple[Site, Observation],
    ],
    ...,
]


@final
class AttentionPolicy(FrozenValue):
    """The single threshold and radius used by attention summaries.

    The fixed threshold marks an individual attention weight as significant. The
    fixed radius includes the query itself and one position on either side when local
    mass is measured. The policy takes no arguments so analysis cannot silently tune
    either choice from one run to another.
    """

    __slots__ = ()

    _noun = "attention policy"

    @property
    def significant_mass(self) -> float:
        """Return the inclusive minimum weight counted as significant."""
        return 0.1

    @property
    def local_radius(self) -> int:
        """Return the inclusive positional radius counted as local."""
        return 1

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is this one fixed policy."""
        if type(other) is not AttentionPolicy:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the hash of the one attention policy."""
        return hash(AttentionPolicy)

    @override
    def __repr__(self) -> str:
        """Return the constructor call for the fixed policy."""
        return "AttentionPolicy()"


ATTENTION_POLICY: Final = AttentionPolicy()
"""The fixed policy under which every attention-pattern result is reported."""


def _observed(
    context: DiagnosticContext,
    taken: Mapping[Site, Observation],
    role: Role,
) -> tuple[tuple[Site, Observation], ...]:
    """Return every published observation for one role, refusing omissions.

    Parameters
    ----------
    context : DiagnosticContext
        What the components under study publish.
    taken : Mapping[Site, Observation]
        What the scope observed.
    role : Role
        The role to collect.

    Returns
    -------
    tuple[tuple[Site, Observation], ...]
        The sites and observations in publication order.

    Raises
    ------
    ValueError
        If a published site was not observed.
    """
    published = context.sites_for(role)

    for site in published:
        if site not in taken:
            msg = f"missing observation at published {role.name} site {site!r}"
            raise ValueError(msg)

    return tuple((site, taken[site]) for site in published)


def _activation(
    context: DiagnosticContext, site: Site, reading: Observation, role: Role
) -> None:
    """Validate one Transformer activation declaration and observation.

    Parameters
    ----------
    context : DiagnosticContext
        What the site publishes.
    site : Site
        The site being checked.
    reading : Observation
        What was observed there.
    role : Role
        The activation role expected there.

    Raises
    ------
    ValueError
        If the role, axes, numeric class or storage dtype is not the Transformer
        activation contract.
    """
    spec = context.spec_at(site)

    if site.role is not role:
        msg = f"{site!r} does not play the expected {role.name} role"
        raise ValueError(msg)

    if spec.axes != _ACTIVATION_AXES or reading.axes != _ACTIVATION_AXES:
        msg = (
            f"{site!r} must publish and observe axes {_ACTIVATION_AXES!r}, not "
            f"{spec.axes!r} and {reading.axes!r}"
        )
        raise ValueError(msg)

    if spec.dtype_class is not DTypeClass.REAL:
        msg = f"{site!r} must publish real numbers, not {spec.dtype_class.value}"
        raise ValueError(msg)

    if not reading.tensor.is_floating_point():
        msg = f"{site!r} must hold floating-point activations"
        raise ValueError(msg)

    if reading.extent("feature") < 1:
        msg = f"{site!r} holds no features"
        raise ValueError(msg)


def _attention_distribution(
    context: DiagnosticContext,
    logits_site: Site,
    logits: Observation,
    weights_site: Site,
    weights: Observation,
) -> Observation:
    """Return normalized observed weights under their observed finite support.

    Parameters
    ----------
    context : DiagnosticContext
        What both sites publish.
    logits_site : Site
        The attention-logit site.
    logits : Observation
        The exact logits observed there.
    weights_site : Site
        The corresponding attention-weight site.
    weights : Observation
        The exact mixing weights observed there.

    Returns
    -------
    Observation
        The weights in float64 with padded and structurally unavailable keys masked.

    Raises
    ------
    ValueError
        If the sites, declarations, observations, masks or values do not form one
        published head.
    """
    logits_spec = context.spec_at(logits_site)
    weights_spec = context.spec_at(weights_site)

    if logits_site.key != weights_site.key:
        msg = f"attention sites {logits_site!r} and {weights_site!r} are not paired"
        raise ValueError(msg)

    if (
        logits_spec.axes != _ATTENTION_AXES
        or weights_spec.axes != _ATTENTION_AXES
        or logits.axes != _ATTENTION_AXES
        or weights.axes != _ATTENTION_AXES
    ):
        msg = f"attention pair {weights_site!r} must use axes {_ATTENTION_AXES!r}"
        raise ValueError(msg)

    if logits_spec.dtype_class is not DTypeClass.REAL:
        msg = f"{logits_site!r} must publish real attention logits"
        raise ValueError(msg)

    if weights_spec.dtype_class is not DTypeClass.PROBABILITY:
        msg = f"{weights_site!r} must publish attention probabilities"
        raise ValueError(msg)

    if not logits.tensor.is_floating_point() or not weights.tensor.is_floating_point():
        msg = f"attention pair {weights_site!r} must hold floating-point tensors"
        raise ValueError(msg)

    if logits.tensor.shape != weights.tensor.shape:
        msg = f"attention pair {weights_site!r} does not have one shape"
        raise ValueError(msg)

    if weights.extent("position") != weights.extent("attended-position"):
        msg = f"attention at {weights_site!r} is not square over its positions"
        raise ValueError(msg)

    if not torch.equal(logits.covering(), weights.covering()):
        msg = f"attention pair {weights_site!r} is masked differently"
        raise ValueError(msg)

    covered = weights.covering()
    valid = covered

    if weights.mask is None or weights.mask.ndim < weights.tensor.ndim:
        occupied = covered.any(dim=weights.position("attended-position"))
        valid = valid & occupied.unsqueeze(-1) & occupied.unsqueeze(-2)

    invalid_logits = torch.isnan(logits.tensor) | torch.isposinf(logits.tensor)

    if bool((invalid_logits & covered).any()):
        msg = f"attention logits at {logits_site!r} contain NaN or positive infinity"
        raise ValueError(msg)

    valid = valid & torch.isfinite(logits.tensor)
    values = weights.tensor.to(torch.float64)

    if bool(((~torch.isfinite(values) | (values < 0)) & covered).any()):
        msg = f"attention weights at {weights_site!r} are not finite nonnegative mass"
        raise ValueError(msg)

    if bool((values.masked_select(~valid) != 0).any()):
        msg = f"attention weights at {weights_site!r} put mass outside observed support"
        raise ValueError(msg)

    values = values.masked_fill(~valid, 0.0)
    mass = values.sum(dim=-1, keepdim=True)
    rows = valid.any(dim=-1, keepdim=True)

    if bool((rows & (mass <= 0)).any()):
        msg = f"a live attention row at {weights_site!r} holds no mass"
        raise ValueError(msg)

    normalized = torch.where(
        rows, values / mass.clamp(min=torch.finfo(values.dtype).tiny), 0.0
    )

    return Observation(weights_site, normalized, axes=_ATTENTION_AXES, mask=valid)


def _mean(reading: Observation) -> tuple[float, int]:
    """Return the mean over every live scalar position and its count.

    Parameters
    ----------
    reading : Observation
        A scalar score over any number of axes.

    Returns
    -------
    tuple[float, int]
        The live mean, or zero where none is live, and the live count.
    """
    live = reading.covering()
    count = int(live.sum())

    if not count:
        return 0.0, 0

    return float(reading.tensor.masked_select(live).to(torch.float64).mean()), count


def _attention_measures(reading: Observation) -> tuple[dict[str, float], int]:
    """Return every per-head summary over one normalized weight observation.

    Parameters
    ----------
    reading : Observation
        Normalized attention weights under their exact observed support.

    Returns
    -------
    tuple[dict[str, float], int]
        The five summaries and number of live query rows.
    """
    entropy = normalized_entropy(
        reading, DTypeClass.PROBABILITY, over="attended-position"
    )
    concentration = maximum_probability(
        reading, DTypeClass.PROBABILITY, over="attended-position"
    )
    entropy_mean, rows = _mean(entropy)
    concentration_mean, _ = _mean(concentration)
    valid = reading.covering()
    live = valid.any(dim=-1)
    significant = (reading.tensor >= ATTENTION_POLICY.significant_mass) & valid
    coverage = reading.tensor.masked_fill(~significant, 0.0).sum(dim=-1)
    positions = reading.extent("position")
    coordinates = torch.arange(positions, device=reading.tensor.device)
    distance = (coordinates.unsqueeze(1) - coordinates.unsqueeze(0)).abs()
    local = distance <= ATTENTION_POLICY.local_radius
    local_mass = reading.tensor.masked_fill(~(valid & local.unsqueeze(0)), 0.0).sum(
        dim=-1
    )
    diagonal = reading.tensor.diagonal(dim1=-2, dim2=-1)

    return (
        {
            _ENTROPY: entropy_mean,
            _COVERAGE: float(coverage.masked_select(live).mean()) if rows else 0.0,
            _CONCENTRATION: concentration_mean,
            _SELF_MASS: float(diagonal.masked_select(live).mean()) if rows else 0.0,
            _LOCAL_MASS: float(local_mass.masked_select(live).mean()) if rows else 0.0,
        },
        rows,
    )


def _attention_layers(
    heads: Sequence[tuple[Site, Observation]],
) -> tuple[tuple[tuple[Site, Observation], ...], ...]:
    """Return consecutive heads grouped by their public layer-key prefix.

    Parameters
    ----------
    heads : Sequence[tuple[Site, Observation]]
        Normalized heads in publication order.

    Returns
    -------
    tuple[tuple[tuple[Site, Observation], ...], ...]
        The heads of each layer, preserving both orders.

    Raises
    ------
    ValueError
        If a head key is malformed or repeats within a layer.
    """
    layers: list[tuple[tuple[Site, Observation], ...]] = []
    current: list[tuple[Site, Observation]] = []
    prefix: tuple[int | str, ...] | None = None
    indices: set[int | str] = set()

    for site, reading in heads:
        if len(site.key) < 2:
            msg = f"attention site {site!r} has no layer and head key"
            raise ValueError(msg)

        held = site.key[:-1]

        if current and held != prefix:
            layers.append(tuple(current))
            current = []
            indices = set()

        head = site.key[-1]

        if head in indices:
            msg = f"attention layer {held!r} repeats head key {head!r}"
            raise ValueError(msg)

        prefix = held
        indices.add(head)
        current.append((site, reading))

    if current:
        layers.append(tuple(current))

    return tuple(layers)


def _head_consistency(heads: Sequence[tuple[Site, Observation]]) -> float | None:
    """Return one minus mean normalized pairwise Jensen-Shannon divergence.

    Parameters
    ----------
    heads : Sequence[tuple[Site, Observation]]
        The heads of one layer.

    Returns
    -------
    float | None
        The cross-head consistency, or None where fewer than two heads or no live
        query pairs exist.
    """
    divergences: list[Tensor] = []

    for (_, left), (_, right) in combinations(heads, 2):
        scored = normalized_js_divergence(
            left, right, DTypeClass.PROBABILITY, over="attended-position"
        )
        live = scored.covering()

        if bool(live.any()):
            divergences.append(scored.tensor.masked_select(live).to(torch.float64))

    if not divergences:
        return None

    mean = torch.cat(divergences).mean().clamp(min=0.0, max=1.0)

    return float(1.0 - mean)


class AttentionPatterns:
    """Summarizes exact observed attention distributions without labeling heads."""

    __slots__ = ()

    tag: ClassVar[str] = "attention-patterns"
    requires: ClassVar[frozenset[Role]] = frozenset(
        {Role.ATTENTION_LOGITS, Role.ATTENTION_WEIGHTS}
    )
    scores: ClassVar[frozenset[str]] = frozenset(
        {
            _ENTROPY,
            _COVERAGE,
            _CONCENTRATION,
            _CONSISTENCY,
            _SELF_MASS,
            _LOCAL_MASS,
        }
    )

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return per-head summaries and layerwise head consistency.

        Every published logit and weight site must be observed. Logits determine the
        observed finite support, and weights supply every measured distribution. No
        attention score or weight is reconstructed.
        """
        taken = dict(context.readings(trace))
        logits = _observed(context, taken, Role.ATTENTION_LOGITS)
        weights = _observed(context, taken, Role.ATTENTION_WEIGHTS)

        if not logits and not weights:
            return DiagnosticResult(self.tag, [])

        if len(logits) != len(weights):
            msg = (
                f"published attention has {len(logits)} logit sites and "
                f"{len(weights)} weight sites, so the observations cannot be paired"
            )
            raise ValueError(msg)

        heads = tuple(
            (
                weights_site,
                _attention_distribution(
                    context,
                    logits_site,
                    logits_reading,
                    weights_site,
                    weights_reading,
                ),
            )
            for (logits_site, logits_reading), (weights_site, weights_reading) in zip(
                logits, weights, strict=True
            )
        )
        measured: list[Measurement] = []

        for site, reading in heads:
            summaries, rows = _attention_measures(reading)
            measured.append(Measurement(site, "queries", float(rows)))

            if rows:
                measured.extend(
                    Measurement(site, name, value) for name, value in summaries.items()
                )

        for layer in _attention_layers(heads):
            consistency = _head_consistency(layer)

            if consistency is not None:
                measured.extend(
                    Measurement(site, _CONSISTENCY, consistency) for site, _ in layer
                )

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "AttentionPatterns()"


def _group_effects(values: Tensor, grouping: ObservationGrouping) -> Tensor | None:
    """Return eta squared for every feature under declared row groups.

    Parameters
    ----------
    values : Tensor
        Rank-two float64 values over flat rows and features.
    grouping : ObservationGrouping
        The caller-declared row memberships.

    Returns
    -------
    Tensor | None
        One effect size per feature, or None where no group is declared.
    """
    if not grouping.groups:
        return None

    rows = tuple(row for group in grouping.groups for row in sorted(group))
    selected = values[list(rows)]
    grand = selected.mean(dim=0)
    total = (selected - grand).square().sum(dim=0)
    between = torch.zeros_like(total)

    for group in grouping.groups:
        member = values[sorted(group)]
        between = between + member.shape[0] * (member.mean(dim=0) - grand).square()

    return torch.where(total > 0, between / total, 0.0).clamp(min=0.0, max=1.0)


def _checked_grouping(
    site: Site, reading: Observation, grouping: ObservationGrouping
) -> tuple[Tensor, Tensor]:
    """Return flat values and live rows after checking a grouping against a reading.

    Parameters
    ----------
    site : Site
        The feed-forward site.
    reading : Observation
        Its preactivations.
    grouping : ObservationGrouping
        The caller-declared row groups.

    Returns
    -------
    tuple[Tensor, Tensor]
        Rank-two float64 values and a rank-one row-liveness mask.

    Raises
    ------
    ValueError
        If the grouping extent differs from the flattened observation or references a
        row that is not completely live.
    """
    features = reading.extent("feature")
    flat = reading.tensor.to(torch.float64).reshape(-1, features)
    live = reading.covering().all(dim=reading.position("feature")).reshape(-1)

    if grouping.rows != flat.shape[0]:
        msg = (
            f"the grouping at {site!r} declares {grouping.rows} rows, but the "
            f"observation flattens to {flat.shape[0]}"
        )
        raise ValueError(msg)

    for position, group in enumerate(grouping.groups):
        missing = sorted(row for row in group if not bool(live[row]))

        if missing:
            msg = (
                f"observation group {position} at {site!r} references non-live rows "
                f"{missing}"
            )
            raise ValueError(msg)

    return flat, live


class FeedForwardSpecialization:
    """Measures feature sparsity and declared-group effects in FFN preactivations."""

    __slots__ = ()

    tag: ClassVar[str] = "feed-forward-specialization"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.FFN_PREACTIVATION})
    scores: ClassVar[frozenset[str]] = frozenset({_SPARSITY, _ETA_MEAN, _ETA_MAXIMUM})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return sparsity and per-feature eta-squared summaries at every FFN site."""
        taken = dict(context.readings(trace))
        readings = _observed(context, taken, Role.FFN_PREACTIVATION)
        measured: list[Measurement] = []

        for site, reading in readings:
            _activation(context, site, reading, Role.FFN_PREACTIVATION)
            grouping = context.grouping_at(site)
            flat, live = _checked_grouping(site, reading, grouping)
            rows = int(live.sum())
            features = reading.extent("feature")
            measured.extend(
                (
                    Measurement(site, "rows", float(rows)),
                    Measurement(site, "neurons", float(features)),
                    Measurement(site, "groups", float(len(grouping.groups))),
                )
            )

            if rows:
                reshaped = Observation(
                    site,
                    flat,
                    axes=("row", "feature"),
                    mask=live,
                )
                sparse = sparsity(reshaped, DTypeClass.REAL, over="feature")
                sparse_mean, _ = _mean(sparse)
                measured.append(Measurement(site, _SPARSITY, sparse_mean))

            effects = _group_effects(flat, grouping)

            if effects is not None:
                measured.extend(
                    (
                        Measurement(site, _ETA_MEAN, float(effects.mean())),
                        Measurement(site, _ETA_MAXIMUM, float(effects.max())),
                    )
                )

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "FeedForwardSpecialization()"


def _residual_layers(
    readings: Sequence[tuple[Site, Observation]],
) -> _ResidualLayers:
    """Return residual observations as input, post-attention, output triples.

    Parameters
    ----------
    readings : Sequence[tuple[Site, Observation]]
        Residual sites in publication order.

    Returns
    -------
    _ResidualLayers
        One exact stage triple per layer.

    Raises
    ------
    ValueError
        If the sites do not form complete stage triples.
    """
    if len(readings) % 3:
        msg = f"{len(readings)} residual sites do not form complete stage triples"
        raise ValueError(msg)

    layers = []

    for offset in range(0, len(readings), 3):
        first, middle, last = readings[offset : offset + 3]
        stages = (first[0].key, middle[0].key, last[0].key)

        if any(len(key) < 2 for key in stages):
            msg = f"residual sites {stages!r} do not name a layer and stage"
            raise ValueError(msg)

        prefixes = tuple(key[:-1] for key in stages)
        names = tuple(key[-1] for key in stages)

        if len(set(prefixes)) != 1 or names != (_INPUT, _POST_ATTENTION, _OUTPUT):
            msg = (
                f"residual sites {stages!r} are not an input, post-attention, "
                "output triple"
            )
            raise ValueError(msg)

        layers.append((first, middle, last))

    return tuple(layers)


def _residual_shares(
    before_site: Site,
    before: Observation,
    after_site: Site,
    after: Observation,
) -> tuple[float, float, int]:
    """Return resultant and skip norm shares over one observed residual addition.

    Parameters
    ----------
    before_site : Site
        The stage supplying the skip vector.
    before : Observation
        The observed skip vectors.
    after_site : Site
        The stage after the addition.
    after : Observation
        The observed sum.

    Returns
    -------
    tuple[float, float, int]
        Mean resultant share, mean skip share and live-row count. The resultant is
        ``after - before``. If both norms are zero, both shares are one half.

    Raises
    ------
    ValueError
        If the stages differ in axes, shape or mask.
    """
    if before.axes != after.axes or before.tensor.shape != after.tensor.shape:
        msg = f"residual stages {before_site!r} and {after_site!r} are mispaired"
        raise ValueError(msg)

    if not torch.equal(before.covering(), after.covering()):
        msg = (
            f"residual stages {before_site!r} and {after_site!r} are masked differently"
        )
        raise ValueError(msg)

    dimension = before.position("feature")
    live = before.covering().all(dim=dimension)
    count = int(live.sum())

    if not count:
        return 0.0, 0.0, 0

    before_values = before.tensor.to(torch.float64)
    after_values = after.tensor.to(torch.float64)
    skip = before_values.norm(dim=dimension)
    resultant = (after_values - before_values).norm(dim=dimension)
    total = skip + resultant
    halves = torch.full_like(total, 0.5)
    resultant_share = torch.where(total > 0, resultant / total, halves)
    skip_share = torch.where(total > 0, skip / total, halves)

    return (
        float(resultant_share.masked_select(live).mean()),
        float(skip_share.masked_select(live).mean()),
        count,
    )


class ResidualContributions:
    """Measures update and skip norm shares from paired residual stages.

    The complete Transformer-internal role bundle distinguishes these staged streams
    from a generic residual-stream publication, whose keys carry no input,
    post-attention and output contract.
    """

    __slots__ = ()

    tag: ClassVar[str] = "residual-contributions"
    requires: ClassVar[frozenset[Role]] = frozenset(
        {
            Role.ATTENTION_LOGITS,
            Role.ATTENTION_WEIGHTS,
            Role.FFN_PREACTIVATION,
            Role.RESIDUAL_STREAM,
        }
    )
    scores: ClassVar[frozenset[str]] = frozenset({_RESULTANT_SHARE, _SKIP_SHARE})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return both norm shares for attention and feed-forward additions."""
        taken = dict(context.readings(trace))
        readings = _observed(context, taken, Role.RESIDUAL_STREAM)
        measured: list[Measurement] = []

        for first, middle, last in _residual_layers(readings):
            for site, reading in (first, middle, last):
                _activation(context, site, reading, Role.RESIDUAL_STREAM)

            for before, after in ((first, middle), (middle, last)):
                resultant, skip, rows = _residual_shares(
                    before[0], before[1], after[0], after[1]
                )
                measured.append(Measurement(after[0], "rows", float(rows)))

                if rows:
                    measured.extend(
                        (
                            Measurement(after[0], _RESULTANT_SHARE, resultant),
                            Measurement(after[0], _SKIP_SHARE, skip),
                        )
                    )

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "ResidualContributions()"

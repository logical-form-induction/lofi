from typing import TYPE_CHECKING, Self, cast, override

import pytest
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import one_hot

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._erasure import (
    ErasureReport,
    LinearErasure,
    MetricDirection,
    fit_leace,
)
from lofi.probe._instrument import instrument
from lofi.probe._values import Observation
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from lofi.nn.ports import Patch

SITE = Site(ComponentId("encoder"), Role.NODE_ENCODING)


class _Held:
    """A masked activation exposed by the instrumented test component."""

    def __init__(self, tensor: Tensor, mask: Tensor) -> None:
        self._tensor = tensor
        self._mask = mask

    @property
    def site(self) -> Site:
        return SITE

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return ("item", "node", "width")

    @property
    def mask(self) -> Tensor:
        return self._mask


class _MaskedScope:
    """Applies public probe patches inside one model pass."""

    def __init__(
        self,
        model: _MaskedModel,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[_Held]],
    ) -> None:
        self._model = model
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Held] = {}

    @property
    def readings(self) -> Mapping[Site, _Held]:
        return self._readings

    def intervene(self, tensor: Tensor, mask: Tensor) -> Tensor:
        held = _Held(tensor, mask)
        patch = self._patches.get(SITE)
        changed = tensor if patch is None else patch.stands_in_for(held)

        if SITE in self._sites:
            self._readings[SITE] = _Held(changed, mask)

        return changed

    def __enter__(self) -> Self:
        self._model.scopes.append(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        del kind, error, traceback
        self._model.scopes.remove(self)

        return None


class _MaskedModel(torch.nn.Module):
    """Publishes one masked rank-three activation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(3, 3, dtype=torch.float32)
        self.register_buffer("offset", torch.tensor([0.25, -0.5, 0.75]))
        self.scopes: list[_MaskedScope] = []

    def site_map(self) -> SiteMap:
        return SiteMap.custom(
            SITE.component,
            {
                SITE: SiteSpec(
                    ("item", "node", "width"),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE | Access.PATCH,
                )
            },
        )

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _MaskedScope:
        return _MaskedScope(self, sites, patches)

    @override
    def forward(self, source: Tensor, mask: Tensor) -> Tensor:
        hidden = self.linear(source) + self.offset

        for scope in self.scopes:
            hidden = scope.intervene(hidden, mask)

        return hidden


def state_bytes(model: _MaskedModel) -> tuple[tuple[str, bytes], ...]:
    """Return raw model-state bytes in state-dictionary order."""
    return tuple(
        (name, value.detach().numpy().tobytes())
        for name, value in model.state_dict().items()
    )


def parameter_bytes(model: _MaskedModel) -> tuple[bytes, ...]:
    """Return raw parameter bytes in module order."""
    return tuple(value.detach().numpy().tobytes() for value in model.parameters())


def fixture(
    repetitions: int, *, nuisance_offset: float = 0.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Return anisotropic representations and two independent binary concepts."""
    mixing = torch.tensor(
        [[1.0, 2.0, 0.2], [0.4, -1.0, 0.5], [1.2, 0.3, 2.0]],
        dtype=torch.float64,
    )
    rows: list[Tensor] = []
    targets: list[int] = []
    controls: list[int] = []

    for repetition in range(repetitions):
        for target in (0, 1):
            for control in (0, 1):
                latent = torch.tensor(
                    [
                        3.0 * (2 * target - 1),
                        0.7 * (2 * control - 1),
                        ((repetition % 5) - 2) / 10 + nuisance_offset,
                    ],
                    dtype=torch.float64,
                )
                rows.append(latent @ mixing)
                targets.append(target)
                controls.append(control)

    return torch.stack(rows), torch.tensor(targets), torch.tensor(controls)


def accuracy(
    train: Tensor,
    train_labels: Tensor,
    test: Tensor,
    test_labels: Tensor,
) -> float:
    """Return held-out least-squares classification accuracy."""
    classes = int(train_labels.max().item()) + 1
    targets = one_hot(train_labels, classes).to(dtype=torch.float64)
    design = torch.cat((torch.ones(len(train), 1), train.to(torch.float64)), dim=1)
    coefficients = torch.linalg.lstsq(design, targets).solution
    heldout = torch.cat((torch.ones(len(test), 1), test.to(torch.float64)), dim=1)

    return float((heldout @ coefficients).argmax(dim=1).eq(test_labels).double().mean())


def covariance(left: Tensor, right: Tensor) -> Tensor:
    """Return the empirical cross-covariance of two row-aligned matrices."""
    right = right.to(torch.float64)
    centered_left = left - left.mean(dim=0)
    centered_right = right - right.mean(dim=0)

    return centered_left.T @ centered_right / len(left)


def applied(erasure: LinearErasure, source: Tensor) -> Tensor:
    """Apply an erasure through its activation-intervention surface."""
    return erasure.stands_in_for(
        Observation(
            SITE,
            source,
            axes=tuple(f"axis-{axis}" for axis in range(source.ndim)),
        )
    )


def test_leace_erases_target_predictability_without_erasing_an_unrelated_concept() -> (
    None
):
    train, target, control = fixture(20)
    test, heldout_target, heldout_control = fixture(10, nuisance_offset=0.025)
    concepts = one_hot(target, 2)
    erasure = fit_leace(train, concepts)
    erased_train = applied(erasure, train)
    erased_test = applied(erasure, test)

    before_target = accuracy(train, target, test, heldout_target)
    after_target = accuracy(erased_train, target, erased_test, heldout_target)
    before_control = accuracy(train, control, test, heldout_control)
    after_control = accuracy(erased_train, control, erased_test, heldout_control)
    zeroed_control = accuracy(
        torch.zeros_like(train),
        control,
        torch.zeros_like(test),
        heldout_control,
    )

    assert before_target == 1.0
    assert after_target == 0.5
    assert abs(after_control - before_control) <= 0.02
    assert zeroed_control == 0.5
    assert after_control - zeroed_control > 0.02
    assert covariance(erased_train, concepts).abs().max().item() <= 1e-9


def test_the_affine_map_uses_the_column_projector_on_row_batches() -> None:
    representations, target, _ = fixture(20)
    concepts = one_hot(target, 2)
    erasure = fit_leace(representations, concepts)
    centered = representations - erasure.mean
    correct = centered @ erasure.projection.T + erasure.mean
    wrong = centered @ erasure.projection + erasure.mean

    assert torch.allclose(applied(erasure, representations), correct, atol=1e-12)
    assert covariance(correct, concepts).abs().max().item() <= 1e-9
    assert covariance(wrong, concepts).abs().max().item() > 1e-3


def test_the_erasure_preserves_the_mean_and_is_idempotent() -> None:
    representations, target, _ = fixture(12)
    erasure = fit_leace(representations, one_hot(target, 2))
    once = applied(erasure, representations)
    twice = applied(erasure, once)

    assert torch.allclose(once.mean(dim=0), representations.mean(dim=0), atol=1e-12)
    assert torch.allclose(twice, once, atol=1e-12)
    assert torch.allclose(
        erasure.projection @ erasure.projection,
        erasure.projection,
        atol=1e-12,
    )

    exposed_mean = erasure.mean
    exposed_projection = erasure.projection
    exposed_mean.zero_()
    exposed_projection.zero_()

    assert torch.allclose(applied(erasure, representations), once, atol=1e-12)


def test_singular_representations_are_erased_through_their_supported_subspace() -> None:
    representations, target, _ = fixture(12)
    singular = torch.cat((representations, representations[:, :1]), dim=1)
    concepts = one_hot(target, 2)
    erasure = fit_leace(singular, concepts)
    erased = applied(erasure, singular)

    assert torch.isfinite(erased).all()
    assert covariance(erased, concepts).abs().max().item() <= 1e-9
    assert torch.allclose(applied(erasure, erased), erased, atol=1e-11)


def test_public_instrument_applies_erasure_only_at_live_masked_positions() -> None:
    representations, target, _ = fixture(10)
    source = representations[:12].to(torch.float32).reshape(2, 6, 3)
    model = _MaskedModel()
    fitted = model(representations.to(torch.float32), torch.ones(40, dtype=torch.bool))
    erasure = fit_leace(fitted.detach(), one_hot(target, 2))
    mask = torch.tensor(
        [
            [True, False, True, False, True, False],
            [True, True, False, False, False, True],
        ]
    )
    clean = model(source, mask).detach()
    input_snapshot = source.numpy().tobytes()
    model_snapshot = state_bytes(model)
    parameters_snapshot = parameter_bytes(model)

    with instrument(
        model,
        observe=frozenset({SITE}),
        patch={SITE: erasure},
    ) as run:
        changed = run(lambda: model(source, mask)).detach()

    covering = mask.unsqueeze(-1).expand_as(source)

    assert changed.shape == clean.shape
    assert changed.dtype is clean.dtype
    assert changed.device == clean.device
    assert torch.equal(changed[~covering], clean[~covering])
    assert not torch.equal(changed[covering], clean[covering])
    assert run.trace[SITE].mask is not None
    assert torch.equal(cast("Tensor", run.trace[SITE].mask), mask)
    assert source.numpy().tobytes() == input_snapshot
    assert state_bytes(model) == model_snapshot
    assert parameter_bytes(model) == parameters_snapshot
    assert model.scopes == []


def test_direct_application_preserves_shape_dtype_and_device_at_rank_four() -> None:
    representations, target, _ = fixture(10)
    erasure = fit_leace(representations, one_hot(target, 2))
    source = representations[:12].to(torch.float32).reshape(2, 2, 3, 3)
    replacement = applied(erasure, source)

    assert replacement.shape == source.shape
    assert replacement.dtype is source.dtype
    assert replacement.device == source.device


def test_fitting_does_not_mutate_representations_or_concepts() -> None:
    representations, target, _ = fixture(10)
    fitting_bytes = representations.numpy().tobytes()
    concepts = one_hot(target, 2)
    concept_bytes = concepts.numpy().tobytes()

    _ = fit_leace(representations, concepts)

    assert representations.numpy().tobytes() == fitting_bytes
    assert concepts.numpy().tobytes() == concept_bytes


@pytest.mark.parametrize(
    ("representations", "concepts", "error", "match"),
    [
        (torch.ones(3), torch.eye(3), ValueError, "representations"),
        (
            torch.ones(3, 2, dtype=torch.int64),
            torch.eye(3),
            TypeError,
            "representations",
        ),
        (torch.empty(0, 2), torch.empty(0, 2), ValueError, "representations"),
        (
            torch.tensor([[1.0, float("nan")]]),
            torch.tensor([[1.0, 0.0]]),
            ValueError,
            "representations",
        ),
        (torch.ones(3, 2), torch.ones(3), ValueError, "concepts"),
        (torch.ones(3, 2), torch.empty(3, 0), ValueError, "concepts"),
        (
            torch.ones(3, 2),
            torch.tensor([[1.0, 0.0], [0.0, float("inf")], [1.0, 0.0]]),
            ValueError,
            "concepts",
        ),
        (
            torch.ones(3, 2),
            torch.tensor([[1, 0], [0, 0], [0, 1]]),
            ValueError,
            "one-hot",
        ),
        (torch.ones(3, 2), torch.eye(2), ValueError, "rows"),
    ],
)
def test_malformed_fitting_data_are_refused_by_name(
    representations: Tensor,
    concepts: Tensor,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        fit_leace(representations, concepts)


def test_application_refuses_a_nonfloating_nonfinite_or_wrong_width_source() -> None:
    representations, target, _ = fixture(10)
    erasure = fit_leace(representations, one_hot(target, 2))

    with pytest.raises(TypeError, match="source"):
        applied(erasure, torch.ones(2, 3, dtype=torch.int64))

    with pytest.raises(ValueError, match="source"):
        applied(erasure, torch.tensor([[1.0, float("nan"), 2.0]]))

    with pytest.raises(ValueError, match="innermost"):
        applied(erasure, torch.ones(2, 2))


def score_pairs() -> tuple[
    tuple[Sequence[float], Sequence[float]],
    tuple[Sequence[float], Sequence[float]],
    tuple[Sequence[float], Sequence[float]],
]:
    """Return three nondegenerate paired comparisons."""
    return (
        ((0.9, 0.7, 0.8, 0.6), (0.5, 0.8, 0.4, 0.55)),
        ((1.0, 0.8, 0.9, 0.7), (0.55, 0.6, 0.5, 0.65)),
        ((0.8, 0.65, 0.75, 0.7), (0.79, 0.66, 0.73, 0.71)),
    )


def test_an_erasure_report_separates_three_paired_drops_and_their_intervals() -> None:
    behavioral, decodability, control = score_pairs()
    report = ErasureReport(
        behavioral,
        decodability,
        control,
        behavioral_direction="higher",
        decodability_direction="higher",
        control_direction="higher",
        bootstrap=Bootstrap(unit="item", replicates=99, seed=7, level=0.95),
    )

    assert report.behavioral_drop.value > 0.0
    assert report.decodability_drop.value > 0.0
    assert abs(report.control_drop.value) < report.behavioral_drop.value
    assert report.behavioral_drop.interval.unit == "item"
    assert report.decodability_drop.interval.replicates == 99
    assert report.control_drop.interval.seed == 7
    assert report.behavioral_direction == "higher"
    assert report.decodability_direction == "higher"
    assert report.control_direction == "higher"
    assert (
        len({report.behavioral_drop, report.decodability_drop, report.control_drop})
        == 3
    )


def test_lower_is_better_scores_report_worsening_as_a_positive_drop() -> None:
    higher_is_better = ErasureReport(
        ((0.9, 0.8, 0.7, 0.6), (0.5, 0.4, 0.3, 0.2)),
        ((1.0, 0.9, 0.8, 0.7), (0.6, 0.5, 0.4, 0.3)),
        ((0.7, 0.6, 0.5, 0.4), (0.69, 0.61, 0.49, 0.41)),
        behavioral_direction="higher",
        decodability_direction="higher",
        control_direction="higher",
        bootstrap=Bootstrap(unit="item", replicates=19, seed=5, level=0.95),
    )
    lower_is_better = ErasureReport(
        ((0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8)),
        ((0.2, 0.3, 0.4, 0.5), (0.6, 0.7, 0.8, 0.9)),
        ((0.4, 0.5, 0.6, 0.7), (0.41, 0.49, 0.61, 0.69)),
        behavioral_direction="lower",
        decodability_direction="lower",
        control_direction="lower",
        bootstrap=Bootstrap(unit="item", replicates=19, seed=5, level=0.95),
    )

    assert higher_is_better.behavioral_drop.value > 0.0
    assert higher_is_better.decodability_drop.value > 0.0
    assert lower_is_better.behavioral_drop.value > 0.0
    assert lower_is_better.decodability_drop.value > 0.0
    assert lower_is_better.behavioral_drop.value == pytest.approx(
        higher_is_better.behavioral_drop.value
    )
    assert lower_is_better.decodability_drop.value == pytest.approx(
        higher_is_better.decodability_drop.value
    )
    assert lower_is_better != higher_is_better


def test_an_erasure_report_is_immutable() -> None:
    report = ErasureReport(
        *score_pairs(),
        behavioral_direction="higher",
        decodability_direction="higher",
        control_direction="higher",
        bootstrap=Bootstrap(unit="item", replicates=19, seed=3, level=0.95),
    )
    field = "behavioral_drop"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(report, field, cast("object", report.control_drop))


@pytest.mark.parametrize(
    ("behavioral", "decodability", "control", "bootstrap", "error", "match"),
    [
        (
            ((1.0,), ()),
            ((1.0,), (0.0,)),
            ((1.0,), (0.0,)),
            Bootstrap(unit="item", replicates=9, seed=1, level=0.95),
            ValueError,
            "behavioral",
        ),
        (
            ((), ()),
            ((1.0,), (0.0,)),
            ((1.0,), (0.0,)),
            Bootstrap(unit="item", replicates=9, seed=1, level=0.95),
            ValueError,
            "behavioral",
        ),
        (
            ((1.0, float("nan")), (0.0, 0.0)),
            ((1.0, 0.5), (0.0, 0.25)),
            ((1.0, 0.5), (0.0, 0.25)),
            Bootstrap(unit="item", replicates=9, seed=1, level=0.95),
            ValueError,
            "behavioral",
        ),
        (
            ((1.0, 0.5), (0.0, 0.25)),
            ((1.0,), (0.0, 0.0)),
            ((1.0, 0.5), (0.0, 0.25)),
            Bootstrap(unit="item", replicates=9, seed=1, level=0.95),
            ValueError,
            "decodability",
        ),
        (
            ((1.0, 0.5), (0.0, 0.25)),
            ((1.0, 0.5), (0.0, 0.25)),
            ((1.0,), (0.0, 0.0)),
            Bootstrap(unit="item", replicates=9, seed=1, level=0.95),
            ValueError,
            "control",
        ),
        (
            ((1.0, 0.5), (0.0, 0.25)),
            ((1.0, 0.5), (0.0, 0.25)),
            ((1.0, 0.5), (0.0, 0.25)),
            cast("Bootstrap", "item"),
            TypeError,
            "bootstrap",
        ),
    ],
)
def test_an_erasure_report_refuses_bad_data_by_name(
    behavioral: tuple[Sequence[float], Sequence[float]],
    decodability: tuple[Sequence[float], Sequence[float]],
    control: tuple[Sequence[float], Sequence[float]],
    bootstrap: Bootstrap,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ErasureReport(
            behavioral,
            decodability,
            control,
            behavioral_direction="higher",
            decodability_direction="higher",
            control_direction="higher",
            bootstrap=bootstrap,
        )


def test_an_erasure_report_refuses_an_undeclared_metric_direction_by_name() -> None:
    with pytest.raises(ValueError, match=r"behavioral.*direction"):
        ErasureReport(
            *score_pairs(),
            behavioral_direction=cast("MetricDirection", "sideways"),
            decodability_direction="higher",
            control_direction="higher",
            bootstrap=Bootstrap(unit="item", replicates=9, seed=1, level=0.95),
        )

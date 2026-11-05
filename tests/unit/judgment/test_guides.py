import importlib
import pkgutil
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
import torch

import lofi.judgment
from lofi.judgment import InferenceMethod
from lofi.judgment.crossed import CrossedScales, exact_posterior, simulate_crossed
from lofi.judgment.guides import (
    MEAN_FIELD_GUIDE,
    PARTIALLY_FACTORIZED_GUIDE,
    CrossedState,
    MeanFieldGuide,
    PartiallyFactorizedGuide,
    fit_guide,
    resolve_guide,
)
from lofi.registry import Registry, RegistryError, UnknownTagError

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.judgment._ports import VariationalGuide
    from lofi.judgment.crossed import CrossedModel

ROWS = 240
PARTICIPANTS = 12
ITEMS = 9
COEFFICIENTS = 3
SWEEPS = 40
SCALES = CrossedScales(0.8, 2.0, (0.7, 0.5))


def model_of(
    rows: int = ROWS,
    participants: int = PARTICIPANTS,
    items: int = ITEMS,
    seed: int = 4,
) -> CrossedModel:
    generator = torch.Generator().manual_seed(seed)
    fixed = torch.randn((rows, COEFFICIENTS), generator=generator, dtype=torch.float64)
    fixed[:, 0] = 1.0
    participant = torch.randint(0, participants, (rows,), generator=generator)
    item = torch.randint(0, items, (rows,), generator=generator)
    simulated = simulate_crossed(
        fixed,
        (participant, item),
        (participants, items),
        SCALES,
        generator=generator,
    )

    return simulated.model


def bounds_over_sweeps(
    guide: VariationalGuide[CrossedState], model: CrossedModel, sweeps: int
) -> list[float]:
    state = guide.initialize(model)
    reached = []

    for _ in range(sweeps):
        state = guide.update(model, state)
        reached.append(float(guide.evidence_bound(model, state)))

    return reached


class GuideLaws:
    def test_a_state_covers_every_parameter_of_the_model(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        model = model_of()
        posterior = guide.posterior(guide.initialize(model))

        assert posterior.n_parameters == model.n_parameters
        assert bool(torch.isfinite(posterior.means).all())
        assert bool((posterior.variances > 0.0).all())

    def test_no_sweep_lowers_the_evidence_bound(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        reached = bounds_over_sweeps(guide, model_of(), SWEEPS)
        steps = [later - earlier for earlier, later in pairwise(reached)]

        assert all(step > -1e-9 for step in steps)
        assert reached[-1] > reached[0]
        assert steps[-1] < steps[0]

    def test_a_sweep_leaves_the_state_it_was_given_alone(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        model = model_of()
        state = guide.update(model, guide.initialize(model))
        held = state.effect_means.clone()
        first = guide.update(model, state)
        second = guide.update(model, state)

        assert torch.equal(state.effect_means, held)
        assert torch.equal(first.effect_means, second.effect_means)
        assert torch.equal(first.fixed_mean, second.fixed_mean)
        assert torch.equal(first.fixed_covariance, second.fixed_covariance)

    def test_a_fit_recovers_the_posterior_mean_the_model_has_exactly(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        model = model_of()
        exact = exact_posterior(model)
        fitted = fit_guide(guide, model, iterations=200)
        scale = exact.variances.sqrt()

        assert float(((fitted.means - exact.means) / scale).abs().max()) < 1e-3

    def test_no_marginal_variance_exceeds_the_one_the_model_has(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        model = model_of()
        exact = exact_posterior(model)
        fitted = fit_guide(guide, model, iterations=200)

        assert float((fitted.variances - exact.variances).max()) < 1e-9

    def test_a_state_over_another_model_is_refused_by_the_bound(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        model = model_of()
        other = model_of(rows=120, participants=5, items=4, seed=8)
        state = guide.initialize(other)

        with pytest.raises(ValueError, match="was scored against a state over"):
            guide.evidence_bound(model, state)

    def test_the_guide_is_installed_under_the_inference_it_records(
        self, guide: VariationalGuide[CrossedState]
    ) -> None:
        resolved = resolve_guide(guide.method)

        assert type(resolved) is type(guide)
        assert resolved.method is guide.method


class TestMeanFieldGuide(GuideLaws):
    @pytest.fixture
    def guide(self) -> VariationalGuide[CrossedState]:
        return MeanFieldGuide()


class TestPartiallyFactorizedGuide(GuideLaws):
    @pytest.fixture
    def guide(self) -> VariationalGuide[CrossedState]:
        return PartiallyFactorizedGuide()


def test_the_relaxation_reaches_a_higher_bound_than_mean_field_does() -> None:
    model = model_of()
    relaxed = bounds_over_sweeps(PartiallyFactorizedGuide(), model, SWEEPS)
    factorized = bounds_over_sweeps(MeanFieldGuide(), model, SWEEPS)

    assert relaxed[-1] > factorized[-1]


def test_the_relaxation_keeps_the_coefficient_spread_mean_field_shrinks() -> None:
    model = model_of()
    exact = exact_posterior(model).variances[: model.n_fixed].sqrt()
    relaxed = (
        fit_guide(PartiallyFactorizedGuide(), model, iterations=200)
        .variances[: model.n_fixed]
        .sqrt()
    )
    factorized = (
        fit_guide(MeanFieldGuide(), model, iterations=200)
        .variances[: model.n_fixed]
        .sqrt()
    )

    assert torch.allclose(relaxed, exact, rtol=1e-3)
    assert float((factorized / exact).min()) < 0.5


def test_only_the_relaxation_lets_an_effect_follow_the_coefficients() -> None:
    model = model_of()
    relaxed = PartiallyFactorizedGuide().update(
        model, PartiallyFactorizedGuide().initialize(model)
    )
    factorized = MeanFieldGuide().update(model, MeanFieldGuide().initialize(model))
    loadings = relaxed.effect_loadings

    assert factorized.effect_loadings is None
    assert loadings is not None
    assert loadings.shape == (model.n_effects, model.n_fixed)
    assert float(loadings.abs().max()) > 0.0


def test_a_fit_runs_at_least_one_sweep() -> None:
    with pytest.raises(ValueError, match="at least one sweep"):
        fit_guide(MeanFieldGuide(), model_of(), iterations=0)


def test_a_state_refuses_parts_that_do_not_line_up() -> None:
    mean = torch.zeros(2, dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)
    effects = torch.zeros(3, dtype=torch.float64)
    variances = torch.ones(3, dtype=torch.float64)

    with pytest.raises(ValueError, match="square covariance"):
        CrossedState(mean, torch.eye(3, dtype=torch.float64), effects, variances, None)

    with pytest.raises(ValueError, match="one mean and one variance per effect"):
        CrossedState(mean, covariance, effects, torch.ones(2), None)

    with pytest.raises(ValueError, match="not positive"):
        CrossedState(
            mean, covariance, effects, torch.zeros(3, dtype=torch.float64), None
        )

    with pytest.raises(ValueError, match="loadings over"):
        CrossedState(
            mean,
            covariance,
            effects,
            variances,
            torch.zeros((3, 3), dtype=torch.float64),
        )


def test_the_guide_group_lists_both_families_and_refuses_an_unknown_one() -> None:
    guides: Registry[VariationalGuide[CrossedState]] = Registry("guide")

    assert guides.group == "lofi.guide"
    assert guides.available() == ("mean_field", "structured")
    assert resolve_guide(InferenceMethod.MEAN_FIELD) is MEAN_FIELD_GUIDE
    assert resolve_guide(InferenceMethod.STRUCTURED) is PARTIALLY_FACTORIZED_GUIDE

    with pytest.raises(UnknownTagError):
        resolve_guide(InferenceMethod.NUTS)


def _install_guide(root: Path, monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    """Install a distribution listing the mean-field guide under another tag."""
    site = root / "site"
    site.mkdir()
    module = f"lofi_guide_plugin_{root.name.replace('-', '_')}"
    distribution = module.replace("_", "-")
    (site / f"{module}.py").write_text(
        "from lofi.judgment.guides import MeanFieldGuide\n\nGUIDE = MeanFieldGuide()\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = site / f"{module}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Name: {distribution}\nVersion: 1.0\n", encoding="utf-8", newline="\n"
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.guide]\n{tag} = {module}:GUIDE\n", encoding="utf-8", newline="\n"
    )
    monkeypatch.syspath_prepend(str(site))


def test_a_guide_listed_under_another_inference_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_guide(tmp_path, monkeypatch, str(InferenceMethod.NUTS))

    with pytest.raises(RegistryError, match="records the inference mean_field"):
        resolve_guide(InferenceMethod.NUTS)


def test_importing_the_judgment_package_registers_nothing() -> None:
    modules = [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(
            lofi.judgment.__path__, prefix="lofi.judgment."
        )
    ]
    held = [
        f"{module.__name__}.{name}"
        for module in (lofi.judgment, *modules)
        for name, value in vars(module).items()
        if isinstance(value, Registry)
    ]

    assert held == []

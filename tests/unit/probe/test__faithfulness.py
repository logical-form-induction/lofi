from statistics import fmean
from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import torch
from tests.conformance._suite import Conformance, ConformanceSuite

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._circuits import Circuit, Unit
from lofi.probe._faithfulness import (
    Ablation,
    Completeness,
    Contrast,
    Endpoints,
    Minimality,
    RecoveryPoint,
    Sufficiency,
    Summary,
    completeness,
    endpoints,
    minimality,
    recovery,
    sufficiency,
)
from lofi.probe._instrument import instrument
from lofi.probe._interventions import Resample, Zero
from lofi.probe._provenance import Provenance
from lofi.stats import Bootstrap, paired_effect_size

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._values import Trace

MODEL: Final = ComponentId("backup")
ITEMS: Final = 16
AXES: Final = ("item", "width")
A: Final = Site(MODEL, Role.RESIDUAL_STREAM, ("a",))
B: Final = Site(MODEL, Role.RESIDUAL_STREAM, ("b",))
N: Final = Site(MODEL, Role.RESIDUAL_STREAM, ("n",))
SITES: Final = frozenset({A, B, N})

# the readout weights on the shared signal, the private signal and the nuisance
ON_SHARED: Final = 2.0
ON_PRIVATE: Final = 1.0
ON_NUISANCE: Final = 0.25

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    MODEL,
    {
        site: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=False,
            access=Access.OBSERVE | Access.PATCH,
        )
        for site in SITES
    },
)

A0: Final = Unit(A, {"width": 0})
A1: Final = Unit(A, {"width": 1})
B0: Final = Unit(B)
N0: Final = Unit(N)

PROVENANCE: Final = Provenance("sufficiency", "backup", None)


class _Held:
    __slots__ = ("_site", "_tensor")

    def __init__(self, site: Site, tensor: Tensor) -> None:
        self._site = site
        self._tensor = tensor

    @property
    def site(self) -> Site:
        return self._site

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return AXES

    @property
    def mask(self) -> Tensor | None:
        return None


class _Scope:
    def __init__(
        self,
        model: _Backup,
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

    def pass_through(self, site: Site, tensor: Tensor) -> Tensor:
        standing = self._patches.get(site)

        if standing is not None:
            tensor = standing.stands_in_for(_Held(site, tensor))

        if site in self._sites:
            self._readings[site] = _Held(site, tensor)

        return tensor

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
        self._model.scopes.remove(self)

        return None


class _Backup:
    """Reads a shared signal off two points, one backing the other up.

    The input holds three positive numbers per item. Point A holds the shared
    signal in its first column and a private signal in its second, point B holds
    the shared signal again, and point N holds a nuisance. The score takes the
    larger of the two copies of the shared signal, so removing either copy alone
    changes nothing, plus the private signal and a little of the nuisance.
    """

    def __init__(self) -> None:
        self.scopes: list[_Scope] = []
        self.forwards = 0
        self.seen: dict[Site, Tensor] = {}

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def at(self, site: Site, tensor: Tensor) -> Tensor:
        for scope in self.scopes:
            tensor = scope.pass_through(site, tensor)

        self.seen[site] = tensor

        return tensor

    def forward(self, given: Tensor) -> Tensor:
        self.forwards += 1
        shared_and_private = self.at(A, given[:, :2])
        backup = self.at(B, given[:, :1])
        nuisance = self.at(N, given[:, 2:])

        return (
            ON_SHARED * torch.maximum(shared_and_private[:, 0], backup[:, 0])
            + ON_PRIVATE * shared_and_private[:, 1]
            + ON_NUISANCE * nuisance[:, 0]
        )


def a_batch(seed: int = 0, items: int = ITEMS) -> Tensor:
    """Return positive inputs, so that zero is below every signal."""
    draws = torch.Generator().manual_seed(seed)

    return torch.rand(items, 3, generator=draws) + 0.5


def a_bootstrap(seed: int = 0) -> Bootstrap:
    return Bootstrap(unit="item", replicates=199, seed=seed, level=0.95)


def an_ablation(
    model: _Backup, given: Tensor, *, intervention: Resample | Zero | None = None
) -> Ablation[_Held]:
    return Ablation(
        model,
        sites=SITES,
        metric=lambda: model.forward(given),
        intervention=Zero() if intervention is None else intervention,
        dataset="backup",
    )


def a_setting() -> tuple[_Backup, Tensor, Ablation[_Held], Endpoints]:
    model = _Backup()
    given = a_batch()
    ablation = an_ablation(model, given)

    return model, given, ablation, endpoints(ablation)


def column_means(given: Tensor) -> tuple[float, float, float]:
    """Return the mean shared, private and nuisance signal over the items."""
    shared, private, nuisance = given.mean(dim=0).tolist()

    return shared, private, nuisance


def clean_mean(given: Tensor) -> float:
    shared, private, nuisance = column_means(given)

    return ON_SHARED * shared + ON_PRIVATE * private + ON_NUISANCE * nuisance


def a_source_trace(model: _Backup, seed: int = 5) -> Trace:
    with torch.no_grad(), instrument(model, observe=SITES) as run:
        run(lambda: model.forward(a_batch(seed) + 2.0))

    return run.trace


CLEAN: Final = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
CORRUPT: Final = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
GAP: Final = fmean(CLEAN) - fmean(CORRUPT)


def some_endpoints(shift: float = 0.0) -> Endpoints:
    return Endpoints((CLEAN[0] + shift, *CLEAN[1:]), CORRUPT)


def a_contrast(shift: float = 0.0) -> Contrast:
    """Return a contrast whose per-item differences are distinct and sum to zero."""
    return Contrast(
        (1.5 + shift, 2.75, 2.0, 5.5, 3.75, 7.0, 5.25, 8.25),
        CLEAN,
        bootstrap=a_bootstrap(),
        estimator="paired-cohens-d",
    )


def a_sufficiency(shift: float = 0.0) -> Sufficiency:
    return Sufficiency(
        Circuit([A0, B0]),
        some_endpoints(),
        a_contrast(shift),
        baseline="zero",
        provenance=PROVENANCE,
    )


def a_completeness(shift: float = 0.0) -> Completeness:
    return Completeness(
        Circuit([A0, B0]),
        some_endpoints(),
        {frozenset({A0}): a_contrast(shift), frozenset({A0, B0}): a_contrast(1.0)},
        baseline="zero",
        provenance=Provenance("completeness", "backup", 3),
    )


def a_minimality(shift: float = 0.0) -> Minimality:
    return Minimality(
        Circuit([A0, B0]),
        some_endpoints(),
        {A0: a_contrast(shift), B0: a_contrast(1.0)},
        baseline="zero",
        provenance=Provenance("minimality", "backup", None),
    )


def a_summary(shift: float = 0.0) -> Summary:
    return Summary([a_sufficiency(shift), a_sufficiency(1.0)])


def a_record(
    circuit: Circuit,
    shift: float = 0.0,
    *,
    endpoints: Endpoints | None = None,
    contrast: Contrast | None = None,
    baseline: str = "zero",
    provenance: Provenance = PROVENANCE,
) -> Sufficiency:
    """Return a record of a circuit whose recovery is one plus the shift over 8 gaps."""
    return Sufficiency(
        circuit,
        some_endpoints() if endpoints is None else endpoints,
        a_contrast(shift) if contrast is None else contrast,
        baseline=baseline,
        provenance=provenance,
    )


def a_recovery_point(shift: float = 0.0) -> RecoveryPoint:
    return RecoveryPoint(2, 0.75, a_contrast(shift).effect_size)


class TestEndpoints(ConformanceSuite):
    """Tests endpoints against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Endpoints]:
        """Return the conformance the laws run endpoints through."""
        return Conformance(
            build=some_endpoints,
            unequal=some_endpoints(0.5),
            operations=(
                ("clean", lambda held: held.clean),
                ("corrupt", lambda held: held.corrupt),
                ("gap", lambda held: held.gap),
                ("len", len),
            ),
        )


class TestAContrast(ConformanceSuite):
    """Tests contrasts against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Contrast]:
        """Return the conformance the laws run a contrast through."""
        return Conformance(
            build=a_contrast,
            unequal=a_contrast(0.5),
            operations=(
                ("scores", lambda held: held.scores),
                ("reference", lambda held: held.reference),
                ("difference", lambda held: held.difference),
                ("effect_size", lambda held: held.effect_size.value),
                ("len", len),
            ),
        )


class TestASufficiency(ConformanceSuite):
    """Tests sufficiency records against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Sufficiency]:
        """Return the conformance the laws run a sufficiency through."""
        return Conformance(
            build=a_sufficiency,
            unequal=a_sufficiency(0.5),
            operations=(
                ("recovery", lambda held: held.recovery),
                ("evidence", lambda held: held.evidence.effect),
                ("circuit", lambda held: held.circuit.id),
            ),
        )


class TestACompleteness(ConformanceSuite):
    """Tests completeness records against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Completeness]:
        """Return the conformance the laws run a completeness through."""
        return Conformance(
            build=a_completeness,
            unequal=a_completeness(0.5),
            operations=(
                ("incompleteness", lambda held: held.incompleteness),
                ("worst", lambda held: len(held.worst)),
                ("knockouts", lambda held: len(held.knockouts)),
            ),
        )


class TestAMinimality(ConformanceSuite):
    """Tests minimality records against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Minimality]:
        """Return the conformance the laws run a minimality through."""
        return Conformance(
            build=a_minimality,
            unequal=a_minimality(0.5),
            operations=(
                ("contribution", lambda held: held.contribution(A0)),
                ("redundant", lambda held: len(held.redundant(0.1))),
                ("removals", lambda held: len(held.removals)),
            ),
        )


class TestARecoveryPoint(ConformanceSuite):
    """Tests curve points against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[RecoveryPoint]:
        """Return the conformance the laws run a curve point through."""
        return Conformance(
            build=a_recovery_point,
            unequal=a_recovery_point(0.5),
            operations=(
                ("size", lambda held: held.size),
                ("recovery", lambda held: held.recovery),
                ("effect_size", lambda held: held.effect_size.value),
            ),
        )


class TestASummary(ConformanceSuite):
    """Tests summaries against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Summary]:
        """Return the conformance the laws run a summary through."""
        return Conformance(
            build=a_summary,
            unequal=a_summary(0.5),
            operations=(
                ("sizes", lambda held: tuple(point.size for point in held.curve)),
                (
                    "recoveries",
                    lambda held: tuple(point.recovery for point in held.curve),
                ),
                (
                    "effect_sizes",
                    lambda held: tuple(point.effect_size.value for point in held.curve),
                ),
                ("len", len),
            ),
        )


def test_recovery_is_where_the_patched_metric_lies_between_the_endpoints() -> None:
    assert recovery(0.75, corrupt=0.0, clean=1.0) == 0.75
    assert recovery(3.0, corrupt=1.0, clean=5.0) == 0.5
    assert recovery(1.0, corrupt=1.0, clean=5.0) == 0.0
    assert recovery(5.0, corrupt=1.0, clean=5.0) == 1.0
    assert recovery(7.0, corrupt=1.0, clean=5.0) == 1.5
    assert recovery(-1.0, corrupt=1.0, clean=5.0) == -0.5


def test_recovery_reads_the_same_when_the_metric_runs_downhill() -> None:
    assert recovery(-3.0, corrupt=-1.0, clean=-5.0) == 0.5
    assert recovery(2.0, corrupt=4.0, clean=0.0) == 0.5


def test_recovery_is_undefined_between_coincident_endpoints() -> None:
    with pytest.raises(ValueError, match="nothing lies between"):
        recovery(1.0, corrupt=2.0, clean=2.0)


def test_recovery_refuses_a_metric_that_is_not_a_finite_number() -> None:
    with pytest.raises(ValueError, match="not finite"):
        recovery(float("nan"), corrupt=0.0, clean=1.0)

    with pytest.raises(TypeError, match="not a number"):
        recovery(cast("float", "0.5"), corrupt=0.0, clean=1.0)


def test_the_endpoints_are_the_clean_pass_and_the_pass_with_everything_replaced() -> (
    None
):
    model, given, ablation, between = a_setting()

    assert model.forwards == 2
    assert between.clean == pytest.approx(tuple(model.forward(given).tolist()))
    assert between.corrupt == (0.0,) * ITEMS
    assert between.clean_mean == pytest.approx(clean_mean(given))
    assert between.corrupt_mean == 0.0
    assert between.gap == pytest.approx(clean_mean(given))
    assert len(between) == ITEMS
    assert ablation.baseline == "zero"
    assert ablation.dataset == "backup"
    assert ablation.sites == SITES


def test_sufficiency_of_the_points_alone_is_their_share_of_the_clean_metric() -> None:
    model, given, ablation, between = a_setting()
    shared, private, nuisance = column_means(given)
    forwards = model.forwards

    assessed = sufficiency(
        ablation, Circuit([A0, A1]), endpoints=between, bootstrap=a_bootstrap()
    )

    without_nuisance = ON_SHARED * shared + ON_PRIVATE * private

    assert model.forwards == forwards + 1
    assert assessed.recovery == pytest.approx(without_nuisance / clean_mean(given))
    assert 0.0 < assessed.recovery < 1.0
    assert assessed.evidence.effect == pytest.approx(-ON_NUISANCE * nuisance)
    assert assessed.evidence.baseline == "zero"
    assert assessed.evidence.provenance == PROVENANCE
    assert assessed.contrast.reference == between.clean
    assert assessed.contrast.effect_size.estimator == "paired-cohens-d"
    assert assessed.contrast.effect_size.value < 0.0
    assert assessed.endpoints == between
    assert assessed.circuit == Circuit([A1, A0])


def test_sufficiency_keeps_the_circuit_and_replaces_everything_else() -> None:
    model, given, ablation, between = a_setting()

    sufficiency(ablation, Circuit([A0, N0]), endpoints=between, bootstrap=a_bootstrap())

    assert torch.equal(model.seen[A][:, 0], given[:, 0])
    assert torch.equal(model.seen[A][:, 1], torch.zeros(ITEMS))
    assert torch.equal(model.seen[B], torch.zeros(ITEMS, 1))
    assert torch.equal(model.seen[N], given[:, 2:])


def test_a_backup_copy_alone_recovers_what_the_shared_signal_carries() -> None:
    _, given, ablation, between = a_setting()
    shared, _, _ = column_means(given)

    assessed = sufficiency(
        ablation, Circuit([B0]), endpoints=between, bootstrap=a_bootstrap()
    )

    assert assessed.recovery == pytest.approx(ON_SHARED * shared / clean_mean(given))


def test_the_whole_circuit_recovers_the_clean_metric_exactly() -> None:
    _, _, ablation, between = a_setting()

    assessed = sufficiency(
        ablation,
        Circuit([A0, A1, B0, N0]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        estimator="paired-cliffs-delta",
    )

    assert assessed.recovery == 1.0
    assert assessed.evidence.effect == 0.0
    assert assessed.contrast.effect_size.value == 0.0


def test_completeness_finds_the_backup_a_sufficient_circuit_leaves_out() -> None:
    model, given, ablation, between = a_setting()
    shared, _, nuisance = column_means(given)
    forwards = model.forwards

    assessed = completeness(
        ablation,
        Circuit([A0, A1]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        draws=4,
        seed=1,
    )

    tried = set(assessed.knockouts)
    hidden = ON_SHARED * shared + ON_NUISANCE * nuisance

    assert model.forwards == forwards + 2 * len(tried)
    assert {frozenset({A0}), frozenset({A1}), frozenset({A0, A1})} <= tried
    assert assessed.disagreement(frozenset({A0})) == pytest.approx(
        -hidden / between.gap
    )
    assert assessed.disagreement(frozenset({A1})) == pytest.approx(
        -ON_NUISANCE * nuisance / between.gap
    )
    assert assessed.disagreement(frozenset({A0, A1})) == pytest.approx(
        -hidden / between.gap
    )
    assert assessed.incompleteness == pytest.approx(hidden / between.gap)
    assert assessed.worst in {frozenset({A0}), frozenset({A0, A1})}
    assert assessed.baseline == "zero"
    assert assessed.provenance == Provenance("completeness", "backup", 1)


def test_removing_a_subset_from_the_circuit_and_from_the_model_are_two_passes() -> None:
    _, given, ablation, between = a_setting()

    assessed = completeness(
        ablation,
        Circuit([A0, A1]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        draws=0,
        seed=0,
    )
    knockout = assessed.knockouts[frozenset({A0})]

    assert knockout.scores == pytest.approx(tuple((ON_PRIVATE * given[:, 1]).tolist()))
    assert knockout.reference == pytest.approx(
        tuple(
            (
                ON_SHARED * given[:, 0]
                + ON_PRIVATE * given[:, 1]
                + ON_NUISANCE * given[:, 2]
            ).tolist()
        )
    )
    assert set(assessed.knockouts) == {
        frozenset({A0}),
        frozenset({A1}),
        frozenset({A0, A1}),
    }


def test_a_circuit_holding_both_copies_is_short_only_the_nuisance() -> None:
    _, given, ablation, between = a_setting()
    _, _, nuisance = column_means(given)

    assessed = completeness(
        ablation,
        Circuit([A0, A1, B0]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        draws=6,
        seed=2,
    )

    for removed in assessed.knockouts:
        assert assessed.disagreement(removed) == pytest.approx(
            -ON_NUISANCE * nuisance / between.gap
        )

    assert assessed.incompleteness == pytest.approx(
        ON_NUISANCE * nuisance / between.gap
    )


def test_the_whole_circuit_is_complete() -> None:
    _, _, ablation, between = a_setting()

    assessed = completeness(
        ablation,
        Circuit([A0, A1, B0, N0]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        estimator="paired-cliffs-delta",
        draws=8,
        seed=3,
    )

    assert assessed.incompleteness == 0.0
    assert len(assessed.knockouts) > 5


def test_random_subsets_are_drawn_under_the_seed() -> None:
    _, _, ablation, between = a_setting()
    circuit = Circuit([A0, A1, B0, N0])

    def tried(seed: int) -> frozenset[frozenset[Unit]]:
        return frozenset(
            completeness(
                ablation,
                circuit,
                endpoints=between,
                bootstrap=a_bootstrap(),
                estimator="paired-cliffs-delta",
                draws=8,
                seed=seed,
            ).knockouts
        )

    assert tried(3) == tried(3)
    assert tried(3) != tried(4)
    assert all(subset and subset <= circuit.units for subset in tried(4))


def test_minimality_reads_what_each_point_alone_contributes() -> None:
    model, given, ablation, between = a_setting()
    shared, private, _ = column_means(given)
    forwards = model.forwards

    assessed = minimality(
        ablation, Circuit([A0, A1]), endpoints=between, bootstrap=a_bootstrap()
    )

    assert model.forwards == forwards + 3
    assert assessed.contribution(A0) == pytest.approx(ON_SHARED * shared / between.gap)
    assert assessed.contribution(A1) == pytest.approx(
        ON_PRIVATE * private / between.gap
    )
    assert assessed.redundant(0.01) == frozenset()
    assert assessed.removals[A0].reference == assessed.removals[A1].reference
    assert assessed.removals[A0].effect_size.value < 0.0
    assert assessed.provenance == Provenance("minimality", "backup", None)


def test_minimality_calls_a_backed_up_point_redundant() -> None:
    _, given, ablation, between = a_setting()
    _, private, _ = column_means(given)

    assessed = minimality(
        ablation,
        Circuit([A0, A1, B0]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        estimator="paired-cliffs-delta",
    )

    assert assessed.contribution(A0) == 0.0
    assert assessed.contribution(B0) == 0.0
    assert assessed.contribution(A1) == pytest.approx(
        ON_PRIVATE * private / between.gap
    )
    assert assessed.redundant(1e-6) == frozenset({A0, B0})
    assert assessed.redundant(0.0) == frozenset()


def test_a_summary_reads_each_record_where_its_evidence_was_written() -> None:
    _, given, ablation, between = a_setting()
    _, _, nuisance = column_means(given)
    partial = [
        sufficiency(ablation, circuit, endpoints=between, bootstrap=a_bootstrap())
        for circuit in (Circuit([A0, A1]), Circuit([A0, A1, B0]))
    ]
    whole = sufficiency(
        ablation,
        Circuit([A0, A1, B0, N0]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        estimator="paired-cliffs-delta",
    )

    summary = Summary([whole, *partial])

    assert len(summary) == 3
    assert [point.size for point in summary.curve] == [2, 3, 4]
    assert summary.curve[2].recovery == 1.0
    assert summary.curve[0].recovery == pytest.approx(partial[0].recovery)
    assert summary.curve[0].effect_size == paired_effect_size(
        partial[0].contrast.scores,
        between.clean,
        a_bootstrap(),
        estimator="paired-cohens-d",
    )
    assert summary.curve[2].effect_size.estimator == "paired-cliffs-delta"
    # the backup copy of the shared signal adds nothing, so the two agree
    assert summary.curve[1].effect_size == summary.curve[0].effect_size
    assert summary.curve[0].effect_size.value < 0.0
    assert summary.curve[2].effect_size.value == 0.0
    assert whole.evidence.effect == pytest.approx(0.0, abs=1e-6)

    for record in partial:
        assert record.evidence.effect == pytest.approx(
            -ON_NUISANCE * nuisance, abs=1e-6
        )


def test_the_recovery_curve_matches_an_analytically_known_circuit() -> None:
    _, given, ablation, between = a_setting()
    shared, private, nuisance = column_means(given)
    clean = clean_mean(given)
    assessed = [
        sufficiency(ablation, circuit, endpoints=between, bootstrap=a_bootstrap())
        for circuit in (Circuit([A0]), Circuit([A0, A1]), Circuit([A0, A1, B0]))
    ]
    whole = sufficiency(
        ablation,
        Circuit([A0, A1, B0, N0]),
        endpoints=between,
        bootstrap=a_bootstrap(),
        estimator="paired-cliffs-delta",
    )

    curve = Summary([whole, *assessed]).curve

    # zeroing every point scores nothing, so recovery is the share of the clean score
    assert between.corrupt_mean == pytest.approx(0.0, abs=1e-6)
    assert [point.size for point in curve] == [1, 2, 3, 4]
    assert curve[0].recovery == pytest.approx(ON_SHARED * shared / clean, abs=1e-6)
    assert curve[1].recovery == pytest.approx(
        (ON_SHARED * shared + ON_PRIVATE * private) / clean, abs=1e-6
    )
    # the backup copy of the shared signal adds nothing the first copy did not
    assert curve[2].recovery == pytest.approx(curve[1].recovery, abs=1e-6)
    assert curve[3].recovery == pytest.approx(1.0, abs=1e-6)
    # the score the three signals predict is the clean run the ablation read
    assert clean == pytest.approx(between.clean_mean, abs=1e-6)
    # and leaving the nuisance point out of the circuit costs exactly its share
    assert assessed[2].evidence.effect == pytest.approx(
        -ON_NUISANCE * nuisance, abs=1e-6
    )

    expected = [
        paired_effect_size(
            record.contrast.scores,
            between.clean,
            a_bootstrap(),
            estimator="paired-cohens-d",
        )
        for record in assessed
    ]
    expected.append(
        paired_effect_size(
            whole.contrast.scores,
            between.clean,
            a_bootstrap(),
            estimator="paired-cliffs-delta",
        )
    )

    # each point carries its own circuit's run against the clean run, not another's
    assert [point.effect_size for point in curve] == expected
    # and the smaller the circuit, the further its run sits below the clean run
    assert curve[0].effect_size.value < curve[1].effect_size.value < 0.0
    assert curve[2].effect_size.value == pytest.approx(curve[1].effect_size.value)
    assert curve[3].effect_size.value == 0.0

    for point in curve:
        interval = point.effect_size.interval
        assert interval.unit == "item"
        assert interval.units == ITEMS
        assert interval.replicates == 199
        assert interval.seed == 0
        assert interval.bca.low <= point.effect_size.value <= interval.bca.high
        assert (
            interval.percentile.low
            <= point.effect_size.value
            <= interval.percentile.high
        )


def test_a_summary_publishes_its_circuits_only_along_the_size_axis() -> None:
    published = {name for name in vars(Summary) if not name.startswith("_")}

    assert published == {"records", "curve"}


def test_resample_ablation_draws_every_replaced_entry_from_the_source() -> None:
    model = _Backup()
    given = a_batch()
    source = a_source_trace(model)
    resampling = Resample(source, 9)
    ablation = an_ablation(model, given, intervention=resampling)

    between = endpoints(ablation)

    for site, column in ((A, given[:, :2]), (B, given[:, :1]), (N, given[:, 2:])):
        drawn = resampling.stands_in_for(source[site])
        expected = torch.zeros_like(column)
        expected.copy_(drawn.reshape(column.shape))

        assert torch.equal(model.seen[site], expected)

    assert between.corrupt_mean > between.clean_mean
    assert ablation.baseline == "resample"

    assessed = sufficiency(
        ablation, Circuit([A0, A1]), endpoints=between, bootstrap=a_bootstrap()
    )

    assert torch.equal(model.seen[A], given[:, :2])
    assert assessed.evidence.baseline == "resample"
    assert assessed.recovery < 1.0


def test_a_run_replacing_nothing_is_the_plain_forward_pass() -> None:
    model = _Backup()
    given = a_batch()
    ablation = an_ablation(model, given)

    assert ablation.without(()) == pytest.approx(tuple(model.forward(given).tolist()))
    assert ablation.only((A0, A1, B0, N0)) == pytest.approx(
        tuple(model.forward(given).tolist())
    )


def test_a_unit_at_a_point_the_ablation_does_not_study_is_refused() -> None:
    model = _Backup()
    ablation = Ablation(
        model,
        sites=(A, B),
        metric=lambda: model.forward(a_batch()),
        intervention=Zero(),
        dataset="backup",
    )

    with pytest.raises(ValueError, match="does not study"):
        ablation.only((N0,))

    with pytest.raises(ValueError, match="does not study"):
        ablation.without((N0,))

    assert model.forwards == 0


def test_an_ablation_studies_points_named_by_sites() -> None:
    model = _Backup()

    with pytest.raises(TypeError, match="not a Site"):
        Ablation(
            model,
            sites=(cast("Site", "a"),),
            metric=lambda: model.forward(a_batch()),
            intervention=Zero(),
            dataset="backup",
        )

    with pytest.raises(ValueError, match="at least one point"):
        Ablation(
            model,
            sites=(),
            metric=lambda: model.forward(a_batch()),
            intervention=Zero(),
            dataset="backup",
        )

    with pytest.raises(ValueError, match="dataset"):
        Ablation(
            model,
            sites=SITES,
            metric=lambda: model.forward(a_batch()),
            intervention=Zero(),
            dataset="",
        )


def test_an_ablation_removes_units_only() -> None:
    model = _Backup()
    ablation = an_ablation(model, a_batch())

    with pytest.raises(TypeError, match="units"):
        ablation.without((cast("Unit", A),))

    with pytest.raises(TypeError, match="units"):
        ablation.only((cast("Unit", "a"),))


def test_a_metric_pass_returns_one_score_per_item() -> None:
    model = _Backup()
    ablation = Ablation(
        model,
        sites=SITES,
        metric=lambda: model.forward(a_batch()).sum(),
        intervention=Zero(),
        dataset="backup",
    )

    with pytest.raises(ValueError, match="one score per item"):
        ablation.without(())

    with pytest.raises(ValueError, match="one score per item"):
        endpoints(
            Ablation(
                model,
                sites=SITES,
                metric=lambda: model.forward(a_batch()).unsqueeze(-1),
                intervention=Zero(),
                dataset="backup",
            )
        )


def test_a_metric_pass_scoring_no_item_is_refused() -> None:
    model = _Backup()
    ablation = an_ablation(model, a_batch(items=0))

    with pytest.raises(ValueError, match="no items"):
        ablation.without(())


def test_a_point_the_component_does_not_publish_is_refused_at_scope_entry() -> None:
    model = _Backup()
    elsewhere = Site(MODEL, Role.RESIDUAL_STREAM, ("elsewhere",))
    ablation = Ablation(
        model,
        sites=(A, elsewhere),
        metric=lambda: model.forward(a_batch()),
        intervention=Zero(),
        dataset="backup",
    )

    with pytest.raises(ValueError, match="not published"):
        ablation.only(())

    assert model.forwards == 0


def test_endpoints_refuse_runs_that_score_alike_on_average() -> None:
    with pytest.raises(ValueError, match="nothing lies between"):
        Endpoints((1.0, 2.0), (2.0, 1.0))


def test_endpoints_refuse_runs_over_different_items() -> None:
    with pytest.raises(ValueError, match="items"):
        Endpoints((1.0, 2.0), (0.0,))

    with pytest.raises(ValueError, match="no items"):
        Endpoints((), ())


def test_endpoints_refuse_a_score_that_is_not_a_finite_number() -> None:
    with pytest.raises(ValueError, match="not finite"):
        Endpoints((1.0, float("inf")), (0.0, 0.0))

    with pytest.raises(TypeError, match="not a number"):
        Endpoints((1.0, cast("float", None)), (0.0, 0.0))


def test_endpoints_report_their_means_without_the_numbers() -> None:
    assert repr(some_endpoints()) == "Endpoints(clean=4.5, corrupt=1.75, items=8)"
    assert some_endpoints().gap == GAP


def test_a_contrast_holds_the_difference_of_means_and_its_effect_size() -> None:
    contrast = a_contrast()

    assert contrast.mean == 4.5
    assert contrast.reference_mean == 4.5
    assert contrast.difference == 0.0
    assert len(contrast) == 8
    assert contrast.effect_size.estimator == "paired-cohens-d"
    assert contrast.effect_size.value == 0.0
    assert contrast.effect_size.interval.unit == "item"
    assert repr(contrast) == "Contrast(mean=4.5, reference=4.5, effect_size=0.0)"

    shifted = a_contrast(1.0)

    assert shifted.difference == 0.125
    assert shifted.effect_size.value > 0.0


def test_a_contrast_honors_the_estimator_it_is_asked_for() -> None:
    contrast = Contrast(
        (2.0, 3.0, 1.0),
        (1.0, 1.0, 2.0),
        bootstrap=a_bootstrap(),
        estimator="paired-cliffs-delta",
    )

    assert contrast.effect_size.estimator == "paired-cliffs-delta"
    assert contrast.effect_size.value == pytest.approx(1 / 3)


def test_a_contrast_refuses_runs_over_different_items() -> None:
    with pytest.raises(ValueError, match="items"):
        Contrast((1.0, 2.0), (1.0,), bootstrap=a_bootstrap())


def test_a_contrast_needs_a_bootstrap_for_its_interval() -> None:
    with pytest.raises(TypeError, match="bootstrap"):
        Contrast((1.0, 2.0), (1.0, 3.0), bootstrap=cast("Bootstrap", 199))


def test_a_contrast_with_no_spread_is_undefined_under_a_standardized_difference() -> (
    None
):
    with pytest.raises(ValueError, match="same difference"):
        Contrast((1.0, 2.0), (1.0, 2.0), bootstrap=a_bootstrap())


def test_a_sufficiency_computes_its_recovery_and_evidence_from_the_contrast() -> None:
    assessed = a_sufficiency()
    shifted = a_sufficiency(1.0)

    assert assessed.recovery == recovery(4.5, corrupt=1.75, clean=4.5)
    assert assessed.recovery == 1.0
    assert assessed.evidence.effect == 0.0
    assert assessed.evidence.kind == "ablation"
    assert repr(assessed) == "Sufficiency(units=2, recovery=1.0, effect=0.0)"
    assert shifted.recovery == pytest.approx(1.0 + 0.125 / GAP)
    assert shifted.evidence.effect == 0.125


def test_two_sufficiencies_agreeing_on_every_mean_still_differ_by_their_items() -> None:
    reordered = Contrast(
        (2.75, 1.5, 2.0, 5.5, 3.75, 7.0, 5.25, 8.25),
        CLEAN,
        bootstrap=a_bootstrap(),
        estimator="paired-cohens-d",
    )
    other = Sufficiency(
        Circuit([A0, B0]),
        some_endpoints(),
        reordered,
        baseline="zero",
        provenance=PROVENANCE,
    )

    assert other.evidence == a_sufficiency().evidence
    assert other.recovery == a_sufficiency().recovery
    assert other != a_sufficiency()
    assert Summary([other]) != Summary([a_sufficiency()])


def test_a_sufficiency_is_read_against_the_clean_run_it_was_scored_beside() -> None:
    with pytest.raises(ValueError, match="other scores"):
        Sufficiency(
            Circuit([A0]),
            some_endpoints(0.5),
            a_contrast(),
            baseline="zero",
            provenance=PROVENANCE,
        )


def test_an_assessment_is_of_a_circuit_with_at_least_one_unit() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        Sufficiency(
            Circuit([]),
            some_endpoints(),
            a_contrast(),
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(TypeError, match="Circuit"):
        Sufficiency(
            cast("Circuit", frozenset({A0})),
            some_endpoints(),
            a_contrast(),
            baseline="zero",
            provenance=PROVENANCE,
        )


def test_an_assessment_checks_the_kinds_of_what_it_records() -> None:
    with pytest.raises(TypeError, match="Endpoints"):
        Sufficiency(
            Circuit([A0]),
            cast("Endpoints", (1.0, 0.0)),
            a_contrast(),
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(TypeError, match="Contrast"):
        Sufficiency(
            Circuit([A0]),
            some_endpoints(),
            cast("Contrast", (1.0, 0.0)),
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(TypeError, match="Provenance"):
        Sufficiency(
            Circuit([A0]),
            some_endpoints(),
            a_contrast(),
            baseline="zero",
            provenance=cast("Provenance", "eap"),
        )

    with pytest.raises(ValueError, match="baseline"):
        Sufficiency(
            Circuit([A0]),
            some_endpoints(),
            a_contrast(),
            baseline="",
            provenance=PROVENANCE,
        )


def test_a_completeness_removes_non_empty_subsets_of_its_circuit() -> None:
    with pytest.raises(ValueError, match="at least one unit at a time"):
        Completeness(
            Circuit([A0]),
            some_endpoints(),
            {frozenset(): a_contrast()},
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(ValueError, match="not within the circuit"):
        Completeness(
            Circuit([A0]),
            some_endpoints(),
            {frozenset({B0}): a_contrast()},
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(ValueError, match="at least one subset"):
        Completeness(
            Circuit([A0]), some_endpoints(), {}, baseline="zero", provenance=PROVENANCE
        )

    with pytest.raises(TypeError, match="Contrast"):
        Completeness(
            Circuit([A0]),
            some_endpoints(),
            {frozenset({A0}): cast("Contrast", 0.5)},
            baseline="zero",
            provenance=PROVENANCE,
        )


def test_a_completeness_scales_each_disagreement_by_the_gap() -> None:
    assessed = a_completeness()

    assert assessed.disagreement(frozenset({A0})) == 0.0
    assert assessed.disagreement(frozenset({A0, B0})) == pytest.approx(0.125 / GAP)
    assert assessed.incompleteness == pytest.approx(0.125 / GAP)
    assert assessed.worst == frozenset({A0, B0})
    assert repr(assessed).startswith("Completeness(units=2, subsets=2, ")

    with pytest.raises(KeyError):
        assessed.disagreement(frozenset({B0}))


def test_a_minimality_removes_each_unit_of_its_circuit_once() -> None:
    with pytest.raises(ValueError, match="each unit of the circuit once"):
        Minimality(
            Circuit([A0, B0]),
            some_endpoints(),
            {A0: a_contrast()},
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(ValueError, match="other scores"):
        Minimality(
            Circuit([A0, B0]),
            some_endpoints(),
            {
                A0: a_contrast(),
                B0: Contrast(CLEAN, CORRUPT, bootstrap=a_bootstrap()),
            },
            baseline="zero",
            provenance=PROVENANCE,
        )

    with pytest.raises(TypeError, match="Contrast"):
        Minimality(
            Circuit([A0]),
            some_endpoints(),
            {A0: cast("Contrast", 0.5)},
            baseline="zero",
            provenance=PROVENANCE,
        )


def test_a_minimality_scales_each_contribution_by_the_gap() -> None:
    assessed = a_minimality()

    assert assessed.contribution(A0) == 0.0
    assert assessed.contribution(B0) == pytest.approx(-0.125 / GAP)
    assert assessed.redundant(0.01) == frozenset({A0})
    assert assessed.redundant(0.1) == frozenset({A0, B0})
    assert repr(assessed) == "Minimality(units=2, contributions=[0.0, -0.0455])"

    with pytest.raises(ValueError, match="negative"):
        assessed.redundant(-0.1)

    with pytest.raises(KeyError):
        assessed.contribution(N0)


def test_a_summary_aggregates_sufficiency_records_only() -> None:
    with pytest.raises(TypeError, match="Sufficiency"):
        Summary([cast("Sufficiency", a_completeness())])

    with pytest.raises(ValueError, match="at least one"):
        Summary([])


def test_a_summary_orders_its_records_smallest_circuit_first() -> None:
    small = Sufficiency(
        Circuit([A0]),
        some_endpoints(),
        a_contrast(),
        baseline="zero",
        provenance=PROVENANCE,
    )
    large = a_sufficiency()

    summary = Summary([large, small])

    assert summary.records == (small, large)
    assert [(point.size, point.recovery) for point in summary.curve] == [
        (1, 1.0),
        (2, 1.0),
    ]
    assert repr(summary) == "Summary(circuits=2, curve=[(1, 1.0), (2, 1.0)])"


def test_a_summary_orders_the_curve_by_size_and_not_by_recovery() -> None:
    small = a_record(Circuit([A0]), 16.0)
    large = a_record(Circuit([A0, A1, B0]), -10.3)

    curve = Summary([small, large]).curve

    # the small circuit recovers the more of the two, so the two axes disagree
    assert small.recovery > large.recovery
    assert [point.size for point in curve] == [1, 3]
    assert [round(point.recovery, 4) for point in curve] == [1.7273, 0.5318]
    assert Summary([large, small]).curve == curve


def test_a_summary_publishes_a_circuit_that_did_worse_than_the_corrupt_run() -> None:
    worse = a_record(Circuit([A0]), -35.8)
    better = a_record(Circuit([A0, A1]), 16.0)

    curve = Summary([worse, better]).curve

    assert worse.recovery < 0.0
    assert better.recovery > 1.0
    assert [round(point.recovery, 4) for point in curve] == [-0.6273, 1.7273]
    assert repr(Summary([worse])) == "Summary(circuits=1, curve=[(1, -0.6273)])"


def test_a_summary_breaks_a_tie_in_size_by_circuit_identifier() -> None:
    first = a_record(Circuit([B0, N0]), -10.0)
    second = a_record(Circuit([A0, A1]), -10.4)

    curve = Summary([second, first]).curve

    # the identifier and the units order the two circuits oppositely
    assert first.circuit.id < second.circuit.id
    assert repr(second.circuit) < repr(first.circuit)
    assert [point.size for point in curve] == [2, 2]
    assert [round(point.recovery, 4) for point in curve] == [0.5455, 0.5273]
    assert Summary([first, second]).records == (first, second)
    assert Summary([first, second]).curve == curve


def test_a_summary_refuses_records_that_were_not_one_measurement() -> None:
    held = a_record(Circuit([A0]), -10.0)
    farther = Endpoints(tuple(2.0 * score for score in CLEAN), CORRUPT)
    elsewhere = a_record(
        Circuit([A0, A1]),
        endpoints=farther,
        contrast=Contrast(
            tuple(2.0 * score for score in a_contrast().scores),
            farther.clean,
            bootstrap=a_bootstrap(),
            estimator="paired-cohens-d",
        ),
    )

    assert held.recovery == pytest.approx(0.5455, abs=1e-4)
    assert elsewhere.recovery == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same endpoints"):
        Summary([held, elsewhere])

    # endpoints the same distance apart but not in the same place are not the same
    shifted = Endpoints(
        tuple(10.0 + score for score in CLEAN), tuple(10.0 + score for score in CORRUPT)
    )
    higher = a_record(
        Circuit([A0, A1]),
        endpoints=shifted,
        contrast=Contrast(
            tuple(10.0 + score for score in a_contrast().scores),
            shifted.clean,
            bootstrap=a_bootstrap(),
            estimator="paired-cohens-d",
        ),
    )

    assert shifted.gap == some_endpoints().gap
    assert shifted.clean_mean != some_endpoints().clean_mean

    with pytest.raises(ValueError, match="same endpoints"):
        Summary([held, higher])

    with pytest.raises(ValueError, match="ablation baseline"):
        Summary([held, a_record(Circuit([A0, A1]), -10.0, baseline="resample")])

    with pytest.raises(ValueError, match="one dataset"):
        Summary(
            [
                held,
                a_record(
                    Circuit([A0, A1]),
                    -10.0,
                    provenance=Provenance("sufficiency", "other", None),
                ),
            ]
        )

    assert len(Summary([held, a_record(Circuit([A0, A1]), -10.4)])) == 2


def test_a_summary_reports_the_recovery_it_reached_at_each_size() -> None:
    shifted = Sufficiency(
        Circuit([A0]),
        some_endpoints(),
        a_contrast(0.5),
        baseline="zero",
        provenance=PROVENANCE,
    )

    assert shifted.recovery == pytest.approx(0.0625 / GAP + 1.0)
    assert repr(Summary([shifted])) == "Summary(circuits=1, curve=[(1, 1.0227)])"

from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import torch
from hypothesis import strategies as st

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._circuitry import (
    CircuitryReport,
    Criterion,
    CriterionResult,
    MatchedRandomCircuits,
    circuitry,
)
from lofi.probe._circuits import Circuit, Unit
from lofi.probe._faithfulness import Ablation, Endpoints, endpoints
from lofi.probe._interventions import Zero
from lofi.stats import NullModel, PValue, p_value_against_replicates
from lofi.stats.laws import NULL_MODEL_LAWS, NullModelFixture, check_null_model_laws

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.stats import Tail

MODEL: Final = ComponentId("planted")
AXES: Final = ("item", "width")
H: Final = Site(MODEL, Role.RESIDUAL_STREAM, ("h",))
K: Final = Site(MODEL, Role.RESIDUAL_STREAM, ("k",))

WIDTH: Final = 24
ITEMS: Final = 12
PLANTED: Final = (2, 5, 11, 17)
DECOY: Final = (0, 1, 3, 4)
SPARE: Final = 7

REPLICATES: Final = 199
SEED: Final = 7
LEVEL: Final = 0.01

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    MODEL,
    {
        H: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=False,
            access=Access.OBSERVE | Access.PATCH,
        )
    },
)

POOL: Final = tuple(Unit(H, {"width": column}) for column in range(WIDTH))
PLANTED_CIRCUIT: Final = Circuit(POOL[column] for column in PLANTED)
DECOY_CIRCUIT: Final = Circuit(POOL[column] for column in DECOY)
PADDED_CIRCUIT: Final = Circuit((*(POOL[column] for column in PLANTED), POOL[SPARE]))
ELSEWHERE: Final = Unit(K, {"width": 0})


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
        model: _Planted,
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


class _Planted:
    """Sums four of twenty-four columns, so the circuit that reads it is known.

    The score is the sum of the planted columns of the one published point. Every
    other column is nuisance: keeping it recovers nothing and removing it costs
    nothing. The inputs are positive, so zeroing a planted column always costs the
    score that column's own contribution.
    """

    def __init__(self) -> None:
        self.scopes: list[_Scope] = []
        self.forwards = 0

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def forward(self, given: Tensor) -> Tensor:
        self.forwards += 1
        held = given

        for scope in self.scopes:
            held = scope.pass_through(H, held)

        return torch.stack([held[:, column] for column in PLANTED], dim=1).sum(dim=1)


def a_batch(seed: int = 0) -> Tensor:
    draws = torch.Generator().manual_seed(seed)

    return torch.rand(ITEMS, WIDTH, generator=draws) + 0.5


def a_null(
    pool: tuple[Unit, ...] = POOL, *, replicates: int = REPLICATES, seed: int = SEED
) -> MatchedRandomCircuits:
    return MatchedRandomCircuits(pool, replicates=replicates, seed=seed)


def a_setting() -> tuple[_Planted, Ablation[_Held], Endpoints]:
    model = _Planted()
    given = a_batch()
    ablation = Ablation(
        model,
        sites=frozenset({H}),
        metric=lambda: model.forward(given),
        intervention=Zero(),
        dataset="planted",
    )

    return model, ablation, endpoints(ablation)


def width_total(circuit: Circuit) -> float:
    return float(sum(unit.index["width"] for unit in circuit.units))


def a_p_value(value: float, tail: Tail = "greater") -> PValue:
    return p_value_against_replicates(
        value,
        (0.0, 0.5, 1.0),
        tail=tail,
        null="a null",
        family="matched",
        exact=False,
        seed=1,
    )


@pytest.fixture(scope="module")
def reports() -> Mapping[Circuit, CircuitryReport]:
    """Return the three criteria of the planted circuit and of two rivals."""
    _, ablation, between = a_setting()
    null = a_null()

    return {
        circuit: circuitry(ablation, circuit, endpoints=between, null=null)
        for circuit in (PLANTED_CIRCUIT, DECOY_CIRCUIT, PADDED_CIRCUIT)
    }


def test_a_draw_holds_as_many_units_at_each_point_as_the_observed_circuit() -> None:
    pool = (*POOL, ELSEWHERE, Unit(K, {"width": 1}), Unit(K, {"width": 2}))
    observed = Circuit((POOL[0], POOL[1], POOL[2], ELSEWHERE))
    wanted = {H: 3, K: 1}
    drawn = a_null(pool, replicates=25).draw(observed)

    assert len(drawn) == 25

    for circuit in drawn:
        counts: dict[Site, int] = {}

        for unit in circuit.units:
            counts[unit.site] = counts.get(unit.site, 0) + 1

        assert counts == wanted
        assert circuit.units <= frozenset(pool)


def test_one_seed_draws_one_reference_set_and_another_seed_draws_another() -> None:
    once = a_null(replicates=25, seed=3).draw(PLANTED_CIRCUIT)

    assert once == a_null(replicates=25, seed=3).draw(PLANTED_CIRCUIT)
    assert once != a_null(replicates=25, seed=4).draw(PLANTED_CIRCUIT)


def test_a_circuit_holding_a_unit_the_pool_does_not_cannot_be_matched() -> None:
    null = a_null(replicates=5)

    with pytest.raises(ValueError, match="cannot match a circuit holding"):
        null.draw(Circuit((POOL[0], ELSEWHERE)))

    with pytest.raises(ValueError, match="cannot match a circuit holding"):
        null.complement(Circuit((ELSEWHERE,)))


def test_the_complement_is_every_unit_of_the_pool_the_circuit_leaves_out() -> None:
    null = a_null(replicates=5)
    left = null.complement(PLANTED_CIRCUIT)

    assert left == frozenset(POOL) - PLANTED_CIRCUIT.units
    assert null.complement(Circuit(POOL)) == frozenset()


def test_a_null_needs_units_to_draw_from_and_a_budget_to_draw_with() -> None:
    with pytest.raises(ValueError, match="empty pool"):
        MatchedRandomCircuits((), seed=1)

    with pytest.raises(ValueError, match="no replicate budget"):
        MatchedRandomCircuits(POOL, replicates=0, seed=1)

    with pytest.raises(TypeError, match="a circuit is drawn from units"):
        MatchedRandomCircuits(cast("tuple[Unit, ...]", (H,)), seed=1)

    with pytest.raises(ValueError, match="holds no unit"):
        a_null(replicates=5).draw(Circuit(()))


def test_the_null_is_a_null_model_by_type_and_by_shape() -> None:
    null: NullModel[Circuit] = a_null(replicates=5)

    assert isinstance(null, NullModel)
    assert null.family == "matched"
    assert dict(null.matching) == {
        "criterion": "units per site",
        "sites": 1,
        "pool": len(POOL),
    }


def test_the_planted_circuit_is_recovered_below_the_one_percent_level(
    reports: Mapping[Circuit, CircuitryReport],
) -> None:
    planted = reports[PLANTED_CIRCUIT]

    assert planted.circuit == PLANTED_CIRCUIT
    assert planted.preservation.statistic == pytest.approx(1.0)
    assert planted.localization.statistic == pytest.approx(0.0)
    assert planted.minimality.statistic > 0.0

    for criterion in Criterion:
        assert planted.results[criterion].p_value.value < LEVEL


def test_a_size_matched_random_circuit_is_not_recovered_at_that_level(
    reports: Mapping[Circuit, CircuitryReport],
) -> None:
    decoy = reports[DECOY_CIRCUIT]

    assert len(decoy.circuit.units) == len(PLANTED_CIRCUIT.units)
    assert decoy.preservation.statistic == pytest.approx(0.0)
    assert decoy.localization.statistic == pytest.approx(1.0)
    assert decoy.minimality.statistic == pytest.approx(0.0)

    for criterion in Criterion:
        assert decoy.results[criterion].p_value.value > LEVEL


def test_a_circuit_passing_one_criterion_and_failing_another_is_reported_as_such(
    reports: Mapping[Circuit, CircuitryReport],
) -> None:
    padded = reports[PADDED_CIRCUIT]

    assert padded.preservation.p_value.value < LEVEL
    assert padded.localization.p_value.value < LEVEL
    assert padded.minimality.p_value.value > LEVEL
    assert padded.minimality.statistic == pytest.approx(0.0)


def test_each_criterion_reads_its_own_tail_of_the_reference_distribution(
    reports: Mapping[Circuit, CircuitryReport],
) -> None:
    planted = reports[PLANTED_CIRCUIT]

    assert Criterion.LOCALIZATION.tail == "less"
    assert Criterion.PRESERVATION.tail == "greater"
    assert Criterion.MINIMALITY.tail == "greater"

    for criterion, result in planted.results.items():
        assert result.p_value.provenance.tail == criterion.tail


def test_the_provenance_names_the_matched_random_circuit_null(
    reports: Mapping[Circuit, CircuitryReport],
) -> None:
    null = a_null()
    planted = reports[PLANTED_CIRCUIT]

    for result in planted.results.values():
        recorded = result.p_value.provenance

        assert recorded.null == null.name
        assert recorded.family == null.family
        assert recorded.replicates == REPLICATES
        assert recorded.seed == SEED
        assert dict(recorded.matching) == dict(null.matching)
        assert not recorded.exact


def test_the_reference_distribution_costs_a_pass_for_every_circuit_it_draws() -> None:
    model, ablation, between = a_setting()
    drawn = 9
    before = model.forwards
    circuitry(
        ablation,
        PLANTED_CIRCUIT,
        endpoints=between,
        null=a_null(replicates=drawn),
    )
    runs = model.forwards - before
    units = len(PLANTED_CIRCUIT.units)

    assert runs >= drawn * (2 + units)


def test_a_p_value_read_off_the_wrong_tail_is_refused() -> None:
    with pytest.raises(ValueError, match="calls the 'less' tail extreme"):
        CriterionResult(Criterion.LOCALIZATION, 0.5, a_p_value(0.6))

    with pytest.raises(TypeError, match="records a PValue"):
        CriterionResult(Criterion.MINIMALITY, 0.5, cast("PValue", 0.5))

    with pytest.raises(TypeError, match="is of a Criterion"):
        CriterionResult(cast("Criterion", "preservation"), 0.5, a_p_value(0.6))


def test_a_result_filed_under_another_criterion_is_refused() -> None:
    preserved = CriterionResult(Criterion.PRESERVATION, 1.0, a_p_value(0.9))
    localized = CriterionResult(Criterion.LOCALIZATION, 0.0, a_p_value(0.1, "less"))
    minimal = CriterionResult(Criterion.MINIMALITY, 0.2, a_p_value(0.9))

    filed_wrong = "preservation result is filed under minimality"

    with pytest.raises(ValueError, match=filed_wrong):
        CircuitryReport(
            PLANTED_CIRCUIT,
            preservation=preserved,
            localization=localized,
            minimality=preserved,
        )

    with pytest.raises(ValueError, match="filed under localization"):
        CircuitryReport(
            PLANTED_CIRCUIT,
            preservation=preserved,
            localization=minimal,
            minimality=minimal,
        )


def test_a_report_holds_all_three_criteria_and_is_equal_on_them() -> None:
    preserved = CriterionResult(Criterion.PRESERVATION, 1.0, a_p_value(0.9))
    localized = CriterionResult(Criterion.LOCALIZATION, 0.0, a_p_value(0.1, "less"))
    minimal = CriterionResult(Criterion.MINIMALITY, 0.2, a_p_value(0.9))
    built = CircuitryReport(
        PLANTED_CIRCUIT,
        preservation=preserved,
        localization=localized,
        minimality=minimal,
    )
    again = CircuitryReport(
        PLANTED_CIRCUIT,
        preservation=preserved,
        localization=localized,
        minimality=minimal,
    )

    assert set(built.results) == set(Criterion)
    assert (built.preservation, built.localization, built.minimality) == (
        preserved,
        localized,
        minimal,
    )
    assert built == again
    assert hash(built) == hash(again)
    assert built != CircuitryReport(
        DECOY_CIRCUIT,
        preservation=preserved,
        localization=localized,
        minimality=minimal,
    )
    assert "units=4" in repr(built)


def test_a_circuitry_test_needs_endpoints_a_null_and_a_circuit_with_a_unit() -> None:
    _, ablation, between = a_setting()
    null = a_null(replicates=3)

    with pytest.raises(ValueError, match="a tested circuit holds no unit"):
        circuitry(ablation, Circuit(()), endpoints=between, null=null)

    with pytest.raises(TypeError, match="read between Endpoints"):
        circuitry(
            ablation,
            PLANTED_CIRCUIT,
            endpoints=cast("Endpoints", None),
            null=null,
        )

    with pytest.raises(TypeError, match="runs against MatchedRandomCircuits"):
        circuitry(
            ablation,
            PLANTED_CIRCUIT,
            endpoints=between,
            null=cast("MatchedRandomCircuits", None),
        )


CIRCUITS = st.lists(st.sampled_from(POOL), min_size=1, max_size=4).map(Circuit)
MATCHING_CIRCUITS = NullModelFixture(
    MatchedRandomCircuits(POOL, replicates=19, seed=11),
    width_total,
    CIRCUITS,
    10,
)


@pytest.mark.conformance
def test_every_null_model_law_holds_of_the_matched_random_circuit_null() -> None:
    check_null_model_laws(MATCHING_CIRCUITS)


@pytest.mark.conformance
@pytest.mark.parametrize("law", NULL_MODEL_LAWS, ids=lambda law: law.__name__)
def test_each_null_model_law_runs_against_it_on_its_own(
    law: Callable[[NullModelFixture[Circuit]], None],
) -> None:
    law(MATCHING_CIRCUITS)

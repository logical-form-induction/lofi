import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
import torch

from lofi.judgment import (
    DynamicRangeCriterion,
    DynamicRangeError,
    DynamicRangeGate,
    DynamicRangeReport,
    read_criterion,
)

if TYPE_CHECKING:
    from torch import Tensor
INVENTORY: Final = "ccg-core"
REGISTERED: Final = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
OBSERVED: Final = REGISTERED + timedelta(days=1)
CRITERION_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "data"
    / "judgment"
    / "pgram_criterion.json"
)


def criterion(
    *,
    registered_at: str = REGISTERED.isoformat(),
    minimum_interquartile_range: float = 1.0,
    ceiling: float = 10.0,
    maximum_ceiling_fraction: float = 0.9,
) -> DynamicRangeCriterion:
    return DynamicRangeCriterion(
        registered_at=registered_at,
        minimum_interquartile_range=minimum_interquartile_range,
        ceiling=ceiling,
        maximum_ceiling_fraction=maximum_ceiling_fraction,
    )


def gate(
    *,
    minimum_interquartile_range: float = 1.0,
    ceiling: float = 10.0,
    maximum_ceiling_fraction: float = 0.9,
) -> DynamicRangeGate:
    return DynamicRangeGate(
        criterion(
            minimum_interquartile_range=minimum_interquartile_range,
            ceiling=ceiling,
            maximum_ceiling_fraction=maximum_ceiling_fraction,
        )
    )


def spread(items: int = 200, *, seed: int = 1) -> Tensor:
    generator = torch.Generator().manual_seed(seed)

    return 12.0 * torch.rand(items, generator=generator, dtype=torch.float64) - 6.0


def test_the_report_states_the_quartiles_of_the_recorded_log_odds() -> None:
    odds = spread()
    ordered = odds.sort().values

    report = gate().report(odds, inventory=INVENTORY, observed_at=OBSERVED)

    assert report.items == odds.shape[0]
    assert report.inventory == INVENTORY
    assert report.observed_at == OBSERVED.isoformat()
    assert report.criterion == criterion()
    assert report.minimum == pytest.approx(float(ordered[0]))
    assert report.maximum == pytest.approx(float(ordered[-1]))
    assert report.median == pytest.approx(float(odds.median()), abs=0.1)
    assert report.lower_quartile == pytest.approx(
        float(torch.quantile(odds, 0.25)), abs=1e-12
    )
    assert report.upper_quartile == pytest.approx(
        float(torch.quantile(odds, 0.75)), abs=1e-12
    )
    assert report.interquartile_range == pytest.approx(
        report.upper_quartile - report.lower_quartile
    )
    assert report.ceiling_fraction == pytest.approx(
        float((odds >= 10.0).double().mean())
    )
    assert report.passed
    assert report.failures == ()


def test_the_report_is_the_same_in_any_item_order() -> None:
    odds = spread()
    shuffled = odds[
        torch.randperm(odds.shape[0], generator=torch.Generator().manual_seed(7))
    ]

    ordered = gate().report(odds, inventory=INVENTORY, observed_at=OBSERVED)
    permuted = gate().report(shuffled, inventory=INVENTORY, observed_at=OBSERVED)

    assert permuted == ordered


def test_a_constant_distribution_is_refused_and_a_spread_one_admitted() -> None:
    constant = torch.full((200,), 2.5, dtype=torch.float64)
    checked = gate()

    admitted = checked.admit(spread(), inventory=INVENTORY, observed_at=OBSERVED)

    with pytest.raises(DynamicRangeError) as caught:
        checked.admit(constant, inventory=INVENTORY, observed_at=OBSERVED)

    assert admitted.passed
    assert not caught.value.report.passed
    assert caught.value.report.interquartile_range == 0.0
    assert len(caught.value.report.failures) == 1
    assert "interquartile range" in caught.value.report.failures[0]
    assert INVENTORY in str(caught.value)
    assert REGISTERED.isoformat() in str(caught.value)
    assert "200 items" in str(caught.value)


def test_a_distribution_piled_at_the_ceiling_is_refused_on_that_clause() -> None:
    generator = torch.Generator().manual_seed(3)
    high = 12.0 + 20.0 * torch.rand(190, generator=generator, dtype=torch.float64)
    low = torch.linspace(-3.0, 3.0, 10, dtype=torch.float64)
    odds = torch.cat((high, low))

    with pytest.raises(DynamicRangeError) as caught:
        gate().admit(odds, inventory=INVENTORY, observed_at=OBSERVED)

    report = caught.value.report

    assert report.interquartile_range > 1.0
    assert report.ceiling_fraction == pytest.approx(0.95)
    assert len(report.failures) == 1
    assert "at or above" in report.failures[0]


def test_a_distribution_failing_both_clauses_names_both() -> None:
    constant = torch.full((50,), 30.0, dtype=torch.float64)

    report = gate().report(constant, inventory=INVENTORY, observed_at=OBSERVED)

    assert not report.passed
    assert len(report.failures) == 2


def test_the_criterion_is_read_where_a_spread_is_judged() -> None:
    odds = spread()
    report = gate().report(odds, inventory=INVENTORY, observed_at=OBSERVED)

    assert report.passed
    assert (
        not gate(minimum_interquartile_range=report.interquartile_range + 0.1)
        .report(odds, inventory=INVENTORY, observed_at=OBSERVED)
        .passed
    )
    assert (
        not gate(ceiling=0.0, maximum_ceiling_fraction=0.1)
        .report(odds, inventory=INVENTORY, observed_at=OBSERVED)
        .passed
    )


def test_the_spread_is_read_where_the_probability_scale_shows_none() -> None:
    exponents = range(5, 13)
    log_masses = torch.tensor(
        [math.log1p(-(10.0**-exponent)) for exponent in exponents],
        dtype=torch.float64,
    )
    log_complements = torch.tensor(
        [math.log(10.0**-exponent) for exponent in exponents], dtype=torch.float64
    )
    odds = log_masses - log_complements

    with localcontext() as context:
        context.prec = 50
        mass = 1 - Decimal("1e-12")
        reference = float(mass.ln() - (1 - mass).ln())

    report = gate().report(odds, inventory=INVENTORY, observed_at=OBSERVED)

    assert float(log_masses.exp().max() - log_masses.exp().min()) < 1e-4
    assert report.maximum == pytest.approx(reference, abs=1e-9)
    assert report.interquartile_range > 1.0
    assert report.ceiling_fraction == 1.0
    assert report.failures == (report.failures[0],)
    assert "at or above" in report.failures[0]


def test_a_distribution_observed_before_the_registration_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="registered after"):
        gate().report(
            spread(), inventory=INVENTORY, observed_at=REGISTERED - timedelta(seconds=1)
        )


def test_an_observation_at_the_registration_instant_is_admitted() -> None:
    report = gate().report(spread(), inventory=INVENTORY, observed_at=REGISTERED)

    assert report.observed_at == REGISTERED.isoformat()


def test_an_observation_without_an_offset_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="UTC offset"):
        gate().report(
            spread(), inventory=INVENTORY, observed_at=OBSERVED.replace(tzinfo=None)
        )


def test_an_unnamed_inventory_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="inventory"):
        gate().report(spread(), inventory="", observed_at=OBSERVED)


def test_an_empty_or_higher_rank_distribution_is_refused() -> None:
    checked = gate()

    with pytest.raises(ValueError, match="no item"):
        checked.report(torch.empty(0), inventory=INVENTORY, observed_at=OBSERVED)

    with pytest.raises(ValueError, match="rank 2"):
        checked.report(torch.zeros(3, 2), inventory=INVENTORY, observed_at=OBSERVED)


def test_a_single_precision_distribution_is_read_in_double() -> None:
    odds = spread().float()

    report = gate().report(odds, inventory=INVENTORY, observed_at=OBSERVED)

    assert report.items == odds.shape[0]
    assert report.minimum == pytest.approx(float(odds.min()))


def test_the_report_round_trips_through_its_wire_form() -> None:
    report = gate().report(spread(), inventory=INVENTORY, observed_at=OBSERVED)

    restored = DynamicRangeReport.model_validate_json(report.model_dump_json())

    assert restored == report
    assert restored.passed == report.passed
    assert restored.failures == report.failures


def test_the_registered_criterion_is_committed_with_its_timestamp() -> None:
    registered = read_criterion(CRITERION_PATH)
    document = json.loads(CRITERION_PATH.read_text(encoding="utf-8"))

    assert registered.registration_time().tzinfo is not None
    assert registered.registration_time() <= datetime.now(UTC)
    assert set(document) == {
        "registered_at",
        "minimum_interquartile_range",
        "ceiling",
        "maximum_ceiling_fraction",
    }


def test_the_registered_criterion_separates_a_constant_from_a_spread() -> None:
    checked = DynamicRangeGate(read_criterion(CRITERION_PATH))
    now = datetime.now(UTC)
    constant = torch.full((200,), 4.0, dtype=torch.float64)

    admitted = checked.admit(spread(), inventory=INVENTORY, observed_at=now)

    with pytest.raises(DynamicRangeError):
        checked.admit(constant, inventory=INVENTORY, observed_at=now)

    assert admitted.passed


def test_a_criterion_file_that_is_not_a_criterion_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "criterion.json"
    path.write_text(
        json.dumps({"registered_at": REGISTERED.isoformat()}),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(dx.ValidationError):
        read_criterion(path)

    path.write_text("not json", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="Expecting"):
        read_criterion(path)


def test_the_repr_names_the_criterion() -> None:
    assert repr(gate()) == f"DynamicRangeGate({criterion()!r})"
    assert gate().criterion == criterion()

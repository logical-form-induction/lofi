from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.conformance.declared_probe import (
    CELLS,
    CHART,
    LOG_Z,
    MU,
    PUBLISHED,
    ToyChart,
    ToyPoint,
)

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._interventions import Zero
from lofi.probe._probing import (
    GradientMode,
    MissingProbeCapabilityError,
    ProbeBuildContext,
    UnpublishedSiteError,
    observable,
    observed_pass,
    published_sites,
    reached_sites,
    read_probe_records,
    require_roles,
    site_label,
    write_probe_records,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tests.conformance.declared_probe import _Reading
    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.probe import PassLayout

HIDDEN: Final = Site(CHART, Role.GATE, ("hidden",))


class _Unreaching(ToyChart):
    """A chart that never reaches its log partition and draws noise on each pass."""

    @override
    def site_map(self) -> SiteMap:
        return SiteMap.custom(
            CHART,
            {
                **dict(PUBLISHED.items()),
                HIDDEN: SiteSpec(
                    ("item",),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.PATCH,
                ),
            },
        )

    @override
    def _published(self, reading: _Reading) -> Tensor:
        if reading.site == LOG_Z:
            return reading.tensor

        return super()._published(reading)

    @override
    def forward(self, layout: PassLayout) -> Tensor:
        torch.rand(4)
        return super().forward(layout)


def test_a_site_is_labeled_by_component_role_and_key() -> None:
    assert site_label(CELLS) == "chart/CHART_CELL/cells"
    assert site_label(Site(ComponentId("encoder"), Role.GATE, (0, "reset"))) == (
        "encoder/GATE/0/reset"
    )
    assert site_label(Site(ComponentId("parser"), Role.LOG_PARTITION, ())) == (
        "parser/LOG_PARTITION"
    )


def test_build_context_carries_role_and_names_absent_pass_capabilities() -> None:
    context = ProbeBuildContext("causal", ToyPoint())

    assert context.role == "causal"

    with pytest.raises(MissingProbeCapabilityError, match="computation graph"):
        context.graph_passes()

    with pytest.raises(MissingProbeCapabilityError, match="causal population"):
        context.causal_passes()


def test_the_published_sites_are_every_components_points() -> None:
    assert dict(published_sites(ToyPoint().components)) == dict(PUBLISHED.items())


def test_only_observable_points_are_observable() -> None:
    assert observable(PUBLISHED) == frozenset({CELLS, MU, LOG_Z})
    assert observable({}) == frozenset()


def test_a_role_published_without_the_needed_access_is_refused() -> None:
    require_roles((Role.CHART_CELL, Role.LOG_PARTITION), PUBLISHED, Access.OBSERVE)
    require_roles((Role.CHART_CELL,), PUBLISHED, Access.PATCH)

    with pytest.raises(UnpublishedSiteError, match="LOG_PARTITION permitting PATCH"):
        require_roles((Role.LOG_PARTITION,), PUBLISHED, Access.PATCH)

    with pytest.raises(UnpublishedSiteError, match=r"publishes \['CHART_CELL'\]"):
        require_roles((Role.GATE,), PUBLISHED, Access.PATCH)


def test_an_observed_pass_reads_only_what_it_names() -> None:
    point = ToyPoint()
    trace, layout = observed_pass(
        point,
        point.passes[0],
        observe=frozenset({LOG_Z}),
        patch={},
        gradients=GradientMode.DISABLED,
    )

    assert trace.closed
    assert trace.sites() == (LOG_Z,)
    assert layout.words == (("a", "b", "c"), ("d", "e"))
    assert point.chart.scopes == []


def test_an_observed_pass_applies_its_patches() -> None:
    point = ToyPoint()
    clean, _ = observed_pass(
        point,
        point.passes[0],
        observe=frozenset({LOG_Z}),
        patch={},
        gradients=GradientMode.DISABLED,
    )
    patched, _ = observed_pass(
        point,
        point.passes[0],
        observe=frozenset({LOG_Z, CELLS}),
        patch={CELLS: Zero()},
        gradients=GradientMode.DISABLED,
    )

    assert bool((clean[LOG_Z].tensor != 0).all())
    assert torch.equal(patched[LOG_Z].tensor, torch.zeros_like(clean[LOG_Z].tensor))
    assert torch.equal(patched[CELLS].tensor, torch.zeros_like(patched[CELLS].tensor))


def test_an_observed_pass_refuses_a_point_no_component_publishes() -> None:
    point = ToyPoint()
    stray = Site(ComponentId("elsewhere"), Role.GATE, ())

    with pytest.raises(ValueError, match="elsewhere/GATE"):
        observed_pass(
            point,
            point.passes[0],
            observe=frozenset({stray}),
            patch={},
            gradients=GradientMode.DISABLED,
        )


def test_an_observed_pass_refuses_to_patch_an_observation_only_point() -> None:
    point = ToyPoint()

    with pytest.raises(ValueError, match="cannot be named"):
        observed_pass(
            point,
            point.passes[0],
            observe=frozenset(),
            patch={MU: Zero()},
            gradients=GradientMode.DISABLED,
        )

    assert point.chart.scopes == []


def test_the_reached_sites_are_every_published_point_a_pass_reads() -> None:
    assert dict(reached_sites(ToyPoint())) == dict(PUBLISHED.items())


def test_a_point_no_pass_reaches_is_not_reached_and_an_unreadable_one_is_kept() -> None:
    point = ToyPoint(_Unreaching())
    before = torch.random.get_rng_state()

    reached = reached_sites(point)

    assert sorted(reached, key=repr) == sorted((CELLS, MU, HIDDEN), key=repr)
    assert reached[HIDDEN].access is Access.PATCH
    assert torch.equal(torch.random.get_rng_state(), before)
    assert point.chart.scopes == []


def test_probe_records_round_trip_under_their_roles(tmp_path: Path) -> None:
    path = tmp_path / "probe-records.json"
    records: dict[str, Mapping[str, JsonValue]] = {
        "arm": {"pass-0": {"arm": "dense-linear-probe", "cells": []}}
    }

    write_probe_records(path, records)

    assert read_probe_records(path) == records

    path.write_text("[1]", encoding="utf-8")

    with pytest.raises(TypeError, match="object per role"):
        read_probe_records(path)

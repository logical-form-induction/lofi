from typing import TYPE_CHECKING, ClassVar, Final, override

import pytest
import torch
from tests.conformance.declared_probe import (
    CELLS,
    LOG_Z,
    MU,
    DeclaredProbeSuite,
    ToyChart,
    ToyPoint,
)

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe import ProbeValue, UnpublishedSiteError, observed_pass, published_sites
from lofi.probe._boundary import DiagnosticsProbeSpec
from lofi.probe._catalog import DiagnosticCatalog, installed_diagnostics
from lofi.probe._diagnose import (
    Aggregation,
    DiagnosticContext,
    DiagnosticResult,
    Measurement,
)
from lofi.probe._diagnostics_probe import PLUGIN, DiagnosticsProbe, _weighting
from lofi.probe._marginals import SpanAggregate
from lofi.probe._probing import GradientMode, ProbeBuildContext
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from tests.conformance.declared_probe import _Reading
    from torch import Tensor


class MeanCells:
    tag: ClassVar[str] = "mean-cells"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.CHART_CELL})
    scores: ClassVar[frozenset[str]] = frozenset({"mean"})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        return DiagnosticResult(
            self.tag,
            [
                Measurement(site, "mean", float(trace[site].tensor.mean()))
                for site in context.sites_for(Role.CHART_CELL)
            ],
        )


class Weighed:
    tag: ClassVar[str] = "weighed"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.CONSTITUENT_MARGINAL})
    scores: ClassVar[frozenset[str]] = frozenset({"total", "spans"})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        del trace
        weighting = context.weighting
        index = context.span_index
        assert weighting is not None
        assert index is not None

        return DiagnosticResult(
            self.tag,
            [
                Measurement(MU, "total", float(weighting.tensor.sum())),
                Measurement(MU, "spans", float(index.num_spans)),
            ],
        )


class Refusing:
    tag: ClassVar[str] = "refusing"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.LOG_PARTITION})
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        del trace, context
        msg = "these readings do not support the measurement"
        raise ValueError(msg)


class Spread:
    tag: ClassVar[str] = "spread"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.CONSTITUENT_MARGINAL})
    scores: ClassVar[frozenset[str]] = frozenset({"cells"})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        weighting = context.weighting
        assert weighting is not None
        cells = trace[CELLS].tensor[..., 1]

        return DiagnosticResult(
            self.tag,
            [],
            [
                Aggregation(
                    CELLS,
                    "length",
                    SpanAggregate.over(cells, weighting.tensor, trace[MU].covering()),
                )
            ],
        )


CATALOG: Final = DiagnosticCatalog(
    {"mean-cells": MeanCells, "weighed": Weighed, "refusing": Refusing}
)


class TestDiagnosticsProbeConformance(DeclaredProbeSuite):
    @pytest.fixture
    def declared_probe(self) -> DiagnosticsProbe:
        return DiagnosticsProbe(None, installed_diagnostics())


class TestDeclaredRolesConformance(DeclaredProbeSuite):
    @pytest.fixture
    def declared_probe(self) -> DiagnosticsProbe:
        return DiagnosticsProbe(frozenset({Role.CHART_CELL}), CATALOG)


def _readings(point: ToyPoint, site: Site) -> list[Tensor]:
    return [
        observed_pass(
            point,
            held_out,
            observe=frozenset({site}),
            patch={},
            gradients=GradientMode.DISABLED,
        )[0][site].tensor
        for held_out in point.passes
    ]


def test_a_measurement_is_its_mean_over_the_passes() -> None:
    point = ToyPoint()
    values = DiagnosticsProbe(None, CATALOG).run(point)
    cells = _readings(point, CELLS)
    expected = sum(float(reading.mean()) for reading in cells) / len(cells)

    (mean,) = (value for value in values if value.measure == "mean-cells/mean")

    assert (mean.site, mean.count) == (CELLS, 3)
    assert mean.value == pytest.approx(expected)


def test_the_weighting_is_the_constituent_marginals_read_inside_their_mask() -> None:
    point = ToyPoint()
    values = {
        value.measure: value for value in DiagnosticsProbe(None, CATALOG).run(point)
    }
    marginals = [
        observed_pass(
            point,
            held_out,
            observe=frozenset({MU}),
            patch={},
            gradients=GradientMode.DISABLED,
        )
        for held_out in point.passes
    ]
    totals = [
        float(trace[MU].tensor[trace[MU].covering()].sum()) for trace, _ in marginals
    ]
    spans = [layout.spans.num_spans for _, layout in marginals if layout.spans]

    assert values["weighed/total"].value == pytest.approx(sum(totals) / 3)
    assert values["weighed/spans"].value == pytest.approx(sum(spans) / 3)


def test_an_aggregated_quantity_reports_both_means_over_the_passes() -> None:
    point = ToyPoint()
    values = {
        value.measure: value
        for value in DiagnosticsProbe(None, DiagnosticCatalog({"spread": Spread})).run(
            point
        )
    }
    weighted: list[float] = []
    uniform: list[float] = []

    for held_out in point.passes:
        trace, _ = observed_pass(
            point,
            held_out,
            observe=frozenset({CELLS, MU}),
            patch={},
            gradients=GradientMode.DISABLED,
        )
        live = trace[MU].covering()
        lengths = trace[CELLS].tensor[..., 1][live]
        marginals = trace[MU].tensor[live]
        weighted.append(float((lengths * marginals).sum() / marginals.sum()))
        uniform.append(float(lengths.mean()))

    assert sorted(values) == ["spread/length/uniform", "spread/length/weighted"]
    assert values["spread/length/weighted"].value == pytest.approx(sum(weighted) / 3)
    assert values["spread/length/uniform"].value == pytest.approx(sum(uniform) / 3)
    assert values["spread/length/weighted"].value != pytest.approx(
        values["spread/length/uniform"].value
    )
    assert {value.site for value in values.values()} == {CELLS}
    assert {value.count for value in values.values()} == {3}


def test_a_refused_pass_is_counted_rather_than_dropped() -> None:
    values = DiagnosticsProbe(None, CATALOG).run(ToyPoint())

    assert values[-1] == ProbeValue(
        "refusing/refused", None, 3.0, 3, None, None, None, None, None, None
    )


class RefusingWide:
    tag: ClassVar[str] = "refusing-wide"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.CHART_CELL})
    scores: ClassVar[frozenset[str]] = frozenset({"spans"})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        del trace
        index = context.span_index
        assert index is not None

        if index.num_spans > 6:
            msg = "this pass holds an item longer than three words"
            raise ValueError(msg)

        return DiagnosticResult(
            self.tag, [Measurement(CELLS, "spans", float(index.num_spans))]
        )


def test_refusals_are_counted_over_the_passes_a_diagnostic_was_selected_for() -> None:
    catalog = DiagnosticCatalog({"refusing-wide": RefusingWide})
    values = DiagnosticsProbe(None, catalog).run(ToyPoint())

    assert values == (
        ProbeValue(
            "refusing-wide/spans",
            CELLS,
            6.0,
            2,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        ProbeValue(
            "refusing-wide/refused",
            None,
            1.0,
            3,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def marginal(site: Site, value: float) -> Observation:
    return Observation(site, torch.full((2, 3), value), axes=("item", "span"))


def test_the_weighting_is_the_single_constituent_marginal_a_pass_read() -> None:
    other = Site(ComponentId("other"), Role.CONSTITUENT_MARGINAL, ("mu",))
    one = Trace()
    one.record(marginal(MU, 0.25))
    one.close()
    two = Trace()
    two.record(marginal(MU, 0.25))
    two.record(marginal(other, 0.75))
    two.close()
    weighting = _weighting(one)

    assert weighting is not None
    assert torch.equal(weighting.tensor, torch.full((2, 3), 0.25))
    assert _weighting(two) is None
    assert _weighting(Trace()) is None


def test_declared_roles_select_the_diagnostics_their_points_support() -> None:
    values = DiagnosticsProbe(frozenset({Role.LOG_PARTITION}), CATALOG).run(ToyPoint())

    assert values == (
        ProbeValue(
            "refusing/refused", None, 3.0, 3, None, None, None, None, None, None
        ),
    )


def test_declared_roles_restrict_the_points_read() -> None:
    values = DiagnosticsProbe(frozenset({Role.CHART_CELL}), CATALOG).run(ToyPoint())

    assert [value.measure for value in values] == ["mean-cells/mean"]


def test_a_declared_role_no_component_publishes_is_refused() -> None:
    probe = DiagnosticsProbe(
        frozenset({Role.ATTENTION_WEIGHTS, Role.CHART_CELL}), CATALOG
    )

    with pytest.raises(UnpublishedSiteError, match="ATTENTION_WEIGHTS") as refused:
        probe.admit(published_sites(ToyPoint().components))

    assert "CHART_CELL" in str(refused.value).split(";")[1]


def test_every_role_admits_any_observable_point() -> None:
    DiagnosticsProbe(None, CATALOG).admit(published_sites(ToyPoint().components))


class _Silent(ToyChart):
    @override
    def _published(self, reading: _Reading) -> Tensor:
        if reading.site == LOG_Z:
            return reading.tensor

        return super()._published(reading)


def test_a_declared_role_no_pass_reaches_is_an_error() -> None:
    probe = DiagnosticsProbe(frozenset({Role.LOG_PARTITION}), CATALOG)

    with pytest.raises(ValueError, match="LOG_PARTITION"):
        probe.run(ToyPoint(_Silent()))


def test_the_plugin_builds_the_declared_roles_over_the_installed_catalog() -> None:
    declaration = DiagnosticsProbeSpec.model_validate(
        {"kind": "diagnostics", "roles": ["CHART_CELL", "GATE"]}
    )
    built = PLUGIN.construct(declaration, ProbeBuildContext("diag", ToyPoint()))

    assert built.roles == frozenset({Role.CHART_CELL, Role.GATE})
    assert repr(built) == "DiagnosticsProbe(roles=['CHART_CELL', 'GATE'])"


def test_the_plugin_reads_every_role_by_default() -> None:
    built = PLUGIN.construct(
        DiagnosticsProbeSpec.model_validate({"kind": "diagnostics"}),
        ProbeBuildContext("diag", ToyPoint()),
    )

    assert built.roles is None

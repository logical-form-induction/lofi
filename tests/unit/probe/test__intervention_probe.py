from typing import TYPE_CHECKING, cast, override

import pytest
from tests.conformance.declared_probe import (
    CELLS,
    CHART,
    LOG_Z,
    MU,
    PUBLISHED,
    DeclaredProbeSuite,
    ToyChart,
    ToyPoint,
)

from lofi.nn.sites import Access, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe import UnpublishedSiteError, observed_pass, published_sites
from lofi.probe._boundary import (
    InterventionDecl,
    InterventionProbeSpec,
    MeanDecl,
    NoiseDecl,
    ResampleDecl,
    ShuffleDecl,
    ZeroDecl,
)
from lofi.probe._intervention_probe import PLUGIN, InterventionProbe
from lofi.probe._probing import GradientMode, ProbeBuildContext

if TYPE_CHECKING:
    from typing import Protocol

    from lofi.probe._probing import InterventionSource

    class _InterventionDeclaration(Protocol):
        """A declaration that supplies an intervention source."""

        def source(self) -> InterventionSource:
            """Return what draws the intervention from the unpatched pass."""
            ...


class TestInterventionProbeConformance(DeclaredProbeSuite):
    @pytest.fixture
    def declared_probe(self) -> InterventionProbe:
        return InterventionProbe(
            ((CELLS, NoiseDecl(sigma=0.5, seed=2).source()),), Role.LOG_PARTITION
        )


def clean_magnitudes(point: ToyPoint) -> list[float]:
    return [
        float(
            observed_pass(
                point,
                held_out,
                observe=frozenset({LOG_Z}),
                patch={},
                gradients=GradientMode.DISABLED,
            )[0][LOG_Z]
            .tensor.abs()
            .mean()
        )
        for held_out in point.passes
    ]


def test_zeroing_the_span_vectors_changes_the_outcome_by_its_own_size() -> None:
    point = ToyPoint()
    magnitudes = clean_magnitudes(point)
    values = InterventionProbe(((CELLS, ZeroDecl().source()),), Role.LOG_PARTITION).run(
        point
    )

    (measured,) = values

    assert (measured.measure, measured.site, measured.count) == (
        "zero/effect",
        CELLS,
        3,
    )
    assert measured.value == pytest.approx(sum(magnitudes) / 3)


def test_each_intervention_is_scored_alone_in_declared_order() -> None:
    values = InterventionProbe(((CELLS, MeanDecl().source()),), Role.LOG_PARTITION).run(
        ToyPoint()
    )
    noisy = InterventionProbe(
        ((CELLS, NoiseDecl(sigma=0.5, seed=2).source()),), Role.LOG_PARTITION
    ).run(ToyPoint())

    assert [value.measure for value in values] == ["mean-from/effect"]
    assert [value.measure for value in noisy] == ["noise/effect"]
    assert values[0].value > 0.0
    assert noisy[0].value > 0.0


class _PatchableTotal(ToyChart):
    """A chart whose log partition may also be patched."""

    @override
    def site_map(self) -> SiteMap:
        return SiteMap.custom(
            CHART,
            {
                **dict(PUBLISHED.items()),
                LOG_Z: SiteSpec(
                    ("item",),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE | Access.PATCH,
                ),
            },
        )


def test_no_intervention_is_in_place_while_another_is_scored() -> None:
    both = InterventionProbe(
        ((LOG_Z, ZeroDecl().source()), (CELLS, ZeroDecl().source())),
        Role.CONSTITUENT_MARGINAL,
    ).run(ToyPoint(_PatchableTotal()))
    alone = InterventionProbe(
        ((CELLS, ZeroDecl().source()),), Role.CONSTITUENT_MARGINAL
    ).run(ToyPoint(_PatchableTotal()))

    assert [value.site for value in both] == [LOG_Z, CELLS]
    assert both[0].value == 0.0
    assert both[0].count == 3
    assert both[1] == alone[0]
    assert alone[0].value > 0.0


def test_the_outcome_is_compared_inside_its_mask() -> None:
    point = ToyPoint()
    values = InterventionProbe(
        ((CELLS, ZeroDecl().source()),), Role.CONSTITUENT_MARGINAL
    ).run(point)
    expected = []

    for held_out in point.passes:
        trace, _ = observed_pass(
            point,
            held_out,
            observe=frozenset({MU}),
            patch={},
            gradients=GradientMode.DISABLED,
        )
        reading = trace[MU]
        live = reading.covering()
        expected.append(float((reading.tensor - 0.5)[live].abs().mean()))

    assert values[0].value == pytest.approx(sum(expected) / 3)


def test_a_site_that_cannot_be_patched_is_refused() -> None:
    published = published_sites(ToyPoint().components)
    refusing = InterventionProbe(((MU, ZeroDecl().source()),), Role.LOG_PARTITION)

    with pytest.raises(UnpublishedSiteError, match="CONSTITUENT_MARGINAL") as refused:
        refusing.admit(published)

    assert "chart/CHART_CELL/cells" in str(refused.value)


def patch_only_cells() -> dict[Site, SiteSpec]:
    published = dict(published_sites(ToyPoint().components))
    published[CELLS] = SiteSpec(
        ("item", "span", "width"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.PATCH,
    )
    return published


@pytest.mark.parametrize(
    "declaration", [ZeroDecl(), NoiseDecl(sigma=1.0), ShuffleDecl()]
)
def test_a_standing_intervention_needs_only_a_patchable_site(
    declaration: InterventionDecl,
) -> None:
    probe = InterventionProbe(
        ((CELLS, cast("_InterventionDeclaration", declaration).source()),),
        Role.LOG_PARTITION,
    )

    probe.admit(patch_only_cells())


@pytest.mark.parametrize("declaration", [MeanDecl(), ResampleDecl()])
def test_an_intervention_drawn_from_the_unpatched_pass_needs_an_observable_site(
    declaration: InterventionDecl,
) -> None:
    probe = InterventionProbe(
        ((CELLS, cast("_InterventionDeclaration", declaration).source()),),
        Role.LOG_PARTITION,
    )

    with pytest.raises(UnpublishedSiteError, match=r"permitting OBSERVE\|PATCH;"):
        probe.admit(patch_only_cells())

    probe.admit(published_sites(ToyPoint().components))


def test_an_unpublished_outcome_is_refused() -> None:
    published = published_sites(ToyPoint().components)
    refusing = InterventionProbe(
        ((CELLS, ZeroDecl().source()),), Role.ATTENTION_WEIGHTS
    )

    with pytest.raises(UnpublishedSiteError, match="ATTENTION_WEIGHTS"):
        refusing.admit(published)


def test_the_plugin_builds_the_declared_interventions() -> None:
    built = PLUGIN.construct(
        InterventionProbeSpec.model_validate(
            {
                "kind": "intervention",
                "outcome": "CONSTITUENT_MARGINAL",
                "interventions": [
                    {
                        "site": {
                            "component": "chart",
                            "role": "CHART_CELL",
                            "key": ["cells"],
                        },
                        "intervention": {"kind": "shuffle", "seed": 4},
                    }
                ],
            }
        ),
        ProbeBuildContext("patch", ToyPoint()),
    )

    assert built.sites == (CELLS,)
    assert built.outcome is Role.CONSTITUENT_MARGINAL
    assert repr(built) == (
        "InterventionProbe(['chart/CHART_CELL/cells'], outcome=CONSTITUENT_MARGINAL)"
    )

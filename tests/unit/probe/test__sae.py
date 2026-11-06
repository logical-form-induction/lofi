from math import isfinite
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest
import torch
from tests.unit.probe.test__causal_probe import PUBLISHED, SITE, _Point

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._boundary import SaeProbeSpec
from lofi.probe._probing import ProbeBuildContext, UnpublishedSiteError
from lofi.probe._sae import PLUGIN, SaeProbe

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _declaration(**changed: JsonValue) -> SaeProbeSpec:
    document: dict[str, JsonValue] = {
        "kind": "sae",
        "site": {
            "component": str(SITE.component),
            "role": SITE.role.name,
            "key": list(SITE.key),
        },
        "width": 4,
        "sparsity_penalty": 0.02,
        "epochs": 100,
        "learning_rate": 0.03,
        "seed": 7,
    }
    document.update(changed)
    return SaeProbeSpec.model_validate(document)


def _probe(point: _Point | None = None) -> SaeProbe:
    trained = _Point() if point is None else point
    return PLUGIN.construct(
        _declaration(),
        ProbeBuildContext("representations", trained),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("width", 0, "width"),
        ("sparsity_penalty", -0.1, "sparsity penalty"),
        ("epochs", 0, "epoch count"),
        ("learning_rate", 0.0, "learning rate"),
        ("seed", -1, "seed"),
    ],
)
def test_an_sae_declaration_refuses_an_invalid_fitting_setting(
    field: str, value: JsonValue, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        _declaration(**{field: value})


def test_the_plugin_builds_against_the_same_causal_capability_as_das() -> None:
    built = _probe()

    assert isinstance(built, SaeProbe)


def test_an_sae_requires_its_declared_observable_activation_site() -> None:
    probe = _probe()

    probe.admit(PUBLISHED)
    with pytest.raises(UnpublishedSiteError, match="unpublished site"):
        probe.admit({})


def test_the_current_term_encoding_role_is_an_admissible_sae_site() -> None:
    site = Site(ComponentId("term-encoder"), Role.TERM_ENCODING)
    settings = _declaration().settings()._replace(site=site)
    probe = SaeProbe(settings)

    probe.admit(
        {
            site: SiteSpec(
                ("term", "width"),
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
        }
    )


def test_an_sae_reports_reconstruction_sparsity_and_variable_alignments() -> None:
    point = _Point()
    probe = _probe(point)
    probe.admit(PUBLISHED)

    values = probe.run(point)
    measured = {value.measure: value for value in values}

    assert set(measured) == {
        "pass-0/reconstruction-mse",
        "pass-0/activation-sparsity",
        "pass-0/feature-alignment/type-0-4/feature",
        "pass-0/feature-alignment/type-0-4/accuracy",
        "pass-0/feature-alignment/type-1-3/feature",
        "pass-0/feature-alignment/type-1-3/accuracy",
    }
    assert all(value.site == SITE for value in values)
    assert all(value.count > 0 and isfinite(value.value) for value in values)
    assert 0.0 <= measured["pass-0/activation-sparsity"].value <= 1.0
    for variable in ("type-0-4", "type-1-3"):
        feature = measured[f"pass-0/feature-alignment/{variable}/feature"].value
        accuracy = measured[f"pass-0/feature-alignment/{variable}/accuracy"].value
        assert 0.0 <= feature < 4.0
        assert accuracy >= 0.5


def test_an_sae_is_repeatable_without_changing_the_subject_or_global_rng() -> None:
    point = _Point()
    probe = _probe(point)
    before = torch.random.get_rng_state()

    first = probe.run(point)
    middle = torch.random.get_rng_state()
    second = probe.run(point)
    after = torch.random.get_rng_state()

    assert first == second
    assert torch.equal(before, middle)
    assert torch.equal(middle, after)
    assert point.passes[0].subject.scopes == []

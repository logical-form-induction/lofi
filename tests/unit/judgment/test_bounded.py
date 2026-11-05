import math
from typing import NamedTuple, cast

import pytest
import torch
from torch.distributions import Normal

from lofi.data import (
    RESPONSE_MISSING,
    AnnotatorId,
    ItemId,
    RelationalScalar,
    Scalar,
    Scale,
)
from lofi.judgment import (
    AnonymousParticipantPolicy,
    Apply,
    AttributedItem,
    Bind,
    BoundedSource,
    ClosedUnitTruncatedNormal,
    Draw,
    Function,
    ItemFactors,
    Literal,
    Name,
    ObserveBounded,
    Prior,
    TruncatedNormalFamily,
    closed_unit_log_prob,
    related_item_key,
)


class Sentence(NamedTuple):
    item_id: str
    languages: tuple[str, ...]


class Rated(NamedTuple):
    item: Sentence
    supervision: Scalar | RelationalScalar


def continuous(
    values: tuple[float | None, ...],
    participants: tuple[str | None, ...],
) -> tuple[Rated, ...]:
    scale = Scale("continuous", 0.0, 1.0)

    return tuple(
        Rated(
            Sentence("pair", ("en",)),
            Scalar(
                value,
                scale,
                None if participant is None else AnnotatorId(participant),
                features=frozenset({RESPONSE_MISSING})
                if value is None
                else frozenset(),
            ),
        )
        for value, participant in zip(values, participants, strict=True)
    )


def test_a_bounded_source_encodes_continuous_rows_without_ordinal_levels() -> None:
    source = BoundedSource(continuous((0.0, 0.73, 1.0), ("p1", "p2", "p3")))
    design = source.design()

    assert source.factors == ()
    assert design.responses.tolist() == [0.0, 0.73, 1.0]
    assert design.unit_responses.tolist() == [0.0, 0.73, 1.0]
    assert design.item_levels == ("pair",)
    assert design.participant_levels == ("p1", "p2", "p3")
    assert design.response_levels is None


def test_a_bounded_source_assigns_unattributed_rows_distinct_singletons() -> None:
    source = BoundedSource(
        continuous((0.25, 0.75), (None, None)),
        participant_policy=AnonymousParticipantPolicy.SINGLETON,
    )
    design = source.design()

    assert design.participant_levels == ("anonymous-0", "anonymous-1")
    assert source.singleton_participants == 2
    assert source.skips == {}


def test_a_bounded_source_retains_a_relational_scalars_typed_pair() -> None:
    premise = ItemId("at://local/expression/premise")
    hypothesis = ItemId("at://local/expression/hypothesis")
    response = RelationalScalar(
        hypothesis,
        "entails",
        Scalar(0.73, Scale("continuous", 0.0, 1.0), AnnotatorId("p1")),
    )
    source = BoundedSource((Rated(Sentence(premise, ("en",)), response),))

    assert source.rows()[0].item == f"{premise} -> {hypothesis}"
    assert source.design().item_levels == (f"{premise} -> {hypothesis}",)


def test_a_bounded_source_reads_factors_from_the_related_item_record() -> None:
    premise = ItemId("at://local/expression/premise")
    hypothesis = ItemId("at://local/expression/hypothesis")
    response = RelationalScalar(
        hypothesis,
        "entails",
        Scalar(0.73, Scale("continuous", 0.0, 1.0), AnnotatorId("p1")),
    )
    key = related_item_key((premise, hypothesis))
    source = BoundedSource(
        (Rated(Sentence(premise, ("en",)), response),),
        factors=ItemFactors(
            {"kind": "item['kind']"}, (AttributedItem(key, {"kind": "k"}, {}),)
        ),
    )

    assert source.factors == ("kind",)
    assert source.design().factor_levels["kind"] == ("k",)


def test_a_bounded_source_counts_missing_values_and_unattributed_rows() -> None:
    source = BoundedSource(continuous((None, 0.73), ("p1", None)))

    assert source.rows() == ()
    assert source.skips == {"missing-response": 1, "missing-participant": 1}


def test_a_bounded_source_refuses_an_ordinal_scale() -> None:
    rated = (
        Rated(
            Sentence("item", ("en",)),
            Scalar(1.0, Scale("ordinal", 1.0, 7.0), AnnotatorId("p1")),
        ),
    )

    with pytest.raises(ValueError, match="noncontinuous or unbounded"):
        BoundedSource(rated)


def test_the_density_has_the_exact_inclusive_normalization() -> None:
    response = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64)
    location = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
    scale = torch.tensor(0.23, dtype=torch.float64)
    normal = Normal(location, scale)
    expected = normal.log_prob(response) - torch.log(
        normal.cdf(torch.ones_like(location)) - normal.cdf(torch.zeros_like(location))
    )

    actual = closed_unit_log_prob(response, location, scale)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert bool(torch.isfinite(actual).all())


def test_the_density_includes_both_endpoints_and_excludes_values_beyond_them() -> None:
    response = torch.tensor(
        [
            -torch.finfo(torch.float64).eps,
            0.0,
            1.0,
            1.0 + torch.finfo(torch.float64).eps,
        ],
        dtype=torch.float64,
    )
    actual = closed_unit_log_prob(
        response,
        torch.full_like(response, 0.5),
        torch.full_like(response, 0.2),
    )

    assert torch.isneginf(actual[[0, 3]]).tolist() == [True, True]
    assert torch.isfinite(actual[[1, 2]]).tolist() == [True, True]


def test_the_density_stays_finite_as_a_large_scale_approaches_uniform() -> None:
    response = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    location = torch.full_like(response, 0.3)
    scale = torch.full_like(response, 1e16)

    actual = closed_unit_log_prob(response, location, scale)

    assert bool(torch.isfinite(actual).all())
    assert torch.allclose(actual, torch.zeros_like(actual), atol=1e-12, rtol=0.0)


def test_the_density_differentiates_location_and_scale() -> None:
    response = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    location = torch.tensor(0.43, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)

    closed_unit_log_prob(response, location, scale).sum().backward()

    assert location.grad is not None
    assert scale.grad is not None
    assert math.isfinite(float(location.grad))
    assert math.isfinite(float(scale.grad))
    assert float(location.grad) != 0.0
    assert float(scale.grad) != 0.0


def test_the_distribution_reads_the_same_density() -> None:
    response = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)
    location = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    scale = torch.tensor(0.25, dtype=torch.float64)
    distribution = ClosedUnitTruncatedNormal(location, scale)

    assert torch.equal(
        distribution.log_prob(response),
        closed_unit_log_prob(response, location, scale),
    )


def test_a_fixed_family_binds_location_scale_and_the_observation() -> None:
    family = TruncatedNormalFamily(0.2)
    steps = family.steps(predictor="eta", response="y", plate="Rows")

    assert steps == (
        Bind("bounded_location", Apply(Function.SIGMOID, Name("eta"))),
        Bind("bounded_scale", Literal(0.2)),
        ObserveBounded(
            "y",
            "Rows",
            location="bounded_location",
            scale="bounded_scale",
        ),
    )


def test_a_drawn_family_draws_one_global_scale() -> None:
    prior = Prior("LogNormal", 0.0, 0.5)
    family = TruncatedNormalFamily(prior)
    steps = family.steps(predictor="eta", response="y", plate="Rows")

    assert steps[1] == Draw("bounded_scale", ("Unit",), prior)
    assert isinstance(steps[-1], ObserveBounded)


@pytest.mark.parametrize("scale", [0.0, -0.1, float("nan"), float("inf")])
def test_a_fixed_family_refuses_a_nonpositive_or_nonfinite_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        TruncatedNormalFamily(scale)


def test_a_fixed_family_refuses_a_boolean_scale() -> None:
    boolean_scale = True

    with pytest.raises(TypeError, match="not a number"):
        TruncatedNormalFamily(cast("float", boolean_scale))

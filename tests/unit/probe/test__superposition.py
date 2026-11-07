from typing import Final

import pytest
import torch

from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders.hrr import HolographicEncoderSettings, HrrTermEncoder
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._superposition import SuperpositionInterference
from lofi.probe._values import Observation, Trace
from lofi.terms import (
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

FIRST: Final = Site(ComponentId("first-encoder"), Role.TERM_ENCODING)
SECOND: Final = Site(ComponentId("second-encoder"), Role.TERM_ENCODING)
OTHER: Final = Site(ComponentId("parser"), Role.CHART_CELL)

REAL: Final = SiteSpec(
    ("term", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
OTHER_SPEC: Final = SiteSpec(
    ("term", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)


def observed(
    site: Site,
    values: torch.Tensor,
    *,
    axes: tuple[str, ...] = ("term", "width"),
    mask: torch.Tensor | None = None,
) -> Observation:
    return Observation(site, values, axes=axes, mask=mask)


def traced(*readings: Observation) -> Trace:
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def run(values: torch.Tensor, *, mask: torch.Tensor | None = None) -> DiagnosticResult:
    trace = traced(observed(FIRST, values, mask=mask))

    return SuperpositionInterference().run(
        trace, context=DiagnosticContext({FIRST: REAL})
    )


def test_the_diagnostic_declares_its_role_scores_and_zero_argument_constructor() -> (
    None
):
    diagnostic = SuperpositionInterference()

    assert diagnostic.tag == "superposition-interference"
    assert diagnostic.requires == frozenset({Role.TERM_ENCODING})
    assert diagnostic.scores == frozenset(
        f"interference-at-{size}" for size in range(1, 6)
    )


def test_orthogonal_prefixes_have_the_exact_superposition_interference() -> None:
    result = run(torch.eye(6, dtype=torch.float32))

    assert result.value(FIRST, "terms") == 6.0
    expected = (0.0, 0.1464466094, 0.2113248654, 0.25, 0.2763932023)

    for size, value in enumerate(expected, start=1):
        assert result.value(FIRST, f"interference-at-{size}") == pytest.approx(value)

    assert len(result.measurements) == 6


@pytest.mark.parametrize("scale", [1e-6, 1e6, -3.0])
def test_a_common_nonzero_scale_leaves_identical_rows_at_zero_interference(
    scale: float,
) -> None:
    values = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]) * scale
    result = run(values)

    assert result.value(FIRST, "interference-at-1") == pytest.approx(0.0)
    assert result.value(FIRST, "interference-at-2") == pytest.approx(0.0)
    assert result.value(FIRST, "interference-at-3") == pytest.approx(0.0)


@pytest.mark.parametrize("scale", [1e-6, 1e6, -3.0])
def test_a_common_nonzero_scale_preserves_nontrivial_interference(
    scale: float,
) -> None:
    values = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-0.5, 2.0]])
    reference = run(values)
    scaled = run(values * scale)

    for size in range(1, 4):
        name = f"interference-at-{size}"

        assert scaled.value(FIRST, name) == pytest.approx(
            reference.value(FIRST, name), abs=1e-12
        )


def test_partial_masks_remove_whole_rows_without_changing_remaining_order() -> None:
    values = torch.tensor([[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]])
    mask = torch.tensor([[True, True], [True, False], [True, True]], dtype=torch.bool)
    result = run(values, mask=mask)

    assert result.value(FIRST, "terms") == 2.0
    assert result.value(FIRST, "interference-at-1") == pytest.approx(0.0)
    assert result.value(FIRST, "interference-at-2") == pytest.approx(0.1464466094)


def test_an_all_masked_reading_is_refused_naming_its_site() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match=r"first-encoder.*masked"):
        run(values, mask=torch.tensor([False, False]))


def test_a_live_zero_norm_member_is_refused_naming_its_site_and_source_row() -> None:
    values = torch.tensor([[5.0, 5.0], [1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([False, True, True])

    with pytest.raises(ValueError, match=r"first-encoder.*row 2"):
        run(values, mask=mask)


def test_a_canceled_prefix_is_omitted_without_stopping_later_prefixes() -> None:
    values = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    result = run(values)

    assert result.value(FIRST, "interference-at-1") == pytest.approx(0.0)

    with pytest.raises(KeyError):
        result.value(FIRST, "interference-at-2")

    assert result.value(FIRST, "interference-at-3") == pytest.approx(1.0 / 3.0)


def test_no_published_type_encoding_returns_an_empty_result() -> None:
    result = SuperpositionInterference().run(
        traced(observed(OTHER, torch.ones(2, 2))),
        context=DiagnosticContext({OTHER: OTHER_SPEC}),
    )

    assert result.measurements == ()
    assert result.aggregations == ()


def test_published_but_unread_type_encodings_raise_a_named_error() -> None:
    with pytest.raises(ValueError, match=r"missing.*first-encoder"):
        SuperpositionInterference().run(
            traced(), context=DiagnosticContext({FIRST: REAL})
        )


def test_a_partial_observation_set_refuses_the_missing_published_site_by_name() -> None:
    third = Site(ComponentId("third-encoder"), Role.TERM_ENCODING)
    context = DiagnosticContext({SECOND: REAL, FIRST: REAL, third: REAL})
    trace = traced(
        observed(FIRST, torch.tensor([[0.0, 1.0]])),
        observed(SECOND, torch.tensor([[1.0, 0.0]])),
    )

    with pytest.raises(
        ValueError, match="missing TERM_ENCODING observation"
    ) as failure:
        SuperpositionInterference().run(trace, context=context)

    assert str(failure.value) == (
        f"missing TERM_ENCODING observation at published site(s): {third!r}"
    )


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            SiteSpec(
                ("width", "term"),
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            "published over",
        ),
        (
            SiteSpec(
                ("term", "width"),
                DTypeClass.PROBABILITY,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            "not real",
        ),
    ],
)
def test_each_published_type_encoding_spec_is_validated(
    spec: SiteSpec, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SuperpositionInterference().run(
            traced(observed(FIRST, torch.ones(2, 2))),
            context=DiagnosticContext({FIRST: spec}),
        )


def test_an_observation_must_use_the_term_and_width_axes() -> None:
    with pytest.raises(ValueError, match="runs over"):
        SuperpositionInterference().run(
            traced(observed(FIRST, torch.ones(2, 2), axes=("width", "term"))),
            context=DiagnosticContext({FIRST: REAL}),
        )


def test_an_observation_must_be_floating_point_and_at_least_two_wide() -> None:
    with pytest.raises(TypeError, match="floating point"):
        run(torch.ones(2, 2, dtype=torch.int64))

    with pytest.raises(ValueError, match="less than two"):
        run(torch.ones(2, 1))


def test_a_real_hrr_encoding_can_be_measured_at_its_published_boundary() -> None:
    atom = Operator("a", "T", family="primitive")
    other = Operator("b", "T", family="primitive")
    signature = Signature("superposition", ("T",), (atom, other))
    index = OperatorIndex(signature)
    codec = TermCodec(
        index,
        CompleteTree(signature, signature.operators, order="bf", depth=1),
    )
    store = TermStore(signature)
    terms = codec.encode_batch((store.op(atom), store.op(other)))
    encoder = HrrTermEncoder(
        codec=NeuralCodecView(codec),
        component=ComponentId("measured-hrr"),
        settings=HolographicEncoderSettings(width=8, magnitude=1.0, floor=1e-3),
    )
    encoded = encoder.encode(terms)
    site = next(
        point for point in encoder.site_map() if point.role is Role.TERM_ENCODING
    )
    observation = Observation(
        site, encoded.tensor, axes=encoded.axes, mask=encoded.mask
    )
    result = SuperpositionInterference().run(
        traced(observation), context=DiagnosticContext.observing(encoder)
    )

    assert result.value(site, "terms") == 2.0
    assert result.value(site, "interference-at-1") == pytest.approx(0.0)
    assert result.value(site, "interference-at-2") >= 0.0

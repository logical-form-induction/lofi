"""Tests for observation-only Transformer diagnostics."""

from copy import copy, deepcopy
from math import log
from types import MappingProxyType
from typing import Final

import pytest
import torch

from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders.transformer import (
    TransformerEncoderSettings,
    TransformerTermEncoder,
)
from lofi.nn.ports import Consumption, StructuralRelation
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._instrument import instrument
from lofi.probe._observation_groups import ObservationGrouping
from lofi.probe._transformer import (
    ATTENTION_POLICY,
    AttentionPatterns,
    AttentionPolicy,
    FeedForwardSpecialization,
    ResidualContributions,
)
from lofi.probe._values import Observation, Trace
from lofi.terms import (
    BatchEncoding,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
    TermStore,
)

ENCODER: Final = ComponentId("transformer-encoder")


def logits(head: int, layer: int = 0) -> Site:
    return Site(ENCODER, Role.ATTENTION_LOGITS, (layer, head))


def weights(head: int, layer: int = 0) -> Site:
    return Site(ENCODER, Role.ATTENTION_WEIGHTS, (layer, head))


def preactivation(layer: int = 0) -> Site:
    return Site(ENCODER, Role.FFN_PREACTIVATION, (layer,))


def residual(stage: str, layer: int = 0) -> Site:
    return Site(ENCODER, Role.RESIDUAL_STREAM, (layer, stage))


ATTENTION_LOGITS: Final = SiteSpec(
    ("batch", "position", "attended-position"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
ATTENTION_WEIGHTS: Final = SiteSpec(
    ("batch", "position", "attended-position"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
ACTIVATION: Final = SiteSpec(
    ("batch", "position", "feature"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)


def observed(
    site: Site,
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> Observation:
    axes = (
        ("batch", "position", "attended-position")
        if site.role in (Role.ATTENTION_LOGITS, Role.ATTENTION_WEIGHTS)
        else ("batch", "position", "feature")
    )

    return Observation(site, tensor, axes=axes, mask=mask)


def traced(*readings: Observation) -> Trace:
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def attention_context(heads: int = 2) -> DiagnosticContext:
    published: dict[Site, SiteSpec] = {}

    for head in range(heads):
        published[logits(head)] = ATTENTION_LOGITS
        published[weights(head)] = ATTENTION_WEIGHTS

    return DiagnosticContext(published)


def attention_trace(
    distributions: tuple[torch.Tensor, ...],
    *,
    mask: torch.Tensor | None = None,
    logit_values: tuple[torch.Tensor, ...] | None = None,
) -> Trace:
    support = (
        tuple(torch.ones_like(distribution) for distribution in distributions)
        if logit_values is None
        else logit_values
    )
    readings: list[Observation] = []

    for head, (distribution, head_logits) in enumerate(
        zip(distributions, support, strict=True)
    ):
        inferred = torch.where(
            head_logits > 0,
            head_logits.log(),
            torch.full_like(head_logits, -torch.inf),
        )
        readings.extend(
            (
                observed(logits(head), inferred, mask=mask),
                observed(weights(head), distribution, mask=mask),
            )
        )

    return traced(*readings)


def result_values(result: DiagnosticResult, site: Site) -> dict[str, float]:
    return {
        measurement.name: measurement.value
        for measurement in result.measurements
        if measurement.site == site
    }


def test_the_fixed_attention_policy_is_public_immutable_and_copies_by_identity() -> (
    None
):
    assert isinstance(ATTENTION_POLICY, AttentionPolicy)
    assert ATTENTION_POLICY.significant_mass == 0.1
    assert ATTENTION_POLICY.local_radius == 1
    assert repr(ATTENTION_POLICY) == "AttentionPolicy()"
    assert copy(ATTENTION_POLICY) is ATTENTION_POLICY
    assert deepcopy(ATTENTION_POLICY) is ATTENTION_POLICY

    with pytest.raises(AttributeError, match="attention policy is immutable"):
        ATTENTION_POLICY.__setattr__("significant_mass", 0.2)


def test_attention_metrics_match_uniform_hand_distributions() -> None:
    uniform = torch.full((1, 3, 3), 1.0 / 3.0)
    result = AttentionPatterns().run(
        attention_trace((uniform, uniform)), context=attention_context()
    )

    for site in (weights(0), weights(1)):
        values = result_values(result, site)

        assert values["queries"] == 3.0
        assert values["normalized-entropy"] == pytest.approx(1.0)
        assert values["significant-mass-coverage"] == pytest.approx(1.0)
        assert values["maximum-concentration"] == pytest.approx(1.0 / 3.0)
        assert values["self-mass"] == pytest.approx(1.0 / 3.0)
        assert values["local-mass"] == pytest.approx(7.0 / 9.0)
        assert values["head-consistency"] == pytest.approx(1.0)


def test_significant_mass_uses_the_inclusive_policy_boundary() -> None:
    boundary = torch.tensor([[[0.8, 0.1, 0.1]]]).repeat(1, 3, 1)
    below = torch.tensor([[[0.82, 0.09, 0.09]]]).repeat(1, 3, 1)
    at_boundary = AttentionPatterns().run(
        attention_trace((boundary,)), context=attention_context(1)
    )
    under_boundary = AttentionPatterns().run(
        attention_trace((below,)), context=attention_context(1)
    )

    assert at_boundary.value(weights(0), "significant-mass-coverage") == 1.0
    assert under_boundary.value(
        weights(0), "significant-mass-coverage"
    ) == pytest.approx(0.82)


def test_identical_and_disjoint_heads_bound_cross_head_consistency() -> None:
    first = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    second = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    identical = AttentionPatterns().run(
        attention_trace((first, first)), context=attention_context()
    )
    disjoint = AttentionPatterns().run(
        attention_trace((first, second)), context=attention_context()
    )

    assert identical.value(weights(0), "head-consistency") == pytest.approx(1.0)
    assert disjoint.value(weights(0), "head-consistency") == pytest.approx(0.0)


def test_padding_does_not_change_attention_summaries() -> None:
    compact = torch.full((1, 3, 3), 1.0 / 3.0)
    padded = torch.zeros(1, 4, 4)
    padded[:, :3, :3] = 1.0 / 3.0
    support = torch.zeros_like(padded)
    support[:, :3, :3] = 1.0
    mask = torch.tensor([[True, True, True, False]])
    without = AttentionPatterns().run(
        attention_trace((compact,)), context=attention_context(1)
    )
    with_padding = AttentionPatterns().run(
        attention_trace((padded,), mask=mask, logit_values=(support,)),
        context=attention_context(1),
    )

    assert result_values(without, weights(0)) == pytest.approx(
        result_values(with_padding, weights(0))
    )


def test_a_full_attention_mask_excludes_individual_keys_and_queries() -> None:
    distribution = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    result = AttentionPatterns().run(
        attention_trace((distribution,), mask=mask),
        context=attention_context(1),
    )

    assert result.value(weights(0), "queries") == 2.0
    assert result.value(weights(0), "normalized-entropy") == pytest.approx(1.0)
    assert result.value(weights(0), "maximum-concentration") == pytest.approx(0.5)


def test_coordinated_query_and_key_reversal_preserves_attention_summaries() -> None:
    head = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]])
    reversed_head = head.flip((1, 2))
    first = AttentionPatterns().run(
        attention_trace((head,)), context=attention_context(1)
    )
    second = AttentionPatterns().run(
        attention_trace((reversed_head,)), context=attention_context(1)
    )

    assert result_values(first, weights(0)) == pytest.approx(
        result_values(second, weights(0))
    )


def test_coordinated_head_permutation_preserves_the_layer_summaries() -> None:
    focused = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    even = torch.full((1, 2, 2), 0.5)
    first = AttentionPatterns().run(
        attention_trace((focused, even)), context=attention_context()
    )
    swapped = AttentionPatterns().run(
        attention_trace((even, focused)), context=attention_context()
    )
    names = (
        "normalized-entropy",
        "significant-mass-coverage",
        "maximum-concentration",
        "self-mass",
        "local-mass",
    )

    assert sorted(
        tuple(first.value(site, name) for name in names)
        for site in (weights(0), weights(1))
    ) == pytest.approx(
        sorted(
            tuple(swapped.value(site, name) for name in names)
            for site in (weights(0), weights(1))
        )
    )
    assert first.value(weights(0), "head-consistency") == pytest.approx(
        swapped.value(weights(0), "head-consistency")
    )


def test_attention_requires_each_published_observation_and_matching_pairs() -> None:
    uniform = torch.full((1, 2, 2), 0.5)

    with pytest.raises(ValueError, match=r"missing observation.*ATTENTION_WEIGHTS"):
        AttentionPatterns().run(
            traced(observed(logits(0), uniform.log())),
            context=attention_context(1),
        )

    malformed = DiagnosticContext(
        {logits(0): ATTENTION_LOGITS, weights(1): ATTENTION_WEIGHTS}
    )

    with pytest.raises(ValueError, match="not paired"):
        AttentionPatterns().run(
            traced(
                observed(logits(0), uniform.log()),
                observed(weights(1), uniform),
            ),
            context=malformed,
        )


def ffn_result(
    values: torch.Tensor,
    grouping: ObservationGrouping,
    *,
    mask: torch.Tensor | None = None,
) -> DiagnosticResult:
    site = preactivation()

    return FeedForwardSpecialization().run(
        traced(observed(site, values, mask=mask)),
        context=DiagnosticContext({site: ACTIVATION}, groupings={site: grouping}),
    )


def test_ffn_sparsity_and_group_eta_squared_match_hand_values() -> None:
    values = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0]]]
    )
    result = ffn_result(values, ObservationGrouping(4, ((0, 1), (2, 3))))

    assert result.value(preactivation(), "rows") == 4.0
    assert result.value(preactivation(), "neurons") == 3.0
    assert result.value(preactivation(), "groups") == 2.0
    assert result.value(preactivation(), "sparsity") == pytest.approx(0.5)
    assert result.value(preactivation(), "eta-squared-mean") == pytest.approx(2 / 3)
    assert result.value(preactivation(), "eta-squared-maximum") == pytest.approx(1.0)


def test_declared_groups_change_specialization_without_hidden_labels() -> None:
    values = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0]]]
    )
    by_first = ffn_result(values, ObservationGrouping(4, ((0, 1), (2, 3))))
    by_second = ffn_result(values, ObservationGrouping(4, ((0, 2), (1, 3))))

    assert by_first.value(preactivation(), "eta-squared-mean") > by_second.value(
        preactivation(), "eta-squared-mean"
    )


def test_coordinated_row_and_group_permutation_preserves_specialization() -> None:
    values = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0]]]
    )
    moved = values[:, [2, 0, 3, 1]]
    original = ffn_result(values, ObservationGrouping(4, ((0, 1), (2, 3))))
    permuted = ffn_result(moved, ObservationGrouping(4, ((1, 3), (0, 2))))

    assert result_values(original, preactivation()) == pytest.approx(
        result_values(permuted, preactivation())
    )


def test_grouping_extent_and_referenced_row_liveness_are_checked() -> None:
    values = torch.ones(1, 3, 2)

    with pytest.raises(ValueError, match=r"declares 2 rows.*flattens to 3"):
        ffn_result(values, ObservationGrouping(2, ((0,),)))

    with pytest.raises(ValueError, match=r"group 0.*non-live rows.*1"):
        ffn_result(
            values,
            ObservationGrouping(3, ((1,),)),
            mask=torch.tensor([[True, False, True]]),
        )


def test_a_published_ffn_site_requires_a_grouping_and_an_observation() -> None:
    site = preactivation()
    context = DiagnosticContext({site: ACTIVATION})

    with pytest.raises(ValueError, match=r"missing observation.*FFN_PREACTIVATION"):
        FeedForwardSpecialization().run(traced(), context=context)

    with pytest.raises(KeyError, match=r"no observation grouping.*transformer-encoder"):
        FeedForwardSpecialization().run(
            traced(observed(site, torch.ones(1, 2, 2))), context=context
        )


def residual_context() -> DiagnosticContext:
    return DiagnosticContext(
        {
            residual("input"): ACTIVATION,
            residual("post-attention"): ACTIVATION,
            residual("output"): ACTIVATION,
        }
    )


def test_residual_resultant_and_skip_shares_match_planted_additions() -> None:
    entered = torch.tensor([[[3.0, 0.0]]])
    attended = torch.tensor([[[4.0, 0.0]]])
    output = torch.tensor([[[6.0, 0.0]]])
    result = ResidualContributions().run(
        traced(
            observed(residual("input"), entered),
            observed(residual("post-attention"), attended),
            observed(residual("output"), output),
        ),
        context=residual_context(),
    )

    assert result.value(
        residual("post-attention"), "resultant-norm-share"
    ) == pytest.approx(0.25)
    assert result.value(residual("post-attention"), "skip-norm-share") == pytest.approx(
        0.75
    )
    assert result.value(residual("output"), "resultant-norm-share") == pytest.approx(
        1.0 / 3.0
    )
    assert result.value(residual("output"), "skip-norm-share") == pytest.approx(
        2.0 / 3.0
    )


def test_an_all_zero_residual_pair_assigns_half_to_each_route() -> None:
    zeros = torch.zeros(1, 1, 2)
    result = ResidualContributions().run(
        traced(
            observed(residual("input"), zeros),
            observed(residual("post-attention"), zeros),
            observed(residual("output"), zeros),
        ),
        context=residual_context(),
    )

    assert result.value(residual("output"), "resultant-norm-share") == 0.5
    assert result.value(residual("output"), "skip-norm-share") == 0.5


def test_residual_stages_must_be_complete_observed_and_paired() -> None:
    incomplete = DiagnosticContext(
        {residual("input"): ACTIVATION, residual("post-attention"): ACTIVATION}
    )

    with pytest.raises(ValueError, match="complete stage triples"):
        ResidualContributions().run(
            traced(
                observed(residual("input"), torch.ones(1, 1, 2)),
                observed(residual("post-attention"), torch.ones(1, 1, 2)),
            ),
            context=incomplete,
        )

    with pytest.raises(ValueError, match=r"missing observation.*output"):
        ResidualContributions().run(
            traced(
                observed(residual("input"), torch.ones(1, 1, 2)),
                observed(residual("post-attention"), torch.ones(1, 1, 2)),
            ),
            context=residual_context(),
        )


def test_registry_resolution_uses_only_declared_transformer_roles() -> None:
    transformer = {AttentionPatterns, FeedForwardSpecialization, ResidualContributions}
    no_transformer = {
        type(one)
        for one in installed_diagnostics().matching(
            DiagnosticContext(
                {
                    Site(ENCODER, Role.TERM_ENCODING): SiteSpec(
                        ("term", "width"),
                        DTypeClass.REAL,
                        differentiable=True,
                        access=Access.OBSERVE,
                    )
                }
            )
        )
    }
    attention = {
        type(one) for one in installed_diagnostics().matching(attention_context(1))
    }

    assert not no_transformer & transformer
    assert AttentionPatterns in attention
    assert FeedForwardSpecialization not in attention
    assert ResidualContributions not in attention


def a_real_encoder() -> tuple[TransformerTermEncoder, BatchEncoding]:
    atom = Operator("a", "T", family="primitive")
    unary = Operator("up", "T", arg_sorts=("T",), family="constructor")
    signature = Signature("probe-transformer", ("T",), (atom, unary))
    codec = TermCodec(OperatorIndex(signature), PreorderArityLayout(3))
    store = TermStore(signature)
    terms = codec.encode_batch((store.op(unary, store.op(atom)),))
    encoder = TransformerTermEncoder(
        codec=NeuralCodecView(codec),
        component=ENCODER,
        settings=TransformerEncoderSettings(
            4, 2, 1, 6, norm_cap=None, activation="gelu"
        ),
        structure=MappingProxyType(
            {StructuralRelation.POSITION: Consumption.EMBEDDING}
        ),
    )

    return encoder, terms


def test_real_transformer_observations_reach_each_diagnostic_without_internals() -> (
    None
):
    encoder, terms = a_real_encoder()
    published = encoder.site_map()
    internal = frozenset(
        site
        for site in published
        if site.role
        in (
            Role.ATTENTION_LOGITS,
            Role.ATTENTION_WEIGHTS,
            Role.FFN_PREACTIVATION,
            Role.RESIDUAL_STREAM,
        )
    )

    with instrument(encoder, observe=internal) as region:
        region(lambda: encoder.encode_nodes(terms))

    (ffn_site,) = tuple(
        site for site in published if site.role is Role.FFN_PREACTIVATION
    )
    context = DiagnosticContext(
        published,
        groupings={ffn_site: ObservationGrouping(3, ((0,), (1,)))},
    )

    assert AttentionPatterns().run(region.trace, context=context).measurements
    assert FeedForwardSpecialization().run(region.trace, context=context).measurements
    assert ResidualContributions().run(region.trace, context=context).measurements


def test_attention_entropy_hand_value_uses_natural_log_normalization() -> None:
    distribution = torch.tensor([[[0.5, 0.25, 0.25]]]).repeat(1, 3, 1)
    result = AttentionPatterns().run(
        attention_trace((distribution,)), context=attention_context(1)
    )
    expected = -(0.5 * log(0.5) + 2 * 0.25 * log(0.25)) / log(3)

    assert result.value(weights(0), "normalized-entropy") == pytest.approx(expected)

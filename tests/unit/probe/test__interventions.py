from typing import TYPE_CHECKING, Final, cast, get_args

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._circuits import Unit
from lofi.probe._erasure import LinearErasure, fit_leace
from lofi.probe._interventions import (
    Interchange,
    Intervention,
    JointInterchange,
    MeanFrom,
    Noise,
    PatchFrom,
    Resample,
    Shuffle,
    Sparing,
    Subspace,
    Within,
    Zero,
)
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from collections.abc import Callable

PARSER = ComponentId("cky-parser")
CELL = Site(PARSER, Role.CHART_CELL, (0, 4))
OTHER = Site(PARSER, Role.CHART_CELL, (0, 2))
WEIGHT = Site(PARSER, Role.PARAMETER, ("combinator",))

WIDTH = 8

type InterventionKind = type[
    Zero
    | MeanFrom
    | Resample
    | PatchFrom
    | Noise
    | Shuffle
    | Subspace
    | Interchange
    | JointInterchange
    | Within
    | Sparing
    | LinearErasure
]
type InterventionArgument = (
    Trace | Site | torch.Tensor | float | tuple[int, ...] | frozenset[Unit] | Zero
)


def an_observation(
    site: Site = CELL, tensor: torch.Tensor | None = None, *, axes: tuple[str, ...] = ()
) -> Observation:
    held = torch.arange(4 * WIDTH, dtype=torch.float32).reshape(4, WIDTH)
    held = held if tensor is None else tensor
    named = axes or ("item", "width")

    return Observation(site, held, axes=named)


def a_closed_trace(*observations: Observation) -> Trace:
    trace = Trace()

    for observation in observations:
        trace.record(observation)

    trace.close()

    return trace


def a_rotation(width: int = WIDTH) -> torch.Tensor:
    orthogonal, _ = torch.linalg.qr(
        torch.randn(width, width, generator=torch.Generator().manual_seed(0))
    )

    return orthogonal


def a_source(items: int = 4, width: int = WIDTH, *, seed: int = 1) -> torch.Tensor:
    return torch.randn(items, width, generator=torch.Generator().manual_seed(seed))


def a_mask(rank: int = 3, width: int = WIDTH) -> torch.Tensor:
    return (torch.arange(width) < rank).to(torch.float32)


VARIANTS: Final[tuple[InterventionKind, ...]] = (
    Zero,
    MeanFrom,
    Resample,
    PatchFrom,
    Noise,
    Shuffle,
    Subspace,
    Interchange,
    JointInterchange,
    Within,
    Sparing,
    LinearErasure,
)

STOCHASTIC: Final[tuple[InterventionKind, ...]] = tuple(
    variant for variant in VARIANTS if hasattr(variant, "seed")
)


def arguments_for(
    variant: InterventionKind, seed: int, trace: Trace
) -> tuple[InterventionArgument, ...]:
    given: dict[InterventionKind, tuple[InterventionArgument, ...]] = {
        Zero: (),
        MeanFrom: (trace,),
        Resample: (trace, seed),
        PatchFrom: (trace, CELL),
        Noise: (0.5, seed),
        Shuffle: (seed,),
        Subspace: (a_rotation(), 4, (0,)),
        Interchange: (a_rotation(), a_mask(4), a_source()),
        JointInterchange: (
            a_rotation(),
            a_mask(3),
            ((torch.arange(WIDTH) >= 3) & (torch.arange(WIDTH) < 6)).to(torch.float32),
            a_source(),
        ),
        Within: (frozenset({Unit(CELL, {"item": 0})}), Zero()),
        Sparing: (frozenset({Unit(CELL, {"item": 0})}), Zero()),
        LinearErasure: (),
    }

    return given[variant]


def builder_for(variant: InterventionKind) -> Callable[..., Intervention]:
    builders: dict[InterventionKind, Callable[..., Intervention]] = {
        Zero: Zero,
        MeanFrom: MeanFrom,
        Resample: Resample,
        PatchFrom: PatchFrom,
        Noise: Noise,
        Shuffle: Shuffle,
        Subspace: Subspace,
        Interchange: Interchange,
        JointInterchange: JointInterchange,
        Within: Within,
        Sparing: Sparing,
        LinearErasure: lambda: fit_leace(
            torch.tensor([[2.0, 0.0], [-2.0, 0.0], [2.0, 1.0], [-2.0, 1.0]]),
            torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]),
        ),
    }

    return builders[variant]


def a_variant(variant: InterventionKind, seed: int, trace: Trace) -> Intervention:
    build = builder_for(variant)

    return build(*arguments_for(variant, seed, trace))


def every_variant(seed: int = 0) -> tuple[Intervention, ...]:
    trace = a_closed_trace(an_observation())

    return tuple(a_variant(variant, seed, trace) for variant in VARIANTS)


def stochastic(seed: int) -> tuple[Intervention, ...]:
    trace = a_closed_trace(an_observation())

    return tuple(a_variant(variant, seed, trace) for variant in STOCHASTIC)


def test_zeroing_replaces_the_reading_with_zeros_of_its_shape() -> None:
    observed = an_observation()

    replacement = Zero().stands_in_for(observed)

    assert torch.equal(replacement, torch.zeros_like(observed.tensor))


def test_the_mean_of_a_trace_is_the_mean_of_what_it_recorded_there() -> None:
    recorded = an_observation()
    trace = a_closed_trace(recorded)

    replacement = MeanFrom(trace).stands_in_for(
        an_observation(tensor=torch.zeros(4, WIDTH))
    )

    assert replacement.shape == (4, WIDTH)
    assert replacement.unique().numel() == 1
    assert replacement[0, 0].item() == pytest.approx(recorded.tensor.mean().item())


def test_the_mean_of_a_trace_leaves_masked_positions_out_of_the_average() -> None:
    numbers = torch.tensor([[1.0, 1.0], [9.0, 9.0]])
    mask = torch.tensor([True, False])
    trace = a_closed_trace(
        Observation(CELL, numbers, axes=("item", "width"), mask=mask)
    )

    replacement = MeanFrom(trace).stands_in_for(
        Observation(CELL, torch.zeros(2, 2), axes=("item", "width"))
    )

    assert replacement[0, 0].item() == pytest.approx(1.0)


def test_a_mean_written_into_a_reading_that_is_not_floating_point_is_refused() -> None:
    trace = a_closed_trace(an_observation())
    counts = Observation(
        CELL, torch.zeros(4, WIDTH, dtype=torch.int64), axes=("i", "w")
    )

    with pytest.raises(TypeError, match="a mean cannot be written into"):
        MeanFrom(trace).stands_in_for(counts)


def test_a_mean_over_a_point_the_trace_never_read_names_the_point() -> None:
    trace = a_closed_trace(an_observation(OTHER))

    with pytest.raises(KeyError, match="nothing was read at"):
        MeanFrom(trace).stands_in_for(an_observation())


def test_a_mean_over_a_reading_entirely_masked_off_is_refused() -> None:
    numbers = torch.ones(2, 2)
    trace = a_closed_trace(
        Observation(
            CELL, numbers, axes=("item", "width"), mask=torch.zeros(2, dtype=torch.bool)
        )
    )

    with pytest.raises(ValueError, match="masked off"):
        MeanFrom(trace).stands_in_for(an_observation(tensor=torch.zeros(2, 2)))


def test_resampling_draws_only_numbers_the_source_recorded_there() -> None:
    pool = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    trace = a_closed_trace(Observation(CELL, pool, axes=("item", "width")))

    replacement = Resample(trace, 17).stands_in_for(
        Observation(CELL, torch.zeros(5, 3), axes=("item", "width"))
    )

    assert replacement.shape == (5, 3)
    assert set(replacement.flatten().tolist()) <= set(pool.flatten().tolist())


def test_patching_writes_in_what_the_trace_held_at_the_named_point() -> None:
    held = torch.full((4, WIDTH), 3.0)
    trace = a_closed_trace(an_observation(OTHER, held))

    replacement = PatchFrom(trace, OTHER).stands_in_for(an_observation())

    assert torch.equal(replacement, held)


def test_patching_from_a_point_of_another_shape_is_refused() -> None:
    trace = a_closed_trace(an_observation(OTHER, torch.zeros(2, 2)))

    with pytest.raises(ValueError, match="cannot replace one of shape"):
        PatchFrom(trace, OTHER).stands_in_for(an_observation())


def test_patching_across_the_floating_boundary_is_refused() -> None:
    trace = a_closed_trace(an_observation(OTHER, torch.zeros(4, WIDTH)))
    counts = Observation(
        CELL, torch.zeros(4, WIDTH, dtype=torch.int64), axes=("item", "width")
    )

    with pytest.raises(TypeError, match="cannot replace one held in"):
        PatchFrom(trace, OTHER).stands_in_for(counts)


def test_one_value_type_covers_weight_ablation_and_activation_patching() -> None:
    weights = torch.full((4, WIDTH), 5.0)
    activations = torch.full((4, WIDTH), 7.0)
    source = a_closed_trace(
        an_observation(WEIGHT, weights), an_observation(OTHER, activations)
    )

    onto_a_weight = PatchFrom(source, WEIGHT).stands_in_for(an_observation(WEIGHT))
    onto_an_activation = PatchFrom(source, OTHER).stands_in_for(an_observation(CELL))

    assert torch.equal(onto_a_weight, weights)
    assert torch.equal(onto_an_activation, activations)


def test_noise_moves_every_entry_by_the_declared_scale() -> None:
    observed = an_observation(tensor=torch.zeros(256, WIDTH))

    replacement = Noise(2.0, 5).stands_in_for(observed)

    assert replacement.std().item() == pytest.approx(2.0, rel=0.1)
    assert replacement.mean().item() == pytest.approx(0.0, abs=0.2)


def test_noise_added_to_a_reading_that_is_not_floating_point_is_refused() -> None:
    counts = Observation(CELL, torch.zeros(4, 2, dtype=torch.int64), axes=("i", "w"))

    with pytest.raises(TypeError, match="noise cannot be added"):
        Noise(1.0, 0).stands_in_for(counts)


def test_a_noise_scale_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="noise scale is positive"):
        Noise(0.0, 0)


def test_shuffling_keeps_the_numbers_and_moves_the_outermost_positions() -> None:
    observed = an_observation()

    replacement = Shuffle(3).stands_in_for(observed)

    assert torch.equal(
        replacement.sort(dim=0).values, observed.tensor.sort(dim=0).values
    )
    assert not torch.equal(replacement, observed.tensor)


def test_shuffling_a_single_number_is_refused_rather_than_silently_a_no_op() -> None:
    one = Observation(CELL, torch.tensor(1.0), axes=())

    with pytest.raises(ValueError, match="no axis to shuffle"):
        Shuffle(0).stands_in_for(one)


def test_a_subspace_removes_the_named_coordinates_of_the_rotated_basis() -> None:
    rotation = a_rotation()
    observed = an_observation(tensor=a_source(16))

    replacement = Subspace(rotation, 4, (0, 2)).stands_in_for(observed)
    rotated = replacement @ rotation

    assert torch.allclose(rotated[:, 0], torch.zeros(16), atol=1e-5)
    assert torch.allclose(rotated[:, 2], torch.zeros(16), atol=1e-5)
    assert torch.allclose(rotated[:, 1], (observed.tensor @ rotation)[:, 1], atol=1e-5)


def test_a_subspace_agrees_with_the_projector_written_out_by_hand() -> None:
    rotation = a_rotation()
    reading = a_source(16)
    directions = rotation[:, [0, 2]]

    replacement = Subspace(rotation, 4, (0, 2)).stands_in_for(
        an_observation(tensor=reading),
    )

    assert torch.allclose(
        replacement, reading - (reading @ directions) @ directions.T, atol=1e-5
    )


def test_a_subspace_over_no_coordinate_is_the_control_that_changes_nothing() -> None:
    rotation = a_rotation()
    observed = an_observation(tensor=a_source(16))

    replacement = Subspace(rotation, 4, ()).stands_in_for(observed)

    assert torch.allclose(replacement, observed.tensor, atol=1e-5)


def test_a_subspace_projection_applied_twice_changes_nothing_the_second_time() -> None:
    rotation = a_rotation()
    projection = Subspace(rotation, WIDTH, (1, 5))
    once = projection.stands_in_for(an_observation(tensor=a_source(16)))

    twice = projection.stands_in_for(
        Observation(CELL, once, axes=("item", "width")),
    )

    assert torch.allclose(twice, once, atol=1e-5)


def test_a_rotation_that_is_not_orthogonal_defines_no_projection_and_is_refused() -> (
    None
):
    with pytest.raises(ValueError, match="not orthogonal"):
        Subspace(torch.eye(WIDTH) * 2.0, 4, (0,))


def test_a_rotation_that_is_not_square_is_refused() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        Subspace(torch.zeros(4, 8), 2, (0,))


def test_a_coordinate_at_or_beyond_the_fitted_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="at or beyond a subspace of rank 4"):
        Subspace(a_rotation(), 4, (4,))


def test_a_coordinate_named_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        Subspace(a_rotation(), 4, (1, 1))


def test_a_rank_beyond_the_basis_is_refused() -> None:
    with pytest.raises(ValueError, match="rank 1 to 8"):
        Subspace(a_rotation(), WIDTH + 1, (0,))


def test_a_subspace_over_a_reading_of_another_width_is_refused() -> None:
    with pytest.raises(ValueError, match="does not act on the innermost axis"):
        Subspace(a_rotation(), 4, (0,)).stands_in_for(
            an_observation(tensor=a_source(4, 3))
        )


def test_an_interchange_takes_the_leading_rotated_coordinates_from_the_source() -> None:
    rotation = a_rotation()
    observed = an_observation(tensor=a_source(16))
    source = a_source(16, seed=2)

    replacement = Interchange(rotation, a_mask(), source).stands_in_for(observed)
    rotated = replacement @ rotation

    assert torch.allclose(rotated[:, :3], (source @ rotation)[:, :3], atol=1e-5)
    assert torch.allclose(
        rotated[:, 3:], (observed.tensor @ rotation)[:, 3:], atol=1e-5
    )


def test_an_interchange_at_full_rank_is_the_source_itself() -> None:
    rotation = a_rotation()
    observed = an_observation(tensor=a_source(16))
    source = a_source(16, seed=2)

    replacement = Interchange(rotation, a_mask(WIDTH), source).stands_in_for(observed)

    assert torch.allclose(replacement, source, atol=1e-5)


def test_an_interchange_from_the_reading_itself_changes_nothing() -> None:
    rotation = a_rotation()
    observed = an_observation(tensor=a_source(16))

    replacement = Interchange(rotation, a_mask(), observed.tensor).stands_in_for(
        observed
    )

    assert torch.allclose(replacement, observed.tensor, atol=1e-5)


def test_an_interchange_keeps_the_gradient_through_its_rotation() -> None:
    generator = torch.zeros(WIDTH, WIDTH, requires_grad=True)
    rotation = torch.linalg.matrix_exp(generator - generator.T)
    observed = an_observation(tensor=a_source(16))

    replacement = Interchange(rotation, a_mask(), a_source(16, seed=2)).stands_in_for(
        observed
    )
    replacement.sum().backward()

    assert Interchange(rotation, a_mask(), a_source(16)).rotation.requires_grad
    assert generator.grad is not None
    assert bool((generator.grad != 0.0).any())


def test_an_interchange_detaches_its_source_but_not_its_rotation() -> None:
    rotation = a_rotation().requires_grad_()
    source = a_source().requires_grad_()

    mask = a_mask(2).requires_grad_()
    interchange = Interchange(rotation, mask, source)

    assert interchange.rotation.requires_grad
    assert not interchange.source.requires_grad
    assert interchange.mask is mask


def test_an_interchange_from_a_source_of_another_shape_is_refused() -> None:
    interchange = Interchange(a_rotation(), a_mask(2), a_source(items=3))

    with pytest.raises(ValueError, match="cannot stand in for"):
        interchange.stands_in_for(an_observation())


def test_an_interchange_source_of_another_width_is_refused() -> None:
    with pytest.raises(ValueError, match="innermost axis of a source"):
        Interchange(a_rotation(), a_mask(2), a_source(width=WIDTH + 1))


def test_an_interchange_source_that_is_not_a_floating_tensor_is_refused() -> None:
    with pytest.raises(TypeError, match="not a tensor"):
        Interchange(a_rotation(), a_mask(2), cast("torch.Tensor", [1.0]))

    with pytest.raises(TypeError, match="held in"):
        Interchange(a_rotation(), a_mask(2), torch.ones(4, WIDTH, dtype=torch.int64))

    with pytest.raises(ValueError, match=r"source is held in torch\.float64"):
        Interchange(a_rotation(), a_mask(2), a_source().to(torch.float64))


def test_an_interchange_refuses_a_reading_at_another_dtype() -> None:
    interchange = Interchange(a_rotation(), a_mask(2), a_source())
    observation = an_observation(tensor=a_source().to(torch.float64))

    with pytest.raises(ValueError, match="do not share a dtype"):
        interchange.stands_in_for(observation)


def test_an_interchange_requires_one_mask_share_per_coordinate() -> None:
    with pytest.raises(ValueError, match="mask of shape"):
        Interchange(a_rotation(), a_mask(width=WIDTH + 1), a_source())

    with pytest.raises(ValueError, match="mask of shape"):
        Interchange(a_rotation(), torch.ones(2, WIDTH), a_source())


def test_an_interchange_over_a_rotation_that_is_not_orthogonal_is_refused() -> None:
    with pytest.raises(ValueError, match="not orthogonal"):
        Interchange(torch.ones(WIDTH, WIDTH), a_mask(2), a_source())


def test_an_interchange_repr_names_the_side_mask_and_source_shapes() -> None:
    assert repr(Interchange(a_rotation(), a_mask(2), a_source())) == (
        f"Interchange(<{WIDTH}x{WIDTH}>, mask=<({WIDTH},)>, source=<(4, {WIDTH})>)"
    )


def test_an_interchange_refuses_a_malformed_mask() -> None:
    with pytest.raises(TypeError, match="mask is not a tensor"):
        Interchange(a_rotation(), cast("torch.Tensor", [1.0]), a_source())

    with pytest.raises(TypeError, match="mask is held"):
        Interchange(a_rotation(), torch.ones(WIDTH, dtype=torch.int64), a_source())

    with pytest.raises(ValueError, match="outside zero to one"):
        Interchange(a_rotation(), torch.full((WIDTH,), 1.1), a_source())

    nonfinite = a_mask()
    nonfinite[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        Interchange(a_rotation(), nonfinite, a_source())

    with pytest.raises(ValueError, match=r"rotation's torch\.float32"):
        Interchange(a_rotation(), a_mask().to(torch.float64), a_source())


def test_an_interchange_keeps_the_gradient_through_the_mask() -> None:
    mask = a_mask().requires_grad_()
    replacement = Interchange(a_rotation(), mask, a_source(seed=2)).stands_in_for(
        an_observation(tensor=a_source())
    )
    replacement.sum().backward()

    assert mask.grad is not None
    assert bool((mask.grad != 0.0).any())


def test_a_subspace_holds_its_rotation_detached() -> None:
    rotation = a_rotation().requires_grad_()

    assert not Subspace(rotation, 2, (0,)).rotation.requires_grad


def test_every_variant_reads_a_trace_only_once_that_trace_has_closed() -> None:
    open_trace = Trace()
    open_trace.record(an_observation())

    with pytest.raises(ValueError, match="still open"):
        MeanFrom(open_trace)

    with pytest.raises(ValueError, match="still open"):
        Resample(open_trace, 0)

    with pytest.raises(ValueError, match="still open"):
        PatchFrom(open_trace, CELL)


def test_a_trace_holding_variant_refuses_a_value_that_is_not_a_trace() -> None:
    with pytest.raises(TypeError, match="is not a Trace"):
        MeanFrom(cast("Trace", {}))


def test_a_patch_names_the_point_it_reads_with_a_site() -> None:
    trace = a_closed_trace(an_observation())

    with pytest.raises(TypeError, match="is not a Site"):
        PatchFrom(trace, cast("Site", "chart-cell"))


def test_a_negative_seed_is_refused_on_every_stochastic_variant() -> None:
    trace = a_closed_trace(an_observation())

    for build in (
        lambda: Resample(trace, -1),
        lambda: Noise(1.0, -1),
        lambda: Shuffle(-1),
    ):
        with pytest.raises(ValueError, match="seed"):
            build()


def test_every_variant_refuses_a_key_it_does_not_declare() -> None:
    trace = a_closed_trace(an_observation())

    for variant in VARIANTS:
        build = builder_for(variant)

        with pytest.raises(TypeError):
            build(*arguments_for(variant, 0, trace), bogus=1)


def test_no_variant_grows_a_field_after_it_is_built() -> None:
    zeroing = Zero()
    field = "bogus"

    with pytest.raises(AttributeError):
        setattr(zeroing, field, 1)


def test_the_union_names_twelve_variants_and_each_answers_to_one_tag() -> None:
    every = every_variant()

    assert get_args(Intervention.__value__) == VARIANTS
    assert len(VARIANTS) == 12
    assert len({variant.tag for variant in every}) == len(every)

    for variant in every:
        assert callable(variant.stands_in_for)


def test_the_variants_spell_the_twelve_kinds_of_replacement() -> None:
    assert {variant.tag for variant in every_variant()} == {
        "zero",
        "mean-from",
        "resample",
        "patch-from",
        "noise",
        "shuffle",
        "subspace",
        "interchange",
        "joint-interchange",
        "within",
        "sparing",
        "linear-erasure",
    }


def test_only_the_stochastic_variants_have_a_seed() -> None:
    seeded_tags = {variant.tag for variant in stochastic(0)}

    assert seeded_tags == {"resample", "noise", "shuffle"}
    assert not any(
        hasattr(variant, "seed")
        for variant in every_variant()
        if variant.tag not in seeded_tags
    )


def test_a_joint_interchange_applies_two_disjoint_masks_simultaneously() -> None:
    rotation = torch.eye(4)
    source = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    observed = an_observation(
        tensor=torch.tensor([[1.0, 2.0, 3.0, 4.0]]), axes=("item", "width")
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])

    forward = JointInterchange(rotation, first, second, source).stands_in_for(observed)
    reversed_masks = JointInterchange(rotation, second, first, source).stands_in_for(
        observed
    )

    assert torch.equal(forward, torch.tensor([[10.0, 20.0, 3.0, 4.0]]))
    assert torch.equal(reversed_masks, forward)


def test_a_joint_interchange_refuses_overlapping_masks() -> None:
    rotation = torch.eye(4)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="overlap"):
        JointInterchange(rotation, first, first, torch.ones(1, 4))


@settings(max_examples=40)
@given(
    which=st.integers(min_value=0, max_value=len(STOCHASTIC) - 1),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_one_seed_gives_bitwise_identical_applications(which: int, seed: int) -> None:
    observed = an_observation(tensor=a_source(64))

    first = stochastic(seed)[which].stands_in_for(observed)
    second = stochastic(seed)[which].stands_in_for(observed)

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "which",
    range(len(STOCHASTIC)),
    ids=[variant.tag for variant in STOCHASTIC],
)
def test_pinned_seeds_give_different_applications(which: int) -> None:
    observed = an_observation(tensor=a_source(64))

    first = stochastic(11)[which].stands_in_for(observed)
    second = stochastic(29)[which].stands_in_for(observed)

    assert not torch.equal(first, second)


def test_two_shuffle_seeds_may_agree_on_a_constant_reading() -> None:
    observed = an_observation(tensor=torch.ones(64, WIDTH))

    first = Shuffle(11).stands_in_for(observed)
    second = Shuffle(29).stands_in_for(observed)

    assert torch.equal(first, second)


@settings(max_examples=20)
@given(
    which=st.integers(min_value=0, max_value=len(STOCHASTIC) - 1),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_no_stochastic_variant_reads_or_moves_the_global_random_source(
    which: int, seed: int
) -> None:
    observed = an_observation(tensor=a_source(64))
    caller_state = torch.get_rng_state()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        before = torch.get_rng_state()

        applied = stochastic(seed)[which].stands_in_for(observed)
        after = torch.get_rng_state()

        torch.manual_seed(9999)
        under_another_global_seed = stochastic(seed)[which].stands_in_for(observed)

    assert torch.equal(before, after)
    assert torch.equal(applied, under_another_global_seed)
    assert torch.equal(torch.get_rng_state(), caller_state)


def test_shuffling_leaves_the_reading_of_the_size_it_was() -> None:
    observed = an_observation()

    replacement = Shuffle(3).stands_in_for(observed)

    assert replacement.norm().item() == pytest.approx(
        observed.tensor.norm().item(), abs=1e-6
    )


def test_noise_moves_a_reading_without_zeroing_any_of_it() -> None:
    numbers = torch.arange(1, 4 * WIDTH + 1, dtype=torch.float32).reshape(4, WIDTH)
    observed = an_observation(tensor=numbers)

    replacement = Noise(0.5, 11).stands_in_for(observed)

    assert not torch.equal(replacement, observed.tensor)
    assert bool((replacement != observed.tensor).all())
    assert bool((replacement != 0.0).all())
    assert replacement.norm().item() > 0.0


def a_column(width: int = WIDTH) -> Unit:
    return Unit(CELL, {"width": width - 1})


def a_row(item: int = 0) -> Unit:
    return Unit(CELL, {"item": item})


def test_within_replaces_exactly_the_entries_its_units_index() -> None:
    observed = an_observation()

    replacement = Within((a_row(1), a_column()), Zero()).stands_in_for(observed)

    expected = observed.tensor.clone()
    expected[1, :] = 0.0
    expected[:, WIDTH - 1] = 0.0

    assert torch.equal(replacement, expected)
    assert not torch.equal(replacement, observed.tensor)


def test_sparing_replaces_every_entry_its_units_do_not_index() -> None:
    observed = an_observation()

    replacement = Sparing((a_row(1), a_column()), Zero()).stands_in_for(observed)

    expected = torch.zeros_like(observed.tensor)
    expected[1, :] = observed.tensor[1, :]
    expected[:, WIDTH - 1] = observed.tensor[:, WIDTH - 1]

    assert torch.equal(replacement, expected)


def test_within_and_sparing_over_the_same_units_partition_the_reading() -> None:
    observed = an_observation()
    units = (a_row(2), Unit(CELL, {"item": 0, "width": 3}))

    inside = Within(units, Zero()).stands_in_for(observed)
    outside = Sparing(units, Zero()).stands_in_for(observed)

    assert torch.equal(inside + outside, observed.tensor)
    assert torch.equal(
        (inside == observed.tensor) & (observed.tensor != 0.0),
        (outside == 0.0) & (observed.tensor != 0.0),
    )


def test_a_unit_with_no_coordinate_covers_the_whole_point() -> None:
    observed = an_observation()

    assert torch.equal(
        Within((Unit(CELL),), Zero()).stands_in_for(observed),
        torch.zeros_like(observed.tensor),
    )
    assert torch.equal(
        Sparing((Unit(CELL),), Zero()).stands_in_for(observed), observed.tensor
    )


def test_within_no_units_changes_nothing_and_sparing_none_replaces_everything() -> None:
    observed = an_observation()

    assert torch.equal(Within((), Zero()).stands_in_for(observed), observed.tensor)
    assert torch.equal(
        Sparing((), Zero()).stands_in_for(observed), torch.zeros_like(observed.tensor)
    )


def test_a_scoped_variant_applies_its_inner_intervention_at_the_covered_entries() -> (
    None
):
    observed = an_observation()
    source = a_closed_trace(an_observation(tensor=torch.full((4, WIDTH), 7.0)))

    replacement = Within((a_row(3),), MeanFrom(source)).stands_in_for(observed)

    assert torch.equal(replacement[3], torch.full((WIDTH,), 7.0))
    assert torch.equal(replacement[:3], observed.tensor[:3])


def test_a_unit_at_another_point_indexes_nothing_here() -> None:
    observed = an_observation()

    with pytest.raises(ValueError, match="is not at"):
        Within((Unit(OTHER),), Zero()).stands_in_for(observed)


def test_a_unit_naming_an_axis_the_reading_lacks_is_refused() -> None:
    observed = an_observation()

    with pytest.raises(KeyError, match="head"):
        Sparing((Unit(CELL, {"head": 0}),), Zero()).stands_in_for(observed)


def test_a_coordinate_beyond_the_axis_is_refused() -> None:
    observed = an_observation()

    with pytest.raises(ValueError, match="positions"):
        Within((Unit(CELL, {"width": WIDTH}),), Zero()).stands_in_for(observed)


def test_a_scoped_variant_indexes_with_units_only() -> None:
    with pytest.raises(TypeError, match="units"):
        Within((cast("Unit", CELL),), Zero())

    with pytest.raises(TypeError, match="units"):
        Sparing((cast("Unit", "cell"),), Zero())


def test_a_scoped_variant_applies_a_declared_intervention_only() -> None:
    with pytest.raises(TypeError, match="declared"):
        Within((), cast("Intervention", torch.zeros))

    with pytest.raises(TypeError, match="declared"):
        Sparing((), cast("Intervention", None))


def test_a_scoped_variant_exposes_its_units_and_inner_intervention() -> None:
    zeroing = Zero()
    scoped = Within((a_row(), a_row()), zeroing)

    assert scoped.units == frozenset({a_row()})
    assert scoped.intervention is zeroing
    assert repr(scoped) == f"Within([{a_row()!r}], Zero())"
    assert repr(Sparing((), zeroing)) == "Sparing([], Zero())"


def test_a_scoped_variant_refuses_an_inner_replacement_of_another_shape() -> None:
    observed = an_observation()
    narrow = a_closed_trace(an_observation(tensor=torch.zeros(4, 1)))

    with pytest.raises(ValueError, match="cannot replace"):
        Within((a_row(),), PatchFrom(narrow, CELL)).stands_in_for(observed)

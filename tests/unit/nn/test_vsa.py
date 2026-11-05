from itertools import combinations, product
from math import sqrt
from typing import TYPE_CHECKING, Final

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from lofi.nn import vsa
from lofi.nn.vsa import (
    AtomTable,
    bind,
    bundle,
    bundle_arity,
    bundle_norm,
    bundle_scale,
    protect,
    superpose,
    unbind,
    unprotect,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.conftest import ChecksReload

DIMENSIONS: Final = st.integers(min_value=256, max_value=2048)
MAGNITUDES: Final = st.sampled_from((0.5, 1.0, 2.0))
SEEDS: Final = st.integers(min_value=0, max_value=2**16)

CHAIN: Final = 25
SIMILARITY_TOLERANCE: Final = 1e-3
MAGNITUDE_TOLERANCE: Final = 0.05
DEPTHS: Final = st.integers(min_value=0, max_value=64)
PRECISION_TOLERANCE: Final = 1e-6
ROLES: Final = ("argument", "result")


def _atoms(
    seed: int, *, count: int, dimension: int, magnitude: float = 1.0
) -> torch.Tensor:
    torch.manual_seed(seed)

    return AtomTable(
        count=count, dimension=dimension, magnitude=magnitude
    ).atoms.detach()


def _fillers(seed: int, *, count: int, dimension: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(count, dimension, generator=generator)


def _similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left, right, dim=-1))


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector, dim=-1))


@given(dimension=DIMENSIONS, seed=SEEDS)
def test_unbinding_a_key_returns_what_was_bound_to_it(
    dimension: int, seed: int
) -> None:
    key, filler = _atoms(seed, count=2, dimension=dimension)
    recovered = unbind(bind(filler, key), key)

    assert _similarity(recovered, filler) > 1.0 - SIMILARITY_TOLERANCE
    assert _norm(recovered) == pytest.approx(_norm(filler), rel=MAGNITUDE_TOLERANCE)


@given(dimension=DIMENSIONS, seed=SEEDS)
def test_binding_holds_over_a_chain_of_twenty_five(dimension: int, seed: int) -> None:
    table = _atoms(seed, count=CHAIN + 1, dimension=dimension)
    filler = table[0]

    for length in range(1, CHAIN + 1):
        keys = table[1 : length + 1]
        bound = filler

        for key in keys:
            bound = bind(bound, key)

        for key in reversed(tuple(keys)):
            bound = unbind(bound, key)

        assert _similarity(bound, filler) > 1.0 - SIMILARITY_TOLERANCE
        assert _norm(bound) == pytest.approx(_norm(filler), rel=MAGNITUDE_TOLERANCE)


@given(
    dimension=DIMENSIONS,
    arity=st.integers(min_value=1, max_value=8),
    magnitude=MAGNITUDES,
    seed=SEEDS,
)
def test_a_superposition_of_bound_pairs_sits_at_the_norm_predicted(
    dimension: int, arity: int, magnitude: float, seed: int
) -> None:
    table = _atoms(seed, count=2 * arity, dimension=dimension, magnitude=magnitude)
    pairs = [
        bind(table[position], table[arity + position]) for position in range(arity)
    ]
    predicted = bundle_norm(magnitude=magnitude, arity=arity, dimension=dimension)

    assert _norm(superpose(pairs)) == pytest.approx(predicted, rel=0.2)


@given(
    dimension=DIMENSIONS,
    arity=st.integers(min_value=1, max_value=8),
    magnitude=MAGNITUDES,
    seed=SEEDS,
)
def test_the_norm_reads_back_how_many_pairs_were_superposed(
    dimension: int, arity: int, magnitude: float, seed: int
) -> None:
    table = _atoms(seed, count=2 * arity, dimension=dimension, magnitude=magnitude)
    pairs = [
        bind(table[position], table[arity + position]) for position in range(arity)
    ]
    read = float(bundle_arity(superpose(pairs), magnitude=magnitude))

    assert read == pytest.approx(arity, rel=0.4)


def test_the_norm_readout_inverts_the_norm_law_exactly() -> None:
    dimension = 512
    magnitude = 0.75

    for arity in (1, 2, 3, 7):
        predicted = bundle_norm(magnitude=magnitude, arity=arity, dimension=dimension)
        vector = torch.full((dimension,), predicted / dimension**0.5)

        assert float(bundle_arity(vector, magnitude=magnitude)) == pytest.approx(arity)


def test_superposition_is_linear_and_distributes_over_binding() -> None:
    left, right, key = _fillers(5, count=3, dimension=256)

    assert torch.equal(superpose((left, right)), left + right)
    assert torch.allclose(
        bind(superpose((left, right)), key),
        superpose((bind(left, key), bind(right, key))),
        atol=1e-6,
    )


def test_a_bundle_divides_a_superposition_by_the_scale_its_arity_names() -> None:
    components = tuple(_fillers(6, count=3, dimension=64))

    assert bundle_scale(1) == 1.0
    assert bundle_scale(4) == 2.0
    assert torch.allclose(
        bundle(components), superpose(components) / bundle_scale(3), atol=0.0
    )


def test_a_bundle_holds_one_component_at_its_own_magnitude() -> None:
    (single,) = _fillers(7, count=1, dimension=512)

    assert _norm(bundle((single,))) == pytest.approx(_norm(single))


def test_cross_talk_stays_inside_the_bound_unequal_role_magnitudes_predict() -> None:
    dimension = 1024
    magnitudes = (0.25, 1.0, 4.0)
    roles = tuple(
        _atoms(11 + position, count=1, dimension=dimension, magnitude=magnitude)[0]
        for position, magnitude in enumerate(magnitudes)
    )
    fillers = tuple(_fillers(31, count=len(magnitudes), dimension=dimension))
    held = superpose(
        tuple(bind(role, filler) for role, filler in zip(roles, fillers, strict=True))
    )
    measured: list[float] = []

    for position, role in enumerate(roles):
        recovered = unbind(held, role)
        cross_talk = recovered - fillers[position]
        predicted = sum(
            (magnitudes[other] / magnitudes[position]) * _norm(fillers[other])
            for other in range(len(magnitudes))
            if other != position
        )

        assert _norm(cross_talk) <= predicted
        measured.append(_norm(cross_talk) / _norm(fillers[position]))

    assert measured[0] > measured[1] > measured[2]
    assert measured[2] < 1.0
    assert measured[0] > 1.0


def test_unbinding_favors_the_filler_whose_role_held_the_most_magnitude() -> None:
    dimension = 1024
    magnitudes = (0.25, 4.0)
    roles = tuple(
        _atoms(41 + position, count=1, dimension=dimension, magnitude=magnitude)[0]
        for position, magnitude in enumerate(magnitudes)
    )
    fillers = tuple(_fillers(43, count=2, dimension=dimension))
    held = superpose(
        tuple(bind(role, filler) for role, filler in zip(roles, fillers, strict=True))
    )
    weak = _similarity(unbind(held, roles[0]), fillers[0])
    strong = _similarity(unbind(held, roles[1]), fillers[1])

    assert strong > 0.9
    assert weak < strong


def test_an_atom_table_draws_every_coordinate_around_its_magnitude() -> None:
    dimension = 2048
    table = AtomTable(count=16, dimension=dimension, magnitude=2.0)
    atoms = table.atoms.detach()

    assert table.count == 16
    assert table.dimension == dimension
    assert table.magnitude == 2.0
    assert atoms.shape == (16, dimension)
    assert float(atoms.abs().mean()) == pytest.approx(2.0, abs=0.05)
    assert float(atoms.abs().sub(2.0).std()) == pytest.approx(
        2.0 / sqrt(dimension), rel=0.05
    )
    assert float(atoms.abs().min()) >= table.floor
    assert bool(atoms.sign().abs().eq(1).all())


def test_an_atom_table_holds_a_driven_down_coordinate_at_its_floor() -> None:
    table = AtomTable(count=3, dimension=64, magnitude=1.0, floor=0.125)
    optimizer = torch.optim.SGD(table.parameters(), lr=200.0)
    loss = table.atoms.abs().sum()

    loss.backward()
    optimizer.step()

    atoms = table.atoms.detach()

    assert float(atoms.abs().max()) == pytest.approx(table.floor)
    assert bool(atoms.abs().ge(table.floor).all())
    assert bool(atoms.sign().abs().eq(1).all())


def test_an_atom_table_reads_rows_by_index_and_keeps_its_signs_on_reload() -> None:
    torch.manual_seed(51)
    written = AtomTable(count=5, dimension=32)
    torch.manual_seed(52)
    read = AtomTable(count=5, dimension=32)
    indices = torch.tensor([4, 0, 0])

    assert not torch.equal(written.atoms, read.atoms)

    read.load_state_dict(written.state_dict())

    assert torch.equal(written.atoms, read.atoms)
    assert torch.equal(written(indices), written.atoms[indices])


def test_an_atom_table_passes_gradient_to_its_magnitudes() -> None:
    table = AtomTable(count=4, dimension=16)
    table.atoms.square().sum().backward()
    gradients = tuple(parameter.grad for _, parameter in table.named_parameters())

    assert gradients
    assert all(gradient is not None for gradient in gradients)


def test_an_atom_table_refuses_a_floor_outside_its_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude floor"):
        AtomTable(count=2, dimension=8, magnitude=1.0, floor=0.0)

    with pytest.raises(ValueError, match="magnitude floor"):
        AtomTable(count=2, dimension=8, magnitude=1.0, floor=2.0)

    with pytest.raises(ValueError, match="atom table's size"):
        AtomTable(count=0, dimension=8)

    with pytest.raises(ValueError, match="atom's dimension"):
        AtomTable(count=2, dimension=0)


def test_an_atom_table_reports_its_shape_and_its_knobs() -> None:
    assert "count=2" in repr(AtomTable(count=2, dimension=8, floor=0.5))
    assert "floor=0.5" in repr(AtomTable(count=2, dimension=8, floor=0.5))


def test_a_superposition_of_nothing_is_refused() -> None:
    empty: Sequence[torch.Tensor] = ()

    with pytest.raises(ValueError, match="at least one vector"):
        superpose(empty)

    with pytest.raises(ValueError, match="at least one vector"):
        bundle(empty)


def test_a_bundle_scale_is_refused_outside_the_positive_integers() -> None:
    with pytest.raises(ValueError, match="bundle's arity"):
        bundle_scale(0)

    with pytest.raises(ValueError, match="bundle's arity"):
        bundle_scale(-2)


def test_the_norm_law_is_refused_an_unusable_magnitude_or_shape() -> None:
    with pytest.raises(ValueError, match="atom's magnitude"):
        bundle_norm(magnitude=float("inf"), arity=1, dimension=8)

    with pytest.raises(ValueError, match="atom's dimension"):
        bundle_norm(magnitude=1.0, arity=1, dimension=0)

    with pytest.raises(ValueError, match="not a scalar"):
        bundle_arity(torch.tensor(1.0), magnitude=1.0)


def test_unbinding_refuses_a_key_holding_a_zero_coordinate(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(vsa, on=True)
    key = torch.tensor([1.0, 0.0, 2.0])
    bound = torch.ones(3)

    with pytest.raises(ValueError, match="no zero coordinate"):
        vsa.unbind(bound, key)


def test_unbinding_does_not_look_at_the_key_when_checks_are_off(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(vsa, on=False)
    divided = vsa.unbind(torch.ones(3), torch.tensor([1.0, 0.0, 2.0]))

    assert bool(divided.isinf().any())


def _relative_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return _norm(left - right) / _norm(left)


def _clamped_fraction(table: AtomTable) -> float:
    return float(table.atoms.detach().abs().le(table.floor).double().mean())


def _tree_code(
    roles: dict[str, torch.Tensor],
    path: tuple[str, ...],
    filler: torch.Tensor,
    *,
    protected: bool,
) -> torch.Tensor:
    """Bind one role per level of a path into a filler.

    Parameters
    ----------
    roles : dict[str, torch.Tensor]
        One vector per structural position, reused at every depth.
    path : tuple[str, ...]
        The positions from the root down to the filler.
    filler : torch.Tensor
        The vector sitting at the end of the path.
    protected : bool
        Whether each level's role is protected by the depth it sits at.
    """
    code = filler

    for depth, role in enumerate(path, start=1):
        key = protect(roles[role], depth) if protected else roles[role]
        code = bind(code, key)

    return code


@given(dimension=DIMENSIONS, depth=DEPTHS, seed=SEEDS)
def test_unprotecting_at_the_depth_a_vector_was_protected_at_returns_it(
    dimension: int, depth: int, seed: int
) -> None:
    (vector,) = _fillers(seed, count=1, dimension=dimension)

    assert torch.equal(unprotect(protect(vector, depth), depth), vector)


def test_protecting_reorders_the_coordinates_and_composes_along_its_depth() -> None:
    (vector,) = _fillers(89, count=1, dimension=64)
    once = protect(vector, 1)

    assert torch.equal(protect(vector, 0), vector)
    assert not torch.equal(once, vector)
    assert torch.equal(once.sort().values, vector.sort().values)
    assert torch.equal(protect(vector, 3), protect(protect(once, 1), 1))
    assert torch.equal(unprotect(once, 1), vector)


def test_protecting_runs_over_the_coordinate_axis_of_a_batch() -> None:
    batch = _fillers(93, count=4, dimension=32)
    shifted = protect(batch, 3)

    assert shifted.shape == batch.shape
    assert all(torch.equal(shifted[row], protect(batch[row], 3)) for row in range(4))


def test_protecting_refuses_a_scalar_or_a_negative_depth() -> None:
    with pytest.raises(ValueError, match="not over a scalar"):
        protect(torch.tensor(1.0), 1)

    with pytest.raises(ValueError, match="not over a scalar"):
        unprotect(torch.tensor(1.0), 1)

    with pytest.raises(ValueError, match="protection's depth"):
        protect(torch.ones(8), -1)

    with pytest.raises(ValueError, match="protection's depth"):
        unprotect(torch.ones(8), -1)

    with pytest.raises(TypeError, match="protection's depth"):
        protect(torch.ones(8), 1.5)  # ty: ignore[invalid-argument-type]


def test_the_published_surface_names_every_public_object_the_module_defines() -> None:
    defined = {
        name
        for name, value in vars(vsa).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == vsa.__name__
    }

    assert set(vsa.__all__) == defined


@given(dimension=DIMENSIONS, seed=SEEDS)
def test_binding_commutes(dimension: int, seed: int) -> None:
    left, right = _fillers(seed, count=2, dimension=dimension)

    assert torch.equal(bind(left, right), bind(right, left))


def test_a_protection_does_not_commute_with_binding() -> None:
    left, right = _fillers(0, count=2, dimension=1024)
    moved = bind(protect(left, 3), right)
    held = protect(bind(left, right), 3)

    assert _relative_difference(moved, held) > 1.0


def test_a_reused_role_tree_code_is_injective_only_once_its_levels_are_protected() -> (
    None
):
    dimension = 1024
    roles = dict(zip(ROLES, _atoms(81, count=2, dimension=dimension), strict=True))
    (filler,) = _fillers(83, count=1, dimension=dimension)
    paths = tuple(product(ROLES, repeat=2))
    reversed_path = (("argument", "result"), ("result", "argument"))

    plain = {path: _tree_code(roles, path, filler, protected=False) for path in paths}
    guarded = {path: _tree_code(roles, path, filler, protected=True) for path in paths}
    collided = tuple(plain[path] for path in reversed_path)
    separated = tuple(guarded[path] for path in reversed_path)

    assert _relative_difference(*collided) < PRECISION_TOLERANCE
    assert _similarity(*collided) > 1.0 - SIMILARITY_TOLERANCE
    assert len({tuple(code.tolist()) for code in guarded.values()}) == len(paths)
    assert abs(_similarity(*separated)) < 0.5

    for one, other in combinations(paths, 2):
        assert abs(_similarity(guarded[one], guarded[other])) < 0.5


def test_a_protected_tree_code_still_unbinds_back_to_its_filler() -> None:
    dimension = 1024
    roles = dict(zip(ROLES, _atoms(85, count=2, dimension=dimension), strict=True))
    (filler,) = _fillers(87, count=1, dimension=dimension)
    path = ("result", "argument")
    code = _tree_code(roles, path, filler, protected=True)
    recovered = code
    unguarded = code

    for depth, role in enumerate(path, start=1):
        recovered = unbind(recovered, protect(roles[role], depth))
        unguarded = unbind(unguarded, roles[role])

    assert _similarity(recovered, filler) > 1.0 - SIMILARITY_TOLERANCE
    assert abs(_similarity(unguarded, filler)) < 0.5


def test_an_atom_table_spreads_its_coordinates_relative_to_its_magnitude() -> None:
    for dimension in (256, 1024):
        torch.manual_seed(91)
        small = AtomTable(count=64, dimension=dimension, magnitude=1 / sqrt(dimension))
        torch.manual_seed(91)
        default = AtomTable(count=64, dimension=dimension)

        assert _clamped_fraction(small) <= _clamped_fraction(default)
        assert _clamped_fraction(small) == 0.0
        assert _clamped_fraction(default) == 0.0
        assert float(
            small.atoms.detach().abs().sub(small.magnitude).std()
        ) == pytest.approx(small.magnitude / sqrt(dimension), rel=0.05)

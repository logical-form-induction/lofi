import pytest
import torch

from lofi.probe._rotation import OrthogonalOptimizer


def an_optimizer(seed: int = 7) -> OrthogonalOptimizer:
    return OrthogonalOptimizer(
        torch.zeros(3, 4), 0.05, torch.Generator().manual_seed(seed)
    )


def test_the_shared_rotation_is_cayley_parameterized_orthogonal_and_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solves = 0
    solve = torch.linalg.solve

    def recorded(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        nonlocal solves
        solves += 1
        return solve(left, right)

    monkeypatch.setattr(torch.linalg, "solve", recorded)
    once = an_optimizer().matrix()
    again = an_optimizer().matrix()

    assert solves == 2
    assert torch.equal(once, again)
    assert torch.allclose(once.T @ once, torch.eye(4), atol=1e-6, rtol=0.0)


def test_a_step_moves_the_rotation_without_leaving_the_orthogonal_group() -> None:
    optimizer = an_optimizer()
    before = optimizer.matrix().detach().clone()
    loss = optimizer.matrix()[0, 0]
    (gradient,) = optimizer.gradients_of(loss)

    optimizer.step(gradient)
    after = optimizer.finish()

    assert not torch.equal(before, after)
    assert torch.allclose(after.T @ after, torch.eye(4), atol=1e-6, rtol=0.0)
    assert not after.requires_grad


def test_additional_parameters_share_one_autograd_traversal() -> None:
    optimizer = an_optimizer()
    additional = torch.ones(4, requires_grad=True)
    loss = (optimizer.matrix()[0] * additional).sum()

    rotation, other = optimizer.gradients_of(loss, (additional,))

    assert rotation.shape == (4, 4)
    assert other.shape == (4,)


def test_a_disconnected_objective_is_refused() -> None:
    optimizer = an_optimizer()
    disconnected = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(ValueError, match="do not depend on the rotation"):
        optimizer.gradients_of(disconnected)

from typing import TYPE_CHECKING, Final, NoReturn, cast, override

import pytest
import torch
from torch import nn

from lofi.probe._hooks import Hooks, module_named, parameter_named

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle

WIDTH: Final = 4
INNER: Final = 8
BATCH: Final = 2

DECLARED: Final = "a declared point"


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(WIDTH, INNER)
        self.second = nn.Linear(INNER, WIDTH)

    @override
    def forward(self, given: Tensor) -> Tensor:
        return self.second(torch.relu(self.first(given)))


class _Sealed(_Net):
    @override
    def named_modules(self, *args: object, **named: object) -> NoReturn:
        message = "this model does not enumerate its modules"

        raise AssertionError(message)

    @override
    def named_parameters(self, *args: object, **named: object) -> NoReturn:
        message = "this model does not enumerate its parameters"

        raise AssertionError(message)

    @override
    def state_dict(self, *args: object, **named: object) -> NoReturn:
        message = "this model does not enumerate its state"

        raise AssertionError(message)


class _Handle:
    def __init__(self, name: str, removed: list[str]) -> None:
        self._name = name
        self._removed = removed

    def remove(self) -> None:
        self._removed.append(self._name)


class _Ordered(nn.Module):
    def __init__(self, name: str, removed: list[str]) -> None:
        super().__init__()
        self.name = name
        self.removed = removed

    @override
    def register_forward_hook(
        self, hook: Callable[..., object], **named: object
    ) -> RemovableHandle:
        del hook, named

        return cast("RemovableHandle", _Handle(self.name, self.removed))


def an_input() -> Tensor:
    return torch.linspace(-1.0, 1.0, BATCH * WIDTH).reshape(BATCH, WIDTH)


def hooked(model: Module) -> int:
    return sum(len(module._forward_hooks) for _, module in model.named_modules())  # noqa: SLF001


def test_a_holder_that_has_installed_nothing_holds_no_hook() -> None:
    assert len(Hooks()) == 0


def test_a_module_hook_is_installed_and_called_on_a_pass() -> None:
    model = _Net()
    hooks = Hooks()
    seen: list[tuple[int, ...]] = []

    hooks.on_module(
        model.first,
        lambda _module, _args, output: seen.append(tuple(output.shape)),
    )
    model(an_input())

    assert len(hooks) == 1
    assert seen == [(BATCH, INNER)]


def test_a_parameter_hook_is_installed_and_called_on_a_backward_pass() -> None:
    model = _Net()
    hooks = Hooks()
    gradients: list[Tensor] = []

    hooks.on_parameter(
        parameter_named(DECLARED, model, "first.weight"),
        lambda gradient: gradients.append(gradient.detach().clone()),
    )
    model(an_input()).sum().backward()

    assert len(hooks) == 1
    assert len(gradients) == 1
    assert tuple(gradients[0].shape) == (INNER, WIDTH)


def test_removing_takes_every_hook_off_the_model() -> None:
    model = _Net()
    hooks = Hooks()

    hooks.on_module(model.first, lambda _module, _args, output: output)
    hooks.on_module(model.second, lambda _module, _args, output: output)

    assert hooked(model) == 2

    hooks.remove()

    assert hooked(model) == 0
    assert len(hooks) == 0


def test_removing_a_second_time_takes_nothing_further_off() -> None:
    removed: list[str] = []
    hooks = Hooks()

    hooks.on_module(_Ordered("only", removed), lambda _module, _args, output: output)
    hooks.remove()
    hooks.remove()

    assert removed == ["only"]


def test_hooks_come_off_in_the_reverse_of_the_order_they_went_on() -> None:
    removed: list[str] = []
    hooks = Hooks()

    for name in ("first", "second", "third"):
        hooks.on_module(_Ordered(name, removed), lambda _module, _args, output: output)

    hooks.remove()

    assert removed == ["third", "second", "first"]


def test_a_holder_reports_how_many_hooks_it_has_installed() -> None:
    model = _Net()
    hooks = Hooks()

    assert repr(hooks) == "Hooks(0 installed)"

    hooks.on_module(model.first, lambda _module, _args, output: output)

    assert repr(hooks) == "Hooks(1 installed)"


def test_a_module_path_the_model_does_not_hold_is_refused_loudly() -> None:
    with pytest.raises(ValueError, match=r"names 'third', and this model holds no"):
        module_named(DECLARED, _Net(), "third")


def test_a_module_path_naming_part_of_another_is_refused_rather_than_matched() -> None:
    with pytest.raises(ValueError, match="'fir'"):
        module_named(DECLARED, _Net(), "fir")


def test_a_module_path_running_through_a_parameter_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'first\.weight'"):
        module_named(DECLARED, _Net(), "first.weight")


def test_the_module_a_path_names_is_the_one_the_model_holds_there() -> None:
    model = _Net()

    assert module_named(DECLARED, model, "first") is model.first


def test_a_parameter_name_the_model_does_not_hold_is_refused_loudly() -> None:
    with pytest.raises(
        ValueError, match=r"names 'third\.weight', and this model holds no"
    ):
        parameter_named(DECLARED, _Net(), "third.weight")


def test_a_parameter_name_that_names_a_module_is_refused() -> None:
    with pytest.raises(ValueError, match="'first'"):
        parameter_named(DECLARED, _Net(), "first")


def test_a_parameter_that_requires_no_gradient_is_refused() -> None:
    model = _Net()
    model.first.weight.requires_grad_(False)  # noqa: FBT003

    with pytest.raises(ValueError, match="requires no gradient"):
        parameter_named(DECLARED, model, "first.weight")


def test_the_parameter_a_name_names_is_the_one_the_model_holds_there() -> None:
    model = _Net()

    assert parameter_named(DECLARED, model, "first.bias") is model.first.bias


def test_a_module_is_resolved_without_enumerating_what_the_model_holds() -> None:
    model = _Sealed()

    assert module_named(DECLARED, model, "second") is model.second

    with pytest.raises(ValueError, match="'third'"):
        module_named(DECLARED, model, "third")


def test_a_parameter_is_resolved_without_enumerating_what_the_model_holds() -> None:
    model = _Sealed()

    assert parameter_named(DECLARED, model, "second.bias") is model.second.bias

    with pytest.raises(ValueError, match=r"'third\.bias'"):
        parameter_named(DECLARED, model, "third.bias")

"""Resolution and removal of the hooks a region installs on a torch model."""

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.nn import Module, Parameter
    from torch.utils.hooks import RemovableHandle

__all__ = ["Hooks", "module_named", "parameter_named"]


def module_named(field: str, model: Module, path: str) -> Module:
    """Return the module a model holds at one dotted path.

    The path is followed one segment at a time from the model itself, so it is
    matched whole: a path naming part of another, or one running through something
    that is not a module, names no module here rather than the nearest thing to it.

    Parameters
    ----------
    field : str
        What names the path, for the message.
    model : Module
        The model to look in.
    path : str
        The dotted path to follow.

    Returns
    -------
    Module
        The module found there.

    Raises
    ------
    ValueError
        If the model holds no module at the path.
    """
    try:
        return model.get_submodule(path)

    except AttributeError as absent:
        msg = f"{field} names {path!r}, and this model holds no module there"
        raise ValueError(msg) from absent


def parameter_named(field: str, model: Module, name: str) -> Parameter:
    """Return the parameter a model holds under one dotted name.

    Parameters
    ----------
    field : str
        What names the parameter, for the message.
    model : Module
        The model to look in.
    name : str
        The dotted name to follow.

    Returns
    -------
    Parameter
        The parameter found there.

    Raises
    ------
    ValueError
        If the model holds no parameter under the name, or the one it holds requires
        no gradient, since nothing would ever reach a hook on it.
    """
    try:
        found = model.get_parameter(name)

    except AttributeError as absent:
        msg = f"{field} names {name!r}, and this model holds no parameter there"
        raise ValueError(msg) from absent

    if not found.requires_grad:
        msg = (
            f"{field} names {name!r}, which requires no gradient, so no gradient "
            "reaches a hook there"
        )
        raise ValueError(msg)

    return found


class Hooks:
    """Holds the hooks one region installed on a model, and takes them off again.

    A hook goes on through :meth:`on_module` or :meth:`on_parameter`, each of which
    keeps what removes it. :meth:`remove` takes every hook off in the reverse of the
    order it went on and empties the holder, so a second removal takes nothing off.
    """

    __slots__ = ("_handles",)

    def __init__(self) -> None:
        self._handles: list[RemovableHandle] = []

    def __len__(self) -> int:
        """Return how many hooks are installed."""
        return len(self._handles)

    def on_module(self, module: Module, hook: Callable[..., object], /) -> None:
        """Install one forward hook on a module and keep what removes it.

        Parameters
        ----------
        module : Module
            The module to install it on.
        hook : Callable[..., object]
            What runs when the pass leaves the module, taking the module, its
            arguments and its output, and returning what the pass carries on with
            or None to carry on with the output itself.
        """
        self._handles.append(module.register_forward_hook(hook))

    def on_parameter(
        self, parameter: Parameter, hook: Callable[[Tensor], Tensor | None], /
    ) -> None:
        """Install one gradient hook on a parameter and keep what removes it.

        Parameters
        ----------
        parameter : Parameter
            The parameter to install it on.
        hook : Callable[[Tensor], Tensor | None]
            What runs when a gradient reaches the parameter, taking that gradient
            and returning what stands in its place or None to leave it alone.
        """
        self._handles.append(parameter.register_hook(hook))

    def remove(self) -> None:
        """Take every hook off, in the reverse of the order it went on."""
        for handle in reversed(self._handles):
            handle.remove()

        self._handles.clear()

    @override
    def __repr__(self) -> str:
        """Return how many hooks are installed."""
        return f"Hooks({len(self._handles)} installed)"

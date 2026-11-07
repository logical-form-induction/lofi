import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType


class Poison:
    def __init__(self, names: Sequence[str]) -> None:
        self._names = frozenset(names)

    def find_spec(
        self,
        fullname: str,
        _path: Sequence[str] | None = None,
        _target: ModuleType | None = None,
        /,
    ) -> ModuleSpec | None:
        """Reject configured imports and decline all other modules.

        Parameters
        ----------
        fullname
            The module being imported.
        _path
            The parent package's search path.
        _target
            The module replaced by a reload, if any.

        Returns
        -------
        importlib.machinery.ModuleSpec or None
            ``None`` for every module not rejected.

        Raises
        ------
        ModuleNotFoundError
            If the module or its top-level package is rejected.
        """
        if fullname.partition(".")[0] in self._names:
            message = f"import of {fullname} refused"
            raise ModuleNotFoundError(message, name=fullname)

        return None


def poison(names: Sequence[str]) -> None:
    """Remove named libraries from the module table and reject re-imports.

    Parameters
    ----------
    names
        Top-level library names to reject.
    """
    for name in names:
        prefix = f"{name}."
        loaded = [
            module
            for module in sys.modules
            if module == name or module.startswith(prefix)
        ]

        for module in loaded:
            del sys.modules[module]

    sys.meta_path.insert(0, Poison(names))


def reraise(name: str) -> None:
    """Re-raise the import failure recorded for a submodule.

    Parameters
    ----------
    name
        The module whose import failed.

    Raises
    ------
    ImportError
        The recorded failure, or a replacement if none was recorded.
    """
    failure = sys.exception()

    if failure is None:
        message = f"import of {name} failed and the interpreter recorded no error"
        raise ImportError(message, name=name)

    raise failure


def main() -> None:
    package, *refused = sys.stdin.read().split()

    poison(refused)

    root = importlib.import_module(package)

    for found in pkgutil.walk_packages(root.__path__, f"{package}.", reraise):
        importlib.import_module(found.name)


if __name__ == "__main__":
    main()

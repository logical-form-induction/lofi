import importlib
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol
from unittest.mock import patch

import didactic.api as dx
import pytest
import torch
from hypothesis import Verbosity, settings

import lofi
from lofi import _checks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

pytest_plugins = ["pytester"]

# the everyday run is checked and a benchmark run is not, since a checked build
# times the checks rather than the operation. the default is read off the
# command line rather than the parsed options because the initial conftests
# import lofi, and the flag binds at import
os.environ.setdefault("LOFI_CHECK", "0" if "--benchmark-only" in sys.argv else "1")

# every Python process a test starts reads and writes UTF-8, whatever the
# platform's locale encoding, so its text output decodes the same everywhere.
# the variable is read at interpreter start, so it reaches children only
os.environ["PYTHONUTF8"] = "1"

# reload the import-time check flag after setting its default, then the package
# that publishes its law error, so both name one class
importlib.reload(_checks)
importlib.reload(lofi)

# development varies examples, while CI uses reproducible examples
settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None, derandomize=True)
settings.register_profile(
    "debug", max_examples=25, deadline=None, verbosity=Verbosity.verbose
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

# marker, command-line flag, and cost for each opt-in test family
_OPT_IN: Final = (
    ("integration", "--run-integration", "real IO, optional deps or credentials"),
    ("slow", "--run-slow", "long-running"),
    ("generality", "--run-generality", "trains on large generated declarations"),
)


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker, flag, cost in _OPT_IN:
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run tests marked {marker} ({cost})",
        )

    parser.addini(
        "wall_clock_budgets",
        "hold each benchmark to its absolute wall-clock budget, which only a "
        "fixed machine can meet; a relative comparison turns this off",
        type="bool",
        default=True,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Anchor the benchmark store to the repository, not the caller's directory."""
    declared = config.getoption("benchmark_storage", default=None)

    if not isinstance(declared, str):
        return

    relative = declared.removeprefix("file://")

    if Path(relative).is_absolute():
        return

    config.option.benchmark_storage = str(config.rootpath / relative)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    for marker, flag, _cost in _OPT_IN:
        if config.getoption(flag):
            continue

        skip = pytest.mark.skip(reason=f"needs {flag}")

        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


class ChecksReload(Protocol):
    def __call__(self, *modules: ModuleType, on: bool) -> None:
        """Reload modules with debug checks enabled or disabled.

        Parameters
        ----------
        modules
            Modules that bind the flag during import, in reload order.
        on
            Whether reloaded modules enable debug checks.
        """
        ...


def _module_snapshots[Value](
    module: ModuleType,
    snapshot: dict[str, Value],
    /,
) -> dict[ModuleType, dict[str, Value]]:
    return {module: snapshot}


@pytest.fixture
def reload_with_checks(monkeypatch: pytest.MonkeyPatch) -> Iterator[ChecksReload]:
    """Provide a function that reloads modules under a debug-check setting.

    Yields
    ------
    ChecksReload
        The reload function.
    """
    snapshots = _module_snapshots(_checks, dict(vars(_checks)))

    def reload(*modules: ModuleType, on: bool) -> None:
        monkeypatch.setenv("LOFI_CHECK", "1" if on else "0")

        for module in modules:
            if module not in snapshots:
                snapshots[module] = dict(vars(module))

        importlib.reload(_checks)

        for module in modules:
            importlib.reload(module)

    yield reload

    monkeypatch.undo()

    for module, snapshot in snapshots.items():
        module.__dict__.clear()
        module.__dict__.update(snapshot)


@pytest.fixture(scope="session")
def ccgbank_root() -> Path:
    """Return the configured CCGBank root or skip the test session.

    Returns
    -------
    pathlib.Path
        The directory named by ``LOFI_CCGBANK``.
    """
    named = os.environ.get("LOFI_CCGBANK")

    if named is None:
        pytest.skip("set LOFI_CCGBANK to a CCGBank root to run this test")

    root = Path(named)

    if not root.is_dir():
        pytest.skip(f"LOFI_CCGBANK names {root}, which is not a directory")

    return root


@pytest.fixture(autouse=True)
def restore_global_torch_state() -> Iterator[None]:
    """Return torch's process-wide settings to what the test found them.

    The device policy sets determinism and matmul precision for the process, as
    a run's declaration asks. A test that resolves one therefore leaves those
    settings behind for every later test, and a deterministic algorithm the
    accelerator has no kernel for then fails a benchmark that ran alone.

    Yields
    ------
    None
        Once, with the settings restored afterwards.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()

    yield

    torch.use_deterministic_algorithms(deterministic)
    torch.set_float32_matmul_precision(precision)


@pytest.fixture(scope="session", autouse=True)
def restore_sigint_disposition() -> Iterator[None]:
    """Check that the session ends with the interrupt disposition it began with.

    A session launched in the background inherits an ignored interrupt, so the
    disposition the session found, not the interpreter default, is the one every
    test has to leave in place.

    Yields
    ------
    None
        Once, before the final disposition is checked and restored.
    """
    found = signal.getsignal(signal.SIGINT)

    yield

    disposition = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, found)

    assert disposition is found


class ConstructionCount:
    """The number of didactic models constructed while a count was open."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


def _counting[**P](
    construct: Callable[P, None], counter: ConstructionCount
) -> Callable[P, None]:
    """Return a constructor that counts each call before delegating."""

    def counted(*args: P.args, **kwargs: P.kwargs) -> None:
        counter.count += 1
        construct(*args, **kwargs)

    return counted


@contextmanager
def counted_constructions() -> Iterator[ConstructionCount]:
    """Count every didactic model constructed inside the block.

    Yields
    ------
    ConstructionCount
        The running count, final once the block exits.
    """
    counter = ConstructionCount()

    with patch.object(dx.Model, "__init__", _counting(dx.Model.__init__, counter)):
        yield counter

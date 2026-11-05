"""Metric-sink conformance laws."""

import ast
from typing import TYPE_CHECKING, Final, NamedTuple

from lofi._checks import fail_law

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.run._ports import MetricSink

__all__ = [
    "METRIC_SINK_LAWS",
    "MetricSinkFixture",
    "metric_keys_are_supplied_as_data",
    "telemetry_backends_are_confined_to_run",
    "unit_tests_do_not_import_telemetry_backends",
]


class MetricSinkFixture(NamedTuple):
    """Configures metric-sink conformance laws."""

    sink: MetricSink
    root: Path
    backend_modules: frozenset[str]
    registry_metric_keys: tuple[str, ...]


def _imported_root(node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    """Return top-level modules imported by one syntax node."""
    if isinstance(node, ast.Import):
        return tuple(alias.name.partition(".")[0] for alias in node.names)

    if node.level or node.module is None:
        return ()

    return (node.module.partition(".")[0],)


def _importlib_aliases(tree: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    """Return aliases for the importlib module and import function."""
    modules: set[str] = set()
    functions: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "importlib"
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "importlib"
        ):
            functions.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "import_module"
            )

    return frozenset(modules), frozenset(functions)


def _dynamic_import(
    node: ast.Call,
    *,
    importlib_modules: frozenset[str],
    import_module_functions: frozenset[str],
) -> str | None:
    """Return the literal module loaded by a dynamic import call."""
    if not node.args:
        return None

    argument = node.args[0]

    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None

    function = node.func
    direct = isinstance(function, ast.Name) and (
        function.id == "__import__" or function.id in import_module_functions
    )
    qualified = (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in importlib_modules
        and function.attr == "import_module"
    )

    if direct or qualified:
        return argument.value.partition(".")[0]

    return None


def _backend_imports(
    directory: Path, backend_modules: frozenset[str]
) -> tuple[Path, ...]:
    """Return Python files importing a named telemetry backend."""
    found: list[Path] = []

    if not directory.exists():
        return ()

    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        importlib_modules, import_module_functions = _importlib_aliases(tree)
        imported = {
            module
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for module in _imported_root(node)
        }
        imported.update(
            module
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            if (
                module := _dynamic_import(
                    node,
                    importlib_modules=importlib_modules,
                    import_module_functions=import_module_functions,
                )
            )
            is not None
        )

        if imported & backend_modules:
            found.append(path)

    return tuple(found)


def _relative(paths: tuple[Path, ...], root: Path) -> tuple[str, ...]:
    """Return paths relative to the fixture root."""
    return tuple(path.relative_to(root).as_posix() for path in paths)


def unit_tests_do_not_import_telemetry_backends(fixture: MetricSinkFixture) -> None:
    """Check that unit tests run without telemetry packages."""
    law = "unit_tests_do_not_import_telemetry_backends"
    fixture.sink.log_table("empty", ())
    imported = _backend_imports(
        fixture.root / "tests" / "unit", fixture.backend_modules
    )

    if imported:
        files = _relative(imported, fixture.root)
        fail_law(
            f"unit tests import telemetry backends in {files!r}",
            law=law,
        )


def telemetry_backends_are_confined_to_run(fixture: MetricSinkFixture) -> None:
    """Check that lower layers import no telemetry backend."""
    law = "telemetry_backends_are_confined_to_run"
    fixture.sink.close()
    imported = tuple(
        path
        for path in _backend_imports(fixture.root / "lofi", fixture.backend_modules)
        if path.relative_to(fixture.root / "lofi").parts[0] != "run"
    )

    if imported:
        files = _relative(imported, fixture.root)
        fail_law(
            f"lower layers import telemetry backends in {files!r}",
            law=law,
        )


def metric_keys_are_supplied_as_data(fixture: MetricSinkFixture) -> None:
    """Check that registry metric keys enter through scalar data."""
    law = "metric_keys_are_supplied_as_data"
    keys = fixture.registry_metric_keys

    if not keys:
        fail_law("the loss registry supplied no metric keys", law=law)

    if len(keys) != len(set(keys)) or any(not key for key in keys):
        fail_law(f"the loss registry supplied invalid metric keys {keys!r}", law=law)

    values = {key: float(index) for index, key in enumerate(keys, start=1)}
    fixture.sink.log_scalars(values, 3)


METRIC_SINK_LAWS: Final = (
    unit_tests_do_not_import_telemetry_backends,
    telemetry_backends_are_confined_to_run,
    metric_keys_are_supplied_as_data,
)
"""Metric-sink laws in conformance order."""

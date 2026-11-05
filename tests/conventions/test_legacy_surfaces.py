"""The package keeps one current spelling of everything, and no legacy surface.

Each rule reads the source structurally:

- a shim is a module, other than a package or a ``__main__`` module, that
  defines nothing and only imports;
- a deprecation path warns with a deprecation category, carries a
  ``deprecated`` decorator, declares a didactic field ``deprecated`` or with an
  ``alias``, or resolves module attributes through a module ``__getattr__``;
- an import fallback catches a failed import and carries on instead of
  raising;
- an alias binds a public name to another name the package defines, or to an
  attribute of a class it defines, or re-exports an import under a new name;
- a duplicated public name is an object with more than one package entry-point
  export, or with a public-module re-export outside its defining module.

Every finding fails the suite: an unreleased package has no compatibility
surface to preserve.
"""

import ast
from typing import TYPE_CHECKING, Final

import pytest
from tests.structure import (
    TREE,
    Finding,
    Package,
    Symbol,
    defined_names,
    exported,
    public,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

DEPRECATION_CATEGORIES: Final = frozenset(
    {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}
)
"""The built-in warning categories that announce a deprecation."""

IMPORT_FAILURES: Final = frozenset({"ImportError", "ModuleNotFoundError"})
"""The built-in exceptions a failed import raises."""


def _scoped(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Yield every node of a module with the dotted scope holding it."""
    pending: list[tuple[str, ast.AST]] = [("<module>", tree)]

    while pending:
        scope, node = pending.pop()
        yield scope, node

        for child in ast.iter_child_nodes(node):
            inner = scope

            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                inner = child.name if scope == "<module>" else f"{scope}.{child.name}"

            pending.append((inner, child))


def shim_modules(package: Package) -> list[Finding]:
    """Return the modules that define nothing and only import.

    Parameters
    ----------
    package : Package
        The source package.
    """
    found: list[Finding] = []

    for module, path in package.modules.items():
        if path.name in {"__init__.py", "__main__.py"}:
            continue

        tree = package.tree(module)
        imports = [
            statement
            for statement in tree.body
            if isinstance(statement, ast.Import | ast.ImportFrom)
        ]

        if imports and set(defined_names(tree)) <= {"__all__"}:
            found.append(
                Finding(
                    module,
                    f"{package.relative(module)}:1",
                    "defines nothing and only re-exports what it imports",
                )
            )

    return found


def deprecation_paths(package: Package) -> list[Finding]:
    """Return the warnings, markers and hooks that keep a deprecated spelling alive.

    Parameters
    ----------
    package : Package
        The source package.
    """
    found: list[Finding] = []

    for module in package.modules:
        tree = package.tree(module)

        def report(
            node: ast.stmt | ast.expr, scope: str, what: str, module: str = module
        ) -> None:
            found.append(
                Finding(
                    f"{module}:{scope}:{what}",
                    f"{package.relative(module)}:{node.lineno}",
                    f"{what} in {scope}",
                )
            )

        for scope, node in _scoped(tree):
            if isinstance(node, ast.Call):
                arguments = [*node.args, *(keyword.value for keyword in node.keywords)]

                if any(
                    isinstance(argument, ast.Name)
                    and argument.id in DEPRECATION_CATEGORIES
                    for argument in arguments
                ):
                    report(node, scope, "warns of a deprecation")

                target = package.reference(module, node.func)

                if (
                    target is not None
                    and target.module.startswith("didactic")
                    and target.name == "field"
                ):
                    named = {keyword.arg for keyword in node.keywords}

                    if "deprecated" in named:
                        report(node, scope, "declares a deprecated field")

                    if "alias" in named:
                        report(node, scope, "declares a field alias")

            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for decorator in node.decorator_list:
                    marker = (
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                    target = package.reference(module, marker)

                    if target is not None and target.name == "deprecated":
                        report(node, scope, "is marked deprecated")

        for statement in tree.body:
            if (
                isinstance(statement, ast.FunctionDef)
                and statement.name == "__getattr__"
            ):
                report(statement, "<module>", "resolves module attributes dynamically")

    return found


def import_fallbacks(package: Package) -> list[Finding]:
    """Return the handlers that catch a failed import and carry on.

    Parameters
    ----------
    package : Package
        The source package.
    """
    found: list[Finding] = []

    for module in package.modules:
        for scope, node in _scoped(package.tree(module)):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue

            caught = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            names = {item.id for item in caught if isinstance(item, ast.Name)}
            raises = any(isinstance(inner, ast.Raise) for inner in ast.walk(node))

            if names & IMPORT_FAILURES and not raises:
                found.append(
                    Finding(
                        f"{module}:{scope}",
                        f"{package.relative(module)}:{node.lineno}",
                        f"falls back when an import fails in {scope}",
                    )
                )

    return found


def _aliased(package: Package, module: str, value: ast.expr) -> Symbol | None:
    """Return the package symbol a bound value merely renames, if it is one."""
    if not isinstance(value, ast.Name | ast.Attribute):
        return None

    target = package.reference(module, value)

    if target is None or not target.name or target.module not in package.modules:
        return None

    defined = defined_names(package.tree(target.module)).get(target.name)

    if defined is None:
        return None

    is_class = isinstance(defined, ast.ClassDef)
    attribute = isinstance(value, ast.Attribute) and is_class

    if attribute or not target.name.startswith("_"):
        return target

    return None


def aliases(package: Package) -> list[Finding]:
    """Return the public names bound to another spelling of a package name.

    Parameters
    ----------
    package : Package
        The source package.
    """
    found: list[Finding] = []

    for module in package.modules:
        tree = package.tree(module)
        listed = set(exported(tree))

        def report(node: ast.stmt, key: str, what: str, module: str = module) -> None:
            found.append(
                Finding(
                    f"{module}:{key}",
                    f"{package.relative(module)}:{node.lineno}",
                    what,
                )
            )

        for statement in tree.body:
            match statement:
                case (
                    ast.Assign(targets=[ast.Name(id=name)], value=value)
                    | ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value)
                ) if not name.startswith("_"):
                    target = _aliased(package, module, value)

                    if target is not None:
                        report(
                            statement,
                            name,
                            f"{name} is another name for {target.describe()}",
                        )
                case ast.ImportFrom(names=names):
                    for alias in names:
                        if (
                            alias.asname
                            and alias.asname != alias.name
                            and (alias.asname in listed)
                        ):
                            report(
                                statement,
                                alias.asname,
                                f"re-exports {alias.name} as {alias.asname}",
                            )
                case ast.ClassDef(name=owner, body=body):
                    for inner in body:
                        if (
                            isinstance(inner, ast.Assign)
                            and len(inner.targets) == 1
                            and isinstance(inner.targets[0], ast.Name)
                            and not inner.targets[0].id.startswith("_")
                        ):
                            name = inner.targets[0].id
                            methods = {
                                method.name
                                for method in body
                                if isinstance(method, ast.FunctionDef)
                            }
                            renamed = (
                                isinstance(inner.value, ast.Name)
                                and inner.value.id in methods
                            )
                            target = _aliased(package, module, inner.value)

                            if renamed or target is not None:
                                spelled = ast.unparse(inner.value)
                                report(
                                    inner,
                                    f"{owner}.{name}",
                                    f"{owner}.{name} is another name for {spelled}",
                                )
                case _:
                    pass

    return found


def public_spellings(package: Package) -> dict[Symbol, list[str]]:
    """Return every object the public modules export, with the modules exporting it.

    Parameters
    ----------
    package : Package
        The source package.

    Returns
    -------
    dict[Symbol, list[str]]
        Each exported object, by its defining module and name, mapped to the
        public modules whose ``__all__`` lists it, as ``module.name``.
    """
    spellings: dict[Symbol, list[str]] = {}

    for module in sorted(package.modules):
        if not public(module):
            continue

        for name in exported(package.tree(module)):
            target = package.resolve(Symbol(module, name))

            if target.name and target.module in package.modules:
                spellings.setdefault(target, []).append(f"{module}.{name}")

    return spellings


def duplicate_public_names(package: Package) -> list[Finding]:
    """Return the objects with more than one package API home.

    Parameters
    ----------
    package : Package
        The source package.
    """
    found: list[Finding] = []

    for target, spelled in sorted(public_spellings(package).items()):
        defining = f"{target.module}.{target.name}"
        routes = [name for name in spelled if name != defining]
        package_routes = [
            name
            for name in routes
            if package.modules[name.rsplit(".", 1)[0]].name == "__init__.py"
        ]
        module_routes = [name for name in routes if name not in package_routes]
        defining_is_public = public(target.module)
        duplicated = (
            len(package_routes) > 1
            or (defining_is_public and bool(module_routes))
            or (not defining_is_public and len(spelled) > 1)
        )

        if not duplicated:
            continue

        statement = defined_names(package.tree(target.module))[target.name]
        found.append(
            Finding(
                target.describe(),
                f"{package.relative(target.module)}:{statement.lineno}",
                f"{target.describe()} is public as {', '.join(spelled)}",
            )
        )

    return found


RULES: Final[Mapping[str, Callable[[Package], list[Finding]]]] = {
    "shim-modules": shim_modules,
    "deprecation-paths": deprecation_paths,
    "import-fallbacks": import_fallbacks,
    "aliases": aliases,
    "duplicate-public-names": duplicate_public_names,
}

PACKAGE: Final = Package(TREE)


@pytest.mark.parametrize("rule", sorted(RULES))
def test_no_legacy_surface(rule: str) -> None:
    findings = RULES[rule](PACKAGE)

    assert not findings, f"{rule}:\n" + "\n".join(
        finding.describe() for finding in findings
    )


def write_tree(root: Path, files: Mapping[str, str]) -> None:
    """Write a miniature repository from relative paths and their text."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


BASE: Final = {
    "pyproject.toml": '[project]\nname = "lofi"\n',
    "lofi/__init__.py": "",
    "lofi/core/__init__.py": (
        "from lofi.core._values import Term, build\n__all__ = ['Term', 'build']\n"
    ),
    "lofi/core/_values.py": (
        "class Term:\n    pass\n\ndef build():\n    return Term()\n"
    ),
}


def planted(tmp_path: Path, files: Mapping[str, str]) -> Package:
    """Return the package of a miniature tree with some files added."""
    write_tree(tmp_path, {**BASE, **files})

    return Package(tmp_path)


def test_the_miniature_tree_has_no_legacy_surface(tmp_path: Path) -> None:
    package = planted(tmp_path, {})

    assert [finding for rule in RULES.values() for finding in rule(package)] == []


def test_a_definition_module_and_one_package_entry_point_are_one_api(
    tmp_path: Path,
) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/__init__.py": (
                "from lofi.core.values import Term\n__all__ = ['Term']\n"
            ),
            "lofi/core/values.py": "class Term:\n    pass\n\n__all__ = ['Term']\n",
        },
    )

    assert duplicate_public_names(package) == []


@pytest.mark.parametrize(
    ("rule", "files", "keys"),
    [
        (
            "shim-modules",
            {
                "lofi/core/old.py": (
                    "from lofi.core._values import Term\n__all__ = ['Term']\n"
                )
            },
            ["lofi.core.old"],
        ),
        (
            "deprecation-paths",
            {
                "lofi/core/old.py": (
                    "import warnings\n"
                    "def build():\n"
                    "    warnings.warn('use lofi.core.build', DeprecationWarning)\n"
                )
            },
            ["lofi.core.old:build:warns of a deprecation"],
        ),
        (
            "deprecation-paths",
            {
                "lofi/core/old.py": (
                    "from warnings import deprecated\n"
                    "@deprecated('use build')\ndef make():\n    return 1\n"
                )
            },
            ["lofi.core.old:make:is marked deprecated"],
        ),
        (
            "deprecation-paths",
            {
                "lofi/core/spec.py": (
                    "import didactic.api as dx\n"
                    "class Spec(dx.Model):\n"
                    "    width: int = dx.field(default=1, alias='size')\n"
                )
            },
            ["lofi.core.spec:Spec:declares a field alias"],
        ),
        (
            "deprecation-paths",
            {"lofi/core/old.py": ("def __getattr__(name):\n    return name\n")},
            ["lofi.core.old:<module>:resolves module attributes dynamically"],
        ),
        (
            "import-fallbacks",
            {
                "lofi/core/fast.py": (
                    "try:\n    import fastterm as engine\n"
                    "except ImportError:\n    engine = None\n"
                )
            },
            ["lofi.core.fast:<module>"],
        ),
        (
            "aliases",
            {
                "lofi/core/_values.py": (
                    BASE["lofi/core/_values.py"] + "Expression = Term\n"
                )
            },
            ["lofi.core._values:Expression"],
        ),
        (
            "aliases",
            {
                "lofi/core/__init__.py": (
                    "from lofi.core._values import Term as Expression, build\n"
                    "__all__ = ['Expression', 'build']\n"
                )
            },
            ["lofi.core:Expression"],
        ),
        (
            "aliases",
            {
                "lofi/core/_values.py": (
                    "class Term:\n    def build(self):\n        return self\n"
                    "    make = build\n\n"
                    "def build():\n    return Term()\n"
                )
            },
            ["lofi.core._values:Term.make"],
        ),
        (
            "duplicate-public-names",
            {
                "lofi/other/__init__.py": (
                    "from lofi.core import Term\n__all__ = ['Term']\n"
                )
            },
            ["lofi.core._values:Term"],
        ),
    ],
)
def test_each_planted_legacy_surface_is_reported(
    tmp_path: Path, rule: str, files: Mapping[str, str], keys: list[str]
) -> None:
    package = planted(tmp_path, files)

    assert [finding.key for finding in RULES[rule](package)] == keys


def test_an_import_failure_turned_into_an_error_is_not_a_fallback(
    tmp_path: Path,
) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/backend.py": (
                "def load():\n    try:\n        import fastterm\n"
                "    except ModuleNotFoundError as missing:\n"
                "        raise RuntimeError('install fastterm') from missing\n"
                "    return fastterm\n"
            )
        },
    )

    assert import_fallbacks(package) == []

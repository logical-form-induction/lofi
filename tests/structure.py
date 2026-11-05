"""The source package's structure, read from its files.

Every function here reads a repository tree: the modules under ``lofi/``, the
import statements that join them, the names each module binds, the entry
points ``pyproject.toml`` declares, and the didactic declaration classes and
the field annotations that join those. Nothing is imported or executed, so a
miniature tree written by a test is read exactly as the checkout is.

A module is reachable when the package exposes it through a generated API
reference, or when a console script or plugin entry point loads it directly
or through the modules those import. A configuration document reaches code
through those execution routes: the command line composes it, and each
``kind`` it names is a variant a plugin entry point registers.
"""

import ast
import re
import tomllib
from functools import cache
from pathlib import Path
from typing import Final, NamedTuple

TREE: Final = Path(__file__).resolve().parents[1]
"""The repository checkout the tests read."""

DECLARATION_BASES: Final = frozenset({"Model", "TaggedUnion"})
"""The didactic classes a declaration class derives from."""

_API_DIRECTIVE: Final = re.compile(
    r"^:::\s+(lofi(?:\.[A-Za-z_]\w*)*)\s*$", re.MULTILINE
)
"""A mkdocstrings directive declaring one supported package API module."""


class Symbol(NamedTuple):
    """One name bound at module level.

    Attributes
    ----------
    module : str
        The dotted module that defines the name.
    name : str
        The name, or the empty string when the binding is the module itself.
    """

    module: str
    name: str

    def describe(self) -> str:
        """Return the symbol as ``module:name``."""
        return f"{self.module}:{self.name}" if self.name else self.module


@cache
def _parsed(text: str) -> ast.Module:
    """Return the syntax tree of a module's text."""
    return ast.parse(text)


def syntax(path: Path) -> ast.Module:
    """Return the syntax tree of a file.

    Parameters
    ----------
    path : Path
        The file to read.
    """
    return _parsed(path.read_text(encoding="utf-8"))


def source_modules(root: Path) -> dict[str, Path]:
    """Return every module of the source package keyed by its dotted name.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    dict[str, Path]
        The dotted name of each module, a package standing for its
        ``__init__.py``, mapped to its file.
    """
    modules: dict[str, Path] = {}

    for path in sorted((root / "lofi").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue

        parts = path.relative_to(root).with_suffix("").parts

        if parts[-1] == "__init__":
            parts = parts[:-1]

        modules[".".join(parts)] = path

    return modules


def public(module: str) -> bool:
    """Return whether every component of a dotted module name is public.

    Parameters
    ----------
    module : str
        The dotted module name.
    """
    return not any(
        part.startswith("_") and not part.startswith("__") for part in module.split(".")
    )


def _origin(name: str | None, path: Path, node: ast.ImportFrom) -> str:
    """Return the module a ``from`` import reads, relative imports resolved."""
    if not node.level or name is None:
        return node.module or ""

    package = name.split(".")

    if path.name != "__init__.py":
        package = package[:-1]

    package = package[: len(package) - node.level + 1]

    return ".".join([*package, *([node.module] if node.module else [])])


def imported_names(path: Path, name: str | None) -> set[str]:
    """Return every dotted name an import statement of a file could load.

    A ``from`` import contributes both its module and each imported name
    qualified by it, so importing a submodule by name counts. A call to
    ``import_module`` with one literal module name loads that module when it
    runs, and counts as an import of it.

    Parameters
    ----------
    path : Path
        The file to read.
    name : str | None
        The file's dotted module name, against which relative imports resolve;
        None for a file outside the source package.

    Returns
    -------
    set[str]
        The candidate module names.
    """
    names: set[str] = set()

    for node in ast.walk(syntax(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom):
            module = _origin(name, path, node)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)

        elif literal := _literal_import(node):
            names.add(literal)

    return names


def _literal_import(node: ast.AST, /) -> str | None:
    """Return the module name an ``import_module`` call names literally, or None."""
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return None

    function = node.func
    called = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else None
    )
    argument = node.args[0]

    if (
        called != "import_module"
        or not isinstance(argument, ast.Constant)
        or not isinstance(argument.value, str)
    ):
        return None

    return argument.value


def loaded_modules(names: set[str], modules: dict[str, Path]) -> set[str]:
    """Return the source modules that importing some names loads.

    Importing a dotted name loads every package on its path as well.

    Parameters
    ----------
    names : set[str]
        The imported dotted names.
    modules : dict[str, Path]
        The source modules by dotted name.

    Returns
    -------
    set[str]
        The source modules loaded.
    """
    loaded: set[str] = set()

    for name in names:
        parts = name.split(".")
        loaded.update(
            prefix
            for prefix in (".".join(parts[:end]) for end in range(1, len(parts) + 1))
            if prefix in modules
        )

    return loaded


def entry_references(root: Path) -> tuple[Symbol, ...]:
    """Return the objects the distribution's console scripts and entry points name.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    tuple[Symbol, ...]
        Each declared object reference; a reference naming a whole module has
        an empty name.
    """
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    references = [*project.get("scripts", {}).values()]

    for group in project.get("entry-points", {}).values():
        references.extend(group.values())

    return tuple(
        Symbol(module, attribute.split(".")[0])
        for module, _, attribute in (
            reference.partition(":") for reference in references
        )
    )


def documented_modules(root: Path) -> set[str]:
    """Return public source modules exposed by the generated API reference.

    The reference pages are executable package boundaries: mkdocstrings imports
    every directive target while building the published API. A private module
    cannot become public merely by appearing in prose.
    """
    modules = source_modules(root)
    reference = root / "docs" / "reference"

    if not reference.is_dir():
        return set()

    return {
        name
        for path in reference.rglob("*.md")
        for name in _API_DIRECTIVE.findall(path.read_text(encoding="utf-8"))
        if name in modules and public(name)
    }


def reachable_modules(root: Path) -> set[str]:
    """Return the source modules reachable from the distribution's entry points.

    The roots are the modules named by the console scripts and the plugin
    entry points, and every ``__main__`` module, which ``python -m`` runs;
    reachability follows import statements from there.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    set[str]
        The dotted names of the reachable source modules.
    """
    modules = source_modules(root)
    seeds = (
        {reference.module for reference in entry_references(root)}
        | documented_modules(root)
        | {name for name in modules if name.split(".")[-1] == "__main__"}
    )
    reached: set[str] = set()
    pending = sorted(loaded_modules(seeds, modules))

    while pending:
        name = pending.pop()

        if name in reached:
            continue

        reached.add(name)
        pending.extend(
            loaded_modules(imported_names(modules[name], name), modules) - reached
        )

    return reached


def _module_statements(tree: ast.Module) -> list[ast.stmt]:
    """Return a module's top-level statements, conditional blocks flattened."""
    statements: list[ast.stmt] = []
    pending = list(tree.body)

    while pending:
        statement = pending.pop(0)

        if isinstance(statement, ast.If):
            pending[:0] = [*statement.body, *statement.orelse]
        elif isinstance(statement, ast.Try):
            pending[:0] = [
                *statement.body,
                *(inner for handler in statement.handlers for inner in handler.body),
                *statement.orelse,
                *statement.finalbody,
            ]
        else:
            statements.append(statement)

    return statements


def defined_names(tree: ast.Module) -> dict[str, ast.stmt]:
    """Return the names a module defines itself, with the defining statement.

    Parameters
    ----------
    tree : ast.Module
        The module's syntax tree.

    Returns
    -------
    dict[str, ast.stmt]
        Every class, function, type alias and assigned name bound at module
        level, imports excluded.
    """
    defined: dict[str, ast.stmt] = {}

    for statement in _module_statements(tree):
        match statement:
            case ast.ClassDef(name=name) | ast.FunctionDef(name=name):
                defined[name] = statement
            case ast.AsyncFunctionDef(name=name):
                defined[name] = statement
            case ast.TypeAlias(name=ast.Name(id=name)):
                defined[name] = statement
            case ast.AnnAssign(target=ast.Name(id=name)):
                defined[name] = statement
            case ast.Assign(targets=targets):
                defined.update(
                    (target.id, statement)
                    for target in targets
                    if isinstance(target, ast.Name)
                )
            case _:
                pass

    return defined


def bindings(path: Path, name: str) -> dict[str, Symbol]:
    """Return what each module-level name of a module refers to.

    Parameters
    ----------
    path : Path
        The module's file.
    name : str
        The module's dotted name.

    Returns
    -------
    dict[str, Symbol]
        A defined name maps to itself in this module; an imported name maps
        to the symbol it was imported as, a module binding having an empty
        name.
    """
    tree = syntax(path)
    bound: dict[str, Symbol] = {
        defined: Symbol(name, defined) for defined in defined_names(tree)
    }

    for statement in _module_statements(tree):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname:
                    bound[alias.asname] = Symbol(alias.name, "")
                else:
                    head = alias.name.split(".")[0]
                    bound[head] = Symbol(head, "")

        elif isinstance(statement, ast.ImportFrom):
            origin = _origin(name, path, statement)

            for alias in statement.names:
                bound[alias.asname or alias.name] = Symbol(origin, alias.name)

    return bound


def exported(tree: ast.Module) -> tuple[str, ...]:
    """Return the names a module lists in ``__all__``.

    Parameters
    ----------
    tree : ast.Module
        The module's syntax tree.
    """
    for statement in tree.body:
        match statement:
            case (
                ast.Assign(
                    targets=[ast.Name(id="__all__")],
                    value=ast.List() | ast.Tuple() as value,
                )
                | ast.AnnAssign(
                    target=ast.Name(id="__all__"),
                    value=ast.List() | ast.Tuple() as value,
                )
            ):
                return tuple(
                    item.value
                    for item in value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
            case _:
                pass

    return ()


class Package:
    """The modules of one source tree, with name resolution across them.

    Parameters
    ----------
    root : Path
        The repository root.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.modules = source_modules(root)
        self._bindings = {
            name: bindings(path, name) for name, path in self.modules.items()
        }

    def tree(self, module: str) -> ast.Module:
        """Return a module's syntax tree.

        Parameters
        ----------
        module : str
            The dotted module name.
        """
        return syntax(self.modules[module])

    def relative(self, module: str) -> str:
        """Return a module's file relative to the repository root.

        Parameters
        ----------
        module : str
            The dotted module name.
        """
        return self.modules[module].relative_to(self.root).as_posix()

    def bound(self, module: str) -> dict[str, Symbol]:
        """Return what each module-level name of a module refers to.

        Parameters
        ----------
        module : str
            The dotted module name.
        """
        return self._bindings[module]

    def resolve(self, symbol: Symbol) -> Symbol:
        """Return the symbol a name finally refers to, imports followed.

        Parameters
        ----------
        symbol : Symbol
            A name as bound in some module.

        Returns
        -------
        Symbol
            The defining module and name for a name the source package
            defines; a source module for a name bound to one; otherwise the
            symbol as reached, outside the package.
        """
        seen: set[Symbol] = set()

        while symbol not in seen:
            seen.add(symbol)

            if not symbol.name:
                return symbol

            if symbol.module not in self.modules:
                submodule = f"{symbol.module}.{symbol.name}"

                return Symbol(submodule, "") if submodule in self.modules else symbol

            target = self._bindings[symbol.module].get(symbol.name)

            if target is None:
                submodule = f"{symbol.module}.{symbol.name}"

                return Symbol(submodule, "") if submodule in self.modules else symbol

            if target == symbol:
                return symbol

            symbol = target

        return symbol

    def reference(self, module: str, node: ast.expr) -> Symbol | None:
        """Return the symbol a dotted-name expression in a module refers to.

        Parameters
        ----------
        module : str
            The module the expression appears in.
        node : ast.expr
            A name or an attribute chain over a name.

        Returns
        -------
        Symbol | None
            The symbol referred to, or None for any other expression. An
            attribute of a class resolves to the class.
        """
        attributes: list[str] = []

        while isinstance(node, ast.Attribute):
            attributes.insert(0, node.attr)
            node = node.value

        if not isinstance(node, ast.Name):
            return None

        symbol = self.resolve(Symbol(module, node.id))

        for attribute in attributes:
            if symbol.name:
                return symbol

            symbol = self.resolve(Symbol(symbol.module, attribute))

        return symbol


def declaration_classes(package: Package) -> dict[Symbol, ast.ClassDef]:
    """Return every class deriving from a didactic declaration class.

    Parameters
    ----------
    package : Package
        The source package.

    Returns
    -------
    dict[Symbol, ast.ClassDef]
        Each declaration class by its defining module and name.
    """
    classes = {
        Symbol(module, statement.name): statement
        for module in package.modules
        for statement in _module_statements(package.tree(module))
        if isinstance(statement, ast.ClassDef)
    }
    declared: dict[Symbol, ast.ClassDef] = {}
    changed = True

    while changed:
        changed = False

        for symbol, statement in classes.items():
            if symbol in declared:
                continue

            for base in statement.bases:
                target = package.reference(symbol.module, base)

                if target is None:
                    continue

                if target in declared or (
                    target.module.startswith("didactic")
                    and target.name in DECLARATION_BASES
                ):
                    declared[symbol] = statement
                    changed = True
                    break

    return declared


def annotation_references(
    package: Package, module: str, annotation: ast.expr
) -> set[Symbol]:
    """Return the symbols a type annotation names, string annotations included.

    Parameters
    ----------
    package : Package
        The source package.
    module : str
        The module the annotation appears in.
    annotation : ast.expr
        The annotation.
    """
    found: set[Symbol] = set()

    for node in ast.walk(annotation):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                continue

            found |= annotation_references(package, module, parsed)

        elif isinstance(node, ast.Name | ast.Attribute):
            target = package.reference(module, node)

            if target is not None:
                found.add(target)

    return found


def declaration_fields(
    package: Package,
    symbol: Symbol,
    statement: ast.ClassDef,
    declared: dict[Symbol, ast.ClassDef],
) -> set[Symbol]:
    """Return the symbols a declaration class's field annotations name.

    Parameters
    ----------
    package : Package
        The source package.
    symbol : Symbol
        The class.
    statement : ast.ClassDef
        Its definition.
    declared : dict[Symbol, ast.ClassDef]
        The declaration classes in the package.
    """
    found: set[Symbol] = set()

    for field in statement.body:
        if isinstance(field, ast.AnnAssign) and "ClassVar" not in ast.unparse(
            field.annotation
        ):
            pending = list(
                annotation_references(package, symbol.module, field.annotation)
            )
            seen: set[Symbol] = set()

            while pending:
                target = pending.pop()

                if target in seen:
                    continue

                seen.add(target)

                if target in declared:
                    found.add(target)
                    continue

                if target.module not in package.modules:
                    continue

                definition = defined_names(package.tree(target.module)).get(target.name)

                if isinstance(definition, ast.TypeAlias):
                    pending.extend(
                        annotation_references(
                            package,
                            target.module,
                            definition.value,
                        )
                    )

    return found


def _class_arguments(package: Package, module: str, scope: ast.AST) -> set[Symbol]:
    """Return the classes a scope's ``type[...]`` annotations admit."""
    found: set[Symbol] = set()

    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "type"
        ):
            found |= annotation_references(package, module, node.slice)

    return found


def document_roots(package: Package, modules: set[str]) -> set[Symbol]:
    """Return the declaration classes a document is read as.

    A root is a class passed as the ``schema`` of a call anywhere in the
    package, as the run, experiment, site and release documents are each
    composed against one, or a class whose ``model_validate`` or
    ``model_validate_json`` reads a document from text. A schema passed
    through a variable stands for the classes the ``type[...]`` annotations
    of the enclosing class or function admit.

    Parameters
    ----------
    package : Package
        The source package.
    modules : set[str]
        The modules whose reads count, the reachable ones.
    """
    roots: set[Symbol] = set()

    for module in sorted(modules):
        tree = package.tree(module)
        pending: list[tuple[ast.AST, ast.AST]] = [(tree, tree)]

        while pending:
            node, scope = pending.pop()

            if scope is tree and isinstance(
                node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                scope = node

            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg != "schema":
                        continue

                    target = package.reference(module, keyword.value)

                    if (
                        target is not None
                        and target.name
                        and (
                            isinstance(keyword.value, ast.Name)
                            and keyword.value.id in package.bound(module)
                        )
                    ):
                        roots.add(target)
                    else:
                        roots |= _class_arguments(package, module, scope)

                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "model_validate",
                    "model_validate_json",
                }:
                    target = package.reference(module, node.func.value)

                    if target is not None:
                        roots.add(target)

            pending.extend((child, scope) for child in ast.iter_child_nodes(node))

    return roots


def reachable_declarations(package: Package) -> set[Symbol]:
    """Return the declaration classes a document can hold.

    The roots are the document roots of the reachable modules. From each, a
    class reaches the classes its fields name, the declaration classes it
    derives from, and every variant derived from it in a reachable module,
    since a field typed by a union admits each registered variant.

    Parameters
    ----------
    package : Package
        The source package.
    """
    modules = reachable_modules(package.root)
    declared = declaration_classes(package)
    variants: dict[Symbol, set[Symbol]] = {}

    for symbol, statement in declared.items():
        for base in statement.bases:
            target = package.reference(symbol.module, base)

            if target in declared and symbol.module in modules:
                variants.setdefault(target, set()).add(symbol)

    reached: set[Symbol] = set()
    pending = sorted(document_roots(package, modules) & declared.keys())

    while pending:
        symbol = pending.pop()

        if symbol in reached:
            continue

        reached.add(symbol)
        statement = declared[symbol]
        bases = {
            target
            for base in statement.bases
            if (target := package.reference(symbol.module, base)) in declared
        }
        named = (
            declaration_fields(package, symbol, statement, declared) & declared.keys()
        )
        pending.extend((named | bases | variants.get(symbol, set())) - reached)

    return reached


def spec_roots(package: Package) -> dict[Symbol, ast.ClassDef]:
    """Return the family roots: the declaration classes deriving directly from a union.

    A family root is what a document field is typed by when it admits any
    registered variant of one kind of component.

    Parameters
    ----------
    package : Package
        The source package.
    """
    return {
        symbol: statement
        for symbol, statement in declaration_classes(package).items()
        if any(
            (target := package.reference(symbol.module, base)) is not None
            and target.module.startswith("didactic")
            and target.name == "TaggedUnion"
            for base in statement.bases
        )
    }


class Finding(NamedTuple):
    """One violation of a structural convention.

    Attributes
    ----------
    key : str
        The stable name an expectation listing records the violation under,
        free of line numbers.
    where : str
        The ``path:line`` the violation is at.
    what : str
        What the violation is.
    """

    key: str
    where: str
    what: str

    def describe(self) -> str:
        """Return the finding as a ``path:line: what`` report."""
        return f"{self.where}: {self.what} [{self.key}]"


def known_violations(path: Path) -> dict[str, dict[str, str]]:
    """Return the listed violations of each rule with their reasons.

    The listing holds, per rule, groups of entries that share one reason.

    Parameters
    ----------
    path : Path
        The listing.

    Returns
    -------
    dict[str, dict[str, str]]
        Each listed entry's reason, by rule.
    """
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    return {
        rule: {entry: group["reason"] for group in groups for entry in group["entries"]}
        for rule, groups in document.items()
    }


def compare(
    found: list[Finding], listing: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Return the unlisted findings and the listed entries no longer found.

    Parameters
    ----------
    found : list[Finding]
        The violations in the tree.
    listing : dict[str, str]
        The listed entries with their reasons.

    Returns
    -------
    tuple[list[str], list[str]]
        Each new violation's report, and each stale entry.
    """
    keys = {finding.key for finding in found}

    return (
        sorted(finding.describe() for finding in found if finding.key not in listing),
        sorted(listing.keys() - keys),
    )

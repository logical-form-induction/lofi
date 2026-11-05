"""Each behavior, value and public name exists once, and is used.

Each rule reads the source or the packaged configuration structurally:

- a parallel implementation is two public functions in one module whose
  bodies are the same once their parameters and locals are renamed in order
  of appearance, so they compute the same result in one namespace;
- a duplicated spec default is one field name given one literal, non-empty
  default by two variants of the same declaration family;
- a restated packaged value is a leaf a packaged run document sets to the
  default its declaration already gives, or a leaf any packaged document or
  fragment sets to the value a fragment it composes already sets (a group
  fragment may pin a default on purpose, since it is layered over a document
  that may set the leaf otherwise);
- an unused public symbol is a public module-level name of a reachable
  module that no module of the package reads, no entry point or ``__all__``
  names, and no document can hold;
- a duplicated native implementation is a Python function or class with the
  name of one a Rust extension exports, which does not call through to it.

Every finding fails the suite: duplicate or unused implementation is not a
supported compatibility surface.
"""

import ast
import copy
import enum
import json
import re
import types
import typing
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, override

import didactic.api as dx
import pytest
import yaml
from didactic.fields import MISSING
from didactic.settings import compose_traced
from tests.structure import (
    TREE,
    Finding,
    Package,
    Symbol,
    declaration_classes,
    defined_names,
    entry_references,
    exported,
    reachable_declarations,
    reachable_modules,
    spec_roots,
)

from lofi.spec import RunSpec, load_plugins
from lofi.spec._groups import PACKAGED_CONFIG_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from didactic.settings import ConfigValue

SMALLEST_BODY: Final = 20
"""The fewest syntax nodes a copied body needs to count as a second implementation."""

EMPTY_FACTORIES: Final = frozenset({"dict", "frozenset", "list", "set", "tuple"})
"""The built-in constructors whose call with no argument builds an empty value."""


class _Renamer(ast.NodeTransformer):
    """Renames the names a function binds in order of appearance."""

    def __init__(self, bound: frozenset[str]) -> None:
        self._bound = bound
        self._names: dict[str, str] = {}

    def _renamed(self, name: str) -> str:
        if name not in self._bound:
            return name

        return self._names.setdefault(name, f"_{len(self._names)}")

    @override
    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._renamed(node.id)

        return node

    @override
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._renamed(node.arg)
        node.annotation = None

        return node


def _shape(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, int] | None:
    """Return a function's body with its own names renamed, and the body's size.

    The docstring, decorators, annotations and the function's name are left
    out, so two functions with one shape compute the same result from the same
    arguments.
    """
    body = function.body

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if not body:
        return None

    copied = copy.deepcopy(function)
    copied.body = copy.deepcopy(body)
    copied.name = ""
    copied.decorator_list = []
    copied.returns = None
    copied.type_params = []
    bound = frozenset(
        {node.arg for node in ast.walk(copied.args) if isinstance(node, ast.arg)}
        | {
            node.id
            for node in ast.walk(copied)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
    )
    _Renamer(bound).visit(copied)
    size = sum(1 for statement in body for _ in ast.walk(statement))

    return ast.dump(copied, annotate_fields=False), size


def _functions(
    tree: ast.Module,
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield every function and method of a module with its dotted name."""
    pending: list[tuple[str, ast.AST]] = [("", tree)]

    while pending:
        prefix, node = pending.pop()

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                yield f"{prefix}{child.name}", child
                pending.append((f"{prefix}{child.name}.", child))
            elif isinstance(child, ast.ClassDef):
                pending.append((f"{prefix}{child.name}.", child))


def parallel_implementations(package: Package) -> list[Finding]:
    """Return public functions in one namespace that share one renamed body.

    Methods are implementations of their receiver's protocol and can read
    different properties despite sharing syntax. Functions in different
    modules likewise resolve their global names in different namespaces.
    Neither equality is enough to establish a parallel implementation.

    Parameters
    ----------
    package : Package
        The source package.
    """
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for module in sorted(package.modules):
        for name, function in _functions(package.tree(module)):
            if "." in name or name.startswith("_"):
                continue

            shaped = _shape(function)

            if shaped is not None and shaped[1] >= SMALLEST_BODY:
                groups.setdefault((module, shaped[0]), []).append(
                    (
                        f"{module}:{name}",
                        f"{package.relative(module)}:{function.lineno}",
                    )
                )

    return [
        Finding(
            " = ".join(sorted(name for name, _ in members)),
            min(where for _, where in members),
            "one body implemented at " + ", ".join(where for _, where in members),
        )
        for members in groups.values()
        if len(members) > 1
    ]


def _neutral(default: ast.expr) -> bool:
    """Return whether a default declares absence: None or an empty collection."""
    match default:
        case ast.Constant(value=None):
            return True
        case ast.Tuple(elts=[]) | ast.List(elts=[]) | ast.Set(elts=[]):
            return True
        case ast.Dict(keys=[]):
            return True
        case ast.Call(func=ast.Name(id=name), args=[], keywords=[]):
            return name in EMPTY_FACTORIES
        case ast.Call(keywords=keywords):
            factory = next(
                (kw.value for kw in keywords if kw.arg == "default_factory"), None
            )

            return isinstance(factory, ast.Name) and factory.id in EMPTY_FACTORIES
        case _:
            return False


def _default(field: ast.AnnAssign) -> ast.expr | None:
    """Return the default a declaration field declares, or None."""
    value = field.value

    if value is None:
        return None

    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        if value.func.attr != "field":
            return value

        explicit = next(
            (kw.value for kw in value.keywords if kw.arg == "default"), None
        )

        return explicit if explicit is not None else value

    return value


def _declaration_families(
    package: Package, declarations: Mapping[Symbol, ast.ClassDef]
) -> dict[Symbol, frozenset[Symbol]]:
    """Return the tagged-union roots each declaration descends from."""
    roots = frozenset(spec_roots(package))
    parents = {
        symbol: frozenset(
            target
            for base in statement.bases
            if (target := package.reference(symbol.module, base)) in declarations
            if target is not None
        )
        for symbol, statement in declarations.items()
    }
    families: dict[Symbol, frozenset[Symbol]] = {}

    for symbol in declarations:
        found: set[Symbol] = set()
        seen: set[Symbol] = set()
        pending = [symbol]

        while pending:
            candidate = pending.pop()

            if candidate in seen:
                continue

            seen.add(candidate)

            if candidate in roots:
                found.add(candidate)

            pending.extend(parents.get(candidate, ()))

        families[symbol] = frozenset(found)

    return families


def duplicated_defaults(package: Package) -> list[Finding]:
    """Return literal defaults repeated by variants of one declaration family.

    A ``Literal`` field names its own variant's tag and is left out, as is a
    default of None or of an empty collection, which declares absence. A name
    or attribute is a reference to one shared default rather than a second
    declaration of its value.

    Parameters
    ----------
    package : Package
        The source package.
    """
    declarations = declaration_classes(package)
    families = _declaration_families(package, declarations)
    groups: dict[tuple[Symbol, str], list[str]] = {}

    for symbol, statement in sorted(declarations.items()):
        for field in statement.body:
            if not isinstance(field, ast.AnnAssign) or not isinstance(
                field.target, ast.Name
            ):
                continue

            annotation = ast.unparse(field.annotation)
            default = _default(field)

            if (
                default is None
                or annotation.startswith(("Literal[", "ClassVar"))
                or _neutral(default)
                or isinstance(default, ast.Name | ast.Attribute)
            ):
                continue

            key = f"{field.target.id} = {ast.unparse(default)}"

            for family in families[symbol]:
                groups.setdefault((family, key), []).append(
                    f"{package.relative(symbol.module)}:{field.lineno}"
                )

    return [
        Finding(key, places[0], f"{key} is declared at {', '.join(places)}")
        for (_, key), places in sorted(groups.items())
        if len(places) > 1
    ]


type Annotation = (
    type
    | types.UnionType
    | types.GenericAlias
    | typing.TypeAliasType
    | typing.TypeVar
    | typing.ForwardRef
)
"""A field annotation as didactic records it."""


def _members(annotation: Annotation) -> list[Annotation]:
    """Return the types an annotation admits, absence left out."""
    origin = typing.get_origin(annotation)

    if origin in {typing.Union, types.UnionType}:
        return [
            member
            for argument in typing.get_args(annotation)
            for member in _members(argument)
            if member is not type(None)
        ]

    if origin is typing.Annotated:
        return _members(typing.get_args(annotation)[0])

    return [annotation]


def _declaration(candidate: Annotation) -> type[dx.Model] | None:
    """Return a candidate type as a declaration class, or None."""
    if isinstance(candidate, type) and issubclass(candidate, dx.Model):
        return candidate

    return None


def _union(candidate: type[dx.Model]) -> type[dx.TaggedUnion] | None:
    """Return a declaration class as a union root with registered variants, or None."""
    if issubclass(candidate, dx.TaggedUnion) and "__variants__" in vars(candidate):
        return candidate

    return None


def _variants(candidate: type[dx.Model]) -> list[type[dx.Model]]:
    """Return the registered variants of a union root, or the class itself."""
    root = _union(candidate)

    return [candidate] if root is None else list(root.__variants__.values())


def _chosen(
    candidates: list[Annotation], document: Mapping[str, ConfigValue]
) -> list[type[dx.Model]]:
    """Return the declaration classes a document mapping may be read as.

    A union root narrows to the variant the mapping's discriminator names, or
    stands for every variant when the mapping names none.
    """
    chosen: list[type[dx.Model]] = []

    for candidate in candidates:
        declared = _declaration(candidate)

        if declared is None:
            continue

        variants = _variants(declared)
        root = _union(declared)

        if root is not None and root.__discriminator__ is not None:
            tag = document.get(root.__discriminator__)

            if isinstance(tag, str) and tag in root.__variants__:
                variants = [root.__variants__[tag]]

        chosen.extend(variant for variant in variants if variant not in chosen)

    return chosen


def _field_types(declared: list[type[dx.Model]], key: str) -> list[Annotation]:
    """Return the types the field of a key admits in some declaration classes."""
    found: list[Annotation] = []

    for variant in declared:
        field = variant.__field_specs__.get(key)

        if field is not None:
            found.extend(
                member for member in _members(field.annotation) if member not in found
            )

    return found


def _step(candidates: list[Annotation], key: str) -> list[Annotation]:
    """Return the types a document key admits below some candidate types."""
    stepped = [
        member
        for candidate in candidates
        if typing.get_origin(candidate) is dict
        for member in _members(typing.get_args(candidate)[1])
    ]

    for declared in candidates:
        root = _declaration(declared)

        if root is not None:
            stepped.extend(
                member
                for member in _field_types(_variants(root), key)
                if member not in stepped
            )

    return stepped


def _plain[V](value: V) -> ConfigValue:
    """Return a declared default in the form a document writes it."""
    if isinstance(value, enum.Enum):
        return _plain(value.value)

    if isinstance(value, dx.Model):
        return json.loads(value.model_dump_json())

    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}

    if isinstance(value, bool | int | float | str) or value is None:
        return value

    return repr(value)


def _same(written: ConfigValue, declared: ConfigValue) -> bool:
    """Return whether a document value equals a declared default, booleans apart."""
    if isinstance(written, bool) or isinstance(declared, bool):
        return type(written) is type(declared) and written == declared

    if isinstance(written, list | tuple) and isinstance(declared, list | tuple):
        return len(written) == len(declared) and all(
            _same(left, right) for left, right in zip(written, declared, strict=True)
        )

    if isinstance(written, Mapping) and isinstance(declared, Mapping):
        return written.keys() == declared.keys() and all(
            _same(written[key], declared[key]) for key in written
        )

    return written == declared


def _restated_defaults(
    document: Mapping[str, ConfigValue], candidates: list[Annotation], prefix: str
) -> Iterator[str]:
    """Yield the dotted leaves of a document that restate their declared default.

    A leaf counts only when every declaration class the mapping may be read as
    declares the field with that same default.
    """
    declared = _chosen(candidates, document)
    keyed = [
        member
        for candidate in candidates
        if typing.get_origin(candidate) is dict
        for member in _members(typing.get_args(candidate)[1])
    ]

    for key, value in document.items():
        if key.endswith("+") or (key == "defaults" and not prefix):
            continue

        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict) and value:
            yield from _restated_defaults(
                value, [*keyed, *_field_types(declared, key)], path
            )
            continue

        fields = [
            variant.__field_specs__[key]
            for variant in declared
            if key in variant.__field_specs__
        ]
        defaults = [
            field.default_factory()
            if field.default_factory is not None
            else field.default
            for field in fields
            if typing.get_origin(field.annotation) is not typing.Literal
            and (field.default_factory is not None or field.default is not MISSING)
        ]

        if (
            defaults
            and len(defaults) == len(fields)
            and all(_same(value, _plain(default)) for default in defaults)
        ):
            yield path


def _leaves(
    document: Mapping[str, ConfigValue], prefix: str = ""
) -> Iterator[tuple[str, ConfigValue]]:
    """Yield each dotted leaf a document sets, with its value."""
    for key, value in document.items():
        if key == "defaults" and not prefix:
            continue

        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict) and value:
            yield from _leaves(value, path)
        else:
            yield path, value


def restated_packaged_values(package: Package) -> list[Finding]:
    """Return the packaged leaves that restate a declared default or a lower layer.

    The packaged configuration root is ``lofi/configs``. A fragment's slot is
    its directory below it; a file in a directory that names no slot is a run
    document, the lowest layer of any composition, so only a run document's
    leaves are compared with their declared defaults.

    Parameters
    ----------
    package : Package
        The source package.
    """
    load_plugins()
    root = package.root / "lofi" / "configs"
    found: list[Finding] = []

    for path in sorted(root.rglob("*.yaml")):
        relative = path.relative_to(root)
        candidates: list[Annotation] = [RunSpec]

        for key in relative.parent.parts:
            candidates = _step(candidates, key)

        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name = path.relative_to(package.root).as_posix()

        if not candidates:
            candidates = [RunSpec]
            found.extend(
                Finding(f"{name}:{leaf}", name, f"{leaf} restates its declared default")
                for leaf in _restated_defaults(document, candidates, "")
            )

        if "defaults" not in document:
            continue

        schemas = [
            variant
            for candidate in candidates
            if (declared := _declaration(candidate)) is not None
            for variant in _variants(declared)
        ]
        lower: dict[str, tuple[ConfigValue, str]] = {}

        for schema in schemas:
            try:
                composed = compose_traced(path, schema=schema, search_path=(root,))
            except dx.ValidationError, ValueError:
                continue

            for layer in composed.layers:
                if layer.origin.kind != "file":
                    lower.update(
                        (leaf, (value, layer.origin.name))
                        for leaf, value in _leaves(dict(layer.document))
                    )

            break

        found.extend(
            Finding(
                f"{name}:{leaf}",
                name,
                f"{leaf} restates the value {lower[leaf][1]} sets",
            )
            for leaf, value in _leaves(document)
            if leaf in lower and _same(value, lower[leaf][0])
        )

    return found


def _identifiers(node: ast.AST) -> Counter[str]:
    """Return how often a syntax tree reads each identifier.

    A string annotation, or the type a ``cast`` names as a string, is read
    as the expression it spells.
    """
    read: Counter[str] = Counter()

    for part in ast.walk(node):
        if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
            read[part.id] += 1
        elif isinstance(part, ast.Attribute):
            read[part.attr] += 1

        annotations = []

        if isinstance(part, ast.arg) and part.annotation is not None:
            annotations.append(part.annotation)
        elif isinstance(part, ast.FunctionDef | ast.AsyncFunctionDef) and part.returns:
            annotations.append(part.returns)
        elif isinstance(part, ast.AnnAssign):
            annotations.append(part.annotation)
        elif (
            isinstance(part, ast.Call)
            and part.args
            and (
                (isinstance(part.func, ast.Name) and part.func.id == "cast")
                or (isinstance(part.func, ast.Attribute) and part.func.attr == "cast")
            )
        ):
            annotations.append(part.args[0])

        for annotation in annotations:
            for text in ast.walk(annotation):
                if isinstance(text, ast.Constant) and isinstance(text.value, str):
                    try:
                        parsed = ast.parse(text.value, mode="eval")
                    except SyntaxError:
                        continue

                    read.update(
                        name.id
                        for name in ast.walk(parsed)
                        if isinstance(name, ast.Name)
                    )

    return read


def unused_public_symbols(package: Package) -> list[Finding]:
    """Return the public names of reachable modules nothing reads or reaches.

    Parameters
    ----------
    package : Package
        The source package.
    """
    modules = reachable_modules(package.root)
    read: Counter[str] = Counter()
    own: dict[Symbol, Counter[str]] = {}

    for module in package.modules:
        tree = package.tree(module)
        read.update(_identifiers(tree))

        for name, statement in defined_names(tree).items():
            own[Symbol(module, name)] = _identifiers(statement)

    named = set(entry_references(package.root)) | {
        package.resolve(Symbol(module, name))
        for module in modules
        for name in exported(package.tree(module))
    }
    held = reachable_declarations(package)
    found: list[Finding] = []

    for module in sorted(modules):
        for name, statement in defined_names(package.tree(module)).items():
            symbol = Symbol(module, name)

            if (
                name.startswith("_")
                or symbol in named
                or symbol in held
                or read[name] - own[symbol][name] > 0
            ):
                continue

            found.append(
                Finding(
                    symbol.describe(),
                    f"{package.relative(module)}:{statement.lineno}",
                    f"{symbol.describe()} is read by nothing in the package and "
                    "reached by no entry point or document",
                )
            )

    return found


NATIVE_EXPORT: Final = re.compile(
    r"#\[(?P<kind>pyfunction|pyclass)(?:\((?P<options>[^)]*)\))?\]"
    r"(?P<between>(?:\s*#\[[^\]]*\])*)\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:fn|struct|enum)\s+(?P<name>\w+)"
)
"""A Rust item exported to Python by a PyO3 attribute, with any rename."""

RENAMED: Final = re.compile(r'name\s*=\s*"(?P<name>\w+)"')
"""A PyO3 rename option."""


def native_exports(root: Path) -> set[str]:
    """Return the Python names the repository's Rust extensions export.

    Parameters
    ----------
    root : Path
        The repository root.
    """
    exports: set[str] = set()

    for path in sorted(root.rglob("*.rs")):
        if {"target", ".venv"} & set(path.relative_to(root).parts):
            continue

        for match in NATIVE_EXPORT.finditer(path.read_text(encoding="utf-8")):
            renamed = RENAMED.search(
                (match.group("options") or "") + match.group("between")
            )
            exports.add(renamed.group("name") if renamed else match.group("name"))

    return exports


def native_duplicates(package: Package) -> list[Finding]:
    """Return the Python definitions that reimplement a Rust export.

    A definition named like an export is a binding when its body reads the
    export's name, which a wrapper calling through to the extension does.

    Parameters
    ----------
    package : Package
        The source package.
    """
    exports = native_exports(package.root)
    found: list[Finding] = []

    for module in sorted(package.modules):
        for name, statement in defined_names(package.tree(module)).items():
            if name in exports and not _identifiers(statement)[name]:
                found.append(
                    Finding(
                        f"{module}:{name}",
                        f"{package.relative(module)}:{statement.lineno}",
                        f"{module}:{name} reimplements the native {name}",
                    )
                )

    return found


PACKAGE: Final = Package(TREE)

RULES: Final[Mapping[str, Callable[[Package], list[Finding]]]] = {
    "parallel-implementations": parallel_implementations,
    "duplicated-spec-defaults": duplicated_defaults,
    "restated-packaged-values": restated_packaged_values,
    "unused-public-symbols": unused_public_symbols,
    "native-duplicates": native_duplicates,
}


@pytest.mark.parametrize("rule", sorted(RULES))
def test_no_redundancy(rule: str) -> None:
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
    "pyproject.toml": (
        '[project]\nname = "lofi"\n[project.scripts]\nlofi = "lofi.core._main:main"\n'
    ),
    "lofi/__init__.py": "",
    "lofi/core/__init__.py": "",
    "lofi/core/_main.py": (
        "from lofi.core._chart import Chart, fill\n"
        "def main():\n    return fill(Chart(), 3)\n"
    ),
    "lofi/core/_chart.py": (
        "class Chart:\n"
        "    def cells(self, length):\n"
        "        spans = []\n"
        "        for width in range(1, length + 1):\n"
        "            for start in range(length - width + 1):\n"
        "                spans.append((start, start + width))\n"
        "        return tuple(spans)\n\n"
        "def fill(chart, length):\n    return chart.cells(length)\n"
    ),
}


def planted(tmp_path: Path, files: Mapping[str, str]) -> Package:
    """Return the package of a miniature tree with some files added."""
    write_tree(tmp_path, {**BASE, **files})

    return Package(tmp_path)


def test_the_miniature_tree_has_no_redundancy(tmp_path: Path) -> None:
    package = planted(tmp_path, {})

    assert parallel_implementations(package) == []
    assert duplicated_defaults(package) == []
    assert unused_public_symbols(package) == []
    assert native_duplicates(package) == []


def test_a_second_public_implementation_in_one_module_is_reported(
    tmp_path: Path,
) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/_chart.py": (
                BASE["lofi/core/_chart.py"] + "\ndef packed_cells(n):\n"
                "    found = []\n"
                "    for size in range(1, n + 1):\n"
                "        for left in range(n - size + 1):\n"
                "            found.append((left, left + size))\n"
                "    return tuple(found)\n\n"
                "def alternate_cells(length):\n"
                "    spans = []\n"
                "    for width in range(1, length + 1):\n"
                "        for start in range(length - width + 1):\n"
                "            spans.append((start, start + width))\n"
                "    return tuple(spans)\n"
            ),
        },
    )

    assert [finding.key for finding in parallel_implementations(package)] == [
        "lofi.core._chart:alternate_cells = lofi.core._chart:packed_cells"
    ]


def test_a_method_computing_something_else_is_not_reported(tmp_path: Path) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/_packed.py": (
                "class PackedChart:\n"
                "    def cells(self, n):\n"
                "        found = []\n"
                "        for size in range(1, n + 1):\n"
                "            for left in range(n - size):\n"
                "                found.append((left, left + size))\n"
                "        return tuple(found)\n"
            ),
        },
    )

    assert parallel_implementations(package) == []


def test_protocol_methods_on_different_receivers_are_not_parallel_routes(
    tmp_path: Path,
) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/_packed.py": (
                "class PackedChart:\n"
                "    def cells(self, n):\n"
                "        found = []\n"
                "        for size in range(1, n + 1):\n"
                "            for left in range(n - size + 1):\n"
                "                found.append((left, left + size))\n"
                "        return tuple(found)\n"
            ),
            "lofi/core/_main.py": (
                "from lofi.core._chart import Chart, fill\n"
                "from lofi.core._packed import PackedChart\n"
                "def main():\n    return fill(Chart(), 3), fill(PackedChart(), 3)\n"
            ),
        },
    )

    assert parallel_implementations(package) == []


def test_a_duplicated_spec_default_is_reported(tmp_path: Path) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/spec.py": (
                "import didactic.api as dx\n"
                "from typing import Literal\n"
                "class EncoderSpec(dx.TaggedUnion, discriminator='kind'):\n"
                "    pass\n"
                "class RnnSpec(EncoderSpec):\n"
                "    kind: Literal['rnn'] = 'rnn'\n"
                "    padding: str = '<pad>'\n"
                "    norm_cap: float | None = None\n"
                "class GruSpec(EncoderSpec):\n"
                "    kind: Literal['gru'] = 'gru'\n"
                "    padding: str = dx.field(default='<pad>')\n"
                "    norm_cap: float | None = None\n"
            ),
        },
    )

    assert [finding.key for finding in duplicated_defaults(package)] == [
        "padding = '<pad>'"
    ]


def test_same_spelled_defaults_of_unrelated_declarations_are_not_one_setting(
    tmp_path: Path,
) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/spec.py": (
                "import didactic.api as dx\n"
                "class TrainingSpec(dx.Model):\n"
                "    seed: int = 0\n"
                "class SamplingSpec(dx.Model):\n"
                "    seed: int = 0\n"
            ),
        },
    )

    assert duplicated_defaults(package) == []


def test_an_unused_public_function_is_reported(tmp_path: Path) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/_chart.py": (
                BASE["lofi/core/_chart.py"]
                + "\ndef spans_of(chart, length):\n    return spans_of(chart, length)\n"
            ),
        },
    )

    assert [finding.key for finding in unused_public_symbols(package)] == [
        "lofi.core._chart:spans_of"
    ]


def test_an_explicitly_exported_function_is_a_public_use(tmp_path: Path) -> None:
    package = planted(
        tmp_path,
        {
            "lofi/core/_chart.py": (
                BASE["lofi/core/_chart.py"]
                + "\n__all__ = ['Chart', 'fill', 'spans_of']\n"
                "def spans_of(chart, length):\n    return spans_of(chart, length)\n"
            ),
        },
    )

    assert unused_public_symbols(package) == []


def test_a_python_copy_of_a_native_function_is_reported(tmp_path: Path) -> None:
    package = planted(
        tmp_path,
        {
            "native/src/lib.rs": (
                "use pyo3::prelude::*;\n"
                "#[pyfunction]\nfn fill(length: usize) -> usize { length }\n"
                '#[pyfunction]\n#[pyo3(name = "cells")]\n'
                "pub fn cells_native(length: usize) -> usize { length }\n"
            ),
            "lofi/core/_bound.py": (
                "from lofi.core import _native\n"
                "def cells(length):\n    return _native.cells(length)\n"
            ),
        },
    )

    assert native_exports(tmp_path) == {"fill", "cells"}
    assert [finding.key for finding in native_duplicates(package)] == [
        "lofi.core._chart:fill"
    ]


def test_a_packaged_value_restating_a_default_or_a_fragment_is_reported(
    tmp_path: Path,
) -> None:
    write_tree(
        tmp_path / "lofi" / "configs",
        {
            "data/datasets/train/dataset/grammar/cfg-collapse.yaml": (
                PACKAGED_CONFIG_ROOT
                / "data/datasets/train/dataset/grammar/cfg-collapse.yaml"
            ).read_text(encoding="utf-8"),
            "data/datasets/train/dataset/synthetic.yaml": (
                PACKAGED_CONFIG_ROOT / "data/datasets/train/dataset/synthetic.yaml"
            ).read_text(encoding="utf-8"),
            "data/datasets/train/featurizer/unlabeled.yaml": (
                PACKAGED_CONFIG_ROOT / "data/datasets/train/featurizer/unlabeled.yaml"
            ).read_text(encoding="utf-8"),
            "optimizer/adamw.yaml": "kind: adamw\nweight_decay: 0.0\n",
            "run/cfg-collapse.yaml": (
                PACKAGED_CONFIG_ROOT / "run/cfg-collapse.yaml"
            ).read_text(encoding="utf-8"),
            "run/base.yaml": (PACKAGED_CONFIG_ROOT / "run/synthetic-parser.yaml")
            .read_text(encoding="utf-8")
            .replace(
                "  active_groups:",
                "  max_grad_norm: null\n  active_groups:",
            ),
            "run/child.yaml": (
                "defaults:\n  - base\ntrainer:\n  kind: parser\n  epochs: 2\n"
            ),
        },
    )
    write_tree(tmp_path, BASE)
    keys = {finding.key for finding in restated_packaged_values(Package(tmp_path))}

    assert "lofi/configs/run/child.yaml:trainer.epochs" in keys
    assert "lofi/configs/run/child.yaml:trainer.kind" in keys
    assert "lofi/configs/run/base.yaml:trainer.max_grad_norm" in keys
    assert not any(key.startswith("lofi/configs/optimizer/") for key in keys)

"""No component branches on a name its declaration supplies.

A component reads operator, category, connective, quantifier, rule and
combinator names from the declaration it is given. Comparing a declared name
with a literal, looking an operator up by a literal name, matching a name
against a literal case, or keeping a table keyed by literal names makes the
component work for the packaged declarations and for no other, so the scan
refuses each of them in any module under ``lofi/`` outside the packaged
declaration data.

The names the scan knows are read from the packaged declarations themselves:
the logical-form languages ``lofi.spec.language`` ships, the rule kinds the
rule declaration union registers, and every primitive, operator and constructor a
packaged configuration document declares. A name made only of punctuation,
such as a slash spelling, is left out, since the same characters separate
paths and escape text and a scan of the literal cannot tell the uses apart.

A comparison counts when one side is a literal name and another reads a
declared name: an attribute or variable called ``name``, ``op``, ``rule``,
``category`` or the like, a ``"rule"`` or ``"name"`` entry of a declaration
mapping, or a local bound from one of these, as ``parts`` is by
``parts = node.rule.split(":")``. A prefix or suffix test on a declared name
counts the same way. A module-level constant bound to literal names stands
for its literals; one made only of declared names is a fixed vocabulary, so
defining it, testing membership in it and set arithmetic with it each count
whatever the other side reads. A class field whose default is a declared
name counts too, since a declaration that omits it takes the packaged name.

Two further readings of a declaration are refused. Comparing a text property
of a term operator with a literal, such as a decoder-head family or a payload
sort, branches on what one declaration happens to call its operators; the
properties are the keyword parameters of the term operator's constructor that
default to a string. Indexing a signature's sorts by position, as in
``signature.sorts[0]``, picks a sort by where one declaration lists it; the
sort sequences are the constructor parameters of the signature typed as a
sequence of sorts. The term package that declares the operator and the
signature defines what those properties mean, so its own reading of them is
not scanned for either form.
"""

import ast
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
from tests.declarations import packaged_names

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

TREE: Final = Path(__file__).resolve().parents[2]
SOURCE: Final = TREE / "lofi"

PACKAGED_DECLARATIONS: Final = (
    "lofi/spec/language/_standard.py",
    "lofi/run/_benchmark_fixture.py",
)
"""Modules that are declaration data rather than components.

The shipped logical-form languages and the benchmark grammar.
"""

CODEC_MODULES: Final = (
    "_codec.py",
    "_auto.py",
    "_auto_parse.py",
    "_parg.py",
    "_parg_parse.py",
    "_drs.py",
    "_translate.py",
    "roundtrip.py",
)
"""Corpus modules that translate one external format, by module name.

A codec, and the Layers translation, declares the fixed vocabulary of its
format; every other corpus module is a component like any other.
"""

READ_BY_NAME: Final = Counter(
    {
        ("lofi/data/synthetic/_semantics.py", "<module>", "exists"): 1,
        ("lofi/data/synthetic/_semantics.py", "<module>", "forall"): 1,
        ("lofi/data/synthetic/spec/interpretation.py", "InterpretationSpec", "e"): 1,
        ("lofi/data/synthetic/spec/interpretation.py", "InterpretationSpec", "t"): 1,
        (
            "lofi/judgment/_spec.py",
            "BoundedResponseSpec._the_predictor_is_declared_once",
            "logit",
        ): 1,
        (
            "lofi/judgment/_spec.py",
            "ResponseSpec._a_rank_consistent_head_has_one_identified_scalar_link",
            "logit",
        ): 1,
        ("lofi/judgment/linear.py", "LinearLinking.__init__", "linear"): 1,
        ("lofi/judgment/morcela.py", "MorcelaLinking.__init__", "morcela"): 1,
        ("lofi/judgment/runtime.py", "QuiversRuntime.load", "LKJCholesky"): 1,
        ("lofi/judgment/runtime.py", "_clamped_sites", "LKJCholesky"): 1,
        ("lofi/judgment/runtime.py", "_drawn", "LKJCholesky"): 1,
        ("lofi/spec/_rule.py", "ClausalAdjunctSpec", "NP"): 1,
        ("lofi/spec/_rule.py", "ClausalAdjunctSpec", "S"): 1,
        ("lofi/spec/_rule.py", "CombinatorSetSpec", "N"): 1,
        ("lofi/spec/_rule.py", "CombinatorSetSpec", "S"): 1,
        ("lofi/spec/_rule.py", "CoordinationSpec", "conj"): 1,
        ("lofi/spec/_rule.py", "TypeRaisingSpec", "N"): 1,
        ("lofi/spec/_rule.py", "TypeRaisingSpec", "S"): 1,
        ("lofi/spec/grammar/_grammar.py", "RootGoalSpec", "S"): 1,
        (
            "lofi/spec/grammar/_signature.py",
            "SignatureSpec._declarations_are_not_redundant",
            "primitive",
        ): 1,
        ("lofi/spec/language/_decl.py", "LanguageDecl", "eval"): 1,
        ("lofi/spec/language/_decl.py", "LanguageDecl", "lam"): 1,
    }
)
"""Sites that still read the packaged logic, rule kinds or categories by name.

They include the sites that test an operator's text property against a
literal or pick a signature's sort by position. Each key is a module, the
function or class body holding the site, and the literal read there (a
positional lookup reads as ``sorts[0]``); each count is how many sites of
that function read it. A site leaves this table when its component reads the
name from the declaration it is given. A site added anywhere, a listed
function included, fails the scan.
"""

DECLARED_ATTRIBUTES: Final = frozenset(
    {
        "categories",
        "category",
        "connective",
        "connectives",
        "constructor",
        "name",
        "op",
        "operator",
        "primitives",
        "quantifier",
        "quantifiers",
        "rule",
    }
)
"""Attributes whose value is a declared name or a collection of them."""

DECLARED_VARIABLES: Final = frozenset(
    {
        "category",
        "connective",
        "constructor",
        "name",
        "op",
        "operator",
        "quantifier",
        "rule",
        "symbol",
    }
)
"""Variable names that hold a declared name."""

DECLARED_KEYS: Final = frozenset({"name", "op", "rule"})
"""Keys under which a declaration mapping holds a declared name."""

LOOKUPS: Final = frozenset({"by_name", "get", "index", "pop"})
"""Methods that look an entry up by the name they are given."""

AFFIXES: Final = frozenset({"endswith", "removeprefix", "removesuffix", "startswith"})
"""String methods that test or strip a name's prefix or suffix."""

EXPORTS: Final = frozenset({"__all__", "__slots__", "__match_args__"})
"""Module-level names whose strings name Python attributes, not declared names."""

TERM_VALUES: Final = SOURCE / "terms" / "_values.py"
"""The module declaring the term operator and the signature."""


def constructor_parameters(path: Path, owner: str) -> tuple[ast.arguments, ...]:
    """Return the parameters of a class's constructor in a module.

    Parameters
    ----------
    path : Path
        The module.
    owner : str
        The class.
    """
    return tuple(
        method.args
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == owner
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "__init__"
    )


def operator_text_fields(path: Path) -> frozenset[str]:
    """Return the term operator's keyword parameters that default to a string.

    Parameters
    ----------
    path : Path
        The module declaring the term operator.
    """
    return frozenset(
        parameter.arg
        for arguments in constructor_parameters(path, "Operator")
        for parameter, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        )
        if isinstance(default, ast.Constant) and isinstance(default.value, str)
    )


def sort_sequences(path: Path) -> frozenset[str]:
    """Return the signature's constructor parameters typed as a sequence of sorts.

    Parameters
    ----------
    path : Path
        The module declaring the signature.
    """
    return frozenset(
        parameter.arg
        for arguments in constructor_parameters(path, "Signature")
        for parameter in (*arguments.args, *arguments.kwonlyargs)
        if parameter.annotation is not None
        and ast.unparse(parameter.annotation) == "Sequence[Sort]"
    )


TEXT_FIELDS: Final = operator_text_fields(TERM_VALUES)
"""The text properties of a term operator."""

SORT_SEQUENCES: Final = sort_sequences(TERM_VALUES)
"""The properties of a signature that list sorts in declaration order."""

KERNEL: Final = f"{TERM_VALUES.parent.relative_to(TREE).as_posix()}/"
"""The package that declares the term operator and the signature."""


class Site(NamedTuple):
    """One place a module reads declared names as literals.

    Attributes
    ----------
    path : str
        The module, relative to the repository root.
    line : int
        The line of the site.
    scope : str
        The dotted function and class path holding the site, or
        ``"<module>"`` at module level.
    form : str
        What the site does with the names.
    names : frozenset[str]
        The literal declared names it reads.
    """

    path: str
    line: int
    scope: str
    form: str
    names: frozenset[str]

    def describe(self) -> str:
        """Return the site as a ``path:line`` report."""
        listed = ", ".join(repr(name) for name in sorted(self.names))

        return f"{self.path}:{self.line}: {self.form} against {listed} in {self.scope}"


def packaged(path: str) -> bool:
    """Return whether a module is declaration data rather than a component.

    Parameters
    ----------
    path : str
        The module, relative to the repository root.

    Returns
    -------
    bool
        Whether the module is listed among the packaged declarations, or is
        a law suite, whose fixtures declare their own signatures.
    """
    name = Path(path).name

    return (
        name == "laws.py"
        or name.endswith("_laws.py")
        or path in PACKAGED_DECLARATIONS
        or (path.startswith("lofi/data/corpora/") and name in CODEC_MODULES)
    )


class _Scanner(ast.NodeVisitor):
    """Collects the sites of one module that read declared names as literals."""

    def __init__(self, path: str, names: frozenset[str], tree: ast.Module) -> None:
        self._path = path
        self._names = names
        self._scope: list[str] = []
        self._held: list[frozenset[str]] = [frozenset()]
        self._classes: list[int] = []
        self._constants = self._module_constants(tree)
        self._vocabularies = frozenset(
            name
            for name, strings in self._constants.items()
            if strings and strings <= names
        )
        self.sites: list[Site] = []
        self._record_vocabularies(tree)

    def _module_constants(self, tree: ast.Module) -> dict[str, frozenset[str]]:
        """Return the module-level names bound to literal strings."""
        constants: dict[str, frozenset[str]] = {}

        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets, value = [statement.target], statement.value
            else:
                continue

            held = self._strings(value, {})

            for target in targets:
                if isinstance(target, ast.Name) and held:
                    constants[target.id] = held

        return constants

    def _strings(
        self, node: ast.expr, constants: Mapping[str, frozenset[str]]
    ) -> frozenset[str]:
        """Return the literal strings an expression denotes."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return frozenset({node.value})

        if isinstance(node, ast.Tuple | ast.List | ast.Set):
            return frozenset(
                text for item in node.elts for text in self._strings(item, constants)
            )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"frozenset", "set", "tuple"}
            and len(node.args) == 1
        ):
            return self._strings(node.args[0], constants)

        if isinstance(node, ast.Name):
            return constants.get(node.id, frozenset())

        return frozenset()

    def _record_vocabularies(self, tree: ast.Module) -> None:
        """Record each module-level collection made only of declared names."""
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets, value = [statement.target], statement.value
            else:
                continue

            bound = [
                target.id
                for target in targets
                if isinstance(target, ast.Name) and target.id not in EXPORTS
            ]
            collection = isinstance(value, ast.Tuple | ast.List | ast.Set) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {"frozenset", "set", "tuple"}
            )

            if collection and any(name in self._vocabularies for name in bound):
                self._record(statement, "a vocabulary", self._declared(value))

    def _vocabulary(self, node: ast.expr) -> bool:
        """Return whether an expression names a vocabulary of declared names."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"frozenset", "set", "sorted", "tuple"}
            and len(node.args) == 1
        ):
            return self._vocabulary(node.args[0])

        return isinstance(node, ast.Name) and node.id in self._vocabularies

    def _holding(self, body: list[ast.stmt]) -> frozenset[str]:
        """Return the local names a body binds from a declared name.

        A local bound from an expression that reads a declared name holds
        one too, so ``parts = node.rule.split(":")`` makes ``parts`` read
        the rule's name.
        """
        held = set(self._held[-1])
        bindings: list[tuple[ast.expr, ast.expr]] = []

        for statement in body:
            for node in ast.walk(statement):
                match node:
                    case ast.Assign(targets=targets, value=value):
                        bindings.extend((target, value) for target in targets)
                    case (
                        ast.AnnAssign(target=target, value=ast.expr() as value)
                        | ast.AugAssign(target=target, value=value)
                        | ast.NamedExpr(target=target, value=value)
                        | ast.For(target=target, iter=value)
                        | ast.AsyncFor(target=target, iter=value)
                        | ast.comprehension(target=target, iter=value)
                    ):
                        bindings.append((target, value))

        changed = True

        while changed:
            changed = False

            for target, value in bindings:
                if not self._derives(value, frozenset(held)):
                    continue

                for part in ast.walk(target):
                    if isinstance(part, ast.Name) and part.id not in held:
                        held.add(part.id)
                        changed = True

        return frozenset(held)

    def _derives(self, node: ast.expr, held: frozenset[str]) -> bool:
        """Return whether an expression's value is derived from a declared name.

        The value is followed through attributes, subscripts and method
        calls to the object they start from; a call's arguments, such as the
        parts of a message, do not pass a name on.
        """
        match node:
            case ast.Name(id=name):
                return name in DECLARED_VARIABLES or name in held
            case ast.Attribute(attr=attribute, value=owner):
                return attribute in DECLARED_ATTRIBUTES or self._derives(owner, held)
            case ast.Subscript(slice=ast.Constant(value=key)) if key in DECLARED_KEYS:
                return True
            case ast.Subscript(value=owner) | ast.Call(func=owner):
                return self._derives(owner, held)
            case ast.Tuple(elts=items) | ast.List(elts=items):
                followed = list(items)
            case ast.IfExp(body=body, orelse=orelse):
                followed = [body, orelse]
            case _:
                followed = []

        return any(self._derives(item, held) for item in followed)

    def _declared(self, node: ast.expr) -> frozenset[str]:
        """Return the declared names an expression denotes as literals."""
        return self._strings(node, self._constants) & self._names

    def _reads_declared_name(
        self, node: ast.expr, held: frozenset[str] | None = None
    ) -> bool:
        """Return whether an expression reads a declared name."""
        local = self._held[-1] if held is None else held

        for part in ast.walk(node):
            if isinstance(part, ast.Attribute) and part.attr in DECLARED_ATTRIBUTES:
                return True

            if isinstance(part, ast.Name) and (
                part.id in DECLARED_VARIABLES or part.id in local
            ):
                return True

            if (
                isinstance(part, ast.Subscript)
                and isinstance(part.slice, ast.Constant)
                and part.slice.value in DECLARED_KEYS
            ):
                return True

            if (
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr == "get"
                and part.args
                and isinstance(part.args[0], ast.Constant)
                and part.args[0].value in DECLARED_KEYS
            ):
                return True

        return False

    def _record(
        self, node: ast.expr | ast.stmt, form: str, names: frozenset[str]
    ) -> None:
        """Record one site."""
        self.sites.append(
            Site(
                self._path,
                node.lineno,
                ".".join(self._scope) or "<module>",
                form,
                names,
            )
        )

    def _scoped(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        """Visit a body under its name and the locals it binds from names."""
        self._scope.append(node.name)

        if isinstance(node, ast.ClassDef):
            self._classes.append(len(self._scope))

        self._held.append(
            self._holding(node.body)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            else self._held[-1]
        )
        self.generic_visit(node)
        self._held.pop()

        if isinstance(node, ast.ClassDef):
            self._classes.pop()

        self._scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function body under its name."""
        self._scoped(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit a coroutine body under its name."""
        self._scoped(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class body under its name."""
        self._scoped(node)

    @override
    def visit_Compare(self, node: ast.Compare) -> None:
        """Record a comparison of a declared name with a literal one."""
        sides = (node.left, *node.comparators)
        literal = frozenset(name for side in sides for name in self._declared(side))
        others = (side for side in sides if not self._declared(side))

        if literal and (
            any(self._vocabulary(side) for side in sides)
            or any(self._reads_declared_name(side) for side in others)
        ):
            self._record(node, "a comparison", literal)

        if not self._path.startswith(KERNEL):
            for side in sides:
                if isinstance(side, ast.Attribute) and side.attr in TEXT_FIELDS:
                    texts = frozenset(
                        text
                        for other in sides
                        if other is not side
                        for text in self._strings(other, self._constants)
                    )

                    if texts:
                        self._record(node, f"a {side.attr} test", texts)

        self.generic_visit(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Record a class field whose default is a declared name.

        A declaration that omits such a field silently takes a packaged
        name. A ``Literal`` field, such as a variant's discriminator, names
        its own tag rather than a declared one.
        """
        in_class = bool(self._classes) and self._classes[-1] == len(self._scope)
        annotation = node.annotation
        tagged = (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "Literal"
        )

        if in_class and not tagged and node.value is not None:
            default = node.value

            if isinstance(default, ast.Call):
                default = next(
                    (kw.value for kw in default.keywords if kw.arg == "default"),
                    default,
                )

            literal = self._declared(default)

            if literal:
                self._record(node, "a default", literal)

        self.generic_visit(node)

    @override
    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Record set arithmetic with a vocabulary of declared names."""
        sides = (node.left, node.right)

        if any(self._vocabulary(side) for side in sides):
            self._record(
                node,
                "set arithmetic",
                frozenset(name for side in sides for name in self._declared(side)),
            )

        self.generic_visit(node)

    @override
    def visit_Match(self, node: ast.Match) -> None:
        """Record a match of a declared name against literal cases."""
        if self._reads_declared_name(node.subject):
            literal = frozenset(
                name
                for case in node.cases
                for pattern in ast.walk(case.pattern)
                if isinstance(pattern, ast.MatchValue)
                for name in self._declared(pattern.value)
            )

            if literal:
                self._record(node, "a match", literal)

        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """Record a lookup by a literal name."""
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in LOOKUPS
            and node.args
            and isinstance(node.args[0], ast.Constant | ast.Name)
        ):
            literal = self._declared(node.args[0])

            if literal:
                self._record(node, f"a {node.func.attr} lookup", literal)

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in AFFIXES
            and node.args
            and self._reads_declared_name(node.func.value)
        ):
            literal = self._declared(node.args[0])

            if literal:
                self._record(node, f"a {node.func.attr} test", literal)

        self.generic_visit(node)

    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Record an index by a literal name, outside a ``Literal`` annotation."""
        annotation = isinstance(node.value, ast.Name) and node.value.id == "Literal"

        if not annotation and isinstance(node.slice, ast.Constant | ast.Name):
            literal = self._declared(node.slice)

            if literal:
                self._record(node, "an index", literal)

        index: int | None = None
        position = node.slice

        if isinstance(position, ast.UnaryOp) and isinstance(position.op, ast.USub):
            position = position.operand
            sign = -1
        else:
            sign = 1

        if (
            isinstance(position, ast.Constant)
            and isinstance(position.value, int)
            and not isinstance(position.value, bool)
        ):
            index = sign * position.value

        if (
            not self._path.startswith(KERNEL)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in SORT_SEQUENCES
            and index is not None
        ):
            self._record(
                node,
                "a positional sort lookup",
                frozenset({f"{node.value.attr}[{index}]"}),
            )

        self.generic_visit(node)

    @override
    def visit_Dict(self, node: ast.Dict) -> None:
        """Record a table keyed only by literal names."""
        keys = [self._declared(key) for key in node.keys if key is not None]
        declared = [key for key in node.keys if key is not None]

        if declared and all(
            literal and literal == self._strings(key, self._constants)
            for literal, key in zip(keys, declared, strict=True)
        ):
            self._record(node, "a table", frozenset().union(*keys))

        self.generic_visit(node)


def scan(path: str, source: str, names: frozenset[str]) -> tuple[Site, ...]:
    """Return the sites of one module that read declared names as literals.

    Parameters
    ----------
    path : str
        The module, relative to the repository root.
    source : str
        The module's text.
    names : frozenset[str]
        The declared names to look for.
    """
    tree = ast.parse(source)
    scanner = _Scanner(path, names, tree)
    scanner.visit(tree)

    return tuple(scanner.sites)


def components() -> tuple[tuple[str, str], ...]:
    """Return every component module with its text, in path order."""
    modules = sorted(
        path
        for path in SOURCE.rglob("*.py")
        if "__pycache__" not in path.parts
        and not packaged(path.relative_to(TREE).as_posix())
    )

    if not modules:
        pytest.fail(f"no module found under {SOURCE}, so nothing was scanned")

    return tuple(
        (path.relative_to(TREE).as_posix(), path.read_text(encoding="utf-8"))
        for path in modules
    )


def tally(sites: Iterable[Site]) -> Counter[tuple[str, str, str]]:
    """Return how many sites of each function read each name."""
    return Counter(
        (site.path, site.scope, name) for site in sites for name in site.names
    )


NAMES: Final = packaged_names()
COMPONENTS: Final = components()

PLANTED: Final = (
    '\n\ndef _planted(op):\n    if op.name == "forall":\n        return op\n'
)


def test_text_fields_and_sort_sequences_come_from_the_term_module() -> None:
    assert {"family", "payload_sort"} <= TEXT_FIELDS
    assert "name" not in TEXT_FIELDS
    assert "sorts" in SORT_SEQUENCES


def test_the_scanned_names_cover_the_packaged_logic_and_rules() -> None:
    assert {"forall", "not", "and", "S", "NP", "application"} <= NAMES
    assert not any(name in NAMES for name in ("/", "\\", "->"))


def test_no_component_reads_a_declared_name_as_a_literal() -> None:
    sites = tuple(
        site for path, source in COMPONENTS for site in scan(path, source, NAMES)
    )
    found = tally(sites)
    added = found - READ_BY_NAME
    removed = READ_BY_NAME - found
    offending = [
        site.describe()
        for site in sites
        if any((site.path, site.scope, name) in added for name in site.names)
    ]

    assert not added, "components read declared names as literals:\n" + "\n".join(
        offending
    )
    assert not removed, (
        "these sites no longer read names; drop them from the table:\n"
        + (
            "\n".join(
                f"{path} {scope} {name!r}" for path, scope, name in sorted(removed)
            )
        )
    )


def test_a_name_branch_planted_in_any_component_is_reported_at_its_line() -> None:
    unreported = []

    for path, source in COMPONENTS:
        planted = source + PLANTED
        line = planted.count("\n", 0, planted.index('if op.name == "forall"')) + 1
        expected = f"{path}:{line}: a comparison against 'forall' in _planted"

        if expected not in {site.describe() for site in scan(path, planted, NAMES)}:
            unreported.append(expected)

    assert not unreported, "planted branches went unreported:\n" + "\n".join(unreported)


@pytest.mark.parametrize(
    ("source", "forms"),
    [
        (
            (
                'def f(term):\n    match term.op.name:\n        case "not":\n'
                "            pass\n"
            ),
            ["a match"],
        ),
        (
            'def f(signature):\n    return signature.by_name("exists")\n',
            ["a by_name lookup"],
        ),
        (
            '_RULE = "composition"\ndef f(spec):\n    return spec["rule"] == _RULE\n',
            ["a comparison"],
        ),
        (
            (
                '_PAIR = frozenset(("and", "or"))\n'
                "def f(value):\n    return value in _PAIR\n"
            ),
            ["a vocabulary", "a comparison"],
        ),
        (
            (
                '_BOUND = ("forall", "exists")\n'
                "def f(held):\n    return held - set(_BOUND)\n"
            ),
            ["a vocabulary", "set arithmetic"],
        ),
        (
            (
                'def f(node):\n    parts = node.rule.split(":")\n'
                '    return parts[0] == "application"\n'
            ),
            ["a comparison"],
        ),
        (
            'def f(node):\n    return node.rule.startswith("composition")\n',
            ["a startswith test"],
        ),
        (
            (
                'def f(spec):\n    label = spec["rule"]\n'
                '    return label == "application"\n'
            ),
            ["a comparison"],
        ),
        (
            (
                "def f(node, flag):\n"
                '    label = node.rule if flag else ""\n'
                '    return label == "application"\n'
            ),
            ["a comparison"],
        ),
        (
            'def f(spec):\n    return spec.get("rule") == "composition"\n',
            ["a comparison"],
        ),
        (
            (
                "def f(steps):\n    previous = None\n    for step in steps:\n"
                '        if previous == "application":\n            return step\n'
                "        previous = parts[0]\n"
                '        parts = step.rule.split(":")\n'
            ),
            ["a comparison"],
        ),
        (
            '_KEY = "exists"\ndef f(signature):\n    return signature.by_name(_KEY)\n',
            ["a by_name lookup"],
        ),
        (
            'class Goal:\n    category: str = field(default="S")\n',
            ["a default"],
        ),
        ('TABLE = {"forall": 1, "exists": 2}\n', ["a table"]),
        (
            'class Goal:\n    categories: tuple[str, ...] = ("S",)\n',
            ["a default"],
        ),
        ('def f(table):\n    return table["iff"]\n', ["an index"]),
        (
            "def f(store):\n    return store.hole(store.signature.sorts[0])\n",
            ["a positional sort lookup"],
        ),
        (
            'def f(op):\n    return op.payload_sort == "constant"\n',
            ["a payload_sort test"],
        ),
        (
            '_VARIABLE = "variable"\ndef f(op):\n    return op.family != _VARIABLE\n',
            ["a family test"],
        ),
        (
            'def f(op):\n    return op.payload_sort not in {"none", "index"}\n',
            ["a payload_sort test"],
        ),
    ],
)
def test_each_form_of_reading_a_name_is_reported(source: str, forms: list[str]) -> None:
    sites = scan("lofi/planted.py", source, NAMES)

    assert [site.form for site in sites] == forms


@pytest.mark.parametrize(
    "source",
    [
        'kind: Literal["application"] = "application"\n',
        'class Rule:\n    kind: Literal["application"] = "application"\n',
        'def f(path):\n    return path.split("/")\n',
        'def f(record):\n    return record.kind == "eval"\n',
        'LABELS = {"e": "entails", "c": "contradicts"}\n',
        'def f(op):\n    return op.name == "frobnicate"\n',
        (
            "def f(name, data):\n"
            '    kind = data.get("kind", f"in {name}")\n'
            '    return kind == "eval"\n'
        ),
        '__all__ = ["S", "NP"]\n',
        (
            "def f(name, describe):\n"
            "    kind = describe(name)\n"
            '    return kind == "eval"\n'
        ),
    ],
)
def test_reading_other_text_is_not_reported(source: str) -> None:
    assert scan("lofi/planted.py", source, NAMES) == ()


def test_packaged_declarations_and_law_suites_are_not_components() -> None:
    assert packaged("lofi/spec/language/_standard.py")
    assert packaged("lofi/data/corpora/megaacceptability/_codec.py")
    assert packaged("lofi/data/corpora/layers/_translate.py")
    assert not packaged("lofi/data/corpora/ccgbank/spec.py")
    assert not packaged("lofi/data/corpora/entailment_graph/_generator.py")
    assert packaged("lofi/spec/_raising_laws.py")
    assert packaged("lofi/terms/laws.py")
    assert not packaged("lofi/lf/_propositional.py")


ITEM_RECORDS: Final = frozenset({"item_metadata", "rendered_elements", "slot_fillers"})
"""The item-record mappings whose keys a document declares."""


def literal_record_keys(path: str, source: str) -> tuple[str, ...]:
    """Return each site reading a declared item-record key as a literal.

    A component reads an item's attributes, its elements or a filling's slots
    only by the names a document declares, so a string literal indexing one of
    them, or passed to its ``get``, or a bead expression literal naming an item
    attribute outside a docstring, is reported.
    """
    tree = ast.parse(source)
    documentation = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    found: list[str] = []
    for node in ast.walk(tree):
        if id(node) in documentation:
            continue
        match node:
            case ast.Subscript(
                value=ast.Attribute(attr=attribute),
                slice=ast.Constant(value=str() as key),
            ) if attribute in ITEM_RECORDS:
                found.append(f"{path}:{node.lineno}: {attribute}[{key!r}]")
            case ast.Call(
                func=ast.Attribute(attr="get", value=ast.Attribute(attr=attribute)),
                args=[ast.Constant(value=str() as key), *_],
            ) if attribute in ITEM_RECORDS:
                found.append(f"{path}:{node.lineno}: {attribute}.get({key!r})")
            case ast.Constant(value=str() as text) if "item['" in text or (
                'item["' in text
            ):
                found.append(f"{path}:{node.lineno}: expression {text!r}")
            case _:
                pass
    return tuple(found)


def test_no_component_reads_an_item_record_key_as_a_literal() -> None:
    sites = tuple(
        site
        for path, source in COMPONENTS
        for site in literal_record_keys(path, source)
    )

    assert not sites, "components read item-record keys as literals:\n" + "\n".join(
        sites
    )


@pytest.mark.parametrize(
    "planted",
    [
        "x = item.item_metadata['verb']",
        "x = item.rendered_elements['sentence']",
        "x = item.item_metadata.get('frame')",
        "x = filled.slot_fillers['predicate']",
        "x = parse(f\"item['{name}']\")",
        "rule = \"item['structure']\"",
    ],
)
def test_a_planted_item_record_literal_is_reported(planted: str) -> None:
    assert literal_record_keys("lofi/planted.py", planted)


def test_a_declared_item_record_key_is_not_reported() -> None:
    assert not literal_record_keys(
        "lofi/planted.py", "x = item.item_metadata[name]\ny = filled.slot_fillers[slot]"
    )

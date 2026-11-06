"""The numpydoc type slots copy the signature, and are checked against it."""

import ast
import pathlib
import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "lofi"

type Function = ast.FunctionDef | ast.AsyncFunctionDef
type Subject = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
type Documented = tuple[str, str, Subject, str]
type Declaration = tuple[str, int, str, str, str]
type BareSlot = tuple[str, int, str, str]


def _entries(lines: list[str], indent: int) -> list[str]:
    """Return the lines at one indentation until the section ends.

    Parameters
    ----------
    lines : list[str]
        The lines following a section underline.
    indent : int
        The column the section heading starts at.

    Returns
    -------
    list[str]
        One line per entry, stripped of its indentation and of its description.
    """
    found = []

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        column = len(line) - len(line.lstrip())

        if column < indent:
            break

        if column > indent:
            continue

        if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            break

        found.append(line.strip())

    return found


def _section(doc: str, heading: str) -> list[str]:
    """Return the entry lines of one numpydoc section.

    Parameters
    ----------
    doc : str
        The docstring to read, with its indentation intact.
    heading : str
        The heading of the section to read.

    Returns
    -------
    list[str]
        One line per entry, empty when the docstring has no such section.
    """
    lines = doc.splitlines()

    for i, line in enumerate(lines[:-2]):
        if line.strip() != heading or set(lines[i + 1].strip()) != {"-"}:
            continue

        return _entries(lines[i + 2 :], len(line) - len(line.lstrip()))

    return []


def _returns_slot(doc: str) -> str | None:
    """Return the type in a numpydoc Returns block, or None if it is absent or empty.

    Parameters
    ----------
    doc : str
        The docstring to read, with its indentation intact.
    """
    entries = _section(doc, "Returns")

    if not entries:
        return None

    slot = entries[0]

    return slot.split(" : ", 1)[1].strip() if " : " in slot else slot


def _named_slots(doc: str, heading: str) -> list[tuple[str, str | None]]:
    """Return the name and declared type of every entry in a named-entry section.

    Parameters
    ----------
    doc : str
        The docstring to read, with its indentation intact.
    heading : str
        The heading of the section to read, Parameters or Attributes.

    Returns
    -------
    list[tuple[str, str | None]]
        One pair per documented name: the name, with the leading stars of a variadic
        removed, and the type it declares or None where it declares none.
    """
    found = []

    for entry in _section(doc, heading):
        names, _, declared = entry.partition(" : ")

        found.extend(
            (name.strip().lstrip("*"), declared.strip() or None)
            for name in names.split(",")
        )

    return found


def _annotations(text: str, node: Function) -> dict[str, str | None]:
    """Return the annotation of each parameter a function takes, by name.

    Parameters
    ----------
    text : str
        The source the function was parsed from.
    node : Function
        The function to read.

    Returns
    -------
    dict[str, str | None]
        The annotation as written for each name, or None where it has none.
    """
    args = node.args
    taken = [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *([args.vararg] if args.vararg is not None else []),
        *([args.kwarg] if args.kwarg is not None else []),
    ]

    return {
        arg.arg: None
        if arg.annotation is None
        else ast.get_source_segment(text, arg.annotation)
        for arg in taken
    }


def _is_property(node: Function) -> bool:
    """Return whether a function is decorated as a property.

    Parameters
    ----------
    node : Function
        The function to read.
    """
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "property")
        or (isinstance(decorator, ast.Attribute) and decorator.attr == "property")
        for decorator in node.decorator_list
    )


def _members(text: str, node: ast.ClassDef) -> dict[str, str | None]:
    """Return the annotation of each attribute a class body declares, by name.

    An attribute is declared by an annotated assignment, or by a property, whose
    return annotation is the attribute's.

    Parameters
    ----------
    text : str
        The source the class was parsed from.
    node : ast.ClassDef
        The class to read.

    Returns
    -------
    dict[str, str | None]
        The annotation as written for each name, or None where the source it was
        parsed from does not hold it.
    """
    found: dict[str, str | None] = {}

    for member in node.body:
        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            found[member.target.id] = ast.get_source_segment(text, member.annotation)
        elif (
            isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
            and member.returns is not None
            and _is_property(member)
        ):
            found[member.name] = ast.get_source_segment(text, member.returns)

    return found


def _signature(node: Subject) -> Function | None:
    """Return the function whose parameters a docstring documents, if there is one.

    Parameters
    ----------
    node : Subject
        The function or class the docstring belongs to.

    Returns
    -------
    Function | None
        The function itself, the initializer a class body defines, or None for a class
        that defines no initializer of its own.
    """
    if not isinstance(node, ast.ClassDef):
        return node

    for member in node.body:
        if (
            isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
            and member.name == "__init__"
        ):
            return member

    return None


def _documented() -> list[Documented]:
    """Return every function and class in the package that has a docstring.

    Returns
    -------
    list[Documented]
        The path, the source it was parsed from, the node and its docstring.
    """
    found = []

    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)

        for node in ast.walk(tree):
            if not isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                continue

            doc = ast.get_docstring(node, clean=False)

            if doc:
                found.append((str(path), text, node, doc))

    return found


def _return_declarations(documented: list[Documented]) -> list[Declaration]:
    """Return every function whose docstring declares a return type, with that type.

    Parameters
    ----------
    documented : list[Documented]
        The functions and classes to read, of which a class declares no return.

    Returns
    -------
    list[Declaration]
        The path, the line, the function's name, the declared type and the annotation.
    """
    found = []

    for path, text, node, doc in documented:
        if isinstance(node, ast.ClassDef) or node.returns is None:
            continue

        annotation = ast.get_source_segment(text, node.returns)

        if annotation is None or annotation == "None":
            continue

        declared = _returns_slot(doc)

        if declared is None:
            continue

        found.append((path, node.lineno, node.name, declared, annotation))

    return found


def _parameter_declarations(documented: list[Documented]) -> list[Declaration]:
    """Return every documented parameter that declares a type, with that type.

    A class docstring documents the parameters of the initializer its body defines.

    Parameters
    ----------
    documented : list[Documented]
        The functions and classes to read.

    Returns
    -------
    list[Declaration]
        The path, the line, the function or class and parameter named together, the
        declared type and the annotation.
    """
    found = []

    for path, text, node, doc in documented:
        slots = _named_slots(doc, "Parameters")
        signature = _signature(node)

        if not slots or signature is None:
            continue

        annotations = _annotations(text, signature)

        for parameter, declared in slots:
            annotation = annotations.get(parameter)

            if declared is None or annotation is None:
                continue

            site = f"{node.name}({parameter})"

            found.append((path, node.lineno, site, declared, annotation))

    return found


def _attribute_declarations(documented: list[Documented]) -> list[Declaration]:
    """Return every documented attribute that declares a type, with that type.

    Parameters
    ----------
    documented : list[Documented]
        The functions and classes to read, of which a function has no attributes.

    Returns
    -------
    list[Declaration]
        The path, the line, the class and attribute named together, the declared type
        and the annotation.
    """
    found = []

    for path, text, node, doc in documented:
        if not isinstance(node, ast.ClassDef):
            continue

        members = _members(text, node)

        for attribute, declared in _named_slots(doc, "Attributes"):
            annotation = members.get(attribute)

            if declared is None or annotation is None:
                continue

            site = f"{node.name}.{attribute}"

            found.append((path, node.lineno, site, declared, annotation))

    return found


def _bare_slots(documented: list[Documented]) -> list[BareSlot]:
    """Return Parameters and Attributes entries with no declared type.

    Parameters
    ----------
    documented : list[Documented]
        Functions and classes to inspect.
    """
    found: list[BareSlot] = []

    for path, _, node, doc in documented:
        for heading in ("Parameters", "Attributes"):
            found.extend(
                (path, node.lineno, f"{node.name}.{name}", heading)
                for name, declared in _named_slots(doc, heading)
                if declared is None
            )

    return found


def _orphans(documented: list[Documented]) -> list[tuple[str, int, str, str]]:
    """Return every documented parameter the function it documents does not take.

    Parameters
    ----------
    documented : list[Documented]
        The functions and classes to read.

    Returns
    -------
    list[tuple[str, int, str, str]]
        The path, the line, the function's or class's name and the name documented
        under it.
    """
    found = []

    for path, text, node, doc in documented:
        slots = _named_slots(doc, "Parameters")
        signature = _signature(node)

        if not slots or signature is None:
            continue

        annotations = _annotations(text, signature)

        found.extend(
            (path, node.lineno, node.name, parameter)
            for parameter, _ in slots
            if parameter not in annotations
        )

    return found


def _flattened(text: str) -> str:
    """Return the type with its wrapping removed, so a slot compares to an annotation.

    A long annotation is broken across lines and gains a comma before the closing
    bracket, and a docstring slot written on one line repeats neither.

    Parameters
    ----------
    text : str
        The type as written.
    """
    return re.sub(r",(?=[)\]}])", "", re.sub(r"\s+", "", text))


DOCUMENTED = _documented()
FUNCTIONS = [entry for entry in DOCUMENTED if not isinstance(entry[2], ast.ClassDef)]
CLASSES = [entry for entry in DOCUMENTED if isinstance(entry[2], ast.ClassDef)]
DECLARATIONS = _return_declarations(FUNCTIONS)
CLASS_PARAMETERS = _parameter_declarations(CLASSES)
PARAMETERS = [*_parameter_declarations(FUNCTIONS), *CLASS_PARAMETERS]
ATTRIBUTES = _attribute_declarations(CLASSES)
ORPHANS = _orphans(DOCUMENTED)
BARE_SLOTS = _bare_slots(DOCUMENTED)


def test_the_tree_has_return_declarations_to_check() -> None:
    """A collection that finds nothing would pass every other test in this module."""
    assert len(DECLARATIONS) > 500


def test_the_tree_has_parameter_declarations_to_check() -> None:
    """A collection that finds nothing would pass every other test in this module."""
    assert len(PARAMETERS) > 500


def test_the_tree_has_class_parameter_declarations_to_check() -> None:
    """A collection that reads no class docstring would pass the function checks."""
    assert len(CLASS_PARAMETERS) > 200


def test_the_tree_has_attribute_declarations_to_check() -> None:
    """A collection that finds nothing would pass every other test in this module."""
    assert len(ATTRIBUTES) > 20


def test_a_section_ends_where_the_next_section_begins() -> None:
    """A heading below the block opens a section rather than naming a parameter."""
    doc = (
        "Summary.\n"
        "\n"
        "    Parameters\n"
        "    ----------\n"
        "    value : int\n"
        "        The value.\n"
        "\n"
        "    Returns\n"
        "    -------\n"
        "    str\n"
        "        The rendering.\n"
        "    "
    )

    assert _section(doc, "Parameters") == ["value : int"]
    assert _section(doc, "Returns") == ["str"]


def test_a_section_ends_where_the_docstring_dedents() -> None:
    """A dedent ends the block, and nothing below it is read back as a parameter."""
    doc = (
        "Summary.\n"
        "\n"
        "    Parameters\n"
        "    ----------\n"
        "    value : int\n"
        "        The value.\n"
        "\n"
        "Notes\n"
        "-----\n"
        "    A line the block does not reach.\n"
        "    "
    )

    assert _section(doc, "Parameters") == ["value : int"]


def test_a_wrapped_annotation_flattens_onto_its_slot() -> None:
    """An annotation ruff broke across lines equals the slot written on one."""
    union = "Expression\n    | Filling\n    | JudgmentSet"
    nested = "tuple[\n    ProjectedDependency,\n    tuple[LexicalHead, ...],\n]"

    assert _flattened(union) == _flattened("Expression | Filling | JudgmentSet")
    assert _flattened(nested) == _flattened(
        "tuple[ProjectedDependency, tuple[LexicalHead, ...]]"
    )


def _crafted(text: str) -> Documented:
    """Return the first statement of a source as a documented subject.

    Parameters
    ----------
    text : str
        The source of one function or class with a docstring.
    """
    node = ast.parse(text).body[0]
    assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    doc = ast.get_docstring(node, clean=False)
    assert doc

    return ("crafted.py", text, node, doc)


OPTIONS = _crafted(
    "class Options:\n"
    '    """Summary.\n'
    "\n"
    "    Parameters\n"
    "    ----------\n"
    "    rate : float | None\n"
    "        The rate.\n"
    "    name\n"
    "        The name.\n"
    "    scale : int\n"
    "        A parameter the initializer does not take.\n"
    "\n"
    "    Attributes\n"
    "    ----------\n"
    "    rate : float\n"
    "        The rate.\n"
    "    limit : int\n"
    "        The limit.\n"
    "    name\n"
    "        The name.\n"
    "    scale : int\n"
    "        An attribute the body does not declare.\n"
    "    tag : str\n"
    "        A method that is not a property.\n"
    '    """\n'
    "\n"
    "    limit: int = 3\n"
    "\n"
    "    def tag(self) -> str:\n"
    "        return self.name\n"
    "\n"
    "    def __init__(self, rate: float | None = None, name: str = '') -> None:\n"
    "        self._rate = 1.0 if rate is None else rate\n"
    "        self.name = name\n"
    "\n"
    "    @property\n"
    "    def rate(self) -> float:\n"
    "        return self._rate\n"
)

BARE = _crafted(
    "class Bare:\n"
    '    """Summary.\n'
    "\n"
    "    Parameters\n"
    "    ----------\n"
    "    x : int\n"
    "        A field, with no initializer to read it against.\n"
    '    """\n'
    "\n"
    "    x: int\n"
)


def test_a_class_documents_the_initializer_its_body_defines() -> None:
    """A class docstring's Parameters are read against __init__, typed ones only."""
    assert _parameter_declarations([OPTIONS]) == [
        ("crafted.py", 1, "Options(rate)", "float | None", "float | None")
    ]
    assert _orphans([OPTIONS]) == [("crafted.py", 1, "Options", "scale")]


def test_a_class_documents_the_attributes_its_body_declares() -> None:
    """An Attributes slot is read against an annotated name or a property, only."""
    assert _attribute_declarations([OPTIONS]) == [
        ("crafted.py", 1, "Options.rate", "float", "float"),
        ("crafted.py", 1, "Options.limit", "int", "int"),
    ]
    assert _return_declarations([OPTIONS]) == []


def test_bare_named_slots_are_detected() -> None:
    """A Parameters or Attributes entry without a type is reported."""
    assert _bare_slots([OPTIONS]) == [
        ("crafted.py", 1, "Options.name", "Parameters"),
        ("crafted.py", 1, "Options.name", "Attributes"),
    ]


def test_a_class_without_an_initializer_documents_no_parameters() -> None:
    """A class whose body defines no __init__ is checked against no signature."""
    assert _parameter_declarations([BARE]) == []
    assert _orphans([BARE]) == []
    assert _attribute_declarations([BARE]) == []


def test_a_documented_parameter_is_one_the_function_takes() -> None:
    """A name under Parameters that no signature holds is checked against nothing."""
    assert not ORPHANS, "\n".join(
        f"{path}:{lineno} {name}() documents {parameter!r}, which it does not take"
        for path, lineno, name, parameter in ORPHANS
    )


def test_every_named_slot_declares_its_exact_type() -> None:
    """Every Parameters and Attributes entry includes a type slot."""
    assert not BARE_SLOTS, "\n".join(
        f"{path}:{lineno} {site} has a bare {heading} entry"
        for path, lineno, site, heading in BARE_SLOTS
    )


@pytest.mark.parametrize(
    ("path", "lineno", "name", "declared", "annotation"),
    DECLARATIONS,
    ids=[f"{pathlib.Path(p).name}:{n}:{f}" for p, n, f, _, _ in DECLARATIONS],
)
def test_a_declared_return_type_copies_its_annotation(
    path: str, lineno: int, name: str, declared: str, annotation: str
) -> None:
    """The Returns slot holds the annotation verbatim, not a paraphrase of it."""
    assert _flattened(declared) == _flattened(annotation), (
        f"{path}:{lineno} {name}() declares {declared!r},"
        f" and is annotated {annotation!r}"
    )


@pytest.mark.parametrize(
    ("path", "lineno", "name", "declared", "annotation"),
    PARAMETERS,
    ids=[f"{pathlib.Path(p).name}:{n}:{f}" for p, n, f, _, _ in PARAMETERS],
)
def test_a_declared_parameter_type_copies_its_annotation(
    path: str, lineno: int, name: str, declared: str, annotation: str
) -> None:
    """A Parameters slot holds the annotation verbatim, not a paraphrase of it."""
    assert _flattened(declared) == _flattened(annotation), (
        f"{path}:{lineno} {name} declares {declared!r}, and is annotated {annotation!r}"
    )


@pytest.mark.parametrize(
    ("path", "lineno", "name", "declared", "annotation"),
    ATTRIBUTES,
    ids=[f"{pathlib.Path(p).name}:{n}:{f}" for p, n, f, _, _ in ATTRIBUTES],
)
def test_a_declared_attribute_type_copies_its_annotation(
    path: str, lineno: int, name: str, declared: str, annotation: str
) -> None:
    """An Attributes slot holds the annotation verbatim, not a paraphrase of it."""
    assert _flattened(declared) == _flattened(annotation), (
        f"{path}:{lineno} {name} declares {declared!r}, and is annotated {annotation!r}"
    )


@pytest.mark.parametrize(
    "check",
    [
        test_a_declared_return_type_copies_its_annotation,
        test_a_declared_parameter_type_copies_its_annotation,
        test_a_declared_attribute_type_copies_its_annotation,
    ],
    ids=["returns", "parameters", "attributes"],
)
def test_a_slot_check_rejects_a_paraphrase(
    check: Callable[[str, int, str, str, str], None],
) -> None:
    """Each slot check fails on a prose paraphrase and names the site and the slot."""
    with pytest.raises(AssertionError, match=r"crafted\.py:1 .*'tuple of int'"):
        check("crafted.py", 1, "f", "tuple of int", "tuple[int, ...]")

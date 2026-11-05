import inspect
from annotationlib import Format
from typing import TYPE_CHECKING, NamedTuple, cast, get_args

import pytest

from lofi import terms

if TYPE_CHECKING:
    from collections.abc import Callable


class Surface(NamedTuple):
    symbol: str
    path: str
    signature: str
    annotation_format: Format | None = Format.STRING


_SURFACES = (
    Surface(
        "BindingSite",
        "BindingSite",
        "(arguments: 'Sequence[int]', sort: 'Sort', annotation: 'int | None' = "
        "None) -> 'None'",
    ),
    Surface(
        "Signature",
        "Signature",
        "(name: 'str', sorts: 'Sequence[Sort]', operators: 'Sequence[Operator]', "
        "*, vocabularies: 'Sequence[Vocabulary]' = (), constant_cores: "
        "'Mapping[Sort, Mapping[str, str]] | None' = None, open_sorts: "
        "'Sequence[Sort]' = (), classifiers: 'Mapping[Sort, Sort] | None' = None, "
        "allows_vacuous_binding: 'bool' = True, features: "
        "'Mapping[OpName, Sequence[str]] | None' = None, holes: 'bool' = False, "
        "constant_family: 'str' = 'constant', variable_family: 'str' = "
        "'variable') "
        "-> 'None'",
    ),
    Surface(
        "Operator",
        "Operator",
        "(name: 'OpName', sort: 'Sort', *, arg_sorts: 'Sequence[Sort]' = (), "
        "binding: 'Sequence[BindingSite]' = (), payload_sort: 'PayloadSort' = "
        "'none', family: 'str' = 'primitive', erases_to: 'OpName | None' = None) "
        "-> 'None'",
    ),
    Surface(
        "TermStore.op",
        "TermStore.op",
        "(self, op, /, *args, payload=None)",
        None,
    ),
    Surface(
        "TermStore.bind",
        "TermStore.bind",
        "(self, binder, variable, /, *args)",
        None,
    ),
    Surface(
        "TermStore.fvar",
        "TermStore.fvar",
        "(self, sort, name, /)",
        None,
    ),
    Surface(
        "TermStore.bvar",
        "TermStore.bvar",
        "(self, sort, index, /)",
        None,
    ),
    Surface(
        "TermStore.const",
        "TermStore.const",
        "(self, sort, name, /)",
        None,
    ),
    Surface(
        "TermStore.hole",
        "TermStore.hole",
        "(self, sort, /)",
        None,
    ),
    Surface(
        "TermStore.view",
        "TermStore.view",
        "(self, sort, tag, /)",
        None,
    ),
    Surface(
        "SortView.op",
        "SortView.op",
        "(self, operator, /, *args, payload=None)",
        None,
    ),
    Surface(
        "SortView.bind",
        "SortView.bind",
        "(self, binder, variable, /, *args)",
        None,
    ),
    Surface(
        "SortView.fvar",
        "SortView.fvar",
        "(self, name, /)",
        None,
    ),
    Surface(
        "SortView.bvar",
        "SortView.bvar",
        "(self, index, /)",
        None,
    ),
    Surface(
        "SortView.const",
        "SortView.const",
        "(self, name, /)",
        None,
    ),
    Surface(
        "SortView.hole",
        "SortView.hole",
        "(self, /)",
        None,
    ),
    Surface(
        "ForeignTermError",
        "ForeignTermError",
        "(*, operator, position, expected, received)",
        None,
    ),
    Surface(
        "Term equality by identity",
        "Term.__eq__",
        "(self, value, /)",
        None,
    ),
    Surface("Term cached hash", "Term.__hash__", "(self, /)", None),
    Surface(
        "instantiate",
        "instantiate",
        "(store, body, value, depth=0)",
        None,
    ),
    Surface(
        "abstract",
        "abstract",
        "(store, body, atom, depth=0)",
        None,
    ),
    Surface(
        "substitute",
        "substitute",
        "(store, body, atom, value)",
        None,
    ),
    Surface(
        "open_binder",
        "open_binder",
        "(store, binder)",
        None,
    ),
    Surface(
        "close_binder",
        "close_binder",
        "(store, binder, atoms)",
        None,
    ),
    Surface(
        "NotLocallyClosedError",
        "NotLocallyClosedError",
        "(*, max_index, depth)",
        None,
    ),
    Surface("MVar", "MVar", "(name: 'str', sort: 'Sort' = 'Ty') -> 'None'"),
    Surface(
        "Node",
        "Node",
        "(op: 'OpName | PVar', args: 'tuple[Pattern, ...]' = (), "
        "payload: 'int | PVar | None' = None) -> 'None'",
    ),
    Surface(
        "match",
        "match",
        "(pattern: 'Pattern', term: 'Term', /, bindings: 'Bindings | None' = "
        "None) -> 'Bindings | None'",
    ),
    Surface(
        "build",
        "build",
        "(pattern: 'Pattern', store: 'TermStore', bindings: 'Bindings', /) -> 'Term'",
    ),
    Surface(
        "TermCodec",
        "TermCodec",
        "(index: 'OperatorIndex', layout: "
        "'Layout[Term, Interner, OperatorIndex]') -> 'None'",
    ),
    Surface(
        "Layout.encode",
        "Layout.encode",
        "(self, t: 'T', index: 'I', /) -> 'Encoding'",
    ),
    Surface(
        "OperatorIndex",
        "OperatorIndex",
        "(signature: 'Signature') -> 'None'",
    ),
    Surface(
        "Frontier",
        "Frontier",
        "(skeleton: 'Term', states: 'Mapping[Path, S]', log_weight: 'float') -> 'None'",
    ),
)


def _resolve(surface: Surface) -> Callable[..., object]:
    target: object = terms

    for part in surface.path.split("."):
        assert hasattr(target, part), (
            f"expected {surface.symbol} with signature {surface.signature}; "
            f"{surface.path} is missing"
        )
        target = getattr(target, part)

    assert callable(target), (
        f"expected {surface.symbol} with signature {surface.signature}; "
        f"{surface.path} is not callable"
    )

    return cast("Callable[..., object]", target)


@pytest.mark.parametrize("surface", _SURFACES, ids=lambda surface: surface.symbol)
def test_required_term_surface_has_exact_signature(surface: Surface) -> None:
    target = _resolve(surface)
    actual = (
        inspect.signature(target)
        if surface.annotation_format is None
        else inspect.signature(target, annotation_format=surface.annotation_format)
    )

    assert str(actual) == surface.signature, (
        f"expected {surface.symbol}{surface.signature}, got {surface.symbol}{actual}"
    )


@pytest.mark.parametrize(
    ("owner", "attribute"),
    [
        (terms.TermStore, "generation"),
        (terms.Term, "is_closed"),
        (terms.Term, "maxidx"),
        (terms.Term, "free"),
        (terms.Term, "size"),
    ],
)
def test_native_term_attributes_are_read_only_descriptors(
    owner: type[object], attribute: str
) -> None:
    descriptor = inspect.getattr_static(owner, attribute)

    assert inspect.isgetsetdescriptor(descriptor)


def test_pattern_is_the_declared_term_shape_union() -> None:
    expected = (terms.MVar, terms.Node, terms.Spine)

    assert hasattr(terms, "Pattern"), (
        "expected Pattern with members (MVar, Node, Spine); Pattern is missing"
    )
    members = get_args(terms.Pattern.__value__)

    assert members == expected, (
        f"expected Pattern with members (MVar, Node, Spine), got {members!r}"
    )

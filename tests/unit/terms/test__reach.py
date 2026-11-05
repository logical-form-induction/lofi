from collections import deque
from types import CellType, FunctionType
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
from tests.support.term_reference import categories

from lofi.terms import Reach, substrate_reaches

if TYPE_CHECKING:
    from collections.abc import Callable

    import panproto

type Substrate = (
    dx.Model
    | type[dx.Model]
    | dx.LoweringRoute
    | dx.GADT
    | dx.Family
    | dx.Operation
    | dx.Equation
    | dx.Motive
    | dx.SortExpr
    | dx.Term
    | dx.Universe
    | dx.IndexedBy
    | panproto.Theory
)
type Node = Bag[list[Node]] | Slotted[Node]


class Slotted[P]:
    """A frozen-shaped holder with slots only."""

    __slots__ = ("_parts",)

    def __init__(self, parts: P) -> None:
        self._parts = parts


class Bag[I]:
    """A holder with an instance dictionary."""

    def __init__(self, items: I) -> None:
        self.items = items


class Transient:
    """A boundary object the qualification refuses."""


class LaterTransient(Transient):
    """A subclass of a refused boundary object."""


class Declared(dx.Model, extra="forbid"):
    """A declaration model."""

    count: int


class Nested(dx.Model, extra="forbid"):
    """A model holding another model."""

    inner: Declared


def _theory() -> dx.GADT:
    builder = dx.GADT("toy")
    naturals = builder.sort("natural", closed=True)
    builder.constructor("zero", returns=naturals())
    builder.constructor("successor", predecessor=naturals(), returns=naturals())

    return builder


def _family() -> dx.Family:
    builder = dx.GADT("indexed")
    naturals = builder.sort("natural", closed=True)

    return builder.family("vector", length=naturals())


def _operation() -> dx.Operation:
    return _theory().operation_named("successor") or pytest.fail("no successor")


def _equation() -> dx.Equation:
    builder = _theory()

    return builder.equation("trivial", dx.App("zero"), dx.App("zero"))


def _compiled() -> panproto.Theory:
    return _theory().compile()


def _indexed_by() -> dx.IndexedBy:
    return dx.indexed_by(_family(), "length")


SUBSTRATE_OBJECTS: Final[dict[str, Callable[[], Substrate]]] = {
    "model": lambda: Declared(count=1),
    "model class": lambda: Declared,
    "the model base": lambda: dx.Model,
    "lowering route": lambda: dx.LoweringRoute("source", "target"),
    "gadt builder": _theory,
    "family": _family,
    "operation": _operation,
    "equation": _equation,
    "motive": lambda: dx.Motive(dx.SortExpr("natural", ())),
    "sort expression": lambda: dx.SortExpr("natural", ()),
    "term": lambda: dx.App("successor", (dx.Var("n"),)),
    "universe": lambda: dx.Universe("code", number=int),
    "indexed by": _indexed_by,
    "panproto theory": _compiled,
}


def _hidden[H](held: H) -> Bag[dict[str, tuple[Slotted[list[H]]]]]:
    """Return a graph holding one object under attributes, keys and indices."""
    return Bag({"deep": (Slotted([held]),)})


def test_a_graph_of_native_values_reaches_nothing() -> None:
    signature = categories()
    graph = Bag(
        {
            "signature": signature,
            "names": frozenset(op.name for op in signature.operators),
            "bytes": b"specification",
            "digest": "abc123",
            "parts": (Slotted([1, 2.5, None, True]), deque([signature])),
        }
    )

    assert substrate_reaches(graph) == ()


@pytest.mark.parametrize("kind", SUBSTRATE_OBJECTS)
def test_a_substrate_object_hidden_at_depth_reports_its_path(kind: str) -> None:
    held = SUBSTRATE_OBJECTS[kind]()
    (reach,) = substrate_reaches(_hidden(held))

    assert reach.path == ".items['deep'][0]._parts[0]"
    assert reach.held is (held if isinstance(held, type) else type(held))
    assert reach.instance is not isinstance(held, type)


def test_a_refused_class_is_reported_with_its_subclasses_and_itself() -> None:
    graph = Bag([Transient(), LaterTransient(), Transient, Bag(())])

    assert substrate_reaches(graph) == ()
    assert substrate_reaches(graph, refused=(Transient,)) == (
        Reach(".items[0]", Transient, instance=True),
        Reach(".items[1]", LaterTransient, instance=True),
        Reach(".items[2]", Transient, instance=False),
    )


def test_a_native_class_the_graph_holds_is_not_reported() -> None:
    assert substrate_reaches(Bag((Bag, Slotted, int, dict))) == ()


def test_every_path_segment_names_how_the_object_was_reached() -> None:
    route = dx.LoweringRoute("source", "target")
    slotted = Slotted(route)
    graph = Bag(
        {
            "attribute": Bag(route),
            "slot": slotted,
            "list": [route],
            "set": frozenset({route}),
            "method": slotted.__init__,
            "deque": deque([route]),
        }
    )
    paths = {reach.path for reach in substrate_reaches(graph)}

    assert paths == {".items['attribute'].items"}

    for name, part in graph.items.items():
        (reach,) = substrate_reaches(part)

        assert reach.held is dx.LoweringRoute, name
        assert (
            reach.path
            == {
                "attribute": ".items",
                "slot": "._parts",
                "list": "[0]",
                "set": "{0}",
                "method": ".__self__._parts",
                "deque": "<referent 1>",
            }[name]
        )


def test_a_substrate_object_used_as_a_mapping_key_is_reported() -> None:
    route = dx.LoweringRoute("source", "target")
    (reach,) = substrate_reaches({route: "value"})

    assert reach.path == "<key 0>"
    assert reach.held is dx.LoweringRoute


def test_a_foreign_object_is_reported_once_and_not_walked_into() -> None:
    nested = Nested(inner=Declared(count=2))

    assert substrate_reaches(Bag((nested, nested))) == (
        Reach(".items[0]", Nested, instance=True),
    )


def test_a_shared_object_is_reported_at_its_first_path_only() -> None:
    route = dx.LoweringRoute("source", "target")
    graph = Bag({"first": route, "second": Slotted(route)})

    assert substrate_reaches(graph) == (
        Reach(".items['first']", dx.LoweringRoute, instance=True),
    )


def test_a_cyclic_graph_terminates() -> None:
    bag: Bag[list[Node]] = Bag([])
    bag.items.append(bag)
    bag.items.append(Slotted(bag))

    assert substrate_reaches(bag) == ()


def test_a_function_or_module_the_graph_holds_is_not_walked_into_the_program() -> None:
    def touches_the_substrate() -> dx.LoweringRoute:
        return dx.LoweringRoute("source", "target")

    graph = Bag((touches_the_substrate, dx, categories, str.upper, len))

    assert substrate_reaches(graph) == ()


def test_a_substrate_object_a_function_closes_over_is_reported() -> None:
    route = dx.LoweringRoute("source", "target")

    def holds() -> dx.LoweringRoute:
        return route

    assert substrate_reaches(Bag(holds)) == (
        Reach(".items<closure route>", dx.LoweringRoute, instance=True),
    )


def test_a_substrate_object_a_function_defaults_to_is_reported() -> None:
    route = dx.LoweringRoute("source", "target")

    def holds(
        first: dx.LoweringRoute = route, *, named: dx.LoweringRoute = route
    ) -> tuple[dx.LoweringRoute, dx.LoweringRoute]:
        return first, named

    def holds_by_keyword(*, named: dx.LoweringRoute = route) -> dx.LoweringRoute:
        return named

    assert substrate_reaches(Bag(holds)) == (
        Reach(".items.__defaults__[0]", dx.LoweringRoute, instance=True),
    )
    assert substrate_reaches(Bag(holds_by_keyword)) == (
        Reach(".items.__kwdefaults__['named']", dx.LoweringRoute, instance=True),
    )


def test_a_bound_method_is_walked_through_its_receiver_and_function() -> None:
    route = dx.LoweringRoute("source", "target")

    class Holder:
        def method(self, default: dx.LoweringRoute = route) -> dx.LoweringRoute:
            return default

    (reach,) = substrate_reaches(Holder().method)

    assert reach == Reach(".__func__.__defaults__[0]", dx.LoweringRoute, instance=True)


def test_an_empty_closure_cell_is_skipped() -> None:
    route = dx.LoweringRoute("source", "target")

    def holds() -> dx.LoweringRoute:
        return route

    emptied = FunctionType(holds.__code__, holds.__globals__, closure=(CellType(),))

    assert substrate_reaches(Bag(holds)) != ()
    assert substrate_reaches(Bag(emptied)) == ()


def test_the_root_itself_is_judged_under_the_empty_path() -> None:
    (reach,) = substrate_reaches(Declared(count=3))

    assert reach == Reach("", Declared, instance=True)


def test_a_slot_left_unset_is_skipped() -> None:
    slotted = Slotted.__new__(Slotted)

    assert substrate_reaches(Bag(slotted)) == ()

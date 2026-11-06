"""Open formalism declarations lowered to the generic grammar boundary."""

from typing import TYPE_CHECKING, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin
from lofi.spec.grammar._grammar import GrammarDecl
from lofi.spec.grammar._spec import GrammarSpec

if TYPE_CHECKING:
    from collections import abc

__all__ = [
    "FormalismPlugin",
    "GrammarSpec",
    "declare_grammar",
]


class FormalismPlugin[Declaration](FrozenValue):
    """Pair one concise formalism declaration with its generic expansion."""

    __slots__ = ("_declare", "_spec")

    _noun = "formalism plugin"

    def __init__(
        self,
        *,
        spec: type[Declaration],
        declare: abc.Callable[[Declaration], GrammarDecl],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_declare", declare)

    @property
    def spec(self) -> type[Declaration]:
        """Return the concise declaration class this plugin expands."""
        return self._spec

    def construct(self, declaration: Declaration, /) -> GrammarDecl:
        """Expand one validated declaration to the generic grammar boundary."""
        return self._declare(declaration)


def _plugin(declaration: GrammarSpec, /) -> FormalismPlugin[GrammarSpec]:
    """Return the unique installed plugin selected by a declaration."""
    plugin = resolve_plugin(
        "formalism",
        cast("str", declaration.formalism),
        FormalismPlugin,
        type(declaration),
    )

    return cast("FormalismPlugin[GrammarSpec]", plugin)


def declare_grammar(source: GrammarSpec, /) -> GrammarDecl:
    """Return the single generic declaration compiled by the grammar boundary."""
    if isinstance(source, GrammarDecl):
        return source

    return _plugin(source).construct(source)

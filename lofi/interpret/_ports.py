"""Derivation interpretation protocols."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lofi.deduce import Derivation
    from lofi.interpret._derive import StepContext
    from lofi.judge import Rule
    from lofi.lexicon import Entry

__all__ = [
    "DerivationReducer",
    "DerivationStyle",
    "Interpretation",
    "Lexicalization",
    "Ordering",
]


@runtime_checkable
class Interpretation[Category, Value, Meaning = Category, Weight = float](Protocol):
    """Homomorphism from derivations to values."""

    @property
    def name(self) -> str:
        """Return the interpretation name."""

    @property
    def refines_items(self) -> bool:
        """Return whether one packed item may have several values."""

    def supports(self, rule: Rule[Category], /) -> bool:
        """Return whether the interpretation defines a rule.

        Parameters
        ----------
        rule : Rule[Category]
            Rule to inspect.

        Returns
        -------
        bool
            Whether the interpretation supports the rule.
        """

    def axiom(self, entry: Entry[Category, Meaning], /) -> Value | None:
        """Interpret a lexical axiom.

        Parameters
        ----------
        entry : Entry[Category, Meaning]
            Lexical entry that licenses the axiom.

        Returns
        -------
        Value | None
            Interpreted value, or ``None`` to reject the axiom.
        """

    def step(
        self,
        context: StepContext[Category, Weight],
        children: tuple[Value, ...],
        /,
    ) -> Value | None:
        """Interpret a rule application from its child values.

        Parameters
        ----------
        context : StepContext[Category, Weight]
            Rule application and local weight.
        children : tuple[Value, ...]
            Interpreted child values in premise order.

        Returns
        -------
        Value | None
            Interpreted value, or ``None`` to reject the application.
        """


@runtime_checkable
class Lexicalization[Category, Meaning = Category](Protocol):
    """The rules and lexical entries a derivation's edges name."""

    def rule(self, name: str, /) -> Rule[Category]:
        """Return the rule a rule edge names.

        Parameters
        ----------
        name : str
            The edge's rule name.

        Returns
        -------
        Rule[Category]
            The named rule.
        """

    def admitted(
        self, position: int, form: str, category: Category, /
    ) -> tuple[Entry[Category, Meaning], ...]:
        """Return the entries a lexical axiom may stand for.

        Parameters
        ----------
        position : int
            The axiom's token position.
        form : str
            The axiom's surface form.
        category : Category
            The axiom's category.

        Returns
        -------
        tuple[Entry[Category, Meaning], ...]
            The admitted entries, in lexicon order.
        """


@runtime_checkable
class Ordering[W](Protocol):
    """Specifies how chart weights rank derivations."""

    @property
    def name(self) -> str:
        """Return the ordering name."""

    def rank(self, weight: W, /) -> float:
        """Return one step's contribution to a derivation score.

        Parameters
        ----------
        weight : W
            Step weight produced by the chart semiring.

        Returns
        -------
        float
            Ranking contribution, with larger values preferred.
        """


@runtime_checkable
class DerivationStyle[T](Protocol):
    """Renders derivations as text."""

    @property
    def name(self) -> str:
        """Return the style name."""

    def render[W](self, derivation: Derivation[T, W], /) -> str:
        """Return a derivation written in this style.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Tree to render.

        Returns
        -------
        str
            Rendering without a trailing newline.
        """


@runtime_checkable
class DerivationReducer[T, R](Protocol):
    """Reduces derivations without materializing another tree."""

    @property
    def name(self) -> str:
        """Return the reduction name."""

    def reduce[W](self, derivation: Derivation[T, W], /) -> R:
        """Reduce a derivation to a value.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Tree to reduce.

        Returns
        -------
        R
            Reduced value.
        """

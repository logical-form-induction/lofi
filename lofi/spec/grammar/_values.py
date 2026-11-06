"""Values at the grammar boundary: the checked grammar and the native assembly."""

from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi._frozen import FrozenValue, Unpicklable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import HeadSelection
    from lofi.formalism import Grammar
    from lofi.lexicon import StaticLexicon
    from lofi.spec.grammar._report import CompileReport
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term, TermStore, TermSyntax

__all__ = ["CategoryMeasure", "GrammarAssembly"]


@final
class CategoryMeasure(FrozenValue):
    """The owner-declared structural measures of a category term.

    The constructor table is an immutable, native-ready sequence of operator
    names and result slots. Arity follows the result child while the current
    operator has a declared result role. Order raises the measure of each
    argument child and follows the result child unchanged. Depth and size read
    the complete term independently of functor roles.
    """

    _noun = "category measure"
    __slots__ = ("_constructors", "_results")

    def __init__(self, constructors: tuple[tuple[str, int], ...], /) -> None:
        names = tuple(name for name, _ in constructors)

        if len(set(names)) != len(names):
            msg = "a category measure declares one result slot per constructor"
            raise ValueError(msg)

        if any(not name or result < 0 for name, result in constructors):
            msg = "a category measure needs named constructors and nonnegative slots"
            raise ValueError(msg)

        object.__setattr__(self, "_constructors", tuple(constructors))
        object.__setattr__(self, "_results", MappingProxyType(dict(constructors)))

    @property
    def constructors(self) -> tuple[tuple[str, int], ...]:
        """Return the native-ready operator-name and result-slot relation."""
        return self._constructors

    def size(self, term: Term, /) -> int:
        """Return the exact number of nodes in a category term."""
        return term.size

    def depth(self, term: Term, /) -> int:
        """Return the greatest number of constructor edges below the root."""
        return 1 + max(map(self.depth, term.args)) if term.args else 0

    def arity(self, term: Term, /) -> int:
        """Return functor spine arity along declared result roles."""
        arity = 0
        current = term

        while (result := self._results.get(current.op.name)) is not None:
            if result >= len(current.args):
                msg = (
                    f"constructor {current.op.name!r} declares result slot {result}, "
                    f"but this term has {len(current.args)} children"
                )
                raise ValueError(msg)

            arity += 1
            current = current.args[result]

        return arity

    def order(self, term: Term, /) -> int:
        """Return category order under the declared result roles."""
        child_orders = tuple(self.order(child) for child in term.args)
        result = self._results.get(term.op.name)

        if result is None:
            return max(child_orders, default=0)

        if result >= len(child_orders):
            msg = (
                f"constructor {term.op.name!r} declares result slot {result}, "
                f"but this term has {len(child_orders)} children"
            )
            raise ValueError(msg)

        return max(
            child_orders[result],
            *(order + 1 for index, order in enumerate(child_orders) if index != result),
        )


@final
class GrammarAssembly(Unpicklable, FrozenValue):
    """A compiled grammar inside its learner language.

    Parameters
    ----------
    declaration : bytes
        The canonical bytes of the declaration the assembly was built from.
    grammar : Grammar
        The compiled grammar.
    lexicon : StaticLexicon[Term]
        The static lexicon over the grammar's store.
    syntax : TermSyntax
        The syntax category text is read and written in.
    language : LanguageAssembly
        The exact learner language whose store contains every category.
    compile_report : CompileReport
        Frozen measurements and diagnostics of the compiled grammar.
    category_measure : CategoryMeasure
        Owner-declared category size and functor spine arity.
    head_selections : Mapping[str, HeadSelection]
        The declared head policy and slot of every compiled rule, by name.
    max_category_arguments : int | None
        The generated lexicon's declared argument-count bound, or no bound.
    max_category_order : int | None
        The generated lexicon's declared category-order bound, or no bound.
    source_version : str
        The declaration version the assembly was lowered from.
    target_version : str
        The native version the assembly realizes.
    """

    _noun = "grammar boundary value"
    _because = "retains term identities"

    __slots__ = (
        "_category_measure",
        "_compile_report",
        "_declaration",
        "_digest",
        "_grammar",
        "_head_selections",
        "_language",
        "_lexicon",
        "_max_category_arguments",
        "_max_category_order",
        "_source_version",
        "_syntax",
        "_target_version",
    )

    def __init__(
        self,
        declaration: bytes,
        grammar: Grammar,
        lexicon: StaticLexicon[Term],
        syntax: TermSyntax,
        language: LanguageAssembly,
        compile_report: CompileReport,
        *,
        category_measure: CategoryMeasure,
        head_selections: Mapping[str, HeadSelection],
        max_category_arguments: int | None,
        max_category_order: int | None,
        source_version: str,
        target_version: str,
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_digest", sha256(declaration).hexdigest())
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_lexicon", lexicon)
        object.__setattr__(self, "_syntax", syntax)
        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_compile_report", compile_report)
        object.__setattr__(self, "_category_measure", category_measure)
        object.__setattr__(
            self, "_head_selections", MappingProxyType(dict(head_selections))
        )
        object.__setattr__(self, "_max_category_arguments", max_category_arguments)
        object.__setattr__(self, "_max_category_order", max_category_order)
        object.__setattr__(self, "_source_version", source_version)
        object.__setattr__(self, "_target_version", target_version)

    @property
    def declaration(self) -> bytes:
        """Return the declaration's canonical bytes."""
        return self._declaration

    @property
    def declaration_digest(self) -> str:
        """Return the SHA-256 of the declaration bytes as lowercase hexadecimal."""
        return self._digest

    @property
    def grammar(self) -> Grammar:
        """Return the compiled grammar."""
        return self._grammar

    @property
    def store(self) -> TermStore:
        """Return the learner language's one root store."""
        return self._language.store

    @property
    def language(self) -> LanguageAssembly:
        """Return the learner language containing every grammar category."""
        return self._language

    @property
    def lexicon(self) -> StaticLexicon[Term]:
        """Return the static lexicon."""
        return self._lexicon

    @property
    def syntax(self) -> TermSyntax:
        """Return the category syntax."""
        return self._syntax

    @property
    def category_measure(self) -> CategoryMeasure:
        """Return owner-declared category size and functor spine arity."""
        return self._category_measure

    @property
    def head_selections(self) -> Mapping[str, HeadSelection]:
        """Return the declared head policy and slot of each compiled rule."""
        return self._head_selections

    @property
    def compile_report(self) -> CompileReport:
        """Return the frozen compilation diagnostics for this grammar."""
        return self._compile_report

    @property
    def max_category_arguments(self) -> int | None:
        """Return the declared lexical argument-count bound, when one is set."""
        return self._max_category_arguments

    @property
    def max_category_order(self) -> int | None:
        """Return the declared lexical category-order bound, when one is set."""
        return self._max_category_order

    @property
    def source_version(self) -> str:
        """Return the declaration version the assembly was lowered from."""
        return self._source_version

    @property
    def target_version(self) -> str:
        """Return the native version the assembly realizes."""
        return self._target_version

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another assembly was built from the same declaration."""
        if type(other) is not GrammarAssembly:
            return NotImplemented

        return (
            self._digest == other._digest
            and self._grammar.fingerprint == other._grammar.fingerprint
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the declaration digest and grammar fingerprint."""
        return hash((GrammarAssembly, self._digest, self._grammar.fingerprint))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the grammar and its declaration digest."""
        return (
            f"GrammarAssembly({self._grammar.name!r}, "
            f"declaration={self._digest[:12]!r}, "
            f"fingerprint={self._grammar.fingerprint[:12]!r})"
        )

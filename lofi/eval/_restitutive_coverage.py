"""Restitutive-scope coverage over declared logical-form lexicons."""

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.data.datasets import dataset_plugin
from lofi.eval._comparison import Reduction
from lofi.registry import ComponentPlugin
from lofi.spec import RunContext, RunSpec, run_context
from lofi.spec.reducers import (
    RestitutiveCoverageReducerSettings,
    RestitutiveCoverageReducerSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.data.synthetic import InterpretationFunction
    from lofi.eval._comparison import PointMetadata
    from lofi.eval._reducers import ReductionContext
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term, TermStore

__all__ = [
    "PLUGIN",
    "RestitutiveCoverageEntry",
    "restitutive_coverage",
]

type Path = tuple[int, ...]


@runtime_checkable
class _InterpretedSource(Protocol):
    """A checked dataset holding the grammar and interpretation it is drawn under."""

    @property
    def source_grammar(self) -> GrammarAssembly: ...

    @property
    def source_interpretation(self) -> InterpretationFunction | None: ...


def _interpreted(
    checked: object, role: str, /
) -> tuple[GrammarAssembly, InterpretationFunction]:
    """Return the source grammar and interpretation a checked dataset holds.

    Raises
    ------
    TypeError
        If the checked dataset holds no source grammar and interpretation.
    ValueError
        If the dataset declares no interpretation.
    """
    if not isinstance(checked, _InterpretedSource):
        msg = f"dataset {role!r} does not declare a grammar and source semantics"
        raise TypeError(msg)

    interpretation = checked.source_interpretation

    if interpretation is None:
        msg = f"dataset {role!r} declares no interpretation"
        raise ValueError(msg)

    return checked.source_grammar, interpretation


@final
class RestitutiveCoverageEntry(FrozenValue):
    """One lexical entry's restitutive-scope result and structural witness."""

    _noun = "restitutive coverage entry"
    __slots__ = (
        "_category",
        "_change",
        "_covered",
        "_form",
        "_meaning_count",
        "_modifier",
    )

    def __init__(
        self,
        form: str,
        category: str,
        meaning_count: int,
        *,
        covered: bool,
        change: str | None,
        modifier: str | None,
    ) -> None:
        object.__setattr__(self, "_form", form)
        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_meaning_count", meaning_count)
        object.__setattr__(self, "_covered", covered)
        object.__setattr__(self, "_change", change)
        object.__setattr__(self, "_modifier", modifier)

    @property
    def form(self) -> str:
        """Return the lexical surface form."""
        return self._form

    @property
    def category(self) -> str:
        """Return the lexical category in the grammar's surface syntax."""
        return self._category

    @property
    def meaning_count(self) -> int:
        """Return how many declared meanings were queried for the entry."""
        return self._meaning_count

    @property
    def covered(self) -> bool:
        """Return whether at least one meaning admits restitutive scope."""
        return self._covered

    @property
    def change(self) -> str | None:
        """Return the change operator witnessing coverage, when one exists."""
        return self._change

    @property
    def modifier(self) -> str | None:
        """Return the modifier witnessing coverage, when one exists."""
        return self._modifier


def _constant_name(language: LanguageAssembly, term: Term, /) -> str | None:
    """Return a constant occurrence's declared name, or nothing for another node."""
    if not term.op.is_constant or term.payload is None:
        return None
    return language.store.signature.vocabulary(term.op.sort).name(term.payload)


def _nodes(term: Term, /) -> tuple[tuple[Path, Term], ...]:
    """Return root-first term occurrences with their stable argument paths."""
    pending: list[tuple[Path, Term]] = [((), term)]
    held: list[tuple[Path, Term]] = []
    while pending:
        path, node = pending.pop()
        held.append((path, node))
        pending.extend(
            ((*path, index), child)
            for index, child in reversed(tuple(enumerate(node.args)))
        )
    return tuple(held)


def _application_spine(
    term: Term, path: Path, application: str, /
) -> tuple[Term, tuple[tuple[Path, Term], ...]]:
    """Return a curried application's head and argument occurrences in order."""
    current = term
    current_path = path
    reversed_arguments: list[tuple[Path, Term]] = []
    while current.op.name == application and len(current.args) == 2:
        reversed_arguments.append(((*current_path, 1), current.args[1]))
        current = current.args[0]
        current_path = (*current_path, 0)
    return current, tuple(reversed(reversed_arguments))


def _replace(store: TermStore, term: Term, path: Path, replacement: Term, /) -> Term:
    """Return a term with one occurrence replaced and every other node shared."""
    if not path:
        return replacement
    position, *remaining = path
    arguments = list(term.args)
    arguments[position] = _replace(
        store, arguments[position], tuple(remaining), replacement
    )
    return store.op(term.op, *arguments, payload=term.payload)


def _witness(
    interpretation: InterpretationFunction,
    meaning: Term,
    *,
    change_class: str,
    modifiers: tuple[str, ...],
) -> tuple[str, str] | None:
    """Return a typed change/modifier witness for restitutive scope."""
    language = interpretation.language
    original = interpretation.check(meaning)
    scheme = original.scheme
    if not original.is_well_typed or scheme is None:
        msg = "a declared lexical meaning is not well typed at coverage time"
        raise ValueError(msg)

    for path, node in _nodes(meaning):
        head, arguments = _application_spine(node, path, language.calculus.app)
        change = _constant_name(language, head)
        if change is None or change_class not in language.constant_classes[change]:
            continue

        for argument_path, argument in arguments:
            inferred = original.assignment.get(argument_path)
            if inferred != interpretation.truth:
                continue
            for modifier in modifiers:
                modified = language.apply(language.constant(modifier), argument)
                candidate = _replace(language.store, meaning, argument_path, modified)
                checked = interpretation.check(candidate)
                if (
                    checked.is_well_typed
                    and checked.scheme is not None
                    and checked.scheme.body == scheme.body
                ):
                    return change, modifier
    return None


def restitutive_coverage(
    grammar: GrammarAssembly,
    interpretation: InterpretationFunction,
    *,
    change_class: str,
    modifier_class: str,
) -> tuple[RestitutiveCoverageEntry, ...]:
    """Return whether each lexical entry admits typed modifier scope below change."""
    language = interpretation.language
    by_class = language.constant_classes
    changes = tuple(
        sorted(name for name, classes in by_class.items() if change_class in classes)
    )
    modifiers = tuple(
        sorted(name for name, classes in by_class.items() if modifier_class in classes)
    )
    if not changes:
        msg = f"the language declares no constant in change class {change_class!r}"
        raise ValueError(msg)
    if not modifiers:
        msg = f"the language declares no constant in modifier class {modifier_class!r}"
        raise ValueError(msg)

    rows: list[RestitutiveCoverageEntry] = []
    for (form, category), meanings in sorted(
        interpretation.entries.items(),
        key=lambda row: (row[0][0], row[0][1].key),
    ):
        witnesses = tuple(
            witness
            for meaning in meanings
            if (
                witness := _witness(
                    interpretation,
                    meaning,
                    change_class=change_class,
                    modifiers=modifiers,
                )
            )
            is not None
        )
        witness = None if not witnesses else witnesses[0]
        rows.append(
            RestitutiveCoverageEntry(
                form,
                grammar.syntax.render(grammar.store, category),
                len(meanings),
                covered=witness is not None,
                change=None if witness is None else witness[0],
                modifier=None if witness is None else witness[1],
            )
        )
    return tuple(rows)


@final
class _RestitutiveCoverageReducer:
    """Lower each declared point lexicon and emit its restitutive coverage rows."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self,
        settings: RestitutiveCoverageReducerSettings,
        context: ReductionContext,
    ) -> None:
        self._settings = settings
        self._context = context

    def _point_rows(
        self, config_hash: str, point: PointMetadata, /
    ) -> tuple[Mapping[str, JsonValue], ...]:
        settings = self._settings
        run = RunSpec.model_validate(point.config)
        section = run.data.datasets.get(settings.dataset)
        if section is None:
            msg = f"point {point.point_id!r} declares no dataset {settings.dataset!r}"
            raise ValueError(msg)
        with run_context(RunContext(self._context.root, point.seed)):
            checked = dataset_plugin(section.dataset).check(section.dataset)

        grammar, interpretation = _interpreted(checked, settings.dataset)
        try:
            coverage = restitutive_coverage(
                grammar,
                interpretation,
                change_class=settings.change_class,
                modifier_class=settings.modifier_class,
            )
        finally:
            interpretation.service.close()

        return tuple(
            {
                "reducer": "restitutive-coverage",
                "config_hash": config_hash,
                "point_id": point.point_id,
                "seed": point.seed,
                "dataset": settings.dataset,
                "form": entry.form,
                "category": entry.category,
                "meaning_count": entry.meaning_count,
                "status": "covered" if entry.covered else "uncovered",
                "restitutive": entry.covered,
                "change": entry.change,
                "modifier": entry.modifier,
            }
            for entry in coverage
        )

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return one coverage row per lexical entry and expanded point."""
        del rows
        if not self._context.points:
            msg = "a restitutive-coverage reducer has no expanded points"
            raise ValueError(msg)
        emitted = tuple(
            row
            for config_hash, points in sorted(self._context.points.items())
            for point in points
            for row in self._point_rows(config_hash, point)
        )
        return Reduction(emitted, ())


def _build(
    settings: RestitutiveCoverageReducerSettings,
    context: ReductionContext,
    /,
) -> _RestitutiveCoverageReducer:
    """Build the declared lexical coverage reducer."""
    return _RestitutiveCoverageReducer(settings, context)


PLUGIN: Final = ComponentPlugin(spec=RestitutiveCoverageReducerSpec, build=_build)

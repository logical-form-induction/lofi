"""Type-directed admission of decoded terms."""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.lf.models import ModelLanguage, truth_type
from lofi.terms import MVar, TermStore, build, match, replace_at
from lofi.typecheck import (
    InfiniteTypeError,
    LexicalSignature,
    Scheme,
    TypeChecker,
    TypingRule,
    Unifier,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.decoders.rnn import TypeDirectedSettings
    from lofi.spec.language import LanguageCodec
    from lofi.terms import ChoiceIndex, HoleContext, Path, Term

__all__ = [
    "UNRESTRICTED_ADMISSION",
    "TermAdmission",
    "TypeDirectedAdmission",
    "TypeDirectedLanguage",
    "term_admission",
]


class _FrontierTyping(NamedTuple):
    """Solved types and variable environments for one partial frontier."""

    assignment: Mapping[Path, Term]
    bound: Mapping[Path, Mapping[str, tuple[Term, ...]]]
    free: Mapping[tuple[str, str], Term]


@final
class _BoundedMemo[K, V]:
    """Hold a bounded least-recently-used memo owned by one admission policy."""

    __slots__ = ("_capacity", "_values")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._values: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, /) -> V | None:
        """Return and refresh one held value."""
        value = self._values.get(key)

        if value is not None:
            self._values.move_to_end(key)

        return value

    def put(self, key: K, value: V, /) -> V:
        """Hold one value and evict the least recently used excess entry."""
        self._values[key] = value
        self._values.move_to_end(key)

        while len(self._values) > self._capacity:
            self._values.popitem(last=False)

        return value


class TypeDirectedLanguage(ModelLanguage, Protocol):
    """Expose the declared typing objects needed during decoder search."""

    @property
    def type_sort(self) -> str:
        """Return the sort containing semantic types."""

    @property
    def lexicon(self) -> LexicalSignature:
        """Return the schemes of the language's constants."""

    @property
    def rules(self) -> Mapping[str, TypingRule]:
        """Return structural typing rules keyed by term operator."""

    @property
    def codec(self) -> LanguageCodec:
        """Return the language-owned finite term codecs."""


@runtime_checkable
class TermAdmission(Protocol):
    """Decide whether one completed decoder term belongs to its declared language."""

    def admits(self, term: Term, /) -> bool:
        """Return whether the completed term is admitted."""

    def admitted_choices(
        self,
        decisions: tuple[tuple[Path, int], ...],
        context: HoleContext,
        index: ChoiceIndex,
        /,
    ) -> tuple[bool, ...]:
        """Return which next choices preserve the declared admission policy."""


@final
class _UnrestrictedAdmission(FrozenValue):
    """Admit every completed term under an explicitly untyped decoder policy."""

    _noun = "term-admission policy"
    __slots__ = ()

    def admits(self, term: Term, /) -> bool:
        """Admit a term without consulting a type grammar."""
        del term
        return True

    def admitted_choices(
        self,
        decisions: tuple[tuple[Path, int], ...],
        context: HoleContext,
        index: ChoiceIndex,
        /,
    ) -> tuple[bool, ...]:
        """Admit every choice under an explicitly untyped policy."""
        del decisions, context
        return (True,) * index.width


UNRESTRICTED_ADMISSION: TermAdmission = _UnrestrictedAdmission()
"""The admission policy selected by ``prefix: none`` and ``mask: none``."""


def term_admission(
    language: TypeDirectedLanguage,
    settings: TypeDirectedSettings,
    /,
) -> TermAdmission:
    """Build the one admission policy declared for a recursive decoder."""
    if settings.prefix == "none" and settings.mask == "none":
        return UNRESTRICTED_ADMISSION

    return TypeDirectedAdmission(language, settings)


@final
class TypeDirectedAdmission(FrozenValue):
    """Admit well-typed terms under one declared type-direction policy."""

    _noun = "term-admission policy"
    __slots__ = (
        "_checker",
        "_frontier",
        "_frontier_unifier",
        "_language",
        "_lexicon",
        "_replacement_memo",
        "_rules",
        "_settings",
        "_skeleton_memo",
        "_truth",
        "_typing_memo",
        "_unifier",
    )

    def __init__(
        self,
        language: TypeDirectedLanguage,
        settings: TypeDirectedSettings,
    ) -> None:
        """Bind admission to the exact language used by the decoder."""
        frontier = TermStore(language.store.signature.with_holes())
        lexicon = LexicalSignature(
            {
                symbol: Scheme(
                    _rehome(language.store, frontier, scheme.body),
                    tuple(
                        frontier.atom(
                            language.store.atom_sort(atom),
                            language.store.atom_name(atom),
                        )
                        for atom in scheme.quantified
                    ),
                )
                for symbol, scheme in language.lexicon.schemes.items()
            }
        )
        hole = frontier.signature.hole(language.term_sort)
        rules = {
            **language.rules,
            hole.name: TypingRule(
                hole.name,
                (),
                (),
                MVar("Decoded", language.type_sort),
            ),
        }
        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_unifier", Unifier(language.store))
        object.__setattr__(self, "_frontier", frontier)
        object.__setattr__(self, "_frontier_unifier", Unifier(frontier))
        object.__setattr__(self, "_checker", TypeChecker(frontier))
        object.__setattr__(self, "_lexicon", lexicon)
        object.__setattr__(self, "_rules", MappingProxyType(rules))
        truth = (
            None
            if not language.profile.logical_constants
            else _rehome(language.store, frontier, truth_type(language))
        )
        object.__setattr__(self, "_truth", truth)
        capacity = (
            language.codec.at(language.term_sort).capacity
            * language.codec.choices.width
        )
        typing_memo: _BoundedMemo[Term, _FrontierTyping] = _BoundedMemo(capacity)
        replacement_memo: _BoundedMemo[tuple[Term, Term], bool] = _BoundedMemo(capacity)
        skeleton_memo: _BoundedMemo[tuple[tuple[Path, int], ...], Term] = _BoundedMemo(
            capacity
        )
        object.__setattr__(self, "_typing_memo", typing_memo)
        object.__setattr__(self, "_replacement_memo", replacement_memo)
        object.__setattr__(self, "_skeleton_memo", skeleton_memo)

    def admits(self, term: Term, /) -> bool:
        """Return whether a completed term satisfies the declared typing policy."""
        language = self._language
        settings = self._settings

        if term.op.sort != language.term_sort:
            return True

        try:
            inferred = language.type_of(term)
        except ValueError:
            return False

        if settings.prefix == "none":
            return True

        return self._unifier.unify(inferred, truth_type(language)) is not None

    def admitted_choices(
        self,
        decisions: tuple[tuple[Path, int], ...],
        context: HoleContext,
        index: ChoiceIndex,
        /,
    ) -> tuple[bool, ...]:
        """Return choices whose partial term has a completion of the required type."""
        language = self._language
        settings = self._settings

        if index.signature is not language.store.signature:
            msg = "type-directed admission and decoder choices must share a signature"
            raise ValueError(msg)

        root_sort = (
            context.sort if not decisions else index.operator(decisions[0][1]).sort
        )

        if root_sort != language.term_sort:
            return (True,) * index.width

        if settings.prefix == "type-grammar" and self._truth is None:
            msg = "type-directed term decoding requires a declared truth type"
            raise ValueError(msg)

        skeleton = self._skeleton(decisions, index)
        typing = self._frontier_typing(skeleton)

        if typing is None:
            return (False,) * index.width

        return tuple(
            admitted
            and self._choice_completable(
                typing,
                context.path,
                choice,
                index,
            )
            for choice, admitted in enumerate(index.legal(context))
        )

    def _skeleton(
        self,
        decisions: tuple[tuple[Path, int], ...],
        index: ChoiceIndex,
        /,
    ) -> Term:
        """Rebuild one partial frontier from its chosen paths."""
        cached = self._skeleton_memo.get(decisions)

        if cached is not None:
            return cached

        if not decisions:
            skeleton = self._frontier.hole(self._language.term_sort)
        else:
            path, code = decisions[-1]
            skeleton = self._fill(
                self._skeleton(decisions[:-1], index),
                path,
                code,
                index,
            )

        return self._skeleton_memo.put(decisions, skeleton)

    def _fill(
        self,
        skeleton: Term,
        path: Path,
        choice: int,
        index: ChoiceIndex,
        /,
    ) -> Term:
        """Fill one frontier hole with one payload-aware choice."""
        store = self._frontier
        selected = index.choice(choice)
        arguments = tuple(store.hole(sort) for sort in selected.operator.arg_sorts)
        replacement = index.instantiate(store, choice, *arguments)

        return replace_at(store, skeleton, path, replacement)

    def _frontier_typing(self, skeleton: Term, /) -> _FrontierTyping | None:
        """Infer constrained node types and variable environments once."""
        cached = self._typing_memo.get(skeleton)

        if cached is not None:
            return cached

        try:
            result = self._checker.check(skeleton, self._lexicon, self._rules)
        except ValueError:
            return None

        if result.residual or result.scheme is None:
            return None

        assignment = result.assignment

        if self._settings.prefix != "none":
            truth = self._truth

            if truth is None:
                msg = "type-directed term decoding requires a declared truth type"
                raise ValueError(msg)

            substitution = self._frontier_unifier.unify(result.scheme.body, truth)

            if substitution is None:
                return None

            assignment = MappingProxyType(
                {
                    path: self._frontier_unifier.apply(substitution, inferred)
                    for path, inferred in assignment.items()
                }
            )

        bound, free = self._variable_types(skeleton, assignment)
        return self._typing_memo.put(
            skeleton,
            _FrontierTyping(assignment, bound, free),
        )

    def _variable_types(
        self,
        skeleton: Term,
        assignment: Mapping[Path, Term],
        /,
    ) -> tuple[
        Mapping[Path, Mapping[str, tuple[Term, ...]]],
        Mapping[tuple[str, str], Term],
    ]:
        """Propagate solved binder and free-assumption types through a frontier."""
        environments: dict[Path, Mapping[str, tuple[Term, ...]]] = {}
        free: dict[tuple[str, str], Term] = {}

        def visit(
            node: Term,
            path: Path,
            environment: Mapping[str, tuple[Term, ...]],
        ) -> None:
            environments[path] = environment

            if node.op.is_free_variable and node.payload is not None:
                free.setdefault(
                    (node.op.sort, self._frontier.atom_name(node.payload)),
                    assignment[path],
                )

            if not node.args:
                return

            rule = self._rules[node.op.name]
            annotation_positions = frozenset(
                site.annotation
                for site in node.op.binding
                if site.annotation is not None
            )
            logical_positions = tuple(
                position
                for position in range(len(node.args))
                if position not in annotation_positions
            )
            bindings = match(rule.result, assignment[path])

            if bindings is None:
                msg = f"typing rule {rule.op!r} does not match its solved result"
                raise ValueError(msg)

            for position, premise in zip(
                logical_positions,
                rule.premises,
                strict=True,
            ):
                bindings = match(
                    premise,
                    assignment[(*path, position)],
                    bindings,
                )

                if bindings is None:
                    msg = f"typing rule {rule.op!r} does not match its solved premise"
                    raise ValueError(msg)

            for position, child in enumerate(node.args):
                child_environment = environment

                if position in logical_positions:
                    premise_at = logical_positions.index(position)
                    binding = rule.binds[premise_at]

                    if binding is not None:
                        inferred = build(binding, self._frontier, bindings)
                        held = dict(environment)

                        for site in node.op.scope(position):
                            held[site.sort] = (
                                inferred,
                                *held.get(site.sort, ()),
                            )

                        child_environment = MappingProxyType(held)

                visit(child, (*path, position), child_environment)

        visit(skeleton, (), MappingProxyType({}))
        return MappingProxyType(environments), MappingProxyType(free)

    def _choice_completable(
        self,
        typing: _FrontierTyping,
        path: Path,
        choice: int,
        index: ChoiceIndex,
        /,
    ) -> bool:
        """Return whether one choice can inhabit the inferred frontier type."""
        selected = index.choice(choice)
        expected = typing.assignment[path]

        if selected.operator.is_bound_variable:
            payload = selected.payload
            environment = typing.bound[path].get(selected.operator.sort, ())

            return (
                payload is not None
                and payload < len(environment)
                and self._compatible(environment[payload], expected)
            )

        if selected.operator.is_free_variable:
            payload = selected.payload

            if payload is None:
                return False

            existing = typing.free.get((selected.operator.sort, f"$decoded{payload}"))
            return existing is None or self._compatible(existing, expected)

        store = self._frontier
        replacement = index.instantiate(
            store,
            choice,
            *(store.hole(sort) for sort in selected.operator.arg_sorts),
        )
        return self._replacement_completable(replacement, expected)

    def _replacement_completable(self, replacement: Term, expected: Term, /) -> bool:
        """Compare one local candidate's principal result with a hole type."""
        key = (replacement, expected)
        cached = self._replacement_memo.get(key)

        if cached is not None:
            return cached

        try:
            result = self._checker.check(replacement, self._lexicon, self._rules)
        except ValueError:
            return False

        if result.residual or result.scheme is None:
            return False

        scheme = result.scheme
        body = scheme.body

        if scheme.quantified:
            substitutions = {
                atom: self._frontier.fvar(
                    self._language.type_sort,
                    f"$admission{atom}",
                )
                for atom in scheme.quantified
            }
            body = Unifier(self._frontier).apply(substitutions, body)

        return self._replacement_memo.put(
            key,
            self._compatible(body, expected),
        )

    def _compatible(self, left: Term, right: Term, /) -> bool:
        """Return whether two frontier types have a finite unifier."""
        try:
            return self._frontier_unifier.unify(left, right) is not None
        except InfiniteTypeError:
            return False


def _rehome(source: TermStore, target: TermStore, term: Term, /) -> Term:
    """Re-intern one language term in its hole-extended typing store."""
    payload = term.payload

    if source.signature.is_free_variable(term.op):
        if payload is None:
            msg = f"free variable {term.op.name!r} has no atom handle"
            raise ValueError(msg)

        return target.fvar(term.op.sort, source.atom_name(payload))

    if source.signature.is_bound_variable(term.op):
        if payload is None:
            msg = f"bound variable {term.op.name!r} has no de Bruijn index"
            raise ValueError(msg)

        return target.bvar(term.op.sort, payload)

    if source.signature.is_constant(term.op):
        return target.const(term.op.sort, source.payload_name(term))

    arguments = tuple(_rehome(source, target, argument) for argument in term.args)
    return target.op(
        target.signature.by_name(term.op.name),
        *arguments,
        payload=payload,
    )

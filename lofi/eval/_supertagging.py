"""Supertagging accuracy and beam diagnostics."""

from collections import Counter
from math import fsum, isclose
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Never, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.eval._values import MetricResult, Resampling
from lofi.terms import Term, TermKey, TermStore, TermSyntax

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.eval._values import ConventionVector


__all__ = [
    "SUPERTAG_ROWS",
    "CategoryProbability",
    "SupertagAccuracy",
    "SupertagConfusionRow",
    "SupertagLengthError",
    "SupertagPrediction",
    "TokenSupertags",
]

SUPERTAG_ROWS: Final = (
    "tagger-accuracy",
    "parser-category-accuracy",
    "coverage",
    "multi-tag-accuracy",
    "ambiguity",
)
"""Supertagging result rows in report order."""

_PROBABILITY_TOLERANCE: Final = 1e-9
_TOKEN_UNITS: Final = Resampling("token")
_SENTENCE_UNITS: Final = Resampling("sentence")

type _Variant = Literal["tagger", "parser"]
type _ConfusionCell = tuple[_Variant, str, str | None]
type _Fingerprint = tuple[
    tuple[tuple[tuple[TermKey, float], ...], ...],
    tuple[TermKey, ...] | None,
    tuple[TermKey, ...],
]


@final
class CategoryProbability:
    """A category paired with its linear probability."""

    __slots__ = ("_category", "_probability")

    def __init__(self, category: Term, probability: float) -> None:
        if not isinstance(category, Term):
            msg = f"a supertag category is not a term: {category!r}"
            raise TypeError(msg)

        held = ensure_finite("a category probability", probability)

        if not 0.0 <= held <= 1.0:
            msg = f"a category probability is outside [0, 1]: {probability!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_probability", held)

    @property
    def category(self) -> Term:
        """Return the category."""
        return self._category

    @property
    def probability(self) -> float:
        """Return the linear probability."""
        return self._probability

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a category probability is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a category probability is immutable"
        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another entry has the same category and probability."""
        if type(other) is not CategoryProbability:
            return NotImplemented

        return (
            self._category is other._category
            and self._probability == other._probability
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the category and probability."""
        return hash((CategoryProbability, self._category, self._probability))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"CategoryProbability({self._category!r}, {self._probability!r})"


@final
class TokenSupertags:
    """An ordered probability distribution over one token's categories."""

    __slots__ = ("_candidates",)

    def __init__(self, candidates: Sequence[CategoryProbability]) -> None:
        held = tuple(candidates)

        if not held:
            msg = "a token supertag distribution has no candidates"
            raise ValueError(msg)

        for position, candidate in enumerate(held):
            if not isinstance(candidate, CategoryProbability):
                msg = (
                    f"supertag candidate {position} is not a category probability: "
                    f"{candidate!r}"
                )
                raise TypeError(msg)

        keys = tuple(candidate.category.key() for candidate in held)

        if len(set(keys)) != len(keys):
            msg = "a token supertag distribution repeats a structural category"
            raise ValueError(msg)

        total = fsum(candidate.probability for candidate in held)

        if not isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            msg = f"a token supertag distribution sums to {total!r}, not 1.0"
            raise ValueError(msg)

        object.__setattr__(self, "_candidates", held)

    @property
    def candidates(self) -> tuple[CategoryProbability, ...]:
        """Return candidates in their declared order."""
        return self._candidates

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a token supertag distribution is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a token supertag distribution is immutable"
        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another distribution has the same ordered candidates."""
        if type(other) is not TokenSupertags:
            return NotImplemented

        return self._candidates == other._candidates

    @override
    def __hash__(self) -> int:
        """Return a hash over the ordered candidates."""
        return hash((TokenSupertags, self._candidates))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"TokenSupertags({self._candidates!r})"


@final
class SupertagPrediction:
    """Tagger probabilities and optional parser categories for one sentence."""

    __slots__ = ("_parser_categories", "_tokens")

    def __init__(
        self,
        tokens: Sequence[TokenSupertags],
        parser_categories: Sequence[Term] | None,
    ) -> None:
        held_tokens = tuple(tokens)

        for position, token in enumerate(held_tokens):
            if not isinstance(token, TokenSupertags):
                msg = f"token distribution {position} is not TokenSupertags: {token!r}"
                raise TypeError(msg)

        held_parser = None if parser_categories is None else tuple(parser_categories)

        if held_parser is not None:
            for position, category in enumerate(held_parser):
                if not isinstance(category, Term):
                    msg = f"parser category {position} is not a term: {category!r}"
                    raise TypeError(msg)

        object.__setattr__(self, "_tokens", held_tokens)
        object.__setattr__(self, "_parser_categories", held_parser)

    @property
    def tokens(self) -> tuple[TokenSupertags, ...]:
        """Return per-token distributions in sentence order."""
        return self._tokens

    @property
    def parser_categories(self) -> tuple[Term, ...] | None:
        """Return parser categories, or None for an unparsed sentence."""
        return self._parser_categories

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a supertag prediction is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a supertag prediction is immutable"
        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another prediction has the same two fields."""
        if type(other) is not SupertagPrediction:
            return NotImplemented

        return (
            self._tokens == other._tokens
            and self._parser_categories == other._parser_categories
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over both prediction fields."""
        return hash((SupertagPrediction, self._tokens, self._parser_categories))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"SupertagPrediction({self._tokens!r}, {self._parser_categories!r})"


@final
class SupertagConfusionRow:
    """One serialized category-confusion count."""

    __slots__ = ("_count", "_gold", "_predicted", "_variant")

    def __init__(
        self,
        variant: _Variant,
        gold: str,
        predicted: str | None,
        count: int,
    ) -> None:
        if variant not in ("tagger", "parser"):
            msg = f"an unknown supertagging variant was reported: {variant!r}"
            raise ValueError(msg)

        held_count = ensure_non_negative("a confusion count", count)

        if held_count == 0:
            msg = "a confusion row has a zero count"
            raise ValueError(msg)

        object.__setattr__(self, "_variant", variant)
        object.__setattr__(self, "_gold", ensure_name("a gold category", gold))
        object.__setattr__(
            self,
            "_predicted",
            None
            if predicted is None
            else ensure_name("a predicted category", predicted),
        )
        object.__setattr__(self, "_count", held_count)

    @property
    def variant(self) -> _Variant:
        """Return the scored prediction variant."""
        return self._variant

    @property
    def gold(self) -> str:
        """Return the canonical gold category."""
        return self._gold

    @property
    def predicted(self) -> str | None:
        """Return the canonical prediction, or None when parsing failed."""
        return self._predicted

    @property
    def count(self) -> int:
        """Return the token count."""
        return self._count

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a supertag confusion row is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a supertag confusion row is immutable"
        raise AttributeError(msg)

    def _identity(self) -> tuple[_Variant, str, str | None, int]:
        """Return fields in comparison order."""
        return (self._variant, self._gold, self._predicted, self._count)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row has the same fields."""
        if type(other) is not SupertagConfusionRow:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the row fields."""
        return hash((SupertagConfusionRow, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"SupertagConfusionRow({self._variant!r}, {self._gold!r}, "
            f"{self._predicted!r}, {self._count!r})"
        )


@final
class SupertagLengthError(ValueError):
    """A sentence-level supertag sequence-length mismatch."""

    __slots__ = ("expected", "field", "item_id", "received")

    def __init__(
        self,
        *,
        item_id: ItemId,
        field: str,
        expected: int,
        received: int,
    ) -> None:
        self.item_id = item_id
        self.field = ensure_name("a mismatched field", field)
        self.expected = ensure_non_negative("an expected length", expected)
        self.received = ensure_non_negative("a received length", received)
        super().__init__(
            f"item {item_id!r} has {received} {field} entries for "
            f"{expected} gold tokens"
        )


class _Item(NamedTuple):
    """Frozen contributions from one scored sentence."""

    fingerprint: _Fingerprint
    tokens: int
    tagger_correct: int
    parser_correct: int
    parsed: bool
    multi_correct: int
    survivors: int
    confusion: tuple[_ConfusionCell, ...]


@final
class SupertagAccuracy:
    """Accumulator for supertagging accuracy and beam diagnostics."""

    __slots__ = ("_beta", "_items", "_store", "_syntax")

    def __init__(
        self,
        store: TermStore,
        syntax: TermSyntax,
        *,
        beta: float,
    ) -> None:
        if not isinstance(store, TermStore):
            msg = f"a supertag metric store is not a TermStore: {store!r}"
            raise TypeError(msg)

        if not isinstance(syntax, TermSyntax):
            msg = f"a supertag metric syntax does not implement TermSyntax: {syntax!r}"
            raise TypeError(msg)

        if syntax.signature is not store.signature:
            msg = (
                f"supertag syntax for {syntax.signature.name!r} cannot score "
                f"terms from {store.signature.name!r}"
            )
            raise ValueError(msg)

        threshold = ensure_finite("a supertag beta", beta)

        if not 0.0 <= threshold < 1.0:
            msg = f"a supertag beta is outside [0, 1): {beta!r}"
            raise ValueError(msg)

        self._store = store
        self._syntax = syntax
        self._beta = threshold
        self._items: dict[ItemId, _Item] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "supertag-accuracy"

    @property
    def conventions(self) -> ConventionVector | None:
        """Return no span conventions for supertag metrics."""
        return None

    @property
    def order_invariant(self) -> bool:
        """Return whether the aggregate is independent of update order."""
        return True

    @property
    def beta(self) -> float:
        """Return the C&C pruning threshold."""
        return self._beta

    def reset(self) -> None:
        """Forget every sentence scored so far."""
        self._items = {}

    def update(
        self,
        pred: SupertagPrediction,
        gold: Sequence[Term],
        *,
        key: ItemId,
    ) -> None:
        """Score one sentence's tagger and parser predictions."""
        if not isinstance(pred, SupertagPrediction):
            msg = f"a supertag prediction is not one: {pred!r}"
            raise TypeError(msg)

        scored = self._score(pred, tuple(gold), key)
        standing = self._items.get(key)

        if standing is not None:
            if standing.fingerprint != scored.fingerprint:
                msg = f"item {key!r} was already scored against another prediction"
                raise ValueError(msg)

            return

        self._items[key] = scored

    def compute(self) -> MetricResult:
        """Return tagger accuracy over every scored token."""
        return self.rows()[SUPERTAG_ROWS[0]]

    def rows(self) -> Mapping[str, MetricResult]:
        """Return both accuracies, coverage, and beam diagnostics."""
        items = tuple(self._items.values())
        tokens = sum(item.tokens for item in items)
        sentences = len(items)
        tagger_correct = sum(item.tagger_correct for item in items)
        parser_correct = sum(item.parser_correct for item in items)
        parsed = sum(item.parsed for item in items)
        multi_correct = sum(item.multi_correct for item in items)
        survivors = sum(item.survivors for item in items)
        rows = {
            "tagger-accuracy": self._token_result(
                tagger_correct,
                tokens,
                component="correct",
            ),
            "parser-category-accuracy": self._token_result(
                parser_correct,
                tokens,
                component="correct",
                extra={"unparsed_sentences": float(sentences - parsed)},
            ),
            "coverage": MetricResult(
                None if sentences == 0 else parsed / sentences,
                sentences,
                _SENTENCE_UNITS,
                components={
                    "parsed": float(parsed),
                    "sentences": float(sentences),
                },
            ),
            "multi-tag-accuracy": self._token_result(
                multi_correct,
                tokens,
                component="correct",
                extra={"beta": self._beta},
            ),
            "ambiguity": self._token_result(
                survivors,
                tokens,
                component="surviving",
                extra={"beta": self._beta},
            ),
        }

        return MappingProxyType({name: rows[name] for name in SUPERTAG_ROWS})

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return sentence-level tagger accuracy keyed by item ID."""
        return MappingProxyType(
            {
                key: self._token_result(
                    item.tagger_correct,
                    item.tokens,
                    component="correct",
                )
                for key, item in self._items.items()
            }
        )

    def confusion(self) -> tuple[SupertagConfusionRow, ...]:
        """Return serialized tagger and parser confusion rows."""
        counts = Counter(
            cell for item in self._items.values() for cell in item.confusion
        )
        ordered = sorted(
            counts.items(),
            key=lambda entry: (
                0 if entry[0][0] == "tagger" else 1,
                entry[0][1],
                entry[0][2] is not None,
                entry[0][2] or "",
            ),
        )

        return tuple(SupertagConfusionRow(*cell, count) for cell, count in ordered)

    def _score(
        self,
        pred: SupertagPrediction,
        gold: tuple[Term, ...],
        key: ItemId,
    ) -> _Item:
        """Freeze one sentence's metric contributions."""
        expected = len(gold)
        self._require_length(key, "tagger", expected, len(pred.tokens))

        if pred.parser_categories is not None:
            self._require_length(
                key,
                "parser",
                expected,
                len(pred.parser_categories),
            )

        gold_data = tuple(
            self._category(category, "gold", at) for at, category in enumerate(gold)
        )
        tagger_data = tuple(
            tuple(
                (
                    self._category(candidate.category, "tagger", at),
                    candidate.probability,
                )
                for candidate in token.candidates
            )
            for at, token in enumerate(pred.tokens)
        )
        parser_data = (
            None
            if pred.parser_categories is None
            else tuple(
                self._category(category, "parser", at)
                for at, category in enumerate(pred.parser_categories)
            )
        )
        tagger_correct = 0
        parser_correct = 0
        multi_correct = 0
        survivors = 0
        confusion: list[_ConfusionCell] = []

        for position, ((gold_key, gold_text), candidates) in enumerate(
            zip(gold_data, tagger_data, strict=True)
        ):
            best = max(candidates, key=lambda candidate: candidate[1])
            threshold = self._beta * best[1]
            beam = tuple(
                candidate for candidate in candidates if candidate[1] > threshold
            )
            tagger_correct += best[0][0] == gold_key
            multi_correct += any(candidate[0][0] == gold_key for candidate in beam)
            survivors += len(beam)
            confusion.append(("tagger", gold_text, best[0][1]))

            if parser_data is None:
                confusion.append(("parser", gold_text, None))
            else:
                parser_key, parser_text = parser_data[position]
                parser_correct += parser_key == gold_key
                confusion.append(("parser", gold_text, parser_text))

        fingerprint: _Fingerprint = (
            tuple(
                tuple((category[0], probability) for category, probability in token)
                for token in tagger_data
            ),
            None
            if parser_data is None
            else tuple(category for category, _ in parser_data),
            tuple(category for category, _ in gold_data),
        )

        return _Item(
            fingerprint,
            tokens=expected,
            tagger_correct=tagger_correct,
            parser_correct=parser_correct,
            parsed=parser_data is not None,
            multi_correct=multi_correct,
            survivors=survivors,
            confusion=tuple(confusion),
        )

    def _category(
        self,
        category: Term,
        field: str,
        position: int,
    ) -> tuple[TermKey, str]:
        """Validate and serialize one category."""
        if not isinstance(category, Term):
            msg = f"{field} category {position} is not a term: {category!r}"
            raise TypeError(msg)

        return category.key(), self._syntax.render(self._store, category)

    @staticmethod
    def _require_length(
        key: ItemId,
        field: str,
        expected: int,
        received: int,
    ) -> None:
        """Reject a sentence-level category-count mismatch."""
        if received != expected:
            raise SupertagLengthError(
                item_id=key,
                field=field,
                expected=expected,
                received=received,
            )

    @staticmethod
    def _token_result(
        numerator: int,
        denominator: int,
        *,
        component: str,
        extra: Mapping[str, float] | None = None,
    ) -> MetricResult:
        """Return one token-micro result."""
        components = {
            component: float(numerator),
            "tokens": float(denominator),
            **({} if extra is None else dict(extra)),
        }

        return MetricResult(
            None if denominator == 0 else numerator / denominator,
            denominator,
            _TOKEN_UNITS,
            components=components,
        )

    @override
    def __repr__(self) -> str:
        """Return the threshold and scored sentence count."""
        return f"SupertagAccuracy(beta={self._beta!r}, sentences={len(self._items)})"

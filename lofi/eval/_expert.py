"""Expert annotation packages and agreement over induced categories."""

import json
from collections import Counter
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data.datasets import artifact_dataset_plugin
from lofi.data.expert import ExpertRating, ExpertRatings
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._declared_alignment import fit_declared_alignment
from lofi.eval._held_out import derivation_dataset
from lofi.eval._lexical_type_evaluator import LexicalTypeEvaluator
from lofi.eval._ports import EvaluationArtifact
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.expert import (
    ExpertCategoryEvaluatorSettings,
    ExpertCategoryEvaluatorSpec,
)
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import ItemId, JsonValue
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval._held_out import HeldOutEvaluationContext
    from lofi.eval._ports import HeldOutData, HeldOutItems
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.evaluators.alignment import AlignmentSettings
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

__all__ = [
    "PLUGIN",
    "CategoryExample",
    "CategoryOccurrence",
    "ExpertAgreementRow",
    "ExpertAnnotationPackage",
    "ExpertCategory",
    "ExpertCategoryEvaluator",
    "ExpertEvaluation",
    "ExpertGold",
    "ExpertTaxonomyRow",
    "evaluate_experts",
    "export_annotation_package",
]

_CATEGORY_UNIT: Final = "induced-category"


class CategoryExample(NamedTuple):
    """One derivation illustrating an induced category's use."""

    item_id: ItemId
    words: tuple[str, ...]
    derivation: str

    def to_spec(self) -> dict[str, JsonValue]:
        """Return serializable annotation-package columns."""
        return {
            "item_id": self.item_id,
            "words": list(self.words),
            "derivation": self.derivation,
        }


class CategoryOccurrence(NamedTuple):
    """One induced category paired with its held-out gold category."""

    item_id: ItemId
    words: tuple[str, ...]
    induced: str
    gold: str
    derivation: str


class ExpertCategory(NamedTuple):
    """One category presented to annotators."""

    category_id: str
    aligned_gold: str | None
    occurrences: int
    examples: tuple[CategoryExample, ...]

    def to_spec(self) -> dict[str, JsonValue]:
        """Return serializable annotation-package columns."""
        return {
            "category_id": self.category_id,
            "aligned_gold": self.aligned_gold,
            "occurrences": self.occurrences,
            "examples": [example.to_spec() for example in self.examples],
        }


@final
class ExpertAnnotationPackage(FrozenValue):
    """A deterministic expert task over aligned induced categories."""

    _noun = "expert annotation package"
    __slots__ = ("_categories", "_digest")

    def __init__(self, categories: Sequence[ExpertCategory], /) -> None:
        held = tuple(categories)
        identifiers = tuple(category.category_id for category in held)

        if identifiers != tuple(sorted(identifiers)):
            msg = "an expert annotation package is not ordered by category id"
            raise ValueError(msg)

        if len(set(identifiers)) != len(identifiers):
            msg = "an expert annotation package repeats a category id"
            raise ValueError(msg)

        payload = {"kind": "expert-categories", "categories": self._spec(held)}
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        object.__setattr__(self, "_categories", held)
        object.__setattr__(self, "_digest", sha256(encoded.encode()).hexdigest())

    @staticmethod
    def _spec(categories: Sequence[ExpertCategory]) -> list[JsonValue]:
        """Return the category records shared by serialization and hashing."""
        return [category.to_spec() for category in categories]

    @property
    def categories(self) -> tuple[ExpertCategory, ...]:
        """Return annotation tasks in category-id order."""
        return self._categories

    @property
    def digest(self) -> str:
        """Return the SHA-256 binding ratings to these exact tasks."""
        return self._digest

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the complete annotation package."""
        return {
            "kind": "expert-categories",
            "digest": self._digest,
            "categories": self._spec(self._categories),
        }

    def serialize(self) -> str:
        """Return canonical UTF-8 JSON text."""
        return (
            json.dumps(
                self.to_spec(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    @override
    def __repr__(self) -> str:
        """Return a summary naming the task count and digest."""
        return (
            f"ExpertAnnotationPackage({len(self._categories)} categories, "
            f"digest={self._digest!r})"
        )


class ExpertAgreementRow(NamedTuple):
    """Krippendorff's nominal alpha and its category bootstrap interval."""

    result: MetricResult
    annotators: int
    categories: int

    def to_spec(self) -> dict[str, JsonValue]:
        """Return one registered evaluator row."""
        interval = self.result.ci

        return {
            "metric": "expert-rating-alpha",
            "value": self.result.value,
            "n": self.result.n,
            "annotators": self.annotators,
            "categories": self.categories,
            "resampling_unit": self.result.resampling.unit,
            "resampling_seed": self.result.resampling.seed,
            "ci_lo": None if interval is None else interval.low,
            "ci_hi": None if interval is None else interval.high,
            "ci_level": None if interval is None else interval.level,
        }


class ExpertTaxonomyRow(NamedTuple):
    """The prevalence of one declared expert error label."""

    error: str
    result: MetricResult
    categories: int

    def to_spec(self) -> dict[str, JsonValue]:
        """Return one error-taxonomy evaluator row."""
        interval = self.result.ci

        return {
            "metric": f"expert-error-{self.error}",
            "error": self.error,
            "value": self.result.value,
            "n": self.result.n,
            "categories": self.categories,
            "resampling_unit": self.result.resampling.unit,
            "resampling_seed": self.result.resampling.seed,
            "ci_lo": None if interval is None else interval.low,
            "ci_hi": None if interval is None else interval.high,
            "ci_level": None if interval is None else interval.level,
        }


class ExpertEvaluation(NamedTuple):
    """Agreement followed by the declared error-taxonomy table."""

    agreement: ExpertAgreementRow
    taxonomy: tuple[ExpertTaxonomyRow, ...]

    @property
    def rows(self) -> tuple[ExpertAgreementRow | ExpertTaxonomyRow, ...]:
        """Return every evaluator row in report order."""
        return (self.agreement, *self.taxonomy)


def export_annotation_package(
    occurrences: Iterable[CategoryOccurrence],
    alignment: Alignment,
    /,
    *,
    examples_per_category: int,
) -> ExpertAnnotationPackage:
    """Export aligned induced categories and deterministic example derivations."""
    limit = ensure_non_negative("an example-derivation limit", examples_per_category)

    if limit == 0:
        msg = "an annotation package needs at least one example per category"
        raise ValueError(msg)

    if not isinstance(alignment, Alignment):
        msg = f"an expert-package alignment is not one: {alignment!r}"
        raise TypeError(msg)

    grouped: dict[str, list[CategoryOccurrence]] = {}

    for position, occurrence in enumerate(occurrences):
        if not isinstance(occurrence, CategoryOccurrence):
            msg = f"category occurrence {position} is not one: {occurrence!r}"
            raise TypeError(msg)

        induced = ensure_name("an induced category", occurrence.induced)
        ensure_name("a gold category", occurrence.gold)
        ensure_name("an example derivation", occurrence.derivation)
        grouped.setdefault(induced, []).append(occurrence)

    categories = []

    for category_id, members in sorted(grouped.items()):
        examples = tuple(
            CategoryExample(member.item_id, member.words, member.derivation)
            for member in sorted(
                members,
                key=lambda row: (row.item_id, row.words, row.derivation, row.gold),
            )[:limit]
        )
        categories.append(
            ExpertCategory(
                category_id,
                alignment.apply(category_id),
                len(members),
                examples,
            )
        )

    if not categories:
        msg = "an expert annotation package contains no induced category"
        raise ValueError(msg)

    return ExpertAnnotationPackage(categories)


type _RatingGroup = tuple[str, tuple[ExpertRating, ...]]


def _groups(ratings: Sequence[ExpertRating]) -> tuple[_RatingGroup, ...]:
    """Group responses by category, retaining categories with two raters."""
    held: dict[str, list[ExpertRating]] = {}

    for response in ratings:
        held.setdefault(response.category_id, []).append(response)

    return tuple(
        (category, tuple(responses))
        for category, responses in sorted(held.items())
        if len(responses) >= 2
    )


def _alpha(groups: Sequence[_RatingGroup]) -> float:
    """Return Krippendorff's alpha under nominal disagreement."""
    coincidences: dict[str, float] = {}
    observed_disagreement = 0.0
    coincidence_total = 0.0

    for _, responses in groups:
        count = len(responses)

        if count < 2:
            continue

        labels = Counter(response.rating for response in responses)
        scale = count - 1
        coincidence_total += count

        for label, frequency in labels.items():
            coincidences[label] = coincidences.get(label, 0.0) + frequency

        observed_disagreement += (
            count * count - sum(frequency * frequency for frequency in labels.values())
        ) / scale

    if coincidence_total < 2:
        msg = "expert agreement needs at least two pairable ratings"
        raise ValueError(msg)

    observed = observed_disagreement / coincidence_total
    expected_numerator = sum(
        frequency * (coincidence_total - frequency)
        for frequency in coincidences.values()
    )
    expected = expected_numerator / (coincidence_total * (coincidence_total - 1))

    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0

    return 1.0 - observed / expected


def _prevalence(groups: Sequence[_RatingGroup], error: str) -> float:
    """Return the fraction of expert responses carrying one error label."""
    responses = tuple(response for _, group in groups for response in group)

    if not responses:
        msg = "an error prevalence needs at least one expert response"
        raise ValueError(msg)

    return sum(error in response.errors for response in responses) / len(responses)


def evaluate_experts(
    package: ExpertAnnotationPackage,
    ratings: ExpertRatings,
    taxonomy: Sequence[str],
    /,
    *,
    bootstrap: Bootstrap,
) -> ExpertEvaluation:
    """Compute nominal agreement and an error-taxonomy table."""
    if ratings.package_digest != package.digest:
        msg = (
            f"expert ratings name package {ratings.package_digest!r}, not "
            f"{package.digest!r}"
        )
        raise ValueError(msg)

    categories = {category.category_id for category in package.categories}
    unknown = sorted(
        {response.category_id for response in ratings.ratings} - categories
    )

    if unknown:
        msg = f"expert ratings name categories absent from their package: {unknown!r}"
        raise ValueError(msg)

    errors = tuple(ensure_name("an expert error label", label) for label in taxonomy)

    if len(set(errors)) != len(errors):
        msg = f"an expert error taxonomy repeats a label: {errors!r}"
        raise ValueError(msg)

    undeclared = sorted(
        {
            error
            for response in ratings.ratings
            for error in response.errors
            if error not in errors
        }
    )

    if undeclared:
        msg = f"expert ratings use undeclared error labels: {undeclared!r}"
        raise ValueError(msg)

    return _evaluate_ratings(ratings.ratings, errors, bootstrap=bootstrap)


def _evaluate_ratings(
    ratings: Sequence[ExpertRating],
    taxonomy: tuple[str, ...],
    /,
    *,
    bootstrap: Bootstrap,
) -> ExpertEvaluation:
    """Reduce already-bound responses to agreement and taxonomy rows."""
    grouped = _groups(ratings)

    if len(grouped) < 2:
        msg = "expert agreement needs at least two categories rated by two experts"
        raise ValueError(msg)

    if bootstrap.unit != _CATEGORY_UNIT:
        msg = f"expert agreement resamples {bootstrap.unit!r}, not {_CATEGORY_UNIT!r}"
        raise ValueError(msg)

    alpha = bootstrap.interval(grouped, _alpha)
    annotators = len({response.annotator for response in ratings})
    agreement = ExpertAgreementRow(
        MetricResult(
            alpha.observed,
            sum(len(group) for _, group in grouped),
            Resampling(_CATEGORY_UNIT, bootstrap.seed),
            components={
                "annotators": float(annotators),
                "categories": float(len(grouped)),
                "bootstrap_replicates": float(bootstrap.replicates),
            },
            ci=Interval(alpha.bca.low, alpha.bca.high, alpha.bca.level),
        ),
        annotators,
        len(grouped),
    )
    taxonomy_rows = []

    for error in taxonomy:
        drawn = bootstrap.interval(
            grouped,
            lambda sample, label=error: _prevalence(sample, label),
        )
        responses = sum(len(group) for _, group in grouped)
        taxonomy_rows.append(
            ExpertTaxonomyRow(
                error,
                MetricResult(
                    drawn.observed,
                    responses,
                    Resampling(_CATEGORY_UNIT, bootstrap.seed),
                    components={
                        "categories": float(len(grouped)),
                        "bootstrap_replicates": float(bootstrap.replicates),
                    },
                    ci=Interval(drawn.bca.low, drawn.bca.high, drawn.bca.level),
                ),
                len(grouped),
            )
        )

    return ExpertEvaluation(agreement, tuple(taxonomy_rows))


class ExpertGold(NamedTuple):
    """Gold lexical categories and a bracketed example derivation."""

    words: tuple[str, ...]
    categories: tuple[Term, ...]
    derivation: str


def _derivation_text(derivation: Derivation[str], /) -> str:
    """Render a gold derivation as a deterministic bracketed tree."""

    def visit(node_id: int) -> str:
        node = derivation.node(node_id)
        label = json.dumps(node.label, ensure_ascii=False)

        if not node.children:
            return label

        return f"({label} {' '.join(visit(child) for child in node.children)})"

    return visit(derivation.root)


class _ExpertMetric:
    """Accumulate expert-evaluation inputs over held-out lexical predictions."""

    __slots__ = (
        "_alignment",
        "_bound_categories",
        "_conventions",
        "_error",
        "_items",
        "_ratings",
        "_store",
        "_syntax",
    )

    def __init__(
        self,
        conventions: ConventionVector,
        grammar: GrammarAssembly,
        alignment: Alignment,
        ratings: ExpertRatings | None,
        *,
        error: str | None,
    ) -> None:
        self._conventions = conventions
        self._syntax = grammar.syntax
        self._store = grammar.store
        self._alignment = alignment
        self._bound_categories: frozenset[str] | None = None
        self._ratings = ratings
        self._error = error
        self._items: dict[ItemId, tuple[tuple[str, ...], ExpertGold]] = {}

    @property
    def name(self) -> str:
        """Return the registered metric name."""
        return (
            "expert-rating-alpha"
            if self._error is None
            else f"expert-error-{self._error}"
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions carried by the evaluator row."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that held-out item order does not affect expert rows."""
        return True

    def reset(self) -> None:
        """Forget every held-out item accumulated so far."""
        self._items = {}

    def update(self, pred: tuple[str, ...], gold: ExpertGold, *, key: ItemId) -> None:
        """Accumulate one item's predicted and gold lexical categories."""
        held = self._items.get(key)

        if held is not None and held != (pred, gold):
            msg = f"expert evaluation already saw another analysis for {key!r}"
            raise ValueError(msg)

        self._items[key] = pred, gold

    def package(self) -> ExpertAnnotationPackage:
        """Return the annotation package induced by accumulated held-out items."""
        occurrences = []

        for item_id, (predicted, gold) in sorted(self._items.items()):
            for induced, expected in zip(predicted, gold.categories, strict=True):
                occurrences.append(
                    CategoryOccurrence(
                        item_id,
                        gold.words,
                        induced,
                        self._syntax.render(self._store, expected),
                        gold.derivation,
                    )
                )

        return export_annotation_package(
            occurrences,
            self._alignment,
            examples_per_category=3,
        )

    def _relevant(self) -> tuple[ExpertRating, ...]:
        """Return responses for categories present in accumulated predictions."""
        if self._ratings is None:
            return ()

        categories = {
            induced
            for induced_categories, _ in self._items.values()
            for induced in induced_categories
        }
        return tuple(
            response
            for response in self._ratings.ratings
            if response.category_id in categories
        )

    def compute(self) -> MetricResult:
        """Return the selected agreement or taxonomy result."""
        if not self._items:
            return MetricResult(None, 0, Resampling("sentence"))

        if self._ratings is None:
            return MetricResult(None, len(self._items), Resampling("sentence"))

        if self._bound_categories is None:
            package = self.package()

            if package.digest != self._ratings.package_digest:
                msg = (
                    f"expert ratings name package {self._ratings.package_digest!r}, "
                    f"not the evaluated package {package.digest!r}"
                )
                raise ValueError(msg)

            self._bound_categories = frozenset(
                category.category_id for category in package.categories
            )
            unknown = sorted(
                {
                    response.category_id
                    for response in self._ratings.ratings
                    if response.category_id not in self._bound_categories
                }
            )

            if unknown:
                msg = (
                    "expert ratings name categories absent from their package: "
                    f"{unknown!r}"
                )
                raise ValueError(msg)

        else:
            present = {
                induced
                for induced_categories, _ in self._items.values()
                for induced in induced_categories
            }
            unbound = sorted(present - self._bound_categories)

            if unbound:
                msg = f"expert evaluation introduced unbound categories: {unbound!r}"
                raise ValueError(msg)

        relevant = self._relevant()
        grouped = _groups(relevant)

        if not grouped:
            return MetricResult(None, 0, Resampling("sentence"))

        value = (
            _alpha(grouped)
            if self._error is None
            else _prevalence(grouped, self._error)
        )
        return MetricResult(
            value,
            len(relevant),
            Resampling("sentence"),
            components={
                "annotators": float(len({row.annotator for row in relevant})),
                "categories": float(len(grouped)),
            },
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return the result after each held-out item is considered alone."""
        if self._ratings is None:
            return MappingProxyType({})

        pairable = {category for category, responses in _groups(self._ratings.ratings)}
        return MappingProxyType(
            {
                item_id: MetricResult(
                    1.0 if pairable.intersection(categories) else None,
                    1,
                    Resampling("sentence"),
                )
                for item_id, (categories, _) in sorted(self._items.items())
            }
        )


class ExpertCategoryEvaluator:
    """Publish lexical categories and score their bound expert judgments."""

    __slots__ = ("_grammar", "_lexical", "_metrics", "_ratings")

    def __init__(
        self,
        grammar: GrammarAssembly,
        *,
        readout: str,
        conventions: ConventionVector,
        alignment: AlignmentSettings,
        ratings: ExpertRatings | None,
        taxonomy: tuple[str, ...],
    ) -> None:
        lexical = LexicalTypeEvaluator(
            grammar,
            readout=readout,
            conventions=conventions,
            alignment=alignment,
        )
        rendered = Counter(
            grammar.syntax.render(grammar.store, entry.category)
            for form in grammar.lexicon.forms
            for entry in grammar.lexicon.entries(form)
        )
        category_alignment = fit_declared_alignment(
            Contingency(
                {(label, label): count for label, count in rendered.items()},
                split=alignment.fit_on,
            ),
            alignment,
        ).alignment
        declared = lexical.conventions.extended({"expert_dataset": "expert-ratings"})
        if ratings is not None:
            undeclared = sorted(
                {
                    error
                    for response in ratings.ratings
                    for error in response.errors
                    if error not in taxonomy
                }
            )

            if undeclared:
                msg = f"expert ratings use undeclared error labels: {undeclared!r}"
                raise ValueError(msg)

        self._lexical = lexical
        self._grammar = grammar
        self._ratings = ratings
        self._metrics = tuple(
            _ExpertMetric(
                declared,
                grammar,
                category_alignment,
                ratings,
                error=error,
            )
            for error in (None, *taxonomy)
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return conventions shared by every expert row."""
        return self._metrics[0].conventions

    @property
    def metrics(self) -> tuple[_ExpertMetric, ...]:
        """Return agreement followed by the declared taxonomy rows."""
        return self._metrics

    def annotation_package(self) -> ExpertAnnotationPackage:
        """Return categories accumulated by the first metric as an export package."""
        return self._metrics[0].package()

    def artifacts(self) -> tuple[EvaluationArtifact, ...]:
        """Return the annotation package only when no ratings were declared."""
        if self._ratings is not None:
            return ()

        package = self.annotation_package()
        return (
            EvaluationArtifact(
                "expert-categories.json", package.serialize().encode("utf-8")
            ),
        )

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[ExpertGold]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> ExpertGold:
        """Return gold lexical categories and a display derivation."""
        lexical = self._lexical.annotate(item, derivation)
        return ExpertGold(
            lexical.words, lexical.categories, _derivation_text(derivation)
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, tuple[str, ...]]:
        """Return rendered induced categories at each token position."""
        grammar = self._grammar
        return MappingProxyType(
            {
                item_id: tuple(
                    grammar.syntax.render(grammar.store, category)
                    for category in categories
                )
                for item_id, categories in self._lexical.predict(outputs).items()
            }
        )


def _build(
    settings: ExpertCategoryEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> ExpertCategoryEvaluator:
    """Build expert evaluation from its declared ratings dataset."""
    ratings = None

    if settings.ratings is not None:
        loaded = artifact_dataset_plugin(settings.ratings).load(settings.ratings)
        if not isinstance(loaded, ExpertRatings):
            msg = (
                "the expert-ratings dataset produced "
                f"{type(loaded).__name__}, not ExpertRatings"
            )
            raise TypeError(msg)
        ratings = loaded

    return ExpertCategoryEvaluator(
        context.grammar,
        readout=settings.readout,
        conventions=ConventionVector.from_settings(settings.conventions),
        alignment=settings.alignment,
        ratings=ratings,
        taxonomy=settings.taxonomy,
    )


PLUGIN: Final = ComponentPlugin(spec=ExpertCategoryEvaluatorSpec, build=_build)

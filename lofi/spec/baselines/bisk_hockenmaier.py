"""Declaration of the Bisk--Hockenmaier induction baseline."""

from functools import cache
from hashlib import sha256
from json import dumps, loads
from math import exp
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec._interpolation import lower_package_resource
from lofi.spec._loader import SpecLoader
from lofi.spec._rule import CombinatorSetSpec, TypeRaisingSpec
from lofi.spec.components.baseline import BaselineSystemSpec
from lofi.spec.components.run import ParseSpec
from lofi.spec.grammar import GrammarDecl, canonical_bytes
from lofi.spec.language import LanguageDecl, resolve_language
from lofi.spec.language import canonical_bytes as language_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from lofi._types import JsonValue
    from lofi.deduce import ParseSettings
__all__ = [
    "BiskHockenmaierBaselineSettings",
    "BiskHockenmaierBaselineSpec",
    "bisk_hockenmaier_grammar",
]

type PosRootPolicy = Literal["s-to-top-except-vbg"]
type NounArgumentPolicy = Literal["none"]
type TransitiveCategoryPolicy = Literal["prefer-(S\\N)/N-over-(S/N)\\N"]
type LexiconInduction = Literal["iterative"]
type ExpansionFactor = Literal["Lex", "Unary", "Left", "Right"]
type BiskEstimator = Literal["smoothed-k-best-em"]
type KBestRefresh = Literal["every-iteration"]
type ForestInterpolation = Literal["c-k-over-c-k-plus-c-full"]
type EventInitialization = Literal["uniform-observed-events"]
type StoppingRule = Literal["convergence"]
type KProvenance = Literal["published-development-section-00"]
type PosSource = Literal["gold"]
type DependencyConversion = Literal["johansson-nugues-ptb"]
type DependencyScoring = Literal["conll-2008-unlabeled-directed-attachment"]
type DependencyDirection = Literal["modifier-to-head"]
type ConjunctionConvention = Literal["bisk-hockenmaier"]

_DEFAULT_PARSE: Final = ParseSpec(retain="forest", reachability="roots")
_GRAMMAR_RESOURCE: Final = "lofi.resource:system/grammar/bisk-hockenmaier.yaml"
_DEFAULT_LANGUAGE: Final = "propositional"


@cache
def bisk_hockenmaier_grammar() -> GrammarDecl:
    """Load the packaged grammar the published system induces over."""
    path = Path(lower_package_resource(_GRAMMAR_RESOURCE))
    return SpecLoader().compose_traced(path, schema=GrammarDecl).spec


def _canonical_language[V](value: V, /) -> LanguageDecl:
    """Return a declared language with every mapping in sorted key order.

    A language read back from a key-sorted manifest then equals the one the
    fit declared.
    """
    declared = resolve_language(value)
    document = (
        loads(declared.model_dump_json())
        if isinstance(declared, LanguageDecl)
        else declared
    )
    return LanguageDecl.model_validate_json(dumps(document, sort_keys=True))


def _default_language() -> LanguageDecl:
    """Return the registered language the baseline grammar compiles within."""
    return _canonical_language(_DEFAULT_LANGUAGE)


def _json_array(values: tuple[str | int | None, ...], /) -> list[JsonValue]:
    """Return one explicitly JSON-typed copy of scalar declaration values."""
    result: list[JsonValue] = []
    result.extend(values)
    return result


class BiskHockenmaierBaselineSettings(NamedTuple):
    """Resolved controls for one Bisk--Hockenmaier baseline fit."""

    ccgbank_source: str
    dependency_gold_source: str
    reference_source: str
    reference_tolerance: float
    training_sections: tuple[int, ...]
    training_max_length: int
    development_section: int
    evaluation_section: int
    evaluation_length_bands: tuple[int | None, ...]
    length_excludes_punctuation: bool
    pos_source: PosSource
    grammar: GrammarDecl
    language: LanguageDecl
    nominal_category: str
    sentence_category: str
    coordination_category: str
    gold_sentence_atom: str
    gold_coordination_atom: str
    nominal_tags: frozenset[str]
    verbal_tags: frozenset[str]
    coordination_tags: frozenset[str]
    other_tags: frozenset[str]
    punctuation_tags: frozenset[str]
    pos_root_policy: PosRootPolicy
    gerund_tag: str
    noun_argument_policy: NounArgumentPolicy
    headed_or_modifier_arguments: tuple[str, ...]
    sentence_arguments: tuple[str, ...]
    second_order_modifiers: bool
    max_arity: int
    transitive_category_policy: TransitiveCategoryPolicy
    lexicon_induction: LexiconInduction
    category_induction_steps: int
    derived_constituent_pass: bool
    expansion_factors: tuple[ExpansionFactor, ...]
    em_max_iterations: int
    estimator: BiskEstimator
    k: int
    k_best_refresh: KBestRefresh
    forest_interpolation: ForestInterpolation
    initialization: EventInitialization
    probability_floor: float
    stopping_rule: StoppingRule
    convergence_tolerance: float
    k_provenance: KProvenance
    dependency_conversion: DependencyConversion
    dependency_scoring: DependencyScoring
    dependency_direction: DependencyDirection
    dependency_root: int
    conjunction_convention: ConjunctionConvention
    parse: ParseSettings[float]
    threads: int
    conventions: Mapping[str, JsonValue]


class BiskHockenmaierBaselineSpec(BaselineSystemSpec):
    """Declare the published CCG induction system and scoring treatment."""

    kind: Literal["bisk-hockenmaier"] = "bisk-hockenmaier"
    ccgbank_source: str
    dependency_gold_source: str
    reference_source: str
    reference_tolerance: float
    training_sections: tuple[int, ...] = tuple(range(2, 22))
    training_max_length: int = 10
    development_section: int = 0
    evaluation_section: int = 23
    evaluation_length_bands: tuple[int | None, ...] = (10, 20, None)
    length_excludes_punctuation: bool = True
    pos_source: PosSource = "gold"
    grammar: GrammarDecl = dx.field(default_factory=bisk_hockenmaier_grammar)
    language: LanguageDecl = dx.field(
        default_factory=_default_language, converter=_canonical_language
    )
    nominal_category: str
    sentence_category: str
    coordination_category: str
    gold_sentence_atom: str
    gold_coordination_atom: str
    nominal_tags: frozenset[str]
    verbal_tags: frozenset[str]
    coordination_tags: frozenset[str]
    other_tags: frozenset[str]
    punctuation_tags: frozenset[str]
    pos_root_policy: PosRootPolicy = "s-to-top-except-vbg"
    gerund_tag: str = "VBG"
    noun_argument_policy: NounArgumentPolicy = "none"
    headed_or_modifier_arguments: tuple[str, ...]
    sentence_arguments: tuple[str, ...]
    second_order_modifiers: bool = True
    max_arity: int = 3
    transitive_category_policy: TransitiveCategoryPolicy = (
        "prefer-(S\\N)/N-over-(S/N)\\N"
    )
    lexicon_induction: LexiconInduction = "iterative"
    category_induction_steps: int = 2
    derived_constituent_pass: bool = True
    expansion_factors: tuple[ExpansionFactor, ...] = ("Lex", "Unary", "Left", "Right")
    em_max_iterations: int
    estimator: BiskEstimator = "smoothed-k-best-em"
    k: Literal[100] = 100
    k_best_refresh: KBestRefresh = "every-iteration"
    forest_interpolation: ForestInterpolation = "c-k-over-c-k-plus-c-full"
    initialization: EventInitialization = "uniform-observed-events"
    probability_floor: float = exp(-6.0)
    stopping_rule: StoppingRule = "convergence"
    convergence_tolerance: float
    k_provenance: KProvenance = "published-development-section-00"
    dependency_conversion: DependencyConversion = "johansson-nugues-ptb"
    dependency_scoring: DependencyScoring = "conll-2008-unlabeled-directed-attachment"
    dependency_direction: DependencyDirection = "modifier-to-head"
    dependency_root: int = 0
    conjunction_convention: ConjunctionConvention = "bisk-hockenmaier"
    parse: ParseSpec = _DEFAULT_PARSE
    threads: int = 1

    @dx.validates(
        "ccgbank_source",
        "dependency_gold_source",
        "reference_source",
        "nominal_category",
        "sentence_category",
        "coordination_category",
        "gold_sentence_atom",
        "gold_coordination_atom",
        "gerund_tag",
    )
    @classmethod
    def _names_are_nonempty(cls, value: str) -> str:
        """Return a named source or POS tag."""
        return ensure_name("a baseline source or POS tag", value)

    @dx.validates(
        "headed_or_modifier_arguments",
        "sentence_arguments",
    )
    @classmethod
    def _category_inventory_is_named(
        cls, categories: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return a nonempty, distinct category inventory."""
        if not categories or len(set(categories)) != len(categories):
            msg = f"a baseline category inventory is empty or repeated: {categories!r}"
            raise ValueError(msg)
        for category in categories:
            ensure_name("a baseline category", category)
        return categories

    @dx.validates("training_sections")
    @classmethod
    def _sections_are_distinct(cls, sections: tuple[int, ...]) -> tuple[int, ...]:
        """Return nonempty, distinct, nonnegative training sections."""
        if not sections:
            msg = "a baseline training-section selection is empty"
            raise ValueError(msg)
        if any(isinstance(section, bool) or section < 0 for section in sections):
            msg = (
                "a baseline training section is negative or not an integer: "
                f"{sections!r}"
            )
            raise ValueError(msg)
        if len(sections) != len(set(sections)):
            msg = f"a baseline training section is repeated: {sections!r}"
            raise ValueError(msg)
        return sections

    @dx.validates("development_section", "evaluation_section")
    @classmethod
    def _held_out_section_is_nonnegative(cls, section: int) -> int:
        """Return a nonnegative development or evaluation section."""
        if isinstance(section, bool) or section < 0:
            msg = (
                "a baseline held-out section is negative or not an integer: "
                f"{section!r}"
            )
            raise ValueError(msg)
        return section

    @dx.validates("evaluation_length_bands")
    @classmethod
    def _evaluation_bands_are_ordered(
        cls, bands: tuple[int | None, ...]
    ) -> tuple[int | None, ...]:
        """Return increasing positive limits followed by one unbounded band."""
        if not bands:
            msg = "a baseline evaluation length-band selection is empty"
            raise ValueError(msg)
        bounded = tuple(limit for limit in bands if limit is not None)
        if any(isinstance(limit, bool) or limit < 1 for limit in bounded):
            msg = f"a baseline evaluation length band is not positive: {bands!r}"
            raise ValueError(msg)
        if tuple(sorted(set(bounded))) != bounded:
            msg = (
                "baseline evaluation length bands are not strictly increasing: "
                f"{bands!r}"
            )
            raise ValueError(msg)
        if bands.count(None) != 1 or bands[-1] is not None:
            msg = (
                "baseline evaluation length bands must end in one unbounded band: "
                f"{bands!r}"
            )
            raise ValueError(msg)
        return bands

    @dx.validates(
        "nominal_tags",
        "verbal_tags",
        "coordination_tags",
        "other_tags",
        "punctuation_tags",
    )
    @classmethod
    def _tags_are_named(cls, tags: frozenset[str]) -> frozenset[str]:
        """Return a nonempty set of named POS tags."""
        if not tags:
            msg = "a baseline POS class is empty"
            raise ValueError(msg)
        for tag in tags:
            ensure_name("a baseline POS tag", tag)
        return tags

    @dx.validates(
        "training_max_length",
        "max_arity",
        "category_induction_steps",
        "em_max_iterations",
    )
    @classmethod
    def _counts_are_positive(cls, value: int) -> int:
        """Return a positive induction count."""
        return ensure_positive("a baseline induction count", value)

    @dx.validates("threads")
    @classmethod
    def _threads_are_positive(cls, threads: int) -> int:
        """Return a positive count of concurrent chart-cell workers."""
        return ensure_positive("a baseline parse's threads", threads)

    @dx.validates("probability_floor")
    @classmethod
    def _probability_floor_is_positive(cls, floor: float) -> float:
        """Return a finite, positive probability floor."""
        held = ensure_finite("a baseline probability floor", floor)
        if not 0.0 < held <= 1.0:
            msg = f"a baseline probability floor is outside (0, 1]: {floor!r}"
            raise ValueError(msg)
        return held

    @dx.validates("convergence_tolerance", "reference_tolerance")
    @classmethod
    def _tolerance_is_nonnegative(cls, tolerance: float) -> float:
        """Return a finite, nonnegative convergence tolerance."""
        held = ensure_finite("a baseline convergence tolerance", tolerance)
        if held < 0.0:
            msg = f"a baseline convergence tolerance is negative: {tolerance!r}"
            raise ValueError(msg)
        return held

    @dx.model_validator()
    def _treatment_is_coherent(self) -> Self:
        """Return a declaration with disjoint POS groups and held-out sections."""
        groups = (
            self.nominal_tags,
            self.verbal_tags,
            self.coordination_tags,
            self.other_tags,
        )
        repeated = set().union(
            *(
                left & right
                for index, left in enumerate(groups)
                for right in groups[index + 1 :]
            )
        )
        if repeated:
            msg = f"baseline POS classes overlap: {sorted(repeated)!r}"
            raise ValueError(msg)
        declared_tags = set().union(*groups)
        if not self.punctuation_tags <= declared_tags:
            msg = (
                "baseline punctuation tags are outside the POS groups: "
                f"{sorted(self.punctuation_tags - declared_tags)!r}"
            )
            raise ValueError(msg)
        held_out = {self.development_section, self.evaluation_section}
        overlap = held_out & set(self.training_sections)
        if overlap:
            msg = f"baseline held-out sections occur in training: {sorted(overlap)!r}"
            raise ValueError(msg)
        if self.development_section == self.evaluation_section:
            msg = "the baseline development and evaluation sections are identical"
            raise ValueError(msg)
        if self.gerund_tag not in self.verbal_tags:
            msg = f"the gerund tag {self.gerund_tag!r} is not in the verbal POS group"
            raise ValueError(msg)
        self._grammar_is_coherent()
        primitives = set(self.grammar.signature.primitives)
        if set(self.headed_or_modifier_arguments) - primitives:
            msg = (
                "baseline head or modifier arguments are not atomic: "
                f"{self.headed_or_modifier_arguments!r}"
            )
            raise ValueError(msg)
        if set(self.sentence_arguments) - primitives:
            msg = (
                "baseline sentence arguments are not atomic: "
                f"{self.sentence_arguments!r}"
            )
            raise ValueError(msg)
        if self.expansion_factors != ("Lex", "Unary", "Left", "Right"):
            msg = (
                "the baseline expansion factors are incomplete or reordered: "
                f"{self.expansion_factors!r}"
            )
            raise ValueError(msg)
        if self.dependency_root != 0:
            msg = (
                f"the baseline dependency root is not token 0: {self.dependency_root!r}"
            )
            raise ValueError(msg)
        if self.parse.retain != "forest":
            msg = (
                "the baseline reads its induction supports from a packed forest; "
                "declare parse.retain: forest"
            )
            raise ValueError(msg)
        refusal = self.parse.semiring_refusal(
            "log-probability", consumer="the Bisk--Hockenmaier baseline"
        )
        if refusal is not None:
            raise refusal
        return self

    def _grammar_is_coherent(self) -> None:
        """Refuse a grammar whose atoms, goal or budget the baseline cannot read."""
        grammar = self.grammar
        goal = grammar.goal
        if goal.kind != "root" or len(goal.categories) != 1:
            msg = (
                "the baseline grammar's goal is not one root category: "
                f"{goal.kind!r} over {goal.categories!r}"
            )
            raise ValueError(msg)
        roles = (
            self.nominal_category,
            self.sentence_category,
            self.coordination_category,
            goal.categories[0],
        )
        if len(set(roles)) != len(roles):
            msg = (
                "the baseline nominal, sentence, coordination and goal categories "
                f"are not distinct: {roles!r}"
            )
            raise ValueError(msg)
        if self.gold_sentence_atom == self.gold_coordination_atom:
            msg = (
                "the baseline gold sentence and coordination atoms are one atom: "
                f"{self.gold_sentence_atom!r}"
            )
            raise ValueError(msg)
        primitives = grammar.signature.primitives
        if set(roles) != set(primitives):
            msg = (
                "the baseline grammar's primitives differ from its nominal, "
                f"sentence, coordination and goal categories: {primitives!r}"
            )
            raise ValueError(msg)
        conjunctions = grammar.signature.conjunctions
        if conjunctions is None or self.coordination_category not in (
            conjunctions.markers
        ):
            msg = (
                f"the baseline coordination category {self.coordination_category!r} "
                "is not a conjunction marker of its grammar"
            )
            raise ValueError(msg)
        if any(
            (
                isinstance(rule, CombinatorSetSpec)
                and rule.type_raising
                and (rule.raising_targets is None or rule.raising_arguments is None)
            )
            or (
                isinstance(rule, TypeRaisingSpec)
                and (rule.targets is None or rule.arguments is None)
            )
            for rule in grammar.rules.rules
        ):
            msg = (
                "the baseline bounds its category set by finite raising targets "
                "and arguments; declare both for every type-raising rule"
            )
            raise ValueError(msg)
        if grammar.budget.threads != 1:
            msg = (
                "the baseline parse takes its threads from the baseline's own "
                "threads field; leave the grammar budget's threads at 1"
            )
            raise ValueError(msg)

    @override
    def metrics(self) -> tuple[str, ...]:
        """Return supertagging then attachment within each length band."""
        return tuple(
            f"{metric}-{band}"
            for limit in self.evaluation_length_bands
            for band in (f"length-{limit}" if limit is not None else "all-lengths",)
            for metric in ("supertagging", "dependencies")
        )

    @override
    def conventions(self) -> Mapping[str, JsonValue]:
        """Return the induced grammar and dependency scoring conventions."""
        factors: dict[str, JsonValue] = {
            "formalism": "ccg",
            "grammar_digest": "sha256:"
            + sha256(canonical_bytes(self.grammar)).hexdigest(),
            "language_digest": (
                "sha256:" + sha256(language_bytes(self.language)).hexdigest()
            ),
            "training_sections": _json_array(self.training_sections),
            "training_max_length": self.training_max_length,
            "development_section": self.development_section,
            "evaluation_section": self.evaluation_section,
            "evaluation_length_bands": _json_array(self.evaluation_length_bands),
            "nominal_category": self.nominal_category,
            "sentence_category": self.sentence_category,
            "coordination_category": self.coordination_category,
            "gold_sentence_atom": self.gold_sentence_atom,
            "gold_coordination_atom": self.gold_coordination_atom,
            "nominal_tags": _json_array(tuple(sorted(self.nominal_tags))),
            "verbal_tags": _json_array(tuple(sorted(self.verbal_tags))),
            "coordination_tags": _json_array(tuple(sorted(self.coordination_tags))),
            "other_tags": _json_array(tuple(sorted(self.other_tags))),
            "punctuation_tags": _json_array(tuple(sorted(self.punctuation_tags))),
            "pos_root_policy": self.pos_root_policy,
            "gerund_tag": self.gerund_tag,
            "noun_argument_policy": self.noun_argument_policy,
            "headed_or_modifier_arguments": _json_array(
                self.headed_or_modifier_arguments
            ),
            "sentence_arguments": _json_array(self.sentence_arguments),
            "second_order_modifiers": self.second_order_modifiers,
            "max_arity": self.max_arity,
            "transitive_category_policy": self.transitive_category_policy,
            "lexicon_induction": self.lexicon_induction,
            "category_induction_steps": self.category_induction_steps,
            "derived_constituent_pass": self.derived_constituent_pass,
            "expansion_factors": _json_array(self.expansion_factors),
            "em_max_iterations": self.em_max_iterations,
            "estimator": self.estimator,
            "k": self.k,
            "k_provenance": self.k_provenance,
            "k_best_refresh": self.k_best_refresh,
            "forest_interpolation": self.forest_interpolation,
            "initialization": self.initialization,
            "probability_floor": self.probability_floor,
            "stopping_rule": self.stopping_rule,
            "convergence_tolerance": self.convergence_tolerance,
            "dependency_conversion": self.dependency_conversion,
            "dependency_scoring": self.dependency_scoring,
            "dependency_direction": self.dependency_direction,
            "dependency_root": self.dependency_root,
            "conjunction_convention": self.conjunction_convention,
            "length_excludes_punctuation": self.length_excludes_punctuation,
            "pos_source": self.pos_source,
            "reference_tolerance": self.reference_tolerance,
            "parse": cast("JsonValue", loads(self.parse.model_dump_json())),
        }
        return MappingProxyType(factors)

    def settings(self) -> BiskHockenmaierBaselineSettings:
        """Return the immutable controls consumed by the registered runner."""
        return BiskHockenmaierBaselineSettings(
            ccgbank_source=self.ccgbank_source,
            dependency_gold_source=self.dependency_gold_source,
            reference_source=self.reference_source,
            reference_tolerance=self.reference_tolerance,
            training_sections=self.training_sections,
            training_max_length=self.training_max_length,
            development_section=self.development_section,
            evaluation_section=self.evaluation_section,
            evaluation_length_bands=self.evaluation_length_bands,
            length_excludes_punctuation=self.length_excludes_punctuation,
            pos_source=self.pos_source,
            grammar=self.grammar,
            language=self.language,
            nominal_category=self.nominal_category,
            sentence_category=self.sentence_category,
            coordination_category=self.coordination_category,
            gold_sentence_atom=self.gold_sentence_atom,
            gold_coordination_atom=self.gold_coordination_atom,
            nominal_tags=self.nominal_tags,
            verbal_tags=self.verbal_tags,
            coordination_tags=self.coordination_tags,
            other_tags=self.other_tags,
            punctuation_tags=self.punctuation_tags,
            pos_root_policy=self.pos_root_policy,
            gerund_tag=self.gerund_tag,
            noun_argument_policy=self.noun_argument_policy,
            headed_or_modifier_arguments=self.headed_or_modifier_arguments,
            sentence_arguments=self.sentence_arguments,
            second_order_modifiers=self.second_order_modifiers,
            max_arity=self.max_arity,
            transitive_category_policy=self.transitive_category_policy,
            lexicon_induction=self.lexicon_induction,
            category_induction_steps=self.category_induction_steps,
            derived_constituent_pass=self.derived_constituent_pass,
            expansion_factors=self.expansion_factors,
            em_max_iterations=self.em_max_iterations,
            estimator=self.estimator,
            k=self.k,
            k_best_refresh=self.k_best_refresh,
            forest_interpolation=self.forest_interpolation,
            initialization=self.initialization,
            probability_floor=self.probability_floor,
            stopping_rule=self.stopping_rule,
            convergence_tolerance=self.convergence_tolerance,
            k_provenance=self.k_provenance,
            dependency_conversion=self.dependency_conversion,
            dependency_scoring=self.dependency_scoring,
            dependency_direction=self.dependency_direction,
            dependency_root=self.dependency_root,
            conjunction_convention=self.conjunction_convention,
            parse=cast("ParseSettings[float]", self.parse.build()),
            threads=self.threads,
            conventions=self.conventions(),
        )

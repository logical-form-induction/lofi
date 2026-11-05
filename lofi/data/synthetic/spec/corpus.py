"""The synthetic corpus declaration and the boundary that lowers it.

A declaration composes a grammar declaration with the controls a draw takes:
the number of items, the seed, the length distribution, the ambiguity
target, the corruption, the label noise, the interpretation, the readings
target and the ill-typed forms. Lowering checks it by compiling the grammar
and the logical-form language and building the interior request values,
then draws the corpus and returns it beside the declaration's canonical
bytes.
"""

from hashlib import sha256
from json import dumps, loads
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, Self, cast, final, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.data.datasets import DatasetPlugin, DatasetVocabulary, RefusedTraining
from lofi.data.synthetic._corrupt import CorruptionContext
from lofi.data.synthetic._injection import IllTypingContext
from lofi.data.synthetic._lengths import (
    NegativeBinomialLength,
    OutOfSupportPolicy,
    TabularLength,
    UniformLength,
)
from lofi.data.synthetic._resemblance import LexiconEstimate, estimate
from lofi.data.synthetic._scope import UndecidedScopePolicy
from lofi.data.synthetic._semantics import SemanticReport
from lofi.data.synthetic._values import (
    AmbiguityTarget,
    Corruption,
    CorruptionDraw,
    LabelNoise,
    NegativeDataPolicy,
    NegativeDataRegime,
    ScopeTarget,
    ShapeSchedule,
    ShapeTarget,
    SyntheticDataset,
    Target,
)
from lofi.data.synthetic.generator import RuleWeighting, SyntheticGenerator
from lofi.data.synthetic.spec._splits import (
    split_assignments,
    training_occurrences,
    validate_retained,
    validate_training_split,
)
from lofi.data.synthetic.spec._validation import nonempty as _nonempty
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    lower_interpretation,
)
from lofi.data.synthetic.spec.operators import (
    WeightedCorruptionOperatorSpec,
    WeightedIllTypingOperatorSpec,
)
from lofi.data.synthetic.training import (
    CorpusSentences,
    corpus_featurizer_resources,
    corpus_sentences,
)
from lofi.deduce import ParsePolicy, Retain
from lofi.spec import (
    DeclarationError,
    ensure_data_seed,
    fixed_seed,
    resolved_seed,
)
from lofi.spec.components.data import (
    TRAINING_SPLIT,
    DatasetSpec,
    TrainingBatchKind,
)
from lofi.spec.grammar import (
    ClassSpec,
    GrammarIso,
    GrammarSpec,
    LexiconSpec,
    LexiconStructure,
    ParametricAmbiguityLawSpec,
    PowerLawRankLawSpec,
    TabularJointSpec,
    declare_grammar,
)
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl, LanguageIso

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue
    from lofi.data.datasets import ReservedForms
    from lofi.data.synthetic._corrupt import CorruptionOperator
    from lofi.data.synthetic._injection import IllTypingOperator
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._lengths import LengthDistribution
    from lofi.data.synthetic._ports import SourceLexicon
    from lofi.judge import RuleSet
    from lofi.lexicon import StaticLexicon
    from lofi.spec.grammar import CategoryMeasure, GrammarAssembly
    from lofi.terms import CategoryResolution, TargetSet, Term, TermStore, TermSyntax

__all__ = [
    "PLUGIN",
    "SOURCE_VERSION",
    "TARGET_VERSION",
    "AmbiguitySpec",
    "CheckedSyntheticCorpus",
    "CorruptionOperatorsSpec",
    "CorruptionProportionSpec",
    "CorruptionSpec",
    "DerivationSamplerSpec",
    "EstimatedClassSpec",
    "IllTypingOperatorsSpec",
    "IllTypingSpec",
    "LabelNoiseRateSpec",
    "LabelNoiseSpec",
    "LengthMassSpec",
    "LengthSpec",
    "LexiconEstimateSpec",
    "NegativeBinomialLengthSpec",
    "NegativeDataSpec",
    "ResemblingLexiconSpec",
    "ScopeSpec",
    "ShapeScheduleSpec",
    "ShapeTargetSpec",
    "SyntheticCorpusDecl",
    "SyntheticCorpusIso",
    "SyntheticCorpusLowerer",
    "TabularLengthSpec",
    "TargetSpec",
    "UniformDerivationSamplerSpec",
    "UniformLengthSpec",
    "WeightedRulesDerivationSamplerSpec",
    "canonical_bytes",
    "fingerprint",
]

SOURCE_VERSION: Final = "lofi-synthetic-corpus-decl"
"""The declaration language version the lowerer accepts."""

TARGET_VERSION: Final = "lofi-synthetic-corpus-native"
"""The native dataset version the lowerer produces."""

_DEFAULT_CORRUPTION_ATTEMPTS: Final = 64
"""Candidate budget shared by string and logical-form corruption declarations."""

_MINIMUM_LENGTH: Final = 1
"""The shared lower bound of every positive token-count distribution."""


class TargetSpec(dx.Model, extra="forbid"):
    """A value to aim at and the half-width of the band that counts as a hit.

    Attributes
    ----------
    value : float
        The aimed-at value.
    tolerance : float
        How far from the value a measurement may fall and still be admitted.
    """

    value: float
    tolerance: float

    __axioms__ = (dx.axiom("tolerance >= 0", message="tolerance must be nonnegative"),)

    @dx.validates("value", "tolerance")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite number."""
        return ensure_finite("a target", number)

    def build(self) -> Target:
        """Return the interior target."""
        return Target(self.value, self.tolerance)


class AmbiguitySpec(dx.Model, extra="forbid"):
    """What a draw aims at in lexical and derivational ambiguity.

    Attributes
    ----------
    lexical : TargetSpec | None
        The mean number of lexical categories per word token of a string, or
        nothing to leave lexical ambiguity uncontrolled.
    derivations : TargetSpec | None
        The number of root derivations of a string, or nothing to leave the
        derivation count uncontrolled.
    attempts : int
        How many draws one accepted string may take before the target is
        declared unreachable.
    undecided : UndecidedScopePolicy
        Whether an equivalence the oracle cannot decide keeps readings apart
        or counts them as equivalent.
    """

    lexical: TargetSpec | None = None
    derivations: TargetSpec | None = None
    attempts: int

    __axioms__ = (dx.axiom("attempts > 0", message="attempts must be positive"),)

    def build(self) -> AmbiguityTarget:
        """Return the interior ambiguity target."""
        return AmbiguityTarget(
            lexical=None if self.lexical is None else self.lexical.build(),
            derivations=(
                None if self.derivations is None else self.derivations.build()
            ),
            attempts=self.attempts,
        )


class ScopeSpec(dx.Model, extra="forbid"):
    """What a draw aims at in logical-form readings per string.

    Each equivalence decision runs under the declared interpretation's clock.

    Attributes
    ----------
    readings : TargetSpec | None
        How many logically distinct forms a string has, or nothing to
        measure the readings without banding them.
    derivations : int
        The most derivations one string's readings are read from. A drawn
        string with more is discarded and drawn again.
    attempts : int
        How many draws one accepted string may take before the target is
        declared unreachable.
    """

    readings: TargetSpec | None = None
    derivations: int
    attempts: int
    undecided: UndecidedScopePolicy = "distinct"

    __axioms__ = (
        dx.axiom("derivations > 0", message="derivations must be positive"),
        dx.axiom("attempts > 0", message="attempts must be positive"),
    )

    def build(self) -> ScopeTarget:
        """Return the interior scope target."""
        return ScopeTarget(
            readings=None if self.readings is None else self.readings.build(),
            derivations=self.derivations,
            attempts=self.attempts,
            undecided=self.undecided,
        )


class ShapeTargetSpec(dx.Model, extra="forbid"):
    """Bands sampled derivations must meet along structural dimensions."""

    depth: TargetSpec | None = None
    right_branching: TargetSpec | None = None
    slash_balance: TargetSpec | None = None
    arity: TargetSpec | None = None
    attempts: int

    __axioms__ = (dx.axiom("attempts > 0", message="attempts must be positive"),)

    @dx.model_validator()
    def _targets_have_valid_ranges(self) -> Self:
        """Require at least one target and valid proportion/count centers."""
        if all(
            target is None
            for target in (
                self.depth,
                self.right_branching,
                self.slash_balance,
                self.arity,
            )
        ):
            msg = "a derivation-shape target names no measurement"
            raise ValueError(msg)

        for name, target in (
            ("right_branching", self.right_branching),
            ("slash_balance", self.slash_balance),
        ):
            if target is not None and not 0.0 <= target.value <= 1.0:
                msg = f"{name} target must lie between zero and one"
                raise ValueError(msg)

        if self.depth is not None and self.depth.value <= 0.0:
            msg = "depth target must be positive"
            raise ValueError(msg)

        if self.arity is not None and self.arity.value < 1.0:
            msg = "arity target must be at least one"
            raise ValueError(msg)

        return self

    def build(self) -> ShapeTarget:
        """Return the interior derivation-shape target."""
        return ShapeTarget(
            depth=None if self.depth is None else self.depth.build(),
            right_branching=(
                None if self.right_branching is None else self.right_branching.build()
            ),
            slash_balance=(
                None if self.slash_balance is None else self.slash_balance.build()
            ),
            arity=None if self.arity is None else self.arity.build(),
            attempts=self.attempts,
        )


class ShapeScheduleSpec(dx.Model, extra="forbid"):
    """Select derivation-shape targets by corpus split."""

    default: ShapeTargetSpec | None = None
    splits: dict[str, ShapeTargetSpec] = dx.field(default_factory=dict)

    @dx.validates("splits")
    @classmethod
    def _splits_are_named(
        cls, values: dict[str, ShapeTargetSpec]
    ) -> dict[str, ShapeTargetSpec]:
        """Return split targets in a stable name order."""
        return {
            ensure_name("a shape-schedule split", name): values[name]
            for name in sorted(values)
        }

    @dx.model_validator()
    def _names_a_target(self) -> Self:
        """Require a default or at least one split target."""
        if self.default is None and not self.splits:
            msg = "a derivation-shape schedule names no target"
            raise ValueError(msg)
        return self

    def build(self) -> ShapeSchedule:
        """Return the immutable split-conditioned schedule."""
        return ShapeSchedule(
            None if self.default is None else self.default.build(),
            {name: target.build() for name, target in self.splits.items()},
        )


class _LengthDeclaration(Protocol):
    """A declaration that builds a token-count distribution."""

    def build(self) -> LengthDistribution:
        """Return the interior distribution."""
        ...


class LengthSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A distribution over positive token counts."""


class UniformLengthSpec(LengthSpec, extra="forbid"):
    """Token counts equally likely at every value between the bounds."""

    kind: Literal["uniform"] = "uniform"
    min_length: int = _MINIMUM_LENGTH
    max_length: int

    __axioms__ = (
        dx.axiom("min_length > 0", message="min_length must be positive"),
        dx.axiom(
            "min_length <= max_length",
            message="max_length must be at least min_length",
        ),
    )

    def build(self) -> LengthDistribution:
        """Return the uniform distribution over the bounds."""
        return UniformLength(self.min_length, self.max_length)


class NegativeBinomialLengthSpec(LengthSpec, extra="forbid"):
    """Token counts a negative binomial above the lower bound.

    Attributes
    ----------
    r : float
        The number of successes the trials run until, positive and not
        necessarily whole.
    p : float
        The probability each trial succeeds with, strictly between zero and
        one.
    """

    kind: Literal["negative-binomial"] = "negative-binomial"
    min_length: int = _MINIMUM_LENGTH
    max_length: int
    r: float
    p: float
    out_of_support: OutOfSupportPolicy = "resample"

    __axioms__ = (
        dx.axiom("min_length > 0", message="min_length must be positive"),
        dx.axiom(
            "min_length <= max_length",
            message="max_length must be at least min_length",
        ),
        dx.axiom("r > 0", message="r must be positive"),
        dx.axiom("p > 0 and p < 1", message="p must lie strictly between 0 and 1"),
    )

    @dx.validates("r", "p")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite shape parameter."""
        return ensure_finite("a negative binomial parameter", number)

    def build(self) -> LengthDistribution:
        """Return the negative binomial shifted to the lower bound."""
        return NegativeBinomialLength(
            self.r,
            self.p,
            min_length=self.min_length,
            max_length=self.max_length,
            out_of_support=self.out_of_support,
        )


class LengthMassSpec(dx.Model, extra="forbid"):
    """One token count and its unnormalized probability mass."""

    length: int
    mass: float

    __axioms__ = (
        dx.axiom("length > 0", message="a tabular length must be positive"),
        dx.axiom("mass >= 0", message="a tabular length mass must be nonnegative"),
    )

    @dx.validates("mass")
    @classmethod
    def _mass_is_finite(cls, mass: float) -> float:
        """Return a finite probability mass."""
        return ensure_finite("a tabular length mass", mass)


class TabularLengthSpec(LengthSpec, extra="forbid"):
    """A finite distribution over explicitly weighted token counts."""

    kind: Literal["tabular"] = "tabular"
    masses: tuple[LengthMassSpec, ...]

    @dx.model_validator()
    def _masses_form_a_distribution(self) -> Self:
        """Reject duplicate lengths and tables without positive mass."""
        lengths = tuple(row.length for row in self.masses)

        if len(set(lengths)) != len(lengths):
            msg = f"a tabular length distribution repeats a length: {lengths!r}"
            raise ValueError(msg)

        TabularLength({row.length: row.mass for row in self.masses})

        return self

    def build(self) -> LengthDistribution:
        """Return the normalized finite distribution."""
        return TabularLength({row.length: row.mass for row in self.masses})


class _DerivationSamplerDeclaration(Protocol):
    """A declaration that supplies rule-application mass."""

    def build(self, rules: RuleSet[Term], /) -> RuleWeighting:
        """Return the weighting checked against the compiled rule inventory."""
        ...


class _DeclaredRuleWeighting(FrozenValue):
    """The checked relative mass of every rule in one grammar."""

    __slots__ = ("_weights",)

    _noun = "rule weighting"

    def __init__(self, weights: dict[str, float]) -> None:
        object.__setattr__(self, "_weights", MappingProxyType(dict(weights)))

    def weight(self, rule: str, /) -> float:
        """Return the declared factor for the named rule."""
        return self._weights[rule]


class _UniformRuleWeighting(FrozenValue):
    """The neutral mass shared by every rule application."""

    __slots__ = ()

    _noun = "uniform rule weighting"

    def weight(self, rule: str, /) -> float:
        """Return the neutral factor for the named rule."""
        del rule

        return 1.0


_UNIFORM_RULE_WEIGHTING: Final[RuleWeighting] = _UniformRuleWeighting()


class DerivationSamplerSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """How rule applications contribute mass when derivations are drawn."""


class UniformDerivationSamplerSpec(DerivationSamplerSpec, extra="forbid"):
    """Give every rule application equal local mass."""

    kind: Literal["uniform"] = "uniform"

    def build(self, rules: RuleSet[Term], /) -> RuleWeighting:
        """Return the neutral weighting."""
        del rules

        return _UNIFORM_RULE_WEIGHTING


class WeightedRulesDerivationSamplerSpec(DerivationSamplerSpec, extra="forbid"):
    """Give each declared grammar rule its own positive local mass."""

    kind: Literal["weighted-rules"] = "weighted-rules"
    weights: dict[str, float]

    __axioms__ = (
        dx.axiom("len(weights) > 0", message="rule weights must not be empty"),
    )

    @dx.validates("weights")
    @classmethod
    def _weights_are_named_positive_and_finite(
        cls, weights: dict[str, float]
    ) -> dict[str, float]:
        """Return rule weights with named keys and positive finite values."""
        checked: dict[str, float] = {}

        for name, value in weights.items():
            held_name = ensure_name("a weighted rule", name)
            held_value = ensure_finite(f"the weight of rule {name!r}", value)

            if held_value <= 0.0:
                msg = f"the weight of rule {name!r} is not positive: {value!r}"
                raise ValueError(msg)

            checked[held_name] = held_value

        return checked

    def build(self, rules: RuleSet[Term], /) -> RuleWeighting:
        """Return weights that name every compiled rule exactly once.

        Raises
        ------
        DeclarationError
            If a compiled rule has no weight or a weight names no compiled rule.
        """
        available = frozenset(rule.name for rule in rules)
        declared = frozenset(self.weights)
        missing = sorted(available - declared)
        unknown = sorted(declared - available)

        if missing or unknown:
            faults: list[str] = []

            if missing:
                faults.append(f"missing {', '.join(repr(name) for name in missing)}")

            if unknown:
                faults.append(f"unknown {', '.join(repr(name) for name in unknown)}")

            msg = "rule weights do not match the compiled grammar: " + "; ".join(faults)
            raise DeclarationError(msg)

        return _DeclaredRuleWeighting(self.weights)


class NegativeDataSpec(dx.Model, extra="forbid"):
    """Declare how certified negatives are presented to training.

    Attributes
    ----------
    regime : NegativeDataRegime
        Whether negatives retain their label, enter positive-only data, or
        undergo independent label flips.
    noise : float
        Flip probability in the ``label-flips`` regime. Fixed-label regimes
        require zero.
    """

    regime: NegativeDataRegime = "labeled-negatives"
    noise: float = 0.0

    __axioms__ = (
        dx.axiom("noise >= 0 and noise <= 1", message="noise must lie between 0 and 1"),
    )

    @dx.validates("noise")
    @classmethod
    def _noise_is_finite(cls, rate: float) -> float:
        """Return a finite label-noise rate."""
        return ensure_finite("a negative-data noise rate", rate)

    @dx.model_validator()
    def _noise_belongs_to_the_stochastic_regime(self) -> Self:
        """Require noise only where the regime draws label flips."""
        if self.regime != "label-flips" and self.noise != 0.0:
            msg = f"negative-data regime {self.regime!r} does not take label noise"
            raise ValueError(msg)

        return self

    def build_policy(self) -> NegativeDataPolicy:
        """Return the interior negative-data policy."""
        return NegativeDataPolicy(self.regime, noise=self.noise)


class CorruptionProportionSpec(dx.Model, extra="forbid"):
    """The share to corrupt and how its item count is drawn."""

    value: float
    draw: CorruptionDraw = "exact"

    __axioms__ = (
        dx.axiom(
            "value >= 0 and value <= 1",
            message="proportion must lie between 0 and 1",
        ),
    )

    @dx.validates("value")
    @classmethod
    def _is_finite(cls, share: float) -> float:
        """Return a finite proportion."""
        return ensure_finite("a corruption proportion", share)


class CorruptionOperatorsSpec(dx.Model, extra="forbid"):
    """Weighted token corruption operators in report order."""

    entries: tuple[WeightedCorruptionOperatorSpec, ...]

    @dx.validates("entries")
    @classmethod
    def _names_each_operator_once(
        cls, entries: tuple[WeightedCorruptionOperatorSpec, ...]
    ) -> tuple[WeightedCorruptionOperatorSpec, ...]:
        if not entries:
            msg = "a corruption names no operator"
            raise ValueError(msg)
        names = tuple(entry.operator.kind for entry in entries)
        if len(set(names)) != len(names):
            msg = f"a corruption names one operator twice: {names!r}"
            raise ValueError(msg)
        return entries


class IllTypingOperatorsSpec(dx.Model, extra="forbid"):
    """Weighted ill-typing operators in report order."""

    entries: tuple[WeightedIllTypingOperatorSpec, ...]

    @dx.validates("entries")
    @classmethod
    def _names_each_operator_once(
        cls, entries: tuple[WeightedIllTypingOperatorSpec, ...]
    ) -> tuple[WeightedIllTypingOperatorSpec, ...]:
        if not entries:
            msg = "an ill-typing names no operator"
            raise ValueError(msg)
        names = tuple(entry.operator.kind for entry in entries)
        if len(set(names)) != len(names):
            msg = f"an ill-typing names one operator twice: {names!r}"
            raise ValueError(msg)
        return entries


class CorruptionSpec(dx.Model, extra="forbid"):
    """How much of a draw to corrupt with weighted installed operators.

    A corrupted string is labeled ungrammatical. The label-noise block of
    the corpus decides how often that label is observed as grammatical.

    Attributes
    ----------
    proportion : CorruptionProportionSpec
        The share of the items to emit and its count draw.
    operators : CorruptionOperatorsSpec
        Installed operators and their relative draw weights.
    attempts : int
        How many candidate strings one corrupted item may take before the
        corruption is declared exhausted.
    """

    proportion: CorruptionProportionSpec
    operators: CorruptionOperatorsSpec
    attempts: int = _DEFAULT_CORRUPTION_ATTEMPTS

    __axioms__ = (dx.axiom("attempts > 0", message="attempts must be positive"),)

    def build(self, context: CorruptionContext, /) -> Corruption[CorruptionOperator]:
        """Return the interior corruption request."""
        return Corruption(
            self.proportion.value,
            operators=tuple(entry.build(context) for entry in self.operators.entries),
            attempts=self.attempts,
            draw=self.proportion.draw,
        )


class IllTypingSpec(NegativeDataSpec):
    """How many logical forms to corrupt with weighted installed operators.

    Attributes
    ----------
    proportion : CorruptionProportionSpec
        The share of the grammatical items whose logical form to corrupt
        into one the type checker refuses.
    operators : IllTypingOperatorsSpec
        Installed operators and their relative draw weights.
    attempts : int
        How many candidate forms one injected form may take before the
        injection is declared exhausted.
    """

    proportion: CorruptionProportionSpec
    operators: IllTypingOperatorsSpec
    attempts: int = _DEFAULT_CORRUPTION_ATTEMPTS

    __axioms__ = (dx.axiom("attempts > 0", message="attempts must be positive"),)

    def build(self, context: IllTypingContext, /) -> Corruption[IllTypingOperator]:
        """Return the interior injection request."""
        return Corruption(
            self.proportion.value,
            operators=tuple(entry.build(context) for entry in self.operators.entries),
            attempts=self.attempts,
            draw=self.proportion.draw,
            policy=self.build_policy(),
        )


class LabelNoiseRateSpec(dx.Model, extra="forbid"):
    """One independently swappable grammaticality-label flip rate."""

    rate: float = 0.0

    __axioms__ = (
        dx.axiom("rate >= 0 and rate <= 1", message="rate must lie between 0 and 1"),
    )

    @dx.validates("rate")
    @classmethod
    def _is_finite(cls, rate: float) -> float:
        """Return a finite rate."""
        return ensure_finite("a label-noise rate", rate)


class LabelNoiseSpec(dx.Model, extra="forbid"):
    """The rates at which items' grammaticality labels are flipped, by true label.

    Equal rates flip every item alike. An ungrammatical rate with no
    grammatical rate contaminates the grammatical data with corrupted items
    observed as grammatical, and an ungrammatical rate of one observes every
    corrupted item as grammatical, which is positive-only data holding the
    corruption share of bad strings.

    Attributes
    ----------
    grammatical : LabelNoiseRateSpec
        The probability that an item the grammar derives is observed as
        ungrammatical.
    ungrammatical : LabelNoiseRateSpec
        The probability that a corrupted item is observed as grammatical.
    """

    grammatical: LabelNoiseRateSpec = dx.field(default_factory=LabelNoiseRateSpec)
    ungrammatical: LabelNoiseRateSpec = dx.field(default_factory=LabelNoiseRateSpec)

    def build(self) -> LabelNoise:
        """Return the interior label-noise request."""
        return LabelNoise(
            grammatical=self.grammatical.rate,
            ungrammatical=self.ungrammatical.rate,
        )


class EstimatedClassSpec(dx.Model, extra="forbid"):
    """One measured category class in a source lexicon."""

    category: str
    size: int
    share: float

    __axioms__ = (dx.axiom("size > 0", message="an estimated class is nonempty"),)

    @dx.validates("category")
    @classmethod
    def _category_is_named(cls, category: str) -> str:
        """Return a nonempty category name."""
        if not category:
            msg = "an estimated class needs a category"
            raise ValueError(msg)

        return category

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite positive token share."""
        held = ensure_finite("an estimated class's share", share)

        if held <= 0.0:
            msg = f"an estimated class's share is not positive: {share!r}"
            raise ValueError(msg)

        return held


class LexiconEstimateSpec(dx.Model, extra="forbid"):
    """The joint a source lexicon was measured to have, beside its digest.

    Attributes
    ----------
    digest : str
        The digest of the source the joint was read from. Two declarations
        seeded from different sources differ here, and so in their bytes.
    forms : int
        How many forms the source holds.
    exponent : float
        The exponent fitted to the rank-frequency curve of the source's
        forms.
    rows : tuple[EstimatedClassSpec, ...]
        Each category of the source with how many of its forms bear the
        category and the fraction of its token mass the category carries.
    """

    digest: str
    forms: int
    exponent: float
    rows: tuple[EstimatedClassSpec, ...]

    __axioms__ = (
        _nonempty("digest", "an estimate needs the digest of its source"),
        dx.axiom("forms > 0", message="a source lexicon holds a form"),
        dx.axiom("len(rows) > 0", message="an estimate needs a category"),
    )

    @dx.validates("exponent")
    @classmethod
    def _exponent_is_finite(cls, exponent: float) -> float:
        """Return a finite exponent.

        Parameters
        ----------
        exponent : float
            The measured exponent.

        Raises
        ------
        ValueError
            If the exponent is infinite or not a number.
        """
        return ensure_finite("an estimated exponent", exponent)

    @dx.model_validator()
    def _rows_fit_the_source(self) -> Self:
        """Reject a repeated category or a class larger than the source's inventory.

        Raises
        ------
        ValueError
            If two rows name one category, or a class bears more forms than
            the source holds.
        """
        categories = [row.category for row in self.rows]

        if len(set(categories)) != len(categories):
            msg = f"an estimate names a category twice: {categories!r}"
            raise ValueError(msg)

        for row in self.rows:
            if row.size > self.forms:
                msg = (
                    f"category {row.category!r} bears {row.size} of the "
                    f"source's {self.forms} forms"
                )
                raise ValueError(msg)

        return self

    @classmethod
    def measured[T](
        cls,
        source: SourceLexicon[T],
        render: Callable[[T], str],
        digest: str,
        /,
    ) -> Self:
        """Return the estimate read from a source lexicon.

        Parameters
        ----------
        source : SourceLexicon[T]
            The lexicon to read.
        render : Callable[[T], str]
            The text a category is recorded under, which must parse back to
            that category in the syntax the drawn lexicon is read in.
        digest : str
            The digest of the source.

        Returns
        -------
        Self
            The declaration of the joint the source was measured to have.

        Raises
        ------
        DeclarationError
            If the source holds no form, holds a form no category bears, or
            holds too few forms for a rank-frequency curve to be fitted.
        """
        try:
            measurement = estimate(source, render, digest)
        except ValueError as error:
            msg = f"the source lexicon cannot be estimated: {error}"
            raise DeclarationError(msg) from error

        return cls(
            digest=measurement.digest,
            forms=measurement.forms,
            exponent=measurement.exponent,
            rows=tuple(
                EstimatedClassSpec(category=category, size=size, share=share)
                for category, size, share in measurement.rows
            ),
        )

    def build(self) -> LexiconEstimate:
        """Return the interior estimate."""
        return LexiconEstimate(
            self.digest,
            tuple((row.category, row.size, row.share) for row in self.rows),
            self.forms,
            self.exponent,
        )


class ResemblingLexiconSpec(LexiconSpec, extra="forbid"):
    """A lexicon drawn at a declared size from the joint a source was measured to have.

    The categories and the joint are read from the source rather than
    declared. Below the source's inventory each class is subsampled in
    proportion, so the categories drawn are a subset of the source's; above
    it each class is extrapolated in the same proportion, and the exponent
    fitted to the source spreads a class's token share over the ranks it
    gains.

    Attributes
    ----------
    kind : Literal["resembling"]
        Registry tag.
    forms : int
        How many forms the drawn lexicon holds.
    estimate : LexiconEstimateSpec
        The joint the source was measured to have, with the source's digest.
    seed : int | str
        The seed of the draw, or ``derived`` to take the seed each point
        derives for the lexicon. Equal seeds draw equal lexicons.
    """

    kind: Literal["resembling"] = "resembling"
    forms: int
    estimate: LexiconEstimateSpec
    seed: int | str

    __axioms__ = (dx.axiom("forms > 0", message="a form inventory must be positive"),)

    @dx.validates("seed")
    @classmethod
    def _seed_is_a_data_seed(cls, value: int | str) -> int | str:
        """Return a nonnegative seed or the derived spelling."""
        return ensure_data_seed("a lexicon's seed", value)

    @override
    def with_derived_seed(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the lexicon drawn under the seed a point derives for it.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the lexicon in the run document.
        """
        return self.with_(seed=resolved_seed(self.seed, derive, f"{path}.seed"))

    def structure(self) -> LexiconStructure:
        """Return the structured declaration the drawn lexicon comes from.

        Returns
        -------
        LexiconStructure
            The estimate's joint scaled to the declared inventory, spelled
            as a table, under the rank law the source's exponent fits.

        Raises
        ------
        DeclarationError
            If no class of the source survives the declared inventory.
        """
        try:
            rows = self.estimate.build().at(self.forms)
        except ValueError as error:
            msg = f"the source lexicon cannot be resembled: {error}"
            raise DeclarationError(msg) from error

        return LexiconStructure(
            forms=self.forms,
            joint=TabularJointSpec(
                rows=tuple(
                    ClassSpec(category=category, share=share)
                    for category, _, share in rows
                )
            ),
            ambiguity=ParametricAmbiguityLawSpec(
                mean=sum(row.size for row in self.estimate.rows) / self.estimate.forms,
                dispersion=0.0,
                maximum=len(rows),
                placement=0.0,
            ),
            rank_law=PowerLawRankLawSpec(exponent=self.estimate.exponent),
            seed=self.seed,
        )

    def build(
        self,
        store: TermStore,
        syntax: TermSyntax,
        resolve_category: CategoryResolution,
        rules: RuleSet[Term],
        goal: TargetSet[Term],
        reserved: frozenset[Term],
        category_measure: CategoryMeasure,
        /,
    ) -> StaticLexicon[Term]:
        """Draw the lexicon the scaled joint declares over the interned categories.

        Parameters
        ----------
        store : TermStore
            The term store used to intern categories.
        syntax : TermSyntax
            The syntax used to parse categories.
        rules : RuleSet[Term]
            The grammar's rules, which the drawn lexicon does not consult.
        goal : TargetSet[Term]
            The grammar's goal, which the drawn lexicon does not consult.
        reserved : frozenset[Term]
            The signature's reserved categories, which the drawn lexicon does
            not consult.

        Returns
        -------
        StaticLexicon[Term]
            The drawn lexicon.

        Raises
        ------
        DeclarationError
            If no class of the source survives the declared inventory.
        """
        return self.structure().build(
            store,
            syntax,
            resolve_category,
            rules,
            goal,
            reserved,
            category_measure,
        )


class SyntheticCorpusDecl(DatasetSpec):
    """The complete declaration of one synthetic corpus.

    Attributes
    ----------
    kind : Literal["synthetic"]
        Registry tag.
    retained : dict[str, float]
        Fractions assigned to exact named retained splits; the remainder trains.
    training_split : str
        The name of the split the remainder is assigned to.
    grammar : GrammarSpec
        The grammar the items are drawn from, with its lexicon.
    language : LanguageDecl
        The learner language whose store contains the grammar's categories.
    size : int
        How many items to draw, corrupted ones included.
    seed : int | str
        The seed of the draw, or ``derived`` to take the seed each point of an
        experiment derives for the corpus, so that sweep seeds resample the
        data. Equal seeds draw equal corpora.
    length : LengthSpec
        The distribution a string's token count is drawn from.
    sampler : DerivationSamplerSpec
        How rule applications contribute mass when derivations are drawn.
    shape : ShapeScheduleSpec | None
        Structural bands selected by split for each accepted derivation, or
        nothing to leave derivation shape uncontrolled.
    ambiguity : AmbiguitySpec | None
        The ambiguity to aim at, or nothing to leave it uncontrolled.
    scope : ScopeSpec | None
        The readings per string to aim at, or nothing to leave the readings
        unmeasured.
    corruption : CorruptionSpec | None
        The share of the items to corrupt and how, or nothing to corrupt none.
    noise : LabelNoiseSpec | None
        The rates at which labels are flipped, or nothing to flip none.
    interpretation : InterpretationSpec | None
        The interpretation every grammatical item's form is read under, or
        nothing to draw strings without forms.
    ill_typing : IllTypingSpec | None
        The share of the grammatical items whose logical form to corrupt
        into a refused one, or nothing to corrupt none.
    """

    kind: Literal["synthetic"] = "synthetic"
    retained: dict[str, float]
    training_split: str = TRAINING_SPLIT
    grammar: GrammarSpec
    language: LanguageDecl = LANGUAGE_FIELD
    size: int
    seed: int | str
    length: LengthSpec
    sampler: DerivationSamplerSpec = dx.field(
        default_factory=UniformDerivationSamplerSpec
    )
    shape: ShapeScheduleSpec | None = None
    ambiguity: AmbiguitySpec | None = None
    scope: ScopeSpec | None = None
    corruption: CorruptionSpec | None = None
    noise: LabelNoiseSpec | None = None
    interpretation: InterpretationSpec | None = None
    ill_typing: IllTypingSpec | None = None

    __axioms__ = (
        dx.axiom("size >= 0", message="size must be nonnegative"),
        dx.axiom(
            "isNone(ill_typing) or isNone(interpretation) == False",
            message="ill-typed forms are injected under an interpretation",
        ),
        dx.axiom(
            "isNone(scope) or isNone(interpretation) == False",
            message="readings are read under an interpretation",
        ),
    )

    @dx.validates("seed")
    @classmethod
    def _seed_is_a_data_seed(cls, value: int | str) -> int | str:
        """Return a nonnegative seed or the derived spelling."""
        return ensure_data_seed("a corpus's seed", value)

    @dx.validates("retained")
    @classmethod
    def _retained_are_named(cls, values: dict[str, float]) -> dict[str, float]:
        """Return validated name-ordered retained allocations."""
        return validate_retained(values)

    @dx.model_validator()
    def _training_split_is_not_retained(self) -> Self:
        """Require a named training split that no retained split shares."""
        validate_training_split(self.training_split, self.retained)
        return self

    @dx.model_validator()
    def _shape_schedule_covers_declared_splits(self) -> Self:
        """Require a target for every split when the schedule has no default."""
        schedule = self.shape
        if schedule is None:
            return self
        declared = frozenset((self.training_split, *self.retained))
        named = frozenset(schedule.splits)
        unknown = named - declared
        if unknown:
            msg = f"shape schedule names undeclared splits {sorted(unknown)!r}"
            raise ValueError(msg)
        missing = frozenset() if schedule.default is not None else declared - named
        if missing:
            msg = f"shape schedule has no target for splits {sorted(missing)!r}"
            raise ValueError(msg)
        return self

    def retained_splits(self) -> frozenset[str]:
        """Return the exact retained split names."""
        return frozenset(self.retained)

    def learner_language(self) -> LanguageDecl:
        """Return the language containing the source grammar's categories."""
        return self.language

    def training_occurrences(self, total: int, /) -> int:
        """Return occurrences left after exact retained allocation."""
        return training_occurrences(total, self.retained)

    @override
    def observed_occurrences(self) -> int:
        """Return the size, since every drawn item is observed once."""
        return self.size

    @override
    def with_derived_seeds(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the corpus with its derived seeds replaced by the ones derived.

        The corpus seed and the seed of its grammar's lexicon are each kept when
        fixed and derived for their own dotted path otherwise.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the corpus in the run document.
        """
        grammar = self.grammar

        return self.with_(
            seed=resolved_seed(self.seed, derive, f"{path}.seed"),
            grammar=grammar.with_(
                lexicon=grammar.lexicon.with_derived_seed(
                    derive, f"{path}.grammar.lexicon"
                )
            ),
        )

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the sentences and any certified forms the lowering yields."""
        batches = {TrainingBatchKind.SENTENCES}

        if self.interpretation is not None:
            batches.add(TrainingBatchKind.GOLD_FORMS)

        return frozenset(batches)


def canonical_bytes(declaration: SyntheticCorpusDecl, /) -> bytes:
    """Return the one byte form of a synthetic corpus declaration.

    Parameters
    ----------
    declaration : SyntheticCorpusDecl
        The declaration to encode.

    Returns
    -------
    bytes
        The declaration as JSON with fields in schema order, mapping entries
        in declaration order, no whitespace and every non-ASCII character
        escaped, encoded as UTF-8. Validating these bytes rebuilds an equal
        declaration.
    """
    document = dumps(
        loads(declaration.model_dump_json()),
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return document.encode("utf-8")


def fingerprint(declaration: SyntheticCorpusDecl, /) -> str:
    """Return the SHA-256 of a declaration's canonical bytes as lowercase hexadecimal.

    Parameters
    ----------
    declaration : SyntheticCorpusDecl
        The declaration to fingerprint.
    """
    return sha256(canonical_bytes(declaration)).hexdigest()


@final
class CheckedSyntheticCorpus:
    """A declaration's compiled grammar and request values after the check stage.

    Parameters
    ----------
    source : SyntheticCorpusDecl
        The validated declaration.
    declaration : bytes
        The declaration's canonical bytes.
    source_grammar : GrammarAssembly
        The compiled grammar the items are drawn from.
    length : LengthDistribution
        The distribution token counts are drawn from.
    weighting : RuleWeighting
        The relative mass of each compiled rule application.
    ambiguity : AmbiguityTarget | None
        The ambiguity target, when one is declared.
    scope : ScopeTarget | None
        The readings target, when one is declared.
    shape : ShapeSchedule | None
        The split-conditioned derivation-shape schedule, when one is declared.
    corruption : Corruption[CorruptionOperator] | None
        The corruption request, when one is declared.
    noise : LabelNoise | None
        The label-noise request, when one is declared.
    source_interpretation : InterpretationFunction | None
        The interpretation, when one is declared.
    ill_typing : Corruption[IllTypingOperator] | None
        The injection request, when one is declared.
    """

    __slots__ = (
        "_ambiguity",
        "_corruption",
        "_declaration",
        "_ill_typing",
        "_length",
        "_noise",
        "_scope",
        "_shape",
        "_source",
        "_source_grammar",
        "_source_interpretation",
        "_weighting",
    )

    def __init__(
        self,
        source: SyntheticCorpusDecl,
        declaration: bytes,
        source_grammar: GrammarAssembly,
        *,
        length: LengthDistribution,
        weighting: RuleWeighting,
        ambiguity: AmbiguityTarget | None,
        corruption: Corruption[CorruptionOperator] | None,
        noise: LabelNoise | None,
        scope: ScopeTarget | None = None,
        shape: ShapeSchedule | None = None,
        source_interpretation: InterpretationFunction | None = None,
        ill_typing: Corruption[IllTypingOperator] | None = None,
    ) -> None:
        self._source = source
        self._declaration = declaration
        self._source_grammar = source_grammar
        self._length = length
        self._weighting = weighting
        self._ambiguity = ambiguity
        self._corruption = corruption
        self._noise = noise
        self._scope = scope
        self._shape = shape
        self._source_interpretation = source_interpretation
        self._ill_typing = ill_typing

    @property
    def source(self) -> SyntheticCorpusDecl:
        """Return the validated declaration."""
        return self._source

    @property
    def declaration(self) -> bytes:
        """Return the declaration's canonical bytes."""
        return self._declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the items are drawn from."""
        return self._source_grammar

    @property
    def length(self) -> LengthDistribution:
        """Return the distribution token counts are drawn from."""
        return self._length

    @property
    def weighting(self) -> RuleWeighting:
        """Return the relative mass of each compiled rule application."""
        return self._weighting

    @property
    def ambiguity(self) -> AmbiguityTarget | None:
        """Return the ambiguity target, when one is declared."""
        return self._ambiguity

    @property
    def scope(self) -> ScopeTarget | None:
        """Return the readings target, when one is declared."""
        return self._scope

    @property
    def shape(self) -> ShapeSchedule | None:
        """Return the derivation-shape schedule, when one is declared."""
        return self._shape

    @property
    def corruption(self) -> Corruption[CorruptionOperator] | None:
        """Return the corruption request, when one is declared."""
        return self._corruption

    @property
    def noise(self) -> LabelNoise | None:
        """Return the label-noise request, when one is declared."""
        return self._noise

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        """Return the interpretation, when one is declared."""
        return self._source_interpretation

    @property
    def source_interpretation_declaration(self) -> dict[str, JsonValue] | None:
        """Return the source interpretation as a typed JSON object."""
        declared = self._source.interpretation
        return (
            None
            if declared is None
            else cast("dict[str, JsonValue]", loads(declared.model_dump_json()))
        )

    @property
    def ill_typing(self) -> Corruption[IllTypingOperator] | None:
        """Return the injection request, when one is declared."""
        return self._ill_typing


@final
class SyntheticCorpusLowerer:
    """Checks, lowers and validates a synthetic corpus declaration on one route.

    Parameters
    ----------
    source_version : str
        The declaration version accepted.
    target_version : str
        The native version produced.
    grammars : GrammarIso | None
        The converter grammars are compiled through, or a fresh one.
    languages : LanguageIso | None
        The converter logical-form languages are lowered through, or a fresh
        one.
    """

    __slots__ = ("_grammars", "_languages", "_route")

    def __init__(
        self,
        source_version: str = SOURCE_VERSION,
        target_version: str = TARGET_VERSION,
        *,
        grammars: GrammarIso | None = None,
        languages: LanguageIso | None = None,
    ) -> None:
        self._route = dx.LoweringRoute(source_version, target_version)
        self._languages = LanguageIso() if languages is None else languages
        self._grammars = GrammarIso(self._languages) if grammars is None else grammars

    @property
    def route(self) -> dx.LoweringRoute:
        """Return the one version pair this lowerer implements."""
        return self._route

    @property
    def grammars(self) -> GrammarIso:
        """Return the converter grammars are compiled through."""
        return self._grammars

    @property
    def languages(self) -> LanguageIso:
        """Return the converter logical-form languages are lowered through."""
        return self._languages

    def check(self, source: SyntheticCorpusDecl, /) -> CheckedSyntheticCorpus:
        """Compile the grammar and the interpretation and build the request values.

        Parameters
        ----------
        source : SyntheticCorpusDecl
            The validated declaration.

        Returns
        -------
        CheckedSyntheticCorpus
            The parts the lower stage draws with.

        Raises
        ------
        DeclarationError
            If the grammar or the interpretation does not lower.
        UnresolvedSeedError
            If the corpus or its lexicon is declared with a derived seed that no
            point has resolved.
        """
        fixed_seed("a corpus's seed", source.seed)
        declared_grammar = declare_grammar(source.grammar)
        if source.interpretation is None:
            grammar = self._grammars.forward(declared_grammar, source.language)
            interpretation = None
        else:
            grammar, interpretation = lower_interpretation(
                declared_grammar,
                source.language,
                source.interpretation,
                self._grammars,
                self._languages,
            )

        length = cast("_LengthDeclaration", source.length).build()
        corruption = (
            None
            if source.corruption is None
            else source.corruption.build(
                CorruptionContext(
                    grammar.lexicon,
                    grammar.grammar.matching,
                    length.max_length,
                )
            )
        )
        ill_typing = (
            None
            if source.ill_typing is None or interpretation is None
            else source.ill_typing.build(IllTypingContext(interpretation))
        )

        return CheckedSyntheticCorpus(
            source,
            canonical_bytes(source),
            grammar,
            length=length,
            weighting=cast("_DerivationSamplerDeclaration", source.sampler).build(
                grammar.grammar.rules
            ),
            ambiguity=None if source.ambiguity is None else source.ambiguity.build(),
            shape=None if source.shape is None else source.shape.build(),
            scope=None if source.scope is None else source.scope.build(),
            corruption=corruption,
            noise=None if source.noise is None else source.noise.build(),
            source_interpretation=interpretation,
            ill_typing=ill_typing,
        )

    def lower(
        self,
        checked: CheckedSyntheticCorpus,
        policy: ParsePolicy[Retain],
        /,
    ) -> SyntheticDataset:
        """Draw the corpus from the checked parts.

        Parameters
        ----------
        checked : CheckedSyntheticCorpus
            The parts the check stage built.
        policy : ParsePolicy[Retain]
            The run-owned parsing policy used to certify every draw.

        Returns
        -------
        SyntheticDataset
            The drawn corpus beside the declaration bytes and the grammar.

        Raises
        ------
        UnproductiveGrammarError
            If no goal category is derivable within the length bounds.
        UncertifiedDerivationError
            If the parser does not rebuild a drawn derivation.
        AmbiguityUnreachableError
            If the draws do not reach the ambiguity target within budget.
        ScopeUnreachableError
            If the draws do not reach the readings target within budget.
        CorruptionExhaustedError
            If no corruption within budget yields a non-derivable string, or
            no injection within budget yields a refused form.
        """
        source = checked.source
        if policy.retain is not Retain.FOREST:
            msg = "synthetic corpus generation requires packed-forest retention"
            raise ValueError(msg)
        corpus = SyntheticGenerator(
            checked.source_grammar,
            interpretation=checked.source_interpretation,
            policy=cast("ParsePolicy[Literal[Retain.FOREST]]", policy),
            weighting=checked.weighting,
        ).sample(
            source.size,
            seed=fixed_seed("a corpus's seed", source.seed),
            splits=split_assignments(
                source.size,
                source.retained,
                fixed_seed("a corpus's seed", source.seed),
                training_split=source.training_split,
            ),
            length=checked.length,
            ambiguity=checked.ambiguity,
            shape=checked.shape,
            scope=checked.scope,
            corruption=checked.corruption,
            noise=checked.noise,
            ill_typing=checked.ill_typing,
        )

        interpretation = checked.source_interpretation
        semantic_lexicon = (
            None if interpretation is None else interpretation.semantic_lexicon
        )

        return SyntheticDataset(
            checked.declaration,
            checked.source_grammar,
            corpus,
            source_version=self._route.source_version,
            target_version=self._route.target_version,
            source_interpretation=checked.source_interpretation,
            semantic=(
                None
                if semantic_lexicon is None
                else SemanticReport.of(semantic_lexicon, (corpus,))
            ),
        )

    def validate(self, target: SyntheticDataset, /) -> None:
        """Reject a dataset whose corpus disagrees with its declaration.

        Parameters
        ----------
        target : SyntheticDataset
            The dataset the lower stage built.

        Raises
        ------
        DeclarationError
            If the corpus was drawn at another seed, holds another number of
            items or forms or another length distribution than the
            declaration names, its mean number of lexical categories per
            token lies outside the declared lexical band, or the dataset
            names another route.
        """
        source = SyntheticCorpusDecl.model_validate_json(target.declaration)
        corpus = target.corpus
        lexical = None if source.ambiguity is None else source.ambiguity.lexical
        band = None if lexical is None else lexical.build()
        tokens = corpus.ambiguity.lexical
        mean = tokens.mean if tokens.total else None
        faults = (
            (corpus.seed != source.seed, "the corpus was drawn at another seed"),
            (len(corpus) != source.size, "the corpus holds another number of items"),
            (
                corpus.length.distribution
                != cast("_LengthDeclaration", source.length).build(),
                "the corpus drew its lengths from another length distribution",
            ),
            (
                bool(corpus.forms)
                != (source.interpretation is not None and bool(corpus.examples)),
                "the corpus holds forms against the declared interpretation",
            ),
            (
                band is not None and mean is not None and not band.admits(mean),
                (
                    f"the corpus has a mean of {mean} lexical categories per "
                    f"token, outside the lexical target {band!r}"
                ),
            ),
            (
                target.source_version != self._route.source_version
                or target.target_version != self._route.target_version,
                "the dataset names another route",
            ),
        )

        for broken, reason in faults:
            if broken:
                msg = (
                    f"synthetic corpus {target.fingerprint[:12]!r} was lowered "
                    f"inconsistently: {reason}"
                )
                raise DeclarationError(msg)


class SyntheticCorpusIso(dx.Iso[SyntheticCorpusDecl, SyntheticDataset]):
    """Converts between synthetic corpus declarations and drawn datasets.

    Each distinct declaration is lowered once per converter and served from
    a table keyed by the declaration afterward. Grammars are compiled one
    level down through a grammar converter, so two corpora declared over one
    grammar share one compiled grammar. The inverse reads the canonical bytes
    the dataset holds back into a declaration.

    Parameters
    ----------
    grammars : GrammarIso | None
        The converter grammars are compiled through, or a fresh one.
    languages : LanguageIso | None
        The converter logical-form languages are lowered through, or a fresh
        one.
    """

    def __init__(
        self,
        grammars: GrammarIso | None = None,
        languages: LanguageIso | None = None,
        *,
        policy: ParsePolicy[Retain],
    ) -> None:
        self._lowerer = SyntheticCorpusLowerer(grammars=grammars, languages=languages)
        self._lowered: dict[SyntheticCorpusDecl, SyntheticDataset] = {}
        self._policy = policy

    @property
    def grammars(self) -> GrammarIso:
        """Return the converter every grammar this converter compiles goes through."""
        return self._lowerer.grammars

    @property
    def languages(self) -> LanguageIso:
        """Return the converter every language this converter lowers goes through."""
        return self._lowerer.languages

    @override
    def forward(self, a: SyntheticCorpusDecl, /) -> SyntheticDataset:
        """Lower a declaration to its drawn dataset.

        Parameters
        ----------
        a : SyntheticCorpusDecl
            The validated declaration.

        Returns
        -------
        SyntheticDataset
            The dataset, the same object for every equal declaration.
        """
        held = self._lowered.get(a)

        if held is not None:
            return held

        dataset = self._lowerer.lower(self._lowerer.check(a), self._policy)
        self._lowerer.validate(dataset)
        self._lowered[a] = dataset

        return dataset

    @override
    def backward(self, b: SyntheticDataset, /) -> SyntheticCorpusDecl:
        """Recover the declaration a dataset was drawn from.

        Parameters
        ----------
        b : SyntheticDataset
            A drawn dataset.

        Returns
        -------
        SyntheticCorpusDecl
            The declaration validated from the dataset's canonical bytes.
        """
        return SyntheticCorpusDecl.model_validate_json(b.declaration)


_PLUGIN_LOWERER: Final = SyntheticCorpusLowerer()


def _check_dataset(declaration: SyntheticCorpusDecl, /) -> CheckedSyntheticCorpus:
    """Check one synthetic declaration through the installed dataset route."""
    return _PLUGIN_LOWERER.check(declaration)


def _lower_dataset(
    checked: CheckedSyntheticCorpus, policy: ParsePolicy[Retain], /
) -> SyntheticDataset:
    """Draw one checked synthetic corpus."""
    return _PLUGIN_LOWERER.lower(checked, policy)


def _validate_dataset(dataset: SyntheticDataset, /) -> None:
    """Validate one dataset produced by the installed route."""
    _PLUGIN_LOWERER.validate(dataset)


def _dataset_vocabulary(
    dataset: SyntheticDataset, reserved: ReservedForms, /
) -> DatasetVocabulary:
    """Return the vocabulary induced by the synthetic grammar's lexicon."""
    return DatasetVocabulary(
        tuple(dataset.source_grammar.lexicon.forms),
        reserved=reserved.indices,
        unknown=reserved.unknown,
    )


def _dataset_sentences(dataset: SyntheticDataset, /) -> CorpusSentences:
    """Return the corpus's sentences under its declared training split."""
    source = SyntheticCorpusDecl.model_validate_json(dataset.declaration)

    return corpus_sentences(dataset, source.training_split)


def _dataset_streams(declaration: SyntheticCorpusDecl, /) -> tuple[str, ...]:
    """Return the streams lowering consumes outside the declaration seed."""
    del declaration
    return ()


PLUGIN: Final = DatasetPlugin(
    spec=SyntheticCorpusDecl,
    checked=CheckedSyntheticCorpus,
    dataset=SyntheticDataset,
    check=_check_dataset,
    lower=_lower_dataset,
    validate=_validate_dataset,
    vocabulary=_dataset_vocabulary,
    resources=corpus_featurizer_resources,
    sentences=_dataset_sentences,
    pairs=RefusedTraining(
        "a synthetic corpus supplies sentence training, not inference pairs"
    ),
    streams=_dataset_streams,
)
"""The installed synthetic-corpus dataset route."""

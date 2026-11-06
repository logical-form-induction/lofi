"""The synthetic inference corpus declaration and the boundary that lowers it.

A declaration names the grammar and source semantics the pairs are drawn and
labeled under. It also declares the readings and refused forms each side's strings are
drawn with and what a labeled pair needs: a length distribution per side, the
complexity bands each form and their overlap fall in, the share of decided
pairs each relation is aimed at, and the label noise. Lowering checks both
semantic boundaries and draws the labeled pairs only through the source.
"""

from hashlib import sha256
from json import dumps, loads
from math import isclose
from typing import TYPE_CHECKING, Final, Literal, Protocol, Self, cast, final, override

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.data.datasets import DatasetPlugin, DatasetVocabulary, RefusedTraining
from lofi.data.synthetic._complexity import (
    ComplexityTarget,
    PairComplexityTarget,
    complexity_controls,
)
from lofi.data.synthetic._inference import (
    InferenceNoise,
    LabelConfusion,
    LabelProportion,
    PairRelations,
    SyntheticInference,
)
from lofi.data.synthetic._injection import IllTypingContext
from lofi.data.synthetic._semantics import SemanticReport
from lofi.data.synthetic.generator import SyntheticGenerator
from lofi.data.synthetic.spec import pairing as _pairing
from lofi.data.synthetic.spec._splits import (
    split_assignments,
    training_occurrences,
    validate_retained,
    validate_training_split,
)
from lofi.data.synthetic.spec.corpus import (
    IllTypingSpec,
    LengthSpec,
    ScopeSpec,
    TargetSpec,
)
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    lower_interpretation,
)
from lofi.data.synthetic.training import (
    CorpusPair,
    InferencePairs,
    synthetic_inference_featurizer_resources,
)
from lofi.deduce import ParsePolicy, Retain
from lofi.lf import (
    RELATION_LABELS,
    InferenceLabel,
    InferenceLabeling,
    registered_labeling,
)
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
from lofi.spec.grammar import GrammarAssembly, GrammarDecl, GrammarIso, canonical_bytes
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl, LanguageIso

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import ItemId, JsonValue
    from lofi.data._supervision import Example, GoldDerivation
    from lofi.data.datasets import ReservedForms
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._lengths import LengthDistribution

__all__ = [
    "INFERENCE_SOURCE_VERSION",
    "INFERENCE_TARGET_VERSION",
    "PLUGIN",
    "CheckedSyntheticInference",
    "ComplexityControlSpec",
    "ConfusionRowSpec",
    "InferenceNoiseRateSpec",
    "InferenceNoiseSpec",
    "InferenceSemanticsSpec",
    "LabelConfusionSpec",
    "LabelProportionSpec",
    "PairComplexitySpec",
    "PairRelationsSpec",
    "SyntheticInferenceDataset",
    "SyntheticInferenceDecl",
    "SyntheticInferenceLowerer",
    "TabularLabelConfusionSpec",
    "UniformLabelConfusionSpec",
    "inference_bytes",
    "inference_fingerprint",
]

INFERENCE_SOURCE_VERSION: Final = "lofi-synthetic-inference-decl"
"""The declaration version the inference lowerer accepts."""

INFERENCE_TARGET_VERSION: Final = "lofi-synthetic-inference-native"
"""The native version the inference lowerer produces."""


class ComplexityControlSpec(dx.Model, extra="forbid"):
    """The band each complexity control of one side of a pair must fall in.

    Attributes
    ----------
    controls : dict[str, TargetSpec]
        Target bands keyed by installed ``lofi.lf_complexity`` entry name.
    """

    controls: dict[str, TargetSpec]

    @dx.validates("controls")
    @classmethod
    def _controls_are_installed(
        cls, controls: dict[str, TargetSpec]
    ) -> dict[str, TargetSpec]:
        """Return nonempty target bands whose controls are installed."""
        if not controls:
            msg = "a complexity target names no control"
            raise ValueError(msg)

        available = frozenset(complexity_controls())
        unknown = sorted(set(controls) - available)

        if unknown:
            msg = (
                f"complexity controls are not installed: {unknown!r}; "
                f"installed controls are {sorted(available)!r}"
            )
            raise ValueError(msg)

        return controls

    def build(self) -> ComplexityTarget:
        """Return the interior complexity target."""
        return ComplexityTarget(
            {control: target.build() for control, target in self.controls.items()}
        )


class PairComplexitySpec(dx.Model, extra="forbid"):
    """What a draw of a premise and hypothesis aims at on each side and across the pair.

    Attributes
    ----------
    premise : ComplexityControlSpec | None
        The bands the premise's form must fall in, or nothing to leave the
        premise uncontrolled.
    hypothesis : ComplexityControlSpec | None
        The bands the hypothesis's form must fall in, or nothing to leave the
        hypothesis uncontrolled.
    overlap : TargetSpec | None
        The number of predicate symbols the two sides share, or nothing to
        leave the overlap uncontrolled.
    """

    premise: ComplexityControlSpec | None = None
    hypothesis: ComplexityControlSpec | None = None
    overlap: TargetSpec | None = None

    def build(self) -> PairComplexityTarget:
        """Return the interior pair complexity target."""
        return PairComplexityTarget(
            premise=None if self.premise is None else self.premise.build(),
            hypothesis=None if self.hypothesis is None else self.hypothesis.build(),
            overlap=None if self.overlap is None else self.overlap.build(),
        )


class LabelProportionSpec(dx.Model, extra="forbid"):
    """The share of the decided pairs each relation is aimed at.

    Attributes
    ----------
    entails : float
        The share of the decided pairs the premise entails the hypothesis in.
    contradicts : float
        The share the premise entails the hypothesis's negation in.
    neutral : float
        The share the premise decides the hypothesis neutral way in.
    tolerance : float
        How far an achieved share may fall from the one aimed at.
    """

    entails: float
    contradicts: float
    neutral: float
    tolerance: float

    __axioms__ = (
        dx.axiom("entails >= 0", message="a share must be nonnegative"),
        dx.axiom("contradicts >= 0", message="a share must be nonnegative"),
        dx.axiom("neutral >= 0", message="a share must be nonnegative"),
        dx.axiom("tolerance >= 0", message="tolerance must be nonnegative"),
    )

    @dx.validates("entails", "contradicts", "neutral", "tolerance")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite number."""
        return ensure_finite("a proportion", number)

    @dx.model_validator()
    def _sums_to_one(self) -> Self:
        """Return the proportion when its three shares sum to one.

        Raises
        ------
        ValueError
            If the shares sum to anything else.
        """
        total = self.entails + self.contradicts + self.neutral

        if not isclose(total, 1.0):
            msg = f"a label proportion sums to {total!r} rather than one"
            raise ValueError(msg)

        return self

    def build(self) -> LabelProportion:
        """Return the interior label proportion."""
        return LabelProportion(
            {
                InferenceLabel.ENTAILS: self.entails,
                InferenceLabel.CONTRADICTS: self.contradicts,
                InferenceLabel.NEUTRAL: self.neutral,
            },
            tolerance=self.tolerance,
        )


class ConfusionRowSpec(dx.Model, extra="forbid"):
    """The share of one relation's flips that each relation takes.

    Attributes
    ----------
    oracle : InferenceLabel
        The relation the oracle reached.
    entails : float
        The share of its flips observed as entailment.
    contradicts : float
        The share of its flips observed as contradiction.
    neutral : float
        The share of its flips observed as neutral.
    """

    oracle: InferenceLabel
    entails: float = 0.0
    contradicts: float = 0.0
    neutral: float = 0.0

    __axioms__ = (
        dx.axiom("entails >= 0", message="a share must be nonnegative"),
        dx.axiom("contradicts >= 0", message="a share must be nonnegative"),
        dx.axiom("neutral >= 0", message="a share must be nonnegative"),
    )

    @dx.validates("entails", "contradicts", "neutral")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite number."""
        return ensure_finite("a confusion share", number)

    @dx.validates("oracle")
    @classmethod
    def _oracle_is_a_relation(cls, label: InferenceLabel) -> InferenceLabel:
        """Return a canonical proof relation rather than a coarsened output."""
        if label not in RELATION_LABELS:
            msg = f"a confusion row needs a logical relation, not {label!r}"
            raise ValueError(msg)
        return label

    @dx.model_validator()
    def _sums_to_one_or_nothing(self) -> Self:
        """Return the row when its shares sum to one or nothing.

        A share on the row's own relation is kept: a flip landing there
        leaves the relation as the oracle reached it.

        Raises
        ------
        ValueError
            If the shares sum to anything other than one or zero.
        """
        shares = self.shares()
        total = sum(shares.values())

        if total != 0.0 and not isclose(total, 1.0):
            msg = (
                f"the confusion of {self.oracle} sums to {total!r} rather than "
                "one or nothing"
            )
            raise ValueError(msg)

        return self

    def shares(self) -> dict[InferenceLabel, float]:
        """Return the share each relation takes, keyed by relation.

        Returns
        -------
        dict[InferenceLabel, float]
            The three shares, in report order.
        """
        return {
            InferenceLabel.ENTAILS: self.entails,
            InferenceLabel.CONTRADICTS: self.contradicts,
            InferenceLabel.NEUTRAL: self.neutral,
        }


class _LabelConfusionDeclaration(Protocol):
    """A declaration that builds a label-confusion structure."""

    def build(self) -> LabelConfusion:
        """Return the interior confusion structure."""
        ...


class _LengthDeclaration(Protocol):
    """A declaration that builds a token count distribution."""

    def build(self) -> LengthDistribution:
        """Return the interior distribution."""
        ...


class LabelConfusionSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A spelling of where a flipped relation lands."""


class UniformLabelConfusionSpec(LabelConfusionSpec, extra="forbid"):
    """Flips spread evenly over the two relations a pair does not bear."""

    kind: Literal["uniform"] = "uniform"

    def build(self) -> LabelConfusion:
        """Return the confusion spreading a relation's flips evenly."""
        return LabelConfusion.uniform()


class TabularLabelConfusionSpec(LabelConfusionSpec, extra="forbid"):
    """Flips routed by an explicit row per relation.

    A relation with no row is never confused for another, so a declaration
    carrying the entailment row alone spells the entailment-to-neutral
    confusion a human annotator produces.

    Attributes
    ----------
    rows : tuple[ConfusionRowSpec, ...]
        One row per relation that is ever confused.
    """

    kind: Literal["tabular"] = "tabular"
    rows: tuple[ConfusionRowSpec, ...]

    __axioms__ = (dx.axiom("len(rows) > 0", message="a tabular confusion needs a row"),)

    @dx.validates("rows")
    @classmethod
    def _relations_are_distinct(
        cls, rows: tuple[ConfusionRowSpec, ...]
    ) -> tuple[ConfusionRowSpec, ...]:
        """Return the rows when no relation has two.

        Raises
        ------
        ValueError
            If a relation has two rows.
        """
        relations = [row.oracle for row in rows]

        if len(set(relations)) != len(relations):
            msg = f"a tabular confusion lists a relation twice: {relations!r}"
            raise ValueError(msg)

        return rows

    def build(self) -> LabelConfusion:
        """Return the confusion the rows spell."""
        return LabelConfusion({row.oracle: row.shares() for row in self.rows})


class PairRelationsSpec(dx.Model, extra="forbid"):
    """Which relations a draw admits between premises and hypotheses.

    A pair is trivial when its two logical forms are one term, which covers a
    reflexive pair and a pair of alphabetic variants.

    Attributes
    ----------
    trivial : float
        The share of the pairs that are trivial, taken to the nearest pair.
        Zero excludes them.
    entailment : int
        The least relation complexity an entailing pair may have: how many
        subterms one of its forms has that the other does not. A trivial pair
        entails at complexity zero, so a positive least complexity holds no
        trivial pair.
    """

    trivial: float = 0.0
    entailment: int = 0

    __axioms__ = (
        dx.axiom("trivial >= 0", message="a share must be nonnegative"),
        dx.axiom("trivial <= 1", message="a share must be at most one"),
        dx.axiom("entailment >= 0", message="a complexity must be nonnegative"),
    )

    @dx.validates("trivial")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite share."""
        return ensure_finite("a trivial-pair share", number)

    @dx.model_validator()
    def _trivial_pairs_are_admitted(self) -> Self:
        """Return a policy whose trivial pairs its entailment bound admits.

        Raises
        ------
        ValueError
            If the policy asks for trivial pairs and refuses every entailing
            pair at complexity zero.
        """
        if self.trivial > 0.0 and self.entailment > 0:
            msg = (
                f"a share of {self.trivial} trivial pairs is declared, and an "
                f"entailing pair must lie {self.entailment} subterms apart, "
                "which no trivial pair does"
            )
            raise ValueError(msg)

        return self

    def build(self) -> PairRelations:
        """Return the interior relation policy."""
        return PairRelations(trivial=self.trivial, entailment=self.entailment)


class InferenceNoiseRateSpec(dx.Model, extra="forbid"):
    """The probability that one oracle relation is flipped."""

    relation: InferenceLabel
    rate: float

    __axioms__ = (
        dx.axiom("rate >= 0", message="a rate must be nonnegative"),
        dx.axiom("rate <= 1", message="a rate must be at most one"),
    )

    @dx.validates("rate")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite rate."""
        return ensure_finite("an inference-noise rate", number)

    @dx.validates("relation")
    @classmethod
    def _relation_is_canonical(cls, label: InferenceLabel) -> InferenceLabel:
        """Return a canonical proof relation rather than a coarsened output."""
        if label not in RELATION_LABELS:
            msg = f"an inference-noise rate needs a logical relation, not {label!r}"
            raise ValueError(msg)
        return label


class InferenceNoiseSpec(dx.Model, extra="forbid"):
    """Per-relation flip rates and where the flips land.

    Attributes
    ----------
    rates : tuple[InferenceNoiseRateSpec, ...]
        A sparse rate table by oracle relation. An omitted relation is never
        flipped.
    confusion : LabelConfusionSpec
        Where a flip lands.
    tolerance : float
        How far an achieved confusion matrix entry may fall from the declared
        one.
    """

    rates: tuple[InferenceNoiseRateSpec, ...]
    confusion: LabelConfusionSpec
    tolerance: float

    __axioms__ = (dx.axiom("tolerance >= 0", message="tolerance must be nonnegative"),)

    @dx.validates("tolerance")
    @classmethod
    def _is_finite(cls, number: float) -> float:
        """Return a finite number."""
        return ensure_finite("an inference-noise number", number)

    @dx.validates("rates")
    @classmethod
    def _relations_are_distinct(
        cls, rows: tuple[InferenceNoiseRateSpec, ...]
    ) -> tuple[InferenceNoiseRateSpec, ...]:
        """Return a rate table that names each relation at most once."""
        relations = tuple(row.relation for row in rows)
        if len(set(relations)) != len(relations):
            msg = f"inference-noise rates list a relation twice: {relations!r}"
            raise ValueError(msg)
        return rows

    def build(self) -> InferenceNoise:
        """Return the interior inference-noise request."""
        return InferenceNoise(
            {row.relation: row.rate for row in self.rates},
            confusion=cast("_LabelConfusionDeclaration", self.confusion).build(),
            tolerance=self.tolerance,
        )


class InferenceSemanticsSpec(dx.Model, extra="forbid"):
    """One semantic treatment of the shared inference grammar.

    Attributes
    ----------
    language : LanguageDecl
        The logical form language whose store contains categories and meanings.
        Its declared name is the stable identity of this treatment.
    interpretation : InterpretationSpec
        The interpretation of the shared grammar in that language.
    """

    language: LanguageDecl = LANGUAGE_FIELD
    interpretation: InterpretationSpec


class SyntheticInferenceDecl(DatasetSpec):
    """The complete declaration of one synthetic inference corpus.

    Each side's strings are drawn from the grammar as a sample of their own,
    under the readings target and with the refused forms declared here, and
    every pair among them is labeled by the interpretation's oracle under the
    interpretation's clock.

    Attributes
    ----------
    kind : Literal["synthetic-inference"]
        Registry tag.
    retained : dict[str, float]
        Fractions assigned to exact named retained pair splits.
    training_split : str
        The name of the split the remaining pairs are assigned to.
    grammar : GrammarDecl
        The grammar the source uses to draw both sides.
    source : InferenceSemanticsSpec
        The language and interpretation used exclusively to draw strings and
        label their relations.
    scope : ScopeSpec | None
        The readings each side's strings are drawn within, or nothing to
        leave the readings unmeasured.
    ill_typing : IllTypingSpec | None
        The share of each side's strings whose logical form to corrupt into
        a refused one, or nothing to corrupt none.
    size : int
        How many labeled pairs to draw.
    seed : int | str
        The fixed seed of the draw or ``derived`` until point planning resolves
        it. Equal resolved seeds and source declarations draw equal runs,
        independently of the learner treatment.
    candidates : int
        How many strings to draw on each side to pair among.
    attempts : int
        How many candidate pairs one accepted pair may take before the
        complexity target or the label proportion is declared unreachable.
    premise : LengthSpec
        The distribution a premise string's token count is drawn from.
    hypothesis : LengthSpec
        The distribution a hypothesis string's token count is drawn from.
    complexity : PairComplexitySpec | None
        The bands the two forms and their overlap must fall in, or nothing to
        leave the complexity uncontrolled.
    proportion : LabelProportionSpec | None
        The share of the decided pairs each relation is aimed at, or nothing
        to take the labels as they fall.
    label_noise : InferenceNoiseSpec | None
        The rate a pair's relation is emitted as another at and where the
        flips land, or nothing to emit every relation the oracle reached.
    relations : PairRelationsSpec
        The share of the pairs that are trivial and how complex an entailing
        pair must be. By default no pair is trivial.
    pairing : _pairing.PairingSpec
        How candidates from the premise and hypothesis pools are paired.
    """

    kind: Literal["synthetic-inference"] = "synthetic-inference"
    labeling: str
    retained: dict[str, float]
    training_split: str = TRAINING_SPLIT
    grammar: GrammarDecl
    source: InferenceSemanticsSpec
    scope: ScopeSpec | None = None
    ill_typing: IllTypingSpec | None = None
    size: int
    seed: int | str
    candidates: int
    attempts: int
    premise: LengthSpec
    hypothesis: LengthSpec
    complexity: PairComplexitySpec | None = None
    proportion: LabelProportionSpec | None = None
    label_noise: InferenceNoiseSpec | None = None
    judgments: tuple[int, ...] | None = None
    relations: PairRelationsSpec = dx.field(default_factory=PairRelationsSpec)
    pairing: _pairing.PairingSpec = dx.field(
        default_factory=_pairing.IndependentPairingSpec
    )

    __axioms__ = (
        dx.axiom("size >= 0", message="size must be nonnegative"),
        dx.axiom("candidates > 0", message="candidates must be positive"),
        dx.axiom("attempts > 0", message="attempts must be positive"),
    )

    @dx.validates("seed")
    @classmethod
    def _seed_is_a_data_seed(cls, value: int | str) -> int | str:
        """Return a nonnegative seed or the derived spelling."""
        return ensure_data_seed("an inference run's seed", value)

    @dx.validates("labeling")
    @classmethod
    def _labeling_is_registered(cls, value: str) -> str:
        """Return the name of an installed inference label vocabulary."""
        registered_labeling(value)
        return value

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
    def _judgments_align_to_pairs(self) -> Self:
        """Return the declaration when every supplied response names one pair."""
        judgments = self.judgments

        if judgments is None:
            return self

        if len(judgments) != self.size:
            msg = (
                f"an inference run with {self.size} pairs declares "
                f"{len(judgments)} judgment responses"
            )
            raise ValueError(msg)

        invalid = tuple(response for response in judgments if response < 0)

        if invalid:
            msg = f"judgment responses are zero-based levels, not {invalid!r}"
            raise ValueError(msg)

        return self

    def retained_splits(self) -> frozenset[str]:
        """Return the exact retained split names."""
        return frozenset(self.retained)

    def inference_labels(self) -> tuple[str, ...]:
        """Return the selected output vocabulary in model-column order."""
        return tuple(label.value for label in registered_labeling(self.labeling).labels)

    def retained_pair_side_derivation_splits(self) -> frozenset[str]:
        """Return retained pair splits whose two sides keep gold derivations."""
        return self.retained_splits()

    def training_occurrences(self, total: int, /) -> int:
        """Return occurrences left after exact retained allocation."""
        return training_occurrences(total, self.retained)

    @override
    def with_derived_seeds(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the run with its derived lexicon seed replaced when needed.

        The inference seed and the shared grammar's lexicon seed are kept when
        fixed and derived independently at their dotted paths otherwise.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the run in the run document.
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
        """Return the labeled pairs, their sentences and their certified forms."""
        batches = {
            TrainingBatchKind.SENTENCES,
            TrainingBatchKind.INFERENCE_PAIRS,
            TrainingBatchKind.GOLD_FORMS,
        }

        if self.judgments is not None:
            batches.add(TrainingBatchKind.JUDGMENTS)

        return frozenset(batches)


def inference_bytes(declaration: SyntheticInferenceDecl, /) -> bytes:
    """Return the one byte form of a synthetic inference declaration.

    Parameters
    ----------
    declaration : SyntheticInferenceDecl
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


def inference_fingerprint(declaration: SyntheticInferenceDecl, /) -> str:
    """Return the SHA-256 of the data-generating declaration as lowercase hex.

    Parameters
    ----------
    declaration : SyntheticInferenceDecl
        The declaration whose shared grammar, source semantics and pair drawing
        controls identify the generated data.
    """
    return sha256(inference_bytes(declaration)).hexdigest()


@final
class CheckedSyntheticInference(FrozenValue):
    """A declaration beside its compiled source treatment."""

    __slots__ = (
        "_declaration",
        "_labeling",
        "_source",
        "_source_grammar",
        "_source_interpretation",
    )
    _noun = "checked synthetic-inference dataset"

    def __init__(
        self,
        source: SyntheticInferenceDecl,
        declaration: bytes,
        source_grammar: GrammarAssembly,
        source_interpretation: InterpretationFunction,
        labeling: InferenceLabeling,
    ) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_source_grammar", source_grammar)
        object.__setattr__(self, "_source_interpretation", source_interpretation)
        object.__setattr__(self, "_labeling", labeling)

    @property
    def source(self) -> SyntheticInferenceDecl:
        """Return the validated source declaration."""
        return self._source

    @property
    def declaration(self) -> bytes:
        """Return the canonical dataset declaration bytes."""
        return self._declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar used to draw the pair sample."""
        return self._source_grammar

    @property
    def source_interpretation(self) -> InterpretationFunction:
        """Return the interpretation used to label the pair sample."""
        return self._source_interpretation

    @property
    def source_interpretation_declaration(self) -> dict[str, JsonValue]:
        """Return the source interpretation as a typed JSON object."""
        return cast(
            "dict[str, JsonValue]",
            loads(self._source.source.interpretation.model_dump_json()),
        )

    @property
    def labeling(self) -> InferenceLabeling:
        """Return the declared inference label vocabulary."""
        return self._labeling


@final
class SyntheticInferenceDataset(FrozenValue):
    """A lowered sample with its source treatment."""

    __slots__ = (
        "_corpus_pairs",
        "_declaration",
        "_fingerprint",
        "_inference",
        "_judgments",
        "_labeling",
        "_source_grammar",
        "_source_interpretation",
    )
    _noun = "synthetic-inference dataset"

    def __init__(
        self,
        declaration: bytes,
        inference: SyntheticInference,
        source_grammar: GrammarAssembly,
        source_interpretation: InterpretationFunction,
        labeling: InferenceLabeling,
        judgments: tuple[int, ...] | None,
    ) -> None:
        if inference.declaration != source_grammar.declaration:
            msg = "a synthetic inference sample was drawn under another grammar"
            raise ValueError(msg)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_fingerprint", sha256(declaration).hexdigest())
        object.__setattr__(self, "_inference", inference)
        object.__setattr__(self, "_source_grammar", source_grammar)
        object.__setattr__(self, "_source_interpretation", source_interpretation)
        object.__setattr__(self, "_labeling", labeling)
        object.__setattr__(self, "_judgments", judgments)
        object.__setattr__(self, "_corpus_pairs", _corpus_pairs(inference, judgments))

    @property
    def declaration(self) -> bytes:
        """Return the canonical dataset declaration bytes."""
        return self._declaration

    @property
    def fingerprint(self) -> str:
        """Return the digest of the dataset declaration."""
        return self._fingerprint

    @property
    def inference(self) -> SyntheticInference:
        """Return the labeled inference pairs and their reports."""
        return self._inference

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar used to draw both sides."""
        return self._source_grammar

    @property
    def source_interpretation(self) -> InterpretationFunction:
        """Return the interpretation used to label the retained pairs."""
        return self._source_interpretation

    @property
    def judgments(self) -> tuple[int, ...] | None:
        """Return one declared ordinal response per pair, when supplied."""
        return self._judgments

    @property
    def labeling(self) -> InferenceLabeling:
        """Return the label vocabulary shared by training and evaluation."""
        return self._labeling

    @property
    def corpus_pairs(self) -> tuple[CorpusPair, ...]:
        """Return every drawn pair in draw order, with its emitted relation."""
        return self._corpus_pairs

    @property
    def derivations(self) -> tuple[Example[GoldDerivation[str]], ...]:
        """Return the gold derivations of the premise and hypothesis samples."""
        return (
            *self._inference.premises.examples,
            *self._inference.hypotheses.examples,
        )

    @property
    def gold_forms(self) -> tuple[tuple[ItemId, str], ...]:
        """Return the gold logical form of every sampled premise and hypothesis."""
        return tuple(
            (item_id, form.text)
            for sample in (self._inference.premises, self._inference.hypotheses)
            for item_id, form in sample.forms.items()
        )

    @property
    def drawn(self) -> SyntheticInference:
        """Return the draw the pairs came from."""
        return self._inference


def _corpus_pairs(
    inference: SyntheticInference, judgments: tuple[int, ...] | None, /
) -> tuple[CorpusPair, ...]:
    """Return each drawn pair with its items, emitted relation and gold forms."""
    premises = {
        example.item.item_id: example.item for example in inference.premises.examples
    }
    hypotheses = {
        example.item.item_id: example.item for example in inference.hypotheses.examples
    }
    return tuple(
        CorpusPair(
            premises[pair.premise.item_id],
            hypotheses[pair.hypothesis.item_id],
            relation=label.observed,
            split=pair.split,
            judgment=None if judgments is None else judgments[position],
            gold=(pair.premise.text, pair.hypothesis.text),
        )
        for position, (pair, label) in enumerate(
            zip(inference.pairs, inference.labels, strict=True)
        )
    )


@final
class SyntheticInferenceLowerer:
    """Checks and lowers a synthetic inference declaration on one route.

    Parameters
    ----------
    source_version : str
        The declaration version accepted.
    target_version : str
        The native version produced.
    grammars : GrammarIso | None
        The converter grammars are compiled through, or a fresh one.
    languages : LanguageIso | None
        The converter logical form languages are lowered through, or a fresh
        one.
    """

    __slots__ = ("_grammars", "_languages", "_route")

    def __init__(
        self,
        source_version: str = INFERENCE_SOURCE_VERSION,
        target_version: str = INFERENCE_TARGET_VERSION,
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
        """Return the converter logical form languages are lowered through."""
        return self._languages

    def lower(
        self, source: SyntheticInferenceDecl, policy: ParsePolicy[Retain], /
    ) -> SyntheticInference:
        """Compile the grammar and the interpretation and draw the pairs.

        Parameters
        ----------
        source : SyntheticInferenceDecl
            The validated declaration.
        policy : ParsePolicy[Retain]
            The run-owned parsing policy used to certify every draw.

        Returns
        -------
        SyntheticInference
            The labeled pairs beside the sample each side was drawn as, the
            achieved complexity distributions and the achieved label counts.

        Raises
        ------
        DeclarationError
            If the grammar or the interpretation does not lower.
        UnresolvedSeedError
            If the run or its lexicon is declared with a derived seed that no
            point has resolved.
        ScopeUnreachableError
            If a side's draws do not reach the readings target within budget.
        CorruptionExhaustedError
            If no injection within budget yields a refused form.
        ComplexityUnreachableError
            If no candidate within the attempt budget falls in every band.
        ProportionUnreachableError
            If a declared proportion's quota cannot be filled within budget.
        PairRelationUnreachableError
            If the relation policy refuses every candidate within budget.
        """
        inference, _ = self.lower_with_interpretation(source, policy)
        return inference

    def lower_with_interpretation(
        self,
        source: SyntheticInferenceDecl,
        policy: ParsePolicy[Retain],
        /,
    ) -> tuple[SyntheticInference, InterpretationFunction]:
        """Lower one declaration together with its source interpretation."""
        semantics = source.source
        assembly, interpretation = lower_interpretation(
            source.grammar,
            semantics.language,
            semantics.interpretation,
            self._grammars,
            self._languages,
        )

        return _draw_inference(source, assembly, interpretation, policy), interpretation

    def validate(
        self, source: SyntheticInferenceDecl, target: SyntheticInference, /
    ) -> None:
        """Reject a run that disagrees with the declaration it was drawn from.

        Parameters
        ----------
        source : SyntheticInferenceDecl
            The declaration the run was drawn from.
        target : SyntheticInference
            The run the lower stage produced.

        Raises
        ------
        DeclarationError
            If the run was drawn at another seed, holds another number of
            pairs, drew either side from another length distribution or under
            another readings target, injected refused forms against the
            declaration, was drawn under another noise declaration, or left a
            declared target unmet.
        """
        scope = None if source.scope is None else source.scope.build()
        semantics = source.source
        _, interpretation = lower_interpretation(
            source.grammar,
            semantics.language,
            semantics.interpretation,
            self._grammars,
            self._languages,
        )
        if source.ill_typing is None:
            ill_typing = None
        else:
            ill_typing = source.ill_typing.build(IllTypingContext(interpretation))
        semantic_lexicon = interpretation.semantic_lexicon
        semantic = (
            None
            if semantic_lexicon is None
            else SemanticReport.of(
                semantic_lexicon,
                (target.premises, target.hypotheses),
            )
        )
        faults = (
            (
                target.declaration != canonical_bytes(source.grammar),
                "the run was drawn under another source grammar",
            ),
            (target.seed != source.seed, "the run was drawn at another seed"),
            (
                len(target.pairs) != source.size,
                "the run holds another number of pairs",
            ),
            (
                target.premises.length.distribution
                != cast("_LengthDeclaration", source.premise).build(),
                "the run drew its premises from another length distribution",
            ),
            (
                target.hypotheses.length.distribution
                != cast("_LengthDeclaration", source.hypothesis).build(),
                "the run drew its hypotheses from another length distribution",
            ),
            (
                any(
                    (None if sample.scope is None else sample.scope.target) != scope
                    for sample in (target.premises, target.hypotheses)
                ),
                "the run drew its strings under another readings target",
            ),
            (
                any(
                    (
                        None
                        if sample.ill_typing is None
                        else sample.ill_typing.injection.requested
                    )
                    != ill_typing
                    for sample in (target.premises, target.hypotheses)
                ),
                "the run injected refused forms against the declaration",
            ),
            (
                not target.complexity.satisfied(),
                "the run left a declared complexity band unmet",
            ),
            (
                not target.inference.satisfied(),
                "the run left the declared label proportion unmet",
            ),
            (
                target.relations.requested != source.relations.build(),
                "the run was drawn under another relation policy",
            ),
            (
                not target.relations.satisfied(),
                "the run holds another share of trivial pairs than declared",
            ),
            (
                (source.label_noise is None) != (target.noise is None),
                "the run was drawn under another noise declaration",
            ),
            (
                target.noise is not None and not target.noise.satisfied(),
                "the run left the declared confusion matrix unmet",
            ),
            (
                target.semantic != semantic,
                "the run carries another semantic report",
            ),
        )

        for broken, reason in faults:
            if broken:
                msg = (
                    f"synthetic inference {inference_fingerprint(source)[:12]!r} "
                    f"was lowered inconsistently: {reason}"
                )

                raise DeclarationError(msg)


def _draw_inference(
    source: SyntheticInferenceDecl,
    assembly: GrammarAssembly,
    interpretation: InterpretationFunction,
    policy: ParsePolicy[Retain],
    /,
) -> SyntheticInference:
    """Draw pairs through an already compiled grammar and interpretation."""
    seed = fixed_seed("an inference run's seed", source.seed)

    if policy.retain is not Retain.FOREST:
        msg = "synthetic inference generation requires packed-forest retention"
        raise ValueError(msg)

    return SyntheticGenerator(
        assembly,
        interpretation=interpretation,
        policy=cast("ParsePolicy[Literal[Retain.FOREST]]", policy),
    ).infer(
        source.size,
        seed=seed,
        splits=split_assignments(
            source.size, source.retained, seed, training_split=source.training_split
        ),
        premise=cast("_LengthDeclaration", source.premise).build(),
        hypothesis=cast("_LengthDeclaration", source.hypothesis).build(),
        candidates=source.candidates,
        attempts=source.attempts,
        scope=None if source.scope is None else source.scope.build(),
        ill_typing=(
            None
            if source.ill_typing is None
            else source.ill_typing.build(IllTypingContext(interpretation))
        ),
        complexity=(None if source.complexity is None else source.complexity.build()),
        proportion=(None if source.proportion is None else source.proportion.build()),
        noise=(None if source.label_noise is None else source.label_noise.build()),
        relations=source.relations.build(),
        pairing=source.pairing.build(),
    )


_PLUGIN_LOWERER: Final = SyntheticInferenceLowerer()


def _check_dataset(declaration: SyntheticInferenceDecl, /) -> CheckedSyntheticInference:
    """Check both semantic treatments without drawing pairs."""
    fixed_seed("an inference run's seed", declaration.seed)
    source_grammar, source_interpretation = lower_interpretation(
        declaration.grammar,
        declaration.source.language,
        declaration.source.interpretation,
        _PLUGIN_LOWERER.grammars,
        _PLUGIN_LOWERER.languages,
    )
    return CheckedSyntheticInference(
        declaration,
        inference_bytes(declaration),
        source_grammar,
        source_interpretation,
        registered_labeling(declaration.labeling),
    )


def _lower_dataset(
    checked: CheckedSyntheticInference, policy: ParsePolicy[Retain], /
) -> SyntheticInferenceDataset:
    """Draw inference pairs through the installed dataset route."""
    inference = _draw_inference(
        checked.source,
        checked.source_grammar,
        checked.source_interpretation,
        policy,
    )
    return SyntheticInferenceDataset(
        checked.declaration,
        inference,
        checked.source_grammar,
        checked.source_interpretation,
        checked.labeling,
        checked.source.judgments,
    )


def _validate_dataset(dataset: SyntheticInferenceDataset, /) -> None:
    """Reject an inference dataset inconsistent with its declaration."""
    source = SyntheticInferenceDecl.model_validate_json(dataset.declaration)
    declared_grammar = canonical_bytes(source.grammar)
    if dataset.source_grammar.declaration != declared_grammar:
        msg = "a synthetic-inference dataset was drawn under another grammar"
        raise DeclarationError(msg)
    if dataset.fingerprint != inference_fingerprint(source):
        msg = "a synthetic-inference dataset carries another source fingerprint"
        raise DeclarationError(msg)
    _PLUGIN_LOWERER.validate(source, dataset.inference)
    if dataset.judgments != source.judgments:
        msg = "a synthetic-inference dataset carries other judgment responses"
        raise DeclarationError(msg)
    if dataset.labeling != registered_labeling(source.labeling):
        msg = "a synthetic-inference dataset carries another label vocabulary"
        raise DeclarationError(msg)


def _dataset_vocabulary(
    dataset: SyntheticInferenceDataset, reserved: ReservedForms, /
) -> DatasetVocabulary:
    """Return forms from both sides in first-observed order."""
    forms = tuple(
        dict.fromkeys(
            token.surface
            for sample in (
                dataset.inference.premises,
                dataset.inference.hypotheses,
            )
            for example in sample.examples
            for token in example.item.tokens
        )
    )
    return DatasetVocabulary(
        forms,
        reserved=reserved.indices,
        unknown=reserved.unknown,
    )


def _dataset_pairs(dataset: SyntheticInferenceDataset, /) -> InferencePairs:
    """Return the decided pairs of the dataset, split for training."""
    source = SyntheticInferenceDecl.model_validate_json(dataset.declaration)

    return InferencePairs(dataset, training_split=source.training_split)


def _dataset_streams(declaration: SyntheticInferenceDecl, /) -> tuple[()]:
    """Return no stream beyond the declaration's inference seed."""
    del declaration
    return ()


PLUGIN: Final = DatasetPlugin(
    spec=SyntheticInferenceDecl,
    checked=CheckedSyntheticInference,
    dataset=SyntheticInferenceDataset,
    check=_check_dataset,
    lower=_lower_dataset,
    validate=_validate_dataset,
    vocabulary=_dataset_vocabulary,
    resources=synthetic_inference_featurizer_resources,
    sentences=RefusedTraining(
        "a synthetic inference dataset supplies inference pairs, not sentence training"
    ),
    pairs=_dataset_pairs,
    streams=_dataset_streams,
)
"""The installed synthetic-inference dataset route."""

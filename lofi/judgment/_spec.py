"""Response-model declarations and result values."""

import enum
import json
import math
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast, override
from uuid import UUID

import didactic.api as dx
from bead.data.base import JsonValue as FeatureValue
from bead.items.item_template import ItemTemplate
from bead.resources.constraints import Constraint
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.judgment._design import POSITION, BoundedDesign, JudgmentDesign
from lofi.judgment.cumulative import (
    DEFAULT_BASE_PRIOR,
    DEFAULT_INCREMENT_PRIOR,
)
from lofi.judgment.splits import FoldCount, Splitter, SplitterFactory
from lofi.registry import Registry
from lofi.spec import HeldOutParticipantPolicy, OptimizerSpec, SeedDerivationRecord
from lofi.spec.artifacts import (
    ArtifactCacheSettings,
    ArtifactCacheSpec,
    PretrainedSourceSpec,
    source_artifact,
)
from lofi.spec.evaluators.interannotator import InterannotatorSpec
from lofi.spec.lexicon_generators import LexiconGeneratorSpec

if TYPE_CHECKING:
    from lofi._artifacts import PinnedCheckpoint
    from lofi._types import JsonValue
    from lofi.judgment._ports import LinkingCompiler
    from lofi.judgment._program import Prior

_DEFAULT_DISCRIMINATION: Final = False

__all__ = [
    "GUIDE_METHODS",
    "AgreementBin",
    "AuxiliaryParameter",
    "Bounded",
    "BoundedResponseSpec",
    "Choice",
    "ColumnInventory",
    "ColumnLabels",
    "ColumnLevel",
    "ColumnLift",
    "ComparisonSpec",
    "ConditionalLikelihood",
    "ConstraintCost",
    "ConstraintFile",
    "ConstraintFilling",
    "ConstraintSource",
    "CorrelatedRandomSlope",
    "Cumulative",
    "DesignProvenance",
    "DynamicRangeCriterion",
    "DynamicRangeReport",
    "DynamicRangeSpec",
    "Elicitation",
    "ElicitationMethod",
    "ExhaustiveFilling",
    "FactorTerm",
    "FillDirection",
    "FillingSpec",
    "FitArtifact",
    "FitDiagnostics",
    "FitInference",
    "FitSchedule",
    "FitSummary",
    "FixedCutpointVariance",
    "FixedDiscrimination",
    "GlazingLexicon",
    "IdentityLinkOnLogit",
    "InferenceMethod",
    "InlineConstraint",
    "InlineItemTemplate",
    "InlineLexicon",
    "InlineTemplate",
    "InteractionTerm",
    "ItemConstructionSpec",
    "ItemTemplateFile",
    "ItemTemplateSource",
    "JointTypeTensor",
    "JudgmentFitCalibration",
    "JudgmentFitManifest",
    "JudgmentFitSpec",
    "LexicalPossession",
    "LexicalPrimitiveInventory",
    "LexiconFile",
    "LexiconSource",
    "LikelihoodObjective",
    "LinkTerm",
    "LinkingSpec",
    "MapInference",
    "MaskedLMFilling",
    "MaskedLMGeneratorSpec",
    "MaskedLMSettings",
    "Measurement",
    "MemorizedCells",
    "Metric",
    "MixedFilling",
    "ModelComparisonCriterion",
    "ModelComparisonSpec",
    "ModelWeighting",
    "NutsInference",
    "OffsetTerm",
    "ParameterFormula",
    "PinnedPrefix",
    "PossessionAndClasses",
    "PosteriorHandle",
    "PosteriorVarianceRow",
    "PredictiveDensityRow",
    "Predictor",
    "PriorBlock",
    "PriorSpec",
    "PseudoRSquaredRow",
    "RandomFilling",
    "RandomIntercept",
    "RandomSlope",
    "RankConsistent",
    "RankConsistentScheme",
    "ReadBoundedDesign",
    "ReadCorpusDesign",
    "ReadDesign",
    "ResponseDeclaration",
    "ResponseFamily",
    "ResponsePrediction",
    "ResponseSpec",
    "RowClasses",
    "ScaleIdentification",
    "ScaleNormalization",
    "ScaleProfile",
    "StratifiedFilling",
    "StreamingFilling",
    "TableAblationSpec",
    "TableColumns",
    "TableLexicon",
    "TableRows",
    "TemplateCollectionFile",
    "TemplateFile",
    "TemplateSource",
    "ThresholdPriors",
    "TruncatedMarginalLikelihood",
    "UniMorphLexicon",
    "VariationalInference",
    "decode_dynamic_range_criterion",
    "decode_fit_manifest",
    "decode_item_construction",
    "encode_fit_manifest",
    "installed_linkings",
    "make_dynamic_range_report",
    "refuse_unidentified",
    "spec_digest",
    "splitter_for",
]


class TemplateSource(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where bead templates are read from."""


class TemplateFile(TemplateSource):
    """Read bead templates from a JSON Lines file, one template per line.

    Parameters
    ----------
    path : str
        File relative to the directory holding the document.
    """

    kind: Literal["jsonl"] = "jsonl"
    path: str

    @dx.validates("path")
    @classmethod
    def _path_is_named(cls, value: str) -> str:
        return ensure_name("a template file", value)


class TemplateCollectionFile(TemplateSource):
    """Read the templates of one bead template collection from a JSON file.

    Parameters
    ----------
    path : str
        File relative to the directory holding the document.
    """

    kind: Literal["collection"] = "collection"
    path: str

    @dx.validates("path")
    @classmethod
    def _path_is_named(cls, value: str) -> str:
        return ensure_name("a template collection file", value)


class InlineTemplate(TemplateSource):
    """Declare one bead template in bead's serialized shape."""

    kind: Literal["inline"] = "inline"
    template: dx.Embed[Template]


class LexiconSource(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where a bead lexicon is read from."""


class LexiconFile(LexiconSource):
    """Read a bead lexicon from a JSON Lines file, one lexical item per line.

    Parameters
    ----------
    path : str
        File relative to the directory holding the document.
    name : str
        Name the lexicon is held under.
    """

    kind: Literal["jsonl"] = "jsonl"
    path: str
    name: str

    @dx.validates("path", "name")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return ensure_name("a lexicon file field", value)


class TableLexicon(LexiconSource):
    """Read a bead lexicon from a delimited table through bead's loaders.

    bead draws a loaded item's identity at random, so each item's identity is
    derived instead from the construction's namespace, the lexicon's name and
    the item's declared content.

    Parameters
    ----------
    path : str
        File relative to the directory holding the document.
    name : str
        Name the lexicon is held under.
    format : {"csv", "tsv"}
        The table's delimiter, read by bead's ``from_csv`` or ``from_tsv``.
    language_code : str
        Language code every item carries.
    column_mapping : dict[str, str]
        The item field each renamed column fills, keyed by column; a table
        with a ``lemma`` column maps none.
    feature_columns : tuple[str, ...]
        Columns recorded in every item's features.
    pos : str | None
        Part of speech every item's features record, or None.
    """

    kind: Literal["table"] = "table"
    path: str
    name: str
    format: Literal["csv", "tsv"]
    language_code: str
    column_mapping: dict[str, str]
    feature_columns: tuple[str, ...]
    pos: str | None

    @dx.validates("path", "name", "language_code")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return ensure_name("a lexicon table field", value)


class GlazingLexicon(LexiconSource):
    """Fetch a bead lexicon from VerbNet, PropBank or FrameNet through glazing.

    Fetched identities are derived as a table lexicon's are.

    Parameters
    ----------
    name : str
        Name the lexicon is held under.
    resource : {"verbnet", "propbank", "framenet"}
        The resource bead's glazing adapter reads.
    query : str | None
        The adapter's query, or None for the adapter's whole resource.
    language_code : str | None
        Language code the fetched items are filtered by, or None.
    options : dict[str, FeatureValue]
        Further keyword arguments of the adapter's fetch.
    """

    kind: Literal["glazing"] = "glazing"
    name: str
    resource: Literal["verbnet", "propbank", "framenet"]
    query: str | None
    language_code: str | None
    options: dict[str, FeatureValue]

    @dx.validates("name")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return ensure_name("a glazing lexicon name", value)


class UniMorphLexicon(LexiconSource):
    """Fetch a bead lexicon of inflected forms through bead's UniMorph adapter.

    Fetched identities are derived as a table lexicon's are.

    Parameters
    ----------
    name : str
        Name the lexicon is held under.
    query : str | None
        The lemma queried, or None for the language's whole paradigm table.
    language_code : str
        Language whose paradigms are read.
    options : dict[str, FeatureValue]
        Further keyword arguments of the adapter's fetch.
    """

    kind: Literal["unimorph"] = "unimorph"
    name: str
    query: str | None
    language_code: str
    options: dict[str, FeatureValue]

    @dx.validates("name", "language_code")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return ensure_name("a UniMorph lexicon field", value)


class InlineLexicon(LexiconSource):
    """Declare one bead lexicon in bead's serialized shape."""

    kind: Literal["inline"] = "inline"
    lexicon: dx.Embed[Lexicon]


class ItemTemplateSource(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where the bead item template is read from."""


class ItemTemplateFile(ItemTemplateSource):
    """Read the bead item template from a JSON file.

    Parameters
    ----------
    path : str
        File relative to the directory holding the document.
    """

    kind: Literal["json"] = "json"
    path: str

    @dx.validates("path")
    @classmethod
    def _path_is_named(cls, value: str) -> str:
        return ensure_name("an item-template file", value)


class InlineItemTemplate(ItemTemplateSource):
    """Declare the bead item template in bead's serialized shape."""

    kind: Literal["inline"] = "inline"
    item_template: dx.Embed[ItemTemplate]


class ConstraintSource(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where the bead constraints an item template names are read from."""


class ConstraintFile(ConstraintSource):
    """Read bead constraints from a JSON Lines file, one constraint per line.

    Parameters
    ----------
    path : str
        File relative to the directory holding the document.
    """

    kind: Literal["jsonl"] = "jsonl"
    path: str

    @dx.validates("path")
    @classmethod
    def _path_is_named(cls, value: str) -> str:
        return ensure_name("a constraint file", value)


class InlineConstraint(ConstraintSource):
    """Declare one bead constraint in bead's serialized shape."""

    kind: Literal["inline"] = "inline"
    constraint: dx.Embed[Constraint]


class FillingSpec(dx.TaggedUnion, discriminator="strategy", extra="forbid"):
    """Declare the bead strategy a template is filled by.

    A strategy that draws at random takes the seed it declares; one that
    declares none takes a seed derived from the construction's namespace and
    the template's identity, so a document always constructs the same items.
    """


class ExhaustiveFilling(FillingSpec):
    """Fill a template with every combination its slots admit."""

    strategy: Literal["exhaustive"] = "exhaustive"


class RandomFilling(FillingSpec):
    """Fill a template with a seeded sample of the combinations it admits.

    Parameters
    ----------
    samples : int
        Combinations drawn.
    seed : int | None
        Seed of the draw, or None for the derived seed.
    """

    strategy: Literal["random"] = "random"
    samples: int
    seed: int | None

    @dx.validates("samples")
    @classmethod
    def _samples_are_positive(cls, value: int) -> int:
        return ensure_positive("a random filling's samples", value)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return ensure_non_negative("a random filling's seed", value)


class StratifiedFilling(FillingSpec):
    """Fill a template with a sample balanced over the groups of a filler property.

    Parameters
    ----------
    samples : int
        Combinations drawn.
    grouping_property : str
        The lexical-item property, such as ``pos`` or ``features.number``,
        whose groups bead's stratified strategy balances.
    seed : int | None
        Seed of the draw, or None for the derived seed.
    """

    strategy: Literal["stratified"] = "stratified"
    samples: int
    grouping_property: str
    seed: int | None

    @dx.validates("samples")
    @classmethod
    def _samples_are_positive(cls, value: int) -> int:
        return ensure_positive("a stratified filling's samples", value)

    @dx.validates("grouping_property")
    @classmethod
    def _property_is_named(cls, value: str) -> str:
        return ensure_name("a stratified filling's grouping property", value)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return ensure_non_negative("a stratified filling's seed", value)


class ConstraintFilling(FillingSpec):
    """Fill a template by bead's backtracking constraint solver.

    The solver honors a template's multi-slot constraints, which the
    enumerating strategies do not read.

    Parameters
    ----------
    count : int
        Most fillings found.
    max_attempts : int
        Most partial assignments the solver tries.
    """

    strategy: Literal["csp"] = "csp"
    count: int
    max_attempts: int

    @dx.validates("count", "max_attempts")
    @classmethod
    def _bounds_are_positive(cls, value: int) -> int:
        return ensure_positive("a constraint filling bound", value)


type FillDirection = Literal[
    "left_to_right", "right_to_left", "inside_out", "outside_in", "custom"
]
"""The order bead's masked-LM strategy fills a template's slots in."""


class MaskedLMFilling(FillingSpec):
    """Fill a template by beam search over a masked language model's proposals.

    bead's masked-LM strategy ranks the lexicon's admissible items for each
    slot by the model's completions; it fills slots and proposes no new
    lexical item, which a lexicon generator does.

    Parameters
    ----------
    checkpoint : PretrainedSourceSpec
        The hub repository and commit of the model bead's Hugging Face
        masked-LM adapter loads, from the directory it is fetched into.
    artifacts : ArtifactCacheSpec
        The cache the model is fetched into.
    device : {"cpu", "cuda", "mps"}
        The device the model runs on.
    beam_size : int
        Hypotheses kept at each step.
    fill_direction : FillDirection
        The order slots are filled in.
    custom_order : tuple[int, ...] | None
        Slot positions in fill order, declared exactly for ``custom``.
    top_k : int
        Model completions considered for each slot.
    budget : int | None
        Most fillings produced, or None for no bound.
    per_slot_max_fills : dict[str, int]
        Most distinct fillers of each named slot.
    per_slot_enforce_unique : dict[str, bool]
        Whether a named slot's filler differs across hypotheses.
    """

    strategy: Literal["mlm"] = "mlm"
    checkpoint: PretrainedSourceSpec
    artifacts: ArtifactCacheSpec
    device: Literal["cpu", "cuda", "mps"]
    beam_size: int
    fill_direction: FillDirection
    custom_order: tuple[int, ...] | None
    top_k: int
    budget: int | None
    per_slot_max_fills: dict[str, int]
    per_slot_enforce_unique: dict[str, bool]

    @dx.validates("beam_size", "top_k")
    @classmethod
    def _widths_are_positive(cls, value: int) -> int:
        return ensure_positive("a masked-LM filling width", value)

    @dx.model_validator()
    def _custom_order_is_declared_for_custom(self) -> MaskedLMFilling:
        """Validate that a custom order is declared exactly for ``custom``."""
        if (self.fill_direction == "custom") != (self.custom_order is not None):
            msg = "a masked-LM filling declares a custom order exactly when custom"
            raise ValueError(msg)
        if self.budget is not None:
            ensure_positive("a masked-LM filling budget", self.budget)
        self.pinned()
        return self

    def pinned(self) -> PinnedCheckpoint:
        """Return the pinned release the declared source names."""
        return source_artifact(self.checkpoint).hub_checkpoint("a masked-LM filling")


class MixedFilling(FillingSpec):
    """Fill named slots of a template by their own bead strategies.

    Bead's mixed strategy fills the slots of every enumerating strategy first,
    then the masked-LM slots, all by the one declared masked-LM filling, in
    the context of each first-pass filling.

    Parameters
    ----------
    slots : dict[str, FillingSpec]
        The enumerating strategy, exhaustive, random or stratified, each named
        slot is filled by.
    masked_lm_slots : tuple[str, ...]
        The slots the masked-LM filling fills.
    masked_lm : MaskedLMFilling | None
        The masked-LM filling, declared exactly when it fills a slot; its
        per-slot bounds are keyed by those slots.
    default : FillingSpec
        The enumerating strategy every other slot is filled by.
    """

    strategy: Literal["mixed"] = "mixed"
    slots: dict[str, FillingSpec]
    masked_lm_slots: tuple[str, ...]
    masked_lm: MaskedLMFilling | None
    default: FillingSpec

    @dx.model_validator()
    def _slots_are_filled_once(self) -> MixedFilling:
        """Validate the enumerating strategies and the masked-LM slots."""
        enumerating = (ExhaustiveFilling, RandomFilling, StratifiedFilling)
        declared = (*self.slots.values(), self.default)
        if not all(isinstance(filling, enumerating) for filling in declared):
            msg = (
                "a mixed filling fills its named and default slots by "
                "exhaustive, random or stratified strategies"
            )
            raise ValueError(msg)
        masked = tuple(
            ensure_name("a masked-LM slot", slot) for slot in self.masked_lm_slots
        )
        overlap = sorted(set(masked) & set(self.slots))
        if overlap or len(set(masked)) != len(masked):
            msg = f"a mixed filling fills the slots {overlap!r} twice"
            raise ValueError(msg)
        if bool(masked) != (self.masked_lm is not None):
            msg = "a mixed filling declares a masked-LM filling exactly for its slots"
            raise ValueError(msg)
        return self


class StreamingFilling(FillingSpec):
    """Fill a template lazily through bead's streaming filler.

    Parameters
    ----------
    max_combinations : int | None
        Most fillings produced, or None for every combination.
    """

    strategy: Literal["streaming"] = "streaming"
    max_combinations: int | None

    @dx.validates("max_combinations")
    @classmethod
    def _bound_is_positive(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return ensure_positive("a streaming filling bound", value)


class MaskedLMSettings(NamedTuple):
    """Resolved pinned masked-language-model proposal of lexical items."""

    kind: Literal["masked-lm"]
    lexicon: str
    language_code: str
    features: Mapping[str, FeatureValue]
    checkpoint: PinnedCheckpoint
    cache: ArtifactCacheSettings
    mask: str
    count: int
    prompts: tuple[str, ...]


class MaskedLMGeneratorSpec(LexiconGeneratorSpec):
    """Propose lexical items as a pinned masked language model's completions.

    Parameters
    ----------
    lexicon : str
        Name of the lexicon the proposed items form.
    language_code : str
        Language code every proposed item carries.
    features : dict[str, FeatureValue]
        Features every proposed item carries, which slot constraints read.
    checkpoint : PretrainedSourceSpec
        The hub repository and commit the masked language model is fetched from.
    artifacts : ArtifactCacheSpec
        The cache the masked language model is fetched into.
    mask : str
        The model's mask token, which every prompt contains exactly once.
    count : int
        Number of distinct single-token completions the lexicon holds.
    prompts : tuple[str, ...]
        Prompts whose masked completions are pooled, best score first.
    """

    kind: Literal["masked-lm"] = "masked-lm"
    lexicon: str
    language_code: str
    features: dict[str, FeatureValue]
    checkpoint: PretrainedSourceSpec
    artifacts: ArtifactCacheSpec
    mask: str
    count: int
    prompts: tuple[str, ...]

    @dx.validates("lexicon", "language_code", "mask")
    @classmethod
    def _text_is_present(cls, value: str) -> str:
        return ensure_name("a masked-LM generator field", value)

    @dx.validates("count")
    @classmethod
    def _count_is_positive(cls, value: int) -> int:
        return ensure_positive("a masked-LM generator count", value)

    @dx.model_validator()
    def _prompts_hold_the_mask_once(self) -> MaskedLMGeneratorSpec:
        """Validate that every prompt contains the mask exactly once."""
        if not self.prompts:
            msg = f"the masked-LM generator {self.lexicon!r} declares no prompt"
            raise ValueError(msg)
        for prompt in self.prompts:
            if prompt.count(self.mask) != 1:
                msg = (
                    f"the masked-LM prompt {prompt!r} does not contain "
                    f"{self.mask!r} exactly once"
                )
                raise ValueError(msg)
        return self

    @dx.model_validator()
    def _loads_a_hub_checkpoint(self) -> MaskedLMGeneratorSpec:
        """Refuse a source that is not a hub repository at one commit."""
        self._checkpoint()
        return self

    def _checkpoint(self) -> PinnedCheckpoint:
        """Return the pinned release the declared source names."""
        return source_artifact(self.checkpoint).hub_checkpoint(
            "a masked-LM lexicon generator"
        )

    def settings(self) -> MaskedLMSettings:
        """Discard the boundary declaration into immutable settings."""
        return MaskedLMSettings(
            self.kind,
            self.lexicon,
            self.language_code,
            MappingProxyType(dict(self.features)),
            self._checkpoint(),
            self.artifacts.settings(),
            self.mask,
            self.count,
            self.prompts,
        )


class ItemConstructionSpec(dx.Model, extra="forbid"):
    """Declare one construction of bead judgment items.

    Parameters
    ----------
    namespace : UUID
        Namespace every constructed identity derives from.
    templates : tuple[TemplateSource, ...]
        The bead templates filled, in construction order.
    lexicons : tuple[LexiconSource, ...]
        The bead lexicons slots are filled from.
    generators : tuple[LexiconGeneratorSpec, ...]
        Generators each proposing one more lexicon slots are filled from.
    filling : FillingSpec
        The bead strategy a template is filled by unless it is named in
        ``template_fillings``.
    template_fillings : dict[str, FillingSpec]
        The bead strategy each named template is filled by instead.
    item_template : ItemTemplateSource
        The bead item template naming each item's rendered elements and the
        constraints every item satisfies.
    constraints : tuple[ConstraintSource, ...]
        The bead constraints the item template names, by identity.
    admit : str | None
        A bead DSL expression over an item's fillings, read as its attributes
        are, that admits the combination as an item, or None to admit every
        combination.
    attributes : dict[str, str]
        The DSL expression each recorded item attribute is read by, keyed by
        the attribute's name.
    """

    namespace: UUID
    templates: tuple[TemplateSource, ...]
    lexicons: tuple[LexiconSource, ...]
    generators: tuple[LexiconGeneratorSpec, ...]
    filling: FillingSpec
    template_fillings: dict[str, FillingSpec]
    item_template: ItemTemplateSource
    constraints: tuple[ConstraintSource, ...]
    admit: str | None
    attributes: dict[str, str]

    @dx.validates("templates")
    @classmethod
    def _templates_are_declared(
        cls, value: tuple[TemplateSource, ...]
    ) -> tuple[TemplateSource, ...]:
        if not value:
            msg = "an item construction declares at least one template source"
            raise ValueError(msg)
        return value

    @dx.validates("admit")
    @classmethod
    def _admission_is_named(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_name("an item admission expression", value)

    @dx.validates("attributes")
    @classmethod
    def _attributes_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            msg = "an item construction declares at least one attribute"
            raise ValueError(msg)
        return {
            ensure_name("an item attribute", name): ensure_name(
                f"attribute {name!r}'s expression", expression
            )
            for name, expression in value.items()
        }

    def files(self) -> tuple[str, ...]:
        """Return every file the construction reads, in declaration order.

        Paths are relative to the directory holding the document.
        """
        declared: list[str] = [
            source.path
            for source in self.templates
            if isinstance(source, TemplateFile | TemplateCollectionFile)
        ]
        declared.extend(
            source.path
            for source in self.lexicons
            if isinstance(source, LexiconFile | TableLexicon)
        )
        if isinstance(self.item_template, ItemTemplateFile):
            declared.append(self.item_template.path)
        declared.extend(
            source.path
            for source in self.constraints
            if isinstance(source, ConstraintFile)
        )
        return tuple(dict.fromkeys(declared))


def refuse_unidentified(value: JsonValue, noun: str, /) -> None:
    """Refuse a declared bead resource whose identity bead would draw at random."""
    if not isinstance(value, dict) or "id" not in value:
        msg = f"{noun} declares no id, so its identity would differ on every read"
        raise ValueError(msg)


def _inline_identities(document: JsonValue, /) -> None:
    """Refuse an inline resource the document declares without its identity."""
    if not isinstance(document, dict):
        return
    for source in _entries(document.get("templates")):
        if source.get("kind") == "inline":
            refuse_unidentified(source.get("template"), "an inline template")
    for source in _entries(document.get("lexicons")):
        if source.get("kind") == "inline":
            lexicon = source.get("lexicon")
            items = lexicon.get("items") if isinstance(lexicon, dict) else None
            for item in items if isinstance(items, list) else ():
                refuse_unidentified(item, "a lexical item of an inline lexicon")
    declared = document.get("item_template")
    if isinstance(declared, dict) and declared.get("kind") == "inline":
        refuse_unidentified(declared.get("item_template"), "the inline item template")
    for source in _entries(document.get("constraints")):
        if source.get("kind") == "inline":
            refuse_unidentified(source.get("constraint"), "an inline constraint")


def _entries(value: JsonValue, /) -> tuple[dict[str, JsonValue], ...]:
    """Return the object entries of one declared resource list."""
    if not isinstance(value, list):
        return ()
    return tuple(entry for entry in value if isinstance(entry, dict))


def decode_item_construction(document: str | bytes, /) -> ItemConstructionSpec:
    """Validate one item-construction document at its declared data boundary.

    Raises
    ------
    ValueError
        If the document is refused, an inline resource among them declaring no
        identity.
    """
    _inline_identities(json.loads(document))
    return ItemConstructionSpec.model_validate_json(document)


def decode_dynamic_range_criterion(document: JsonValue, /) -> DynamicRangeCriterion:
    """Validate a registered dynamic-range criterion at its JSON boundary."""
    return DynamicRangeCriterion.model_validate(
        cast("Mapping[str, JsonValue]", document)
    )


def make_dynamic_range_report(
    criterion: DynamicRangeCriterion,
    /,
    *,
    inventory: str,
    observed_at: str,
    items: int,
    minimum: float,
    lower_quartile: float,
    median: float,
    upper_quartile: float,
    maximum: float,
    ceiling_fraction: float,
) -> DynamicRangeReport:
    """Validate a computed dynamic-range report at its publication boundary."""
    return DynamicRangeReport(
        criterion=criterion,
        inventory=inventory,
        observed_at=observed_at,
        items=items,
        minimum=minimum,
        lower_quartile=lower_quartile,
        median=median,
        upper_quartile=upper_quartile,
        maximum=maximum,
        ceiling_fraction=ceiling_fraction,
    )


class ScaleNormalization(enum.StrEnum):
    """Selects response normalization."""

    #: The recorded response.
    IDENTITY = "identity"

    #: The response's within-annotator empirical rank.
    RIDIT = "ridit"

    #: The natural logarithm of a ratio-scale response.
    LOG = "log"

    #: The response rescaled to the unit interval by the instrument bounds.
    UNIT_INTERVAL = "unit_interval"


class AuxiliaryParameter(enum.StrEnum):
    """Names a distribution parameter beside the location."""

    #: The slope the location and the thresholds are both read at.
    DISCRIMINATION = "discrimination"


class RankConsistentScheme(enum.StrEnum):
    """Selects a rank-consistent tying scheme."""

    #: One shared weight vector and one bias per threshold.
    CORAL = "coral"

    #: One conditional probability per threshold.
    CORN = "corn"


class PriorSpec(dx.Model, extra="forbid"):
    """Declares a prior distribution.

    Parameters
    ----------
    distribution : str
        Distribution-family name.
    parameters : tuple[float, ...]
        Positional distribution parameters.
    """

    distribution: str
    parameters: tuple[float, ...] = ()

    __axioms__ = (
        dx.axiom(
            "len(distribution) > 0",
            message="a prior names a distribution family",
        ),
    )


def _declared(prior: Prior) -> PriorSpec:
    """Return a prior's specification form.

    Parameters
    ----------
    prior : Prior
        Prior to convert.

    Returns
    -------
    PriorSpec
        The same family and positional parameters as a specification value.
    """
    return PriorSpec(distribution=prior.family, parameters=prior.arguments)


class ThresholdPriors(dx.Model, extra="forbid"):
    """Declares priors for ordered thresholds.

    Parameters
    ----------
    base : PriorSpec
        Prior for the lowest threshold.
    increment : PriorSpec
        Prior for each unconstrained threshold increment.
    """

    base: PriorSpec = _declared(DEFAULT_BASE_PRIOR)
    increment: PriorSpec = _declared(DEFAULT_INCREMENT_PRIOR)


class PriorBlock(dx.Model, extra="forbid"):
    """Declares response-family priors.

    Parameters
    ----------
    thresholds : ThresholdPriors or None
        Ordered-threshold priors, if the family draws thresholds.
    discrimination : PriorSpec or None
        Discrimination prior, if the family draws discrimination.
    scale : PriorSpec or None
        Positive observation-scale prior, if the family draws one.
    """

    thresholds: ThresholdPriors | None = None
    discrimination: PriorSpec | None = None
    scale: PriorSpec | None = None


class ResponseFamily(dx.TaggedUnion, discriminator="family", extra="forbid"):
    """A response-family declaration."""

    @dx.computed
    def draws_thresholds(self) -> bool:
        """Return whether this family uses ordered thresholds.

        Returns
        -------
        bool
            Whether the family draws ordered thresholds.
        """
        return False

    @dx.computed
    def draws_discrimination(self) -> bool:
        """Return whether this family draws discrimination.

        Returns
        -------
        bool
            Whether the family draws a discrimination parameter.
        """
        return False

    @dx.computed
    def takes_discrimination(self) -> bool:
        """Return whether this family has a discrimination at all.

        Returns
        -------
        bool
            Whether a discrimination scales what the family observes,
            whether it is drawn, held, or carried by a linear predictor.
        """
        return False

    @dx.computed
    def draws_scale(self) -> bool:
        """Return whether this family draws a positive observation scale."""
        return False

    @dx.computed
    def programmed(self) -> bool:
        """Return whether this family compiles to a probabilistic response program.

        Returns
        -------
        bool
            Whether a posterior over the family's parameters can be drawn by
            a backend that scores compiled programs.
        """
        return False

    @dx.computed
    def takes_formulas(self) -> bool:
        """Return whether an auxiliary parameter of this family may carry a formula."""
        return False


class Cumulative(ResponseFamily, extra="forbid"):
    """Declares an ordered-logit response family.

    Parameters
    ----------
    family : Literal["cumulative"]
        Tagged-union discriminator.
    discrimination : bool
        Whether the family draws discrimination instead of fixing it at one.
    discrimination_group : str | None
        Grouping factor a drawn discrimination varies over, one value per
        level, or None for one discrimination shared by every row.
    """

    family: Literal["cumulative"] = "cumulative"
    discrimination: bool = _DEFAULT_DISCRIMINATION
    discrimination_group: str | None = None

    @dx.model_validator()
    def _a_varying_discrimination_is_drawn(self) -> Cumulative:
        """Validate the discrimination group against the discrimination.

        Returns
        -------
        Cumulative
            This family after validation.

        Raises
        ------
        ValueError
            If a group is named without a drawn discrimination, or the group
            has no name.
        """
        if self.discrimination_group is None:
            return self

        if not self.discrimination_group:
            msg = "a discrimination group names its grouping factor"

            raise ValueError(msg)

        if not self.discrimination:
            msg = (
                f"discrimination varies over {self.discrimination_group!r} and "
                f"the family holds it at one"
            )

            raise ValueError(msg)

        return self

    @dx.computed
    @override
    def draws_thresholds(self) -> bool:
        """Return whether this family uses ordered thresholds.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def draws_discrimination(self) -> bool:
        """Return whether this family draws discrimination.

        Returns
        -------
        bool
            The declared discrimination setting.
        """
        return self.discrimination

    @dx.computed
    @override
    def takes_discrimination(self) -> bool:
        """Return whether this family has a discrimination at all.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def programmed(self) -> bool:
        """Return that this family compiles to a response program."""
        return True

    @dx.computed
    @override
    def takes_formulas(self) -> bool:
        """Return that the discrimination of this family may carry a formula."""
        return True


class RankConsistent(ResponseFamily, extra="forbid"):
    """Declares a rank-consistent response family.

    Parameters
    ----------
    family : Literal["rank_consistent"]
        Tagged-union discriminator.
    scheme : RankConsistentScheme
        Scheme tying the binary threshold tasks together.
    """

    family: Literal["rank_consistent"] = "rank_consistent"
    scheme: RankConsistentScheme

    @dx.computed
    @override
    def draws_thresholds(self) -> bool:
        """Return whether this family draws ordered thresholds.

        Returns
        -------
        bool
            Always ``False``. The binary tasks have fitted biases instead.
        """
        return False


class Metric(ResponseFamily, extra="forbid"):
    """Declares a metric response family.

    Parameters
    ----------
    family : Literal["metric"]
        Tagged-union discriminator.
    """

    family: Literal["metric"] = "metric"


class Bounded(ResponseFamily, extra="forbid"):
    """Declares a truncated-normal response on a closed interval.

    Parameters
    ----------
    family : Literal["bounded"]
        Tagged-union discriminator.
    scale : float | None
        Fixed positive standard deviation on the normalized unit interval,
        or None to draw one global value from the declared scale prior.
    """

    family: Literal["bounded"] = "bounded"
    scale: float | None

    @dx.model_validator()
    def _a_fixed_scale_is_finite_and_positive(self) -> Bounded:
        """Validate a fixed observation scale.

        Returns
        -------
        Bounded
            This family after validation.

        Raises
        ------
        TypeError
            If a fixed scale is a boolean.
        ValueError
            If a fixed scale is nonfinite or nonpositive.
        """
        if self.scale is None:
            return self

        if isinstance(self.scale, bool):
            msg = f"a bounded response scale is not a number: {self.scale!r}"

            raise TypeError(msg)

        if not math.isfinite(self.scale) or self.scale <= 0.0:
            msg = (
                "a bounded response scale is not a finite positive number: "
                f"{self.scale!r}"
            )

            raise ValueError(msg)

        return self

    @dx.computed
    @override
    def draws_scale(self) -> bool:
        """Return whether this family draws the observation scale.

        Returns
        -------
        bool
            Whether no fixed scale was declared.
        """
        return self.scale is None

    @dx.computed
    @override
    def programmed(self) -> bool:
        """Return that this family compiles to a response program."""
        return True


class Choice(ResponseFamily, extra="forbid"):
    """Declares a forced-choice response family.

    Every alternative a row shows is scored on the latent scale the
    predictor defines, and the recorded level names the chosen alternative
    by its place in presentation order. Without ties the level is that
    place and the alternatives are compared by a multinomial logit over
    their latents. With ties the row shows two alternatives and the level
    orders them as the first, a tie, and the second, over the difference
    of the two latents.

    Parameters
    ----------
    family : Literal["choice"]
        Tagged-union discriminator.
    ties : bool
        Whether a tie between the two alternatives is a recordable level.
    discrimination : bool
        Whether the family draws the slope on the difference instead of
        holding it at one.
    discrimination_group : str | None
        Grouping factor a drawn discrimination varies over, one value per
        level, or None for one discrimination shared by every row.
    """

    family: Literal["choice"] = "choice"
    ties: bool = False
    discrimination: bool = _DEFAULT_DISCRIMINATION
    discrimination_group: str | None = None

    def alternatives(self, levels: int, /) -> int:
        """Return how many alternatives a row of a response over levels shows.

        Parameters
        ----------
        levels : int
            Number of recordable levels of the response.

        Returns
        -------
        int
            The levels themselves, less the tie between the alternatives
            when the family records one.
        """
        return levels - 1 if self.ties else levels

    @dx.model_validator()
    def _a_varying_discrimination_is_drawn(self) -> Choice:
        """Validate the discrimination group against the discrimination.

        Returns
        -------
        Choice
            This family after validation.

        Raises
        ------
        ValueError
            If a group is named without a drawn discrimination, or the group
            has no name.
        """
        if self.discrimination_group is None:
            return self

        if not self.discrimination_group:
            msg = "a discrimination group names its grouping factor"

            raise ValueError(msg)

        if not self.discrimination:
            msg = (
                f"discrimination varies over {self.discrimination_group!r} and "
                f"the family holds it at one"
            )

            raise ValueError(msg)

        return self

    @dx.computed
    @override
    def draws_thresholds(self) -> bool:
        """Return whether this family uses ordered thresholds.

        Returns
        -------
        bool
            Whether a tie is recordable, which the two cutpoints separate.
        """
        return self.ties

    @dx.computed
    @override
    def draws_discrimination(self) -> bool:
        """Return whether this family draws discrimination.

        Returns
        -------
        bool
            The declared discrimination setting.
        """
        return self.discrimination

    @dx.computed
    @override
    def takes_discrimination(self) -> bool:
        """Return whether this family has a discrimination at all.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def programmed(self) -> bool:
        """Return that this family compiles to a response program."""
        return True


def _in_name_order[V](entries: V) -> V:
    """Return a mapping in key order.

    A value that is not a mapping is returned as given, for validation to
    refuse.

    Parameters
    ----------
    entries : V
        Entries by name.
    """
    if not isinstance(entries, Mapping):
        return entries

    return cast("V", dict(sorted(entries.items())))


class LinkingSpec(dx.Model, extra="forbid"):
    """Declares the linking a readout enters the latent scale through.

    The linking reads the readout it is the linking of, its argument, and
    whichever further readout columns its family takes, each named here
    under the role the family gives it. A linking that reads its argument
    alone names no columns.

    Parameters
    ----------
    family : str
        Name of the linking family.
    columns : dict[str, str]
        Readout columns the linking reads beyond its argument, by the role
        the family gives each.
    fixed : dict[str, float]
        Linking parameters pinned at declared values, by name. Every
        parameter the family has that is not named here is fitted.
    """

    family: str
    columns: dict[str, str] = dx.field(default_factory=dict, converter=_in_name_order)
    fixed: dict[str, float] = dx.field(default_factory=dict, converter=_in_name_order)

    __axioms__ = (dx.axiom("len(family) > 0", message="a linking names its family"),)

    @dx.model_validator()
    def _columns_are_named_on_both_sides(self) -> LinkingSpec:
        """Validate the further columns.

        Returns
        -------
        LinkingSpec
            This linking after validation.

        Raises
        ------
        ValueError
            If a column has an empty role or an empty name.
        """
        for role, name in self.columns.items():
            if not role:
                msg = f"a linking column has no role: {name!r}"

                raise ValueError(msg)

            if not name:
                msg = f"the linking column {role!r} names no readout"

                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _pinned_values_are_finite(self) -> LinkingSpec:
        """Validate the pinned parameter values.

        Returns
        -------
        LinkingSpec
            This linking after validation.

        Raises
        ------
        ValueError
            If a pinned parameter has an empty name or a value that is not
            finite.
        """
        for name, value in self.fixed.items():
            if not name:
                msg = f"a pinned linking parameter has no name: {value!r}"

                raise ValueError(msg)

            if not math.isfinite(value):
                msg = f"the pinned linking parameter {name!r} is not finite: {value!r}"

                raise ValueError(msg)

        return self


class Predictor(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A linear-predictor term."""

    @dx.computed
    def fixed(self) -> bool:
        """Return whether this term is a fixed effect with fitted coefficients."""
        return False

    @dx.computed
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns this term reads, in declaration order."""
        return ()

    @dx.computed
    def factors(self) -> tuple[str, ...]:
        """Return the factors this term reads, in declaration order."""
        return ()

    @dx.computed
    def groups(self) -> tuple[str, ...]:
        """Return the grouping factors this term varies over."""
        return ()


def _distinct(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return names in first-seen order without repeats.

    Parameters
    ----------
    names : tuple[str, ...]
        Names in declaration order, possibly repeated.
    """
    return tuple(dict.fromkeys(names))


class LinkTerm(Predictor, extra="forbid"):
    """Declares a readout entering the latent scale through a linking.

    The term reads its readout and every further column its linking names,
    and the linking is non-decreasing in the readout.

    Parameters
    ----------
    kind : Literal["link"]
        Tagged-union discriminator.
    readout : str
        Readout column the linking takes as its argument.
    linking : LinkingSpec
        Linking the readout enters through.
    """

    kind: Literal["link"] = "link"
    readout: str
    linking: LinkingSpec

    __axioms__ = (
        dx.axiom("len(readout) > 0", message="a link term names its readout"),
    )

    @dx.computed
    @override
    def fixed(self) -> bool:
        """Return whether this term is a fixed effect with fitted coefficients.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The argument first, then the further columns the linking names,
            in role order.
        """
        return _distinct((self.readout, *self.linking.columns.values()))


class FactorTerm(Predictor, extra="forbid"):
    """Declares one fitted coefficient per level of a factor.

    Parameters
    ----------
    kind : Literal["factor"]
        Tagged-union discriminator.
    name : str
        Factor whose levels the coefficients index.
    """

    kind: Literal["factor"] = "factor"
    name: str

    __axioms__ = (dx.axiom("len(name) > 0", message="a factor term names its factor"),)

    @dx.computed
    @override
    def fixed(self) -> bool:
        """Return whether this term is a fixed effect with fitted coefficients.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def factors(self) -> tuple[str, ...]:
        """Return the factors this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The named factor alone.
        """
        return (self.name,)


class InteractionTerm(Predictor, extra="forbid"):
    """Declares the product of two fixed effects.

    Parameters
    ----------
    kind : Literal["interaction"]
        Tagged-union discriminator.
    left : Predictor
        First fixed effect.
    right : Predictor
        Second fixed effect, distinct from the first.
    """

    kind: Literal["interaction"] = "interaction"
    left: Predictor
    right: Predictor

    @dx.computed
    @override
    def fixed(self) -> bool:
        """Return whether this term is a fixed effect with fitted coefficients.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The readouts of both operands, without repeats.
        """
        return _distinct(self.left.readouts + self.right.readouts)

    @dx.computed
    @override
    def factors(self) -> tuple[str, ...]:
        """Return the factors this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The factors of both operands, without repeats.
        """
        return _distinct(self.left.factors + self.right.factors)

    @dx.model_validator()
    def _operands_are_distinct_fixed_effects(self) -> InteractionTerm:
        """Validate the operands.

        Returns
        -------
        InteractionTerm
            This term after validation.

        Raises
        ------
        ValueError
            If an operand is not a fixed effect, the operands are the same
            term, or two link terms of the interaction read one readout.
        """
        for side, operand in (("left", self.left), ("right", self.right)):
            if not operand.fixed:
                msg = (
                    f"an interaction multiplies fixed effects, and its {side} "
                    f"operand is the {operand.kind} term {operand!r}"
                )

                raise ValueError(msg)

        if self.left == self.right:
            msg = f"an interaction of a term with itself is the term: {self.left!r}"

            raise ValueError(msg)

        arguments = [term.readout for term, _ in _link_terms(self)]
        repeated = _distinct(
            tuple(name for name in arguments if arguments.count(name) > 1)
        )

        if repeated:
            msg = (
                f"an interaction reads each readout through one link term, and "
                f"it reads {list(repeated)!r} through more than one"
            )

            raise ValueError(msg)

        return self


class RandomIntercept(Predictor, extra="forbid"):
    """Declares an intercept varying over the levels of a grouping factor.

    Parameters
    ----------
    kind : Literal["random_intercept"]
        Tagged-union discriminator.
    group : str
        Grouping factor the intercept varies over.
    """

    kind: Literal["random_intercept"] = "random_intercept"
    group: str

    __axioms__ = (
        dx.axiom("len(group) > 0", message="a random intercept names its group"),
    )

    @dx.computed
    @override
    def groups(self) -> tuple[str, ...]:
        """Return the grouping factors this term varies over.

        Returns
        -------
        tuple[str, ...]
            The named group alone.
        """
        return (self.group,)


class CorrelatedRandomSlope(Predictor, extra="forbid"):
    """Declares a jointly varying intercept and scalar slope.

    The coefficient order is intercept, then slope. Their scales are half-normal,
    their standardized group offsets are standard normal, and their correlation is
    given an LKJ Cholesky prior.

    Parameters
    ----------
    kind : Literal["correlated_random_slope"]
        Tagged-union discriminator.
    group : str
        Grouping factor the intercept and slope vary over.
    term : Predictor
        Scalar fixed effect whose coefficient varies.
    concentration : float
        Positive LKJ concentration.
    """

    kind: Literal["correlated_random_slope"] = "correlated_random_slope"
    group: str
    term: Predictor
    concentration: float = 1.0

    __axioms__ = (
        dx.axiom(
            "len(group) > 0",
            message="a correlated random slope names its group",
        ),
        dx.axiom(
            "concentration > 0.0",
            message="an LKJ concentration is positive",
        ),
    )

    @dx.computed
    @override
    def readouts(self) -> tuple[str, ...]:
        """Return the readouts of the varying term."""
        return self.term.readouts

    @dx.computed
    @override
    def factors(self) -> tuple[str, ...]:
        """Return the factors of the varying term, which must be empty."""
        return self.term.factors

    @dx.computed
    @override
    def groups(self) -> tuple[str, ...]:
        """Return the named grouping factor alone."""
        return (self.group,)

    @dx.model_validator()
    def _the_varying_term_is_scalar_and_the_concentration_is_finite(
        self,
    ) -> CorrelatedRandomSlope:
        """Validate the varying term and LKJ concentration."""
        if not self.term.fixed or self.term.factors:
            msg = (
                "a correlated random slope varies one scalar fixed effect, "
                f"and this one varies the {self.term.kind} term {self.term!r}"
            )

            raise ValueError(msg)

        if not math.isfinite(self.concentration):
            msg = f"an LKJ concentration is not finite: {self.concentration!r}"

            raise ValueError(msg)

        return self


class RandomSlope(Predictor, extra="forbid"):
    """Declares a fixed effect's coefficients varying over a grouping factor.

    Parameters
    ----------
    kind : Literal["random_slope"]
        Tagged-union discriminator.
    group : str
        Grouping factor the coefficients vary over.
    term : Predictor
        Fixed effect whose coefficients vary.
    """

    kind: Literal["random_slope"] = "random_slope"
    group: str
    term: Predictor

    __axioms__ = (dx.axiom("len(group) > 0", message="a random slope names its group"),)

    @dx.computed
    @override
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The readouts of the varying term.
        """
        return self.term.readouts

    @dx.computed
    @override
    def factors(self) -> tuple[str, ...]:
        """Return the factors this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The factors of the varying term.
        """
        return self.term.factors

    @dx.computed
    @override
    def groups(self) -> tuple[str, ...]:
        """Return the grouping factors this term varies over.

        Returns
        -------
        tuple[str, ...]
            The named group alone.
        """
        return (self.group,)

    @dx.model_validator()
    def _the_varying_term_is_a_fixed_effect(self) -> RandomSlope:
        """Validate the varying term.

        Returns
        -------
        RandomSlope
            This term after validation.

        Raises
        ------
        ValueError
            If the varying term is not a fixed effect.
        """
        if not self.term.fixed:
            msg = (
                f"a random slope varies a fixed effect, and this one varies the "
                f"{self.term.kind} term {self.term!r}"
            )

            raise ValueError(msg)

        return self


class OffsetTerm(Predictor, extra="forbid"):
    """Declares a readout added to the latent scale with a coefficient of one.

    Parameters
    ----------
    kind : Literal["offset"]
        Tagged-union discriminator.
    readout : str
        Readout column added as is.
    """

    kind: Literal["offset"] = "offset"
    readout: str

    __axioms__ = (
        dx.axiom("len(readout) > 0", message="an offset term names its readout"),
    )

    @dx.computed
    @override
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns this term reads, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The offset readout alone.
        """
        return (self.readout,)


class ParameterFormula(dx.Model, extra="forbid"):
    """Declares the linear predictor one auxiliary parameter carries.

    The parameter is the exponential of the terms' sum, so the terms are
    read on the logarithmic scale, a parameter predicted by an intercept of
    zero is one, and a parameter predicted by nothing at all is refused.

    Parameters
    ----------
    parameter : AuxiliaryParameter
        Parameter the terms predict.
    predictor : tuple[Predictor, ...]
        Terms whose sum is the parameter's logarithm.
    """

    parameter: AuxiliaryParameter
    predictor: tuple[Predictor, ...]

    __axioms__ = (
        dx.axiom(
            "len(predictor) > 0",
            message="a linear predictor has at least one term",
        ),
    )

    @dx.model_validator()
    def _each_term_is_declared_once(self) -> ParameterFormula:
        """Validate the predictor terms.

        Returns
        -------
        ParameterFormula
            This formula after validation.

        Raises
        ------
        ValueError
            If a term is declared twice, or two fixed effects read the same
            readouts and factors.
        """
        for position, term in enumerate(self.predictor):
            if term in self.predictor[:position]:
                msg = (
                    f"the linear predictor of {self.parameter.value} declares "
                    f"the term {term!r} twice"
                )

                raise ValueError(msg)

        _ensure_effects_distinct(
            self.predictor, f"the linear predictor of {self.parameter.value}"
        )

        return self


def _ensure_effects_distinct(terms: tuple[Predictor, ...], formula: str, /) -> None:
    """Refuse two fixed effects of one formula over the same columns.

    A response program identifies a fixed effect by the readouts its link
    terms take as arguments and the factors it crosses, so two fixed effects
    reading the same columns through different linkings would bind one site
    twice.

    Parameters
    ----------
    terms : tuple[Predictor, ...]
        The terms of one formula.
    formula : str
        How the formula is named in a refusal.

    Raises
    ------
    ValueError
        If two fixed effects read the same arguments and factors.
    """
    seen: dict[tuple[tuple[str, ...], tuple[str, ...]], Predictor] = {}

    for term in terms:
        if not term.fixed:
            continue

        effect = (tuple(link.readout for link, _ in _link_terms(term)), term.factors)
        earlier = seen.setdefault(effect, term)

        if earlier is not term:
            msg = (
                f"{formula} enters the readouts {list(effect[0])!r} crossed with "
                f"the factors {list(effect[1])!r} as two fixed effects, "
                f"{earlier!r} and {term!r}; read them through one linking"
            )

            raise ValueError(msg)


def _formula_terms(formulas: tuple[ParameterFormula, ...]) -> tuple[Predictor, ...]:
    """Return every term of every formula, in declaration order.

    Parameters
    ----------
    formulas : tuple[ParameterFormula, ...]
        Formulas whose terms are collected.
    """
    return tuple(term for formula in formulas for term in formula.predictor)


class ScaleIdentification(dx.TaggedUnion, discriminator="constraint", extra="forbid"):
    """A latent-scale constraint."""

    @dx.computed
    def unit_slope(self) -> bool:
        """Return whether every link term enters at a slope held at one."""
        return False

    @dx.computed
    def signed_slope(self) -> bool:
        """Return whether a fitted link slope may cross zero."""
        return False


class FixedCutpointVariance(ScaleIdentification, extra="forbid"):
    """Declares the latent scale identified by pinning the cutpoint variance.

    Parameters
    ----------
    constraint : Literal["fixed_cutpoint_variance"]
        Tagged-union discriminator.
    value : float
        Variance the cutpoints are held at.
    """

    constraint: Literal["fixed_cutpoint_variance"] = "fixed_cutpoint_variance"
    value: float

    __axioms__ = (
        dx.axiom("value > 0.0", message="a pinned cutpoint variance is positive"),
    )

    @dx.model_validator()
    def _the_variance_is_finite(self) -> FixedCutpointVariance:
        """Validate the pinned variance.

        Returns
        -------
        FixedCutpointVariance
            This constraint after validation.

        Raises
        ------
        ValueError
            If the variance is not finite.
        """
        if not math.isfinite(self.value):
            msg = f"a pinned cutpoint variance is not finite: {self.value!r}"

            raise ValueError(msg)

        return self

    @dx.computed
    @override
    def signed_slope(self) -> bool:
        """Return that a fitted slope may cross zero under pinned cutpoints."""
        return True


class IdentityLinkOnLogit(ScaleIdentification, extra="forbid"):
    """Declares the latent scale identified as the log-odds scale itself.

    Every link term of the predictor enters at a slope held at one, so a
    logit link's log-odds are the latent scale rather than a quantity
    proportional to it.

    Parameters
    ----------
    constraint : Literal["identity_link_on_logit"]
        Tagged-union discriminator.
    """

    constraint: Literal["identity_link_on_logit"] = "identity_link_on_logit"

    @dx.computed
    @override
    def unit_slope(self) -> bool:
        """Return that every link term enters at a slope held at one."""
        return True


class FixedDiscrimination(ScaleIdentification, extra="forbid"):
    """Declares the latent scale identified by pinning the discrimination.

    Parameters
    ----------
    constraint : Literal["fixed_discrimination"]
        Tagged-union discriminator.
    value : float
        Discrimination the response family is held at.
    """

    constraint: Literal["fixed_discrimination"] = "fixed_discrimination"
    value: float

    __axioms__ = (
        dx.axiom("value > 0.0", message="a pinned discrimination is positive"),
    )

    @dx.model_validator()
    def _the_discrimination_is_finite(self) -> FixedDiscrimination:
        """Validate the pinned discrimination.

        Returns
        -------
        FixedDiscrimination
            This constraint after validation.

        Raises
        ------
        ValueError
            If the discrimination is not finite.
        """
        if not math.isfinite(self.value):
            msg = f"a pinned discrimination is not finite: {self.value!r}"

            raise ValueError(msg)

        return self


def _free_link_scale(term: Predictor) -> bool:
    """Check whether a composite fixed effect contains an unfixed link slope.

    Parameters
    ----------
    term : Predictor
        Predictor term to inspect.
    """
    match term:
        case LinkTerm():
            return "slope" not in term.linking.fixed
        case InteractionTerm():
            return _free_link_scale(term.left) or _free_link_scale(term.right)
        case _:
            return False


def _scale_degenerate(term: Predictor) -> bool:
    """Check whether a term multiplies two independently free scales.

    Parameters
    ----------
    term : Predictor
        Predictor term to inspect.
    """
    return isinstance(term, InteractionTerm) and _free_link_scale(term)


class LikelihoodObjective(dx.TaggedUnion, discriminator="objective", extra="forbid"):
    """A declared treatment of the response likelihood."""


class ConditionalLikelihood(LikelihoodObjective, extra="forbid"):
    """Declares the ordinary likelihood conditional on every drawn site."""

    objective: Literal["conditional"] = "conditional"


class TruncatedMarginalLikelihood(LikelihoodObjective, extra="forbid"):
    """Declares integration of one complete group-local latent block.

    Parameters
    ----------
    objective : Literal["truncated_marginal"]
        Tagged-union discriminator.
    group : str
        The sole grouping factor whose standardized local block is integrated.
    radius : float
        Radius of the centered finite integration region.
    relative_tolerance : float
        Relative tolerance of the scalar adaptive solve.
    absolute_tolerance : float
        Absolute tolerance of the scalar adaptive solve.
    quadrature_points : int
        Number of positive-weight nodes used for a multivariate block.
    quadrature_seed : int
        Seed of the local generator used to construct those nodes.
    """

    objective: Literal["truncated_marginal"] = "truncated_marginal"
    group: str
    radius: float
    relative_tolerance: float
    absolute_tolerance: float
    quadrature_points: int
    quadrature_seed: int

    @dx.model_validator()
    def _numerical_controls_are_valid(self) -> TruncatedMarginalLikelihood:
        """Validate the integration group and numerical controls."""
        if not self.group:
            msg = "a truncated marginal objective names its local group"

            raise ValueError(msg)

        positive = (
            ("radius", self.radius),
            ("relative tolerance", self.relative_tolerance),
            ("absolute tolerance", self.absolute_tolerance),
        )

        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                msg = f"a truncated marginal {name} is finite and positive: {value!r}"

                raise ValueError(msg)

        if self.quadrature_points < 1:
            msg = (
                "a truncated marginal quadrature has at least one point: "
                f"{self.quadrature_points!r}"
            )

            raise ValueError(msg)

        if self.quadrature_seed < 0:
            msg = (
                "a truncated marginal quadrature seed is non-negative: "
                f"{self.quadrature_seed!r}"
            )

            raise ValueError(msg)

        return self


class ResponseSpec(dx.Model, extra="forbid"):
    """Declares a response model.

    Parameters
    ----------
    family : ResponseFamily
        Observation family for the latent response.
    levels : int
        Number of ordered response levels.
    predictor : tuple[Predictor, ...]
        Terms whose sum defines the latent response.
    identification : ScaleIdentification
        Constraint identifying the latent scale.
    objective : LikelihoodObjective
        Treatment of the response likelihood.
    priors : PriorBlock
        Priors for parameters drawn by the response family.
    normalization : ScaleNormalization
        Transformation applied before observing the response.
    parameters : tuple[ParameterFormula, ...]
        Linear predictor of each auxiliary parameter that carries one, and
        empty where every auxiliary parameter is drawn or held.
    """

    family: ResponseFamily
    levels: int
    predictor: tuple[Predictor, ...]
    identification: ScaleIdentification
    objective: LikelihoodObjective
    priors: PriorBlock = dx.field(default_factory=PriorBlock)
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY
    parameters: tuple[ParameterFormula, ...] = ()

    __axioms__ = (
        dx.axiom(
            "levels >= 2",
            message="a response has at least two levels",
        ),
        dx.axiom(
            "len(predictor) > 0",
            message="a linear predictor has at least one term",
        ),
    )

    @dx.computed
    def terms(self) -> tuple[Predictor, ...]:
        """Return every term of every formula, the latent response's first."""
        return (*self.predictor, *_formula_terms(self.parameters))

    @dx.computed
    def predicted(self) -> tuple[AuxiliaryParameter, ...]:
        """Return the auxiliary parameters carrying a formula, in order."""
        return tuple(formula.parameter for formula in self.parameters)

    @dx.computed
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns the formulas read, in declaration order."""
        return _distinct(tuple(name for term in self.terms for name in term.readouts))

    @dx.computed
    def factors(self) -> tuple[str, ...]:
        """Return the factors the formulas read, in declaration order."""
        return _distinct(tuple(name for term in self.terms for name in term.factors))

    @dx.computed
    def groups(self) -> tuple[str, ...]:
        """Return the grouping factors the formulas vary over, in order."""
        return _distinct(tuple(name for term in self.terms for name in term.groups))

    @dx.model_validator()
    def _each_term_is_declared_once(self) -> ResponseSpec:
        """Validate the predictor terms.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a term is declared twice, or two fixed effects read the same
            readouts and factors.
        """
        for position, term in enumerate(self.predictor):
            if term in self.predictor[:position]:
                msg = f"a linear predictor declares the term {term!r} twice"

                raise ValueError(msg)

        _ensure_effects_distinct(self.predictor, "a linear predictor")

        return self

    @dx.model_validator()
    def _each_parameter_carries_one_formula(self) -> ResponseSpec:
        """Validate the auxiliary formulas against one another.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If two formulas predict the same parameter.
        """
        for position, formula in enumerate(self.parameters):
            if formula.parameter in self.predicted[:position]:
                msg = (
                    f"the {formula.parameter.value} parameter carries two "
                    f"linear predictors"
                )

                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_predicted_discrimination_is_not_also_drawn(self) -> ResponseSpec:
        """Validate a discrimination formula against the response family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If the family has no discrimination to predict, or draws one
            value for it while a formula gives it one per row.
        """
        if AuxiliaryParameter.DISCRIMINATION not in self.predicted:
            return self

        if not self.family.takes_discrimination:
            msg = (
                f"a linear predictor was declared for discrimination, and the "
                f"{self.family.family} family has none"
            )

            raise ValueError(msg)

        if self.family.draws_discrimination:
            msg = (
                f"the {self.family.family} family draws one discrimination and "
                f"a linear predictor gives it one per row"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_predicted_discrimination_leaves_the_scale_identified(self) -> ResponseSpec:
        """Validate a discrimination formula against the identification.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If the latent scale is identified by anything other than the
            variance of the cutpoints.
        """
        if AuxiliaryParameter.DISCRIMINATION not in self.predicted:
            return self

        if isinstance(self.identification, FixedCutpointVariance):
            return self

        msg = (
            f"a predicted discrimination scales the location and the cutpoints "
            f"of every row, so the latent scale is identified by the variance "
            f"of the cutpoints, and the {self.identification.constraint!r} "
            f"constraint was declared"
        )

        raise ValueError(msg)

    @dx.model_validator()
    def _priors_match_the_family(self) -> ResponseSpec:
        """Validate priors against the response family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a drawn parameter lacks a prior or an unused parameter has one.
        """
        declared = (
            ("thresholds", self.family.draws_thresholds, self.priors.thresholds),
            (
                "discrimination",
                self.family.draws_discrimination,
                self.priors.discrimination,
            ),
            ("scale", self.family.draws_scale, self.priors.scale),
        )

        for parameter, drawn, prior in declared:
            if drawn and prior is None:
                msg = (
                    f"the {self.family.family} family draws {parameter} and the "
                    f"prior block states no prior over it"
                )

                raise ValueError(msg)

            if not drawn and prior is not None:
                msg = (
                    f"the prior block states a prior over {parameter}, which the "
                    f"{self.family.family} family does not draw"
                )

                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _bounded_family_uses_its_continuous_declaration(self) -> ResponseSpec:
        """Refuse a bounded family in an ordinal response declaration.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        TypeError
            If the bounded family is paired with ordinal levels.
        """
        if isinstance(self.family, Bounded):
            msg = (
                "the bounded family records a floating response through "
                "BoundedResponseSpec, not an ordinal ResponseSpec"
            )

            raise TypeError(msg)

        return self

    @dx.model_validator()
    def _a_rank_consistent_head_has_one_identified_scalar_link(
        self,
    ) -> ResponseSpec:
        """Validate a rank-consistent head's identified scalar predictor.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        TypeError
            If another scale-identification constraint is paired with this
            family.
        ValueError
            If the predictor is not exactly one logit link, or the response is
            normalized.
        """
        if not isinstance(self.family, RankConsistent):
            return self

        if len(self.predictor) != 1 or not isinstance(self.predictor[0], LinkTerm):
            kinds = tuple(term.kind for term in self.predictor)
            msg = (
                "the rank-consistent conditional construction takes exactly one "
                f"scalar link term, and this predictor declares {kinds!r}"
            )

            raise ValueError(msg)

        term = self.predictor[0]

        if term.linking.family != "logit":
            msg = (
                "the rank-consistent conditional construction is identified on "
                f"the logit scale, and the link declares {term.linking.family!r}"
            )

            raise ValueError(msg)

        if not isinstance(self.identification, IdentityLinkOnLogit):
            msg = (
                "the rank-consistent conditional construction multiplies its "
                "predictor by task weights, so its sole link is identified by "
                "identity_link_on_logit"
            )

            raise TypeError(msg)

        if self.normalization is not ScaleNormalization.IDENTITY:
            msg = (
                "the rank-consistent conditional construction classifies the "
                "recorded level and does not normalize it"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_tie_is_recorded_between_two_alternatives(self) -> ResponseSpec:
        """Validate the level count against a choice family that records ties.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a tie is recordable over other than three levels.
        """
        family = self.family

        if not isinstance(family, Choice) or not family.ties:
            return self

        if self.levels != 3:
            msg = (
                f"a tie is recorded between two alternatives, over the three "
                f"levels they and it make, and this response has {self.levels}"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_choice_predictor_tells_the_alternatives_apart(self) -> ResponseSpec:
        """Validate the predictor terms against a choice family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a term reads neither a readout nor the presentation position,
            and so takes one value on every alternative of a row.
        """
        if not isinstance(self.family, Choice):
            return self

        for term in self.predictor:
            if not term.readouts and POSITION not in term.factors:
                msg = (
                    f"a choice is made between the alternatives of a row, and the "
                    f"{term.kind} term {term!r} takes one value on all of them"
                )

                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_drawn_choice_slope_is_identified(self) -> ResponseSpec:
        """Validate a drawn choice discrimination against the predictor's free scales.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a drawn discrimination scales both a free linking slope and a
            free position bias while the identification pins neither.
        """
        family = self.family

        if not isinstance(family, Choice) or not family.discrimination:
            return self

        if isinstance(self.identification, FixedDiscrimination | IdentityLinkOnLogit):
            return self

        free = tuple(term for term in self.predictor if _free_link_scale(term))
        biased = tuple(term for term in self.predictor if POSITION in term.factors)

        if free and biased:
            msg = (
                f"the choice family draws the slope on the difference, the linking "
                f"over {free[0].readouts[0]!r} leaves its slope free, the "
                f"{POSITION!r} factor leaves the position bias free, and the "
                f"{self.identification.constraint!r} constraint pins neither"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _the_predictor_is_identified(self) -> ResponseSpec:
        """Validate predictor scales against the identification constraint.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If fixed discrimination leaves a composite predictor with two
            independently free multiplicative scales.
        """
        if isinstance(self.identification, FixedCutpointVariance):
            if self.levels < 3:
                msg = "fixed cutpoint variance requires at least two cutpoints"

                raise ValueError(msg)

            return self

        if not isinstance(self.identification, FixedDiscrimination):
            return self

        for term in self.predictor:
            if _scale_degenerate(term):
                msg = (
                    "fixed discrimination leaves the predictor term "
                    f"{term!r} scale-degenerate"
                )

                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_held_discrimination_is_not_drawn(self) -> ResponseSpec:
        """Validate a pinned discrimination against the response family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If the identification holds discrimination at a value while the
            family draws it.
        """
        identification = self.identification

        if (
            isinstance(identification, FixedDiscrimination)
            and self.family.draws_discrimination
        ):
            msg = (
                f"the family draws discrimination and the identification holds "
                f"it at {identification.value!r}"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _the_family_takes_each_formula(self) -> ResponseSpec:
        """Validate the auxiliary formulas against the response family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a formula is declared over an auxiliary parameter of a family
            that takes none.
        """
        if self.parameters and not self.family.takes_formulas:
            msg = (
                f"the {self.family.family} family takes no formula over an "
                f"auxiliary parameter, and this specification declares one over "
                f"{self.parameters[0].parameter.value!r}"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _a_program_observes_the_recorded_level(self) -> ResponseSpec:
        """Validate normalization against a family that compiles to a program.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a family whose program observes the recorded level is declared
            with a normalization of the response.
        """
        if (
            self.family.programmed
            and self.normalization is not ScaleNormalization.IDENTITY
        ):
            msg = (
                f"the {self.family.family} family observes the recorded level "
                f"itself, and this specification declares "
                f"{self.normalization.value!r} normalization of the response"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _each_installed_linking_accepts_its_term(self) -> ResponseSpec:
        """Validate every link term against its installed linking family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If an installed linking family refuses a term declaring it.
        """
        _ensure_installed_linkings_accept(self.terms, self.identification)

        return self

    @dx.model_validator()
    def _normalization_corrects_scale_usage_once(self) -> ResponseSpec:
        """Validate normalization against the response family.

        Returns
        -------
        ResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If ridit scaling is combined with a threshold-based family.
        """
        if (
            self.normalization is ScaleNormalization.RIDIT
            and self.family.draws_thresholds
        ):
            msg = (
                f"ridit scaling corrects for scale usage visibly, before the "
                f"response is read, and the thresholds the {self.family.family} "
                f"family fits correct for it again invisibly, inside the fit, so "
                f"declaring both corrects for it twice"
            )

            raise ValueError(msg)

        return self


class BoundedResponseSpec(dx.Model, extra="forbid"):
    """Declares a bounded-continuous response model.

    Parameters
    ----------
    family : Bounded
        Truncated-normal observation family.
    predictor : tuple[Predictor, ...]
        Terms whose sum defines ``eta`` before the sigmoid location link.
    identification : ScaleIdentification
        Constraint identifying the latent predictor.
    priors : PriorBlock
        Prior of a drawn global observation scale.
    normalization : ScaleNormalization
        Transformation from each row's original scale to the unit interval.
    """

    family: Bounded
    predictor: tuple[Predictor, ...]
    identification: ScaleIdentification
    priors: PriorBlock = dx.field(default_factory=PriorBlock)
    normalization: ScaleNormalization = ScaleNormalization.UNIT_INTERVAL

    __axioms__ = (
        dx.axiom(
            "len(predictor) > 0",
            message="a linear predictor has at least one term",
        ),
    )

    @dx.computed
    def terms(self) -> tuple[Predictor, ...]:
        """Return the latent response's predictor terms."""
        return self.predictor

    @dx.computed
    def predicted(self) -> tuple[AuxiliaryParameter, ...]:
        """Return that no auxiliary parameter has a row-wise formula."""
        return ()

    @dx.computed
    def parameters(self) -> tuple[ParameterFormula, ...]:
        """Return that the family accepts no auxiliary formula."""
        return ()

    @dx.computed
    def readouts(self) -> tuple[str, ...]:
        """Return readout columns in declaration order without repeats."""
        return _distinct(tuple(name for term in self.terms for name in term.readouts))

    @dx.computed
    def factors(self) -> tuple[str, ...]:
        """Return factor columns in declaration order without repeats."""
        return _distinct(tuple(name for term in self.terms for name in term.factors))

    @dx.computed
    def groups(self) -> tuple[str, ...]:
        """Return grouping factors in declaration order without repeats."""
        return _distinct(tuple(name for term in self.terms for name in term.groups))

    @dx.model_validator()
    def _the_predictor_is_declared_once(self) -> BoundedResponseSpec:
        """Validate predictor terms and link families.

        Returns
        -------
        BoundedResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If a term repeats, two fixed effects read the same readouts and
            factors, no link term is declared, or a link is not a logit.
        """
        for position, term in enumerate(self.predictor):
            if term in self.predictor[:position]:
                msg = f"a linear predictor declares the term {term!r} twice"

                raise ValueError(msg)

        _ensure_effects_distinct(self.predictor, "a linear predictor")

        links = tuple(term for term in self.predictor if isinstance(term, LinkTerm))

        if not links:
            msg = "the bounded family needs a logit link term in its predictor"

            raise ValueError(msg)

        unsupported = tuple(
            term.linking.family for term in links if term.linking.family != "logit"
        )

        if unsupported:
            msg = (
                "the bounded family accepts logit linking alone, and the predictor "
                f"declares {unsupported[0]!r}"
            )

            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _each_installed_linking_accepts_its_term(self) -> BoundedResponseSpec:
        """Validate every link term against its installed linking family.

        Returns
        -------
        BoundedResponseSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If an installed linking family refuses a term declaring it.
        """
        _ensure_installed_linkings_accept(self.terms, self.identification)

        return self

    @dx.model_validator()
    def _the_scale_is_identified(self) -> BoundedResponseSpec:
        """Validate identification, normalization and the scale declaration.

        Returns
        -------
        BoundedResponseSpec
            This specification after validation.

        Raises
        ------
        TypeError
            If another scale-identification constraint is declared.
        ValueError
            If normalization or the global scale prior is unsupported.
        """
        if not isinstance(self.identification, IdentityLinkOnLogit):
            msg = (
                "the bounded family's location is sigmoid(eta), so its latent "
                "predictor is identified by identity_link_on_logit"
            )

            raise TypeError(msg)

        if self.normalization is not ScaleNormalization.UNIT_INTERVAL:
            msg = (
                "the bounded family observes the response on its declared closed "
                "scale through 'unit_interval' normalization"
            )

            raise ValueError(msg)

        if self.priors.thresholds is not None or self.priors.discrimination is not None:
            msg = "the bounded family accepts a scale prior alone"

            raise ValueError(msg)

        prior = self.priors.scale

        if self.family.scale is not None:
            if prior is not None:
                msg = (
                    "the bounded family fixes its scale and also declares a scale prior"
                )

                raise ValueError(msg)

            return self

        if prior is None or prior.distribution != "LogNormal":
            declared = None if prior is None else prior.distribution
            msg = (
                "the bounded family's drawn scale is strictly positive under a "
                f"LogNormal prior, not {declared!r}"
            )

            raise ValueError(msg)

        if (
            len(prior.parameters) != 2
            or not all(math.isfinite(value) for value in prior.parameters)
            or prior.parameters[1] <= 0.0
        ):
            msg = (
                "the bounded family's LogNormal scale prior needs a finite location "
                f"and a finite positive spread: {prior.parameters!r}"
            )

            raise ValueError(msg)

        return self


if TYPE_CHECKING:
    type ResponseDeclaration = ResponseSpec | BoundedResponseSpec
else:
    type ResponseDeclaration = ResponseSpec | BoundedResponseSpec | ResponseDeclaration
"""An ordinal or bounded-continuous response-model declaration."""


def _term_tree(term: Predictor) -> tuple[Predictor, ...]:
    """Return a term and every term nested inside it.

    Parameters
    ----------
    term : Predictor
        Root of the predictor-term tree.
    """
    match term:
        case InteractionTerm(left=left, right=right):
            return (term, *_term_tree(left), *_term_tree(right))
        case RandomSlope(term=varying) | CorrelatedRandomSlope(term=varying):
            return (term, *_term_tree(varying))
        case _:
            return (term,)


def _link_terms(
    term: Predictor, /, *, correlated: bool = False
) -> tuple[tuple[LinkTerm, bool], ...]:
    """Return every link term inside a term and whether a correlated slope holds it.

    Parameters
    ----------
    term : Predictor
        Root of the predictor-term tree.
    correlated : bool
        Whether the root lies inside a correlated random slope, whose basis
        enters at a unit slope.
    """
    match term:
        case LinkTerm():
            return ((term, correlated),)
        case InteractionTerm(left=left, right=right):
            return (
                *_link_terms(left, correlated=correlated),
                *_link_terms(right, correlated=correlated),
            )
        case CorrelatedRandomSlope(term=varying):
            return _link_terms(varying, correlated=True)
        case RandomSlope(term=varying):
            return _link_terms(varying, correlated=correlated)
        case _:
            return ()


def installed_linkings() -> Registry[LinkingCompiler]:
    """Return the linking families the ``lofi.linking`` entry-point group lists.

    Returns
    -------
    Registry[LinkingCompiler]
        The discovered registry, read afresh so that a family installed after
        an earlier read is listed.
    """
    linkings: Registry[LinkingCompiler] = Registry("linking")

    return linkings


def _ensure_installed_linkings_accept(
    terms: tuple[Predictor, ...], identification: ScaleIdentification
) -> None:
    """Compile every link term whose linking family is installed, once.

    The program form of each installed family builds the term under the
    slope the identification holds, so a term naming a column role, a pinned
    parameter or a pinned slope its family refuses is refused here. A term
    whose family is not installed is compiled only by a compiler that holds
    that family.

    Parameters
    ----------
    terms : tuple[Predictor, ...]
        Every term of every formula of a response declaration.
    identification : ScaleIdentification
        Constraint fixing the slopes the link terms enter at.

    Raises
    ------
    ValueError
        If an installed family refuses a term.
    """
    linkings = installed_linkings()
    available = frozenset(linkings.available())

    for root in terms:
        for term, correlated in _link_terms(root):
            if term.linking.family not in available:
                continue

            linkings.get(term.linking.family)(
                term,
                term.readout,
                unit_slope=identification.unit_slope or correlated,
                signed_slope=identification.signed_slope,
            )


class ModelComparisonCriterion(enum.StrEnum):
    """Selects the predictive criterion used to compare fitted models."""

    PSIS_LOO = "psis_loo"


class ModelWeighting(enum.StrEnum):
    """Selects the model-weight estimator used by the comparison."""

    STACKING = "stacking"
    BB_PSEUDO_BMA = "BB-pseudo-BMA"
    PSEUDO_BMA = "pseudo-BMA"


class ModelComparisonSpec(dx.Model, extra="forbid"):
    """Declares a fitted-model comparison and its Pareto reliability gate."""

    candidates: tuple[str, ...]
    candidate: str
    criterion: ModelComparisonCriterion
    weighting: ModelWeighting
    pareto_fraction: float
    pareto_threshold: float

    @dx.model_validator()
    def _candidate_belongs_to_the_declared_set(self) -> ModelComparisonSpec:
        """Validate the complete candidate set and this point's member of it."""
        names = tuple(
            ensure_name("a model candidate", name) for name in self.candidates
        )

        if len(names) < 2:
            msg = "a fitted-model comparison declares at least two candidates"
            raise ValueError(msg)

        if len(set(names)) != len(names):
            msg = f"a fitted-model comparison repeats a candidate: {names!r}"
            raise ValueError(msg)

        candidate = ensure_name("this fit's model candidate", self.candidate)

        if candidate not in names:
            msg = (
                f"this fit's candidate {candidate!r} is not among the declared "
                f"candidates {names!r}"
            )
            raise ValueError(msg)

        return self

    @dx.validates("pareto_fraction")
    @classmethod
    def _fraction_is_a_proportion(cls, value: float) -> float:
        """Return a finite proportion of observations admitted above the gate."""
        fraction = ensure_finite("a Pareto gate's fraction", value)

        if not 0.0 <= fraction <= 1.0:
            msg = f"a Pareto gate's fraction is not a proportion: {value!r}"
            raise ValueError(msg)

        return fraction

    @dx.validates("pareto_threshold")
    @classmethod
    def _threshold_is_positive(cls, value: float) -> float:
        """Return a finite positive Pareto-shape threshold."""
        threshold = ensure_finite("a Pareto gate's threshold", value)

        if threshold <= 0.0:
            msg = f"a Pareto gate's threshold is not positive: {value!r}"
            raise ValueError(msg)

        return threshold


class ComparisonSpec(dx.Model, extra="forbid"):
    """Declares how one response model is held out and compared.

    The held-out unit is a name rather than a closed enumeration: ``response``,
    ``participant`` and ``item`` select the built-in row columns, and every other
    name selects a factor column of the design.
    """

    held_out_unit: str
    readout_units: dict[str, str] = dx.field(converter=_in_name_order)
    splitter: str
    folds: FoldCount
    seed: int
    coverage_threshold: float
    unseen_participant: HeldOutParticipantPolicy | None
    models: ModelComparisonSpec

    @dx.validates("held_out_unit")
    @classmethod
    def _unit_is_named(cls, value: str) -> str:
        """Return the named row column or factor held out together."""
        named = ensure_name("a comparison's held-out unit", value)

        if not named.strip():
            msg = "a comparison's held-out unit is blank"
            raise ValueError(msg)

        return named

    @dx.validates("readout_units")
    @classmethod
    def _readout_units_are_named(cls, values: dict[str, str]) -> dict[str, str]:
        """Return a canonical map from named readouts to named grains."""
        checked: dict[str, str] = {}

        for readout, unit in values.items():
            named_readout = ensure_name("a comparison readout", readout)
            named_unit = ensure_name(
                f"the grain of comparison readout {readout!r}", unit
            )

            if not named_readout.strip() or not named_unit.strip():
                msg = "a comparison readout and its grain are not blank"
                raise ValueError(msg)

            checked[named_readout] = named_unit

        return checked

    @dx.validates("splitter")
    @classmethod
    def _splitter_is_installed(cls, value: str) -> str:
        """Return the tag of an installed splitter."""
        named = ensure_name("a comparison splitter", value)
        Registry[SplitterFactory]("splitter").get(named)
        return named

    @dx.validates("coverage_threshold")
    @classmethod
    def _coverage_is_a_proportion(cls, value: float) -> float:
        """Return the largest unseen-participant fraction admitted unguided."""
        threshold = ensure_finite("a comparison's coverage threshold", value)

        if not 0.0 <= threshold <= 1.0:
            msg = f"a comparison's coverage threshold is not a proportion: {value!r}"
            raise ValueError(msg)

        return threshold

    @dx.model_validator()
    def _splitter_accepts_the_declared_fold_count(self) -> ComparisonSpec:
        """Validate the splitter, fold count and seed together."""
        splitter_for(self)
        return self


def splitter_for(comparison: ComparisonSpec, /) -> Splitter:
    """Build the split generator selected by a comparison declaration."""
    factory = Registry[SplitterFactory]("splitter").get(comparison.splitter)

    return factory(
        comparison.held_out_unit,
        folds=comparison.folds,
        seed=comparison.seed,
    )


class ColumnLevel(dx.Model, extra="forbid"):
    """Declares one column level as a sequence of primitive types.

    Parameters
    ----------
    label : str
        Level the column attribute records in a judgment design.
    arguments : tuple[str, ...]
        Primitive type of each argument slot, in order.
    """

    label: str
    arguments: tuple[str, ...]

    __axioms__ = (dx.axiom("len(label) > 0", message="a column level has a label"),)


class ColumnInventory(dx.Model, extra="forbid"):
    """Declares the column levels under comparison and the primitive types they hold.

    Argument sequences are padded to the widest one, and a slot the padding
    fills holds the null type, whose index this inventory assigns after the
    primitive types so that no primitive type shares it.

    Parameters
    ----------
    primitive_types : tuple[str, ...]
        Types an argument slot may hold, in index order, the null type apart.
    null_type : str
        Name of the type an absent argument slot holds.
    levels : tuple[ColumnLevel, ...]
        Column levels under comparison, in index order.
    """

    primitive_types: tuple[str, ...]
    null_type: str
    levels: tuple[ColumnLevel, ...]

    __axioms__ = (
        dx.axiom(
            "len(primitive_types) > 0",
            message="an inventory declares a primitive type",
        ),
        dx.axiom("len(null_type) > 0", message="an inventory names its null type"),
        dx.axiom("len(levels) > 0", message="an inventory declares a column level"),
    )

    @dx.computed
    def labels(self) -> tuple[str, ...]:
        """Return the column labels, in index order."""
        return tuple(level.label for level in self.levels)

    @dx.computed
    def arity(self) -> int:
        """Return how many argument slots every column level is padded to."""
        return max((len(level.arguments) for level in self.levels), default=0)

    @dx.computed
    def null_index(self) -> int:
        """Return the slot index of the null type, which no primitive type holds."""
        return len(self.primitive_types)

    @dx.computed
    def type_count(self) -> int:
        """Return how many types a slot may hold, the null type included."""
        return len(self.primitive_types) + 1

    @dx.model_validator()
    def _every_name_is_declared_once_and_every_argument_is_typed(
        self,
    ) -> ColumnInventory:
        """Validate the type inventory against the column levels declared over it.

        Returns
        -------
        ColumnInventory
            This inventory after validation.

        Raises
        ------
        ValueError
            If a primitive type or a column label is empty or declared twice,
            if the null type repeats a primitive type, if an argument names
            no declared primitive type, or if no level takes an argument.
        """
        for position, name in enumerate(self.primitive_types):
            if not name:
                msg = f"the primitive type at position {position} has no name"

                raise ValueError(msg)

            if name in self.primitive_types[:position]:
                msg = f"the primitive type {name!r} is declared twice"

                raise ValueError(msg)

        if self.null_type in self.primitive_types:
            msg = (
                f"the null type {self.null_type!r} repeats a primitive type, and "
                f"an absent slot is not a slot holding a type"
            )

            raise ValueError(msg)

        declared = frozenset(self.primitive_types)

        for position, level in enumerate(self.levels):
            if level.label in self.labels[:position]:
                msg = f"the column level {level.label!r} is declared twice"

                raise ValueError(msg)

            for slot, argument in enumerate(level.arguments):
                if argument not in declared:
                    msg = (
                        f"slot {slot} of the column level {level.label!r} holds "
                        f"{argument!r}, which the inventory does not declare"
                    )

                    raise ValueError(msg)

        if self.arity < 1:
            msg = "no column level takes an argument, so none decomposes"

            raise ValueError(msg)

        return self


class LexicalPrimitiveInventory(dx.Model, extra="forbid"):
    """Declares the lexical-semantic primitives a row level may possess.

    These are not the primitive types a column level decomposes into. A row
    level possesses any number of them at once, while an argument slot holds
    exactly one primitive type.

    Parameters
    ----------
    names : tuple[str, ...]
        The lexical-semantic primitives, in index order.
    """

    names: tuple[str, ...]

    __axioms__ = (
        dx.axiom(
            "len(names) > 0",
            message="an inventory declares a lexical-semantic primitive",
        ),
    )

    @dx.computed
    def count(self) -> int:
        """Return how many lexical-semantic primitives a row level may possess."""
        return len(self.names)

    @dx.model_validator()
    def _every_primitive_is_named_once(self) -> LexicalPrimitiveInventory:
        """Validate the lexical-semantic primitive names.

        Returns
        -------
        LexicalPrimitiveInventory
            This inventory after validation.

        Raises
        ------
        ValueError
            If a name is empty or declared twice.
        """
        for position, name in enumerate(self.names):
            if not name:
                msg = (
                    f"the lexical-semantic primitive at position {position} has no name"
                )

                raise ValueError(msg)

            if name in self.names[:position]:
                msg = f"the lexical-semantic primitive {name!r} is declared twice"

                raise ValueError(msg)

        return self


type _PinTarget = Literal["semantic_primitives", "item_clusters"]
type _PinMode = Literal["independent_sigmoid", "normalized_softmax"]


class PinnedPrefix(dx.Model, extra="forbid"):
    """Declares special items pinned to a prefix of a latent inventory.

    Parameters
    ----------
    target : Literal["semantic_primitives", "item_clusters"]
        Inventory the special-item rows identify.
    mode : Literal["independent_sigmoid", "normalized_softmax"]
        Link used for the regular rows.
    cardinality : int
        Number of leading item rows and inventory columns reserved together.
    """

    target: _PinTarget
    mode: _PinMode
    cardinality: int

    @dx.model_validator()
    def _target_has_its_mode_and_cardinality_is_nonnegative(self) -> PinnedPrefix:
        """Validate the target, mode and cardinality together.

        Returns
        -------
        PinnedPrefix
            This prefix after validation.

        Raises
        ------
        TypeError
            If the cardinality is not an integer.
        ValueError
            If the cardinality is negative or the mode does not belong to the
            target.
        """
        ensure_non_negative("pinned-prefix cardinality", self.cardinality)

        expected: _PinMode

        if self.target == "semantic_primitives":
            expected = "independent_sigmoid"
        else:
            expected = "normalized_softmax"

        if self.mode != expected:
            msg = (
                f"the pinned-prefix target {self.target!r} requires mode "
                f"{expected!r}, not {self.mode!r}"
            )

            raise ValueError(msg)

        return self


def _unpinned_primitives() -> PinnedPrefix:
    """Return the unpinned semantic-primitive declaration."""
    return PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=0,
    )


def _unpinned_clusters() -> PinnedPrefix:
    """Return the unpinned item-cluster declaration."""
    return PinnedPrefix(
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=0,
    )


def _validate_prefix(
    prefix: PinnedPrefix,
    *,
    target: _PinTarget,
    mode: _PinMode,
    inventory: int,
) -> None:
    """Validate one routed prefix against its target inventory."""
    if prefix.target != target:
        msg = (
            f"the pinned-prefix target is {prefix.target!r}, but this latent "
            f"targets {target!r}"
        )

        raise ValueError(msg)

    if prefix.mode != mode:
        msg = (
            f"the pinned-prefix mode is {prefix.mode!r}, but target "
            f"{target!r} uses {mode!r}"
        )

        raise ValueError(msg)

    if prefix.cardinality > inventory:
        msg = (
            f"the {target!r} pinned-prefix cardinality {prefix.cardinality} "
            f"exceeds its inventory cardinality {inventory}"
        )

        raise ValueError(msg)


class ColumnLift(dx.TaggedUnion, discriminator="lift", extra="forbid"):
    """A declaration of how a column level reaches the latent score."""

    @dx.computed
    def decomposes(self) -> bool:
        """Return whether the column level is read as a sequence of primitive types."""
        return False

    @dx.computed
    def mediators(self) -> tuple[str, ...]:
        """Return the row-side latents the lift declares, in declaration order."""
        return ()


class MemorizedCells(ColumnLift, extra="forbid"):
    """Declares one free score per row and column level, which memorizes the table.

    Parameters
    ----------
    lift : Literal["memorized"]
        Tagged-union discriminator.
    """

    lift: Literal["memorized"] = "memorized"


class ColumnLabels(ColumnLift, extra="forbid"):
    """Declares a column vector read from the column label alone.

    The label is an index into a free table, so two column levels sharing a
    primitive type share nothing. The declared depth is how many affine
    layers the label's vector passes through before a row level reads it.

    Parameters
    ----------
    lift : Literal["labels"]
        Tagged-union discriminator.
    depth : int
        Number of affine layers between the label table and the row level.
    """

    lift: Literal["labels"] = "labels"
    depth: int

    __axioms__ = (dx.axiom("depth >= 1", message="a label lift has a layer"),)


class JointTypeTensor(ColumnLift, extra="forbid"):
    """Declares a column vector lifted by outer product across the slots.

    A complex type is the joint tensor of its slots' primitive types, so a
    column level with a slot the padding filled carries the null type's vector
    in that axis, and the row level reads the flattened joint tensor.

    Parameters
    ----------
    lift : Literal["joint_type"]
        Tagged-union discriminator.
    """

    lift: Literal["joint_type"] = "joint_type"

    @dx.computed
    @override
    def decomposes(self) -> bool:
        """Return whether the column level is read as a sequence of primitive types.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True


class LexicalPossession(ColumnLift, extra="forbid"):
    """Declares a column read through the lexical-semantic primitives a row has.

    Possession is many per row level, so it is read as one independent
    probability per lexical-semantic primitive, and each primitive scores the
    column level's slots through the shared primitive-type map.

    Parameters
    ----------
    lift : Literal["lexical_possession"]
        Tagged-union discriminator.
    lexical_primitives : LexicalPrimitiveInventory
        The lexical-semantic primitives a row level may possess.
    prefix : PinnedPrefix
        Special row levels pinned to matching semantic primitives.
    """

    lift: Literal["lexical_possession"] = "lexical_possession"
    lexical_primitives: LexicalPrimitiveInventory
    prefix: PinnedPrefix = dx.field(default_factory=_unpinned_primitives)

    @dx.model_validator()
    def _prefix_targets_the_primitive_inventory(self) -> LexicalPossession:
        """Validate the pinned prefix against the primitive inventory."""
        _validate_prefix(
            self.prefix,
            target="semantic_primitives",
            mode="independent_sigmoid",
            inventory=self.lexical_primitives.count,
        )

        return self

    @dx.computed
    @override
    def decomposes(self) -> bool:
        """Return whether the column level is read as a sequence of primitive types.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def mediators(self) -> tuple[str, ...]:
        """Return the row-side latents the lift declares, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The possession latent alone.
        """
        return ("possession",)


class RowClasses(ColumnLift, extra="forbid"):
    """Declares a column read through a learned row class and its entailments.

    Class membership is single-label, so it is read as one distribution over
    classes, and one class entails another's column levels through a declared
    class-by-class relation.

    Parameters
    ----------
    lift : Literal["row_classes"]
        Tagged-union discriminator.
    classes : int
        Number of row classes the latent ranges over.
    prefix : PinnedPrefix
        Special row levels pinned to matching item clusters.
    """

    lift: Literal["row_classes"] = "row_classes"
    classes: int
    prefix: PinnedPrefix = dx.field(default_factory=_unpinned_clusters)

    __axioms__ = (
        dx.axiom("classes >= 2", message="a class latent ranges over two classes"),
    )

    @dx.model_validator()
    def _prefix_targets_the_class_inventory(self) -> RowClasses:
        """Validate the pinned prefix against the class inventory."""
        _validate_prefix(
            self.prefix,
            target="item_clusters",
            mode="normalized_softmax",
            inventory=self.classes,
        )

        return self

    @dx.computed
    @override
    def decomposes(self) -> bool:
        """Return whether the column level is read as a sequence of primitive types.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def mediators(self) -> tuple[str, ...]:
        """Return the row-side latents the lift declares, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The membership latent alone.
        """
        return ("membership",)


class PossessionAndClasses(ColumnLift, extra="forbid"):
    """Declares both row-side latents, each with the link its support takes.

    Class membership is single-label and possession of a lexical-semantic
    primitive is not, so the membership latent is a distribution over
    classes and the possession latent is one independent probability per
    lexical-semantic primitive.

    Parameters
    ----------
    lift : Literal["possession_and_classes"]
        Tagged-union discriminator.
    lexical_primitives : LexicalPrimitiveInventory
        The lexical-semantic primitives a row level may possess.
    classes : int
        Number of row classes the membership latent ranges over.
    primitive_prefix : PinnedPrefix
        Special row levels pinned to matching semantic primitives.
    cluster_prefix : PinnedPrefix
        Special row levels pinned to matching item clusters.
    """

    lift: Literal["possession_and_classes"] = "possession_and_classes"
    lexical_primitives: LexicalPrimitiveInventory
    classes: int
    primitive_prefix: PinnedPrefix = dx.field(default_factory=_unpinned_primitives)
    cluster_prefix: PinnedPrefix = dx.field(default_factory=_unpinned_clusters)

    __axioms__ = (
        dx.axiom("classes >= 2", message="a class latent ranges over two classes"),
    )

    @dx.model_validator()
    def _each_prefix_targets_its_inventory(self) -> PossessionAndClasses:
        """Validate the primitive and cluster prefix routes."""
        _validate_prefix(
            self.primitive_prefix,
            target="semantic_primitives",
            mode="independent_sigmoid",
            inventory=self.lexical_primitives.count,
        )
        _validate_prefix(
            self.cluster_prefix,
            target="item_clusters",
            mode="normalized_softmax",
            inventory=self.classes,
        )

        return self

    @dx.computed
    @override
    def decomposes(self) -> bool:
        """Return whether the column level is read as a sequence of primitive types.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @dx.computed
    @override
    def mediators(self) -> tuple[str, ...]:
        """Return the row-side latents the lift declares, in declaration order.

        Returns
        -------
        tuple[str, ...]
            The membership latent and the possession latent.
        """
        return ("membership", "possession")


class FitSchedule(dx.Model, extra="forbid"):
    """Declares optimization and initialization of one table-ablation rung.

    Parameters
    ----------
    optimizer : OptimizerSpec
        Installed optimizer and all of its settings.
    steps : int
        Number of parameter updates.
    batch_size : int | None
        Rows in each deterministic minibatch, or ``None`` for a full batch.
    initial_scale : float
        Standard deviation of each initial parameter draw.
    """

    optimizer: OptimizerSpec
    steps: int
    batch_size: int | None
    initial_scale: float

    @dx.validates("steps")
    @classmethod
    def _steps_are_positive(cls, value: int) -> int:
        """Return a positive update count."""
        return ensure_positive("a fit schedule's step count", value)

    @dx.validates("batch_size")
    @classmethod
    def _batch_is_positive_or_full(cls, value: int | None) -> int | None:
        """Return a positive minibatch size or the explicit full-batch choice."""
        if value is None:
            return None

        return ensure_positive("a fit schedule's batch size", value)

    @dx.validates("initial_scale")
    @classmethod
    def _initial_scale_is_positive(cls, value: float) -> float:
        """Return a finite positive initialization scale."""
        scale = ensure_finite("a fit schedule's initial scale", value)

        if scale <= 0.0:
            msg = f"a fit schedule's initial scale is not positive: {value!r}"
            raise ValueError(msg)

        return scale


class TableRows(dx.Model, extra="forbid"):
    """Declares the attribute a table's rows record and its levels.

    Parameters
    ----------
    attribute : str
        Design factor whose level a judgment's row is.
    levels : tuple[str, ...]
        Row levels under comparison, in index order.
    """

    attribute: str
    levels: tuple[str, ...]

    __axioms__ = (
        dx.axiom("len(attribute) > 0", message="a table names its row attribute"),
        dx.axiom("len(levels) > 0", message="a table declares a row level"),
    )

    @dx.model_validator()
    def _every_level_is_named_once(self) -> TableRows:
        """Validate that every row level is named and declared once.

        Raises
        ------
        ValueError
            If a level has no name or is declared twice.
        """
        for position, level in enumerate(self.levels):
            if not level:
                msg = f"the row level at position {position} has no name"
                raise ValueError(msg)
            if level in self.levels[:position]:
                msg = f"the row level {level!r} is declared twice"
                raise ValueError(msg)
        return self


class TableColumns(dx.Model, extra="forbid"):
    """Declares the attribute a table's columns record and their inventory.

    Parameters
    ----------
    attribute : str
        Design factor whose level a judgment's column is.
    inventory : ColumnInventory
        Column levels under comparison and the primitive types they hold.
    """

    attribute: str
    inventory: ColumnInventory

    __axioms__ = (
        dx.axiom("len(attribute) > 0", message="a table names its column attribute"),
    )


class TableAblationSpec(dx.Model, extra="forbid"):
    """Declares one rung of the structured-versus-unstructured table comparison.

    A rung scores every row level against every column level of a declared
    table. Every rung declares the same rows, the same columns and the same
    width, so two rungs differ in the lift and in nothing else.

    Parameters
    ----------
    rows : TableRows
        The row attribute and its levels.
    columns : TableColumns
        The column attribute and the primitive types its levels hold.
    width : int
        Width of the vector a column level is represented by.
    lift : ColumnLift
        How a column level reaches the latent score, which is what the rung is.
    schedule : FitSchedule
        Initialization, optimizer, batching and update count used to fit the rung.
    """

    rows: TableRows
    columns: TableColumns
    width: int
    lift: ColumnLift
    schedule: FitSchedule

    __axioms__ = (dx.axiom("width > 0", message="a column vector has a width"),)

    @dx.computed
    def terms(self) -> tuple[Predictor, ...]:
        """Return the predictor terms every rung of the comparison implements."""
        return (
            InteractionTerm(
                left=FactorTerm(name=self.rows.attribute),
                right=FactorTerm(name=self.columns.attribute),
            ),
        )

    @dx.model_validator()
    def _the_two_attributes_differ_and_every_prefix_fits_the_rows(
        self,
    ) -> TableAblationSpec:
        """Validate the two attributes and each pinned prefix against the rows.

        Returns
        -------
        TableAblationSpec
            This specification after validation.

        Raises
        ------
        ValueError
            If the rows and the columns record one attribute, or a pinned
            prefix reserves more row levels than the table has.
        """
        if self.rows.attribute == self.columns.attribute:
            msg = (
                f"the rows and the columns are both recorded under "
                f"{self.rows.attribute!r}, and the comparison crosses them"
            )

            raise ValueError(msg)

        match self.lift:
            case LexicalPossession():
                prefixes = ((self.lift.prefix, self.lift.lexical_primitives.count),)
            case RowClasses():
                prefixes = ((self.lift.prefix, self.lift.classes),)
            case PossessionAndClasses():
                prefixes = (
                    (
                        self.lift.primitive_prefix,
                        self.lift.lexical_primitives.count,
                    ),
                    (self.lift.cluster_prefix, self.lift.classes),
                )
            case _:
                prefixes = ()

        rows = len(self.rows.levels)

        for prefix, inventory in prefixes:
            if prefix.cardinality > rows:
                msg = (
                    f"the {prefix.target!r} pinned-prefix cardinality "
                    f"{prefix.cardinality} exceeds the row cardinality {rows}"
                )

                raise ValueError(msg)

            if (
                prefix.mode == "normalized_softmax"
                and prefix.cardinality == inventory
                and prefix.cardinality < rows
            ):
                msg = (
                    f"the {prefix.target!r} pinned-prefix cardinality "
                    f"{prefix.cardinality} reserves every target column but "
                    f"leaves {rows - prefix.cardinality} regular rows"
                )

                raise ValueError(msg)

        return self


def spec_digest(spec: dx.Model) -> str:
    """Return the content hash identifying a declaration.

    The digest is over the specification's JSON form, computed fields
    included, so under one version of this module two specifications
    declaring the same model have the same digest and two differing in any
    declared field have different digests. A change to the module's
    declarations or computed fields changes every digest, so a stored
    digest identifies a specification against the version that wrote it.

    Parameters
    ----------
    spec : dx.Model
        Specification to identify.

    Returns
    -------
    str
        Hexadecimal SHA-256 of the specification's JSON form.
    """
    return sha256(spec.model_dump_json().encode("utf-8")).hexdigest()


class AgreementBin(enum.StrEnum):
    """Selects an item-level agreement stratum."""

    #: Items whose annotators agree least.
    LOW = "low"

    #: Items of middling agreement.
    MEDIUM = "medium"

    #: Items whose annotators agree most.
    HIGH = "high"


class InferenceMethod(enum.StrEnum):
    """Selects the inference a fit ran."""

    #: The No-U-Turn sampler.
    NUTS = "nuts"

    #: A full-rank Gaussian guide.
    FULL_RANK = "full_rank"

    #: A partially factorized guide over crossed random effects.
    STRUCTURED = "structured"

    #: A mean-field Gaussian guide.
    MEAN_FIELD = "mean_field"

    #: A point estimate at the posterior mode, with no draws.
    MAP = "map"


def _ensure_finite_or_absent(field: str, value: float | None) -> None:
    """Check that an optional number is finite when present.

    Parameters
    ----------
    field : str
        Name of the value, used in the error message.
    value : float | None
        Number to check, or None when the value is absent.

    Raises
    ------
    ValueError
        If the number is present and not finite.
    """
    if value is not None and not math.isfinite(value):
        msg = f"{field} is not finite: {value!r}"

        raise ValueError(msg)


class Measurement(dx.Model, extra="forbid"):
    """Declares a measured number with its count, unit and interval.

    A measurement over no units has no value, a value that is undefined has
    no interval, and an interval states the coverage it claims.

    Parameters
    ----------
    value : float | None
        The measured number, or None when it is undefined.
    n : int
        How many units the measurement rests on.
    unit : str
        Name of the unit counted by ``n``, such as an item or a participant.
    lower : float | None
        Lower interval bound, if one was computed.
    upper : float | None
        Upper interval bound, if one was computed.
    level : float | None
        Nominal coverage of the interval as a proportion, if one was computed.
    """

    value: float | None
    n: int
    unit: str
    lower: float | None = None
    upper: float | None = None
    level: float | None = None

    __axioms__ = (
        dx.axiom(
            "n >= 0",
            message="a measurement rests on a nonnegative number of units",
        ),
        dx.axiom("len(unit) > 0", message="a measurement names its unit"),
    )

    @dx.model_validator()
    def _value_count_and_interval_agree(self) -> Measurement:
        """Validate the value against its count and interval.

        Returns
        -------
        Measurement
            This measurement after validation.

        Raises
        ------
        ValueError
            If a number is not finite, a value is reported over no units, an
            interval is reported without a value, with a bound or its level
            missing, or with a level that is not a proportion, or the interval
            does not bracket the value.
        """
        _ensure_finite_or_absent("a measurement's value", self.value)
        _ensure_finite_or_absent("a measurement's lower bound", self.lower)
        _ensure_finite_or_absent("a measurement's upper bound", self.upper)
        _ensure_finite_or_absent("a measurement's coverage level", self.level)

        if self.n == 0 and self.value is not None:
            msg = f"a measurement over no units reports a value: {self.value!r}"

            raise ValueError(msg)

        parts = (self.lower, self.upper, self.level)

        if any(part is None for part in parts) and any(
            part is not None for part in parts
        ):
            msg = (
                f"a measurement's interval is incomplete: bounds {self.lower!r}, "
                f"{self.upper!r} at level {self.level!r}"
            )

            raise ValueError(msg)

        if self.lower is None or self.upper is None or self.level is None:
            return self

        if self.value is None:
            msg = (
                f"a measurement with no value reports an interval: "
                f"[{self.lower}, {self.upper}]"
            )

            raise ValueError(msg)

        if not 0.0 < self.level < 1.0:
            msg = f"a measurement's coverage level is not a proportion: {self.level!r}"

            raise ValueError(msg)

        if not self.lower <= self.value <= self.upper:
            msg = (
                f"a measurement's interval must bracket its value: "
                f"{self.lower!r}, {self.value!r}, {self.upper!r}"
            )

            raise ValueError(msg)

        return self


class ScaleProfile(dx.Model, extra="forbid"):
    """Reports a latent-scale log-posterior profile and its ridge verdict.

    Parameters
    ----------
    scales : tuple[float, ...]
        Strictly increasing positive scale factors bracketing one.
    log_posterior : tuple[float, ...]
        Finite log-posterior value at each scale factor.
    minimum_tail_drop : float
        Smallest drop from the peak required at each endpoint.
    """

    scales: tuple[float, ...]
    log_posterior: tuple[float, ...]
    minimum_tail_drop: float

    @dx.computed
    def peak(self) -> float:
        """Return the largest observed log-posterior value."""
        return max(self.log_posterior)

    @dx.computed
    def peak_scale(self) -> float:
        """Return the first scale factor attaining the observed peak."""
        return self.scales[self.log_posterior.index(self.peak)]

    @dx.computed
    def lower_tail_drop(self) -> float:
        """Return the drop from the peak to the smallest scale factor."""
        return self.peak - self.log_posterior[0]

    @dx.computed
    def upper_tail_drop(self) -> float:
        """Return the drop from the peak to the largest scale factor."""
        return self.peak - self.log_posterior[-1]

    @dx.computed
    def ridge(self) -> bool:
        """Return whether at least one tail fails to fall by the minimum."""
        return (
            self.lower_tail_drop < self.minimum_tail_drop
            or self.upper_tail_drop < self.minimum_tail_drop
        )

    @dx.model_validator()
    def _the_grid_and_profile_define_two_tails(self) -> ScaleProfile:
        """Validate the scale grid, log posterior and required tail drop."""
        if len(self.scales) < 3:
            msg = "a scale profile contains at least three scale factors"

            raise ValueError(msg)

        if len(self.log_posterior) != len(self.scales):
            msg = (
                f"a scale profile has {len(self.scales)} scale factors and "
                f"{len(self.log_posterior)} log-posterior values"
            )

            raise ValueError(msg)

        for position, scale in enumerate(self.scales):
            ensure_finite(f"scale factor {position}", scale)

            if scale <= 0.0:
                msg = f"a scale factor is not positive: {scale!r}"

                raise ValueError(msg)

        for lower, upper in zip(self.scales, self.scales[1:], strict=False):
            if lower >= upper:
                msg = f"scale factors are not strictly increasing: {self.scales!r}"

                raise ValueError(msg)

        if not self.scales[0] < 1.0 < self.scales[-1]:
            msg = f"scale factors do not bracket one: {self.scales!r}"

            raise ValueError(msg)

        for position, value in enumerate(self.log_posterior):
            ensure_finite(f"log-posterior value {position}", value)

        ensure_finite("a scale profile's minimum tail drop", self.minimum_tail_drop)

        if self.minimum_tail_drop <= 0.0:
            msg = (
                "a scale profile's minimum tail drop is not positive: "
                f"{self.minimum_tail_drop!r}"
            )

            raise ValueError(msg)

        ensure_finite(
            "a scale profile's lower tail drop",
            self.lower_tail_drop,
        )
        ensure_finite(
            "a scale profile's upper tail drop",
            self.upper_tail_drop,
        )

        return self

    def peak_measurement(self) -> Measurement:
        """Return the peak in the generic measurement form."""
        return Measurement(
            value=self.peak,
            n=len(self.scales),
            unit="scale_factor",
        )

    def lower_tail_measurement(self) -> Measurement:
        """Return the lower endpoint drop in the generic measurement form."""
        return Measurement(
            value=self.lower_tail_drop,
            n=len(self.scales),
            unit="scale_factor",
        )

    def upper_tail_measurement(self) -> Measurement:
        """Return the upper endpoint drop in the generic measurement form."""
        return Measurement(
            value=self.upper_tail_drop,
            n=len(self.scales),
            unit="scale_factor",
        )


class ConstraintCost(dx.Model, extra="forbid"):
    """Reports the predictive-density cost of a scale constraint.

    Parameters
    ----------
    constrained : Measurement
        Predictive density under the identified fit.
    unconstrained : Measurement
        Predictive density under its otherwise matched unconstrained fit.
    """

    constrained: Measurement
    unconstrained: Measurement

    @dx.computed
    def value(self) -> float:
        """Return unconstrained minus constrained predictive density."""
        unconstrained = cast("float", self.unconstrained.value)
        constrained = cast("float", self.constrained.value)

        return unconstrained - constrained

    @dx.model_validator()
    def _the_measurements_are_defined_and_matched(self) -> ConstraintCost:
        """Validate the paired predictive-density measurements."""
        if self.constrained.value is None or self.unconstrained.value is None:
            msg = "a constraint cost requires two defined predictive densities"

            raise ValueError(msg)

        if (
            self.constrained.n != self.unconstrained.n
            or self.constrained.unit != self.unconstrained.unit
        ):
            msg = (
                "a constraint cost compares predictive densities over the same "
                f"units, and the constrained fit covers {self.constrained.n} "
                f"{self.constrained.unit!r} units while the unconstrained fit "
                f"covers {self.unconstrained.n} {self.unconstrained.unit!r} units"
            )

            raise ValueError(msg)

        ensure_finite("a constraint cost's signed difference", self.value)

        return self

    def as_measurement(self) -> Measurement:
        """Return the signed cost in the generic measurement form."""
        return Measurement(
            value=self.value,
            n=self.constrained.n,
            unit=self.constrained.unit,
        )


class PseudoRSquaredRow(dx.Model, extra="forbid"):
    """Reports one fit's pseudo-R-squared against a named reference.

    Parameters
    ----------
    fit : str
        Name of the fitted specification being reported.
    reference : str
        Name of the specification supplying the reference log-likelihood.
    pseudo_r_squared : Measurement
        One less the ratio of the fit's to the reference's log-likelihood.
    """

    fit: str
    reference: str
    pseudo_r_squared: Measurement

    __axioms__ = (
        dx.axiom("len(fit) > 0", message="a pseudo-R-squared row names its fit"),
        dx.axiom(
            "len(reference) > 0",
            message="a pseudo-R-squared row names its reference",
        ),
    )


class PredictiveDensityRow(dx.Model, extra="forbid"):
    """Reports one fit's mean log predictive density.

    Parameters
    ----------
    fit : str
        Name of the fitted specification being reported.
    predictive_density : Measurement
        Mean log predictive density over the held-out response rows.
    """

    fit: str
    predictive_density: Measurement

    __axioms__ = (
        dx.axiom("len(fit) > 0", message="a predictive-density row names its fit"),
    )

    @dx.model_validator()
    def _density_is_defined(self) -> PredictiveDensityRow:
        """Require a predictive density computed over at least one row."""
        if self.predictive_density.value is None:
            msg = "a predictive-density row has no predictive density"

            raise ValueError(msg)

        return self


class PosteriorVarianceRow(dx.Model, extra="forbid"):
    """Reports one fit's posterior variance component and interval.

    Parameters
    ----------
    fit : str
        Name of the fitted specification being reported.
    group : str
        Random-effect group whose variance component is reported.
    posterior_variance : Measurement
        Posterior mean and interval of the variance component.
    """

    fit: str
    group: str
    posterior_variance: Measurement

    __axioms__ = (
        dx.axiom("len(fit) > 0", message="a posterior-variance row names its fit"),
        dx.axiom(
            "len(group) > 0",
            message="a posterior-variance row names its random-effect group",
        ),
    )

    @dx.model_validator()
    def _variance_has_an_interval(self) -> PosteriorVarianceRow:
        """Require a nonnegative variance with a posterior interval."""
        variance = self.posterior_variance

        if variance.value is None:
            msg = "a posterior-variance row has no posterior variance"

            raise ValueError(msg)

        if variance.lower is None:
            msg = "a posterior-variance row has no posterior interval"

            raise ValueError(msg)

        if variance.value < 0.0 or variance.lower < 0.0:
            msg = "a posterior-variance row reports a negative variance"

            raise ValueError(msg)

        return self


class ResponsePrediction(dx.Model, extra="forbid"):
    """Declares the predicted distribution over one recorded response.

    Parameters
    ----------
    participant : str
        Participant who gave the response.
    item : str
        Item the response was given to.
    response : int
        Zero-based index of the recorded response level.
    distribution : tuple[float, ...]
        Predicted probability of each response level, in level order.
    convention : HeldOutParticipantPolicy | None
        How the participant's effects entered the prediction, or None when the
        participant was in the fit.
    agreement : AgreementBin
        Agreement stratum of the item.
    """

    participant: str
    item: str
    response: int
    distribution: tuple[float, ...]
    convention: HeldOutParticipantPolicy | None
    agreement: AgreementBin

    __axioms__ = (
        dx.axiom(
            "len(participant) > 0",
            message="a prediction names its participant",
        ),
        dx.axiom("len(item) > 0", message="a prediction names its item"),
        dx.axiom(
            "len(distribution) >= 2",
            message="a distribution covers at least two response levels",
        ),
        dx.axiom(
            "response >= 0",
            message="a response index is nonnegative",
        ),
    )

    @dx.computed
    def log_density(self) -> float:
        """Return the natural logarithm of the recorded level's probability."""
        return math.log(self.distribution[self.response])

    @dx.model_validator()
    def _distribution_is_proper_over_the_recorded_level(self) -> ResponsePrediction:
        """Validate the distribution against the recorded level.

        Returns
        -------
        ResponsePrediction
            This prediction after validation.

        Raises
        ------
        ValueError
            If a probability is outside the unit interval, the probabilities do
            not sum to one, the recorded level is outside the distribution, or
            the recorded level has probability zero, so that its log density
            is undefined.
        """
        for level, probability in enumerate(self.distribution):
            if not 0.0 <= probability <= 1.0:
                msg = (
                    f"the probability of level {level} is outside the unit "
                    f"interval: {probability!r}"
                )

                raise ValueError(msg)

        total = math.fsum(self.distribution)

        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            msg = f"a distribution's probabilities sum to {total!r}, not one"

            raise ValueError(msg)

        if self.response >= len(self.distribution):
            msg = (
                f"a recorded level of {self.response} lies outside a distribution "
                f"over {len(self.distribution)} levels"
            )

            raise ValueError(msg)

        if self.distribution[self.response] <= 0.0:
            msg = (
                f"the recorded level {self.response} has probability zero, so its "
                f"log density is undefined"
            )

            raise ValueError(msg)

        return self


class PosteriorHandle(dx.Model, extra="forbid"):
    """Declares where a fit's posterior draws are stored.

    Parameters
    ----------
    location : str
        Path or URI of the stored draws.
    chains : int
        Number of chains drawn.
    draws : int
        Number of draws per chain.
    """

    location: str
    chains: int
    draws: int

    __axioms__ = (
        dx.axiom("len(location) > 0", message="a posterior handle names a location"),
        dx.axiom("chains >= 1", message="a posterior has at least one chain"),
        dx.axiom("draws >= 1", message="a posterior has at least one draw"),
    )


class FitDiagnostics(dx.Model, extra="forbid"):
    """Declares the diagnostics of one fit.

    Parameters
    ----------
    method : InferenceMethod
        Inference the fit ran.
    divergences : int | None
        Number of divergent transitions, or None when the inference has none.
    max_rhat : float | None
        Largest split R-hat over the parameters, if computed.
    min_ess : float | None
        Smallest effective sample size over the parameters, if computed.
    variance_components : dict[str, float]
        Variance of each random effect, keyed by group and parameter.
    """

    method: InferenceMethod
    divergences: int | None = None
    max_rhat: float | None = None
    min_ess: float | None = None
    variance_components: dict[str, float] = dx.field(default_factory=dict)

    @dx.model_validator()
    def _numbers_are_finite_and_nonnegative(self) -> FitDiagnostics:
        """Validate the numeric diagnostics.

        Returns
        -------
        FitDiagnostics
            These diagnostics after validation.

        Raises
        ------
        ValueError
            If a count or a variance is negative, or a diagnostic is not finite.
        """
        if self.divergences is not None and self.divergences < 0:
            msg = f"a divergence count is negative: {self.divergences!r}"

            raise ValueError(msg)

        _ensure_finite_or_absent("the largest R-hat", self.max_rhat)
        _ensure_finite_or_absent("the smallest effective sample size", self.min_ess)

        if self.min_ess is not None and self.min_ess < 0.0:
            msg = f"an effective sample size is negative: {self.min_ess!r}"

            raise ValueError(msg)

        for name, variance in self.variance_components.items():
            if not math.isfinite(variance) or variance < 0.0:
                msg = f"the variance component {name!r} is not a variance: {variance!r}"

                raise ValueError(msg)

        return self


class FitSummary(dx.Model, extra="forbid"):
    """Declares the outcome of fitting one response specification.

    Parameters
    ----------
    spec_hash : str
        Content hash of the fitted specification.
    posterior : PosteriorHandle | None
        Where the posterior draws are stored, or None for a point estimate.
    diagnostics : FitDiagnostics
        Diagnostics of the fit.
    identification : ScaleIdentification
        Constraint that identified the latent scale.
    scale_profile : ScaleProfile
        Observed scale profile and its computed ridge verdict.
    constraint_cost : ConstraintCost
        Paired predictive densities and their computed signed difference.
    """

    spec_hash: str
    posterior: PosteriorHandle | None
    diagnostics: FitDiagnostics
    identification: ScaleIdentification
    scale_profile: ScaleProfile
    constraint_cost: ConstraintCost

    __axioms__ = (
        dx.axiom("len(spec_hash) > 0", message="a fit summary names its spec"),
    )

    @dx.model_validator()
    def _draws_exist_exactly_when_the_method_samples(self) -> FitSummary:
        """Validate the posterior handle against the inference method.

        Returns
        -------
        FitSummary
            This summary after validation.

        Raises
        ------
        ValueError
            If a ridged profile or a point estimate has draws, or an
            identified sampling fit has none.
        """
        sampled = self.diagnostics.method is not InferenceMethod.MAP

        if self.scale_profile.ridge:
            if self.posterior is not None:
                msg = "a ridged scale profile cannot publish posterior draws"

                raise ValueError(msg)

            return self

        if sampled and self.posterior is None:
            msg = (
                f"a fit by {self.diagnostics.method.value} draws from the "
                f"posterior and reports no handle to the draws"
            )

            raise ValueError(msg)

        if not sampled and self.posterior is not None:
            msg = "a point estimate reports a handle to posterior draws"

            raise ValueError(msg)

        return self


def _moment(field: str, text: str) -> datetime:
    """Return the instant an ISO 8601 timestamp with a UTC offset names.

    Parameters
    ----------
    field : str
        Name of the value, used in the error message.
    text : str
        The timestamp.

    Raises
    ------
    ValueError
        If the text is not an ISO 8601 timestamp, or names no UTC offset.
    """
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        msg = f"{field} is not an ISO 8601 timestamp: {text!r}"

        raise ValueError(msg) from error

    if moment.tzinfo is None:
        msg = f"{field} names no UTC offset: {text!r}"

        raise ValueError(msg)

    return moment


class DynamicRangeCriterion(dx.Model, extra="forbid"):
    """Declares the pass criterion for the spread of the root log-odds.

    The criterion is registered before the distribution it is read against
    is observed, and the registration time is part of the declaration. A
    distribution passes when the middle half of the items is at least as
    spread as the minimum and at most the declared fraction of the items sit
    at or above the ceiling.

    Parameters
    ----------
    registered_at : str
        When the criterion was registered, as an ISO 8601 timestamp with a
        UTC offset.
    minimum_interquartile_range : float
        Smallest interquartile range of the log-odds over the items that
        passes.
    ceiling : float
        Log-odds at or above which an item counts as at the ceiling.
    maximum_ceiling_fraction : float
        Largest fraction of the items at the ceiling that passes.
    """

    registered_at: str
    minimum_interquartile_range: float
    ceiling: float
    maximum_ceiling_fraction: float

    __axioms__ = (
        dx.axiom(
            "minimum_interquartile_range > 0.0",
            message="a criterion a constant distribution passes is no criterion",
        ),
        dx.axiom(
            "maximum_ceiling_fraction >= 0.0",
            message="a ceiling fraction is a proportion",
        ),
        dx.axiom(
            "maximum_ceiling_fraction < 1.0",
            message="a criterion every distribution passes is no criterion",
        ),
    )

    @dx.model_validator()
    def _the_registration_is_dated_and_the_bounds_are_finite(
        self,
    ) -> DynamicRangeCriterion:
        """Validate the timestamp and the bounds.

        Returns
        -------
        DynamicRangeCriterion
            This criterion after validation.

        Raises
        ------
        ValueError
            If the registration timestamp is not ISO 8601 with a UTC offset, or
            a bound is not finite.
        """
        _moment("a criterion's registration time", self.registered_at)
        ensure_finite(
            "a criterion's minimum interquartile range",
            self.minimum_interquartile_range,
        )
        ensure_finite("a criterion's ceiling", self.ceiling)

        return self

    def registration_time(self) -> datetime:
        """Return the instant the criterion was registered."""
        return _moment("a criterion's registration time", self.registered_at)


class DynamicRangeSpec(dx.Model, extra="forbid"):
    """Declares the dynamic-range gate a fit's launch applies to one readout.

    Parameters
    ----------
    readout : str
        Predictor column of a response link term whose root mass, as
        log-odds against its complement, is read against the criterion.
    criterion : DynamicRangeCriterion
        The criterion registered for the readout's distribution.
    """

    readout: str
    criterion: DynamicRangeCriterion

    __axioms__ = (
        dx.axiom("len(readout) > 0", message="a dynamic-range gate names a readout"),
    )


class DynamicRangeReport(dx.Model, extra="forbid"):
    """Declares the spread of the root log-odds over a corpus against a criterion.

    Parameters
    ----------
    criterion : DynamicRangeCriterion
        The criterion the distribution is read against.
    inventory : str
        Name of the declared inventory the scores were computed under.
    observed_at : str
        When the distribution was observed, as an ISO 8601 timestamp with a
        UTC offset.
    items : int
        How many items the distribution is over.
    minimum : float
        Smallest log-odds over the items.
    lower_quartile : float
        Log-odds a quarter of the items lie at or below.
    median : float
        Log-odds half of the items lie at or below.
    upper_quartile : float
        Log-odds three quarters of the items lie at or below.
    maximum : float
        Largest log-odds over the items.
    ceiling_fraction : float
        Fraction of the items at or above the criterion's ceiling.
    """

    criterion: DynamicRangeCriterion
    inventory: str
    observed_at: str
    items: int
    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float
    ceiling_fraction: float

    __axioms__ = (
        dx.axiom("len(inventory) > 0", message="a report names its inventory"),
        dx.axiom("items >= 1", message="a distribution is over at least one item"),
        dx.axiom(
            "minimum <= lower_quartile",
            message="the lower quartile is at least the minimum",
        ),
        dx.axiom(
            "lower_quartile <= median",
            message="the median is at least the lower quartile",
        ),
        dx.axiom(
            "median <= upper_quartile",
            message="the upper quartile is at least the median",
        ),
        dx.axiom(
            "upper_quartile <= maximum",
            message="the maximum is at least the upper quartile",
        ),
        dx.axiom(
            "ceiling_fraction >= 0.0",
            message="a ceiling fraction is a proportion",
        ),
        dx.axiom(
            "ceiling_fraction <= 1.0",
            message="a ceiling fraction is a proportion",
        ),
    )

    @dx.computed
    def interquartile_range(self) -> float:
        """Return the spread of the middle half of the items."""
        return self.upper_quartile - self.lower_quartile

    @dx.computed
    def failures(self) -> tuple[str, ...]:
        """Return the criterion's clauses the distribution fails, if any."""
        failed: list[str] = []

        if self.interquartile_range < self.criterion.minimum_interquartile_range:
            failed.append(
                f"the interquartile range of the log-odds is "
                f"{self.interquartile_range:.4g}, below the registered minimum of "
                f"{self.criterion.minimum_interquartile_range:.4g}"
            )

        if self.ceiling_fraction > self.criterion.maximum_ceiling_fraction:
            failed.append(
                f"{self.ceiling_fraction:.4g} of the items lie at or above a "
                f"log-odds of {self.criterion.ceiling:.4g}, more than the "
                f"registered maximum of {self.criterion.maximum_ceiling_fraction:.4g}"
            )

        return tuple(failed)

    @dx.computed
    def passed(self) -> bool:
        """Return whether the distribution meets every clause of the criterion."""
        return not self.failures

    @dx.model_validator()
    def _the_numbers_are_finite_and_the_criterion_came_first(
        self,
    ) -> DynamicRangeReport:
        """Validate the summary and the order of registration and observation.

        Returns
        -------
        DynamicRangeReport
            This report after validation.

        Raises
        ------
        ValueError
            If a number is not finite, the observation timestamp is not ISO 8601
            with a UTC offset, or the distribution was observed before the
            criterion was registered.
        """
        ensure_finite("a report's minimum", self.minimum)
        ensure_finite("a report's maximum", self.maximum)
        observed = _moment("a report's observation time", self.observed_at)
        registered = self.criterion.registration_time()

        if observed < registered:
            msg = (
                f"the distribution was observed at {self.observed_at} and the "
                f"criterion registered at {self.criterion.registered_at}, so the "
                "criterion was registered after the distribution was seen"
            )

            raise ValueError(msg)

        return self


GUIDE_METHODS: frozenset[InferenceMethod] = frozenset(
    (InferenceMethod.STRUCTURED, InferenceMethod.MEAN_FIELD)
)
"""The guides a variational judgment fit may declare."""


def _ensure_learning_rate(value: float) -> float:
    """Return a guide learning rate that is finite and positive.

    Parameters
    ----------
    value : float
        Declared Adam step size of a variational guide.

    Returns
    -------
    float
        The declared step size.

    Raises
    ------
    ValueError
        If the step size is not finite or not positive.
    """
    rate = ensure_finite("a guide's learning rate", value)

    if rate <= 0.0:
        msg = f"a guide's learning rate is positive, not {value!r}"
        raise ValueError(msg)

    return rate


class FitInference(dx.TaggedUnion, discriminator="inference", extra="forbid"):
    """The inference a judgment fit draws its posterior by."""

    @dx.computed
    def method(self) -> InferenceMethod:
        """Return the inference method a fit under this declaration records."""
        return InferenceMethod.NUTS


class NutsInference(FitInference, extra="forbid"):
    """Declares the multi-chain NUTS plan of a judgment fit.

    The plan's seed is not declared here. A fit run as an experiment point takes
    it from the run seed the point derives from its experiment's root.

    Parameters
    ----------
    inference : Literal["nuts"]
        Tagged-union discriminator.
    warmup : int
        Discarded adaptation iterations per chain.
    retained : int
        Retained draws per chain.
    chains : int
        Independent chains.
    """

    inference: Literal["nuts"] = "nuts"
    warmup: int
    retained: int
    chains: int

    __axioms__ = (
        dx.axiom("warmup >= 1", message="a NUTS plan runs at least one warmup step"),
        dx.axiom(
            "retained >= 4",
            message="a NUTS plan retains at least four draws per chain",
        ),
        dx.axiom("chains >= 2", message="a NUTS plan runs at least two chains"),
    )


class VariationalInference(FitInference, extra="forbid"):
    """Declares stochastic variational inference under a declared guide.

    The guide's evidence lower bound is maximized by Adam at the declared
    learning rate for the declared number of steps, and the declared number
    of draws is then taken from the fitted guide. The seed is taken from the
    point's run seed, as for NUTS.

    Parameters
    ----------
    inference : Literal["svi"]
        Tagged-union discriminator.
    guide : InferenceMethod
        The structured guide, which conditions each grouped site on the
        ungrouped sites drawn before it, or the mean-field guide.
    iterations : int
        Optimization steps the guide is fitted for.
    draws : int
        Draws taken from the fitted guide.
    learning_rate : float
        Adam step size every optimization step takes.
    """

    inference: Literal["svi"] = "svi"
    guide: InferenceMethod
    iterations: int
    draws: int
    learning_rate: float

    __axioms__ = (
        dx.axiom("iterations >= 1", message="a guide is fit for at least one step"),
        dx.axiom("draws >= 1", message="a fitted guide is drawn from at least once"),
    )

    @dx.validates("learning_rate")
    @classmethod
    def _learning_rate_is_positive(cls, value: float) -> float:
        """Return a finite positive guide learning rate."""
        return _ensure_learning_rate(value)

    @dx.validates("guide")
    @classmethod
    def _the_guide_is_variational(cls, guide: InferenceMethod) -> InferenceMethod:
        """Return a guide a variational fit builds."""
        if guide not in GUIDE_METHODS:
            msg = (
                f"a variational fit declares one of the guides "
                f"{sorted(method.value for method in GUIDE_METHODS)!r}, not "
                f"{guide.value!r}"
            )
            raise ValueError(msg)

        return guide

    @dx.computed
    @override
    def method(self) -> InferenceMethod:
        """Return the declared guide."""
        return self.guide


class MapInference(FitInference, extra="forbid"):
    """Declare support-preserving optimization of a posterior mode.

    Parameters
    ----------
    inference : Literal["map"]
        Tagged-union discriminator.
    optimizer : OptimizerSpec
        Installed optimizer and all of its settings.
    steps : int
        Number of parameter updates.
    initial_scale : float
        Standard deviation of the initial unconstrained coordinates.
    """

    inference: Literal["map"] = "map"
    optimizer: OptimizerSpec
    steps: int
    initial_scale: float

    @dx.validates("steps")
    @classmethod
    def _steps_are_positive(cls, value: int) -> int:
        """Return a positive update count."""
        return ensure_positive("a MAP fit's step count", value)

    @dx.validates("initial_scale")
    @classmethod
    def _initial_scale_is_positive(cls, value: float) -> float:
        """Return a finite positive unconstrained initialization scale."""
        scale = ensure_finite("a MAP fit's initial scale", value)

        if scale <= 0.0:
            msg = f"a MAP fit's initial scale is not positive: {value!r}"
            raise ValueError(msg)

        return scale

    @dx.computed
    @override
    def method(self) -> InferenceMethod:
        """Return maximum a posteriori inference."""
        return InferenceMethod.MAP


class ElicitationMethod(enum.StrEnum):
    """Selects how the judgments a fit conditions on were elicited."""

    #: Ratings recorded from human participants.
    PARTICIPANTS = "participants"

    #: A language model's answer to a prompt asking for the judgment.
    MODEL_PROMPTING = "model_prompting"

    #: A judgment read off the probability a language model assigns the item.
    MODEL_PROBABILITY = "model_probability"


class Elicitation(dx.Model, extra="forbid"):
    """Declares the versioned protocol the judgments of a fit were elicited under.

    Parameters
    ----------
    method : ElicitationMethod
        How the judgments were elicited.
    version : str
        Version of the elicitation protocol, so two protocols under one
        method are told apart.
    """

    method: ElicitationMethod
    version: str

    @dx.validates("version")
    @classmethod
    def _version_is_named(cls, version: str) -> str:
        """Return a nonempty protocol version."""
        return ensure_name("an elicitation protocol version", version)


class JudgmentFitCalibration(dx.Model, extra="forbid"):
    """Declares simulation-based calibration of a judgment fit.

    Each replicate draws parameters from the prior, samples responses at them,
    and fits the sampled responses by NUTS and by the structured and
    mean-field guides.

    Parameters
    ----------
    replicates : int
        Prior-predictive replicates.
    guide_iterations : int
        Sweeps each variational guide is fit for.
    guide_learning_rate : float
        Adam step size each variational guide is fit at.
    """

    replicates: int
    guide_iterations: int
    guide_learning_rate: float

    __axioms__ = (
        dx.axiom("replicates >= 1", message="calibration runs at least one replicate"),
        dx.axiom(
            "guide_iterations >= 1",
            message="a guide is fit for at least one sweep",
        ),
    )

    @dx.validates("guide_learning_rate")
    @classmethod
    def _guide_learning_rate_is_positive(cls, value: float) -> float:
        """Return a finite positive guide learning rate."""
        return _ensure_learning_rate(value)


class JudgmentFitSpec(dx.Model, extra="forbid"):
    """Declares one judgment fit, the configuration a judgment experiment sweeps.

    A fit's response model compiles to a program over the installed linking
    families. NUTS and variational inference draw every site and therefore use
    the conditional likelihood. MAP may instead optimize the population
    posterior of a truncated-marginal objective.

    Parameters
    ----------
    dataset : str
        Identifier of the judgment dataset the fit conditions on.
    response : ResponseDeclaration
        Response model fit to the judgments.
    inference : FitInference
        Inference the posterior is drawn by.
    elicitation : Elicitation
        Protocol the judgments were elicited under.
    calibration : JudgmentFitCalibration | None
        Simulation-based calibration of the fit, or None for a fit that is not
        calibrated.
    comparison : ComparisonSpec | None
        Held-out and fitted-model comparison, or None for an in-sample fit.
    dynamic_range : DynamicRangeSpec | None
        Dynamic-range gate applied to a declared readout before fitting, or
        None when no launch gate is declared.
    ablation : tuple[TableAblationSpec, ...]
        Table-ablation rungs, empty when the fit runs no table ablation.
    interannotator : InterannotatorSpec | None
        Interannotator correlation ceilings reported beside the fit, or None
        when the fit reports none.
    """

    dataset: str
    response: ResponseDeclaration
    inference: FitInference
    elicitation: Elicitation
    calibration: JudgmentFitCalibration | None
    comparison: ComparisonSpec | None
    dynamic_range: DynamicRangeSpec | None
    ablation: tuple[TableAblationSpec, ...]
    interannotator: InterannotatorSpec | None

    @dx.validates("dataset")
    @classmethod
    def _dataset_is_named(cls, dataset: str) -> str:
        """Return a nonblank dataset identifier."""
        if not dataset.strip():
            msg = f"a judgment fit names its dataset: {dataset!r}"
            raise ValueError(msg)

        return dataset

    @dx.model_validator()
    def _the_response_compiles(self) -> JudgmentFitSpec:
        """Validate that the response compiles to a program over installed linkings.

        Returns
        -------
        JudgmentFitSpec
            This fit after validation.

        Raises
        ------
        ValueError
            If the response family compiles to no program, or a link term
            declares a linking family that is not installed.
        """
        family = self.response.family

        if not family.programmed:
            msg = (
                f"a judgment fit draws a posterior over a compiled response "
                f"program, and the {family.family!r} family compiles to none"
            )
            raise ValueError(msg)

        available = frozenset(installed_linkings().available())
        missing = _distinct(
            tuple(
                term.linking.family
                for root in self.response.terms
                for term, _ in _link_terms(root)
                if term.linking.family not in available
            )
        )

        if missing:
            msg = (
                f"a judgment fit compiles its response over the installed linking "
                f"families {sorted(available)!r}, and it declares {list(missing)!r}"
            )
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _the_inference_supports_the_likelihood(self) -> JudgmentFitSpec:
        """Validate the likelihood objective against posterior inference.

        Returns
        -------
        JudgmentFitSpec
            This fit after validation.

        Raises
        ------
        ValueError
            If a sampled fit declares the truncated-marginal objective.
        """
        match self.response:
            case ResponseSpec(objective=TruncatedMarginalLikelihood(group=group)):
                if self.inference.method is InferenceMethod.MAP:
                    return self

                msg = (
                    f"a {self.inference.method.value} fit draws every site of the "
                    f"program, the group-local block over {group!r} among them, "
                    f"so the truncated-marginal objective leaves its posterior "
                    f"unchanged; declare the conditional objective"
                )
                raise ValueError(msg)
            case _:
                return self

    @dx.model_validator()
    def _comparison_describes_this_response(self) -> JudgmentFitSpec:
        """Validate the held-out comparison against the response it contains."""
        comparison = self.comparison

        if comparison is None:
            return self

        consumed = frozenset(self.response.readouts)
        declared = frozenset(comparison.readout_units)
        missing = tuple(sorted(consumed - declared))
        extra = tuple(sorted(declared - consumed))

        if missing or extra:
            msg = (
                "comparison readout_units name exactly every readout consumed by "
                f"the response; missing {missing!r}; extra {extra!r}"
            )
            raise ValueError(msg)

        if comparison.held_out_unit == "item":
            return self

        terms = tuple(
            nested for root in self.response.terms for nested in _term_tree(root)
        )
        intercept = next(
            (
                term
                for term in terms
                if isinstance(term, RandomIntercept) and term.group == "item"
            ),
            None,
        )
        link = next(
            (
                term
                for term in terms
                if isinstance(term, LinkTerm)
                and all(
                    comparison.readout_units[readout] == "item"
                    for readout in term.readouts
                )
            ),
            None,
        )

        if intercept is not None and link is not None:
            msg = (
                f"{comparison.held_out_unit!r} holdout cannot compare the item "
                f"random intercept {intercept!r} with the item-grain link {link!r}"
            )
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _table_ablation_is_one_comparable_ladder(self) -> JudgmentFitSpec:
        """Validate that declared table rungs can run as one held-out ladder."""
        if not self.ablation:
            return self

        if isinstance(self.response, BoundedResponseSpec):
            msg = "a table ablation scores ordinal levels, not bounded responses"
            raise TypeError(msg)

        if self.comparison is None:
            msg = "a table ablation declares the held-out comparison it reports"
            raise ValueError(msg)

        first = self.ablation[0]

        for rung in self.ablation[1:]:
            if (
                rung.rows != first.rows
                or rung.columns != first.columns
                or rung.width != first.width
                or rung.schedule != first.schedule
            ):
                msg = (
                    "every table-ablation rung declares the same rows, columns, "
                    "width and fit schedule"
                )
                raise ValueError(msg)

        lifts = tuple(rung.lift.lift for rung in self.ablation)

        if len(frozenset(lifts)) != len(lifts):
            msg = f"a table ablation repeats a lift: {lifts!r}"
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _calibration_compares_against_nuts(self) -> JudgmentFitSpec:
        """Validate that a calibrated fit declares its NUTS plan.

        Returns
        -------
        JudgmentFitSpec
            This fit after validation.

        Raises
        ------
        ValueError
            If calibration is declared beside variational inference.
        """
        if self.calibration is None:
            return self

        if isinstance(self.response, BoundedResponseSpec):
            msg = "calibration currently scores ordinal response levels"
            raise TypeError(msg)

        if self.inference.method is not InferenceMethod.NUTS:
            msg = (
                "calibration fits every replicate by NUTS under the fit's NUTS "
                f"plan, and this fit declares {self.inference.method.value!r}"
            )
            raise ValueError(msg)

        return self


def _digest(field: str, value: str) -> str:
    """Return a lowercase hexadecimal SHA-256 digest."""
    if len(value) != 64 or value.strip("0123456789abcdef"):
        msg = f"{field} is not a lowercase SHA-256 digest: {value!r}"
        raise ValueError(msg)

    return value


class DesignProvenance(dx.Model, extra="forbid"):
    """Records what a registered judgment design was read from.

    A design scored by a language model records its checkpoint with the
    precision the model was loaded in. A drawn design records its seed with
    the counts it was drawn at and a digest of the population values its
    responses were drawn from.

    Parameters
    ----------
    tag : str
        The tag the design is registered under.
    checkpoint : str | None
        The ``repository@commit`` identity of the language model whose scores
        the design carries, or None for a design that carries none.
    checkpoint_dtype : str | None
        The precision the language model's parameters were loaded in, such as
        ``float32``, or None for a design that carries no scores.
    unigram_digest : str | None
        SHA-256 digest of the unigram counts the design was scored against, or
        None for a design scored against none.
    corpus_digests : dict[str, str]
        SHA-256 digest of each installed corpus file the design was read from,
        by file name.
    seed : int | None
        Seed a synthetic design is drawn from, or None for a design that is
        not drawn.
    participants : int | None
        Number of participants a synthetic design is drawn over, or None.
    items : int | None
        Number of items a synthetic design is drawn over, or None.
    levels : int | None
        Number of ordered response levels a synthetic design is drawn over,
        or None.
    population_digest : str | None
        SHA-256 digest of the population values a synthetic design's
        responses are drawn at, or None.
    producer_run : str | None
        Run artifact that produced a predicted design, or None.
    producer_point : str | None
        Point within ``producer_run`` that produced the predictions, or None.
    """

    tag: str
    checkpoint: str | None
    checkpoint_dtype: str | None
    unigram_digest: str | None
    corpus_digests: dict[str, str]
    seed: int | None
    participants: int | None
    items: int | None
    levels: int | None
    population_digest: str | None
    producer_run: str | None
    producer_point: str | None

    @dx.validates("tag")
    @classmethod
    def _tag_is_named(cls, tag: str) -> str:
        """Return a nonempty design tag."""
        return ensure_name("a design tag", tag)

    @dx.validates("checkpoint")
    @classmethod
    def _checkpoint_is_named(cls, checkpoint: str | None) -> str | None:
        """Return a nonempty checkpoint identity, or None."""
        return None if checkpoint is None else ensure_name("a checkpoint", checkpoint)

    @dx.validates("checkpoint_dtype")
    @classmethod
    def _dtype_is_named(cls, dtype: str | None) -> str | None:
        """Return a nonempty precision, or None."""
        return None if dtype is None else ensure_name("a checkpoint precision", dtype)

    @dx.validates("unigram_digest")
    @classmethod
    def _unigrams_are_digested(cls, digest: str | None) -> str | None:
        """Return a SHA-256 digest of the unigram counts, or None."""
        return None if digest is None else _digest("a unigram digest", digest)

    @dx.validates("corpus_digests")
    @classmethod
    def _corpus_is_digested(cls, digests: dict[str, str]) -> dict[str, str]:
        """Return a name-ordered mapping of corpus files to SHA-256 digests."""
        return {
            ensure_name("a corpus file", name): _digest(f"the digest of {name}", value)
            for name, value in sorted(digests.items())
        }

    @dx.validates("participants", "items", "levels")
    @classmethod
    def _count_is_positive(cls, count: int | None) -> int | None:
        """Return a positive count, or None."""
        return None if count is None else ensure_positive("a design count", count)

    @dx.validates("population_digest")
    @classmethod
    def _population_is_digested(cls, digest: str | None) -> str | None:
        """Return a SHA-256 digest of the population values, or None."""
        return None if digest is None else _digest("a population digest", digest)

    @dx.validates("producer_run", "producer_point")
    @classmethod
    def _producer_is_named(cls, value: str | None) -> str | None:
        """Return a nonempty producer identity, or None."""
        return None if value is None else ensure_name("a design producer", value)

    @dx.model_validator()
    def _records_are_whole(self) -> DesignProvenance:
        """Validate that a checkpoint and a draw are each recorded whole.

        Returns
        -------
        DesignProvenance
            This record after validation.

        Raises
        ------
        ValueError
            If a checkpoint is recorded without its precision or the reverse,
            or a draw's seed, counts and population digest are not all
            recorded or all absent.
        """
        if (self.checkpoint is None) != (self.checkpoint_dtype is None):
            msg = "a design's checkpoint and its precision are recorded together"
            raise ValueError(msg)

        drawn = (
            self.seed,
            self.participants,
            self.items,
            self.levels,
            self.population_digest,
        )

        if any(value is None for value in drawn) and any(
            value is not None for value in drawn
        ):
            msg = (
                "a drawn design's seed, counts and population digest are "
                "recorded together"
            )
            raise ValueError(msg)

        if (self.producer_run is None) != (self.producer_point is None):
            msg = "a predicted design's producer run and point are recorded together"
            raise ValueError(msg)

        return self


class ReadDesign(NamedTuple):
    """A judgment design and the provenance of the bytes it was decoded from.

    Attributes
    ----------
    design : JudgmentDesign
        The decoded design.
    provenance : DesignProvenance
        What the design was read from, digested over the same bytes.
    """

    design: JudgmentDesign
    provenance: DesignProvenance


class ReadBoundedDesign(NamedTuple):
    """A bounded design and the provenance of the bytes it was decoded from."""

    design: BoundedDesign
    provenance: DesignProvenance


type ReadCorpusDesign = ReadDesign | ReadBoundedDesign
"""A corpus-backed ordinal or bounded judgment design with provenance."""


class FitArtifact(dx.Model, extra="forbid"):
    """Identify the sole persisted artifact of one completed fit.

    Parameters
    ----------
    name : Literal["posterior.nc", "point-estimate.safetensors"]
        Current artifact representation selected by the inference method.
    sha256 : str
        Lowercase SHA-256 digest of the stored artifact bytes.
    """

    name: Literal["posterior.nc", "point-estimate.safetensors"]
    sha256: str

    @dx.validates("sha256")
    @classmethod
    def _digest_is_valid(cls, value: str) -> str:
        """Return a lowercase SHA-256 artifact digest."""
        return _digest("a fit artifact digest", value)


class JudgmentFitManifest(dx.Model, extra="forbid"):
    """Records the fit, design and run seed of one judgment-fit experiment point.

    Parameters
    ----------
    fit : JudgmentFitSpec
        The resolved fit the point ran.
    design : DesignProvenance
        What the fit's design was read from.
    seed_derivation : SeedDerivationRecord
        The experiment root, rule and point the run seed was derived from.
    artifact : FitArtifact
        Name and content digest of the fit's sole persisted result.
    package_versions : dict[str, str]
        Installed version of each recorded distribution.
    git_revision : str
        Source revision the fit ran at.
    """

    fit: JudgmentFitSpec
    design: DesignProvenance
    seed_derivation: SeedDerivationRecord
    artifact: FitArtifact
    package_versions: dict[str, str]
    git_revision: str

    @dx.validates("package_versions")
    @classmethod
    def _versions_are_named(cls, versions: dict[str, str]) -> dict[str, str]:
        """Return a name-ordered mapping with nonempty names and versions."""
        if not versions:
            msg = "a manifest's package versions cannot be empty"
            raise ValueError(msg)

        return {
            ensure_name("package name", name): ensure_name("package version", value)
            for name, value in sorted(versions.items())
        }

    @dx.validates("git_revision")
    @classmethod
    def _revision_is_named(cls, revision: str) -> str:
        """Return a nonempty source revision."""
        return ensure_name("source revision", revision)

    @dx.model_validator()
    def _artifact_matches_inference(self) -> JudgmentFitManifest:
        """Validate the sole artifact representation against the fit method."""
        expected = (
            "point-estimate.safetensors"
            if self.fit.inference.method is InferenceMethod.MAP
            else "posterior.nc"
        )

        if self.artifact.name != expected:
            msg = (
                f"a fit by {self.fit.inference.method.value} persists {expected!r}, "
                f"not {self.artifact.name!r}"
            )
            raise ValueError(msg)

        return self


def decode_fit_manifest(data: bytes, /) -> JudgmentFitManifest:
    """Decode a judgment-fit manifest's JSON bytes, refusing any that do not validate.

    Parameters
    ----------
    data : bytes
        The bytes the manifest was written as.

    Returns
    -------
    JudgmentFitManifest
        The validated manifest.

    Raises
    ------
    ValueError
        If the bytes are not JSON or do not validate as a judgment-fit manifest.
    """
    try:
        return JudgmentFitManifest.model_validate_json(data)

    except dx.ValidationError as refused:
        msg = f"the bytes do not validate as a judgment-fit manifest: {refused}"
        raise ValueError(msg) from refused


def encode_fit_manifest(manifest: JudgmentFitManifest, /) -> bytes:
    """Encode one validated judgment-fit manifest for persistent storage."""
    return manifest.model_dump_json().encode()

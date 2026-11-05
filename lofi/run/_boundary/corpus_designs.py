"""Boundary declarations for judgment designs read beneath one data root.

A design document names where its judgments come from and how they are read.
A fragment design decodes a corpus release through the codec registered under
its declared name, or reads a Layers export, and lowers what the release
states of each judged item into an item record: the release's own feature
names become the record's attributes, and its text the record's one declared
element. A bead design constructs its items from a judgment-item construction
document and reads bead annotation records. Either way the design's factors
are a mapping from factor name to a bead expression over the item records, so
no corpus column is named outside its codec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, Self, cast

import didactic.api as dx
from bead.dsl.errors import DSLError
from bead.dsl.evaluator import DSLEvaluator
from didactic.settings import EnvSource, Settings
from lairs.data import JudgmentStudy
from lairs.integrations.registry import get_codec
from lairs.records import annotation as annotation_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import resource as resource_records

from lofi.data.corpora.layers import (
    LayersCorpusReader,
    LayersExportCodec,
    RecordSkippedError,
    relation_layer_pair,
)
from lofi.judgment import AttributedItem, DesignError, related_item_key
from lofi.spec.artifacts import ArtifactCacheSpec, PretrainedSourceSpec, source_artifact

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bead.items.item import MetadataValue
    from lairs.integrations.codecs import CorpusFragment

    from lofi._artifacts import PinnedCheckpoint
    from lofi.data import Example, Supervision

__all__ = [
    "BeadDesignSpec",
    "BoundedResponsesSpec",
    "CausalLmReadoutsSpec",
    "CodecFragmentSpec",
    "CorpusSettings",
    "DesignSpec",
    "FragmentDesignSpec",
    "FragmentSourceSpec",
    "LayersExportSpec",
    "LmReadoutsSpec",
    "LoweredFragment",
    "OrdinalResponsesSpec",
    "ReadoutSourceSpec",
    "RelationResponsesSpec",
    "ResponseReadingSpec",
    "RootMassSpec",
    "declared_data_root",
    "decode_design",
    "decode_fragment",
    "lower_fragment",
]

_EXPERIMENT_NSID = "pub.layers.judgment.experimentDef"
_ABSENT = object()

type _Record = (
    annotation_records.AnnotationLayer
    | expression_records.Expression
    | resource_records.Filling
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet
)


class DeclaredCodec(Protocol):
    """A registered corpus codec configured by a release declaration document."""

    def decode(self, src: str | bytes, /) -> CorpusFragment:
        """Return the Layers records one release file holds."""
        ...


class _DeclaredCodecType(Protocol):
    """The class of a codec a release declaration document configures."""

    def from_declaration(self, declaration: bytes, /) -> DeclaredCodec:
        """Return the codec the declaration configures."""
        ...


def decode_design(document: bytes, /) -> DesignSpec:
    """Validate one design document at the corpus boundary."""
    return DesignSpec.model_validate_json(document)


def decode_fragment(
    source: FragmentSourceSpec, files: Mapping[str, bytes], /
) -> CorpusFragment:
    """Decode the declared release into Layers records at the corpus boundary.

    Parameters
    ----------
    source : FragmentSourceSpec
        The declared release.
    files : Mapping[str, bytes]
        The bytes of every file the source declares, by declared path.

    Raises
    ------
    UnknownCodecError
        If no codec is registered under the declared name.
    """
    match source:
        case CodecFragmentSpec(codec=name, declaration=declaration, responses=read):
            codec = cast("_DeclaredCodecType", get_codec(name))
            return codec.from_declaration(files[declaration]).decode(files[read])
        case LayersExportSpec(path=path):
            return LayersExportCodec().decode(files[path])
    msg = f"unknown fragment source {source!r}"
    raise TypeError(msg)


def _record_of(nsid: str, value: str) -> _Record:
    """Return the Layers record a supported corpus fragment serializes."""
    match nsid:
        case "pub.layers.annotation.annotationLayer":
            return annotation_records.AnnotationLayer.model_validate_json(value)
        case "pub.layers.expression.expression":
            return expression_records.Expression.model_validate_json(value)
        case "pub.layers.resource.filling":
            return resource_records.Filling.model_validate_json(value)
        case "pub.layers.judgment.experimentDef":
            return judgment_records.ExperimentDef.model_validate_json(value)
        case "pub.layers.judgment.judgmentSet":
            return judgment_records.JudgmentSet.model_validate_json(value)

    msg = f"a corpus design fragment holds no {nsid!r} records"
    raise ValueError(msg)


def _features(judgment: judgment_records.Judgment, /) -> dict[str, MetadataValue]:
    """Return what one judgment states, by feature name."""
    if judgment.features is None:
        return {}
    return {entry.key: entry.value for entry in judgment.features.entries}


def _kept(record: _Record, keep: str | None, evaluator: DSLEvaluator, /) -> _Record:
    """Return a judgment set without the judgments the filter sets aside.

    Raises
    ------
    DesignError
        If the filter cannot be read from a judgment or yields no truth value.
    """
    if keep is None or not isinstance(record, judgment_records.JudgmentSet):
        return record
    kept: list[judgment_records.Judgment] = []
    for judgment in record.judgments:
        try:
            admitted = evaluator.evaluate(keep, {"judgment": _features(judgment)})
        except DSLError as refused:
            msg = f"the judgment filter {keep!r} cannot be read: {refused}"
            raise DesignError(msg) from refused
        match admitted:
            case bool():
                if admitted:
                    kept.append(judgment)
            case _:
                msg = (
                    f"the judgment filter {keep!r} yields {admitted!r}, no truth value"
                )
                raise DesignError(msg)
    return record.with_(judgments=tuple(kept))


def _stated(
    record: _Record, stated: dict[str, list[dict[str, MetadataValue]]], /
) -> None:
    """Record what each judgment of a judgment set states of its target."""
    if not isinstance(record, judgment_records.JudgmentSet):
        return
    for judgment in record.judgments:
        target = judgment.item.recordRef
        if target is not None:
            stated.setdefault(target, []).append(_features(judgment))


def _agreed(stated: list[dict[str, MetadataValue]], /) -> dict[str, MetadataValue]:
    """Return the features every judgment of one target states alike."""
    first, *rest = stated
    return {
        name: value
        for name, value in first.items()
        if all(held.get(name, _ABSENT) == value for held in rest)
    }


def _related_key(record: _Record, local_id: str) -> str | None:
    """Return the related-item key a relation-pair layer stands for."""
    if not isinstance(record, annotation_records.AnnotationLayer):
        return None
    try:
        pair = relation_layer_pair(record)
    except RecordSkippedError as refused:
        msg = f"the relation layer {local_id!r} names no single pair: {refused}"
        raise DesignError(msg) from refused
    if pair is None:
        return None
    return related_item_key((pair.premise, pair.hypothesis))


class LoweredFragment(NamedTuple):
    """A fragment's examples and the item record of every judged target."""

    examples: tuple[Example[Supervision[str]], ...]
    items: tuple[AttributedItem, ...]


def lower_fragment(
    fragment: CorpusFragment, element: str | None, keep: str | None, /
) -> LoweredFragment:
    """Lower one Layers fragment to interior examples and item records.

    A record's attributes are the features every judgment of the item, or of
    the related-item key of a relation pair, states alike; a feature its
    judgments state differently, such as a participant's own, describes the
    judgments rather than the item and is no attribute. Given an element name,
    each judged item's text is rendered under it. Given a judgment filter, a
    bead expression over each judgment's features bound to ``judgment``, the
    judgments it does not hold of are set aside before anything is read.

    Raises
    ------
    DesignError
        If the filter cannot be read from a judgment, or a relation-pair layer
        names no single pair.
    """
    experiment = next(
        record.local_id
        for record in fragment.records
        if record.nsid == _EXPERIMENT_NSID
    )
    study = JudgmentStudy.new(experiment)
    stated: dict[str, list[dict[str, MetadataValue]]] = {}
    keys: dict[str, str] = {}
    evaluator = DSLEvaluator()

    for record in fragment.records:
        model = _kept(_record_of(record.nsid, record.value_json), keep, evaluator)
        study.add_record(record.local_id, model)
        _stated(model, stated)
        key = _related_key(model, record.local_id)
        if key is not None:
            keys[record.local_id] = key

    reader = LayersCorpusReader(study)
    examples = tuple(
        example for split in reader.splits() for example in reader.examples(split)
    )
    attributes = {
        keys.get(target, target): _agreed(held) for target, held in stated.items()
    }
    texts: dict[str, str] = {
        str(example.item.item_id): example.item.text
        for example in examples
        if element is not None and example.item.text is not None
    }
    items = tuple(
        AttributedItem(
            key,
            attributes.get(key, {}),
            {} if element is None or key not in texts else {element: texts[key]},
        )
        for key in dict.fromkeys((*attributes, *texts))
    )
    return LoweredFragment(examples, items)


def _named(value: str, what: str) -> str:
    """Return a nonblank declared string."""
    if not value.strip():
        msg = f"{what} cannot be blank"
        raise ValueError(msg)
    return value


class CorpusSettings(Settings, extra="forbid"):
    """Locate corpus declarations and releases under one data root."""

    data_root: str

    __sources__ = (EnvSource(prefix="LOFI_"),)

    @dx.validates("data_root")
    @classmethod
    def _root_is_named(cls, value: str) -> str:
        return _named(value, "the data root")


def declared_data_root() -> str:
    """Return the data root the environment declares.

    Returns
    -------
    str
        The nonblank value of ``LOFI_DATA_ROOT``.

    Raises
    ------
    ValueError
        If the variable is unset or blank.
    """
    try:
        return CorpusSettings.load().data_root
    except dx.ValidationError as error:
        raise ValueError(str(error)) from error


def _factors_are_named(value: dict[str, str], /) -> dict[str, str]:
    """Return named factors, each read by a named expression."""
    return {
        _named(name, "a factor"): _named(expression, f"factor {name!r}'s expression")
        for name, expression in value.items()
    }


class FragmentSourceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where a design's Layers records are decoded from."""


class CodecFragmentSpec(FragmentSourceSpec):
    """Decode a release through a registered corpus codec.

    Parameters
    ----------
    codec : str
        Name the codec is registered under in the ``lairs.codecs`` group.
    declaration : str
        The release declaration document configuring the codec.
    responses : str
        The released file the codec decodes.
    """

    kind: Literal["codec"] = "codec"
    codec: str
    declaration: str
    responses: str

    @dx.validates("codec", "declaration", "responses")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return _named(value, "a codec fragment field")


class LayersExportSpec(FragmentSourceSpec):
    """Read a Layers export file.

    Parameters
    ----------
    path : str
        The export file, relative to the data root.
    """

    kind: Literal["layers"] = "layers"
    path: str

    @dx.validates("path")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return _named(value, "a Layers export path")


class ResponseReadingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare which responses of a fragment a design reads, and how."""


class OrdinalResponsesSpec(ResponseReadingSpec):
    """Read ordinal ratings of single items, on the scale the release declares."""

    kind: Literal["ordinal"] = "ordinal"


class RelationResponsesSpec(ResponseReadingSpec):
    """Read relations asserted between related items.

    Parameters
    ----------
    inventory : tuple[str, ...]
        The relation names in response-level order.
    """

    kind: Literal["relation"] = "relation"
    inventory: tuple[str, ...]

    @dx.validates("inventory")
    @classmethod
    def _inventory_is_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(_named(value, "a relation") for value in values)
        if len(checked) < 2:
            msg = "a relation inventory has at least two relations"
            raise ValueError(msg)
        if len(checked) != len(set(checked)):
            msg = f"a relation inventory repeats a relation: {checked!r}"
            raise ValueError(msg)
        return checked


class BoundedResponsesSpec(ResponseReadingSpec):
    """Read continuous responses on a finite scale."""

    kind: Literal["bounded"] = "bounded"


class ReadoutSourceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare the readout columns joined to every row of a rating design."""


class RootMassSpec(ReadoutSourceSpec):
    """Declare a published root-mass artifact and the columns its pair fills.

    Parameters
    ----------
    path : str
        The root-mass artifact, relative to the data root.
    mass : str
        Readout column holding each item's root log mass.
    complement : str
        Readout column holding each item's root log complement.
    """

    kind: Literal["root-mass"] = "root-mass"
    path: str
    mass: str
    complement: str

    @dx.validates("path", "mass", "complement")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return _named(value, "a root-mass field")

    @dx.model_validator()
    def _columns_differ(self) -> Self:
        if self.mass == self.complement:
            msg = f"the root mass and its complement both fill {self.mass!r}"
            raise ValueError(msg)
        return self


class LmReadoutsSpec(dx.Model, extra="forbid"):
    """Declare the readout column each language-model measure fills.

    Parameters
    ----------
    log_probability : str
        Column holding each item's log probability under the model.
    unigram_log_probability : str
        Column holding each item's unigram log probability.
    length : str
        Column holding each item's length in model tokens.
    """

    log_probability: str
    unigram_log_probability: str
    length: str

    @dx.validates("log_probability", "unigram_log_probability", "length")
    @classmethod
    def _column_is_named(cls, value: str) -> str:
        return _named(value, "a language-model readout column")


class CausalLmReadoutsSpec(ReadoutSourceSpec):
    """Score every rated item's text by a pinned causal language model.

    Parameters
    ----------
    element : str
        The element each lowered item renders its text under, which the model
        scores.
    checkpoint : PretrainedSourceSpec
        The hub repository and commit the causal model is fetched from.
    artifacts : ArtifactCacheSpec
        The cache the causal model is fetched into.
    unigrams : str
        The unigram count table, relative to the data root.
    columns : LmReadoutsSpec
        The readout column each measure fills.
    """

    kind: Literal["causal-lm"] = "causal-lm"
    element: str
    checkpoint: PretrainedSourceSpec
    artifacts: ArtifactCacheSpec
    unigrams: str
    columns: LmReadoutsSpec

    @dx.validates("element", "unigrams")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return _named(value, "a causal-LM readout field")

    @dx.model_validator()
    def _loads_a_hub_checkpoint(self) -> Self:
        """Refuse a source that is not a hub repository at one commit."""
        self.pinned()
        return self

    def pinned(self) -> PinnedCheckpoint:
        """Return the pinned release the declared source names."""
        return source_artifact(self.checkpoint).hub_checkpoint("a causal-LM readout")


class DesignSpec(dx.TaggedUnion, discriminator="source", extra="forbid"):
    """Declare one judgment design read beneath the data root."""


class FragmentDesignSpec(DesignSpec):
    """Declare a design over the Layers records one release decodes to.

    Parameters
    ----------
    fragment : FragmentSourceSpec
        Where the records are decoded from.
    responses : ResponseReadingSpec
        Which responses are read, and how.
    factors : dict[str, str]
        The bead expression each factor's level is read by, over the item
        records the release lowers to, keyed by factor name in column order.
    readouts : ReadoutSourceSpec | None
        Readout columns joined to every row of an ordinal design, or None.
    keep : str | None
        A bead expression over each judgment's features, bound to
        ``judgment``, that keeps the judgments it holds of, or None to keep
        every judgment.
    """

    source: Literal["fragment"] = "fragment"
    fragment: FragmentSourceSpec
    responses: ResponseReadingSpec
    factors: dict[str, str]
    readouts: ReadoutSourceSpec | None
    keep: str | None

    @dx.validates("factors")
    @classmethod
    def _factors_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        return _factors_are_named(value)

    @dx.validates("keep")
    @classmethod
    def _filter_is_named(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _named(value, "a judgment filter")

    @dx.model_validator()
    def _readouts_join_ratings(self) -> Self:
        if self.readouts is not None and not isinstance(
            self.responses, OrdinalResponsesSpec
        ):
            msg = "readout columns are joined to an ordinal rating design only"
            raise ValueError(msg)
        return self


class BeadDesignSpec(DesignSpec):
    """Declare a design over constructed bead items and bead annotation records.

    Parameters
    ----------
    items : str
        The judgment-item construction document the items are built from.
    judgments : str
        The bead annotation records, one JSON object per line.
    encoding : str
        The bead response encoding the records answer.
    factors : dict[str, str]
        The bead expression each factor's level is read by over the
        constructed items, keyed by factor name in column order.
    """

    source: Literal["bead"] = "bead"
    items: str
    judgments: str
    encoding: str
    factors: dict[str, str]

    @dx.validates("items", "judgments", "encoding")
    @classmethod
    def _is_named(cls, value: str) -> str:
        return _named(value, "a bead design path")

    @dx.validates("factors")
    @classmethod
    def _factors_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        return _factors_are_named(value)

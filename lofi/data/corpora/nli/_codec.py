"""Conversion from normalized NLI files to Layers relation records."""

from __future__ import annotations

import csv
import enum
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from io import StringIO
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

from lairs.data import JudgmentStudy
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import segmentation as segmentation_records

from lofi._types import JsonValue
from lofi.data.registration import registers_as
from lofi.lf import InferenceLabel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.data import CorpusRelease, Iso

__all__ = [
    "ChaosNliCodec",
    "MalformedNliError",
    "MnliCodec",
    "NliDecimal",
    "NliFile",
    "NliFormat",
    "NliIso",
    "NliPair",
    "SnliCodec",
    "UnliCodec",
]

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_SCHEME: Final = "at://local"
_CORPUS_NSID: Final = "pub.layers.corpus.corpus"
_MEMBERSHIP_NSID: Final = "pub.layers.corpus.membership"
_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
_SEGMENTATION_NSID: Final = "pub.layers.segmentation.segmentation"
_LAYER_NSID: Final = "pub.layers.annotation.annotationLayer"
_EXPERIMENT_NSID: Final = "pub.layers.judgment.experimentDef"
_JUDGMENT_SET_NSID: Final = "pub.layers.judgment.judgmentSet"
_PAIR_JSON: Final = "pair_json"
_PAIR_ORDINAL: Final = "pair_ordinal"
_FORMAT: Final = "format"
_RELATION_OBJECT: Final = "relation"
_SOURCE_LABELS: Final = {
    "entailment": InferenceLabel.ENTAILS.value,
    "contradiction": InferenceLabel.CONTRADICTS.value,
    "neutral": InferenceLabel.NEUTRAL.value,
}
_WIRE_LABELS: Final = {value: key for key, value in _SOURCE_LABELS.items()}
_CHAOS_LABELS: Final = {
    "e": InferenceLabel.ENTAILS.value,
    "c": InferenceLabel.CONTRADICTS.value,
    "n": InferenceLabel.NEUTRAL.value,
}
_CHAOS_WIRE: Final = {value: key for key, value in _CHAOS_LABELS.items()}
_RELATIONS: Final = tuple(_WIRE_LABELS)
_WORD: Final = re.compile(r"\S+")
_PAIR_SIZE: Final = 2
_UNLI_HEADER: Final = (
    "ID",
    "Premise",
    "Hypothesis",
    "NLI label",
    "UNLI label",
)
_PROBABILITY: Final = re.compile(r"(?:0(?:\.\d+)?|1(?:\.0+)?)")

type PairMetadata = tuple[tuple[str, JsonValue], ...]
type LabelCounts = tuple[tuple[str, int], ...]


class MalformedNliError(ValueError):
    """An error for malformed NLI data or generated records."""


class NliFormat(enum.StrEnum):
    """A normalized NLI wire format."""

    SNLI = "snli"
    MNLI = "mnli"
    CHAOS_NLI = "chaosnli"
    UNLI = "unli"


class NliDecimal(NamedTuple):
    """An exact decimal coefficient and base-ten exponent."""

    coefficient: int
    exponent: int


def _decimal_of(value: NliDecimal) -> Decimal:
    """Reconstruct one exact decimal without applying a numeric context."""
    return Decimal(f"{value.coefficient}e{value.exponent}")


class NliPair(NamedTuple):
    """A normalized pair consisting of a premise and hypothesis."""

    pair_id: str
    premise: str
    hypothesis: str
    relation: str | None = None
    label_counts: LabelCounts = ()
    metadata: PairMetadata = ()
    probability: NliDecimal | None = None


class NliFile(NamedTuple):
    """A normalized NLI file."""

    format: NliFormat
    pairs: tuple[NliPair, ...]


class _PairRefs(NamedTuple):
    """Generated references for one pair."""

    premise: str
    hypothesis: str
    layer: str


type _LayerModel = (
    corpus_records.Corpus
    | corpus_records.Membership
    | expression_records.Expression
    | segmentation_records.Segmentation
    | annotation_records.AnnotationLayer
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet
)


def _text_of(src: str | bytes) -> str:
    """Return UTF-8 source text."""
    if isinstance(src, str):
        return src

    try:
        return src.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "NLI input is not UTF-8"
        raise MalformedNliError(msg) from error


def _refuse_nonfinite(value: str) -> NoReturn:
    """Reject a nonfinite JSON number."""
    msg = f"nonfinite JSON number {value!r}"
    raise ValueError(msg)


def _object(line: str, number: int) -> dict[str, JsonValue]:
    """Return one JSON object."""
    try:
        value = json.loads(line, parse_constant=_refuse_nonfinite)
    except (json.JSONDecodeError, ValueError) as error:
        msg = f"line {number} is not JSON"
        raise MalformedNliError(msg) from error

    if not isinstance(value, dict):
        msg = f"line {number} is not a JSON object"
        raise MalformedNliError(msg)

    return value


def _required_text(row: Mapping[str, JsonValue], key: str, number: int) -> str:
    """Return one required text field."""
    value = row.get(key)

    if not isinstance(value, str) or not value:
        msg = f"line {number} has no nonempty {key}"
        raise MalformedNliError(msg)

    return value


def _metadata_of(
    row: Mapping[str, JsonValue],
    known: frozenset[str],
) -> PairMetadata:
    """Return fields not consumed by a format."""
    return tuple(sorted((key, value) for key, value in row.items() if key not in known))


def _json_rows(src: str | bytes) -> tuple[dict[str, JsonValue], ...]:
    """Return every nonempty JSONL row."""
    rows = tuple(
        _object(line, number)
        for number, line in enumerate(_text_of(src).splitlines(), start=1)
        if line.strip()
    )

    if not rows:
        msg = "NLI input has no pairs"
        raise MalformedNliError(msg)

    return rows


def _source_relation(value: str, number: int) -> str | None:
    """Return one canonical gold relation."""
    if value == "-":
        return None

    relation = _SOURCE_LABELS.get(value)

    if relation is None:
        msg = f"line {number} has unknown gold label {value!r}"
        raise MalformedNliError(msg)

    return relation


def _static_file(src: str | bytes, flavor: NliFormat) -> NliFile:
    """Return a normalized SNLI or MNLI file."""
    known = frozenset(("pairID", "sentence1", "sentence2", "gold_label"))
    pairs = tuple(
        NliPair(
            pair_id=_required_text(row, "pairID", number),
            premise=_required_text(row, "sentence1", number),
            hypothesis=_required_text(row, "sentence2", number),
            relation=_source_relation(
                _required_text(row, "gold_label", number), number
            ),
            metadata=_metadata_of(row, known),
        )
        for number, row in enumerate(_json_rows(src), start=1)
    )
    return _validated(NliFile(flavor, pairs))


def _chaos_counts(row: Mapping[str, JsonValue], number: int) -> LabelCounts:
    """Return canonical counts from one ChaosNLI row."""
    value = row.get("label_counter")

    if not isinstance(value, dict) or set(value) != set(_CHAOS_LABELS):
        msg = f"line {number} has no exact ChaosNLI label counter"
        raise MalformedNliError(msg)

    counts: list[tuple[str, int]] = []

    for wire, relation in _CHAOS_LABELS.items():
        count = value[wire]

        if type(count) is not int or count < 0:
            msg = f"line {number} has invalid count for {wire!r}"
            raise MalformedNliError(msg)

        counts.append((relation, count))

    if not sum(count for _, count in counts):
        msg = f"line {number} has no source annotations"
        raise MalformedNliError(msg)

    declared = row.get("label_count")

    if type(declared) is not int or declared != sum(count for _, count in counts):
        msg = f"line {number} has a label count that disagrees with its counter"
        raise MalformedNliError(msg)

    return tuple(counts)


def _chaos_file(src: str | bytes) -> NliFile:
    """Return a normalized ChaosNLI count release."""
    known = frozenset(("uid", "example", "label_counter"))
    pairs: list[NliPair] = []

    for number, row in enumerate(_json_rows(src), start=1):
        example = row.get("example")

        if not isinstance(example, dict):
            msg = f"line {number} has no example object"
            raise MalformedNliError(msg)

        pairs.append(
            NliPair(
                pair_id=_required_text(row, "uid", number),
                premise=_required_text(example, "premise", number),
                hypothesis=_required_text(example, "hypothesis", number),
                label_counts=_chaos_counts(row, number),
                metadata=_metadata_of(row, known),
            )
        )

    return _validated(NliFile(NliFormat.CHAOS_NLI, tuple(pairs)))


def _exact_probability(value: str, number: int) -> NliDecimal:
    """Return one exact probability from its source decimal spelling."""
    if _PROBABILITY.fullmatch(value) is None:
        msg = f"line {number} has invalid UNLI probability {value!r}"
        raise MalformedNliError(msg)

    sign, digits, exponent = Decimal(value).as_tuple()

    if not isinstance(exponent, int):
        msg = f"line {number} has nonfinite UNLI probability {value!r}"
        raise MalformedNliError(msg)

    coefficient = int("".join(str(digit) for digit in digits))

    if sign:
        coefficient = -coefficient

    return NliDecimal(coefficient, exponent)


def _unli_file(src: str | bytes) -> NliFile:
    """Return a normalized UNLI probability release."""
    try:
        rows = tuple(
            csv.reader(StringIO(_text_of(src), newline=""), delimiter="\t", strict=True)
        )
    except csv.Error as error:
        msg = "UNLI input is not valid TSV"
        raise MalformedNliError(msg) from error

    if not rows:
        msg = "UNLI input has no header"
        raise MalformedNliError(msg)

    if tuple(rows[0]) != _UNLI_HEADER:
        msg = "UNLI input has no exact release header"
        raise MalformedNliError(msg)

    if len(rows) == 1:
        msg = "UNLI input has no pairs"
        raise MalformedNliError(msg)

    pairs: list[NliPair] = []

    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(_UNLI_HEADER):
            msg = f"line {number} has {len(row)} columns, expected five"
            raise MalformedNliError(msg)

        pair_id, premise, hypothesis, label, probability = row

        if not pair_id or not premise or not hypothesis:
            msg = f"line {number} has an empty ID, Premise, or Hypothesis"
            raise MalformedNliError(msg)

        pairs.append(
            NliPair(
                pair_id=pair_id,
                premise=premise,
                hypothesis=hypothesis,
                relation=_source_relation(label, number),
                probability=_exact_probability(probability, number),
            )
        )

    return _validated(NliFile(NliFormat.UNLI, tuple(pairs)))


def _validated(source: NliFile) -> NliFile:
    """Return a normalized file with one valid shape per format."""
    identifiers: set[str] = set()

    for pair in source.pairs:
        if not pair.pair_id or not pair.premise or not pair.hypothesis:
            msg = "an NLI pair has an empty identifier or expression"
            raise MalformedNliError(msg)

        if pair.pair_id in identifiers:
            msg = f"an NLI file repeats pair id {pair.pair_id!r}"
            raise MalformedNliError(msg)

        identifiers.add(pair.pair_id)
        static = source.format in {NliFormat.SNLI, NliFormat.MNLI}
        continuous = source.format is NliFormat.UNLI

        if pair.relation is not None and pair.relation not in _RELATIONS:
            msg = f"an NLI pair has unknown relation {pair.relation!r}"
            raise MalformedNliError(msg)

        if any(
            relation not in _RELATIONS or type(count) is not int or count < 0
            for relation, count in pair.label_counts
        ):
            msg = "an NLI pair has invalid label counts"
            raise MalformedNliError(msg)

        if len({relation for relation, _ in pair.label_counts}) != len(
            pair.label_counts
        ):
            msg = "an NLI pair repeats a label count"
            raise MalformedNliError(msg)

        if static and pair.label_counts:
            msg = "a labeled NLI pair has elicited responses"
            raise MalformedNliError(msg)

        if static and pair.probability is not None:
            msg = "a labeled NLI pair has a continuous response"
            raise MalformedNliError(msg)

        if source.format is NliFormat.CHAOS_NLI and (
            pair.relation is not None or not pair.label_counts
        ):
            msg = "a ChaosNLI pair has no exact label counts"
            raise MalformedNliError(msg)

        if source.format is NliFormat.CHAOS_NLI and pair.probability is not None:
            msg = "a ChaosNLI pair has a continuous response"
            raise MalformedNliError(msg)

        if continuous and (
            pair.relation is None
            or pair.label_counts
            or pair.probability is None
            or pair.probability.coefficient < 0
            or _decimal_of(pair.probability) > 1
        ):
            msg = "a UNLI pair has no exact probability and categorical source label"
            raise MalformedNliError(msg)

    return source


def _json_line(row: Mapping[str, JsonValue]) -> str:
    """Return one canonical JSONL row."""
    return json.dumps(
        row,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _static_text(source: NliFile) -> str:
    """Return canonical SNLI or MNLI JSONL."""
    lines: list[str] = []

    for pair in source.pairs:
        row = dict(pair.metadata)
        row.update(
            {
                "gold_label": "-"
                if pair.relation is None
                else _WIRE_LABELS[pair.relation],
                "pairID": pair.pair_id,
                "sentence1": pair.premise,
                "sentence2": pair.hypothesis,
            }
        )
        lines.append(_json_line(row))

    return "".join(f"{line}\n" for line in lines)


def _chaos_text(source: NliFile) -> str:
    """Return canonical ChaosNLI JSONL."""
    lines: list[str] = []

    for pair in source.pairs:
        row = dict(pair.metadata)
        row.update(
            {
                "example": {
                    "hypothesis": pair.hypothesis,
                    "premise": pair.premise,
                },
                "label_counter": {
                    _CHAOS_WIRE[relation]: count
                    for relation, count in pair.label_counts
                },
                "uid": pair.pair_id,
            }
        )
        lines.append(_json_line(row))

    return "".join(f"{line}\n" for line in lines)


def _decimal_text(value: NliDecimal) -> str:
    """Return the fixed-point spelling of one exact decimal."""
    return format(_decimal_of(value), "f")


def _unli_text(source: NliFile) -> str:
    """Return canonical UNLI TSV."""
    target = StringIO(newline="")
    writer = csv.writer(target, delimiter="\t", lineterminator="\n")
    writer.writerow(_UNLI_HEADER)

    for pair in source.pairs:
        if pair.relation is None or pair.probability is None:
            msg = "a UNLI pair has no source label or exact probability"
            raise MalformedNliError(msg)

        writer.writerow(
            (
                pair.pair_id,
                pair.premise,
                pair.hypothesis,
                _WIRE_LABELS[pair.relation],
                _decimal_text(pair.probability),
            )
        )

    return target.getvalue()


def _parse(src: str | bytes, flavor: NliFormat) -> NliFile:
    """Return one normalized external file."""
    if flavor in {NliFormat.SNLI, NliFormat.MNLI}:
        return _static_file(src, flavor)

    if flavor is NliFormat.CHAOS_NLI:
        return _chaos_file(src)

    return _unli_file(src)


def _write(source: NliFile) -> str:
    """Return a normalized file in its declared format."""
    if source.format in {NliFormat.SNLI, NliFormat.MNLI}:
        return _static_text(source)

    if source.format is NliFormat.CHAOS_NLI:
        return _chaos_text(source)

    return _unli_text(source)


def _digest(*parts: str) -> str:
    """Return a stable source-derived key."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()[:24]


def _uri(nsid: str, key: str) -> str:
    """Return one local record reference."""
    return f"{_SCHEME}/{nsid}/{key}"


def _features(entries: Iterable[tuple[str, str]]) -> defs_records.FeatureMap:
    """Return a Layers feature map."""
    return defs_records.FeatureMap(
        entries=tuple(
            defs_records.Feature(key=key, value=value) for key, value in entries
        )
    )


def _licensing(spec: CorpusRelease) -> defs_records.Licensing:
    """Return caller-supplied licensing."""
    return defs_records.Licensing(
        expression=spec.spdx,
        licenses=(defs_records.LicenseRef(spdx=spec.spdx, appliesTo="whole"),),
    )


def _record(nsid: str, key: str, value: _LayerModel) -> FragmentRecord:
    """Return one serialized Layers record."""
    return FragmentRecord(
        local_id=_uri(nsid, key),
        nsid=nsid,
        value_json=value.model_dump_json(),
    )


def _pair_refs(pair: NliPair, spec: CorpusRelease) -> _PairRefs:
    """Return stable references for one pair."""
    stem = _digest(spec.source_id, spec.release_id, pair.pair_id)
    return _PairRefs(
        _uri(_EXPRESSION_NSID, f"{stem}-premise"),
        _uri(_EXPRESSION_NSID, f"{stem}-hypothesis"),
        _uri(_LAYER_NSID, f"{stem}-relation"),
    )


def _tokens(text: str) -> tuple[segmentation_records.Token, ...]:
    """Return whitespace tokens with UTF-8 byte spans."""
    return tuple(
        segmentation_records.Token(
            tokenIndex=index,
            text=match.group(),
            textSpan=defs_records.Span(
                byteStart=len(text[: match.start()].encode()),
                byteEnd=len(text[: match.end()].encode()),
            ),
        )
        for index, match in enumerate(_WORD.finditer(text))
    )


def _expression(
    text: str, key: str, spec: CorpusRelease
) -> expression_records.Expression:
    """Return one expression record."""
    return expression_records.Expression(
        id=key,
        kind="sentence",
        createdAt=_EPOCH,
        text=text,
        languages=(spec.language,),
        eprintRefs=spec.eprint_refs,
    )


def _segmentation(
    expression_ref: str,
    text: str,
) -> segmentation_records.Segmentation:
    """Return one whitespace segmentation."""
    return segmentation_records.Segmentation(
        createdAt=_EPOCH,
        expression=expression_ref,
        tokenizations=(
            segmentation_records.Tokenization(
                uuid=defs_records.Uuid(value="whitespace"),
                kind="whitespace",
                tokens=_tokens(text),
            ),
        ),
    )


def _pair_payload(pair: NliPair) -> str:
    """Return canonical JSON for one normalized pair."""
    return _json_line(
        {
            "hypothesis": pair.hypothesis,
            "label_counts": [list(entry) for entry in pair.label_counts],
            "metadata": dict(pair.metadata),
            "pair_id": pair.pair_id,
            "premise": pair.premise,
            "relation": pair.relation,
        }
    )


def _relation_layer(
    pair: NliPair,
    refs: _PairRefs,
    source: NliFile,
    ordinal: int,
    spec: CorpusRelease,
) -> annotation_records.AnnotationLayer:
    """Return the directed relation layer for one pair."""
    return annotation_records.AnnotationLayer(
        createdAt=_EPOCH,
        expression=refs.premise,
        kind="relation",
        annotations=(
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value=_RELATION_OBJECT),
                label=pair.relation,
                arguments=(
                    annotation_records.ArgumentRef(
                        role="hypothesis",
                        target=defs_records.ObjectRef(recordRef=refs.hypothesis),
                    ),
                ),
                features=_features(
                    (
                        (_FORMAT, source.format.value),
                        (_PAIR_ORDINAL, str(ordinal)),
                        (_PAIR_JSON, _pair_payload(pair)),
                    )
                ),
            ),
        ),
        sourceMethod=spec.source_method,
        licensing=_licensing(spec),
        eprintRefs=spec.eprint_refs,
    )


def _experiment(source: NliFile, spec: CorpusRelease) -> FragmentRecord | None:
    """Return the elicitation experiment, if the source has responses."""
    if source.format not in {NliFormat.CHAOS_NLI, NliFormat.UNLI}:
        return None

    key = _digest(spec.source_id, spec.release_id, source.format.value, "experiment")
    refs = tuple(
        reference for pair in source.pairs for reference in _pair_refs(pair, spec)[:2]
    )
    continuous = source.format is NliFormat.UNLI
    experiment = judgment_records.ExperimentDef(
        name=f"{source.format.value} {spec.release_id}",
        createdAt=_EPOCH,
        taskType="continuous" if continuous else "categorical",
        measureType="inference",
        labels=() if continuous else _RELATIONS,
        scaleMin=0 if continuous else None,
        scaleMax=1 if continuous else None,
        languages=(spec.language,),
        licensing=_licensing(spec),
        eprintRefs=spec.eprint_refs,
        stimulusExpressionRefs=refs,
    )
    return _record(_EXPERIMENT_NSID, key, experiment)


def _judgments(
    pair: NliPair,
    refs: _PairRefs,
) -> tuple[judgment_records.Judgment, ...]:
    """Return every elicited source response for one pair."""
    if pair.probability is not None:
        return (
            judgment_records.Judgment(
                item=defs_records.ObjectRef(
                    recordRef=refs.layer,
                    objectId=defs_records.Uuid(value=_RELATION_OBJECT),
                ),
                continuousValue=judgment_records.DecimalValue(
                    coefficient=pair.probability.coefficient,
                    exponent=pair.probability.exponent,
                ),
            ),
        )

    return tuple(
        judgment_records.Judgment(
            item=defs_records.ObjectRef(
                recordRef=refs.layer,
                objectId=defs_records.Uuid(value=_RELATION_OBJECT),
            ),
            categoricalValue=relation,
        )
        for relation, count in pair.label_counts
        for _ in range(count)
    )


def _response_set(
    pair: NliPair,
    refs: _PairRefs,
    experiment_ref: str,
    spec: CorpusRelease,
) -> FragmentRecord:
    """Return anonymous responses for one elicited pair."""
    key = _digest(spec.source_id, spec.release_id, pair.pair_id, "responses")
    return _record(
        _JUDGMENT_SET_NSID,
        key,
        judgment_records.JudgmentSet(
            experimentRef=experiment_ref,
            createdAt=_EPOCH,
            judgments=_judgments(pair, refs),
        ),
    )


def _pair_records(
    pair: NliPair,
    source: NliFile,
    ordinal: int,
    corpus_ref: str,
    spec: CorpusRelease,
) -> tuple[FragmentRecord, ...]:
    """Return the expressions, segmentations, memberships, and relation."""
    refs = _pair_refs(pair, spec)
    premise_key = refs.premise.rsplit("/", maxsplit=1)[-1]
    hypothesis_key = refs.hypothesis.rsplit("/", maxsplit=1)[-1]
    layer_key = refs.layer.rsplit("/", maxsplit=1)[-1]
    return (
        _record(
            _EXPRESSION_NSID,
            premise_key,
            _expression(pair.premise, premise_key, spec),
        ),
        _record(
            _EXPRESSION_NSID,
            hypothesis_key,
            _expression(pair.hypothesis, hypothesis_key, spec),
        ),
        _record(
            _SEGMENTATION_NSID,
            premise_key,
            _segmentation(refs.premise, pair.premise),
        ),
        _record(
            _SEGMENTATION_NSID,
            hypothesis_key,
            _segmentation(refs.hypothesis, pair.hypothesis),
        ),
        _record(
            _MEMBERSHIP_NSID,
            premise_key,
            corpus_records.Membership(
                corpusRef=corpus_ref,
                expressionRef=refs.premise,
                createdAt=_EPOCH,
                split=spec.split,
                ordinal=ordinal * 2,
            ),
        ),
        _record(
            _MEMBERSHIP_NSID,
            hypothesis_key,
            corpus_records.Membership(
                corpusRef=corpus_ref,
                expressionRef=refs.hypothesis,
                createdAt=_EPOCH,
                split=spec.split,
                ordinal=ordinal * 2 + 1,
            ),
        ),
        _record(
            _LAYER_NSID,
            layer_key,
            _relation_layer(pair, refs, source, ordinal, spec),
        ),
    )


def _records_of(source: NliFile, spec: CorpusRelease) -> tuple[FragmentRecord, ...]:
    """Return canonical Layers records for one normalized file."""
    source = _validated(source)
    corpus_key = _digest(spec.source_id, spec.release_id, source.format.value, "corpus")
    corpus_ref = _uri(_CORPUS_NSID, corpus_key)
    records: list[FragmentRecord] = [
        _record(
            _CORPUS_NSID,
            corpus_key,
            corpus_records.Corpus(
                name=f"{source.format.value} {spec.release_id}",
                createdAt=_EPOCH,
                languages=(spec.language,),
                version=spec.release_id,
                expressionCount=len(source.pairs) * 2,
                licensing=_licensing(spec),
                eprintRefs=spec.eprint_refs,
            ),
        )
    ]
    experiment = _experiment(source, spec)
    experiment_ref = None if experiment is None else experiment.local_id

    if experiment is not None:
        records.append(experiment)

    for ordinal, pair in enumerate(source.pairs):
        records.extend(_pair_records(pair, source, ordinal, corpus_ref, spec))

    if experiment_ref is not None:
        records.extend(
            _response_set(
                pair,
                _pair_refs(pair, spec),
                experiment_ref,
                spec,
            )
            for pair in source.pairs
        )

    return tuple(records)


def _feature_values(features: defs_records.FeatureMap | None) -> Mapping[str, str]:
    """Return one feature map without duplicate keys."""
    if features is None:
        return {}

    values = {entry.key: entry.value for entry in features.entries}

    if len(values) != len(features.entries):
        msg = "an NLI record repeats a feature key"
        raise MalformedNliError(msg)

    return values


def _payload_pair(text: str) -> NliPair:
    """Return one pair from canonical record metadata."""
    row = _object(text, 0)
    expected = {
        "hypothesis",
        "label_counts",
        "metadata",
        "pair_id",
        "premise",
        "relation",
    }

    if set(row) != expected:
        msg = "an NLI relation layer has unsupported pair metadata"
        raise MalformedNliError(msg)

    counts_value = row["label_counts"]
    metadata_value = row["metadata"]
    relation_value = row["relation"]

    if not isinstance(counts_value, list) or not isinstance(metadata_value, dict):
        msg = "an NLI relation layer has invalid pair metadata"
        raise MalformedNliError(msg)

    counts: list[tuple[str, int]] = []

    for entry in counts_value:
        if (
            not isinstance(entry, list)
            or len(entry) != _PAIR_SIZE
            or not isinstance(entry[0], str)
            or type(entry[1]) is not int
        ):
            msg = "an NLI relation layer has invalid label counts"
            raise MalformedNliError(msg)

        counts.append((entry[0], entry[1]))

    if relation_value is not None and not isinstance(relation_value, str):
        msg = "an NLI relation layer has an invalid relation"
        raise MalformedNliError(msg)

    return NliPair(
        _required_text(row, "pair_id", 0),
        _required_text(row, "premise", 0),
        _required_text(row, "hypothesis", 0),
        relation_value,
        tuple(counts),
        tuple(sorted(metadata_value.items())),
    )


def _decimal_parts(value: Decimal) -> NliDecimal:
    """Return the coefficient and exponent of one finite Lairs decimal."""
    sign, digits, exponent = value.as_tuple()

    if not isinstance(exponent, int):
        msg = "a UNLI judgment holds a nonfinite continuous response"
        raise MalformedNliError(msg)

    coefficient = int("".join(str(digit) for digit in digits))

    if sign:
        coefficient = -coefficient

    return NliDecimal(coefficient, exponent)


def _unli_probabilities(
    records: tuple[tuple[FragmentRecord, _LayerModel], ...],
) -> dict[str, NliDecimal]:
    """Return exact UNLI responses from Lairs' public judgment rows."""
    study = JudgmentStudy.new()

    for record, model in records:
        study.add_record(record.local_id, model)

    probabilities: dict[str, NliDecimal] = {}

    for row in study.judgments():
        if row.scalar_value is not None or row.categorical_value is not None:
            msg = "a UNLI judgment uses a noncontinuous response arm"
            raise MalformedNliError(msg)

        if row.continuous_value is None:
            msg = "a UNLI judgment has no continuous response"
            raise MalformedNliError(msg)

        if row.item_ref in probabilities:
            msg = f"a UNLI pair has repeated continuous responses: {row.item_ref!r}"
            raise MalformedNliError(msg)

        probabilities[row.item_ref] = _decimal_parts(row.continuous_value)

    return probabilities


def _model_of(record: FragmentRecord) -> _LayerModel:
    """Return one validated Layers model."""
    try:
        model: _LayerModel

        if record.nsid == _CORPUS_NSID:
            model = corpus_records.Corpus.model_validate_json(record.value_json)
        elif record.nsid == _MEMBERSHIP_NSID:
            model = corpus_records.Membership.model_validate_json(record.value_json)
        elif record.nsid == _EXPRESSION_NSID:
            model = expression_records.Expression.model_validate_json(record.value_json)
        elif record.nsid == _SEGMENTATION_NSID:
            model = segmentation_records.Segmentation.model_validate_json(
                record.value_json
            )
        elif record.nsid == _LAYER_NSID:
            model = annotation_records.AnnotationLayer.model_validate_json(
                record.value_json
            )
        elif record.nsid == _EXPERIMENT_NSID:
            model = judgment_records.ExperimentDef.model_validate_json(
                record.value_json
            )
        elif record.nsid == _JUDGMENT_SET_NSID:
            model = judgment_records.JudgmentSet.model_validate_json(record.value_json)
        else:
            msg = f"an NLI fragment contains {record.nsid!r}"
            raise MalformedNliError(msg)
    except ValueError as error:
        msg = f"an NLI record is invalid: {record.local_id!r}"
        raise MalformedNliError(msg) from error

    return model


def _source_of(
    records: tuple[FragmentRecord, ...],
    expected: NliFormat,
    spec: CorpusRelease,
) -> NliFile:
    """Return the normalized file encoded by Layers records."""
    modeled = tuple((record, _model_of(record)) for record in records)
    indexed: list[tuple[int, NliPair, str]] = []

    for record, model in modeled:
        if not isinstance(model, annotation_records.AnnotationLayer):
            continue

        if (
            model.kind != "relation"
            or model.subkind is not None
            or len(model.annotations) != 1
        ):
            msg = "an NLI relation layer has a noncanonical shape"
            raise MalformedNliError(msg)

        values = _feature_values(model.annotations[0].features)

        if set(values) != {_FORMAT, _PAIR_ORDINAL, _PAIR_JSON}:
            msg = "an NLI relation layer lacks exact codec metadata"
            raise MalformedNliError(msg)

        if values[_FORMAT] != expected.value:
            msg = "an NLI fragment mixes formats"
            raise MalformedNliError(msg)

        try:
            ordinal = int(values[_PAIR_ORDINAL])
        except ValueError as error:
            msg = "an NLI relation layer has a noninteger pair ordinal"
            raise MalformedNliError(msg) from error

        indexed.append((ordinal, _payload_pair(values[_PAIR_JSON]), record.local_id))

    indexed.sort(key=lambda entry: entry[0])

    if tuple(ordinal for ordinal, _, _ in indexed) != tuple(range(len(indexed))):
        msg = "NLI pair ordinals are not contiguous from zero"
        raise MalformedNliError(msg)

    if expected is NliFormat.UNLI:
        probabilities = _unli_probabilities(modeled)
        layer_refs = {layer_ref for _, _, layer_ref in indexed}

        if set(probabilities) != layer_refs:
            msg = "UNLI pair layers and continuous judgments do not correspond"
            raise MalformedNliError(msg)

        pairs = tuple(
            NliPair(
                pair.pair_id,
                pair.premise,
                pair.hypothesis,
                pair.relation,
                pair.label_counts,
                pair.metadata,
                probabilities[layer_ref],
            )
            for _, pair, layer_ref in indexed
        )
    else:
        pairs = tuple(pair for _, pair, _ in indexed)

    source = _validated(NliFile(expected, pairs))

    if _records_of(source, spec) != records:
        msg = "NLI records are not in the codec's canonical form"
        raise MalformedNliError(msg)

    return source


class NliIso:
    """A translation from normalized NLI data to Layers records."""

    __slots__ = ("_format", "_spec")

    def __init__(self, flavor: NliFormat, spec: CorpusRelease) -> None:
        self._format = flavor
        self._spec = spec

    def forward(self, source: NliFile, /) -> CorpusFragment:
        """Return canonical Layers records."""
        if source.format is not self._format:
            msg = f"the {self._format.value} codec received {source.format.value}"
            raise MalformedNliError(msg)

        return CorpusFragment(
            records=_records_of(source, self._spec),
            source=self._format.value,
        )

    def backward(self, target: CorpusFragment, /) -> NliFile:
        """Return normalized pairs from Layers records."""
        if target.source != self._format.value:
            msg = f"the {self._format.value} codec received {target.source!r} records"
            raise MalformedNliError(msg)

        return _source_of(target.records, self._format, self._spec)


class _NliCodec:
    """Shared codec mechanics for normalized NLI formats."""

    __slots__ = ("_format", "_iso", "name")

    name: str

    def __init__(self, flavor: NliFormat, spec: CorpusRelease) -> None:
        self.name = flavor.value
        self._format = flavor
        self._iso = NliIso(flavor, spec)

    @property
    def iso(self) -> Iso[NliFile, CorpusFragment]:
        """Return the format-to-record translation."""
        return self._iso

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return Layers records for normalized NLI pairs."""
        decoded = self._iso.forward(_parse(src, self._format))

        if into is None:
            return decoded

        return CorpusFragment(records=into.records + decoded.records, source=self.name)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return normalized NLI text for Layers records."""
        fragment = CorpusFragment(records=tuple(records), source=self.name)
        return _write(self._iso.backward(fragment))


@registers_as(NliFormat.SNLI)
class SnliCodec(_NliCodec):
    """A codec for normalized SNLI JSONL."""

    def __init__(self, spec: CorpusRelease) -> None:
        super().__init__(NliFormat.SNLI, spec)


@registers_as(NliFormat.MNLI)
class MnliCodec(_NliCodec):
    """A codec for normalized MNLI JSONL."""

    def __init__(self, spec: CorpusRelease) -> None:
        super().__init__(NliFormat.MNLI, spec)


@registers_as(NliFormat.CHAOS_NLI)
class ChaosNliCodec(_NliCodec):
    """A codec for normalized ChaosNLI count-release JSONL."""

    def __init__(self, spec: CorpusRelease) -> None:
        super().__init__(NliFormat.CHAOS_NLI, spec)


@registers_as(NliFormat.UNLI)
class UnliCodec(_NliCodec):
    """A codec for normalized UNLI TSV."""

    def __init__(self, spec: CorpusRelease) -> None:
        super().__init__(NliFormat.UNLI, spec)

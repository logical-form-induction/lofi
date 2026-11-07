"""Participant-level MegaAttitude tables as Layers inference judgments."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from io import StringIO
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records

from lofi.data.corpora.megaattitude.spec import MegaAttitudeSpec
from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi.data import Iso

__all__ = [
    "MEGAINTENSIONALITY_NAME",
    "MEGANEGRAISING_NAME",
    "MEGAVERIDICALITY_NAME",
    "MalformedMegaAttitudeError",
    "MegaAttitudeFile",
    "MegaAttitudeFormat",
    "MegaAttitudeIso",
    "MegaIntensionalityCodec",
    "MegaIntensionalityRow",
    "MegaNegRaisingCodec",
    "MegaNegRaisingRow",
    "MegaVeridicalityCodec",
    "MegaVeridicalityRow",
]

MEGAVERIDICALITY_NAME: Final = "megaveridicality"
MEGANEGRAISING_NAME: Final = "meganegraising"
MEGAINTENSIONALITY_NAME: Final = "megaintensionality"

type MegaAttitudeFormat = Literal[
    "megaveridicality", "meganegraising", "megaintensionality"
]

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_SCHEME: Final = "at://local"
_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
_LAYER_NSID: Final = "pub.layers.annotation.annotationLayer"
_EXPERIMENT_NSID: Final = "pub.layers.judgment.experimentDef"
_JUDGMENT_SET_NSID: Final = "pub.layers.judgment.judgmentSet"
_RELATION_OBJECT: Final = "relation"
_ROW_JSON: Final = "row_json"
_ROW_ORDINAL: Final = "row_ordinal"

_VERIDICALITY_HEADER: Final = (
    "participant",
    "list",
    "presentationorder",
    "verb",
    "frame",
    "polarity",
    "conditional",
    "sentence",
    "veridicality",
    "acceptability",
    "nativeenglish",
)
_NEG_RAISING_HEADER: Final = (
    "participant",
    "list",
    "presentationorder",
    "verb",
    "frame",
    "tense",
    "subject",
    "sentence1",
    "sentence2",
    "negraising",
    "acceptability",
    "nativeenglish",
)
_INTENSIONALITY_HEADER: Final = (
    "participant",
    "antecedent_verb",
    "antecedent_polarity",
    "target",
    "antecedent_frame",
    "consequent_verb",
    "consequent_embedded_tense",
    "antecedent_text",
    "consequent_text",
    "response",
    "sanity",
)


class MalformedMegaAttitudeError(ValueError):
    """Report a malformed MegaAttitude table or fragment."""


class ExactDecimal(NamedTuple):
    """An exact base-ten decimal."""

    coefficient: int
    exponent: int


class MegaVeridicalityRow(NamedTuple):
    """One released MegaVeridicality response."""

    participant: str
    list_id: str
    presentation_order: int
    verb: str
    frame: str
    polarity: str
    conditional: str
    antecedent: str
    response: str | None
    acceptability: int
    native_english: str


class MegaNegRaisingRow(NamedTuple):
    """One released MegaNegRaising response."""

    participant: str
    list_id: str
    presentation_order: int
    verb: str
    frame: str
    tense: str
    subject: str
    antecedent: str
    consequent: str
    response: ExactDecimal
    acceptability: ExactDecimal
    native_english: str


class MegaIntensionalityRow(NamedTuple):
    """One released MegaIntensionality response."""

    participant: str
    verb: str
    polarity: str
    target: str
    frame: str
    consequent_verb: str
    consequent_tense: str
    antecedent: str
    consequent: str
    response: ExactDecimal
    sanity: int


type MegaAttitudeRow = MegaVeridicalityRow | MegaNegRaisingRow | MegaIntensionalityRow


class MegaAttitudeFile(NamedTuple):
    """One normalized participant-level MegaAttitude table."""

    format: MegaAttitudeFormat
    rows: tuple[MegaAttitudeRow, ...]


class _Pair(NamedTuple):
    """The pair and response fields shared by record construction."""

    participant: str
    antecedent: str
    consequent: str
    response: str | ExactDecimal | None
    factors: tuple[tuple[str, str], ...]


class _PairRefs(NamedTuple):
    """Record references generated for one expression pair."""

    antecedent: str
    consequent: str
    layer: str


type _RecordModel = (
    annotation_records.AnnotationLayer
    | expression_records.Expression
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
        msg = "MegaAttitude input is not UTF-8"
        raise MalformedMegaAttitudeError(msg) from error


def _rows(
    src: str | bytes, header: tuple[str, ...], *, optional: frozenset[int] = frozenset()
) -> tuple[list[str], ...]:
    """Return a nonempty exact-header TSV table.

    Every field must be nonblank except those at the ``optional`` column
    positions, where the release spells a missing response as an empty field.
    """
    try:
        rows = tuple(
            csv.reader(StringIO(_text_of(src), newline=""), delimiter="\t", strict=True)
        )
    except csv.Error as error:
        msg = "MegaAttitude input is not valid TSV"
        raise MalformedMegaAttitudeError(msg) from error

    if not rows or tuple(rows[0]) != header:
        msg = "MegaAttitude input has no exact release header"
        raise MalformedMegaAttitudeError(msg)

    if len(rows) == 1:
        msg = "MegaAttitude input has no response rows"
        raise MalformedMegaAttitudeError(msg)

    expected = len(header)

    for number, row in enumerate(rows[1:], start=2):
        blank = any(
            not value.strip()
            for column, value in enumerate(row)
            if column not in optional
        )
        if len(row) != expected or blank:
            msg = f"line {number} is not a complete {expected}-column response"
            raise MalformedMegaAttitudeError(msg)

    return rows[1:]


def _integer(value: str, field: str, number: int) -> int:
    """Return one source integer."""
    try:
        return int(value)
    except ValueError as error:
        msg = f"line {number} has noninteger {field} {value!r}"
        raise MalformedMegaAttitudeError(msg) from error


def _decimal(value: str, field: str, number: int) -> ExactDecimal:
    """Return one exact finite decimal."""
    try:
        parsed = Decimal(value)
    except Exception as error:
        msg = f"line {number} has invalid {field} {value!r}"
        raise MalformedMegaAttitudeError(msg) from error

    if not parsed.is_finite():
        msg = f"line {number} has nonfinite {field} {value!r}"
        raise MalformedMegaAttitudeError(msg)

    sign, digits, exponent = parsed.as_tuple()

    if not isinstance(exponent, int):
        msg = f"line {number} has nonfinite {field} {value!r}"
        raise MalformedMegaAttitudeError(msg)

    coefficient = int("".join(str(digit) for digit in digits))

    return ExactDecimal(-coefficient if sign else coefficient, exponent)


def _decimal_value(value: ExactDecimal) -> Decimal:
    """Return the decimal an exact pair denotes."""
    return Decimal(f"{value.coefficient}e{value.exponent}")


def _probability(value: str, field: str, number: int) -> ExactDecimal:
    """Return one exact response on the closed unit scale."""
    parsed = _decimal(value, field, number)

    if not Decimal(0) <= _decimal_value(parsed) <= Decimal(1):
        msg = f"line {number} has {field} outside the closed unit scale"
        raise MalformedMegaAttitudeError(msg)

    return parsed


def _relation_for(response: str, spec: MegaAttitudeSpec) -> str:
    """Return the declared inference relation for one source response."""
    for declared in spec.veridicality_relations:
        if declared.response == response:
            return declared.relation

    msg = f"unknown MegaVeridicality response {response!r}"
    raise MalformedMegaAttitudeError(msg)


def _response_for(relation: str, spec: MegaAttitudeSpec) -> str:
    """Return the declared source response for one inference relation."""
    for declared in spec.veridicality_relations:
        if declared.relation == relation:
            return declared.response

    msg = "a MegaVeridicality response has no declared relation label"
    raise MalformedMegaAttitudeError(msg)


def _veridicality_file(src: str | bytes, spec: MegaAttitudeSpec) -> MegaAttitudeFile:
    """Return a MegaVeridicality v2.1 response table."""
    parsed: list[MegaAttitudeRow] = []

    rows = _rows(src, _VERIDICALITY_HEADER, optional=frozenset({8}))

    for number, row in enumerate(rows, start=2):
        response = row[8] or None

        if response is not None:
            try:
                _relation_for(response, spec)
            except MalformedMegaAttitudeError as error:
                msg = f"line {number} has {error}"
                raise MalformedMegaAttitudeError(msg) from error

        parsed.append(
            MegaVeridicalityRow(
                row[0],
                row[1],
                _integer(row[2], "presentationorder", number),
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                response,
                _integer(row[9], "acceptability", number),
                row[10],
            )
        )

    return MegaAttitudeFile(MEGAVERIDICALITY_NAME, tuple(parsed))


def _negraising_file(src: str | bytes) -> MegaAttitudeFile:
    """Return a MegaNegRaising v1.1 response table."""
    parsed: list[MegaAttitudeRow] = []

    for number, row in enumerate(_rows(src, _NEG_RAISING_HEADER), start=2):
        parsed.append(
            MegaNegRaisingRow(
                row[0],
                row[1],
                _integer(row[2], "presentationorder", number),
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                _probability(row[9], "negraising", number),
                _probability(row[10], "acceptability", number),
                row[11],
            )
        )

    return MegaAttitudeFile(MEGANEGRAISING_NAME, tuple(parsed))


def _intensionality_file(src: str | bytes) -> MegaAttitudeFile:
    """Return a MegaIntensionality v1 response table."""
    parsed: list[MegaAttitudeRow] = []

    for number, row in enumerate(_rows(src, _INTENSIONALITY_HEADER), start=2):
        parsed.append(
            MegaIntensionalityRow(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                _probability(row[9], "response", number),
                _integer(row[10], "sanity", number),
            )
        )

    return MegaAttitudeFile(MEGAINTENSIONALITY_NAME, tuple(parsed))


def _capitalized_clause(text: str) -> str:
    """Return one embedded clause as a sentence."""
    clause = text.strip().rstrip(".")

    if not clause:
        msg = "a MegaVeridicality antecedent has no embedded clause"
        raise MalformedMegaAttitudeError(msg)

    return f"{clause[0].upper()}{clause[1:]}."


def _veridicality_consequent(row: MegaVeridicalityRow) -> str:
    """Generate the release's bleached consequent from its antecedent."""
    if "that S" in row.frame:
        marker = " that "

        if marker not in row.antecedent:
            msg = f"the antecedent for frame {row.frame!r} has no {marker!r}"
            raise MalformedMegaAttitudeError(msg)

        return _capitalized_clause(row.antecedent.split(marker, maxsplit=1)[1])

    if row.frame == "NP Ved for NP to VP":
        if " for a particular thing to happen" not in row.antecedent:
            msg = f"the antecedent for frame {row.frame!r} has no generated clause"
            raise MalformedMegaAttitudeError(msg)

        return "A particular thing happened."

    if "to VP[+eventive]" in row.frame:
        event = "do a particular thing"
        realized = "did a particular thing"
    elif "to VP[-eventive]" in row.frame:
        event = "have a particular thing"
        realized = "had a particular thing"
    else:
        msg = f"unsupported MegaVeridicality frame {row.frame!r}"
        raise MalformedMegaAttitudeError(msg)

    if f" to {event}" not in row.antecedent:
        msg = f"the antecedent for frame {row.frame!r} has no generated clause"
        raise MalformedMegaAttitudeError(msg)

    return f"A particular person {realized}."


def _pair_of(row: MegaAttitudeRow, spec: MegaAttitudeSpec) -> _Pair:
    """Return the inference pair, response and factors of a source row."""
    if isinstance(row, MegaVeridicalityRow):
        return _Pair(
            row.participant,
            row.antecedent,
            _veridicality_consequent(row),
            None if row.response is None else _relation_for(row.response, spec),
            (
                ("verb", row.verb),
                ("frame", row.frame),
                ("polarity", row.polarity),
                ("conditional", row.conditional),
            ),
        )

    if isinstance(row, MegaNegRaisingRow):
        return _Pair(
            row.participant,
            row.antecedent,
            row.consequent,
            row.response,
            (
                ("verb", row.verb),
                ("frame", row.frame),
                ("tense", row.tense),
                ("subject", row.subject),
            ),
        )

    return _Pair(
        row.participant,
        row.antecedent,
        row.consequent.strip(),
        row.response,
        (
            ("verb", row.verb),
            ("frame", row.frame),
            ("polarity", row.polarity),
            ("target", row.target),
            ("consequent_verb", row.consequent_verb),
            ("consequent_tense", row.consequent_tense),
        ),
    )


def _fields_of(row: MegaAttitudeRow) -> tuple[str, ...]:
    """Return a source row in release-column order."""
    if isinstance(row, MegaVeridicalityRow):
        return (
            row.participant,
            row.list_id,
            str(row.presentation_order),
            row.verb,
            row.frame,
            row.polarity,
            row.conditional,
            row.antecedent,
            row.response or "",
            str(row.acceptability),
            row.native_english,
        )

    if isinstance(row, MegaNegRaisingRow):
        return (
            row.participant,
            row.list_id,
            str(row.presentation_order),
            row.verb,
            row.frame,
            row.tense,
            row.subject,
            row.antecedent,
            row.consequent,
            format(_decimal_value(row.response), "f"),
            format(_decimal_value(row.acceptability), "f"),
            row.native_english,
        )

    return (
        row.participant,
        row.verb,
        row.polarity,
        row.target,
        row.frame,
        row.consequent_verb,
        row.consequent_tense,
        row.antecedent,
        row.consequent,
        format(_decimal_value(row.response), "f"),
        str(row.sanity),
    )


def _header(format_: MegaAttitudeFormat) -> tuple[str, ...]:
    """Return the exact header of a release format."""
    if format_ == MEGAVERIDICALITY_NAME:
        return _VERIDICALITY_HEADER
    if format_ == MEGANEGRAISING_NAME:
        return _NEG_RAISING_HEADER
    return _INTENSIONALITY_HEADER


def _write(source: MegaAttitudeFile) -> str:
    """Return one canonical release table."""
    destination = StringIO(newline="")
    writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
    writer.writerow(_header(source.format))

    for row in source.rows:
        writer.writerow(_fields_of(row))

    return destination.getvalue()


def _digest(*parts: str) -> str:
    """Return a stable source-derived identifier."""
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return sha256(value.encode()).hexdigest()[:24]


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


def _licensing(spec: MegaAttitudeSpec) -> defs_records.Licensing:
    """Return the release's declared license."""
    return defs_records.Licensing(
        expression=spec.spdx,
        licenses=(defs_records.LicenseRef(spdx=spec.spdx, appliesTo="whole"),),
    )


def _record(nsid: str, key: str, value: _RecordModel) -> FragmentRecord:
    """Return one serialized Layers record."""
    return FragmentRecord(
        local_id=_uri(nsid, key), nsid=nsid, value_json=value.model_dump_json()
    )


def _refs(pair: _Pair, spec: MegaAttitudeSpec) -> _PairRefs:
    """Return stable references for an antecedent-consequent pair."""
    stem = _digest(spec.source_id, spec.release_id, pair.antecedent, pair.consequent)
    return _PairRefs(
        _uri(_EXPRESSION_NSID, f"{stem}-antecedent"),
        _uri(_EXPRESSION_NSID, f"{stem}-consequent"),
        _uri(_LAYER_NSID, f"{stem}-relation"),
    )


def _expression(text: str, key: str, spec: MegaAttitudeSpec) -> FragmentRecord:
    """Return one sentence expression."""
    return _record(
        _EXPRESSION_NSID,
        key,
        expression_records.Expression(
            id=key,
            kind="sentence",
            createdAt=_EPOCH,
            text=text,
            languages=(spec.language,),
            eprintRefs=spec.eprint_refs,
        ),
    )


def _layer(refs: _PairRefs, spec: MegaAttitudeSpec) -> FragmentRecord:
    """Return the elicited relation carrier for one pair."""
    key = refs.layer.rsplit("/", maxsplit=1)[-1]
    return _record(
        _LAYER_NSID,
        key,
        annotation_records.AnnotationLayer(
            createdAt=_EPOCH,
            expression=refs.antecedent,
            kind="relation",
            annotations=(
                annotation_records.Annotation(
                    uuid=defs_records.Uuid(value=_RELATION_OBJECT),
                    arguments=(
                        annotation_records.ArgumentRef(
                            role="hypothesis",
                            target=defs_records.ObjectRef(recordRef=refs.consequent),
                        ),
                    ),
                ),
            ),
            sourceMethod=spec.source_method,
            licensing=_licensing(spec),
            eprintRefs=spec.eprint_refs,
        ),
    )


def _experiment(source: MegaAttitudeFile, spec: MegaAttitudeSpec) -> FragmentRecord:
    """Return the release's inference experiment."""
    continuous = source.format != MEGAVERIDICALITY_NAME
    key = _digest(spec.source_id, spec.release_id, "experiment")
    pairs = tuple(_pair_of(row, spec) for row in source.rows)
    refs = tuple(reference for pair in pairs for reference in _refs(pair, spec)[:2])
    return _record(
        _EXPERIMENT_NSID,
        key,
        judgment_records.ExperimentDef(
            name=f"{source.format} {spec.release_id}",
            createdAt=_EPOCH,
            taskType="continuous" if continuous else "categorical",
            measureType="inference",
            labels=()
            if continuous
            else tuple(declared.relation for declared in spec.veridicality_relations),
            scaleMin=0 if continuous else None,
            scaleMax=1 if continuous else None,
            languages=(spec.language,),
            licensing=_licensing(spec),
            eprintRefs=spec.eprint_refs,
            stimulusExpressionRefs=tuple(dict.fromkeys(refs)),
        ),
    )


def _row_json(row: MegaAttitudeRow) -> str:
    """Return the lossless non-response payload of a source row."""
    return json.dumps(_fields_of(row), ensure_ascii=False, separators=(",", ":"))


def _judgment(
    row: MegaAttitudeRow,
    ordinal: int,
    refs: _PairRefs,
    spec: MegaAttitudeSpec,
) -> judgment_records.Judgment:
    """Return one typed participant response."""
    pair = _pair_of(row, spec)
    fields = ((_ROW_ORDINAL, str(ordinal)), (_ROW_JSON, _row_json(row)), *pair.factors)
    item = defs_records.ObjectRef(
        recordRef=refs.layer, objectId=defs_records.Uuid(value=_RELATION_OBJECT)
    )

    if isinstance(pair.response, str):
        return judgment_records.Judgment(
            item=item,
            categoricalValue=pair.response,
            features=_features(fields),
        )

    if pair.response is None:
        return judgment_records.Judgment(item=item, features=_features(fields))

    return judgment_records.Judgment(
        item=item,
        continuousValue=judgment_records.DecimalValue(
            coefficient=pair.response.coefficient,
            exponent=pair.response.exponent,
        ),
        features=_features(fields),
    )


def _records(
    source: MegaAttitudeFile, spec: MegaAttitudeSpec
) -> tuple[FragmentRecord, ...]:
    """Return the canonical Layers fragment for one source table."""
    experiment = _experiment(source, spec)
    records: list[FragmentRecord] = [experiment]
    seen: set[str] = set()

    for row in source.rows:
        pair = _pair_of(row, spec)
        refs = _refs(pair, spec)

        if refs.layer in seen:
            continue

        seen.add(refs.layer)
        records.extend(
            (
                _expression(
                    pair.antecedent,
                    refs.antecedent.rsplit("/", maxsplit=1)[-1],
                    spec,
                ),
                _expression(
                    pair.consequent,
                    refs.consequent.rsplit("/", maxsplit=1)[-1],
                    spec,
                ),
                _layer(refs, spec),
            )
        )

    grouped: dict[str, list[tuple[int, MegaAttitudeRow]]] = {}

    for ordinal, row in enumerate(source.rows):
        grouped.setdefault(_pair_of(row, spec).participant, []).append((ordinal, row))

    for participant, held in grouped.items():
        key = _digest(spec.source_id, spec.release_id, "participant", participant)
        judgments = tuple(
            _judgment(row, ordinal, _refs(_pair_of(row, spec), spec), spec)
            for ordinal, row in held
        )
        records.append(
            _record(
                _JUDGMENT_SET_NSID,
                key,
                judgment_records.JudgmentSet(
                    experimentRef=experiment.local_id,
                    createdAt=_EPOCH,
                    agent=defs_records.AgentRef(id=participant),
                    judgments=judgments,
                ),
            )
        )

    return tuple(records)


def _model(record: FragmentRecord) -> _RecordModel:
    """Return a typed record from a fragment entry."""
    models = {
        _EXPRESSION_NSID: expression_records.Expression,
        _LAYER_NSID: annotation_records.AnnotationLayer,
        _EXPERIMENT_NSID: judgment_records.ExperimentDef,
        _JUDGMENT_SET_NSID: judgment_records.JudgmentSet,
    }
    selected = models.get(record.nsid)

    if selected is None:
        msg = f"a MegaAttitude fragment contains {record.nsid!r}"
        raise MalformedMegaAttitudeError(msg)

    return selected.model_validate_json(record.value_json)


def _feature_map(features: defs_records.FeatureMap | None) -> Mapping[str, str]:
    """Return a feature map with no repeated keys."""
    if features is None:
        return {}

    values = {entry.key: entry.value for entry in features.entries}

    if len(values) != len(features.entries):
        msg = "a MegaAttitude response repeats a feature key"
        raise MalformedMegaAttitudeError(msg)

    return values


def _response_text(
    judgment: judgment_records.Judgment,
    format_: MegaAttitudeFormat,
    spec: MegaAttitudeSpec,
) -> str:
    """Return the source spelling of a typed response."""
    if format_ == MEGAVERIDICALITY_NAME:
        relation = judgment.categoricalValue

        if relation is None:
            return ""

        return _response_for(relation, spec)

    value = judgment.continuousValue

    if value is None:
        msg = "a MegaAttitude likelihood response has no exact decimal"
        raise MalformedMegaAttitudeError(msg)

    return format(Decimal(f"{value.coefficient}e{value.exponent}"), "f")


class MegaAttitudeIso:
    """Translate one released table format to Layers records."""

    __slots__ = ("_format", "_spec")

    def __init__(self, format_: MegaAttitudeFormat, spec: MegaAttitudeSpec) -> None:
        self._format = format_
        self._spec = spec

    def forward(self, source: MegaAttitudeFile, /) -> CorpusFragment:
        """Return Layers records for one table."""
        if source.format != self._format:
            msg = f"a {self._format} codec cannot encode {source.format}"
            raise MalformedMegaAttitudeError(msg)

        return CorpusFragment(records=_records(source, self._spec), source=self._format)

    def backward(self, target: CorpusFragment, /) -> MegaAttitudeFile:
        """Return the table represented by one canonical fragment."""
        return _source_of_with_spec(target.records, self._format, self._spec)


def _source_of_with_spec(
    records: tuple[FragmentRecord, ...],
    format_: MegaAttitudeFormat,
    spec: MegaAttitudeSpec,
) -> MegaAttitudeFile:
    """Return and canonically validate a fragment under its declaration."""
    indexed: list[tuple[int, list[str]]] = []

    for record in records:
        model = _model(record)

        if not isinstance(model, judgment_records.JudgmentSet):
            continue

        for judgment in model.judgments:
            features = _feature_map(judgment.features)

            try:
                ordinal = int(features[_ROW_ORDINAL])
                fields = json.loads(features[_ROW_JSON])
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                msg = "a MegaAttitude response lacks its source row"
                raise MalformedMegaAttitudeError(msg) from error

            if not isinstance(fields, list) or not all(
                isinstance(value, str) for value in fields
            ):
                msg = "a MegaAttitude response has a malformed source row"
                raise MalformedMegaAttitudeError(msg)

            response_index = 8 if format_ == MEGAVERIDICALITY_NAME else 9
            fields[response_index] = _response_text(judgment, format_, spec)
            indexed.append((ordinal, fields))

    indexed.sort(key=lambda entry: entry[0])

    if tuple(ordinal for ordinal, _ in indexed) != tuple(range(len(indexed))):
        msg = "MegaAttitude response ordinals are not contiguous from zero"
        raise MalformedMegaAttitudeError(msg)

    destination = StringIO(newline="")
    writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
    writer.writerow(_header(format_))
    writer.writerows(fields for _, fields in indexed)
    text = destination.getvalue()

    if format_ == MEGAVERIDICALITY_NAME:
        source = _veridicality_file(text, spec)
    elif format_ == MEGANEGRAISING_NAME:
        source = _negraising_file(text)
    else:
        source = _intensionality_file(text)

    if _records(source, spec) != records:
        msg = "MegaAttitude records are not in canonical form"
        raise MalformedMegaAttitudeError(msg)

    return source


class _MegaAttitudeCodec:
    """Shared implementation of one concrete release codec."""

    __slots__ = ("_iso", "_parse", "name")

    name: str

    def __init__(
        self,
        name: MegaAttitudeFormat,
        spec: MegaAttitudeSpec,
        parse: Callable[[str | bytes], MegaAttitudeFile],
    ) -> None:
        self.name = name
        self._parse = parse
        self._iso = MegaAttitudeIso(name, spec)

    @property
    def iso(self) -> Iso[MegaAttitudeFile, CorpusFragment]:
        """Return the table-to-record translation."""
        return self._iso

    def decode(
        self, src: str | bytes, *, into: CorpusFragment | None = None
    ) -> CorpusFragment:
        """Return the Layers records in one released table."""
        decoded = self._iso.forward(self._parse(src))

        if into is None:
            return decoded

        return CorpusFragment(records=into.records + decoded.records, source=self.name)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return the released TSV represented by Layers records."""
        fragment = CorpusFragment(records=tuple(records), source=self.name)
        return _write(self._iso.backward(fragment))


@registers_as(MEGAVERIDICALITY_NAME)
class MegaVeridicalityCodec(_MegaAttitudeCodec):
    """Codec for the participant-level MegaVeridicality v2.1 table."""

    @classmethod
    def from_declaration(cls, declaration: bytes, /) -> Self:
        """Return the codec a release declaration document configures."""
        return cls(MegaAttitudeSpec.model_validate_json(declaration))

    def __init__(self, spec: MegaAttitudeSpec) -> None:
        super().__init__(
            MEGAVERIDICALITY_NAME,
            spec,
            lambda source: _veridicality_file(source, spec),
        )


@registers_as(MEGANEGRAISING_NAME)
class MegaNegRaisingCodec(_MegaAttitudeCodec):
    """Codec for the participant-level MegaNegRaising v1.1 table."""

    @classmethod
    def from_declaration(cls, declaration: bytes, /) -> Self:
        """Return the codec a release declaration document configures."""
        return cls(MegaAttitudeSpec.model_validate_json(declaration))

    def __init__(self, spec: MegaAttitudeSpec) -> None:
        super().__init__(MEGANEGRAISING_NAME, spec, _negraising_file)


@registers_as(MEGAINTENSIONALITY_NAME)
class MegaIntensionalityCodec(_MegaAttitudeCodec):
    """Codec for the participant-level MegaIntensionality v1 table."""

    @classmethod
    def from_declaration(cls, declaration: bytes, /) -> Self:
        """Return the codec a release declaration document configures."""
        return cls(MegaAttitudeSpec.model_validate_json(declaration))

    def __init__(self, spec: MegaAttitudeSpec) -> None:
        super().__init__(MEGAINTENSIONALITY_NAME, spec, _intensionality_file)

"""Conversion from normalized MegaAcceptability tables to Layers records."""

import csv
import io
import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import resource as resource_records

from lofi.data.corpora.megaacceptability.spec import MegaAcceptabilitySpec
from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.data import Iso

__all__ = [
    "NAME",
    "MalformedMegaAcceptabilityError",
    "MegaAcceptabilityCodec",
    "MegaAcceptabilityIso",
    "MegaAcceptabilityRow",
    "MegaAcceptabilityTable",
    "TableVersion",
    "parse_table",
]

type TableVersion = Literal["v1", "v2"]

NAME: Final = "megaacceptability"
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_SCHEME: Final = "at://local"
_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
_FILLING_NSID: Final = "pub.layers.resource.filling"
_EXPERIMENT_NSID: Final = "pub.layers.judgment.experimentDef"
_JUDGMENT_SET_NSID: Final = "pub.layers.judgment.judgmentSet"
_HEADER_V1: Final = (
    "participant",
    "list",
    "presentationorder",
    "verb",
    "frame",
    "response",
    "nativeenglish",
    "sentence",
)
_HEADER_V2: Final = (
    "participant",
    "list",
    "presentationorder",
    "verb",
    "frame",
    "tense",
    "response",
    "nativeenglish",
    "sentence",
    "version",
)
_ROW_KEYS: Final = frozenset(
    {
        "row_ordinal",
        "list",
        "presentationorder",
        "verb",
        "frame",
        "nativeenglish",
        "sentence",
    }
)
_SOURCE_ID: Final = "source_id"
_RELEASE_ID: Final = "release_id"
_TABLE_VERSION: Final = "table_version"
_RESPONSE_NA: Final = "response_na"


class MalformedMegaAcceptabilityError(ValueError):
    """An error for malformed MegaAcceptability tables or fragments."""


class MegaAcceptabilityRow(NamedTuple):
    """A normalized response row."""

    participant: str
    list_id: str
    presentation_order: int
    verb: str
    frame: str
    response: int | None
    native_english: str
    sentence: str
    tense: str | None = None
    version: str | None = None


class MegaAcceptabilityTable(NamedTuple):
    """A normalized v1 or v2 table."""

    version: TableVersion
    rows: tuple[MegaAcceptabilityRow, ...]


class _Stimulus(NamedTuple):
    """The records shared by responses to one stimulus."""

    expression_ref: str
    filling_ref: str
    row: MegaAcceptabilityRow


class _FragmentModels(NamedTuple):
    """Models recovered from one fragment."""

    experiment: judgment_records.ExperimentDef
    response_sets: tuple[judgment_records.JudgmentSet, ...]


def _text_of(src: str | bytes) -> str:
    """Return UTF-8 table text."""
    if isinstance(src, str):
        return src

    try:
        return src.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "MegaAcceptability input is not UTF-8"
        raise MalformedMegaAcceptabilityError(msg) from error


def _integer(value: str, field: str, row: int) -> int:
    """Return one integer field."""
    try:
        return int(value)
    except ValueError as error:
        msg = f"row {row} has noninteger {field} {value!r}"
        raise MalformedMegaAcceptabilityError(msg) from error


def _missing(version: TableVersion) -> str:
    """Return how a release version spells a missing response.

    Version 1 tables write ``NA``; version 2 tables leave the field empty.
    """
    return "NA" if version == "v1" else ""


def _response(value: str, version: TableVersion, row: int) -> int:
    """Return a response in the release version's exact spelling."""
    if version == "v1":
        return _integer(value, "response", row)

    whole, separator, fraction = value.partition(".")
    if separator != "." or fraction != "0":
        msg = f"row {row} has noninteger-valued v2 response {value!r}"
        raise MalformedMegaAcceptabilityError(msg)

    return _integer(whole, "response", row)


def _row_of(
    values: list[str], version: TableVersion, ordinal: int
) -> MegaAcceptabilityRow:
    """Return one parsed table row."""
    header = _HEADER_V1 if version == "v1" else _HEADER_V2

    if len(values) != len(header):
        msg = f"row {ordinal + 2} has {len(values)} fields rather than {len(header)}"
        raise MalformedMegaAcceptabilityError(msg)

    fields = dict(zip(header, values, strict=True))
    required = tuple(name for name in header if name != "response")

    if any(not fields[name].strip() for name in required):
        msg = f"row {ordinal + 2} has an empty required field"
        raise MalformedMegaAcceptabilityError(msg)

    response_text = fields["response"]
    response = (
        None
        if response_text == _missing(version)
        else _response(response_text, version, ordinal + 2)
    )

    return MegaAcceptabilityRow(
        participant=fields["participant"],
        list_id=fields["list"],
        presentation_order=_integer(
            fields["presentationorder"], "presentationorder", ordinal + 2
        ),
        verb=fields["verb"],
        frame=fields["frame"],
        response=response,
        native_english=fields["nativeenglish"],
        sentence=fields["sentence"],
        tense=None if version == "v1" else fields["tense"],
        version=None if version == "v1" else fields["version"],
    )


def parse_table(src: str | bytes) -> MegaAcceptabilityTable:
    """Parse one normalized v1 or v2 table.

    Raises
    ------
    MalformedMegaAcceptabilityError
        If the header is neither release's, or a row is malformed.
    """
    reader = csv.reader(io.StringIO(_text_of(src), newline=""), delimiter="\t")

    try:
        header = tuple(next(reader))
    except StopIteration as error:
        msg = "MegaAcceptability input has no header"
        raise MalformedMegaAcceptabilityError(msg) from error

    if header == _HEADER_V1:
        version: TableVersion = "v1"
    elif header == _HEADER_V2:
        version = "v2"
    else:
        msg = f"MegaAcceptability input has unsupported header {header!r}"
        raise MalformedMegaAcceptabilityError(msg)

    rows = tuple(
        _row_of(values, version, ordinal) for ordinal, values in enumerate(reader)
    )

    if not rows:
        msg = "MegaAcceptability input has no response rows"
        raise MalformedMegaAcceptabilityError(msg)

    return MegaAcceptabilityTable(version, rows)


def _fields_of(row: MegaAcceptabilityRow, version: TableVersion) -> tuple[str, ...]:
    """Return one row in table-column order."""
    common = (
        row.participant,
        row.list_id,
        str(row.presentation_order),
        row.verb,
        row.frame,
    )
    response = _missing("v1") if row.response is None else str(row.response)

    if version == "v1":
        if row.tense is not None or row.version is not None:
            msg = "a v1 row has v2-only fields"
            raise MalformedMegaAcceptabilityError(msg)

        return (*common, response, row.native_english, row.sentence)

    if row.tense is None or row.version is None:
        msg = "a v2 row lacks tense or version"
        raise MalformedMegaAcceptabilityError(msg)

    rendered = _missing("v2") if row.response is None else f"{row.response}.0"
    return (*common, row.tense, rendered, row.native_english, row.sentence, row.version)


def _write(table: MegaAcceptabilityTable) -> str:
    """Write one normalized table."""
    destination = io.StringIO(newline="")
    writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
    writer.writerow(_HEADER_V1 if table.version == "v1" else _HEADER_V2)

    for row in table.rows:
        writer.writerow(_fields_of(row, table.version))

    return destination.getvalue()


def _digest(*parts: str) -> str:
    """Return a stable key for source fields."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


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


def _record(
    nsid: str,
    key: str,
    value: expression_records.Expression
    | resource_records.Filling
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet,
) -> FragmentRecord:
    """Return one serialized Layers record."""
    return FragmentRecord(
        local_id=_uri(nsid, key),
        nsid=nsid,
        value_json=value.model_dump_json(),
    )


def _stimulus_of(row: MegaAcceptabilityRow, spec: MegaAcceptabilitySpec) -> _Stimulus:
    """Return stable record references for one stimulus."""
    key = _digest(
        spec.source_id,
        spec.release_id,
        row.verb,
        row.frame,
        row.tense or "",
        row.sentence,
    )
    return _Stimulus(
        _uri(_EXPRESSION_NSID, key),
        _uri(_FILLING_NSID, key),
        row,
    )


def _resources(
    stimulus: _Stimulus, spec: MegaAcceptabilitySpec
) -> tuple[FragmentRecord, FragmentRecord]:
    """Return the expression and filling for one stimulus."""
    row = stimulus.row

    try:
        template_ref = spec.frame_refs[row.frame]
    except KeyError as error:
        msg = f"no template reference is declared for frame {row.frame!r}"
        raise MalformedMegaAcceptabilityError(msg) from error

    try:
        entry_ref = spec.verb_refs[row.verb]
    except KeyError as error:
        msg = f"no entry reference is declared for verb {row.verb!r}"
        raise MalformedMegaAcceptabilityError(msg) from error

    key = stimulus.expression_ref.rsplit("/", maxsplit=1)[-1]
    shared_features = _features(
        ((_SOURCE_ID, spec.source_id), (_RELEASE_ID, spec.release_id))
    )
    expression = expression_records.Expression(
        id=key,
        kind="sentence",
        createdAt=_EPOCH,
        text=row.sentence,
        languages=("en",),
        eprintRefs=spec.eprint_refs,
        features=shared_features,
    )
    filling = resource_records.Filling(
        createdAt=_EPOCH,
        expressionRef=stimulus.expression_ref,
        templateRef=template_ref,
        slotFillings=(
            resource_records.SlotFilling(
                slotName="verb",
                entryRef=entry_ref,
                renderedForm=row.verb,
            ),
        ),
        renderedText=row.sentence,
        strategy="manual",
        features=shared_features,
    )

    return (
        _record(_EXPRESSION_NSID, key, expression),
        _record(_FILLING_NSID, key, filling),
    )


def _row_features(
    row: MegaAcceptabilityRow, ordinal: int, version: TableVersion
) -> defs_records.FeatureMap:
    """Return the columns stored beside one judgment."""
    entries = [
        ("row_ordinal", str(ordinal)),
        ("list", row.list_id),
        ("presentationorder", str(row.presentation_order)),
        ("verb", row.verb),
        ("frame", row.frame),
        ("nativeenglish", row.native_english),
        ("sentence", row.sentence),
    ]

    if version == "v2":
        entries.extend((("tense", row.tense or ""), ("version", row.version or "")))

    if row.response is None:
        entries.append((_RESPONSE_NA, "true"))

    return _features(entries)


def _experiment(
    table: MegaAcceptabilityTable,
    spec: MegaAcceptabilitySpec,
    expressions: tuple[str, ...],
) -> tuple[str, FragmentRecord]:
    """Return the experiment definition and its reference."""
    key = _digest(spec.source_id, spec.release_id, "experiment")
    licensing = defs_records.Licensing(
        expression=spec.spdx,
        licenses=(defs_records.LicenseRef(spdx=spec.spdx, appliesTo="whole"),),
    )
    record = judgment_records.ExperimentDef(
        name=f"MegaAcceptability {spec.release_id}",
        createdAt=_EPOCH,
        taskType="ordinal-scale",
        measureType="acceptability",
        scaleMin=spec.scale_min,
        scaleMax=spec.scale_max,
        languages=("en",),
        licensing=licensing,
        eprintRefs=spec.eprint_refs,
        templateRefs=tuple(
            dict.fromkeys(spec.frame_refs[row.frame] for row in table.rows)
        ),
        stimulusExpressionRefs=expressions,
        features=_features(
            (
                (_SOURCE_ID, spec.source_id),
                (_RELEASE_ID, spec.release_id),
                (_TABLE_VERSION, table.version),
            )
        ),
    )
    return _uri(_EXPERIMENT_NSID, key), _record(_EXPERIMENT_NSID, key, record)


def _records_of(
    table: MegaAcceptabilityTable, spec: MegaAcceptabilitySpec
) -> tuple[FragmentRecord, ...]:
    """Return the Layers records for one table.

    A participant may rate one stimulus more than once; each rating is its own
    judgment, kept apart by its row ordinal.
    """
    stimuli: dict[str, _Stimulus] = {}
    participants: dict[str, list[tuple[int, MegaAcceptabilityRow, _Stimulus]]] = {}

    for ordinal, row in enumerate(table.rows):
        if (
            row.response is not None
            and not spec.scale_min <= row.response <= spec.scale_max
        ):
            msg = (
                f"row {ordinal + 2} has response {row.response} outside "
                "the declared scale"
            )
            raise MalformedMegaAcceptabilityError(msg)

        stimulus = _stimulus_of(row, spec)
        held = stimuli.get(stimulus.expression_ref)

        if held is not None and held.row.sentence != row.sentence:
            msg = f"row {ordinal + 2} collides with another stimulus"
            raise MalformedMegaAcceptabilityError(msg)

        stimuli.setdefault(stimulus.expression_ref, stimulus)
        participants.setdefault(row.participant, []).append((ordinal, row, stimulus))

    expressions = tuple(stimulus.expression_ref for stimulus in stimuli.values())
    experiment_ref, experiment_record = _experiment(table, spec, expressions)
    records: list[FragmentRecord] = [experiment_record]

    for stimulus in stimuli.values():
        records.extend(_resources(stimulus, spec))

    for participant, held in participants.items():
        judgments = tuple(
            judgment_records.Judgment(
                item=defs_records.ObjectRef(recordRef=stimulus.expression_ref),
                scalarValue=row.response,
                fillingRef=stimulus.filling_ref,
                features=_row_features(row, ordinal, table.version),
            )
            for ordinal, row, stimulus in held
        )
        key = _digest(spec.source_id, spec.release_id, "participant", participant)
        records.append(
            _record(
                _JUDGMENT_SET_NSID,
                key,
                judgment_records.JudgmentSet(
                    experimentRef=experiment_ref,
                    createdAt=_EPOCH,
                    agent=defs_records.AgentRef(id=participant),
                    judgments=judgments,
                ),
            )
        )

    return tuple(records)


def _feature_values(features: defs_records.FeatureMap | None) -> Mapping[str, str]:
    """Return one feature map without duplicate keys."""
    if features is None:
        return {}

    values = {entry.key: entry.value for entry in features.entries}

    if len(values) != len(features.entries):
        msg = "a MegaAcceptability record repeats a feature key"
        raise MalformedMegaAcceptabilityError(msg)

    return values


def _required(values: Mapping[str, str], key: str) -> str:
    """Return one required row feature."""
    value = values.get(key)

    if value is None:
        msg = f"a MegaAcceptability judgment lacks {key!r}"
        raise MalformedMegaAcceptabilityError(msg)

    return value


def _fragment_models(records: tuple[FragmentRecord, ...]) -> _FragmentModels:
    """Validate and collect the models in one fragment."""
    experiments: list[judgment_records.ExperimentDef] = []
    response_sets: list[judgment_records.JudgmentSet] = []

    for record in records:
        if record.nsid == _EXPERIMENT_NSID:
            experiments.append(
                judgment_records.ExperimentDef.model_validate_json(record.value_json)
            )
        elif record.nsid == _JUDGMENT_SET_NSID:
            response_sets.append(
                judgment_records.JudgmentSet.model_validate_json(record.value_json)
            )
        elif record.nsid == _EXPRESSION_NSID:
            expression_records.Expression.model_validate_json(record.value_json)
        elif record.nsid == _FILLING_NSID:
            resource_records.Filling.model_validate_json(record.value_json)
        else:
            msg = f"a MegaAcceptability fragment contains {record.nsid!r}"
            raise MalformedMegaAcceptabilityError(msg)

    if len(experiments) != 1 or not response_sets:
        msg = (
            "a MegaAcceptability fragment needs one experiment and "
            "participant responses"
        )
        raise MalformedMegaAcceptabilityError(msg)

    return _FragmentModels(experiments[0], tuple(response_sets))


def _version_of(experiment: judgment_records.ExperimentDef) -> TableVersion:
    """Return the normalized table version declared by an experiment."""
    stated = _required(_feature_values(experiment.features), _TABLE_VERSION)

    if stated not in {"v1", "v2"}:
        msg = f"a MegaAcceptability experiment declares table version {stated!r}"
        raise MalformedMegaAcceptabilityError(msg)

    return "v1" if stated == "v1" else "v2"


def _row_feature_keys(version: TableVersion, *, missing: bool) -> set[str]:
    """Return the exact feature keys for one response row."""
    expected = set(_ROW_KEYS)

    if version == "v2":
        expected.update(("tense", "version"))

    if missing:
        expected.add(_RESPONSE_NA)

    return expected


def _decoded_row(
    judgment: judgment_records.Judgment,
    participant: str,
    version: TableVersion,
) -> tuple[int, MegaAcceptabilityRow]:
    """Return one response row and its source ordinal."""
    values = _feature_values(judgment.features)
    missing = judgment.scalarValue is None

    if missing and values.get(_RESPONSE_NA) != "true":
        msg = "a missing response lacks response_na=true"
        raise MalformedMegaAcceptabilityError(msg)

    if not missing and _RESPONSE_NA in values:
        msg = "a present response is marked response_na"
        raise MalformedMegaAcceptabilityError(msg)

    if set(values) != _row_feature_keys(version, missing=missing):
        msg = "a MegaAcceptability judgment has unsupported row features"
        raise MalformedMegaAcceptabilityError(msg)

    ordinal = _integer(_required(values, "row_ordinal"), "row_ordinal", 0)
    row = MegaAcceptabilityRow(
        participant=participant,
        list_id=_required(values, "list"),
        presentation_order=_integer(
            _required(values, "presentationorder"),
            "presentationorder",
            ordinal + 2,
        ),
        verb=_required(values, "verb"),
        frame=_required(values, "frame"),
        response=judgment.scalarValue,
        native_english=_required(values, "nativeenglish"),
        sentence=_required(values, "sentence"),
        tense=None if version == "v1" else _required(values, "tense"),
        version=None if version == "v1" else _required(values, "version"),
    )
    return ordinal, row


def _rows_of(
    response_set: judgment_records.JudgmentSet, version: TableVersion
) -> tuple[tuple[int, MegaAcceptabilityRow], ...]:
    """Return the response rows in one participant's set."""
    participant = None if response_set.agent is None else response_set.agent.id

    if participant is None:
        msg = "a MegaAcceptability judgment set has no participant agent"
        raise MalformedMegaAcceptabilityError(msg)

    return tuple(
        _decoded_row(judgment, participant, version)
        for judgment in response_set.judgments
    )


def _table_of(
    records: tuple[FragmentRecord, ...], spec: MegaAcceptabilitySpec
) -> MegaAcceptabilityTable:
    """Return the normalized table encoded by Layers records."""
    models = _fragment_models(records)
    version = _version_of(models.experiment)
    indexed = [
        row
        for response_set in models.response_sets
        for row in _rows_of(response_set, version)
    ]

    indexed.sort(key=lambda held: held[0])

    if tuple(ordinal for ordinal, _ in indexed) != tuple(range(len(indexed))):
        msg = "MegaAcceptability row ordinals are not contiguous from zero"
        raise MalformedMegaAcceptabilityError(msg)

    table = MegaAcceptabilityTable(version, tuple(row for _, row in indexed))

    if _records_of(table, spec) != records:
        msg = "MegaAcceptability records are not in the codec's canonical form"
        raise MalformedMegaAcceptabilityError(msg)

    return table


class MegaAcceptabilityIso:
    """A translation from normalized tables to Layers fragments."""

    __slots__ = ("_spec",)

    def __init__(self, spec: MegaAcceptabilitySpec) -> None:
        self._spec = spec

    def forward(self, source: MegaAcceptabilityTable, /) -> CorpusFragment:
        """Return the Layers fragment for one table."""
        return CorpusFragment(records=_records_of(source, self._spec), source=NAME)

    def backward(self, target: CorpusFragment, /) -> MegaAcceptabilityTable:
        """Return the normalized table for one fragment."""
        return _table_of(target.records, self._spec)


@registers_as(NAME)
class MegaAcceptabilityCodec:
    """A codec for normalized English MegaAcceptability v1/v2 tables."""

    __slots__ = ("_iso", "name")

    name: str

    def __init__(self, spec: MegaAcceptabilitySpec) -> None:
        self.name = NAME
        self._iso = MegaAcceptabilityIso(spec)

    @classmethod
    def from_declaration(cls, declaration: bytes, /) -> Self:
        """Return the codec a release declaration document configures."""
        return cls(MegaAcceptabilitySpec.model_validate_json(declaration))

    @property
    def iso(self) -> Iso[MegaAcceptabilityTable, CorpusFragment]:
        """Return the table-to-record translation."""
        return self._iso

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return the Layers records in one normalized table."""
        decoded = self._iso.forward(parse_table(src))

        if into is None:
            return decoded

        return CorpusFragment(records=into.records + decoded.records, source=NAME)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return normalized TSV for Layers records."""
        fragment = CorpusFragment(records=tuple(records), source=NAME)
        return _write(self._iso.backward(fragment))

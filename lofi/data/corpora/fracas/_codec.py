"""Conversion from the FraCaS test suite's XML to Layers relation records."""

from __future__ import annotations

import enum
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Final, NamedTuple

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import segmentation as segmentation_records

from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import JsonValue
    from lofi.data import CorpusRelease, Iso

__all__ = [
    "FRACAS_NAME",
    "FracasAnswer",
    "FracasCodec",
    "FracasFile",
    "FracasIso",
    "FracasProblem",
    "FracasSection",
    "MalformedFracasError",
]

FRACAS_NAME: Final = "fracas"

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_SCHEME: Final = "at://local"
_CORPUS_NSID: Final = "pub.layers.corpus.corpus"
_MEMBERSHIP_NSID: Final = "pub.layers.corpus.membership"
_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
_SEGMENTATION_NSID: Final = "pub.layers.segmentation.segmentation"
_LAYER_NSID: Final = "pub.layers.annotation.annotationLayer"
_PROBLEM_JSON: Final = "problem_json"
_PROBLEM_ORDINAL: Final = "problem_ordinal"
_SECTION: Final = "section"
_SUBSECTION: Final = "subsection"
_FORMAT: Final = "format"
_PROBLEM_OBJECT: Final = "problem"
_HYPOTHESIS_ROLE: Final = "hypothesis"
_WORD: Final = re.compile(r"\S+")

type _LayerModel = (
    corpus_records.Corpus
    | corpus_records.Membership
    | expression_records.Expression
    | segmentation_records.Segmentation
    | annotation_records.AnnotationLayer
)


class MalformedFracasError(ValueError):
    """An error for malformed FraCaS data or generated records."""


class FracasAnswer(enum.StrEnum):
    """The answer a FraCaS problem carries.

    The suite's answers are four-valued rather than three-valued. Twelve of the
    346 problems in the released XML carry ``undef``, because their source answer
    is a qualification ("Not many", "Yes on one scoping; unknown on another")
    that the release declined to force into the three-way inventory.
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"
    UNDEF = "undef"


class FracasSection(enum.StrEnum):
    """One of the nine semantic-phenomenon strata FraCaS partitions problems into.

    The value is the section heading as the release spells it, since the heading
    is where the stratum is written: a ``problem`` element carries no section
    attribute, and the partition is recoverable only by folding the sibling
    ``comment`` elements in document order.
    """

    QUANTIFIERS = "1 GENERALIZED QUANTIFIERS"
    PLURALS = "2 PLURALS"
    ANAPHORA = "3 (NOMINAL) ANAPHORA"
    ELLIPSIS = "4 ELLIPSIS"
    ADJECTIVES = "5 ADJECTIVES"
    COMPARATIVES = "6 COMPARATIVES"
    TEMPORAL = "7 TEMPORAL REFERENCE"
    VERBS = "8 VERBS"
    ATTITUDES = "9 ATTITUDES"

    @property
    def number(self) -> str:
        """Return the section numeral the release numbers this stratum with."""
        return self.value.split(maxsplit=1)[0]


class FracasProblem(NamedTuple):
    """One FraCaS problem: an ordered premise sequence and what follows from it."""

    problem_id: str
    section: FracasSection
    subsection: str
    premises: tuple[str, ...]
    answer: FracasAnswer
    subsubsection: str | None = None
    pattern: str | None = None
    question: str | None = None
    hypothesis: str | None = None
    source_answer: str | None = None
    standard: bool = True
    why: str | None = None
    note: str | None = None

    @property
    def headings(self) -> _Headings:
        """Return the heading stack this problem sits under."""
        return _Headings(
            self.section, self.subsection, self.subsubsection, self.pattern
        )


class FracasFile(NamedTuple):
    """A FraCaS problem set in document order."""

    problems: tuple[FracasProblem, ...]


class _Headings(NamedTuple):
    """The four heading levels a problem sits under, outermost first."""

    section: FracasSection | None
    subsection: str | None
    subsubsection: str | None
    pattern: str | None


class _ProblemRefs(NamedTuple):
    """Generated references for one problem."""

    premises: tuple[str, ...]
    hypothesis: str | None
    layer: str

    @property
    def expressions(self) -> tuple[str, ...]:
        """Return every expression reference, premises first."""
        if self.hypothesis is None:
            return self.premises

        return (*self.premises, self.hypothesis)


_HEADING_CLASSES: Final = ("section", "subsection", "subsubsection", None)
_ANSWER_SPELLINGS: Final = {
    FracasAnswer.YES: "Yes",
    FracasAnswer.NO: "No",
    FracasAnswer.UNKNOWN: "Don't know",
}


def _text_of(src: str | bytes) -> str:
    """Return UTF-8 source text."""
    if isinstance(src, str):
        return src

    try:
        return src.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "FraCaS input is not UTF-8"
        raise MalformedFracasError(msg) from error


def _content(element: ET.Element | None) -> str | None:
    """Return one element's text with whitespace collapsed, or None when it is blank.

    Every text-bearing element in the release is indented and line-wrapped, so the
    raw text of all 1717 of them differs from its collapsed form.
    """
    if element is None:
        return None

    collapsed = " ".join((element.text or "").split())
    return collapsed or None


def _answer_of(element: ET.Element) -> FracasAnswer:
    """Return the canonical answer one problem element declares."""
    declared = element.get("fracas_answer")

    try:
        return FracasAnswer(declared)
    except ValueError as error:
        problem_id = element.get("id")
        msg = f"problem {problem_id!r} declares answer {declared!r}"
        raise MalformedFracasError(msg) from error


def _premises_of(element: ET.Element) -> tuple[str, ...]:
    """Return one problem's premises in the order its indices declare."""
    premises: list[str] = []

    for position, child in enumerate(element.findall("p"), start=1):
        text = _content(child)

        if text is None:
            msg = f"problem {element.get('id')!r} has an empty premise"
            raise MalformedFracasError(msg)

        if child.get("idx") != str(position):
            msg = (
                f"problem {element.get('id')!r} premise {position} is indexed "
                f"{child.get('idx')!r}"
            )
            raise MalformedFracasError(msg)

        premises.append(text)

    return tuple(premises)


def _problem_of(
    element: ET.Element,
    headings: _Headings,
) -> FracasProblem:
    """Return one problem under the heading stack the document order established."""
    problem_id = element.get("id")

    if not problem_id:
        msg = "a FraCaS problem has no identifier"
        raise MalformedFracasError(msg)

    if headings.section is None or headings.subsection is None:
        msg = f"problem {problem_id!r} precedes its section heading"
        raise MalformedFracasError(msg)

    return FracasProblem(
        problem_id=problem_id,
        section=headings.section,
        subsection=headings.subsection,
        premises=_premises_of(element),
        answer=_answer_of(element),
        subsubsection=headings.subsubsection,
        pattern=headings.pattern,
        question=_content(element.find("q")),
        hypothesis=_content(element.find("h")),
        source_answer=_content(element.find("a")),
        standard=element.get("fracas_nonstandard") != "true",
        why=_content(element.find("why")),
        note=_content(element.find("note")),
    )


def _section_of(heading: str) -> FracasSection:
    """Return the stratum one section comment names."""
    try:
        return FracasSection(heading)
    except ValueError as error:
        msg = f"a FraCaS section heading reads {heading!r}"
        raise MalformedFracasError(msg) from error


def _advanced(headings: _Headings, element: ET.Element) -> _Headings:
    """Return the heading stack one comment element leaves behind it.

    A comment at one level resets every level below it, which is what makes the
    stack recoverable from document order alone.
    """
    kind = element.get("class")
    heading = _content(element)

    if kind == "intro" or heading is None:
        return headings

    if kind == "section":
        return _Headings(_section_of(heading), None, None, None)

    if kind == "subsection":
        return _Headings(headings.section, heading, None, None)

    if kind == "subsubsection":
        return _Headings(headings.section, headings.subsection, heading, None)

    if kind is None:
        return headings._replace(pattern=heading)

    msg = f"a FraCaS comment declares class {kind!r}"
    raise MalformedFracasError(msg)


def _parse(src: str | bytes) -> FracasFile:
    """Return the problem set one FraCaS XML document holds."""
    try:
        root = ET.fromstring(_text_of(src))  # noqa: S314
    except ET.ParseError as error:
        msg = "FraCaS input is not well-formed XML"
        raise MalformedFracasError(msg) from error

    if root.tag != "fracas-problems":
        msg = f"a FraCaS document is rooted at {root.tag!r}"
        raise MalformedFracasError(msg)

    headings = _Headings(None, None, None, None)
    problems: list[FracasProblem] = []

    for child in root:
        if child.tag == "comment":
            headings = _advanced(headings, child)
        elif child.tag == "problem":
            problems.append(_problem_of(child, headings))
        else:
            msg = f"a FraCaS document holds a {child.tag!r} element"
            raise MalformedFracasError(msg)

    return _validated(FracasFile(tuple(problems)))


def _check_heading_run(previous: _Headings, current: _Headings) -> None:
    """Refuse a heading stack that drops a level without changing an outer one.

    Emission introduces a problem by writing the comments from the outermost level
    that changed inward, so a run that leaves an outer heading in place while
    dropping an inner one has no spelling to write back.
    """
    changed = False

    for was, now in zip(previous, current, strict=True):
        if not changed and was != now:
            changed = True

            if now is None:
                msg = f"a FraCaS file drops the heading {was!r} in place"
                raise MalformedFracasError(msg)


def _check_answer(problem: FracasProblem) -> None:
    """Refuse a problem whose answer does not agree with how the source spelled it."""
    if problem.hypothesis is None and problem.answer is not FracasAnswer.UNDEF:
        msg = (
            f"problem {problem.problem_id!r} answers {problem.answer} with no "
            f"hypothesis"
        )
        raise MalformedFracasError(msg)

    if problem.answer is FracasAnswer.UNDEF and problem.standard:
        msg = f"problem {problem.problem_id!r} is undefined yet answered as standard"
        raise MalformedFracasError(msg)

    if not problem.standard:
        return

    spelling = _ANSWER_SPELLINGS[problem.answer]

    if problem.source_answer != spelling:
        msg = (
            f"problem {problem.problem_id!r} answers {problem.answer} yet its "
            f"source answer reads {problem.source_answer!r}"
        )
        raise MalformedFracasError(msg)


def _validated(source: FracasFile) -> FracasFile:
    """Return a problem set every FraCaS invariant holds of."""
    identifiers: set[str] = set()
    previous = _Headings(None, None, None, None)

    for problem in source.problems:
        if not problem.problem_id:
            msg = "a FraCaS problem has an empty identifier"
            raise MalformedFracasError(msg)

        if problem.problem_id in identifiers:
            msg = f"a FraCaS file repeats problem id {problem.problem_id!r}"
            raise MalformedFracasError(msg)

        identifiers.add(problem.problem_id)

        if not problem.premises:
            msg = f"problem {problem.problem_id!r} has no premise"
            raise MalformedFracasError(msg)

        if any(not premise for premise in problem.premises):
            msg = f"problem {problem.problem_id!r} has an empty premise"
            raise MalformedFracasError(msg)

        if not problem.subsection:
            msg = f"problem {problem.problem_id!r} has an empty subsection"
            raise MalformedFracasError(msg)

        if not problem.subsection.startswith(f"{problem.section.number}."):
            msg = (
                f"problem {problem.problem_id!r} sits under subsection "
                f"{problem.subsection!r} in section {problem.section.number}"
            )
            raise MalformedFracasError(msg)

        _check_heading_run(previous, problem.headings)
        _check_answer(problem)
        previous = problem.headings

    return source


def _element(tag: str, text: str, **attributes: str) -> ET.Element:
    """Return one element carrying text."""
    made = ET.Element(tag, attributes)
    made.text = text
    return made


def _heading_elements(
    previous: _Headings,
    current: _Headings,
) -> tuple[ET.Element, ...]:
    """Return the comments that introduce one problem's headings."""
    made: list[ET.Element] = []
    changed = False

    for kind, was, now in zip(_HEADING_CLASSES, previous, current, strict=True):
        changed = changed or was != now

        if changed and now is not None:
            attributes = {} if kind is None else {"class": kind}
            made.append(_element("comment", str(now), **attributes))

    return tuple(made)


def _problem_element(problem: FracasProblem) -> ET.Element:
    """Return one problem element in the release's own shape."""
    attributes = {"id": problem.problem_id, "fracas_answer": problem.answer.value}

    if not problem.standard:
        attributes["fracas_nonstandard"] = "true"

    made = ET.Element("problem", attributes)
    made.extend(
        _element("p", premise, idx=str(position))
        for position, premise in enumerate(problem.premises, start=1)
    )
    made.extend(
        _element(tag, text)
        for tag, text in (
            ("q", problem.question),
            ("h", problem.hypothesis),
            ("a", problem.source_answer),
            ("why", problem.why),
            ("note", problem.note),
        )
        if text is not None
    )
    return made


def _write(source: FracasFile) -> str:
    """Return canonical FraCaS XML for a problem set."""
    root = ET.Element("fracas-problems")
    previous = _Headings(None, None, None, None)

    for problem in _validated(source).problems:
        root.extend(_heading_elements(previous, problem.headings))
        root.append(_problem_element(problem))
        previous = problem.headings

    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def _json_line(row: Mapping[str, JsonValue]) -> str:
    """Return one canonical JSON object."""
    return json.dumps(
        row,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(*parts: str) -> str:
    """Return a stable source-derived key."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()[:24]


def _uri(nsid: str, key: str) -> str:
    """Return one local record reference."""
    return f"{_SCHEME}/{nsid}/{key}"


def _key_of(reference: str) -> str:
    """Return the record key one local reference ends in."""
    return reference.rsplit("/", maxsplit=1)[-1]


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


def _problem_refs(problem: FracasProblem, spec: CorpusRelease) -> _ProblemRefs:
    """Return stable references for one problem."""
    stem = _digest(spec.source_id, spec.release_id, problem.problem_id)
    return _ProblemRefs(
        premises=tuple(
            _uri(_EXPRESSION_NSID, f"{stem}-premise-{position}")
            for position in range(len(problem.premises))
        ),
        hypothesis=(
            None
            if problem.hypothesis is None
            else _uri(_EXPRESSION_NSID, f"{stem}-hypothesis")
        ),
        layer=_uri(_LAYER_NSID, f"{stem}-problem"),
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


def _problem_payload(problem: FracasProblem) -> str:
    """Return canonical JSON for one problem."""
    return _json_line(
        {
            "answer": problem.answer.value,
            "hypothesis": problem.hypothesis,
            "note": problem.note,
            "pattern": problem.pattern,
            "premises": list(problem.premises),
            "problem_id": problem.problem_id,
            "question": problem.question,
            "section": problem.section.value,
            "source_answer": problem.source_answer,
            "standard": problem.standard,
            "subsection": problem.subsection,
            "subsubsection": problem.subsubsection,
            "why": problem.why,
        }
    )


def _arguments(refs: _ProblemRefs) -> tuple[annotation_records.ArgumentRef, ...]:
    """Return one problem's premises in order, then its hypothesis.

    The premise sequence is an argument per premise rather than one concatenated
    expression, so a three-premise problem reads back as three premises.
    """
    arguments = [
        annotation_records.ArgumentRef(
            role=f"premise-{position}",
            target=defs_records.ObjectRef(recordRef=reference),
        )
        for position, reference in enumerate(refs.premises)
    ]

    if refs.hypothesis is not None:
        arguments.append(
            annotation_records.ArgumentRef(
                role=_HYPOTHESIS_ROLE,
                target=defs_records.ObjectRef(recordRef=refs.hypothesis),
            )
        )

    return tuple(arguments)


def _problem_layer(
    problem: FracasProblem,
    refs: _ProblemRefs,
    ordinal: int,
    spec: CorpusRelease,
) -> annotation_records.AnnotationLayer:
    """Return the relation layer carrying one problem and its stratum."""
    return annotation_records.AnnotationLayer(
        createdAt=_EPOCH,
        expression=refs.premises[0],
        kind="relation",
        annotations=(
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value=_PROBLEM_OBJECT),
                label=problem.answer.value,
                arguments=_arguments(refs),
                features=_features(
                    (
                        (_FORMAT, FRACAS_NAME),
                        (_PROBLEM_ORDINAL, str(ordinal)),
                        (_SECTION, problem.section.value),
                        (_SUBSECTION, problem.subsection),
                        (_PROBLEM_JSON, _problem_payload(problem)),
                    )
                ),
            ),
        ),
        sourceMethod=spec.source_method,
        licensing=_licensing(spec),
        eprintRefs=spec.eprint_refs,
    )


def _problem_records(
    problem: FracasProblem,
    ordinal: int,
    first_expression: int,
    corpus_ref: str,
    spec: CorpusRelease,
) -> tuple[FragmentRecord, ...]:
    """Return the expressions, segmentations, memberships, and relation layer."""
    refs = _problem_refs(problem, spec)
    texts = (
        problem.premises
        if problem.hypothesis is None
        else (
            *problem.premises,
            problem.hypothesis,
        )
    )
    records: list[FragmentRecord] = []

    for position, (reference, text) in enumerate(
        zip(refs.expressions, texts, strict=True)
    ):
        key = _key_of(reference)
        records.extend(
            (
                _record(_EXPRESSION_NSID, key, _expression(text, key, spec)),
                _record(_SEGMENTATION_NSID, key, _segmentation(reference, text)),
                _record(
                    _MEMBERSHIP_NSID,
                    key,
                    corpus_records.Membership(
                        corpusRef=corpus_ref,
                        expressionRef=reference,
                        createdAt=_EPOCH,
                        split=spec.split,
                        ordinal=first_expression + position,
                    ),
                ),
            )
        )

    records.append(
        _record(
            _LAYER_NSID,
            _key_of(refs.layer),
            _problem_layer(problem, refs, ordinal, spec),
        )
    )
    return tuple(records)


def _records_of(source: FracasFile, spec: CorpusRelease) -> tuple[FragmentRecord, ...]:
    """Return canonical Layers records for one problem set."""
    source = _validated(source)
    corpus_key = _digest(spec.source_id, spec.release_id, FRACAS_NAME, "corpus")
    corpus_ref = _uri(_CORPUS_NSID, corpus_key)
    expressions = sum(
        len(problem.premises) + (problem.hypothesis is not None)
        for problem in source.problems
    )
    records: list[FragmentRecord] = [
        _record(
            _CORPUS_NSID,
            corpus_key,
            corpus_records.Corpus(
                name=f"{FRACAS_NAME} {spec.release_id}",
                createdAt=_EPOCH,
                languages=(spec.language,),
                version=spec.release_id,
                expressionCount=expressions,
                licensing=_licensing(spec),
                eprintRefs=spec.eprint_refs,
            ),
        )
    ]
    first_expression = 0

    for ordinal, problem in enumerate(source.problems):
        records.extend(
            _problem_records(problem, ordinal, first_expression, corpus_ref, spec)
        )
        first_expression += len(problem.premises) + (problem.hypothesis is not None)

    return tuple(records)


def _feature_values(features: defs_records.FeatureMap | None) -> Mapping[str, str]:
    """Return one feature map without duplicate keys."""
    if features is None:
        return {}

    values = {entry.key: entry.value for entry in features.entries}

    if len(values) != len(features.entries):
        msg = "a FraCaS record repeats a feature key"
        raise MalformedFracasError(msg)

    return values


def _optional_text(row: Mapping[str, JsonValue], key: str) -> str | None:
    """Return one optional text field of a problem payload."""
    value = row[key]

    if value is None:
        return None

    if not isinstance(value, str) or not value:
        msg = f"a FraCaS problem payload has an invalid {key}"
        raise MalformedFracasError(msg)

    return value


def _required_text(row: Mapping[str, JsonValue], key: str) -> str:
    """Return one required text field of a problem payload."""
    value = _optional_text(row, key)

    if value is None:
        msg = f"a FraCaS problem payload has no {key}"
        raise MalformedFracasError(msg)

    return value


def _premises_from(row: Mapping[str, JsonValue]) -> tuple[str, ...]:
    """Return the ordered premise sequence a problem payload holds."""
    value = row["premises"]

    if not isinstance(value, list) or not value:
        msg = "a FraCaS problem payload has no premise sequence"
        raise MalformedFracasError(msg)

    if any(not isinstance(premise, str) or not premise for premise in value):
        msg = "a FraCaS problem payload has an invalid premise"
        raise MalformedFracasError(msg)

    return tuple(str(premise) for premise in value)


def _payload_problem(text: str) -> FracasProblem:
    """Return one problem from canonical record metadata."""
    try:
        row = json.loads(text)
    except json.JSONDecodeError as error:
        msg = "a FraCaS relation layer holds a payload that is not JSON"
        raise MalformedFracasError(msg) from error

    expected = {
        "answer",
        "hypothesis",
        "note",
        "pattern",
        "premises",
        "problem_id",
        "question",
        "section",
        "source_answer",
        "standard",
        "subsection",
        "subsubsection",
        "why",
    }

    if not isinstance(row, dict) or set(row) != expected:
        msg = "a FraCaS relation layer has unsupported problem metadata"
        raise MalformedFracasError(msg)

    if type(row["standard"]) is not bool:
        msg = "a FraCaS problem payload has a nonboolean standard flag"
        raise MalformedFracasError(msg)

    try:
        answer = FracasAnswer(row["answer"])
        section = FracasSection(row["section"])
    except ValueError as error:
        msg = "a FraCaS problem payload names an unknown answer or section"
        raise MalformedFracasError(msg) from error

    return FracasProblem(
        problem_id=_required_text(row, "problem_id"),
        section=section,
        subsection=_required_text(row, "subsection"),
        premises=_premises_from(row),
        answer=answer,
        subsubsection=_optional_text(row, "subsubsection"),
        pattern=_optional_text(row, "pattern"),
        question=_optional_text(row, "question"),
        hypothesis=_optional_text(row, "hypothesis"),
        source_answer=_optional_text(row, "source_answer"),
        standard=row["standard"],
        why=_optional_text(row, "why"),
        note=_optional_text(row, "note"),
    )


def _model_of(record: FragmentRecord) -> _LayerModel:
    """Return one validated Layers model."""
    models: Mapping[str, type[_LayerModel]] = {
        _CORPUS_NSID: corpus_records.Corpus,
        _MEMBERSHIP_NSID: corpus_records.Membership,
        _EXPRESSION_NSID: expression_records.Expression,
        _SEGMENTATION_NSID: segmentation_records.Segmentation,
        _LAYER_NSID: annotation_records.AnnotationLayer,
    }
    model = models.get(record.nsid)

    if model is None:
        msg = f"a FraCaS fragment contains {record.nsid!r}"
        raise MalformedFracasError(msg)

    try:
        return model.model_validate_json(record.value_json)
    except ValueError as error:
        msg = f"a FraCaS record is invalid: {record.local_id!r}"
        raise MalformedFracasError(msg) from error


def _source_of(
    records: tuple[FragmentRecord, ...],
    spec: CorpusRelease,
) -> FracasFile:
    """Return the problem set encoded by Layers records."""
    indexed: list[tuple[int, FracasProblem]] = []

    for record in records:
        model = _model_of(record)

        if not isinstance(model, annotation_records.AnnotationLayer):
            continue

        if (
            model.kind != "relation"
            or model.subkind is not None
            or len(model.annotations) != 1
        ):
            msg = "a FraCaS relation layer has a noncanonical shape"
            raise MalformedFracasError(msg)

        values = _feature_values(model.annotations[0].features)

        if set(values) != {
            _FORMAT,
            _PROBLEM_ORDINAL,
            _SECTION,
            _SUBSECTION,
            _PROBLEM_JSON,
        }:
            msg = "a FraCaS relation layer lacks exact codec metadata"
            raise MalformedFracasError(msg)

        if values[_FORMAT] != FRACAS_NAME:
            msg = f"a FraCaS fragment holds {values[_FORMAT]!r} records"
            raise MalformedFracasError(msg)

        try:
            ordinal = int(values[_PROBLEM_ORDINAL])
        except ValueError as error:
            msg = "a FraCaS relation layer has a noninteger problem ordinal"
            raise MalformedFracasError(msg) from error

        indexed.append((ordinal, _payload_problem(values[_PROBLEM_JSON])))

    indexed.sort(key=lambda entry: entry[0])

    if tuple(ordinal for ordinal, _ in indexed) != tuple(range(len(indexed))):
        msg = "FraCaS problem ordinals are not contiguous from zero"
        raise MalformedFracasError(msg)

    source = _validated(FracasFile(tuple(problem for _, problem in indexed)))

    if _records_of(source, spec) != records:
        msg = "FraCaS records are not in the codec's canonical form"
        raise MalformedFracasError(msg)

    return source


class FracasIso:
    """A translation from a FraCaS problem set to Layers records."""

    __slots__ = ("_spec",)

    def __init__(self, spec: CorpusRelease) -> None:
        self._spec = spec

    def forward(self, source: FracasFile, /) -> CorpusFragment:
        """Return canonical Layers records for a problem set.

        Parameters
        ----------
        source : FracasFile
            The problem set to translate.

        Returns
        -------
        CorpusFragment
            The records the problem set encodes to.
        """
        return CorpusFragment(
            records=_records_of(source, self._spec),
            source=FRACAS_NAME,
        )

    def backward(self, target: CorpusFragment, /) -> FracasFile:
        """Return the problem set Layers records encode.

        Parameters
        ----------
        target : CorpusFragment
            The records to translate back.

        Returns
        -------
        FracasFile
            The problem set the records encode.

        Raises
        ------
        MalformedFracasError
            If the fragment was not produced by this codec.
        """
        if target.source != FRACAS_NAME:
            msg = f"the FraCaS codec received {target.source!r} records"
            raise MalformedFracasError(msg)

        return _source_of(target.records, self._spec)


@registers_as(FRACAS_NAME)
class FracasCodec:
    """A codec for the FraCaS test suite's XML problem set."""

    __slots__ = ("_iso", "name")

    name: str

    def __init__(self, spec: CorpusRelease) -> None:
        self.name = FRACAS_NAME
        self._iso = FracasIso(spec)

    @property
    def iso(self) -> Iso[FracasFile, CorpusFragment]:
        """Return the translation between a problem set and a fragment."""
        return self._iso

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return Layers records for a FraCaS XML document.

        Parameters
        ----------
        src : str | bytes
            The document to read.
        into : CorpusFragment | None
            A fragment to extend, whose records come first.

        Returns
        -------
        CorpusFragment
            The decoded records, appended to those of ``into``.
        """
        decoded = self._iso.forward(_parse(src))

        if into is None:
            return decoded

        return CorpusFragment(records=into.records + decoded.records, source=self.name)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return FraCaS XML for Layers records.

        Parameters
        ----------
        records : Iterable[FragmentRecord]
            The records to write out.

        Returns
        -------
        str
            The problem set as FraCaS XML.
        """
        fragment = CorpusFragment(records=tuple(records), source=self.name)
        return _write(self._iso.backward(fragment))

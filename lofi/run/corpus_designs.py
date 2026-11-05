"""Judgment designs read from declared documents beneath one data root.

Every dataset a judgment fit names is read by one reader: its document under
``designs/<tag>.json`` declares a fragment design, whose Layers records a
registered codec decodes from a release, or a bead design, whose items an
item-construction document builds and whose judgments are bead annotation
records. The rows' factors are read from the item records by the declared
attribute expressions, and an ordinal design may join readout columns: the
published root masses of a trained point, or a pinned causal model's scores of
each item's declared element.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, final, override

from bead.evaluation.reliability import AnnotationRecord
from bead.protocol import ResponseEncoding

from lofi.data import DataRoot, Example, Relational, RelationalScalar, Scalar
from lofi.eval._boundary import RootMassDocument, decode_root_mass
from lofi.judgment import (
    AttributedItem,
    BeadJudgmentSource,
    BoundedSource,
    DesignCodec,
    DesignError,
    DesignProvenance,
    ItemFactors,
    JudgmentRow,
    RatingSource,
    ReadBoundedDesign,
    ReadDesign,
    RelationSource,
    RowColumns,
    construct_items,
    decode_item_construction,
    design_of,
)
from lofi.judgment.language_model import (
    CausalLmScore,
    LmReadoutColumns,
    UnigramTable,
    lm_readout_design,
)
from lofi.nn._snapshots import artifact_fetcher
from lofi.nn.tokens.pretrained import HuggingFaceCausalLoader
from lofi.run._boundary.corpus_designs import (
    BeadDesignSpec,
    BoundedResponsesSpec,
    CausalLmReadoutsSpec,
    CodecFragmentSpec,
    FragmentDesignSpec,
    LayersExportSpec,
    OrdinalResponsesSpec,
    RelationResponsesSpec,
    RootMassSpec,
    declared_data_root,
    decode_design,
    decode_fragment,
    lower_fragment,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.judgment import JudgmentDesign, ReadCorpusDesign
    from lofi.nn.tokens.pretrained import CausalLoader
    from lofi.run._boundary.corpus_designs import (
        FragmentSourceSpec,
        ReadoutSourceSpec,
    )
    from lofi.spec.artifacts import ArtifactCacheSettings

__all__ = [
    "DATA_ROOT_VARIABLE",
    "DESIGNS_DIRECTORY",
    "MEGAACCEPTABILITY",
    "MEGAACCEPTABILITY_MORCELA",
    "MEGAINTENSIONALITY",
    "MEGANEGRAISING",
    "MEGAVERIDICALITY",
    "AbsentCorpusError",
    "DeclaredDesign",
    "parse_unigram_table",
]

DATA_ROOT_VARIABLE: Final = "LOFI_DATA_ROOT"
"""The sole environment variable locating corpus releases and declarations."""

DESIGNS_DIRECTORY: Final = "designs"
"""The data-root directory containing one JSON declaration per design."""

_UNIGRAM_HEADER: Final = "token\tcount"


class AbsentCorpusError(FileNotFoundError):
    """Report a missing data root, declaration, or declared release file."""

    __slots__ = ("variable",)

    def __init__(self, reason: str) -> None:
        self.variable = DATA_ROOT_VARIABLE
        super().__init__(f"set {DATA_ROOT_VARIABLE} to a complete data root: {reason}")


def _data_root() -> DataRoot:
    """Return the validated data root declared through didactic-settings."""
    try:
        root = DataRoot(Path(declared_data_root()))
    except ValueError as error:
        raise AbsentCorpusError(str(error)) from error

    if not root.directory.is_dir():
        msg = f"{root.directory} is not a directory"
        raise AbsentCorpusError(msg)

    return root


def _read(root: DataRoot, declared: str, *, key: str) -> bytes:
    """Read one declared file beneath a data root."""
    path = root.resolve(declared, key=key)
    if not path.is_file():
        msg = f"{declared} is not a file"
        raise AbsentCorpusError(msg)
    return path.read_bytes()


def _digests(entries: Mapping[str, bytes]) -> dict[str, str]:
    """Return SHA-256 digests keyed by data-root-relative path."""
    return {name: sha256(data).hexdigest() for name, data in entries.items()}


def _provenance(
    tag: str,
    corpus_digests: Mapping[str, str],
    *,
    checkpoint: str | None = None,
    checkpoint_dtype: str | None = None,
    unigram_digest: str | None = None,
) -> DesignProvenance:
    """Return provenance for a fully read corpus design."""
    return DesignProvenance(
        tag=tag,
        checkpoint=checkpoint,
        checkpoint_dtype=checkpoint_dtype,
        unigram_digest=unigram_digest,
        corpus_digests=dict(corpus_digests),
        seed=None,
        participants=None,
        items=None,
        levels=None,
        population_digest=None,
        producer_run=None,
        producer_point=None,
    )


def _root_mass_design(
    source: RatingSource, artifact: RootMassDocument, columns: tuple[str, str], /
) -> JudgmentDesign:
    """Return a design carrying one published mass pair per item.

    The pair fills the declared mass and complement columns, in that order.
    """
    rows = tuple(source.rows())
    expected = tuple(dict.fromkeys(row.item for row in rows))
    expected_set = frozenset(expected)
    recorded = {
        item.item_id: (item.log_mass, item.log_complement) for item in artifact.items
    }
    missing = tuple(item for item in expected if item not in recorded)
    unexpected = tuple(item for item in recorded if item not in expected_set)

    if missing or unexpected:
        msg = (
            f"published root masses do not match the rated items: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )
        raise DesignError(msg)

    codec = DesignCodec(source.response_levels, source.factors, columns)
    return codec.encode(
        JudgmentRow(
            row.participant,
            row.item,
            row.response,
            RowColumns(
                dict(row.factors),
                dict(
                    zip(
                        columns,
                        recorded[row.item],
                        strict=True,
                    )
                ),
            ),
        )
        for row in rows
    )


def parse_unigram_table(data: bytes, /, *, source: str) -> UnigramTable:
    """Return the exact token counts declared by a UTF-8 TSV file.

    Lines may end in LF or CRLF.
    """
    header, *lines = (
        line.removesuffix("\r")
        for line in data.decode("utf-8").rstrip("\r\n").split("\n")
    )
    if header != _UNIGRAM_HEADER:
        msg = f"{source} has the header {header!r}, not {_UNIGRAM_HEADER!r}"
        raise ValueError(msg)

    counts: dict[str, int] = {}
    for number, line in enumerate(lines, start=2):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[1].isdigit():
            msg = f"{source} line {number} is not a token and a count: {line!r}"
            raise ValueError(msg)
        token, count = fields
        if token in counts:
            msg = f"{source} repeats the token {token!r} at line {number}"
            raise ValueError(msg)
        counts[token] = int(count)
    return UnigramTable(counts)


def _fragment_files(source: FragmentSourceSpec, /) -> tuple[str, ...]:
    """Return the files a fragment source declares, relative to the data root."""
    match source:
        case CodecFragmentSpec(declaration=declaration, responses=responses):
            return (declaration, responses)
        case LayersExportSpec(path=path):
            return (path,)
    msg = f"unknown fragment source {source!r}"
    raise TypeError(msg)


def _levels(ratings: tuple[Example[Scalar], ...], /) -> int:
    """Return the one number of ordered levels every rating's scale has."""
    levels = frozenset(rating.supervision.scale.levels for rating in ratings)
    if len(levels) != 1:
        msg = f"the ratings are on scales of {sorted(map(str, levels))} levels"
        raise DesignError(msg)
    (count,) = levels
    if count is None:
        msg = "the ratings are on a scale without ordered levels"
        raise DesignError(msg)
    return count


@final
class DeclaredDesign:
    """Read the design its document declares beneath the data root.

    Parameters
    ----------
    tag : str
        The dataset identifier, naming the document ``designs/<tag>.json``.
    loader : Callable[[ArtifactCacheSettings], CausalLoader]
        Returns the loader of the causal model a declared language-model readout
        pins, fetching through the cache the readout declares.
    """

    __slots__ = ("_loader", "_tag")

    def __init__(
        self, tag: str, loader: Callable[[ArtifactCacheSettings], CausalLoader]
    ) -> None:
        self._tag = tag
        self._loader = loader

    def read(self) -> ReadCorpusDesign:
        """Return the declared design and the provenance of what it read.

        Raises
        ------
        AbsentCorpusError
            If the data root, the document or a declared file is absent.
        DesignError
            If the judgments cannot be read as the declared design.
        """
        root = _data_root()
        path = f"{DESIGNS_DIRECTORY}/{self._tag}.json"
        document = _read(root, path, key=f"{self._tag}.design")
        files = {path: document}
        match decode_design(document):
            case FragmentDesignSpec() as declaration:
                return self._fragment(root, declaration, files)
            case BeadDesignSpec() as declaration:
                return self._bead(root, declaration, files)
            case declaration:
                msg = f"unknown design source {declaration!r}"
                raise TypeError(msg)

    def _fragment(
        self,
        root: DataRoot,
        declaration: FragmentDesignSpec,
        files: dict[str, bytes],
    ) -> ReadCorpusDesign:
        """Return the design one release's Layers records declare."""
        for declared in _fragment_files(declaration.fragment):
            files[declared] = _read(root, declared, key=declared)
        readouts = declaration.readouts
        element = (
            readouts.element if isinstance(readouts, CausalLmReadoutsSpec) else None
        )
        lowered = lower_fragment(
            decode_fragment(declaration.fragment, files), element, declaration.keep
        )
        factors = ItemFactors(declaration.factors, lowered.items)
        examples = lowered.examples
        match declaration.responses:
            case OrdinalResponsesSpec():
                rated = tuple(
                    Example(example.item, supervision)
                    for example in examples
                    if isinstance(supervision := example.supervision, Scalar)
                )
                source = RatingSource(rated, levels=_levels(rated), factors=factors)
                if readouts is None:
                    design = design_of(source)
                    return ReadDesign(design, _provenance(self._tag, _digests(files)))
                return self._readouts(root, source, readouts, lowered.items, files)
            case RelationResponsesSpec(inventory=inventory):
                relations = tuple(
                    Example(example.item, supervision)
                    for example in examples
                    if isinstance(supervision := example.supervision, Relational)
                )
                design = design_of(
                    RelationSource(relations, inventory=inventory, factors=factors)
                )
                return ReadDesign(design, _provenance(self._tag, _digests(files)))
            case BoundedResponsesSpec():
                bounded = tuple(
                    Example(example.item, supervision)
                    for example in examples
                    if isinstance(supervision := example.supervision, RelationalScalar)
                )
                design = BoundedSource(bounded, factors=factors).design()
                return ReadBoundedDesign(
                    design, _provenance(self._tag, _digests(files))
                )
        msg = f"unknown response reading {declaration.responses!r}"
        raise TypeError(msg)

    def _readouts(
        self,
        root: DataRoot,
        source: RatingSource,
        readouts: ReadoutSourceSpec,
        items: tuple[AttributedItem, ...],
        files: dict[str, bytes],
    ) -> ReadDesign:
        """Return a rating design joined to its declared readout columns."""
        match readouts:
            case RootMassSpec(path=path, mass=mass, complement=complement):
                data = _read(root, path, key="root_mass")
                files[path] = data
                design = _root_mass_design(
                    source, decode_root_mass(data), (mass, complement)
                )
                return ReadDesign(design, _provenance(self._tag, _digests(files)))
            case (
                CausalLmReadoutsSpec(
                    element=element,
                    artifacts=artifacts,
                    unigrams=unigrams,
                    columns=columns,
                ) as declared
            ):
                counts = _read(root, unigrams, key="unigrams")
                files[unigrams] = counts
                checkpoint = declared.pinned()
                scorer = CausalLmScore(
                    self._loader(artifacts.settings()),
                    parse_unigram_table(counts, source=unigrams),
                    checkpoint=checkpoint,
                )
                texts = {
                    str(item.id): item.rendered_elements[element]
                    for item in items
                    if element in item.rendered_elements
                }
                design = lm_readout_design(
                    source,
                    texts,
                    scorer,
                    LmReadoutColumns(
                        columns.log_probability,
                        columns.unigram_log_probability,
                        columns.length,
                    ),
                )
                provenance = _provenance(
                    self._tag,
                    _digests(files),
                    checkpoint=checkpoint.identity,
                    checkpoint_dtype=str(scorer.dtype).removeprefix("torch."),
                    unigram_digest=sha256(counts).hexdigest(),
                )
                return ReadDesign(design, provenance)
        msg = f"unknown readout source {readouts!r}"
        raise TypeError(msg)

    def _bead(
        self, root: DataRoot, declaration: BeadDesignSpec, files: dict[str, bytes]
    ) -> ReadDesign:
        """Return the design bead annotation records give constructed items."""
        construction_path = root.resolve(declaration.items, key="items")
        construction = _read(root, declaration.items, key="items")
        files[declaration.items] = construction
        spec = decode_item_construction(construction)
        for declared in spec.files():
            relative = str(Path(declaration.items).parent / declared)
            files[relative] = _read(root, relative, key=relative)
        items = construct_items(spec, construction_path.parent)
        encoding = _read(root, declaration.encoding, key="encoding")
        judgments = _read(root, declaration.judgments, key="judgments")
        files[declaration.encoding] = encoding
        files[declaration.judgments] = judgments
        records = tuple(
            AnnotationRecord.model_validate_json(stripped)
            for line in judgments.decode("utf-8").splitlines()
            if (stripped := line.strip())
        )
        source = BeadJudgmentSource(
            records,
            ResponseEncoding.model_validate_json(encoding),
            factors=ItemFactors(declaration.factors, items),
        )
        return ReadDesign(design_of(source), _provenance(self._tag, _digests(files)))

    @override
    def __repr__(self) -> str:
        return f"DeclaredDesign({self._tag!r})"


def _causal_loader(cache: ArtifactCacheSettings, /) -> CausalLoader:
    """Return the loader fetching a readout's causal model into its cache."""
    return HuggingFaceCausalLoader(artifact_fetcher(cache))


_LOADER: Final = _causal_loader
"""The loader every registered design's causal-model readouts use."""

MEGAACCEPTABILITY: Final = DeclaredDesign("megaacceptability", _LOADER)
MEGAACCEPTABILITY_MORCELA: Final = DeclaredDesign("megaacceptability-morcela", _LOADER)
MEGAVERIDICALITY: Final = DeclaredDesign("megaveridicality", _LOADER)
MEGANEGRAISING: Final = DeclaredDesign("meganegraising", _LOADER)
MEGAINTENSIONALITY: Final = DeclaredDesign("megaintensionality", _LOADER)

"""Cross-lingual comparison of aligned inventories and type embeddings."""

from math import comb
from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

import torch

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._alignment import Alignment
from lofi.eval._stability import AlignmentStability
from lofi.eval._transfer import TransferLanguage, UnavailableLanguage
from lofi.stats import Interval, LabelPermutation, PValue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._types import JsonValue

__all__ = [
    "AlignmentCoverage",
    "CrosslingualAlignmentCell",
    "CrosslingualAlignmentDiagonal",
    "CrosslingualAlignmentMatrix",
    "JaccardFloor",
    "LanguageAlignment",
    "UnavailableAlignmentCell",
    "crosslingual_alignment_matrix",
    "hypergeometric_jaccard_floor",
]


def _proportion(field: str, value: float) -> float:
    """Return a finite value on the closed unit interval."""
    held = ensure_finite(field, value)

    if not 0.0 <= held <= 1.0:
        msg = f"{field} is outside the closed unit interval: {value!r}"
        raise ValueError(msg)

    return held


@final
class AlignmentCoverage(FrozenValue):
    """Counts how much of one induced inventory its alignment covers."""

    _noun = "cross-lingual alignment value"

    __slots__ = ("_aligned", "_language", "_total")

    def __init__(self, language: TransferLanguage, aligned: int, total: int) -> None:
        if not isinstance(language, TransferLanguage):
            msg = f"an alignment coverage's language is not one: {language!r}"
            raise TypeError(msg)

        held_aligned = ensure_non_negative("an aligned-label count", aligned)
        held_total = ensure_non_negative("an induced-label count", total)

        if held_total == 0:
            msg = f"the language {language.code!r} has an empty induced inventory"
            raise ValueError(msg)

        if held_aligned > held_total:
            msg = (
                f"the language {language.code!r} aligns {held_aligned} labels "
                f"from an inventory of {held_total}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_aligned", held_aligned)
        object.__setattr__(self, "_total", held_total)

    @property
    def language(self) -> TransferLanguage:
        """Return the language whose induced inventory is counted."""
        return self._language

    @property
    def aligned(self) -> int:
        """Return the number of induced labels assigned a reference label."""
        return self._aligned

    @property
    def total(self) -> int:
        """Return the total number of induced labels."""
        return self._total

    @property
    def proportion(self) -> float:
        """Return the aligned fraction of the induced inventory."""
        return self._aligned / self._total

    def _fields(self) -> tuple[TransferLanguage, int, int]:
        """Return fields in equality order."""
        return self._language, self._aligned, self._total

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not AlignmentCoverage:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((AlignmentCoverage, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"AlignmentCoverage({self._language!r}, {self._aligned!r}, {self._total!r})"
        )


@final
class LanguageAlignment(FrozenValue):
    """One language's fitted alignment, stability, and induced-label vectors.

    Vector values are snapshotted at construction and exposed only through fresh
    float64 tensors. Every induced label in the alignment's contingency must have
    exactly one finite nonzero vector, including labels left unaligned.
    """

    _noun = "cross-lingual alignment input"

    __slots__ = ("_alignment", "_language", "_stability", "_vectors", "_width")

    def __init__(
        self,
        language: TransferLanguage,
        alignment: Alignment,
        stability: AlignmentStability,
        vectors: Mapping[str, Tensor],
    ) -> None:
        if not isinstance(language, TransferLanguage):
            msg = f"a language alignment's language is not one: {language!r}"
            raise TypeError(msg)

        if not isinstance(alignment, Alignment):
            msg = (
                f"the language {language.code!r} supplies something other than "
                f"an Alignment: {alignment!r}"
            )
            raise TypeError(msg)

        if not isinstance(stability, AlignmentStability):
            msg = (
                f"the language {language.code!r} supplies something other than "
                f"an AlignmentStability: {stability!r}"
            )
            raise TypeError(msg)

        expected = set(alignment.contingency.induced_labels)
        stability_labels = (*stability.labels, *stability.dropped)
        stability_inventory = set(stability_labels)
        repeated_stability = sorted(
            label for label in stability_inventory if stability_labels.count(label) > 1
        )
        missing_stability = sorted(expected - stability_inventory)
        extra_stability = sorted(stability_inventory - expected)

        if repeated_stability or missing_stability or extra_stability:
            msg = (
                f"the stability for {language.code!r} does not belong to the "
                f"alignment's exact induced labels: repeated {repeated_stability!r}, "
                f"missing {missing_stability!r}, extra {extra_stability!r}"
            )
            raise ValueError(msg)

        observed = set(vectors)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)

        if missing:
            msg = f"the language {language.code!r} has no vector for {missing!r}"
            raise ValueError(msg)

        if extra:
            msg = (
                f"the language {language.code!r} supplies vectors for unknown "
                f"induced labels {extra!r}"
            )
            raise ValueError(msg)

        checked: list[tuple[str, tuple[float, ...]]] = []
        width: int | None = None

        for label in sorted(expected):
            vector = vectors[label]

            if not isinstance(vector, torch.Tensor):
                msg = f"the vector for {label!r} in {language.code!r} is not a tensor"
                raise TypeError(msg)

            if not vector.is_floating_point():
                msg = (
                    f"the vector for {label!r} in {language.code!r} holds "
                    f"{vector.dtype}, not floating-point values"
                )
                raise TypeError(msg)

            if vector.ndim != 1:
                msg = (
                    f"the vector for {label!r} in {language.code!r} has shape "
                    f"{tuple(vector.shape)}, expected one width axis"
                )
                raise ValueError(msg)

            if vector.numel() == 0:
                msg = f"the vector for {label!r} in {language.code!r} has zero width"
                raise ValueError(msg)

            held = vector.detach().to(device="cpu", dtype=torch.float64)

            if not bool(torch.isfinite(held).all()):
                msg = f"the vector for {label!r} in {language.code!r} is not finite"
                raise ValueError(msg)

            if bool(held.norm() == 0.0):
                msg = f"the vector for {label!r} in {language.code!r} is zero"
                raise ValueError(msg)

            current = int(held.numel())

            if width is not None and current != width:
                msg = (
                    f"the vector for {label!r} in {language.code!r} has width "
                    f"{current}, expected {width}"
                )
                raise ValueError(msg)

            width = current
            checked.append((label, tuple(float(value) for value in held.tolist())))

        if width is None:
            msg = f"the language {language.code!r} has no induced-label vectors"
            raise ValueError(msg)

        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_alignment", alignment)
        object.__setattr__(self, "_stability", stability)
        object.__setattr__(self, "_vectors", tuple(checked))
        object.__setattr__(self, "_width", width)

    @property
    def language(self) -> TransferLanguage:
        """Return the language being aligned."""
        return self._language

    @property
    def alignment(self) -> Alignment:
        """Return its induced-to-reference alignment."""
        return self._alignment

    @property
    def stability(self) -> AlignmentStability:
        """Return the alignment's mandatory seed stability."""
        return self._stability

    @property
    def labels(self) -> tuple[str, ...]:
        """Return induced labels in canonical order."""
        return tuple(label for label, _ in self._vectors)

    @property
    def width(self) -> int:
        """Return the common embedding width."""
        return self._width

    @property
    def coverage(self) -> AlignmentCoverage:
        """Return aligned and total induced-label counts."""
        return AlignmentCoverage(
            self._language,
            len(self._alignment.mapping),
            len(self._alignment.contingency.induced_labels),
        )

    def vector(self, label: str, /) -> Tensor:
        """Return a fresh copy of one induced label's vector."""
        for name, values in self._vectors:
            if name == label:
                return torch.tensor(values, dtype=torch.float64)

        msg = f"the language {self._language.code!r} has no vector for {label!r}"
        raise KeyError(msg)

    def _fields(
        self,
    ) -> tuple[
        TransferLanguage,
        Alignment,
        AlignmentStability,
        tuple[tuple[str, tuple[float, ...]], ...],
    ]:
        """Return fields in equality order."""
        return self._language, self._alignment, self._stability, self._vectors

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not LanguageAlignment:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((LanguageAlignment, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"LanguageAlignment(language={self._language!r}, "
            f"alignment={self._alignment!r}, stability={self._stability!r}, "
            f"labels={self.labels!r}, width={self._width!r})"
        )


@final
class JaccardFloor(FrozenValue):
    """Exact fixed-cardinality hypergeometric Jaccard reference."""

    _noun = "cross-lingual alignment value"

    __slots__ = ("_expected", "_interval", "_left", "_right", "_universe")

    def __init__(
        self,
        universe: int,
        left: int,
        right: int,
        expected: float,
        interval: Interval,
    ) -> None:
        held_universe = ensure_non_negative("a reference-universe size", universe)
        held_left = ensure_non_negative("a left inventory size", left)
        held_right = ensure_non_negative("a right inventory size", right)

        if held_left > held_universe or held_right > held_universe:
            msg = (
                f"fixed inventories of {held_left} and {held_right} do not fit "
                f"a reference universe of {held_universe}"
            )
            raise ValueError(msg)

        if not isinstance(interval, Interval):
            msg = f"a Jaccard floor's interval is not one: {interval!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_universe", held_universe)
        object.__setattr__(self, "_left", held_left)
        object.__setattr__(self, "_right", held_right)
        object.__setattr__(
            self, "_expected", _proportion("an expected Jaccard", expected)
        )
        object.__setattr__(self, "_interval", interval)

    @property
    def universe(self) -> int:
        """Return the shared reference-inventory size."""
        return self._universe

    @property
    def left(self) -> int:
        """Return the first fixed cardinality."""
        return self._left

    @property
    def right(self) -> int:
        """Return the second fixed cardinality."""
        return self._right

    @property
    def expected(self) -> float:
        """Return exact expected Jaccard similarity."""
        return self._expected

    @property
    def interval(self) -> Interval:
        """Return the exact central 95% interval."""
        return self._interval

    def _fields(self) -> tuple[int, int, int, float, Interval]:
        """Return fields in equality order."""
        return self._universe, self._left, self._right, self._expected, self._interval

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not JaccardFloor:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((JaccardFloor, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"JaccardFloor({self._universe!r}, {self._left!r}, {self._right!r}, "
            f"{self._expected!r}, {self._interval!r})"
        )


def _jaccard(overlap: int, left: int, right: int) -> float:
    """Return Jaccard similarity from an overlap and two cardinalities."""
    union = left + right - overlap

    return 1.0 if union == 0 else overlap / union


def hypergeometric_jaccard_floor(
    universe: int,
    left: int,
    right: int,
    /,
    *,
    level: float,
) -> JaccardFloor:
    """Return exact Jaccard moments for two fixed-size random subsets.

    The overlap is hypergeometric. The interval endpoints are the overlap
    quantiles leaving equal mass outside the declared coverage in each tail,
    transformed monotonically onto the Jaccard scale.
    """
    held_universe = ensure_non_negative("a reference-universe size", universe)
    held_left = ensure_non_negative("a left inventory size", left)
    held_right = ensure_non_negative("a right inventory size", right)

    if held_left > held_universe or held_right > held_universe:
        msg = (
            f"fixed inventories of {held_left} and {held_right} do not fit "
            f"a reference universe of {held_universe}"
        )
        raise ValueError(msg)

    low = max(0, held_left + held_right - held_universe)
    high = min(held_left, held_right)
    denominator = comb(held_universe, held_right)
    distribution = tuple(
        (
            overlap,
            comb(held_left, overlap)
            * comb(held_universe - held_left, held_right - overlap)
            / denominator,
        )
        for overlap in range(low, high + 1)
    )
    expected = sum(
        probability * _jaccard(overlap, held_left, held_right)
        for overlap, probability in distribution
    )
    lower_overlap = distribution[0][0]
    upper_overlap = distribution[-1][0]
    outside = (1.0 - level) / 2.0
    cumulative = 0.0
    lower_found = False

    for overlap, probability in distribution:
        cumulative += probability

        if cumulative >= outside and not lower_found:
            lower_overlap = overlap
            lower_found = True

        if cumulative >= 1.0 - outside:
            upper_overlap = overlap
            break

    interval = Interval(
        _jaccard(lower_overlap, held_left, held_right),
        _jaccard(upper_overlap, held_left, held_right),
        level,
    )

    return JaccardFloor(
        held_universe,
        held_left,
        held_right,
        expected,
        interval,
    )


@final
class CrosslingualAlignmentCell(FrozenValue):
    """One available off-diagonal inventory and embedding comparison."""

    _noun = "cross-lingual alignment cell"

    __slots__ = (
        "_floor",
        "_jaccard",
        "_left",
        "_left_coverage",
        "_left_stability",
        "_permutation",
        "_procrustes_cosine",
        "_right",
        "_right_coverage",
        "_right_stability",
        "_shared",
    )

    def __init__(
        self,
        left: TransferLanguage,
        right: TransferLanguage,
        *,
        shared: Sequence[str],
        jaccard: float,
        floor: JaccardFloor,
        procrustes_cosine: float,
        permutation: PValue,
        left_coverage: AlignmentCoverage,
        right_coverage: AlignmentCoverage,
        left_stability: AlignmentStability,
        right_stability: AlignmentStability,
    ) -> None:
        for field, language in (("left", left), ("right", right)):
            if not isinstance(language, TransferLanguage):
                msg = f"a cell's {field} language is not one: {language!r}"
                raise TypeError(msg)

        if not isinstance(floor, JaccardFloor):
            msg = f"a cell's Jaccard floor is not one: {floor!r}"
            raise TypeError(msg)

        if not isinstance(permutation, PValue):
            msg = f"a cell's permutation result is not a p-value: {permutation!r}"
            raise TypeError(msg)

        for field, coverage, language in (
            ("left", left_coverage, left),
            ("right", right_coverage, right),
        ):
            if not isinstance(coverage, AlignmentCoverage):
                msg = f"a cell's {field} coverage is not one: {coverage!r}"
                raise TypeError(msg)

            if coverage.language != language:
                msg = f"a cell's {field} coverage belongs to another language"
                raise ValueError(msg)

        for field, stability in (
            ("left", left_stability),
            ("right", right_stability),
        ):
            if not isinstance(stability, AlignmentStability):
                msg = f"a cell's {field} stability is not one: {stability!r}"
                raise TypeError(msg)

        labels = tuple(
            ensure_name("a shared reference label", label) for label in shared
        )

        if not labels:
            msg = f"{left.code!r} and {right.code!r} share no aligned reference label"
            raise ValueError(msg)

        if len(set(labels)) != len(labels):
            repeated = sorted(label for label in set(labels) if labels.count(label) > 1)
            msg = f"an alignment cell repeats shared labels {repeated!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_left", left)
        object.__setattr__(self, "_right", right)
        object.__setattr__(self, "_shared", labels)
        object.__setattr__(
            self, "_jaccard", _proportion("a Jaccard similarity", jaccard)
        )
        object.__setattr__(self, "_floor", floor)
        object.__setattr__(
            self,
            "_procrustes_cosine",
            ensure_finite("a Procrustes mean row cosine", procrustes_cosine),
        )
        object.__setattr__(self, "_permutation", permutation)
        object.__setattr__(self, "_left_coverage", left_coverage)
        object.__setattr__(self, "_right_coverage", right_coverage)
        object.__setattr__(self, "_left_stability", left_stability)
        object.__setattr__(self, "_right_stability", right_stability)

    @property
    def left(self) -> TransferLanguage:
        return self._left

    @property
    def right(self) -> TransferLanguage:
        return self._right

    @property
    def shared(self) -> tuple[str, ...]:
        return self._shared

    @property
    def jaccard(self) -> float:
        return self._jaccard

    @property
    def floor(self) -> JaccardFloor:
        return self._floor

    @property
    def procrustes_cosine(self) -> float:
        return self._procrustes_cosine

    @property
    def permutation(self) -> PValue:
        return self._permutation

    @property
    def left_coverage(self) -> AlignmentCoverage:
        return self._left_coverage

    @property
    def right_coverage(self) -> AlignmentCoverage:
        return self._right_coverage

    @property
    def left_stability(self) -> AlignmentStability:
        return self._left_stability

    @property
    def right_stability(self) -> AlignmentStability:
        return self._right_stability

    @property
    def available(self) -> bool:
        return True

    @property
    def diagonal(self) -> bool:
        return False

    def _fields(
        self,
    ) -> tuple[
        TransferLanguage,
        TransferLanguage,
        tuple[str, ...],
        float,
        JaccardFloor,
        float,
        PValue,
        AlignmentCoverage,
        AlignmentCoverage,
        AlignmentStability,
        AlignmentStability,
    ]:
        return (
            self._left,
            self._right,
            self._shared,
            self._jaccard,
            self._floor,
            self._procrustes_cosine,
            self._permutation,
            self._left_coverage,
            self._right_coverage,
            self._left_stability,
            self._right_stability,
        )

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not CrosslingualAlignmentCell:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((CrosslingualAlignmentCell, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"CrosslingualAlignmentCell({self._left!r}, {self._right!r}, "
            f"jaccard={self._jaccard!r}, "
            f"procrustes_cosine={self._procrustes_cosine!r})"
        )


@final
class CrosslingualAlignmentDiagonal(FrozenValue):
    """An explicit available diagonal cell."""

    _noun = "cross-lingual alignment cell"

    __slots__ = ("_coverage", "_language", "_stability")

    def __init__(
        self,
        language: TransferLanguage,
        coverage: AlignmentCoverage,
        stability: AlignmentStability,
    ) -> None:
        if not isinstance(language, TransferLanguage):
            msg = f"a diagonal's language is not one: {language!r}"
            raise TypeError(msg)

        if not isinstance(coverage, AlignmentCoverage):
            msg = f"a diagonal's coverage is not one: {coverage!r}"
            raise TypeError(msg)

        if coverage.language != language:
            msg = "a diagonal's coverage belongs to another language"
            raise ValueError(msg)

        if not isinstance(stability, AlignmentStability):
            msg = f"a diagonal's stability is not one: {stability!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_coverage", coverage)
        object.__setattr__(self, "_stability", stability)

    @property
    def left(self) -> TransferLanguage:
        return self._language

    @property
    def right(self) -> TransferLanguage:
        return self._language

    @property
    def language(self) -> TransferLanguage:
        return self._language

    @property
    def coverage(self) -> AlignmentCoverage:
        return self._coverage

    @property
    def stability(self) -> AlignmentStability:
        return self._stability

    @property
    def jaccard(self) -> float:
        return 1.0

    @property
    def procrustes_cosine(self) -> float:
        return 1.0

    @property
    def available(self) -> bool:
        return True

    @property
    def diagonal(self) -> bool:
        return True

    def _fields(self) -> tuple[TransferLanguage, AlignmentCoverage, AlignmentStability]:
        return self._language, self._coverage, self._stability

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not CrosslingualAlignmentDiagonal:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((CrosslingualAlignmentDiagonal, self._fields()))

    @override
    def __repr__(self) -> str:
        return f"CrosslingualAlignmentDiagonal({self._language!r})"


@final
class UnavailableAlignmentCell(FrozenValue):
    """A matrix cell unavailable because at least one language is unavailable."""

    _noun = "cross-lingual alignment cell"

    __slots__ = ("_left", "_reasons", "_right")

    def __init__(
        self,
        left: TransferLanguage,
        right: TransferLanguage,
        reasons: Sequence[UnavailableLanguage],
    ) -> None:
        for field, language in (("left", left), ("right", right)):
            if not isinstance(language, TransferLanguage):
                msg = f"an unavailable cell's {field} language is not one: {language!r}"
                raise TypeError(msg)

        held = tuple(reasons)

        if not held:
            msg = "an unavailable cell names no unavailable language"
            raise ValueError(msg)

        declared = {left, right}

        for reason in held:
            if not isinstance(reason, UnavailableLanguage):
                msg = f"an unavailable cell's reason is not typed: {reason!r}"
                raise TypeError(msg)

            if reason.language not in declared:
                msg = (
                    f"the unavailable language {reason.language.code!r} is not "
                    f"one of this cell's endpoints"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_left", left)
        object.__setattr__(self, "_right", right)
        object.__setattr__(self, "_reasons", held)

    @property
    def left(self) -> TransferLanguage:
        return self._left

    @property
    def right(self) -> TransferLanguage:
        return self._right

    @property
    def reasons(self) -> tuple[UnavailableLanguage, ...]:
        return self._reasons

    @property
    def available(self) -> bool:
        return False

    @property
    def diagonal(self) -> bool:
        return self._left == self._right

    def _fields(
        self,
    ) -> tuple[TransferLanguage, TransferLanguage, tuple[UnavailableLanguage, ...]]:
        return self._left, self._right, self._reasons

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not UnavailableAlignmentCell:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((UnavailableAlignmentCell, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"UnavailableAlignmentCell({self._left!r}, {self._right!r}, "
            f"{self._reasons!r})"
        )


type _AlignmentMatrixCell = (
    CrosslingualAlignmentCell | CrosslingualAlignmentDiagonal | UnavailableAlignmentCell
)

type _CoverageEntry = AlignmentCoverage | UnavailableLanguage
type _StabilityEntry = AlignmentStability | UnavailableLanguage


@final
class CrosslingualAlignmentMatrix(FrozenValue):
    """A complete symmetric matrix in declared language order."""

    _noun = "cross-lingual alignment matrix"

    __slots__ = (
        "_cells",
        "_codes",
        "_coverages",
        "_languages",
        "_reference_labels",
        "_stabilities",
    )

    def __init__(
        self,
        reference_labels: Sequence[str],
        languages: Sequence[TransferLanguage],
        cells: Sequence[Sequence[_AlignmentMatrixCell]],
        coverages: Sequence[_CoverageEntry],
        stabilities: Sequence[_StabilityEntry],
    ) -> None:
        references = _references(reference_labels)
        held_languages = tuple(languages)
        held_cells = tuple(tuple(row) for row in cells)
        held_coverages = tuple(coverages)
        held_stabilities = tuple(stabilities)
        size = len(held_languages)

        if not held_languages:
            msg = "a cross-lingual alignment matrix declares no language"
            raise ValueError(msg)

        codes: dict[str, int] = {}

        for position, language in enumerate(held_languages):
            if not isinstance(language, TransferLanguage):
                msg = f"alignment matrix language {position} is not one: {language!r}"
                raise TypeError(msg)

            if language.code in codes:
                msg = f"the alignment matrix repeats language {language.code!r}"
                raise ValueError(msg)

            codes[language.code] = position

        if len(held_cells) != size or any(len(row) != size for row in held_cells):
            msg = f"an alignment matrix over {size} languages is not square"
            raise ValueError(msg)

        if len(held_coverages) != size or len(held_stabilities) != size:
            msg = f"an alignment matrix over {size} languages has incomplete metadata"
            raise ValueError(msg)

        for i in range(size):
            language = held_languages[i]
            coverage = held_coverages[i]
            stability = held_stabilities[i]
            diagonal = held_cells[i][i]

            if isinstance(coverage, AlignmentCoverage):
                if coverage.language != language:
                    msg = (
                        f"alignment coverage {i} belongs to "
                        f"{coverage.language.code!r}, "
                        f"not {language.code!r}"
                    )
                    raise ValueError(msg)

                if not isinstance(stability, AlignmentStability):
                    msg = (
                        f"alignment stability {i} for {language.code!r} is not one: "
                        f"{stability!r}"
                    )
                    raise TypeError(msg)

                if not isinstance(diagonal, CrosslingualAlignmentDiagonal):
                    msg = (
                        f"the diagonal for available language {language.code!r} is not "
                        "an available diagonal"
                    )
                    raise TypeError(msg)

                if (
                    diagonal.language != language
                    or diagonal.coverage != coverage
                    or diagonal.stability != stability
                ):
                    msg = (
                        f"the diagonal for {language.code!r} does not carry its exact "
                        "coverage and stability"
                    )
                    raise ValueError(msg)

            elif isinstance(coverage, UnavailableLanguage):
                if coverage.language != language:
                    msg = (
                        f"alignment unavailability {i} belongs to "
                        f"{coverage.language.code!r}, not {language.code!r}"
                    )
                    raise ValueError(msg)

                if not isinstance(stability, UnavailableLanguage):
                    msg = (
                        f"alignment stability {i} for unavailable language "
                        f"{language.code!r} is not typed unavailability: {stability!r}"
                    )
                    raise TypeError(msg)

                if stability != coverage:
                    msg = (
                        f"coverage and stability give different unavailability for "
                        f"{language.code!r}"
                    )
                    raise ValueError(msg)

                if not isinstance(diagonal, UnavailableAlignmentCell):
                    msg = (
                        f"the diagonal for unavailable language {language.code!r} is "
                        "not unavailable"
                    )
                    raise TypeError(msg)

                if (
                    diagonal.left != language
                    or diagonal.right != language
                    or diagonal.reasons != (coverage,)
                ):
                    msg = (
                        f"the diagonal for {language.code!r} does not carry its exact "
                        "unavailability"
                    )
                    raise ValueError(msg)

            else:
                msg = f"alignment coverage {i} is neither coverage nor unavailability"
                raise TypeError(msg)

            for j in range(size):
                cell = held_cells[i][j]
                left = held_languages[i]
                right = held_languages[j]

                if not isinstance(
                    cell,
                    CrosslingualAlignmentCell
                    | CrosslingualAlignmentDiagonal
                    | UnavailableAlignmentCell,
                ):
                    msg = f"alignment matrix cell ({i}, {j}) is not one: {cell!r}"
                    raise TypeError(msg)

                endpoints = (cell.left, cell.right)

                if i == j:
                    exact_endpoints = endpoints == (left, right)
                else:
                    exact_endpoints = endpoints in ((left, right), (right, left))

                if not exact_endpoints:
                    msg = (
                        f"alignment matrix cell ({i}, {j}) has endpoints "
                        f"{cell.left.code!r}, {cell.right.code!r}, expected "
                        f"{left.code!r}, {right.code!r}"
                    )
                    raise ValueError(msg)

                if i == j:
                    if cell != held_cells[j][i]:
                        msg = f"an alignment matrix is asymmetric at ({i}, {j})"
                        raise ValueError(msg)

                    continue

                endpoint_positions = (codes[cell.left.code], codes[cell.right.code])
                unavailable = tuple(
                    held_coverages[position]
                    for position in endpoint_positions
                    if isinstance(held_coverages[position], UnavailableLanguage)
                )

                if unavailable:
                    if not isinstance(cell, UnavailableAlignmentCell):
                        msg = (
                            f"alignment matrix cell ({i}, {j}) has an unavailable "
                            "endpoint but reports an available result"
                        )
                        raise ValueError(msg)

                    if cell.reasons != unavailable:
                        msg = (
                            f"alignment matrix cell ({i}, {j}) does not carry its "
                            "endpoints' exact unavailability"
                        )
                        raise ValueError(msg)

                elif i != j:
                    if not isinstance(cell, CrosslingualAlignmentCell):
                        msg = (
                            f"alignment matrix cell ({i}, {j}) has two available "
                            "endpoints but no comparison"
                        )
                        raise ValueError(msg)

                    left_at, right_at = endpoint_positions
                    if (
                        cell.left_coverage != held_coverages[left_at]
                        or cell.right_coverage != held_coverages[right_at]
                        or cell.left_stability != held_stabilities[left_at]
                        or cell.right_stability != held_stabilities[right_at]
                    ):
                        msg = (
                            f"alignment matrix cell ({i}, {j}) does not carry its "
                            "endpoints' exact coverage and stability"
                        )
                        raise ValueError(msg)

                    unknown = sorted(set(cell.shared) - set(references))

                    if unknown:
                        msg = (
                            f"alignment matrix cell ({i}, {j}) shares labels outside "
                            f"the reference inventory: {unknown!r}"
                        )
                        raise ValueError(msg)

                    if cell.floor.universe != len(references):
                        msg = (
                            f"alignment matrix cell ({i}, {j}) uses a Jaccard universe "
                            f"of {cell.floor.universe}, expected {len(references)}"
                        )
                        raise ValueError(msg)

                if cell != held_cells[j][i]:
                    msg = f"an alignment matrix is asymmetric at ({i}, {j})"
                    raise ValueError(msg)

        object.__setattr__(self, "_reference_labels", references)
        object.__setattr__(self, "_languages", held_languages)
        object.__setattr__(self, "_cells", held_cells)
        object.__setattr__(self, "_coverages", held_coverages)
        object.__setattr__(self, "_stabilities", held_stabilities)
        object.__setattr__(
            self,
            "_codes",
            MappingProxyType(
                {
                    language.code: position
                    for position, language in enumerate(held_languages)
                }
            ),
        )

    @property
    def reference_labels(self) -> tuple[str, ...]:
        """Return the one shared reference inventory in declared order."""
        return self._reference_labels

    @property
    def languages(self) -> tuple[TransferLanguage, ...]:
        """Return matrix languages in declared order."""
        return self._languages

    @property
    def cells(self) -> tuple[tuple[_AlignmentMatrixCell, ...], ...]:
        """Return the complete symmetric rows."""
        return self._cells

    @property
    def coverages(self) -> tuple[_CoverageEntry, ...]:
        """Return per-language coverage or typed unavailability."""
        return self._coverages

    @property
    def stabilities(self) -> tuple[_StabilityEntry, ...]:
        """Return per-language stability or typed unavailability."""
        return self._stabilities

    def cell(self, left: str, right: str, /) -> _AlignmentMatrixCell:
        """Return one cell by language code."""
        try:
            i = self._codes[left]
            j = self._codes[right]
        except KeyError as error:
            msg = f"the alignment matrix has no language {error.args[0]!r}"
            raise KeyError(msg) from error

        return self._cells[i][j]

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the complete matrix as deterministic JSON-ready values."""
        language_rows: list[JsonValue] = []

        for language, coverage, stability in zip(
            self._languages,
            self._coverages,
            self._stabilities,
            strict=True,
        ):
            if isinstance(coverage, UnavailableLanguage):
                language_rows.append(
                    {
                        **language.to_spec(),
                        "coverage": None,
                        "stability": None,
                        "unavailable": coverage.reason,
                    }
                )
            else:
                if not isinstance(stability, AlignmentStability):
                    msg = "an available language lost its alignment stability"
                    raise TypeError(msg)

                language_rows.append(
                    {
                        **language.to_spec(),
                        "coverage": _coverage_spec(coverage),
                        "stability": stability.to_spec(),
                        "unavailable": None,
                    }
                )

        return {
            "reference_labels": list(self._reference_labels),
            "languages": language_rows,
            "cells": [[_cell_spec(cell) for cell in row] for row in self._cells],
        }

    def _fields(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[TransferLanguage, ...],
        tuple[tuple[_AlignmentMatrixCell, ...], ...],
        tuple[_CoverageEntry, ...],
        tuple[_StabilityEntry, ...],
    ]:
        return (
            self._reference_labels,
            self._languages,
            self._cells,
            self._coverages,
            self._stabilities,
        )

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not CrosslingualAlignmentMatrix:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((CrosslingualAlignmentMatrix, self._fields()))

    @override
    def __repr__(self) -> str:
        codes = tuple(language.code for language in self._languages)

        return (
            f"CrosslingualAlignmentMatrix(reference_labels="
            f"{self._reference_labels!r}, languages={codes!r})"
        )


def _coverage_spec(coverage: AlignmentCoverage) -> dict[str, JsonValue]:
    """Return every field of one language's coverage."""
    return {
        "language": coverage.language.code,
        "aligned": coverage.aligned,
        "total": coverage.total,
        "proportion": coverage.proportion,
    }


def _stability_spec(stability: AlignmentStability) -> dict[str, JsonValue]:
    """Return one alignment stability's public report values."""
    return stability.to_spec()


def _permutation_spec(permutation: PValue) -> dict[str, JsonValue]:
    """Return a label-permutation result and all of its null provenance."""
    provenance = permutation.provenance

    return {
        "value": permutation.value,
        "null": provenance.null,
        "family": provenance.family,
        "replicates": provenance.replicates,
        "tail": provenance.tail,
        "exact": provenance.exact,
        "seed": provenance.seed,
        "matching": dict(sorted(provenance.matching.items())),
    }


def _cell_spec(cell: _AlignmentMatrixCell) -> dict[str, JsonValue]:
    """Return one matrix cell with every value its variant reports."""
    common: dict[str, JsonValue] = {
        "left": cell.left.code,
        "right": cell.right.code,
        "available": cell.available,
        "diagonal": cell.diagonal,
    }

    if isinstance(cell, CrosslingualAlignmentCell):
        floor = cell.floor

        return {
            **common,
            "shared": list(cell.shared),
            "jaccard": cell.jaccard,
            "jaccard_floor": {
                "universe": floor.universe,
                "left": floor.left,
                "right": floor.right,
                "expected": floor.expected,
                "ci_low": floor.interval.low,
                "ci_high": floor.interval.high,
                "ci_level": floor.interval.level,
            },
            "procrustes_cosine": cell.procrustes_cosine,
            "permutation": _permutation_spec(cell.permutation),
            "left_coverage": _coverage_spec(cell.left_coverage),
            "right_coverage": _coverage_spec(cell.right_coverage),
            "left_stability": _stability_spec(cell.left_stability),
            "right_stability": _stability_spec(cell.right_stability),
        }

    if isinstance(cell, CrosslingualAlignmentDiagonal):
        return {
            **common,
            "jaccard": cell.jaccard,
            "procrustes_cosine": cell.procrustes_cosine,
            "coverage": _coverage_spec(cell.coverage),
            "stability": _stability_spec(cell.stability),
        }

    return {
        **common,
        "reasons": [
            {
                **reason.language.to_spec(),
                "reason": reason.reason,
            }
            for reason in cell.reasons
        ],
    }


def _reference_centroids(entry: LanguageAlignment) -> Mapping[str, Tensor]:
    """Return mean induced vectors keyed by their aligned reference labels."""
    grouped: dict[str, list[Tensor]] = {}

    for induced, reference in entry.alignment.mapping.items():
        grouped.setdefault(reference, []).append(entry.vector(induced))

    centroids: dict[str, Tensor] = {}

    for reference, vectors in grouped.items():
        centroid = torch.stack(vectors).mean(dim=0)

        if bool(centroid.norm() == 0.0):
            msg = (
                f"the aligned vectors for reference label {reference!r} in "
                f"{entry.language.code!r} have a zero centroid"
            )
            raise ValueError(msg)

        centroids[reference] = centroid

    return MappingProxyType(centroids)


def _procrustes_mean_row_cosine(left: Tensor, right: Tensor) -> float:
    """Return uncentered O(d) Procrustes mean row cosine."""
    left_singular, _, right_singular = torch.linalg.svd(
        left.mT @ right,
        full_matrices=True,
    )
    rotation = left_singular @ right_singular
    aligned = left @ rotation
    cosine = (aligned * right).sum(dim=1) / (aligned.norm(dim=1) * right.norm(dim=1))

    return float(cosine.clamp(-1.0, 1.0).mean())


def _comparison(
    left: LanguageAlignment,
    right: LanguageAlignment,
    references: tuple[str, ...],
    null: LabelPermutation[int],
    level: float,
) -> CrosslingualAlignmentCell:
    """Return one canonical available off-diagonal cell."""
    left_vectors = _reference_centroids(left)
    right_vectors = _reference_centroids(right)
    left_image = frozenset(left_vectors)
    right_image = frozenset(right_vectors)
    shared = tuple(
        label for label in references if label in left_image and label in right_image
    )

    if not shared:
        msg = (
            f"the languages {left.language.code!r} and {right.language.code!r} "
            "share no aligned reference label"
        )
        raise ValueError(msg)

    if len(shared) < 2:
        msg = (
            f"the label-permutation null for {left.language.code!r} and "
            f"{right.language.code!r} needs at least two shared aligned labels"
        )
        raise ValueError(msg)

    left_rows = torch.stack([left_vectors[label] for label in shared])
    right_rows = torch.stack([right_vectors[label] for label in shared])
    order = tuple(range(len(shared)))
    statistics: dict[tuple[int, ...], float] = {}

    def statistic(permutation: Sequence[int]) -> float:
        key = tuple(permutation)
        found = statistics.get(key)

        if found is not None:
            return found

        index = torch.tensor(key, dtype=torch.int64)
        measured = _procrustes_mean_row_cosine(left_rows, right_rows[index])
        statistics[key] = measured

        return measured

    permutation = null.test(statistic, order, tail="greater")
    overlap = len(left_image & right_image)
    union = len(left_image | right_image)

    return CrosslingualAlignmentCell(
        left.language,
        right.language,
        shared=shared,
        jaccard=overlap / union,
        floor=hypergeometric_jaccard_floor(
            len(references),
            len(left_image),
            len(right_image),
            level=level,
        ),
        procrustes_cosine=statistic(order),
        permutation=permutation,
        left_coverage=left.coverage,
        right_coverage=right.coverage,
        left_stability=left.stability,
        right_stability=right.stability,
    )


def _references(labels: Sequence[str]) -> tuple[str, ...]:
    """Return a nonempty unique reference-label tuple."""
    held = tuple(ensure_name("a reference label", label) for label in labels)

    if not held:
        msg = "a cross-lingual comparison has no reference labels"
        raise ValueError(msg)

    if len(set(held)) != len(held):
        repeated = sorted(label for label in set(held) if held.count(label) > 1)
        msg = f"the shared reference inventory repeats labels {repeated!r}"
        raise ValueError(msg)

    return held


def crosslingual_alignment_matrix(
    evaluations: Sequence[LanguageAlignment | UnavailableLanguage],
    /,
    *,
    reference_labels: Sequence[str],
    null: LabelPermutation[int],
    level: float,
) -> CrosslingualAlignmentMatrix:
    """Compare available languages and propagate typed unavailability.

    Every available alignment must be fitted against exactly the declared
    reference inventory and have the same vector width. The matrix retains the
    caller's language order, computes each available pair once, and places that
    one cell at both symmetric addresses. Each cell's Jaccard floor reports its
    interval at the declared coverage, and its permutation test draws the
    replicates the null declares.
    """
    references = _references(reference_labels)

    if not isinstance(null, LabelPermutation):
        msg = f"the embedding null is not a LabelPermutation: {null!r}"
        raise TypeError(msg)

    held = tuple(evaluations)

    if not held:
        msg = "a cross-lingual comparison declares no language"
        raise ValueError(msg)

    languages: list[TransferLanguage] = []
    seen: set[str] = set()
    width: int | None = None

    for entry in held:
        if not isinstance(entry, LanguageAlignment | UnavailableLanguage):
            msg = (
                f"a cross-lingual evaluation is neither available nor typed: {entry!r}"
            )
            raise TypeError(msg)

        language = entry.language

        if language.code in seen:
            msg = f"the cross-lingual matrix repeats language {language.code!r}"
            raise ValueError(msg)

        seen.add(language.code)
        languages.append(language)

        if isinstance(entry, UnavailableLanguage):
            continue

        actual = entry.alignment.contingency.gold_labels

        if set(actual) != set(references):
            missing = sorted(set(references) - set(actual))
            extra = sorted(set(actual) - set(references))
            msg = (
                f"the alignment for {language.code!r} uses another reference "
                f"inventory: missing {missing!r}, extra {extra!r}"
            )
            raise ValueError(msg)

        if not entry.alignment.mapping:
            msg = f"the alignment for {language.code!r} has an empty reference image"
            raise ValueError(msg)

        if width is not None and entry.width != width:
            msg = (
                f"the vectors for {language.code!r} have width {entry.width}, "
                f"while the shared width is {width}"
            )
            raise ValueError(msg)

        width = entry.width

    size = len(held)
    cells: list[list[_AlignmentMatrixCell | None]] = [
        [None for _ in range(size)] for _ in range(size)
    ]
    coverages: list[_CoverageEntry] = []
    stabilities: list[_StabilityEntry] = []

    for entry in held:
        if isinstance(entry, UnavailableLanguage):
            coverages.append(entry)
            stabilities.append(entry)
        else:
            coverages.append(entry.coverage)
            stabilities.append(entry.stability)

    for i, left in enumerate(held):
        if isinstance(left, UnavailableLanguage):
            cells[i][i] = UnavailableAlignmentCell(
                left.language,
                left.language,
                (left,),
            )
        else:
            cells[i][i] = CrosslingualAlignmentDiagonal(
                left.language,
                left.coverage,
                left.stability,
            )

        for j in range(i + 1, size):
            right = held[j]
            unavailable = tuple(
                entry
                for entry in (left, right)
                if isinstance(entry, UnavailableLanguage)
            )
            cell: _AlignmentMatrixCell

            if unavailable:
                cell = UnavailableAlignmentCell(
                    left.language,
                    right.language,
                    unavailable,
                )
            else:
                if not isinstance(left, LanguageAlignment) or not isinstance(
                    right, LanguageAlignment
                ):
                    msg = "an available comparison lost its language alignment"
                    raise RuntimeError(msg)

                cell = _comparison(left, right, references, null, level)

            cells[i][j] = cell
            cells[j][i] = cell

    complete: list[tuple[_AlignmentMatrixCell, ...]] = []

    for row in cells:
        if any(cell is None for cell in row):
            msg = "a cross-lingual alignment matrix has an incomplete cell"
            raise RuntimeError(msg)

        complete.append(tuple(cell for cell in row if cell is not None))

    return CrosslingualAlignmentMatrix(
        references,
        languages,
        complete,
        coverages,
        stabilities,
    )

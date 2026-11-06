"""Tests for cross-lingual inventory and embedding comparison."""

import json
from copy import copy, deepcopy
from itertools import combinations
from math import ceil
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._crosslingual_alignment import (
    AlignmentCoverage,
    CrosslingualAlignmentCell,
    CrosslingualAlignmentDiagonal,
    CrosslingualAlignmentMatrix,
    LanguageAlignment,
    UnavailableAlignmentCell,
    crosslingual_alignment_matrix,
    hypergeometric_jaccard_floor,
)
from lofi.eval._stability import AlignmentStability
from lofi.eval._transfer import TransferLanguage, UnavailableLanguage
from lofi.eval._values import MetricResult, Resampling
from lofi.stats import Interval, LabelPermutation

if TYPE_CHECKING:
    from lofi._types import JsonValue

REFERENCES: Final = ("A", "B", "C", "D")
ENGLISH: Final = TransferLanguage("en", "gold")
GERMAN: Final = TransferLanguage("de", "gold")
DUTCH: Final = TransferLanguage("nl", "gold")
SPANISH: Final = TransferLanguage("es", "silver")
NO_TREEBANK: Final = UnavailableLanguage(SPANISH, "no converted treebank")
SEED: Final = 19

ENGLISH_VECTORS: Final = {
    "en-a": torch.tensor([1.0, 0.0]),
    "en-b": torch.tensor([0.0, 1.0]),
    "en-c": torch.tensor([1.0, 1.0]),
}
GERMAN_VECTORS: Final = {
    "de-a": torch.tensor([-1.0, 0.0]),
    "de-b": torch.tensor([0.0, 1.0]),
    "de-c": torch.tensor([-1.0, 1.0]),
}


def table(
    induced: tuple[str, ...],
    references: tuple[str, ...] = REFERENCES,
) -> Contingency:
    return Contingency(
        {
            (label, reference): int(position == reference_position)
            for position, label in enumerate(induced)
            for reference_position, reference in enumerate(references)
        },
        split="dev",
    )


def stability(induced: tuple[str, ...]) -> AlignmentStability:
    return AlignmentStability(
        MetricResult(
            0.9,
            4,
            Resampling("seed", 5),
            ci=Interval(0.8, 1.0, 0.95),
        ),
        dict.fromkeys(induced, 0.0),
        labels=induced,
        dropped=(),
        seeds=4,
    )


def aligned(
    language: TransferLanguage,
    mapping: dict[str, str],
    vectors: dict[str, torch.Tensor],
    *,
    references: tuple[str, ...] = REFERENCES,
    induced: tuple[str, ...] | None = None,
) -> LanguageAlignment:
    labels = tuple(mapping) if induced is None else induced

    return LanguageAlignment(
        language,
        Alignment(mapping, table(labels, references)),
        stability(labels),
        vectors,
    )


def english() -> LanguageAlignment:
    return aligned(
        ENGLISH,
        {"en-a": "A", "en-b": "B", "en-c": "C"},
        ENGLISH_VECTORS,
    )


def german() -> LanguageAlignment:
    return aligned(
        GERMAN,
        {"de-a": "A", "de-b": "B", "de-c": "C"},
        GERMAN_VECTORS,
    )


def matrix(
    *evaluations: LanguageAlignment | UnavailableLanguage,
    references: tuple[str, ...] = REFERENCES,
) -> CrosslingualAlignmentMatrix:
    return crosslingual_alignment_matrix(
        evaluations,
        reference_labels=references,
        null=LabelPermutation(seed=SEED, replicates=999),
        level=0.95,
    )


def available(
    held: CrosslingualAlignmentCell
    | CrosslingualAlignmentDiagonal
    | UnavailableAlignmentCell,
) -> CrosslingualAlignmentCell:
    assert isinstance(held, CrosslingualAlignmentCell)

    return held


def test_identical_reference_images_and_an_orthogonal_transform_score_one() -> None:
    report = matrix(english(), german())
    cell = available(report.cell("en", "de"))

    assert cell.jaccard == 1.0
    assert cell.procrustes_cosine == pytest.approx(1.0, abs=1e-12)
    assert cell.shared == ("A", "B", "C")
    assert cell.left_stability is report.stabilities[0]
    assert cell.right_stability is report.stabilities[1]
    assert cell.left_coverage == AlignmentCoverage(ENGLISH, 3, 3)
    assert cell.right_coverage == AlignmentCoverage(GERMAN, 3, 3)
    assert cell.permutation.provenance.null == "label permutation"
    assert cell.permutation.provenance.family == "permutation"
    assert cell.permutation.provenance.tail == "greater"
    assert cell.permutation.provenance.replicates == 999
    assert cell.permutation.provenance.seed == SEED


def test_inventory_jaccard_uses_reference_images_and_fixed_cardinalities() -> None:
    dutch = aligned(
        DUTCH,
        {"nl-b": "B", "nl-c": "C", "nl-d": "D"},
        {
            "nl-b": torch.tensor([1.0, 0.0]),
            "nl-c": torch.tensor([0.0, 1.0]),
            "nl-d": torch.tensor([1.0, 1.0]),
        },
    )
    cell = available(matrix(english(), dutch).cell("en", "nl"))

    assert cell.shared == ("B", "C")
    assert cell.jaccard == 0.5
    assert cell.floor == hypergeometric_jaccard_floor(4, 3, 3, level=0.95)


def test_coordinated_induced_label_and_vector_row_permutations_change_nothing() -> None:
    original = available(matrix(english(), german()).cell("en", "de"))
    renamed = aligned(
        GERMAN,
        {"third": "C", "first": "A", "second": "B"},
        {
            "third": GERMAN_VECTORS["de-c"],
            "first": GERMAN_VECTORS["de-a"],
            "second": GERMAN_VECTORS["de-b"],
        },
    )
    permuted = available(matrix(english(), renamed).cell("en", "de"))

    assert permuted.shared == original.shared
    assert permuted.jaccard == original.jaccard
    assert permuted.floor == original.floor
    assert permuted.procrustes_cosine == pytest.approx(original.procrustes_cosine)
    assert permuted.permutation == original.permutation


def test_unaligned_primitives_affect_coverage_but_neither_statistic() -> None:
    low = aligned(
        DUTCH,
        {"nl-a": "A", "nl-b": "B", "nl-c": "C"},
        {
            "nl-a": ENGLISH_VECTORS["en-a"],
            "nl-b": ENGLISH_VECTORS["en-b"],
            "nl-c": ENGLISH_VECTORS["en-c"],
            "unaligned": torch.tensor([1.0, 1.0]),
        },
        induced=("nl-a", "nl-b", "nl-c", "unaligned"),
    )
    high = aligned(
        DUTCH,
        {"nl-a": "A", "nl-b": "B", "nl-c": "C"},
        {
            "nl-a": ENGLISH_VECTORS["en-a"],
            "nl-b": ENGLISH_VECTORS["en-b"],
            "nl-c": ENGLISH_VECTORS["en-c"],
            "unaligned": torch.tensor([1e9, -1e9]),
        },
        induced=("nl-a", "nl-b", "nl-c", "unaligned"),
    )
    first = available(matrix(english(), low).cell("en", "nl"))
    second = available(matrix(english(), high).cell("en", "nl"))

    assert first.right_coverage == AlignmentCoverage(DUTCH, 3, 4)
    assert first.jaccard == second.jaccard
    assert first.procrustes_cosine == second.procrustes_cosine
    assert first.permutation == second.permutation


def _enumerated_jaccards(universe: int, left: int, right: int) -> list[float]:
    values: list[float] = []
    labels = range(universe)

    for first in combinations(labels, left):
        first_set = set(first)

        for second in combinations(labels, right):
            second_set = set(second)
            union = first_set | second_set
            values.append(
                1.0 if not union else len(first_set & second_set) / len(union)
            )

    return sorted(values)


def test_the_hypergeometric_floor_agrees_with_exhaustive_small_universes() -> None:
    for universe in range(1, 7):
        for left in range(universe + 1):
            for right in range(universe + 1):
                values = _enumerated_jaccards(universe, left, right)
                floor = hypergeometric_jaccard_floor(universe, left, right, level=0.95)
                lower = values[ceil(0.025 * len(values)) - 1]
                upper = values[ceil(0.975 * len(values)) - 1]

                assert floor.expected == pytest.approx(sum(values) / len(values))
                assert floor.interval.low == pytest.approx(lower)
                assert floor.interval.high == pytest.approx(upper)
                assert floor.interval.level == 0.95


def test_the_matrix_is_complete_symmetric_and_in_declared_order() -> None:
    report = matrix(german(), english(), NO_TREEBANK)

    assert tuple(language.code for language in report.languages) == ("de", "en", "es")
    assert len(report.cells) == 3
    assert all(len(row) == 3 for row in report.cells)

    for i in range(3):
        for j in range(3):
            assert report.cells[i][j] is report.cells[j][i]

    assert isinstance(report.cell("de", "de"), CrosslingualAlignmentDiagonal)
    assert isinstance(report.cell("en", "en"), CrosslingualAlignmentDiagonal)
    assert isinstance(report.cell("es", "es"), UnavailableAlignmentCell)
    unavailable = cast("UnavailableAlignmentCell", report.cell("de", "es"))
    assert unavailable.reasons == (NO_TREEBANK,)
    assert report.coverages[2] is NO_TREEBANK
    assert report.stabilities[2] is NO_TREEBANK
    assert available(report.cell("de", "en")).permutation is not None


def test_matrix_values_are_immutable_canonical_snapshots() -> None:
    source = ENGLISH_VECTORS["en-a"].clone()
    entry = aligned(
        ENGLISH,
        {"en-a": "A", "en-b": "B", "en-c": "C"},
        {**ENGLISH_VECTORS, "en-a": source},
    )
    report = matrix(entry, german())
    before = entry.vector("en-a")
    source.fill_(99.0)

    assert torch.equal(entry.vector("en-a"), before)
    assert copy(report) is report
    assert deepcopy(report) is report
    assert report == matrix(entry, german())
    assert hash(report) == hash(matrix(entry, german()))
    assert "languages=('en', 'de')" in repr(report)


def test_a_wrong_reference_inventory_is_refused_by_language() -> None:
    wrong = aligned(
        ENGLISH,
        {"en-a": "A", "en-b": "B", "en-c": "C"},
        ENGLISH_VECTORS,
        references=("A", "B", "C"),
    )

    with pytest.raises(ValueError, match=r"en.*reference inventory"):
        matrix(wrong, german())


def test_duplicate_languages_are_refused_by_code() -> None:
    another = aligned(
        ENGLISH,
        {"other-a": "A", "other-b": "B", "other-c": "C"},
        {
            "other-a": ENGLISH_VECTORS["en-a"],
            "other-b": ENGLISH_VECTORS["en-b"],
            "other-c": ENGLISH_VECTORS["en-c"],
        },
    )

    with pytest.raises(ValueError, match="repeats language 'en'"):
        matrix(english(), another)


def test_duplicate_reference_labels_are_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"repeats labels.*'A'"):
        matrix(english(), german(), references=("A", "B", "A", "D"))


def test_vector_width_mismatch_is_refused_by_language() -> None:
    wide = aligned(
        GERMAN,
        {"de-a": "A", "de-b": "B", "de-c": "C"},
        {
            "de-a": torch.tensor([1.0, 0.0, 0.0]),
            "de-b": torch.tensor([0.0, 1.0, 0.0]),
            "de-c": torch.tensor([0.0, 0.0, 1.0]),
        },
    )

    with pytest.raises(ValueError, match=r"de.*width 3.*shared width is 2"):
        matrix(english(), wide)


def test_a_missing_vector_is_refused_by_language_and_label() -> None:
    with pytest.raises(ValueError, match=r"en.*no vector.*en-c"):
        aligned(
            ENGLISH,
            {"en-a": "A", "en-b": "B", "en-c": "C"},
            {"en-a": torch.ones(2), "en-b": torch.ones(2)},
        )


@pytest.mark.parametrize(
    "vector",
    [
        torch.zeros(2),
        torch.tensor([1.0, float("nan")]),
        torch.ones(1, 2),
        torch.ones(2, dtype=torch.int64),
    ],
)
def test_a_malformed_vector_is_refused_by_language_and_label(
    vector: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"en-a.*en"):
        aligned(
            ENGLISH,
            {"en-a": "A", "en-b": "B", "en-c": "C"},
            {**ENGLISH_VECTORS, "en-a": vector},
        )


def test_missing_alignment_stability_is_refused_by_language() -> None:
    labels = ("en-a", "en-b", "en-c")

    with pytest.raises(TypeError, match=r"en.*AlignmentStability"):
        LanguageAlignment(
            ENGLISH,
            Alignment(
                {"en-a": "A", "en-b": "B", "en-c": "C"},
                table(labels),
            ),
            cast("AlignmentStability", None),
            ENGLISH_VECTORS,
        )


def test_alignment_stability_must_belong_to_the_exact_induced_inventory() -> None:
    labels = ("en-a", "en-b", "en-c")
    alignment = Alignment(
        {"en-a": "A", "en-b": "B", "en-c": "C"},
        table(labels),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"stability.*en.*exact induced labels.*missing \['en-c'\]"
            r".*extra \['other'\]"
        ),
    ):
        LanguageAlignment(
            ENGLISH,
            alignment,
            stability(("en-a", "en-b", "other")),
            ENGLISH_VECTORS,
        )


def test_public_matrix_constructor_validates_inventories_and_exact_endpoints() -> None:
    report = matrix(english(), german())

    with pytest.raises(ValueError, match="no reference labels"):
        CrosslingualAlignmentMatrix(
            (),
            report.languages,
            report.cells,
            report.coverages,
            report.stabilities,
        )

    with pytest.raises(ValueError, match=r"repeats labels.*'A'"):
        CrosslingualAlignmentMatrix(
            ("A", "A"),
            report.languages,
            report.cells,
            report.coverages,
            report.stabilities,
        )

    with pytest.raises(ValueError, match="declares no language"):
        CrosslingualAlignmentMatrix(REFERENCES, (), (), (), ())

    with pytest.raises(ValueError, match="repeats language 'en'"):
        CrosslingualAlignmentMatrix(
            REFERENCES,
            (ENGLISH, ENGLISH),
            report.cells,
            report.coverages,
            report.stabilities,
        )

    with pytest.raises(ValueError, match=r"coverage 0 belongs to 'de'.*not 'en'"):
        CrosslingualAlignmentMatrix(
            REFERENCES,
            report.languages,
            report.cells,
            tuple(reversed(report.coverages)),
            report.stabilities,
        )

    wrong_endpoint = (
        (report.cells[0][0], report.cells[0][0]),
        (report.cells[0][0], report.cells[1][1]),
    )

    with pytest.raises(ValueError, match=r"cell \(0, 1\).*endpoints"):
        CrosslingualAlignmentMatrix(
            REFERENCES,
            report.languages,
            wrong_endpoint,
            report.coverages,
            report.stabilities,
        )


def test_matrix_serialization_emits_every_declared_value_in_matrix_order() -> None:
    report = matrix(german(), english(), NO_TREEBANK)
    serialized = report.to_spec()
    available_cell = available(report.cell("de", "en"))
    cell = cast("list[list[dict[str, JsonValue]]]", serialized["cells"])[0][1]
    language_rows = cast("list[dict[str, JsonValue]]", serialized["languages"])
    permutation = available_cell.permutation.provenance

    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["reference_labels"] == list(report.reference_labels)
    assert [row["language"] for row in language_rows] == ["de", "en", "es"]
    assert language_rows[0]["coverage"] == {
        "language": "de",
        "aligned": available_cell.left_coverage.aligned,
        "total": available_cell.left_coverage.total,
        "proportion": available_cell.left_coverage.proportion,
    }
    assert language_rows[0]["stability"] == available_cell.left_stability.to_spec()
    assert language_rows[2]["unavailable"] == NO_TREEBANK.reason
    assert cell["shared"] == list(available_cell.shared)
    assert cell["jaccard"] == available_cell.jaccard
    assert cell["procrustes_cosine"] == available_cell.procrustes_cosine
    assert cell["left_coverage"] == language_rows[0]["coverage"]
    assert cell["right_coverage"] == language_rows[1]["coverage"]
    assert cell["permutation"] == {
        "value": available_cell.permutation.value,
        "null": permutation.null,
        "family": permutation.family,
        "replicates": permutation.replicates,
        "tail": permutation.tail,
        "exact": permutation.exact,
        "seed": permutation.seed,
        "matching": dict(permutation.matching),
    }
    unavailable = cast("list[list[dict[str, JsonValue]]]", serialized["cells"])[2][2]
    assert unavailable["reasons"] == [
        {**SPANISH.to_spec(), "reason": NO_TREEBANK.reason}
    ]


def test_an_empty_shared_alignment_is_refused_by_both_languages() -> None:
    left = aligned(
        ENGLISH,
        {"en-a": "A", "en-b": "B"},
        {"en-a": torch.tensor([1.0, 0.0]), "en-b": torch.tensor([0.0, 1.0])},
    )
    right = aligned(
        GERMAN,
        {"de-c": "C", "de-d": "D"},
        {"de-c": torch.tensor([1.0, 0.0]), "de-d": torch.tensor([0.0, 1.0])},
    )

    with pytest.raises(ValueError, match=r"en.*de.*share no aligned"):
        matrix(left, right)


def test_the_declared_null_and_coverage_reach_each_cell() -> None:
    report = crosslingual_alignment_matrix(
        (english(), german()),
        reference_labels=REFERENCES,
        null=LabelPermutation(replicates=17, seed=SEED),
        level=0.8,
    )
    cell = available(report.cell("en", "de"))

    assert cell.permutation.provenance.replicates == 17
    assert cell.floor.interval.level == 0.8
    assert cell.floor == hypergeometric_jaccard_floor(4, 3, 3, level=0.8)


def test_a_wider_coverage_widens_the_jaccard_floor() -> None:
    narrow = hypergeometric_jaccard_floor(40, 10, 10, level=0.5).interval
    wide = hypergeometric_jaccard_floor(40, 10, 10, level=0.99).interval

    assert wide.low <= narrow.low
    assert wide.high >= narrow.high
    assert (wide.low, wide.high) != (narrow.low, narrow.high)


def test_a_canceling_reference_centroid_is_refused_by_language_and_label() -> None:
    canceling = aligned(
        GERMAN,
        {"positive": "A", "negative": "A", "de-b": "B"},
        {
            "positive": torch.tensor([1.0, 0.0]),
            "negative": torch.tensor([-1.0, 0.0]),
            "de-b": torch.tensor([0.0, 1.0]),
        },
    )
    left = aligned(
        ENGLISH,
        {"en-a": "A", "en-b": "B"},
        {"en-a": torch.tensor([1.0, 0.0]), "en-b": torch.tensor([0.0, 1.0])},
    )

    with pytest.raises(ValueError, match=r"reference label 'A'.*de.*zero centroid"):
        matrix(left, canceling)

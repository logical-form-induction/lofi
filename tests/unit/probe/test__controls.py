from random import random
from random import seed as global_seed
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._causal import (
    AlignmentPolicy,
    CombinatorAt,
    CounterfactualPair,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._controls import (
    BinaryBracketing,
    BracketingDelta,
    BracketingUnavailable,
    ControlCell,
    ControlReport,
    DeltaIIA,
    DormantMovement,
    DormantPathway,
    IIAPopulation,
    ScrambledDelta,
    TypeScramble,
    bracketing_distance,
    delta_iia,
    matched_binary_bracketings,
    random_binary_bracketing,
    scramble_types,
)
from lofi.probe._das import AlignmentObjective, FixedRankSearch, Schedule
from lofi.probe._ports import ModelReadyCausalPair
from lofi.probe._pricing import CostModel, ProbeUnit, UnitPrice
from lofi.probe._provenance import Provenance
from lofi.probe._sweep import GridCell, Preregistration, Subgrid
from lofi.stats import Bootstrap, LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.deduce import Cell
    from lofi.probe._causal import CellAssignment, HighLevelVariable

LEFT: Final = TypeAt((0, 1))
RIGHT: Final = TypeAt((1, 2))
ROOT_SPLIT: Final = SplitAt((0, 2))
ROOT_RULE: Final = CombinatorAt((0, 2), 1)
POLICY: Final = AlignmentPolicy("first", "exclude", contiguous=True)
PROVENANCE: Final = Provenance("fixture", "controls", 0)
NAMED: Final = NamedVariable("external", "category")
SITE: Final = Site(ComponentId("model"), Role.RESIDUAL_STREAM, (0,))


class _ReadyPair:
    __slots__ = ("_id", "_pair", "_value")

    def __init__(
        self,
        identity: str,
        base_left: str,
        variant_left: str,
        right: str,
        value: int,
    ) -> None:
        base: CellAssignment = {
            LEFT: base_left,
            RIGHT: right,
            ROOT_SPLIT: 1,
            ROOT_RULE: "application",
        }
        variant: CellAssignment = dict(base)
        variant[LEFT] = variant_left
        self._id = identity
        self._pair = CounterfactualPair(base, variant, LEFT, LEFT.cell, PROVENANCE)
        self._value = value

    @property
    def id(self) -> str:
        return self._id

    @property
    def tokens(self) -> tuple[str, ...]:
        return ("one", "two")

    @property
    def base(self) -> CellAssignment:
        return self._pair.base

    @property
    def variant(self) -> CellAssignment:
        return self._pair.variant

    @property
    def varied(self) -> HighLevelVariable:
        return self._pair.varied

    @property
    def span(self) -> Cell:
        return self._pair.span

    @property
    def provenance(self) -> Provenance:
        return self._pair.provenance

    @property
    def policy(self) -> AlignmentPolicy:
        return POLICY

    @property
    def tokenizer(self) -> str:
        return "fixture-tokenizer"

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        offset = 0 if side is Side.BASE else 10
        return {"input_ids": torch.tensor([[self._value + offset]])}

    def rows(self, side: Side, /) -> tuple[int, ...]:
        return (self._value,) if side is Side.BASE else (self._value + 1,)

    def live_rows(self, side: Side, /) -> int:
        del side
        return 20


def population() -> tuple[_ReadyPair, ...]:
    return (
        _ReadyPair("p0", "N", "NP", "S", 0),
        _ReadyPair("p1", "N", "S", "NP", 2),
        _ReadyPair("p2", "NP", "PP", "N", 4),
        _ReadyPair("p3", "S", "N", "PP", 6),
    )


class _NamedReadyPair(_ReadyPair):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("named", "N", "NP", "S", 0)
        self._pair = CounterfactualPair(
            {NAMED: "N"}, {NAMED: "NP"}, NAMED, (0, 1), PROVENANCE
        )


def test_type_scrambling_moves_complete_profiles_without_touching_model_rows() -> None:
    original = population()
    result = scramble_types(original, seed=17)
    by_id = {pair.id: pair for pair in original}

    assert tuple(pair.id for pair in result.pairs) == tuple(
        pair.id for pair in original
    )
    assert set(result.manifest) == {pair.id for pair in original}
    assert all(target != donor for target, donor in result.manifest.items())
    for target, changed in zip(original, result.pairs, strict=True):
        donor = by_id[result.manifest[target.id]]
        assert changed.base[LEFT] == donor.base[LEFT]
        assert changed.variant[LEFT] == donor.variant[LEFT]
        assert changed.base[RIGHT] == donor.base[RIGHT]
        assert changed.variant[RIGHT] == donor.variant[RIGHT]
        assert changed.base[ROOT_SPLIT] == target.base[ROOT_SPLIT]
        assert changed.base[ROOT_RULE] == target.base[ROOT_RULE]
        assert changed.span == target.span
        assert changed.varied == target.varied
        assert changed.tokens == target.tokens
        assert changed.policy == target.policy
        assert changed.tokenizer == target.tokenizer
        assert changed.rows(Side.BASE) == target.rows(Side.BASE)
        assert changed.rows(Side.VARIANT) == target.rows(Side.VARIANT)
        assert changed.live_rows(Side.BASE) == target.live_rows(Side.BASE)
        assert torch.equal(
            changed.inputs(Side.BASE)["input_ids"],
            target.inputs(Side.BASE)["input_ids"],
        )


def test_type_scrambling_is_local_deterministic_and_not_a_global_rename() -> None:
    global_seed(193)
    before = random()
    first = scramble_types(population(), seed=5)
    after = random()
    global_seed(193)

    assert before == random()
    assert after == random()
    assert first.manifest == scramble_types(population(), seed=5).manifest
    relations: dict[str | int, set[str | int]] = {}
    for source, changed in zip(population(), first.pairs, strict=True):
        for variable in (LEFT, RIGHT):
            relations.setdefault(source.base[variable], set()).add(
                changed.base[variable]
            )
            relations.setdefault(source.variant[variable], set()).add(
                changed.variant[variable]
            )
    assert any(len(outputs) > 1 for outputs in relations.values())


def test_type_scrambling_refuses_singleton_unchanged_and_global_surrogates() -> None:
    lone = population()[0]
    with pytest.raises(ValueError, match="only one"):
        scramble_types((lone,), seed=0)

    duplicate = _ReadyPair("copy", "N", "NP", "S", 9)
    with pytest.raises(ValueError, match="no nonidentity"):
        scramble_types((lone, duplicate), seed=0)

    first = _ReadyPair("a", "N", "NP", "N", 0)
    second = _ReadyPair("b", "NP", "N", "NP", 2)
    with pytest.raises(ValueError, match="global label renaming"):
        TypeScramble((first, second), (second, first), seed=0)


def test_type_scrambling_refuses_named_variables_explicitly() -> None:
    with pytest.raises(TypeError, match="NamedVariable"):
        scramble_types((_NamedReadyPair(),), seed=0)


def test_binary_bracketings_validate_full_trees_and_expose_proper_widths() -> None:
    tree = BinaryBracketing(4, ((0, 4), (0, 3), (0, 2)))

    assert tree.spans == ((0, 4), (0, 3), (0, 2))
    assert tree.proper_widths == (3, 2)
    assert BinaryBracketing(1, ()) == BinaryBracketing(1, ())
    with pytest.raises(ValueError, match="internal constituents"):
        BinaryBracketing(4, ((0, 4),))
    with pytest.raises(ValueError, match=r"rooted tree|decompositions"):
        BinaryBracketing(4, ((0, 4), (0, 3), (1, 4)))


def test_catalan_draws_are_seeded_locally_and_cover_valid_full_trees() -> None:
    global_seed(811)
    before = random()
    draws = tuple(random_binary_bracketing(5, seed=value) for value in range(20))
    after = random()
    global_seed(811)

    assert before == random()
    assert after == random()
    assert random_binary_bracketing(5, seed=7) == random_binary_bracketing(5, seed=7)
    assert len(set(draws)) > 5
    assert all(len(draw.spans) == draw.length - 1 for draw in draws)
    assert all((0, draw.length) in draw.spans for draw in draws)


def test_root_splits_follow_the_catalan_subtree_weights() -> None:
    def root_split(tree: BinaryBracketing) -> int:
        spans = frozenset(tree.spans)
        return next(
            split
            for split in range(1, tree.length)
            if (split == 1 or (0, split) in spans)
            and (tree.length - split == 1 or (split, tree.length) in spans)
        )

    counts = {
        split: sum(
            root_split(random_binary_bracketing(4, seed=seed)) == split
            for seed in range(2_000)
        )
        for split in (1, 2, 3)
    }

    assert counts[1] / 2_000 == pytest.approx(2 / 5, abs=0.02)
    assert counts[2] / 2_000 == pytest.approx(1 / 5, abs=0.02)
    assert counts[3] / 2_000 == pytest.approx(2 / 5, abs=0.02)


def test_span_length_matching_returns_only_an_accepted_ordered_population() -> None:
    provisional = matched_binary_bracketings(
        (
            BinaryBracketing(4, ((0, 4), (0, 3), (0, 2))),
            BinaryBracketing(3, ((0, 3), (1, 3))),
        ),
        tolerance=1.0,
        seed=3,
        attempts=1,
    )
    assert isinstance(provisional, tuple)
    exact = matched_binary_bracketings(provisional, tolerance=0.0, seed=3, attempts=1)
    within = matched_binary_bracketings(provisional, tolerance=0.5, seed=4, attempts=20)

    assert exact == provisional
    assert isinstance(within, tuple)
    assert tuple(value.length for value in within) == (4, 3)
    assert bracketing_distance(provisional, within) <= 0.5


def test_span_length_matching_returns_one_typed_failure_not_an_unmatched_draw() -> None:
    reference = (BinaryBracketing(5, ((0, 5), (0, 4), (0, 3), (0, 2))),)
    failed = matched_binary_bracketings(reference, tolerance=0.0, seed=9, attempts=1)

    assert type(failed) is BracketingUnavailable
    assert failed.reason == "no-span-length-match"
    assert failed.lengths == (5,)
    assert failed.tolerance == 0.0
    assert failed.seed == 9
    assert failed.attempts == 1
    assert failed.closest > 0.0


def test_delta_iia_is_the_hand_derived_paired_difference_with_provenance() -> None:
    observed = IIAPopulation((("a", True), ("b", True), ("c", False), ("d", True)))
    control = IIAPopulation((("a", True), ("b", False), ("c", False), ("d", False)))
    result = delta_iia(
        observed,
        control,
        bootstrap=Bootstrap(unit="sentence", replicates=31, seed=7, level=0.95),
    )

    assert observed.accuracy == 0.75
    assert control.accuracy == 0.25
    assert result.headline == 0.5
    assert result.difference.observed == 0.5
    assert result.difference.unit == "sentence"
    assert result.difference.units == 4
    assert result.difference.replicates == 31
    assert result.difference.seed == 7


def test_delta_iia_pairs_by_canonical_stable_id_not_caller_order() -> None:
    observed = IIAPopulation((("b", False), ("a", True)))
    control = IIAPopulation((("a", False), ("b", False)))
    result = delta_iia(
        observed,
        control,
        bootstrap=Bootstrap(unit="sentence", replicates=7, seed=3, level=0.95),
    )

    assert observed.outcomes == (("a", True), ("b", False))
    assert control.outcomes == (("a", False), ("b", False))
    assert result.observed.ids == result.control.ids == ("a", "b")
    assert result.headline == 0.5


def test_delta_iia_refuses_mismatched_ids_and_an_inconsistent_aggregate() -> None:
    observed = IIAPopulation((("a", True), ("b", False)))
    mismatched = IIAPopulation((("a", False), ("c", False)))
    with pytest.raises(ValueError, match="identical stable identities"):
        delta_iia(
            observed,
            mismatched,
            bootstrap=Bootstrap(unit="item", replicates=3, seed=0, level=0.95),
        )
    control = IIAPopulation((("a", False), ("b", False)))
    wrong = Bootstrap(unit="item", replicates=3, seed=0, level=0.95).interval(
        (0.0, 0.0), sum
    )
    with pytest.raises(ValueError, match="aggregate disagrees"):
        DeltaIIA(observed, control, wrong)


def test_control_specific_deltas_bind_the_population_that_licenses_them() -> None:
    scramble = scramble_types(population(), seed=17)
    observed = IIAPopulation((("p0", True), ("p1", True), ("p2", False), ("p3", True)))
    control = IIAPopulation((("p0", False), ("p1", True), ("p2", False), ("p3", False)))
    compared = delta_iia(
        observed,
        control,
        bootstrap=Bootstrap(unit="sentence", replicates=7, seed=0, level=0.95),
    )
    scrambled = ScrambledDelta(scramble, compared)
    reference = tuple(BinaryBracketing(4, ((0, 4), (0, 3), (0, 2))) for _ in range(4))
    matched = tuple(BinaryBracketing(4, ((0, 4), (1, 4), (2, 4))) for _ in range(4))
    bracketed = BracketingDelta(reference, matched, compared, tolerance=0.0)

    assert scrambled.headline == 0.5
    assert scrambled.scramble is scramble
    assert bracketed.headline == 0.5
    assert bracketed.distance == 0.0
    assert bracketed.tolerance == 0.0
    assert bracketed.reference == reference
    assert bracketed.control == matched

    unmatched = tuple(BinaryBracketing(4, ((0, 4), (0, 2), (2, 4))) for _ in range(4))
    with pytest.raises(ValueError, match="exceeds"):
        BracketingDelta(reference, unmatched, compared, tolerance=0.0)


def a_preregistration(*, two_cells: bool = False) -> Preregistration:
    layers = (0, 1) if two_cells else (0,)
    sites = {
        (layer, 0): Site(ComponentId("model"), Role.RESIDUAL_STREAM, (layer,))
        for layer in layers
    }
    subgrid = Subgrid((LEFT,), layers, (0,), sites)
    search = FixedRankSearch(
        rank=1,
        schedule=Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
        null=LabelPermutation(replicates=1, seed=0),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    costs = CostModel(
        (
            UnitPrice(
                ProbeUnit.FIXED_ALIGNMENT_FIT,
                seconds_per_run=1.0,
                seconds_fastest=0.5,
                warmup=1,
                repeats=2,
                device="cpu",
            ),
            UnitPrice(
                ProbeUnit.FIXED_PERMUTATION_REFIT,
                seconds_per_run=1.0,
                seconds_fastest=0.5,
                warmup=1,
                repeats=2,
                device="cpu",
            ),
        )
    )
    return Preregistration(subgrid, search, costs=costs, budget_hours=1.0)


def a_delta() -> DeltaIIA:
    observed = IIAPopulation((("a", True), ("b", False)))
    control = IIAPopulation((("a", False), ("b", False)))
    return delta_iia(
        observed,
        control,
        bootstrap=Bootstrap(unit="sentence", replicates=7, seed=0, level=0.95),
    )


def a_dormant() -> DormantPathway:
    return DormantPathway(
        (
            DormantMovement("p0", "source-p0", moved=True),
            DormantMovement("p1", "source-p1", moved=False),
            DormantMovement("p2", "source-p2", moved=False),
            DormantMovement("p3", "source-p3", moved=True),
        ),
        LEFT,
        "absent",
        POLICY,
        "fixture-tokenizer",
    )


def a_control_cell(
    preregistration: Preregistration,
    cell: GridCell,
    *,
    digest: str | None = None,
) -> ControlCell:
    unavailable = BracketingUnavailable(
        "no-span-length-match",
        (2, 2, 2, 2),
        0.0,
        seed=0,
        attempts=2,
        closest=0.25,
    )
    observed = IIAPopulation((("p0", True), ("p1", True), ("p2", False), ("p3", True)))
    control = IIAPopulation((("p0", False), ("p1", True), ("p2", False), ("p3", False)))
    compared = delta_iia(
        observed,
        control,
        bootstrap=Bootstrap(unit="sentence", replicates=7, seed=0, level=0.95),
    )
    return ControlCell(
        cell,
        ScrambledDelta(scramble_types(population(), seed=17), compared),
        unavailable,
        a_dormant(),
        preregistration.digest if digest is None else digest,
    )


def test_control_cells_require_compatible_complete_control_populations() -> None:
    preregistration = a_preregistration()
    cell = preregistration.subgrid.cells[0]
    result = a_control_cell(preregistration, cell)

    assert result.scrambled.headline == 0.5
    assert type(result.bracketing) is BracketingUnavailable
    assert result.dormant.movement_rate == 0.5
    assert result.preregistration == preregistration.digest

    trees = tuple(BinaryBracketing(2, ((0, 2),)) for _ in range(4))
    foreign = delta_iia(
        IIAPopulation((("x0", True), ("x1", True), ("x2", False), ("x3", False))),
        IIAPopulation((("x0", False), ("x1", False), ("x2", False), ("x3", False))),
        bootstrap=Bootstrap(unit="sentence", replicates=7, seed=0, level=0.95),
    )
    incompatible = BracketingDelta(trees, trees, foreign, tolerance=0.0)
    with pytest.raises(ValueError, match="incompatible"):
        ControlCell(
            cell,
            result.scrambled,
            incompatible,
            result.dormant,
            preregistration.digest,
        )
    reversed_dormant = DormantPathway(
        tuple(reversed(result.dormant.movements)),
        LEFT,
        "absent",
        POLICY,
        "fixture-tokenizer",
    )
    reordered = ControlCell(
        cell,
        result.scrambled,
        result.bracketing,
        reversed_dormant,
        preregistration.digest,
    )

    assert reversed_dormant.target_ids == result.dormant.target_ids
    assert reordered.dormant.movements == result.dormant.movements

    wrong_lengths = BracketingUnavailable(
        "no-span-length-match",
        (2, 2, 2, 3),
        0.0,
        seed=0,
        attempts=2,
        closest=0.25,
    )
    with pytest.raises(ValueError, match="incompatible"):
        ControlCell(
            cell,
            result.scrambled,
            wrong_lengths,
            result.dormant,
            preregistration.digest,
        )


def test_control_report_requires_exact_cell_coverage_and_one_digest() -> None:
    preregistration = a_preregistration(two_cells=True)
    first, second = preregistration.subgrid.cells
    first_result = a_control_cell(preregistration, first)
    second_result = a_control_cell(preregistration, second)
    report = ControlReport(preregistration, (second_result, first_result))

    assert report.digest == preregistration.digest
    assert tuple(result.cell for result in report.results) == (first, second)
    with pytest.raises(ValueError, match="missing preregistered"):
        ControlReport(preregistration, (first_result,))
    with pytest.raises(ValueError, match="reported twice"):
        ControlReport(preregistration, (first_result, first_result, second_result))
    with pytest.raises(ValueError, match="another preregistration"):
        ControlReport(
            preregistration,
            (first_result, a_control_cell(preregistration, second, digest="wrong")),
        )


def test_control_values_are_frozen() -> None:
    scramble = scramble_types(population(), seed=1)
    with pytest.raises(AttributeError, match="immutable"):
        scramble.__setattr__("seed", 2)
    with pytest.raises(TypeError):
        cast("dict[str, str]", scramble.manifest)["p0"] = "p1"


def test_public_value_representations_are_informative() -> None:
    tree = BinaryBracketing(3, ((0, 3), (0, 2)))

    assert repr(tree) == "BinaryBracketing(3, ((0, 3), (0, 2)))"
    assert hash(tree) == hash(BinaryBracketing(3, ((0, 2), (0, 3))))


def test_pair_fixture_conforms_to_the_model_ready_protocol() -> None:
    pair = population()[0]

    assert isinstance(pair, ModelReadyCausalPair)
    assert pair.base[LEFT] != pair.variant[LEFT]
    assert pair.base[RIGHT] == pair.variant[RIGHT]

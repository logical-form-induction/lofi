from typing import Final

import torch

from lofi import probe
from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._das import AlignmentObjective
from lofi.stats import Bootstrap, LabelPermutation

LEFT: Final = probe.TypeAt((0, 1))
RIGHT: Final = probe.TypeAt((1, 2))
ROOT_SPLIT: Final = probe.SplitAt((0, 2))
ROOT_RULE: Final = probe.CombinatorAt((0, 2), 1)
POLICY: Final = probe.AlignmentPolicy("first", "exclude", contiguous=True)
PROVENANCE: Final = probe.Provenance("fixture", "control-composition", 0)
ALIGNMENT: Final = probe.TokenizerAlignment("toy", ((0,), (1,)), (), 2)


def a_pair(
    identity: str,
    base_left: str,
    variant_left: str,
    right: str,
    value: int,
) -> probe.ModelCausalPair:
    base: probe.CellAssignment = {
        LEFT: base_left,
        RIGHT: right,
        ROOT_SPLIT: 1,
        ROOT_RULE: "application",
    }
    variant: probe.CellAssignment = dict(base)
    variant[LEFT] = variant_left
    structural = probe.CounterfactualPair(base, variant, LEFT, LEFT.cell, PROVENANCE)
    return probe.ModelCausalPair(
        identity,
        structural,
        ("one", "two"),
        POLICY,
        base_inputs={"input_ids": torch.tensor([[value, value + 1]])},
        variant_inputs={"input_ids": torch.tensor([[value + 10, value + 11]])},
        base_alignment=ALIGNMENT,
        variant_alignment=ALIGNMENT,
    )


def a_preregistration() -> probe.Preregistration:
    site = Site(ComponentId("control-model"), Role.RESIDUAL_STREAM, (0,))
    subgrid = probe.Subgrid((LEFT,), (0,), (0,), {(0, 0): site})
    search = probe.FixedRankSearch(
        rank=1,
        schedule=probe.Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
        null=LabelPermutation(replicates=1, seed=0),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    prices = probe.CostModel(
        (
            probe.UnitPrice(
                probe.ProbeUnit.FIXED_ALIGNMENT_FIT,
                seconds_per_run=1.0,
                seconds_fastest=0.5,
                warmup=1,
                repeats=2,
                device="cpu",
            ),
            probe.UnitPrice(
                probe.ProbeUnit.FIXED_PERMUTATION_REFIT,
                seconds_per_run=1.0,
                seconds_fastest=0.5,
                warmup=1,
                repeats=2,
                device="cpu",
            ),
        )
    )
    return probe.Preregistration(subgrid, search, costs=prices, budget_hours=1.0)


def test_scrambling_matching_deltas_and_dormancy_form_one_complete_report() -> None:
    pairs = (
        a_pair("p0", "N", "NP", "S", 0),
        a_pair("p1", "N", "S", "NP", 2),
        a_pair("p2", "NP", "PP", "N", 4),
        a_pair("p3", "S", "N", "PP", 6),
    )
    scrambled_population = probe.scramble_types(pairs, seed=17)
    reference = (
        probe.BinaryBracketing(2, ((0, 2),)),
        probe.BinaryBracketing(2, ((0, 2),)),
        probe.BinaryBracketing(2, ((0, 2),)),
        probe.BinaryBracketing(2, ((0, 2),)),
    )
    bracketings = probe.matched_binary_bracketings(
        reference, tolerance=1.0, seed=11, attempts=1
    )
    observed = probe.IIAPopulation(
        (("p0", True), ("p1", True), ("p2", False), ("p3", True))
    )
    scrambled = probe.IIAPopulation(
        (("p0", False), ("p1", True), ("p2", False), ("p3", False))
    )
    bracketed = probe.IIAPopulation(
        (("p0", True), ("p1", False), ("p2", False), ("p3", False))
    )
    bootstrap = Bootstrap(unit="sentence", replicates=31, seed=5, level=0.95)
    scrambled_delta = probe.delta_iia(observed, scrambled, bootstrap=bootstrap)
    bracketing_delta = probe.delta_iia(observed, bracketed, bootstrap=bootstrap)
    scrambled_control = probe.ScrambledDelta(scrambled_population, scrambled_delta)
    assert isinstance(bracketings, tuple)
    bracketing_control = probe.BracketingDelta(
        reference, bracketings, bracketing_delta, tolerance=1.0
    )
    dormant = probe.DormantPathway(
        tuple(
            probe.DormantMovement(identity, f"source-{identity}", moved=moved)
            for identity, moved in zip(
                observed.ids, (True, False, False, True), strict=True
            )
        ),
        LEFT,
        "absent",
        POLICY,
        "toy",
    )
    preregistration = a_preregistration()
    result = probe.ControlCell(
        preregistration.subgrid.cells[0],
        scrambled_control,
        bracketing_control,
        dormant,
        preregistration.digest,
    )
    report = probe.ControlReport(preregistration, (result,))

    assert tuple(value.length for value in bracketings) == (2, 2, 2, 2)
    assert set(scrambled_population.manifest) == set(observed.ids)
    assert all(
        target != donor for target, donor in scrambled_population.manifest.items()
    )
    assert report.digest == preregistration.digest
    assert report.results == (result,)
    assert report.results[0].scrambled.headline == 0.5
    assert report.results[0].bracketing is bracketing_control
    assert bracketing_control.headline == 0.5
    assert report.results[0].dormant.movement_rate == 0.5

import itertools
from typing import TYPE_CHECKING, Final, Self, cast, override

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._causal import (
    AlignmentPolicy,
    CounterfactualPair,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._circuits import Unit
from lofi.probe._controls import dormant_pathway
from lofi.probe._das import (
    AlignmentObjective,
    Annealing,
    Assessment,
    BoundlessSearch,
    DistributedAlignment,
    FitStep,
    Fitted,
    FixedRankSearch,
    Schedule,
    SearchStrategy,
)
from lofi.probe._ports import Alignment, Subject
from lofi.probe._provenance import Provenance
from lofi.stats import LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._causal import CellAssignment, HighLevelVariable

PARSER: Final = ComponentId("toy-parser")
HIDDEN: Final = Site(PARSER, Role.CHART_CELL, (0, 2))
SEALED: Final = Site(PARSER, Role.SPLIT_DISTRIBUTION, (0, 2))
CATEGORY: Final = TypeAt((0, 2))
SPLIT: Final = SplitAt((0, 4))
CATEGORIES: Final = ("N", "NP", "S", "PP", "S/NP", "NP/N", "S\\NP", "conj")
BITS: Final = 3
NUISANCE: Final = 5
WIDTH: Final = BITS + NUISANCE
GAIN: Final = 2.0
GENERATED_BY: Final = Provenance("planted", "toy", 0)
POLICY: Final = AlignmentPolicy("first", "exclude", contiguous=True)

# the category at the cell is planted as three signed bits in a hidden basis
CODES: Final = torch.tensor(
    [
        [1.0 if (index >> bit) & 1 else -1.0 for bit in range(BITS)]
        for index in range(len(CATEGORIES))
    ]
)

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    PARSER,
    {
        HIDDEN: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        SEALED: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
    },
)


class _Held:
    __slots__ = ("_axes", "_mask", "_site", "_tensor")

    def __init__(self, site: Site, tensor: Tensor, axes: tuple[str, ...]) -> None:
        self._site = site
        self._tensor = tensor
        self._axes = axes
        self._mask = None

    @property
    def site(self) -> Site:
        return self._site

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return self._axes

    @property
    def mask(self) -> Tensor | None:
        return self._mask


class _Scope:
    def __init__(
        self, model: _Toy, sites: frozenset[Site], patches: Mapping[Site, Patch[_Held]]
    ) -> None:
        self._model = model
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Held] = {}

    @property
    def readings(self) -> Mapping[Site, _Held]:
        return self._readings

    def observe(self, site: Site, tensor: Tensor, axes: tuple[str, ...]) -> None:
        if site in self._sites:
            self._readings[site] = _Held(site, tensor, axes)

    def intervene(self, site: Site, tensor: Tensor, axes: tuple[str, ...]) -> Tensor:
        standing = self._patches.get(site)

        if standing is None:
            return tensor

        return standing.stands_in_for(_Held(site, tensor, axes))

    def __enter__(self) -> Self:
        self._model.scopes.append(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        self._model.scopes.remove(self)

        return None


class _Toy:
    """Plants the category at the cell in a rotated three-bit subspace.

    The site contains a rotated category code plus nuisance features. Connected,
    scrambled, and disconnected readouts expose the intervention laws' three cases.
    """

    def __init__(
        self,
        basis: Tensor,
        *,
        scrambled_at: frozenset[int] = frozenset(),
        connected: bool = True,
    ) -> None:
        self._basis = basis
        self._scrambled_at = scrambled_at
        self._connected = connected
        self.readout = torch.nn.Parameter(CODES.clone())
        self.scopes: list[_Scope] = []
        self.runs: list[tuple[tuple[CounterfactualPair, ...], Side]] = []

    @property
    def policy(self) -> AlignmentPolicy:
        return POLICY

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def decisions(self, variable: HighLevelVariable) -> tuple[str | int, ...]:
        return CATEGORIES if variable == CATEGORY else ()

    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        self.runs.append((tuple(pairs), side))
        decided = [pair.base if side is Side.BASE else pair.variant for pair in pairs]
        codes = torch.stack(
            [CODES[CATEGORIES.index(str(at[CATEGORY]))] for at in decided]
        )
        splits = torch.tensor([float(at[SPLIT]) for at in decided]).unsqueeze(-1)
        nuisance = torch.cat(
            [splits, splits**2 / 4, torch.sin(splits), torch.cos(splits), -splits],
            dim=-1,
        )
        hidden = torch.cat([codes * GAIN, nuisance], dim=-1) @ self._basis

        for scope in self.scopes:
            hidden = scope.intervene(HIDDEN, hidden, ("item", "width"))
            scope.observe(HIDDEN, hidden, ("item", "width"))
            scope.observe(SEALED, hidden, ("item", "width"))

        if not self._connected:
            return codes @ self.readout.T

        bits = (hidden @ self._basis.T)[:, :BITS] / GAIN
        scores = bits @ self.readout.T
        scrambled = torch.tensor(
            [int(at[SPLIT]) in self._scrambled_at for at in decided]
        ).unsqueeze(-1)

        return torch.where(scrambled, scores.roll(1, dims=-1), scores)


class _Spread(_Toy):
    """Plants the category one-hot across the whole width of the site.

    Every coordinate of the hidden basis holds one category, so no subspace of lower
    rank than the width holds the decision, and the readout turns the value back and
    reads it off directly.
    """

    @override
    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        self.runs.append((tuple(pairs), side))
        decided = [pair.base if side is Side.BASE else pair.variant for pair in pairs]
        index = torch.tensor([CATEGORIES.index(str(at[CATEGORY])) for at in decided])
        planted = torch.nn.functional.one_hot(index, WIDTH).to(CODES.dtype) * GAIN
        hidden = planted @ self._basis

        for scope in self.scopes:
            hidden = scope.intervene(HIDDEN, hidden, ("item", "width"))
            scope.observe(HIDDEN, hidden, ("item", "width"))

        return (hidden @ self._basis.T) / GAIN


class _Wide(_Toy):
    @override
    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        scores = super().run(pairs, side)

        return torch.cat([scores, scores], dim=-1)


class _NanScores(_Toy):
    @override
    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        return super().run(pairs, side) * float("nan")


class _NonfiniteGradient(_Toy):
    @override
    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        scores = super().run(pairs, side)
        zero = (scores - scores).abs()

        return scores + torch.sqrt(zero)


class _PublishedAs(_Toy):
    def __init__(self, basis: Tensor, published: SiteMap) -> None:
        super().__init__(basis)
        self._published = published

    @override
    def site_map(self) -> SiteMap:
        return self._published


def a_basis(seed: int = 3) -> Tensor:
    orthogonal, _ = torch.linalg.qr(
        torch.randn(WIDTH, WIDTH, generator=torch.Generator().manual_seed(seed))
    )

    return orthogonal


def a_pair(
    base: str, variant: str, split: int = 2, varied: HighLevelVariable = CATEGORY
) -> CounterfactualPair:
    before: CellAssignment = {CATEGORY: base, SPLIT: split}
    after: CellAssignment = {CATEGORY: variant, SPLIT: split}

    return CounterfactualPair(before, after, varied, (0, 4), GENERATED_BY)


class _ModelPair:
    __slots__ = ("_id", "_pair", "_policy", "_tokenizer")

    def __init__(
        self,
        identity: str,
        pair: CounterfactualPair,
        tokenizer: str = "toy",
        policy: AlignmentPolicy = POLICY,
    ) -> None:
        self._id = identity
        self._pair = pair
        self._tokenizer = tokenizer
        self._policy = policy

    @property
    def id(self) -> str:
        return self._id

    @property
    def tokens(self) -> tuple[str, ...]:
        return ("one", "two", "three", "four")

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
    def span(self) -> tuple[int, int]:
        return self._pair.span

    @property
    def provenance(self) -> Provenance:
        return self._pair.provenance

    @property
    def policy(self) -> AlignmentPolicy:
        return self._policy

    @property
    def tokenizer(self) -> str:
        return self._tokenizer

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        value = 0 if side is Side.BASE else 1
        return {"input_ids": torch.tensor([[value]])}

    def rows(self, side: Side, /) -> tuple[int, ...]:
        return (0,) if side is Side.BASE else (1,)

    def live_rows(self, side: Side, /) -> int:
        del side
        return 4


def a_control_alignment(*, connected: bool) -> DistributedAlignment[_Held, _ModelPair]:
    discrete = torch.arange(WIDTH) < BITS
    fitting = (_ModelPair("fitting", FITTING[0]),)
    fitted = Fitted(
        torch.eye(WIDTH),
        discrete.to(dtype=torch.float32),
        discrete,
        (0.0, float(BITS)),
        BITS,
        (FitStep(0, None, 0.0, 0.0, None),),
        fitting,
    )
    return DistributedAlignment(
        cast("Subject[_Held, _ModelPair]", _Toy(torch.eye(WIDTH), connected=connected)),
        HIDDEN,
        CATEGORY,
        search=a_search(),
        fitted=fitted,
    )


def every_pair(seed: int = 0) -> tuple[CounterfactualPair, ...]:
    pairs = [
        a_pair(base, variant, split)
        for base, variant in itertools.permutations(CATEGORIES, 2)
        for split in (1, 2, 3)
    ]
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed))

    return tuple(pairs[at] for at in order.tolist())


FITTING: Final = every_pair()[:96]
HELD_OUT: Final = every_pair()[96:144]


def a_schedule(epochs: int = 25, seed: int = 0) -> Schedule:
    return Schedule(epochs=epochs, learning_rate=0.05, batch_size=16, seed=seed)


def a_null(replicates: int = 4) -> LabelPermutation[int]:
    return LabelPermutation(replicates=replicates, seed=1)


def test_alignment_objectives_compute_the_declared_loss() -> None:
    scores = torch.tensor([[3.0, 1.0, 2.0], [1.0, 4.0, 2.0]])
    targets = torch.tensor([0, 2])

    assert float(AlignmentObjective.CROSS_ENTROPY.loss(scores, targets)) == (
        pytest.approx(float(torch.nn.functional.cross_entropy(scores, targets)))
    )
    assert float(AlignmentObjective.LOGIT_DIFFERENCE.loss(scores, targets)) == (
        pytest.approx(0.5)
    )


def test_an_alignment_objective_refuses_malformed_decision_rows() -> None:
    scores = torch.ones(2, 2)

    with pytest.raises(TypeError, match="integer decision indices"):
        AlignmentObjective.LOGIT_DIFFERENCE.loss(scores, torch.ones(2))
    with pytest.raises(ValueError, match="outside"):
        AlignmentObjective.LOGIT_DIFFERENCE.loss(scores, torch.tensor([0, 2]))


def a_search(rank: int = BITS, **schedule: int) -> FixedRankSearch:
    return FixedRankSearch(
        rank=rank,
        schedule=a_schedule(**schedule),
        null=a_null(),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )


def a_boundless(**schedule: int) -> BoundlessSearch:
    return BoundlessSearch(
        boundaries=(0.0, float(BITS)),
        margin=1e-6,
        annealing=Annealing(initial=1.0, final=0.05),
        schedule=a_schedule(**schedule),
        null=a_null(),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )


def an_alignment(
    subject: _Toy | None = None, rank: int = BITS, **schedule: int
) -> DistributedAlignment[_Held]:
    return DistributedAlignment(
        _Toy(a_basis()) if subject is None else subject,
        HIDDEN,
        CATEGORY,
        search=a_search(rank, **schedule),
    )


@pytest.fixture(scope="module")
def planted() -> DistributedAlignment[_Held]:
    return an_alignment().fit(FITTING)


@pytest.fixture(scope="module")
def boundless() -> DistributedAlignment[_Held]:
    return DistributedAlignment(
        _Toy(a_basis()), HIDDEN, CATEGORY, search=a_boundless()
    ).fit(FITTING)


@pytest.fixture(scope="module")
def planted_assessment(planted: DistributedAlignment[_Held]) -> Assessment:
    return planted.assess(HELD_OUT)


def test_the_search_conforms_to_the_alignment_protocol_by_shape() -> None:
    alignment = an_alignment()

    assert isinstance(alignment, Alignment)
    assert isinstance(_Toy(a_basis()), Subject)
    assert alignment.policy == POLICY
    assert alignment.rank == BITS
    assert alignment.site == HIDDEN
    assert alignment.variable == CATEGORY
    assert alignment.fitted is None


def test_a_planted_alignment_is_recovered(
    planted: DistributedAlignment[_Held], planted_assessment: Assessment
) -> None:
    assert planted.fitted is not None
    assert planted.fitted.pairs == FITTING
    assert planted_assessment.result.accuracy > 0.95
    assert planted_assessment.result.variable == CATEGORY
    assert planted_assessment.result.rank == BITS


def test_the_fitted_rotation_is_orthogonal(
    planted: DistributedAlignment[_Held],
) -> None:
    assert planted.fitted is not None
    rotation = planted.fitted.rotation

    assert not rotation.requires_grad
    assert torch.allclose(rotation.T @ rotation, torch.eye(WIDTH), atol=1e-4)
    assert torch.equal(
        planted.fitted.mask,
        (torch.arange(WIDTH) < BITS).to(torch.float32),
    )
    assert torch.equal(
        planted.fitted.discrete_mask,
        torch.arange(WIDTH) < BITS,
    )
    assert planted.fitted.boundaries == (0.0, float(BITS))
    assert planted.fitted.discrete_width == BITS
    assert all(step.temperature is None for step in planted.fitted.trace)
    assert all(step.boundary_gradient_norm is None for step in planted.fitted.trace)


def test_the_fitted_subspace_is_the_planted_one(
    planted: DistributedAlignment[_Held],
) -> None:
    assert planted.fitted is not None
    # the first three rotated coordinates span the three planted bit directions
    found = planted.fitted.rotation[:, :BITS]
    true = a_basis()[:BITS, :].T
    projected = true.T @ found @ found.T @ true

    assert torch.allclose(projected, torch.eye(BITS), atol=0.05)


def test_boundless_search_retains_its_ordered_boundaries_masks_and_every_step(
    boundless: DistributedAlignment[_Held],
) -> None:
    assert boundless.fitted is not None
    fitted = boundless.fitted
    lower, upper = fitted.boundaries
    expected_steps = a_schedule().epochs * (len(FITTING) // a_schedule().batch_size)
    coordinates = torch.arange(WIDTH, dtype=fitted.mask.dtype)
    expected_mask = torch.sigmoid(
        (coordinates - lower) / a_boundless().annealing.final
    ) * torch.sigmoid((upper - coordinates) / a_boundless().annealing.final)

    assert 0.0 <= lower < upper <= WIDTH
    assert torch.equal(fitted.mask, expected_mask)
    assert fitted.discrete_width == int(fitted.discrete_mask.sum())
    assert fitted.aligned is (fitted.discrete_width > 0)
    assert len(fitted.trace) == expected_steps
    assert tuple(step.step for step in fitted.trace) == tuple(range(expected_steps))
    assert fitted.trace[0].temperature == a_boundless().annealing.initial
    assert fitted.trace[-1].temperature == pytest.approx(a_boundless().annealing.final)
    assert all(torch.isfinite(torch.tensor(step.loss)) for step in fitted.trace)
    assert any(step.rotation_gradient_norm > 0.0 for step in fitted.trace)
    assert any(
        step.boundary_gradient_norm is not None and step.boundary_gradient_norm > 0.0
        for step in fitted.trace
    )


def test_boundless_and_fixed_search_agree_after_the_interval_is_discretized(
    boundless: DistributedAlignment[_Held],
) -> None:
    assert boundless.fitted is not None
    fixed = an_alignment(rank=boundless.fitted.discrete_width).fit(FITTING)

    assert abs(boundless.iia(HELD_OUT).accuracy - fixed.iia(HELD_OUT).accuracy) <= 0.02


def test_no_refit_on_exchanged_decisions_does_as_well(
    planted_assessment: Assessment,
) -> None:
    replicates = a_null().replicates

    assert replicates is not None
    assert planted_assessment.p_value.value == pytest.approx(1 / (replicates + 1))
    assert planted_assessment.result.null == planted_assessment.p_value.provenance
    assert planted_assessment.result.null.null == "label permutation"
    assert planted_assessment.result.null.replicates == replicates


def test_iia_is_the_assessed_result(
    planted: DistributedAlignment[_Held], planted_assessment: Assessment
) -> None:
    assert planted.iia(HELD_OUT) == planted_assessment.result


def test_output_movement_uses_one_fitted_patch_route_against_the_target_base() -> None:
    targets = (
        _ModelPair("target-0", a_pair("N", "NP")),
        _ModelPair("target-1", a_pair("S", "NP")),
    )
    sources = (
        _ModelPair("source-0", a_pair("S", "PP")),
        _ModelPair("source-1", a_pair("N", "PP")),
    )

    assert a_control_alignment(connected=True).moves_output(targets, sources) == (
        True,
        True,
    )
    assert a_control_alignment(connected=False).moves_output(targets, sources) == (
        False,
        False,
    )


def test_output_movement_requires_equal_nonempty_populations_for_one_variable() -> None:
    alignment = a_control_alignment(connected=True)
    pair = _ModelPair("pair", a_pair("N", "NP"))
    split_pair = _ModelPair(
        "split-pair",
        CounterfactualPair(
            {CATEGORY: "N", SPLIT: 1},
            {CATEGORY: "N", SPLIT: 2},
            SPLIT,
            (0, 4),
            GENERATED_BY,
        ),
    )

    with pytest.raises(ValueError, match="at least one pair"):
        alignment.moves_output((), ())
    with pytest.raises(ValueError, match="1 targets with 2 sources"):
        alignment.moves_output((pair,), (pair, pair))
    with pytest.raises(ValueError, match="varying"):
        alignment.moves_output((pair,), (split_pair,))


def test_dormant_pathway_reports_planted_and_inert_movement_by_stable_id() -> None:
    targets = (
        _ModelPair("target-0", a_pair("N", "NP")),
        _ModelPair("target-1", a_pair("S", "NP")),
    )
    sources = (
        _ModelPair("source-0", a_pair("S", "PP")),
        _ModelPair("source-1", a_pair("N", "PP")),
    )
    planted = dormant_pathway(
        a_control_alignment(connected=True), targets, sources, absent="PP"
    )
    inert = dormant_pathway(
        a_control_alignment(connected=False), targets, sources, absent="PP"
    )

    assert planted.target_ids == ("target-0", "target-1")
    assert planted.source_ids == ("source-0", "source-1")
    assert tuple(value.moved for value in planted.movements) == (True, True)
    assert planted.movement_rate == 1.0
    assert inert.movement_rate == 0.0
    assert planted.variable == CATEGORY
    assert planted.absent == "PP"
    assert planted.policy == POLICY
    assert planted.tokenizer == "toy"


def test_dormant_pathway_refuses_identity_policy_tokenizer_and_absence_defects() -> (
    None
):
    alignment = a_control_alignment(connected=True)
    target = _ModelPair("target", a_pair("N", "NP"))
    source = _ModelPair("source", a_pair("S", "PP"))

    with pytest.raises(ValueError, match="disjoint"):
        dormant_pathway(
            alignment,
            (target,),
            (_ModelPair("target", a_pair("S", "PP")),),
            absent="PP",
        )
    with pytest.raises(ValueError, match="one tokenizer"):
        dormant_pathway(
            alignment,
            (target,),
            (_ModelPair("source", a_pair("S", "PP"), "other"),),
            absent="PP",
        )
    with pytest.raises(ValueError, match="policy"):
        dormant_pathway(
            alignment,
            (target,),
            (
                _ModelPair(
                    "source",
                    a_pair("S", "PP"),
                    policy=AlignmentPolicy("last", "exclude", contiguous=True),
                ),
            ),
            absent="PP",
        )
    with pytest.raises(ValueError, match="absent decision"):
        dormant_pathway(alignment, (target,), (source,), absent="NP")


def test_a_category_spread_over_the_whole_site_is_recovered_only_at_full_rank() -> None:
    low = an_alignment(_Spread(a_basis())).fit(FITTING).iia(HELD_OUT)
    full = an_alignment(_Spread(a_basis()), rank=WIDTH).fit(FITTING).iia(HELD_OUT)

    assert low.rank == BITS
    assert low.accuracy < 0.95
    assert full.rank == WIDTH
    assert full.accuracy > 0.95


def test_accuracy_and_the_null_are_measured_on_the_held_out_pairs() -> None:
    subject = _Toy(a_basis(), scrambled_at=frozenset({3}))
    fitting = tuple(pair for pair in FITTING if pair.variant[SPLIT] != 3)
    held = tuple(pair for pair in HELD_OUT if pair.variant[SPLIT] == 3)
    fitted = an_alignment(subject).fit(fitting)
    assessment = fitted.assess(held)
    matches = fitted.matches(held)

    assert fitted.iia(fitting).accuracy == 1.0
    assert matches == (False,) * len(held)
    assert assessment.result.accuracy == sum(matches) / len(matches)
    assert assessment.result.accuracy == 0.0
    assert assessment.p_value.value == 1.0


def test_the_null_replicates_are_measured_on_the_held_out_pairs() -> None:
    subject = _Toy(a_basis())
    held = HELD_OUT[:16]
    fitted = an_alignment(subject, epochs=1).fit(FITTING[:32])
    subject.runs.clear()
    fitted.assess(held)
    measured = [
        pairs for pairs, side in subject.runs if side is Side.BASE and pairs == held
    ]
    replicates = a_null().replicates

    assert replicates is not None
    assert len(measured) == 1 + replicates


def test_fitting_leaves_the_subject_without_gradients_and_unchanged() -> None:
    subject = _Toy(a_basis())
    an_alignment(subject, epochs=2).fit(FITTING[:32])

    assert subject.readout.grad is None
    assert torch.equal(subject.readout.detach(), CODES)


def test_boundless_fitting_leaves_the_subject_without_gradients_and_unchanged() -> None:
    subject = _Toy(a_basis())
    DistributedAlignment(subject, HIDDEN, CATEGORY, search=a_boundless(epochs=2)).fit(
        FITTING[:32]
    )

    assert subject.readout.grad is None
    assert torch.equal(subject.readout.detach(), CODES)
    assert subject.scopes == []


def test_accuracy_increases_with_rank_across_the_sweep(
    planted_assessment: Assessment,
) -> None:
    accuracies = [
        an_alignment(rank=rank).fit(FITTING).iia(HELD_OUT).accuracy
        for rank in range(1, BITS)
    ] + [planted_assessment.result.accuracy]

    assert all(lower < higher for lower, higher in itertools.pairwise(accuracies))
    assert accuracies[-1] > 0.95


def test_the_reported_rank_is_the_rank_searched_at() -> None:
    result = an_alignment(rank=2, epochs=1).fit(FITTING[:16]).iia(HELD_OUT[:16])

    assert result.rank == 2


def test_fitting_leaves_the_unfitted_alignment_as_it_was() -> None:
    alignment = an_alignment(epochs=1)
    fitted = alignment.fit(FITTING[:16])

    assert alignment.fitted is None
    assert fitted.fitted is not None
    assert fitted.search is alignment.search
    assert fitted.site == alignment.site


def test_fitting_is_deterministic_under_a_seed() -> None:
    first = an_alignment(epochs=2).fit(FITTING[:32])
    second = an_alignment(epochs=2).fit(FITTING[:32])
    third = an_alignment(epochs=2, seed=5).fit(FITTING[:32])

    assert first.fitted is not None
    assert second.fitted is not None
    assert third.fitted is not None
    assert torch.equal(first.fitted.rotation, second.fitted.rotation)
    assert not torch.allclose(first.fitted.rotation, third.fitted.rotation)


def test_the_units_are_the_site_for_the_fitted_variable_only(
    planted: DistributedAlignment[_Held],
) -> None:
    assert planted.apply(CATEGORY) == frozenset({Unit(HIDDEN)})
    assert planted.apply(SPLIT) == frozenset()
    assert an_alignment().apply(CATEGORY) == frozenset()


def test_an_unfitted_alignment_reports_no_accuracy() -> None:
    with pytest.raises(ValueError, match="not been fitted"):
        an_alignment().iia(HELD_OUT)

    with pytest.raises(ValueError, match="not been fitted"):
        an_alignment().matches(HELD_OUT)


def test_a_pair_varying_another_variable_is_refused() -> None:
    other = CounterfactualPair(
        {CATEGORY: "N", SPLIT: 1},
        {CATEGORY: "N", SPLIT: 2},
        SPLIT,
        (0, 4),
        GENERATED_BY,
    )

    with pytest.raises(ValueError, match="varying SplitAt"):
        an_alignment().fit([other])


def test_a_decision_the_subject_cannot_make_is_refused() -> None:
    with pytest.raises(ValueError, match="does not score"):
        an_alignment().fit([a_pair("N", "VP")])


def test_fitting_on_no_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        an_alignment().fit([])


def test_a_subject_scoring_in_another_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="one row per pair"):
        an_alignment(_Wide(a_basis()), epochs=1).fit(FITTING[:16])


def test_a_site_the_scores_do_not_depend_on_cannot_be_fitted() -> None:
    with pytest.raises(ValueError, match="do not depend"):
        an_alignment(_Toy(a_basis(), connected=False), epochs=1).fit(FITTING[:16])


def test_a_nonfinite_loss_and_nonfinite_gradient_are_refused_by_name() -> None:
    with pytest.raises(ValueError, match="loss is not a finite scalar"):
        an_alignment(_NanScores(a_basis()), epochs=1).fit(FITTING[:16])

    with pytest.raises(ValueError, match="gradient is not finite"):
        an_alignment(_NonfiniteGradient(a_basis()), epochs=1).fit(FITTING[:16])


def test_a_site_that_permits_no_patch_is_refused_at_region_entry() -> None:
    sealed = DistributedAlignment(_Toy(a_basis()), SEALED, CATEGORY, search=a_search())

    with pytest.raises(ValueError, match="does not permit patching"):
        sealed.fit(FITTING[:16])


def test_a_site_declared_non_differentiable_is_refused_before_a_model_run() -> None:
    published = SiteMap.custom(
        PARSER,
        {
            HIDDEN: SiteSpec(
                ("item", "width"),
                DTypeClass.REAL,
                differentiable=False,
                access=Access.OBSERVE | Access.PATCH,
            )
        },
    )
    subject = _PublishedAs(a_basis(), published)

    with pytest.raises(ValueError, match="non-differentiable"):
        an_alignment(subject, epochs=1).fit(FITTING[:16])

    assert subject.runs == []


def test_a_site_without_observation_access_is_refused_before_a_model_run() -> None:
    published = SiteMap.custom(
        PARSER,
        {
            HIDDEN: SiteSpec(
                ("item", "width"),
                DTypeClass.REAL,
                differentiable=True,
                access=Access.PATCH,
            )
        },
    )
    subject = _PublishedAs(a_basis(), published)

    with pytest.raises(ValueError, match="does not permit observation"):
        an_alignment(subject, epochs=1).fit(FITTING[:16])

    assert subject.runs == []


def test_every_step_reads_the_variants_and_then_patches_the_bases() -> None:
    subject = _Toy(a_basis())
    an_alignment(subject, epochs=1).fit(FITTING[:32])

    assert [(len(pairs), side) for pairs, side in subject.runs] == [
        (16, Side.VARIANT),
        (16, Side.BASE),
        (16, Side.VARIANT),
        (16, Side.BASE),
    ]


def test_each_epoch_draws_the_pairs_in_a_fresh_seeded_order() -> None:
    subject = _Toy(a_basis())
    an_alignment(subject, epochs=2).fit(FITTING[:32])
    variants = [pairs for pairs, side in subject.runs if side is Side.VARIANT]
    first = variants[0] + variants[1]
    second = variants[2] + variants[3]

    assert set(first) == set(FITTING[:32])
    assert set(second) == set(FITTING[:32])
    assert first != FITTING[:32]
    assert first != second


def test_a_fixed_search_of_rank_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="rank zero"):
        a_search(rank=0)


def test_a_fixed_search_requires_an_explicit_objective() -> None:
    with pytest.raises(TypeError, match="objective"):
        FixedRankSearch(
            rank=1,
            schedule=a_schedule(),
            null=a_null(),
            objective=cast("AlignmentObjective", None),
        )


def test_logit_difference_drives_the_distributed_fit() -> None:
    search = FixedRankSearch(
        rank=BITS,
        schedule=a_schedule(epochs=1),
        null=a_null(replicates=1),
        objective=AlignmentObjective.LOGIT_DIFFERENCE,
    )
    fitted = DistributedAlignment(_Toy(a_basis()), HIDDEN, CATEGORY, search=search).fit(
        FITTING[:16]
    )

    assert search.objective is AlignmentObjective.LOGIT_DIFFERENCE
    assert fitted.fitted is not None
    assert len(fitted.fitted.trace) == 1


def test_a_fixed_search_requires_a_schedule_and_a_label_permutation() -> None:
    with pytest.raises(TypeError, match="Schedule"):
        FixedRankSearch(
            rank=1,
            schedule=cast("Schedule", None),
            null=a_null(),
            objective=AlignmentObjective.CROSS_ENTROPY,
        )

    with pytest.raises(TypeError, match="LabelPermutation"):
        FixedRankSearch(
            rank=1,
            schedule=a_schedule(),
            null=cast("LabelPermutation[int]", None),
            objective=AlignmentObjective.CROSS_ENTROPY,
        )


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ((-0.1, 2.0), "outside"),
        ((2.0, 2.0), "not ordered"),
        ((3.0, 2.0), "not ordered"),
        ((0.0, float("inf")), "not finite"),
    ],
)
def test_a_boundless_search_refuses_malformed_boundaries(
    boundaries: tuple[float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BoundlessSearch(
            boundaries=boundaries,
            margin=1e-6,
            annealing=Annealing(initial=1.0, final=0.1),
            schedule=a_schedule(),
            null=a_null(),
            objective=AlignmentObjective.CROSS_ENTROPY,
        )


def test_a_boundless_search_refuses_boundaries_outside_the_observed_width() -> None:
    search = BoundlessSearch(
        boundaries=(0.0, WIDTH + 1.0),
        margin=1e-6,
        annealing=Annealing(initial=1.0, final=0.1),
        schedule=a_schedule(epochs=1),
        null=a_null(),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    alignment = DistributedAlignment(_Toy(a_basis()), HIDDEN, CATEGORY, search=search)

    with pytest.raises(ValueError, match="outside a site of width"):
        alignment.fit(FITTING[:16])


@pytest.mark.parametrize(
    ("initial", "final", "message"),
    [
        (0.0, 0.1, "initial temperature is not positive"),
        (1.0, 0.0, "final temperature is not positive"),
        (1.0, 1.0, "ends below"),
        (0.5, 1.0, "ends below"),
    ],
)
def test_annealing_requires_a_decreasing_positive_schedule(
    initial: float, final: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Annealing(initial=initial, final=final)


def test_a_boundless_search_requires_annealing() -> None:
    with pytest.raises(TypeError, match="Annealing"):
        BoundlessSearch(
            boundaries=(0.0, 2.0),
            margin=1e-6,
            annealing=cast("Annealing", None),
            schedule=a_schedule(),
            null=a_null(),
            objective=AlignmentObjective.CROSS_ENTROPY,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "zero epochs"),
        ("learning_rate", 0.0, "not positive"),
        ("learning_rate", -0.1, "not positive"),
        ("batch_size", 0, "zero pairs"),
        ("seed", -1, "negative"),
    ],
)
def test_a_schedule_refuses_a_degenerate_field(
    field: str, value: float, message: str
) -> None:
    fields: dict[str, float] = {
        "epochs": 3,
        "learning_rate": 0.1,
        "batch_size": 4,
        "seed": 0,
    }
    fields[field] = value

    with pytest.raises(ValueError, match=message):
        Schedule(
            epochs=int(fields["epochs"]),
            learning_rate=fields["learning_rate"],
            batch_size=int(fields["batch_size"]),
            seed=int(fields["seed"]),
        )


def test_a_schedule_is_equal_on_its_four_fields() -> None:
    assert a_schedule() == a_schedule()
    assert hash(a_schedule()) == hash(a_schedule())
    assert a_schedule() != a_schedule(seed=1)
    assert a_schedule() != a_schedule(epochs=2)
    assert repr(a_schedule()) == (
        "Schedule(epochs=25, learning_rate=0.05, batch_size=16, seed=0)"
    )


def test_a_fitted_rotation_is_a_square_floating_matrix_over_some_pairs() -> None:
    trace = (FitStep(0, None, 1.0, 1.0, None),)
    mask = torch.ones(2)
    discrete = torch.ones(2, dtype=torch.bool)

    with pytest.raises(TypeError, match="floating point"):
        Fitted(
            torch.ones(2, 2, dtype=torch.int64),
            mask,
            discrete,
            (0.0, 2.0),
            2,
            trace,
            FITTING[:1],
        )

    with pytest.raises(ValueError, match="square"):
        Fitted(
            torch.ones(2, 3),
            mask,
            discrete,
            (0.0, 2.0),
            2,
            trace,
            FITTING[:1],
        )

    with pytest.raises(ValueError, match="no pair"):
        Fitted(torch.eye(2), mask, discrete, (0.0, 2.0), 2, trace, [])

    held = Fitted(
        torch.eye(2).requires_grad_(),
        mask,
        discrete,
        (0.0, 2.0),
        2,
        trace,
        FITTING[:2],
    )

    assert not held.rotation.requires_grad
    assert held.pairs == FITTING[:2]
    assert repr(held) == "Fitted(<2x2>, width=2, pairs=2, steps=1)"

    returned_rotation = held.rotation
    returned_mask = held.mask
    returned_discrete = held.discrete_mask
    returned_rotation.zero_()
    returned_mask.zero_()
    returned_discrete.zero_()

    assert torch.equal(held.rotation, torch.eye(2))
    assert torch.equal(held.mask, torch.ones(2))
    assert torch.equal(held.discrete_mask, torch.ones(2, dtype=torch.bool))

    with pytest.raises(AttributeError):
        held.__setattr__("_discrete_width", 0)

    with pytest.raises(AttributeError):
        trace[0].__setattr__("_loss", 0.0)


def test_a_zero_width_fit_is_assessed_as_an_unaligned_rank_zero_result() -> None:
    trace = (FitStep(0, 1.0, 1.0, 1.0, 1.0),)
    lower, upper = (0.1, 0.2)
    temperature = 0.1
    coordinates = torch.arange(WIDTH, dtype=torch.float32)
    continuous = torch.sigmoid((coordinates - lower) / temperature) * torch.sigmoid(
        (upper - coordinates) / temperature
    )
    fitted = Fitted(
        torch.eye(WIDTH),
        continuous,
        torch.zeros(WIDTH, dtype=torch.bool),
        (lower, upper),
        0,
        trace,
        FITTING[:16],
    )
    search = BoundlessSearch(
        boundaries=(0.1, 0.2),
        margin=1e-6,
        annealing=Annealing(initial=1.0, final=0.1),
        schedule=a_schedule(epochs=1),
        null=a_null(replicates=1),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    alignment = DistributedAlignment(
        _Toy(a_basis()), HIDDEN, CATEGORY, search=search, fitted=fitted
    )
    assessment = alignment.assess(HELD_OUT[:16])
    result = alignment.iia(HELD_OUT[:16])

    assert assessment.result.rank == 0
    assert result.rank == 0
    assert alignment.rank == 0
    assert alignment.apply(CATEGORY) == frozenset()
    assert alignment.matches(HELD_OUT[:16]) == (False,) * 16


def test_an_alignment_checks_its_site_variable_search_and_fit() -> None:
    subject = _Toy(a_basis())

    with pytest.raises(TypeError, match="Site"):
        DistributedAlignment(
            subject, cast("Site", "hidden"), CATEGORY, search=a_search()
        )

    with pytest.raises(TypeError, match="high-level variable"):
        DistributedAlignment(
            subject, HIDDEN, cast("HighLevelVariable", "category"), search=a_search()
        )

    with pytest.raises(TypeError, match="search strategy"):
        DistributedAlignment(
            subject, HIDDEN, CATEGORY, search=cast("SearchStrategy", a_schedule())
        )

    with pytest.raises(TypeError, match="Fitted"):
        DistributedAlignment(
            subject,
            HIDDEN,
            CATEGORY,
            search=a_search(),
            fitted=cast("Fitted", torch.eye(WIDTH)),
        )


def test_an_alignment_built_from_a_fitted_rotation_is_assessed_without_refitting(
    planted: DistributedAlignment[_Held], planted_assessment: Assessment
) -> None:
    assert planted.fitted is not None
    rebuilt = DistributedAlignment(
        _Toy(a_basis()), HIDDEN, CATEGORY, search=a_search(), fitted=planted.fitted
    )

    assert rebuilt.iia(HELD_OUT) == planted_assessment.result


def test_the_repr_names_the_site_the_variable_the_rank_and_the_fit(
    planted: DistributedAlignment[_Held], planted_assessment: Assessment
) -> None:
    assert repr(an_alignment()).endswith("width=3, fitted=False)")
    assert repr(planted).endswith("width=3, fitted=True)")
    assert repr(planted_assessment).startswith("Assessment(accuracy=1.0")


@pytest.mark.parametrize("margin", [0.0, 0.5, float("nan")])
def test_a_boundless_search_refuses_a_share_margin_outside_the_open_interval(
    margin: float,
) -> None:
    with pytest.raises(ValueError, match=r"share margin|not finite"):
        BoundlessSearch(
            boundaries=(0.0, 2.0),
            margin=margin,
            annealing=Annealing(initial=1.0, final=0.1),
            schedule=Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
            null=LabelPermutation(replicates=1, seed=0),
            objective=AlignmentObjective.LOGIT_DIFFERENCE,
        )

from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._causal import (
    AlignmentPolicy,
    IIAResult,
    JointCounterfactualPair,
    NamedVariable,
    Side,
    TypeAt,
)
from lofi.probe._das import (
    AlignmentObjective,
    Annealing,
    Assessment,
    FixedRankSearch,
    Schedule,
)
from lofi.probe._joint import (
    JointDistributedAlignment,
    JointFitStep,
    JointFitted,
    JointLayerSelection,
    JointReport,
    dominance_pairs,
    select_joint_layer,
)
from lofi.probe._ports import JointAlignment, JointSubject
from lofi.probe._pricing import CostModel, ProbeUnit, UnitPrice
from lofi.probe._provenance import Provenance
from lofi.probe._sweep import (
    CellResult,
    Preregistration,
    Subgrid,
    SweepReport,
)
from lofi.stats import Bootstrap, LabelPermutation, p_value_against_replicates

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._causal import CellAssignment, HighLevelVariable
    from lofi.stats import BootstrapResult

MODEL: Final = ComponentId("joint-toy")
SITE: Final = Site(MODEL, Role.RESIDUAL_STREAM, (0,))
OUTER: Final = TypeAt((0, 4))
INNER: Final = TypeAt((1, 3))
POLICY: Final = AlignmentPolicy("all", "exclude", contiguous=True)
PROVENANCE: Final = Provenance("joint-fixture", "synthetic", 3)
PUBLISHED: Final = SiteMap.custom(
    MODEL,
    {
        SITE: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        )
    },
)


class _Pair:
    __slots__ = ("_id", "_pair")

    def __init__(self, identity: str) -> None:
        self._id = identity
        self._pair = JointCounterfactualPair(
            {OUTER: "A", INNER: "A"},
            {OUTER: "B", INNER: "B"},
            (OUTER, INNER),
            OUTER.cell,
            PROVENANCE,
        )

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
    def variables(self) -> tuple[HighLevelVariable, HighLevelVariable]:
        return self._pair.variables

    @property
    def span(self) -> tuple[int, int]:
        return self._pair.span

    @property
    def provenance(self) -> Provenance:
        return self._pair.provenance

    @property
    def policy(self) -> AlignmentPolicy:
        return POLICY

    @property
    def tokenizer(self) -> str:
        return "toy-tokenizer"

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        fill = 0 if side is Side.BASE else 1
        return {"tokens": torch.full((1, 4), fill)}

    def rows(self, side: Side, variable: HighLevelVariable, /) -> tuple[int, ...]:
        del side
        if isinstance(variable, NamedVariable):
            msg = "a joint fixture does not admit NamedVariable"
            raise TypeError(msg)
        return tuple(range(*variable.cell))

    def selected_rows(self, side: Side, /) -> tuple[int, ...]:
        del side
        return (0, 1, 2, 3)

    def live_rows(self, side: Side, /) -> int:
        del side
        return 4


class _Reading:
    __slots__ = ("_tensor",)

    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    @property
    def site(self) -> Site:
        return SITE

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return ("item", "width")

    @property
    def mask(self) -> None:
        return None


class _Scope:
    def __init__(
        self,
        subject: _Toy,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[_Reading]],
    ) -> None:
        self._subject = subject
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Reading] = {}

    @property
    def readings(self) -> Mapping[Site, _Reading]:
        return self._readings

    def apply(self, hidden: Tensor) -> Tensor:
        reading = _Reading(hidden)
        patch = self._patches.get(SITE)
        replaced = hidden if patch is None else patch.stands_in_for(reading)
        if SITE in self._sites:
            self._readings[SITE] = _Reading(replaced)
        return replaced

    def __enter__(self) -> Self:
        self._subject.scopes.append(self)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        del kind, error, traceback
        self._subject.scopes.remove(self)


class _Toy:
    def __init__(self, *, noncompositional: bool = False) -> None:
        self.noncompositional = noncompositional
        self.scopes: list[_Scope] = []

    @property
    def policy(self) -> AlignmentPolicy:
        return POLICY

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Reading]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def decisions(self, variable: HighLevelVariable, /) -> tuple[str | int, ...]:
        if variable not in {OUTER, INNER}:
            return ()
        return ("A", "B")

    def run(self, pairs: Sequence[_Pair], side: Side, /) -> tuple[Tensor, Tensor]:
        assignments = [
            pair.base if side is Side.BASE else pair.variant for pair in pairs
        ]
        hidden = torch.tensor(
            [
                [
                    1.0 if assignment[OUTER] == "B" else -1.0,
                    1.0 if assignment[INNER] == "B" else -1.0,
                    0.0,
                    0.0,
                ]
                for assignment in assignments
            ],
            requires_grad=True,
        )
        for scope in self.scopes:
            hidden = scope.apply(hidden)
        outer_scores = torch.stack((-hidden[:, 0], hidden[:, 0]), dim=-1)
        inner_scores = torch.stack((-hidden[:, 1], hidden[:, 1]), dim=-1)
        if self.noncompositional and side is Side.BASE:
            jointly_changed = (hidden[:, 0] > 0.0) & (hidden[:, 1] > 0.0)
            even = torch.tensor(
                [int(pair.id.rsplit("-", 1)[-1]) % 2 == 0 for pair in pairs]
            )
            outer_scores = torch.where(
                (jointly_changed & even).unsqueeze(-1),
                outer_scores.roll(1, dims=-1),
                outer_scores,
            )
            inner_scores = torch.where(
                (jointly_changed & ~even).unsqueeze(-1),
                inner_scores.roll(1, dims=-1),
                inner_scores,
            )
        return outer_scores, inner_scores


def a_selection() -> JointLayerSelection:
    report = a_sweep_report(layers=(0,))
    selected = select_joint_layer(report, ("dev-0", "dev-1"))
    return JointLayerSelection(report, selected.development_ids)


def a_fitted() -> JointFitted[_Pair]:
    return JointFitted(
        torch.eye(4),
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False]),
        (JointFitStep(0, 0.0, 0.0, 0.0),),
        (_Pair("fit-0"), _Pair("fit-1")),
    )


def an_alignment(subject: _Toy) -> JointDistributedAlignment[_Reading, _Pair]:
    return JointDistributedAlignment(
        subject,
        a_selection(),
        (OUTER, INNER),
        schedule=Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
        annealing=Annealing(initial=1.0, final=0.1),
        bootstrap=Bootstrap(unit="pair", replicates=99, seed=4, level=0.95),
        objective=AlignmentObjective.CROSS_ENTROPY,
        fitted=a_fitted(),
    )


def test_joint_pairs_are_structurally_distinct_and_dominance_ordered() -> None:
    pair = _Pair("pair-0")

    assert pair.variables == (OUTER, INNER)
    assert dominance_pairs((INNER, TypeAt((5, 6)), OUTER)) == ((OUTER, INNER),)

    with pytest.raises(ValueError, match="exactly its two"):
        JointCounterfactualPair(
            {OUTER: "A", INNER: "A"},
            {OUTER: "B", INNER: "A"},
            (OUTER, INNER),
            OUTER.cell,
            PROVENANCE,
        )

    with pytest.raises(TypeError, match="NamedVariable"):
        dominance_pairs((OUTER, NamedVariable("external", "category")))


def test_the_joint_protocols_are_structural() -> None:
    subject = _Toy()
    alignment = an_alignment(subject)

    assert isinstance(subject, JointSubject)
    assert isinstance(alignment, JointAlignment)
    assert JointSubject not in type(subject).__mro__
    assert JointAlignment not in type(alignment).__mro__


def test_a_compositional_model_matches_both_marginals_and_their_product() -> None:
    report = an_alignment(_Toy()).report(tuple(_Pair(f"test-{i}") for i in range(4)))

    assert report.joint_accuracy == 1.0
    assert report.dominant_accuracy == 1.0
    assert report.dominated_accuracy == 1.0
    assert report.marginal_product == 1.0
    assert report.joint_minus_product.observed == 0.0


def test_a_noncompositional_model_collapses_only_under_the_joint_path() -> None:
    alignment = an_alignment(_Toy(noncompositional=True))
    pairs = tuple(_Pair(f"test-{i}") for i in range(4))
    report = alignment.report(pairs)

    assert report.joint_accuracy == 0.0
    assert alignment.matches(pairs) == ((True, True),) * 4
    assert report.dominant_accuracy == 1.0
    assert report.dominated_accuracy == 1.0
    assert report.marginal_product == 1.0
    assert report.joint_minus_product.observed == -1.0


def test_a_joint_report_refuses_a_forged_fifth_quantity() -> None:
    interval = (
        an_alignment(_Toy())
        .report(tuple(_Pair(f"test-{i}") for i in range(4)))
        .joint_minus_product
    )

    with pytest.raises(TypeError, match="BootstrapResult"):
        JointReport(1.0, 1.0, 1.0, 1.0, cast("BootstrapResult", "forged"))

    with pytest.raises(ValueError, match="computed from joint accuracy"):
        JointReport(0.5, 0.5, 0.5, 0.25, interval)


def test_a_joint_fit_step_refuses_coercible_and_nonfinite_scalars() -> None:
    with pytest.raises(TypeError, match="not a number"):
        JointFitStep(0, cast("float", "0.0"), 0.0, 0.0)
    with pytest.raises(TypeError, match="not a number"):
        JointFitStep(0, 0.0, cast("float", "0.0"), 0.0)
    with pytest.raises(TypeError, match="not a number"):
        JointFitStep(0, 0.0, 0.0, cast("float", "0.0"))
    with pytest.raises(ValueError, match="not finite"):
        JointFitStep(0, 0.0, float("nan"), 0.0)


def test_joint_fitting_learns_two_disjoint_masks_under_one_rotation() -> None:
    unfitted = JointDistributedAlignment(
        _Toy(),
        a_selection(),
        (OUTER, INNER),
        schedule=Schedule(epochs=1, learning_rate=0.05, batch_size=2, seed=3),
        annealing=Annealing(initial=1.0, final=0.1),
        bootstrap=Bootstrap(unit="pair", replicates=19, seed=4, level=0.95),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )

    fitted = unfitted.fit((_Pair("fit-0"), _Pair("fit-1")))
    assert fitted.fitted is not None
    first, second = fitted.fitted.masks

    assert fitted.widths[0] > 0
    assert fitted.widths[1] > 0
    assert not bool((first & second).any())
    assert torch.allclose(
        fitted.fitted.rotation.T @ fitted.fitted.rotation,
        torch.eye(4),
        atol=1e-6,
        rtol=0.0,
    )


def test_logit_difference_drives_joint_fitting() -> None:
    alignment = JointDistributedAlignment(
        _Toy(),
        a_selection(),
        (OUTER, INNER),
        schedule=Schedule(epochs=1, learning_rate=0.05, batch_size=2, seed=3),
        annealing=Annealing(initial=1.0, final=0.1),
        bootstrap=Bootstrap(unit="pair", replicates=19, seed=4, level=0.95),
        objective=AlignmentObjective.LOGIT_DIFFERENCE,
    ).fit((_Pair("fit-0"), _Pair("fit-1")))

    assert alignment.objective is AlignmentObjective.LOGIT_DIFFERENCE
    assert alignment.fitted is not None
    assert len(alignment.fitted.trace) == 1


def test_a_public_joint_fit_refuses_nonfinite_or_nonorthogonal_rotations() -> None:
    first = torch.tensor([True, False, False, False])
    second = torch.tensor([False, True, False, False])
    trace = (JointFitStep(0, 0.0, 0.0, 0.0),)
    pairs = (_Pair("fit-0"),)

    nonfinite = torch.eye(4)
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        JointFitted(nonfinite, first, second, trace, pairs)

    with pytest.raises(ValueError, match="not orthogonal"):
        JointFitted(torch.ones(4, 4), first, second, trace, pairs)


def test_development_fitting_and_evaluation_identities_are_disjoint() -> None:
    alignment = an_alignment(_Toy())

    with pytest.raises(ValueError, match="overlap fitting"):
        alignment.matches((_Pair("fit-0"), _Pair("test-1")))

    with pytest.raises(ValueError, match="overlap development"):
        alignment.matches((_Pair("dev-0"), _Pair("test-1")))


def _price(unit: ProbeUnit) -> UnitPrice:
    return UnitPrice(
        unit,
        seconds_per_run=1.0,
        seconds_fastest=0.5,
        warmup=1,
        repeats=2,
        device="cpu",
    )


def _assessment(accuracy: float, variable: HighLevelVariable) -> Assessment:
    tested = p_value_against_replicates(
        accuracy,
        (0.0,),
        tail="greater",
        null="cell-permutation",
        family="permutation",
        exact=False,
        seed=0,
    )
    return Assessment(IIAResult(accuracy, variable, 1, tested.provenance), tested)


def a_sweep_report(
    *,
    complete: bool = True,
    layers: tuple[int, ...] = (2, 1),
    variables: tuple[HighLevelVariable, ...] = (OUTER, INNER),
) -> SweepReport:
    sites = {
        (layer, 0): Site(MODEL, Role.RESIDUAL_STREAM, (layer,)) for layer in layers
    }
    subgrid = Subgrid(variables, layers, (0,), sites)
    search = FixedRankSearch(
        rank=1,
        schedule=Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
        null=LabelPermutation(replicates=1, seed=0),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    costs = CostModel(
        (
            _price(ProbeUnit.FIXED_ALIGNMENT_FIT),
            _price(ProbeUnit.FIXED_PERMUTATION_REFIT),
        )
    )
    preregistration = Preregistration(subgrid, search, costs=costs, budget_hours=1.0)
    by_layer = {0: (1.0, 1.0), 2: (0.7, 0.9), 1: (0.8, 0.8)}
    accuracies: dict[tuple[HighLevelVariable, int], float] = {
        (variable, layer): by_layer[layer][position]
        for position, variable in enumerate(variables)
        for layer in layers
    }
    results = tuple(
        CellResult(
            cell,
            subgrid.site_of(cell),
            _assessment(accuracies[cell.variable, cell.layer], cell.variable),
            preregistration.digest,
        )
        for cell in subgrid.cells
    )
    return SweepReport(preregistration, results if complete else results[:-1])


def test_layer_selection_requires_the_complete_grid_and_breaks_ties_by_order() -> None:
    selected = select_joint_layer(a_sweep_report(), ("dev-0", "dev-1"))

    assert selected.layer == 2
    assert selected.means == ((2, 0.8), (1, 0.8))
    assert selected.variables == (OUTER, INNER)

    with pytest.raises(ValueError, match="every preregistered"):
        select_joint_layer(a_sweep_report(complete=False), ("dev-0", "dev-1"))


def test_layer_selection_and_alignment_are_bound_to_one_dominance_pair() -> None:
    unrelated = TypeAt((5, 6))
    with pytest.raises(ValueError, match="exactly two preregistered variables"):
        select_joint_layer(
            a_sweep_report(variables=(OUTER, unrelated)),
            ("dev-0", "dev-1"),
        )

    other_outer = TypeAt((0, 5))
    other_inner = TypeAt((1, 4))
    with pytest.raises(ValueError, match="development sweep selected"):
        JointDistributedAlignment(
            _Toy(),
            a_selection(),
            (other_outer, other_inner),
            schedule=Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
            annealing=Annealing(initial=1.0, final=0.1),
            bootstrap=Bootstrap(unit="pair", replicates=19, seed=4, level=0.95),
            objective=AlignmentObjective.CROSS_ENTROPY,
        )

from typing import TYPE_CHECKING, Self

import torch

from lofi.nn.ports import Observable
from lofi.nn.sites import ComponentId, Role, Site, SiteMap
from lofi.probe._causal import (
    AlignmentPolicy,
    CounterfactualPair,
    HighLevelVariable,
    IIAResult,
    JointCounterfactualPair,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._circuits import Unit
from lofi.probe._ports import (
    Alignment,
    ModelReadyCausalPair,
    ModelReadyJointCausalPair,
    Subject,
    TermEncoder,
    TermVectors,
)
from lofi.probe._provenance import Provenance
from lofi.stats import p_value_against_replicates
from lofi.terms import BatchEncoding

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch, Reading
ENCODER = ComponentId("term-encoder")
GENERATED_BY = Provenance("grammar-counterfactuals", "ccgbank-dev", 7)

CATEGORY = TypeAt((0, 2))
SPLIT = SplitAt((0, 4))


def a_pair(category: str) -> CounterfactualPair:
    return CounterfactualPair(
        {CATEGORY: "NP", SPLIT: 2},
        {CATEGORY: category, SPLIT: 2},
        CATEGORY,
        (0, 4),
        GENERATED_BY,
    )


class _Rotation:
    def __init__(self, policy: AlignmentPolicy, rank: int, seen: int = 0) -> None:
        self._policy = policy
        self._rank = rank
        self._seen = seen

    @property
    def policy(self) -> AlignmentPolicy:
        return self._policy

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def seen(self) -> int:
        return self._seen

    def fit(self, pairs: Sequence[CounterfactualPair]) -> Self:
        return type(self)(self._policy, self._rank, len(pairs))

    def apply(self, variable: HighLevelVariable) -> frozenset[Unit]:
        if isinstance(variable, NamedVariable):
            return frozenset()
        start, end = variable.cell

        return frozenset(
            Unit(Site(ENCODER, Role.CHART_CELL, (start, end)), {"width": width})
            for width in range(self._rank)
        )

    def iia(self, pairs: Sequence[CounterfactualPair]) -> IIAResult:
        matched = sum(1 for pair in pairs if pair.variant[pair.varied] == "S")
        tested = p_value_against_replicates(
            float(matched),
            [0.0, 1.0],
            tail="greater",
            null="cell-permutation",
            family="permutation",
            exact=False,
            seed=3,
        )

        return IIAResult(
            matched / len(pairs), pairs[0].varied, self._rank, tested.provenance
        )


def a_rotation(rank: int = 4) -> _Rotation:
    return _Rotation(AlignmentPolicy("first", "exclude", contiguous=True), rank)


def test_an_implementation_conforms_by_shape_and_not_by_inheritance() -> None:
    rotation = a_rotation()

    assert isinstance(rotation, Alignment)
    assert Alignment not in type(rotation).__mro__


def test_something_missing_a_method_does_not_conform() -> None:
    class Partial:
        policy = AlignmentPolicy("first", "exclude", contiguous=True)
        rank = 4

    assert not isinstance(Partial(), Alignment)


def test_fitting_returns_a_fitted_alignment_and_leaves_the_original_alone() -> None:
    unfitted = a_rotation()
    fitted = unfitted.fit([a_pair("S"), a_pair("PP")])

    assert fitted.seen == 2
    assert unfitted.seen == 0
    assert fitted is not unfitted
    assert fitted.policy == unfitted.policy


def test_applying_an_alignment_names_the_units_a_variable_lives_at() -> None:
    units = a_rotation(3).apply(CATEGORY)

    assert len(units) == 3
    assert all(unit.site.key == (0, 2) for unit in units)
    assert {unit.index["width"] for unit in units} == {0, 1, 2}


def test_the_accuracy_comes_back_with_its_variable_its_rank_and_its_null() -> None:
    result = a_rotation(8).iia([a_pair("S"), a_pair("PP")])

    assert result.accuracy == 0.5
    assert result.variable == CATEGORY
    assert result.rank == 8
    assert result.null.null == "cell-permutation"


def test_the_policy_fitting_ran_under_is_the_policy_evaluation_runs_under() -> None:
    rotation = a_rotation()
    fitted = rotation.fit([a_pair("S")])

    assert fitted.policy is rotation.policy
    assert fitted.policy.pieces == "first"


def test_an_alignment_reaches_a_model_only_through_the_units_it_names() -> None:
    for unit in a_rotation().apply(SPLIT):
        assert isinstance(unit.site, Site)
        assert unit.site.component == ENCODER
        assert unit.site.role is Role.CHART_CELL


class _Scope:
    def __init__(
        self, sites: frozenset[Site], patches: Mapping[Site, Patch[Reading]]
    ) -> None:
        self._sites = sites
        self._patches = patches

    @property
    def readings(self) -> Mapping[Site, Reading]:
        return {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        return None


class _Decider:
    @property
    def policy(self) -> AlignmentPolicy:
        return AlignmentPolicy("first", "exclude", contiguous=True)

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("decider"), {})

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Reading]]
    ) -> _Scope:
        return _Scope(sites, patches)

    def decisions(self, variable: HighLevelVariable, /) -> tuple[str | int, ...]:
        return ("NP", "S") if variable == CATEGORY else (1, 2, 3)

    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        chosen = [pair.base if side is Side.BASE else pair.variant for pair in pairs]
        decisions = self.decisions(CATEGORY)

        return torch.tensor(
            [
                [
                    1.0 if decisions[column] == at[CATEGORY] else 0.0
                    for column in range(2)
                ]
                for at in chosen
            ]
        )


def test_a_subject_is_a_probed_component_that_also_decides() -> None:
    subject = _Decider()

    assert isinstance(subject, Subject)
    assert isinstance(subject, Observable)
    assert Subject not in type(subject).__mro__


def test_a_probed_component_that_does_not_decide_is_not_a_subject() -> None:
    class Silent:
        def site_map(self) -> SiteMap:
            return SiteMap.custom(ComponentId("silent"), {})

        def instrument(
            self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Reading]]
        ) -> _Scope:
            return _Scope(sites, patches)

    assert isinstance(Silent(), Observable)
    assert not isinstance(Silent(), Subject)


def test_a_subject_scores_each_side_of_a_pair_over_its_decisions() -> None:
    subject = _Decider()
    pairs = [a_pair("S"), a_pair("S")]

    assert subject.run(pairs, Side.BASE).tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert subject.run(pairs, Side.VARIANT).tolist() == [[0.0, 1.0], [0.0, 1.0]]


class _ReadyPair:
    def __init__(self) -> None:
        self._pair = a_pair("S")

    @property
    def id(self) -> str:
        return "ready-pair"

    @property
    def tokens(self) -> tuple[str, ...]:
        return ("a", "sentence")

    @property
    def base(self) -> Mapping[HighLevelVariable, str | int]:
        return self._pair.base

    @property
    def variant(self) -> Mapping[HighLevelVariable, str | int]:
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
        return AlignmentPolicy("first", "exclude", contiguous=True)

    @property
    def tokenizer(self) -> str:
        return "tokenizer"

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        fill = 0.0 if side is Side.BASE else 1.0
        return {"vectors": torch.full((1, 2, 2), fill)}

    def rows(self, side: Side, /) -> tuple[int, ...]:
        del side

        return (0, 1)

    def live_rows(self, side: Side, /) -> int:
        del side

        return 2


def test_a_model_ready_pair_conforms_by_shape_and_not_by_inheritance() -> None:
    pair = _ReadyPair()

    assert isinstance(pair, ModelReadyCausalPair)
    assert ModelReadyCausalPair not in type(pair).__mro__


def test_a_structural_pair_without_model_inputs_is_not_model_ready() -> None:
    assert not isinstance(a_pair("S"), ModelReadyCausalPair)


class _ReadyJointPair:
    def __init__(self) -> None:
        outer = TypeAt((0, 4))
        inner = TypeAt((1, 3))
        self._pair = JointCounterfactualPair(
            {outer: "NP", inner: "N"},
            {outer: "S", inner: "NP"},
            (outer, inner),
            outer.cell,
            GENERATED_BY,
        )

    @property
    def id(self) -> str:
        return "joint-ready-pair"

    @property
    def tokens(self) -> tuple[str, ...]:
        return ("a", "joint", "test", "case")

    @property
    def base(self) -> Mapping[HighLevelVariable, str | int]:
        return self._pair.base

    @property
    def variant(self) -> Mapping[HighLevelVariable, str | int]:
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
        return AlignmentPolicy("all", "exclude", contiguous=True)

    @property
    def tokenizer(self) -> str:
        return "tokenizer"

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        fill = 0.0 if side is Side.BASE else 1.0
        return {"vectors": torch.full((1, 4, 2), fill)}

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


def test_a_model_ready_joint_pair_is_not_a_widened_single_pair() -> None:
    pair = _ReadyJointPair()

    assert isinstance(pair, ModelReadyJointCausalPair)
    assert not isinstance(pair, ModelReadyCausalPair)
    assert ModelReadyJointCausalPair not in type(pair).__mro__


class _Vectors:
    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return ("term", "width")

    @property
    def mask(self) -> Tensor | None:
        return None


class _Bagging:
    def encode(self, terms: BatchEncoding, /) -> _Vectors:
        return _Vectors(torch.zeros(len(terms.rows), 2))


def test_a_term_encoder_conforms_by_shape_and_not_by_inheritance() -> None:
    encoder = _Bagging()

    assert isinstance(encoder, TermEncoder)
    assert TermEncoder not in type(encoder).__mro__
    assert isinstance(encoder.encode(BatchEncoding(())), TermVectors)


def test_something_that_does_not_encode_terms_is_not_a_term_encoder() -> None:
    assert not isinstance(_Vectors(torch.zeros(1, 2)), TermEncoder)
    assert not isinstance(_Bagging(), TermVectors)

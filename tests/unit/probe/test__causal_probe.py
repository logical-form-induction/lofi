from typing import TYPE_CHECKING, Final, Self, cast, overload

import didactic.api as dx
import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._boundary import (
    BoundlessSearchDecl,
    FixedRankSearchDecl,
    NodeVocabularyArmDecl,
)
from lofi.probe._bucketing import AbstractionFeatures
from lofi.probe._causal import (
    AlignmentPolicy,
    CounterfactualPair,
    JointCounterfactualPair,
    Side,
    TypeAt,
)
from lofi.probe._causal_probe import PLUGIN, CausalProbe, CausalProbeSpec
from lofi.probe._controls import BinaryBracketing
from lofi.probe._counterfactuals import AlignmentManifest, CounterfactualGeneration
from lofi.probe._das import BoundlessSearch, FixedRankSearch
from lofi.probe._node_vocabulary import NodeVocabularyArm, NodeVocabularyRow
from lofi.probe._pricing import CostModel, ProbeUnit, UnitPrice
from lofi.probe._probing import (
    GradientMode,
    PassLayout,
    ProbeBuildContext,
    RecordingProbe,
    UnpublishedSiteError,
)
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._causal import CellAssignment, HighLevelVariable
    from lofi.probe._ports import JointSubject, Subject

MODEL: Final = ComponentId("causal-model")
SITE: Final = Site(MODEL, Role.RESIDUAL_STREAM, (0,))
OUTER: Final = TypeAt((0, 4))
INNER: Final = TypeAt((1, 3))
POLICY: Final = AlignmentPolicy("all", "exclude", contiguous=True)
PROVENANCE: Final = Provenance("causal-fixture", "synthetic", 3)
DECISIONS: Final = ("A", "B", "C", "D")
CODES: Final = torch.tensor([-1.5, -0.5, 0.5, 1.5])
PUBLISHED: Final[SiteMap] = SiteMap.custom(
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

    def __init__(
        self,
        identity: str,
        variable: TypeAt,
        base: str,
        variant: str,
        other: str,
    ) -> None:
        self._id = identity
        before: CellAssignment
        after: CellAssignment
        if variable == OUTER:
            before = {OUTER: base, INNER: other}
            after = {OUTER: variant, INNER: other}
        else:
            before = {OUTER: other, INNER: base}
            after = {OUTER: other, INNER: variant}
        self._pair = CounterfactualPair(
            before, after, variable, variable.cell, PROVENANCE
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
        return POLICY

    @property
    def tokenizer(self) -> str:
        return "causal-tokenizer"

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        fill = 0 if side is Side.BASE else 1
        return {"input_ids": torch.full((1, 4), fill)}

    def rows(self, side: Side, /) -> tuple[int, ...]:
        del side
        start, end = self.span
        return tuple(range(start, end))

    def live_rows(self, side: Side, /) -> int:
        del side
        return 4


class _JointPair:
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
        return "causal-tokenizer"

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        fill = 0 if side is Side.BASE else 1
        return {"input_ids": torch.full((1, 4), fill)}

    def rows(self, side: Side, variable: HighLevelVariable, /) -> tuple[int, ...]:
        del side
        if variable == OUTER:
            start, end = OUTER.cell
        elif variable == INNER:
            start, end = INNER.cell
        else:
            msg = f"a joint fixture does not declare {variable!r}"
            raise ValueError(msg)
        return tuple(range(start, end))

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
    __slots__ = ("_patches", "_readings", "_sites", "_subject")

    def __init__(
        self,
        subject: _Subject,
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
        standing = self._patches.get(SITE)
        changed = hidden if standing is None else standing.stands_in_for(reading)
        if SITE in self._sites:
            self._readings[SITE] = _Reading(changed)
        return changed

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


def _code(value: str | int) -> float:
    return float(CODES[DECISIONS.index(str(value))])


class _Subject:
    __slots__ = ("scopes",)

    def __init__(self) -> None:
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
        return DECISIONS if variable in {OUTER, INNER} else ()

    def _hidden(
        self,
        pairs: Sequence[_Pair] | Sequence[_JointPair],
        side: Side,
    ) -> Tensor:
        assignments = [
            pair.base if side is Side.BASE else pair.variant for pair in pairs
        ]
        hidden = torch.tensor(
            [
                [
                    _code(assignment[OUTER]),
                    _code(assignment[INNER]),
                    _code(assignment[OUTER]) * _code(assignment[INNER]),
                    1.0,
                ]
                for assignment in assignments
            ],
            requires_grad=True,
        )
        for scope in self.scopes:
            hidden = scope.apply(hidden)
        return hidden

    @staticmethod
    def _scores(signal: Tensor) -> Tensor:
        return -(signal.unsqueeze(1) - CODES.to(signal.device).unsqueeze(0)).square()

    @overload
    def run(self, pairs: Sequence[_Pair], side: Side, /) -> Tensor: ...

    @overload
    def run(
        self, pairs: Sequence[_JointPair], side: Side, /
    ) -> tuple[Tensor, Tensor]: ...

    def run(
        self,
        pairs: Sequence[_Pair] | Sequence[_JointPair],
        side: Side,
        /,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not pairs:
            msg = "a causal fixture scores at least one pair"
            raise ValueError(msg)
        hidden = self._hidden(pairs, side)
        if isinstance(pairs[0], _JointPair):
            return self._scores(hidden[:, 0]), self._scores(hidden[:, 1])
        variable = pairs[0].varied
        coordinate = 0 if variable == OUTER else 1
        return self._scores(hidden[:, coordinate])


_PROFILES: Final = (
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "D", "A"),
    ("C", "A", "D"),
)


def _population(prefix: str) -> tuple[_Pair, ...]:
    return tuple(
        _Pair(
            f"{prefix}-{variable.cell[0]}-{variable.cell[1]}-{index}",
            variable,
            base,
            variant,
            other,
        )
        for variable in (OUTER, INNER)
        for index, (base, variant, other) in enumerate(_PROFILES)
    )


def _costs() -> CostModel:
    return CostModel(
        tuple(
            UnitPrice(
                unit,
                seconds_per_run=0.001,
                seconds_fastest=0.001,
                warmup=0,
                repeats=1,
                device="cpu",
            )
            for unit in (
                ProbeUnit.FIXED_ALIGNMENT_FIT,
                ProbeUnit.FIXED_PERMUTATION_REFIT,
                ProbeUnit.BOUNDLESS_ALIGNMENT_FIT,
                ProbeUnit.BOUNDLESS_PERMUTATION_REFIT,
            )
        )
    )


class _CausalPass:
    __slots__ = ("_subject",)

    def __init__(self) -> None:
        self._subject = _Subject()

    @property
    def subject(self) -> _Subject:
        return self._subject

    def __call__(self, gradients: GradientMode, /) -> PassLayout:
        del gradients
        return PassLayout((), None, ())

    def causal_population(self) -> CounterfactualGeneration:
        return CounterfactualGeneration((), {}, (), AlignmentManifest(POLICY, (), 0, 0))

    def causal_subject(self) -> Subject[_Reading, _Pair]:
        return cast("Subject[_Reading, _Pair]", self._subject)

    def causal_fitting_pairs(self) -> tuple[_Pair, ...]:
        return _population("fit")

    def causal_held_out_pairs(self) -> tuple[_Pair, ...]:
        return _population("dev")

    def causal_costs(self) -> CostModel:
        return _costs()

    def causal_joint_variables(
        self,
    ) -> tuple[HighLevelVariable, HighLevelVariable]:
        return OUTER, INNER

    def causal_joint_subject(self) -> JointSubject[_Reading, _JointPair]:
        return cast("JointSubject[_Reading, _JointPair]", self._subject)

    def causal_joint_fitting_pairs(self) -> tuple[_JointPair, ...]:
        return (_JointPair("joint-fit-0"), _JointPair("joint-fit-1"))

    def causal_joint_held_out_pairs(self) -> tuple[_JointPair, ...]:
        return (_JointPair("joint-test-0"), _JointPair("joint-test-1"))

    def causal_features(self, pair: _Pair, /) -> AbstractionFeatures:
        return AbstractionFeatures(
            sentence_length=4,
            derivation_depth=1,
            ambiguity=int(pair.id.rsplit("-", 1)[-1]),
            span_length=pair.span[1] - pair.span[0],
        )

    def causal_bracketings(
        self, pairs: Sequence[_Pair], /
    ) -> tuple[BinaryBracketing, ...]:
        return tuple(BinaryBracketing(4, ((0, 4), (0, 3), (0, 2))) for _pair in pairs)

    def causal_bracketing_pairs(
        self,
        pairs: Sequence[_Pair],
        bracketings: Sequence[BinaryBracketing],
        /,
    ) -> tuple[_Pair, ...]:
        if len(pairs) != len(bracketings):
            msg = "a bracketing control preserves the causal population"
            raise ValueError(msg)
        return tuple(pairs)

    def causal_dormant_sources(
        self,
        variable: HighLevelVariable,
        targets: Sequence[_Pair],
        /,
    ) -> tuple[_Pair, ...]:
        if not isinstance(variable, TypeAt):
            msg = f"a causal fixture has no dormant source for {variable!r}"
            raise TypeError(msg)
        other_variable = INNER if variable == OUTER else OUTER
        return tuple(
            _Pair(
                f"dormant-{target.id}",
                variable,
                "B",
                "A",
                str(target.base[other_variable]),
            )
            for target in targets
        )

    def causal_absent_decision(self, variable: HighLevelVariable, /) -> str | int:
        del variable
        return "A"


class _Point:
    __slots__ = ("_pass",)

    def __init__(self) -> None:
        self._pass = _CausalPass()

    @property
    def components(self) -> Mapping[str, _Subject]:
        return {str(MODEL): self._pass.subject}

    @property
    def passes(self) -> tuple[_CausalPass, ...]:
        return (self._pass,)


def _declaration() -> CausalProbeSpec:
    return CausalProbeSpec.model_validate(
        {
            "kind": "causal",
            "sites": [
                {
                    "layer": 0,
                    "position": 0,
                    "site": {
                        "component": str(MODEL),
                        "role": "RESIDUAL_STREAM",
                        "key": [0],
                    },
                }
            ],
            "searches": [
                {"kind": "fixed-rank", "rank": 1},
                {"kind": "boundless", "boundary_lower": 0.0, "boundary_upper": 1.0},
            ],
            "joint_search": "fixed-rank",
            "objective": "logit-difference",
            "epochs": 1,
            "learning_rate": 0.05,
            "batch_size": 4,
            "null_replicates": 1,
            "budget_hours": 1.0,
            "bootstrap_replicates": 3,
            "bracketing_tolerance": 1.0,
            "bracketing_attempts": 1,
            "minimum_bucket_size": 1,
            "minimum_accuracy_gap": 0.0,
            "seed": 3,
            "fdr_alpha": 0.1,
            "dependence": "prds",
        }
    )


def test_a_causal_declaration_requires_sites_and_an_explicit_budget() -> None:
    with pytest.raises(dx.ValidationError, match="sites"):
        CausalProbeSpec.model_validate({"kind": "causal", "budget_hours": 1.0})
    with pytest.raises(dx.ValidationError, match="budget_hours"):
        CausalProbeSpec.model_validate({"kind": "causal", "sites": []})
    with pytest.raises(dx.ValidationError, match="joint_search"):
        _declaration().with_(searches=(BoundlessSearchDecl(),))
    with pytest.raises(dx.ValidationError, match="distinct"):
        _declaration().with_(searches=(FixedRankSearchDecl(), FixedRankSearchDecl()))


def test_each_declared_search_arm_lowers_to_its_strategy_with_its_parameters() -> None:
    settings = (
        _declaration()
        .with_(
            searches=(
                FixedRankSearchDecl(rank=2),
                BoundlessSearchDecl(boundary_lower=0.5, boundary_upper=2.0),
            )
        )
        .settings()
    )
    arms = dict(settings.searches)

    assert tuple(arms) == ("fixed-rank", "boundless")
    assert isinstance(arms["fixed-rank"], FixedRankSearch)
    assert arms["fixed-rank"].rank == 2
    assert isinstance(arms["boundless"], BoundlessSearch)
    assert arms["boundless"].boundaries == (0.5, 2.0)
    assert settings.joint_search == "fixed-rank"


def test_the_joint_condition_may_use_the_boundless_arm() -> None:
    settings = _declaration().with_(joint_search="boundless").settings()

    assert settings.joint_search == "boundless"


def test_a_search_arm_declaration_refuses_invalid_parameters() -> None:
    with pytest.raises(dx.ValidationError, match="rank"):
        FixedRankSearchDecl(rank=0)
    with pytest.raises(dx.ValidationError, match="ordered"):
        BoundlessSearchDecl(boundary_lower=1.0, boundary_upper=0.5)


def test_the_causal_plugin_validates_its_complete_pass_capability_at_the_boundary() -> (
    None
):
    class _PopulationOnly:
        def __call__(self, gradients: GradientMode, /) -> PassLayout:
            del gradients
            return PassLayout((), None, ())

        def causal_population(self) -> CounterfactualGeneration:
            return CounterfactualGeneration(
                (), {}, (), AlignmentManifest(POLICY, (), 0, 0)
            )

    class _PopulationPoint:
        @property
        def components(self) -> Mapping[str, _Subject]:
            return {}

        @property
        def passes(self) -> tuple[_PopulationOnly, ...]:
            return (_PopulationOnly(),)

    with pytest.raises(TypeError, match="model-ready sweep"):
        PLUGIN.construct(
            _declaration(), ProbeBuildContext("causal", _PopulationPoint())
        )


def test_the_causal_probe_executes_both_searches_joint_controls_and_bucketing() -> None:
    point = _Point()
    built = PLUGIN.construct(_declaration(), ProbeBuildContext("causal", point))
    built.admit(PUBLISHED)

    rows = built.run(point)
    measures = {row.measure for row in rows}
    inferential = tuple(row for row in rows if row.raw_p is not None)

    assert isinstance(built, CausalProbe)
    assert {
        "pass-0/joint/iia",
        "pass-0/joint/minus-marginal-product",
    } <= measures
    assert (
        sum(
            measure.startswith("pass-0/fixed-rank/logit-difference/")
            and measure.endswith("/iia")
            for measure in measures
        )
        == 2
    )
    assert (
        sum(
            measure.startswith("pass-0/boundless/logit-difference/")
            and measure.endswith("/iia")
            for measure in measures
        )
        == 2
    )
    for suffix in (
        "type-scramble-delta-iia",
        "bracketing-delta-iia",
        "dormant-movement",
    ):
        assert sum(measure.endswith(suffix) for measure in measures) == 2
    assert any(measure.startswith("pass-0/bucket/") for measure in measures)
    assert len(inferential) == 4
    assert all(row.null is not None for row in inferential)
    assert all(row.adjusted_p is not None for row in inferential)
    assert all(row.rejected is not None for row in inferential)
    assert all(row.family is not None for row in inferential)
    assert all(row.dependence is not None for row in inferential)


def test_a_declared_arm_records_its_named_search_sweep_for_every_pass() -> None:
    point = _Point()
    declaration = _declaration().with_(
        arm=NodeVocabularyArmDecl(
            representation="dense-linear-probe", search="boundless"
        )
    )
    built = PLUGIN.construct(declaration, ProbeBuildContext("causal", point))
    built.admit(PUBLISHED)

    assert isinstance(built, RecordingProbe)
    measured = built.recorded(point)
    row = NodeVocabularyRow.from_record(measured.records["pass-0"])
    boundless = sorted(
        value.value
        for value in measured.values
        if value.measure.startswith("pass-0/boundless/") and value.raw_p is not None
    )

    assert tuple(measured.records) == ("pass-0",)
    assert row.arm is NodeVocabularyArm.DENSE_LINEAR_PROBE
    assert sorted(cell.accuracy for cell in row.cells) == pytest.approx(boundless)
    assert len(row.pair_ids) > 0
    assert [value.measure for value in built.run(point)] == [
        value.measure for value in measured.values
    ]


def test_an_undeclared_arm_records_nothing() -> None:
    point = _Point()
    built = PLUGIN.construct(_declaration(), ProbeBuildContext("causal", point))
    built.admit(PUBLISHED)

    assert isinstance(built, RecordingProbe)
    assert dict(built.recorded(point).records) == {}


def test_an_arm_names_a_declared_search() -> None:
    with pytest.raises(dx.ValidationError, match="arm records search"):
        _declaration().with_(
            arm=NodeVocabularyArmDecl(
                representation="dense-linear-probe", search="missing"
            )
        )
    with pytest.raises(dx.ValidationError, match="representation"):
        NodeVocabularyArmDecl.model_validate(
            {"representation": "dense", "search": "fixed-rank"}
        )


def test_a_causal_probe_admits_each_declared_site_under_the_required_access() -> None:
    probe = CausalProbe("causal", _declaration().settings())
    inaccessible = SiteMap.custom(
        MODEL,
        {
            SITE: SiteSpec(
                ("item", "width"),
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
        },
    )

    with pytest.raises(UnpublishedSiteError, match="patchable"):
        probe.admit(inaccessible)

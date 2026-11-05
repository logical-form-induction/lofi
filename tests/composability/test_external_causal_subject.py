"""External model inputs compose with row-local causal intervention search."""

from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from torch import nn

from lofi import probe
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._das import AlignmentObjective
from lofi.stats import LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

TYPE_VARIABLE: Final = probe.TypeAt((1, 2))
SPLIT_VARIABLE: Final = probe.SplitAt((0, 3))
COMBINATOR_VARIABLE: Final = probe.CombinatorAt((0, 3), 1)
NAMED_VARIABLE: Final = probe.NamedVariable("external", "syntactic-category")
POLICY: Final = probe.AlignmentPolicy("all", "exclude", contiguous=True)
TOKENIZER: Final = "tiny-tokenizer"
MODEL: Final = ComponentId("tiny-causal-model")
SITE: Final = Site(MODEL, Role.RESIDUAL_STREAM, (0,))
AXES: Final = ("batch", "position", "feature")


class TinyModel(nn.Module):
    """Read two decision scores from the sum of three hidden token rows."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(2))

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        pooled = self.hidden(vectors).sum(dim=1)

        return torch.stack((-pooled[:, 0], pooled[:, 0]), dim=-1)


class Truncate(nn.Module):
    """Return fewer tokenizer rows than the input declares live."""

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return vectors[:, :2]


class ShortModel(nn.Module):
    """Expose a row extent inconsistent with retained tokenizer evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = Truncate()

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        pooled = self.hidden(vectors).sum(dim=1)

        return torch.stack((-pooled[:, 0], pooled[:, 0]), dim=-1)


def score(
    output: probe.Output,
    variable: probe.HighLevelVariable,
    decisions: tuple[str | int, ...],
) -> torch.Tensor:
    """Return the tiny model's already ordered decision scores."""
    assert variable in {
        TYPE_VARIABLE,
        SPLIT_VARIABLE,
        COMBINATOR_VARIABLE,
        NAMED_VARIABLE,
    }
    assert len(decisions) == 2
    assert isinstance(output, torch.Tensor)

    return output


def model_pair(
    token: str,
    shift: float,
    variable: probe.HighLevelVariable,
    base_decision: str | int,
    variant_decision: str | int,
    *,
    source_span: tuple[int, int] | None = None,
) -> probe.ModelCausalPair:
    """Return one confirmed pair with exact source and model coordinates."""
    tokens = (token, "runs", "today")
    if source_span is None:
        assert not isinstance(variable, probe.NamedVariable)
        source_span = variable.cell
    pair = probe.CounterfactualPair(
        {variable: base_decision},
        {variable: variant_decision},
        variable,
        source_span,
        probe.Provenance("structural-counterfactual", "unit", 4),
    )
    position = source_span[0]
    alignment = probe.TokenizerAlignment(TOKENIZER, ((0,), (1,), (2,)), (), 3)
    siblings = torch.tensor([[[0.2 + shift, 0.3], [0.0, 0.0], [0.4, -0.1]]])
    base = siblings.clone()
    variant = siblings.clone()
    base[:, position] = torch.tensor([-2.0, 0.5])
    variant[:, position] = torch.tensor([2.0, -0.5])

    return probe.ModelCausalPair(
        f"{variable.kind}:{token}",
        pair,
        tokens,
        POLICY,
        base_inputs={"vectors": base},
        variant_inputs={"vectors": variant},
        base_alignment=alignment,
        variant_alignment=alignment,
    )


def external_model(module: nn.Module) -> probe.ExternalModel:
    """Declare the one public causal site of a tiny module."""
    declared = probe.ExternalSiteMap(
        MODEL,
        (
            probe.ExternalSite(
                "hidden",
                Role.RESIDUAL_STREAM,
                0,
                SiteSpec(
                    AXES,
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE | Access.PATCH,
                ),
            ),
        ),
    )

    return probe.ExternalModel(module, declared)


def causal_subject(
    external: probe.ExternalModel,
    decisions: dict[probe.HighLevelVariable, tuple[str | int, ...]],
    scorer: probe.DecisionScorer,
) -> probe.ExternalSubject:
    """Return a row-local subject over one declared external model."""
    return probe.ExternalSubject(
        external,
        POLICY,
        TOKENIZER,
        decisions,
        scorer,
        batch_axis="batch",
        row_axis="position",
    )


def subject() -> tuple[probe.ExternalModel, probe.ExternalSubject]:
    """Return the public external adapter and its causal subject."""
    external = external_model(TinyModel())

    return (
        external,
        causal_subject(
            external,
            {
                TYPE_VARIABLE: ("A", "B"),
                SPLIT_VARIABLE: (1, 2),
                COMBINATOR_VARIABLE: ("forward", "backward"),
                NAMED_VARIABLE: ("A", "B"),
            },
            score,
        ),
    )


class WrappedPair:
    """A controlled wrapper implementing the public model-ready pair boundary."""

    def __init__(self, pair: probe.ModelCausalPair) -> None:
        self._pair = pair

    @property
    def id(self) -> str:
        return self._pair.id

    @property
    def tokens(self) -> tuple[str, ...]:
        return self._pair.tokens

    @property
    def base(self) -> probe.CellAssignment:
        return self._pair.base

    @property
    def variant(self) -> probe.CellAssignment:
        return self._pair.variant

    @property
    def varied(self) -> probe.HighLevelVariable:
        return self._pair.varied

    @property
    def span(self) -> tuple[int, int]:
        return self._pair.span

    @property
    def provenance(self) -> probe.Provenance:
        return self._pair.provenance

    @property
    def policy(self) -> probe.AlignmentPolicy:
        return self._pair.policy

    @property
    def tokenizer(self) -> str:
        return self._pair.tokenizer

    def inputs(self, side: probe.Side, /) -> Mapping[str, Tensor]:
        return self._pair.inputs(side)

    def rows(self, side: probe.Side, /) -> tuple[int, ...]:
        return self._pair.rows(side)

    def live_rows(self, side: probe.Side, /) -> int:
        return self._pair.live_rows(side)


def test_an_external_subject_observes_only_exact_pair_rows_in_pair_order() -> None:
    _, causal = subject()
    pairs = (
        model_pair("kim", 0.0, TYPE_VARIABLE, "A", "B"),
        model_pair("lee", 0.1, TYPE_VARIABLE, "A", "B"),
    )

    assert isinstance(causal, probe.Subject)

    with probe.instrument(causal, observe=frozenset({SITE})) as run:
        scores = run(lambda: causal.run(pairs, probe.Side.BASE))

    assert scores.shape == (2, 2)
    assert run.trace[SITE].axes == AXES
    assert run.trace[SITE].tensor.shape == (2, 1, 2)
    assert torch.equal(
        run.trace[SITE].tensor[:, 0],
        torch.tensor([[-2.0, 0.5], [-2.0, 0.5]]),
    )


@pytest.mark.parametrize(
    ("variable", "base_decision", "variant_decision"),
    [
        (SPLIT_VARIABLE, 1, 2),
        (COMBINATOR_VARIABLE, "forward", "backward"),
    ],
)
def test_structural_pairs_run_through_exact_row_local_instrumentation(
    variable: probe.HighLevelVariable,
    base_decision: str | int,
    variant_decision: str | int,
) -> None:
    _, causal = subject()
    pair = model_pair(variable.kind, 0.0, variable, base_decision, variant_decision)

    with probe.instrument(causal, observe=frozenset({SITE})) as run:
        scores = run(lambda: causal.run((pair,), probe.Side.BASE))

    expected = pair.inputs(probe.Side.BASE)["vectors"][
        :, list(pair.rows(probe.Side.BASE))
    ]
    assert pair.base[variable] == base_decision
    assert pair.variant[variable] == variant_decision
    assert scores.shape == (1, 2)
    assert run.trace[SITE].tensor.shape == (1, len(pair.rows(probe.Side.BASE)), 2)
    assert torch.equal(run.trace[SITE].tensor, expected)


def test_an_external_subject_consumes_a_structural_model_ready_wrapper() -> None:
    _, causal = subject()
    concrete = model_pair("kim", 0.0, TYPE_VARIABLE, "A", "B")
    wrapped = WrappedPair(concrete)

    assert isinstance(wrapped, probe.ModelReadyCausalPair)
    assert torch.equal(
        causal.run((wrapped,), probe.Side.BASE),
        causal.run((concrete,), probe.Side.BASE),
    )


def test_a_row_local_patch_leaves_every_sibling_model_row_live() -> None:
    external, causal = subject()
    pairs = (
        model_pair("kim", 0.0, TYPE_VARIABLE, "A", "B"),
        model_pair("lee", 0.1, TYPE_VARIABLE, "A", "B"),
    )
    before = tuple(pair.inputs(probe.Side.BASE)["vectors"] for pair in pairs)

    with probe.instrument(causal, patch={SITE: probe.Zero()}) as run:
        patched = run(lambda: causal.run(pairs, probe.Side.BASE))

    zeroed = torch.cat(before).clone()
    zeroed[:, 1] = 0.0
    expected = external.run({"vectors": zeroed})
    assert isinstance(expected, torch.Tensor)
    assert torch.equal(patched, expected)
    assert all(
        torch.equal(pair.inputs(probe.Side.BASE)["vectors"], original)
        for pair, original in zip(pairs, before, strict=True)
    )
    assert torch.equal(
        causal.run(pairs, probe.Side.BASE), causal.run(pairs, probe.Side.BASE)
    )


def test_distributed_alignment_fits_through_the_public_external_subject() -> None:
    _, causal = subject()
    pairs = (
        model_pair("kim", 0.0, TYPE_VARIABLE, "A", "B"),
        model_pair("lee", 0.1, TYPE_VARIABLE, "A", "B"),
    )
    alignment = probe.DistributedAlignment[
        probe.Observation, probe.ModelReadyCausalPair
    ](
        causal,
        SITE,
        TYPE_VARIABLE,
        search=probe.FixedRankSearch(
            rank=1,
            schedule=probe.Schedule(epochs=1, learning_rate=0.05, batch_size=2, seed=3),
            null=LabelPermutation(replicates=2, seed=5),
            objective=AlignmentObjective.CROSS_ENTROPY,
        ),
    )

    fitted = alignment.fit(pairs)

    assert fitted.fitted is not None
    assert fitted.fitted.pairs == pairs


def test_named_pairs_keep_pair_local_spans_and_rows_through_fit_and_assessment() -> (
    None
):
    _, causal = subject()
    fitting: tuple[probe.ModelReadyCausalPair, ...] = (
        model_pair(
            "kim",
            0.0,
            NAMED_VARIABLE,
            "A",
            "B",
            source_span=(0, 1),
        ),
        model_pair(
            "lee",
            0.1,
            NAMED_VARIABLE,
            "A",
            "B",
            source_span=(2, 3),
        ),
    )
    held_out: tuple[probe.ModelReadyCausalPair, ...] = (
        model_pair(
            "pat",
            0.2,
            NAMED_VARIABLE,
            "A",
            "B",
            source_span=(2, 3),
        ),
        model_pair(
            "sam",
            0.3,
            NAMED_VARIABLE,
            "A",
            "B",
            source_span=(0, 1),
        ),
    )
    search = probe.FixedRankSearch(
        rank=1,
        schedule=probe.Schedule(epochs=1, learning_rate=0.05, batch_size=2, seed=3),
        null=LabelPermutation(replicates=2, seed=5),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    costs = probe.CostModel(
        (
            probe.UnitPrice(
                probe.ProbeUnit.FIXED_ALIGNMENT_FIT,
                seconds_per_run=0.01,
                seconds_fastest=0.01,
                warmup=0,
                repeats=1,
                device="cpu",
            ),
            probe.UnitPrice(
                probe.ProbeUnit.FIXED_PERMUTATION_REFIT,
                seconds_per_run=0.01,
                seconds_fastest=0.01,
                warmup=0,
                repeats=1,
                device="cpu",
            ),
        )
    )
    preregistration = probe.Preregistration(
        probe.Subgrid((NAMED_VARIABLE,), (0,), (0,), {(0, 0): SITE}),
        search,
        costs=costs,
        budget_hours=1.0,
    )

    assert tuple(pair.span for pair in fitting) == ((0, 1), (2, 3))
    assert tuple(pair.rows(probe.Side.BASE) for pair in fitting) == ((0,), (2,))
    assert tuple(pair.span for pair in held_out) == ((2, 3), (0, 1))
    assert tuple(pair.rows(probe.Side.BASE) for pair in held_out) == ((2,), (0,))
    with probe.instrument(causal, observe=frozenset({SITE})) as run:
        run(lambda: causal.run(fitting, probe.Side.BASE))
    assert torch.equal(
        run.trace[SITE].tensor[:, 0],
        torch.tensor([[-2.0, 0.5], [-2.0, 0.5]]),
    )

    report = probe.sweep(
        causal,
        preregistration,
        {NAMED_VARIABLE: fitting},
        {NAMED_VARIABLE: held_out},
    )
    result = report.result_at(probe.GridCell(NAMED_VARIABLE, 0, 0))

    assert report.cells == (probe.GridCell(NAMED_VARIABLE, 0, 0),)
    assert result.site == SITE
    assert result.result.variable is NAMED_VARIABLE
    assert not hasattr(result.result, "cell")


def test_an_external_subject_refuses_duplicate_pairs_and_unknown_decisions() -> None:
    external, causal = subject()
    pair = model_pair("kim", 0.0, NAMED_VARIABLE, "A", "B", source_span=(0, 1))

    with pytest.raises(ValueError, match="repeats a causal pair identity"):
        causal.run((pair, pair), probe.Side.BASE)

    unknown = causal_subject(external, {NAMED_VARIABLE: ("B", "C")}, score)

    with pytest.raises(ValueError, match="outside"):
        unknown.run((pair,), probe.Side.BASE)


def test_an_external_subject_refuses_mixed_variables_before_the_model_runs() -> None:
    external, _ = subject()
    other = probe.NamedVariable("external", "semantic-role")
    causal = causal_subject(
        external,
        {NAMED_VARIABLE: ("A", "B"), other: ("A", "B")},
        score,
    )

    with pytest.raises(ValueError, match="mixes"):
        causal.run(
            (
                model_pair("kim", 0.0, NAMED_VARIABLE, "A", "B", source_span=(0, 1)),
                model_pair("lee", 0.1, other, "A", "B", source_span=(2, 3)),
            ),
            probe.Side.BASE,
        )


def test_an_external_subject_refuses_malformed_scores() -> None:
    external = external_model(TinyModel())
    pairs = (
        model_pair("kim", 0.0, NAMED_VARIABLE, "A", "B", source_span=(0, 1)),
        model_pair("lee", 0.1, NAMED_VARIABLE, "A", "B", source_span=(2, 3)),
    )

    def wrong_shape(
        output: probe.Output,
        variable: probe.HighLevelVariable,
        decisions: tuple[str | int, ...],
    ) -> torch.Tensor:
        del output, variable, decisions

        return torch.zeros(1, 2)

    def nonfinite(
        output: probe.Output,
        variable: probe.HighLevelVariable,
        decisions: tuple[str | int, ...],
    ) -> torch.Tensor:
        del output, variable, decisions

        return torch.full((2, 2), float("nan"))

    with pytest.raises(ValueError, match="returned shape"):
        causal_subject(external, {NAMED_VARIABLE: ("A", "B")}, wrong_shape).run(
            pairs, probe.Side.BASE
        )

    with pytest.raises(ValueError, match="nonfinite score"):
        causal_subject(external, {NAMED_VARIABLE: ("A", "B")}, nonfinite).run(
            pairs, probe.Side.BASE
        )


def test_an_external_subject_refuses_a_runtime_row_extent_mismatch() -> None:
    external = external_model(ShortModel())
    causal = causal_subject(external, {NAMED_VARIABLE: ("A", "B")}, score)

    with (
        pytest.raises(ValueError, match="returned row extent 2"),
        probe.instrument(causal, observe=frozenset({SITE})) as run,
    ):
        run(
            lambda: causal.run(
                (
                    model_pair(
                        "kim",
                        0.0,
                        NAMED_VARIABLE,
                        "A",
                        "B",
                        source_span=(0, 1),
                    ),
                ),
                probe.Side.BASE,
            )
        )

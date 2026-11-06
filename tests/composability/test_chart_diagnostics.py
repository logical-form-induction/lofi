from typing import Final

import torch
from tests._controllers import mlp_controller

from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.parsers import (
    DeepInsideOutside,
    LogSumExpReduction,
    MarginalSplitWeigher,
    MeanFieldComposition,
    MlpSpanFunction,
)
from lofi.nn.psi import RuleScorer
from lofi.nn.sites import ComponentId, Role
from lofi.probe import (
    CutpointEntropy,
    DiagnosticContext,
    Observation,
    SpanLengthBias,
    SpanWeighting,
    Trace,
)

INPUT_WIDTH: Final = 3
CELL_WIDTH: Final = 4
HIDDEN: Final = 5
PARSER: Final = ComponentId("chart")
CONTROLLER: Final = ComponentId("controller")

APPLY: Final = Action("apply", "application")
COMPOSE: Final = Action("compose", "composition")
KEEP: Final = Action("keep")
ACTIONS: Final = ActionSet((APPLY, COMPOSE, KEEP), KEEP)


def scorer(seed: int) -> DeepInsideOutside:
    """Return one chart scorer initialized under a fixed seed.

    Parameters
    ----------
    seed : int
        The parameter initialization seed.
    """
    torch.manual_seed(seed)
    controller = mlp_controller(
        ACTIONS,
        width=CELL_WIDTH,
        component=CONTROLLER,
        hidden=HIDDEN,
    )
    rules = RuleScorer(controller, width=CELL_WIDTH)

    return DeepInsideOutside(
        rules,
        composition=MeanFieldComposition(
            MlpSpanFunction((INPUT_WIDTH,), CELL_WIDTH, layers=1, nonlinearity="tanh"),
            MlpSpanFunction(
                (CELL_WIDTH, CELL_WIDTH),
                CELL_WIDTH,
                layers=1,
                nonlinearity="tanh",
            ),
            width=CELL_WIDTH,
        ),
        split_left=MlpSpanFunction(
            (CELL_WIDTH, CELL_WIDTH),
            CELL_WIDTH,
            layers=1,
            nonlinearity="tanh",
        ),
        split_right=MlpSpanFunction(
            (CELL_WIDTH, CELL_WIDTH),
            CELL_WIDTH,
            layers=1,
            nonlinearity="tanh",
        ),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=PARSER,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    ).to(torch.float64)


def leaves() -> Encoded:
    """Return a literal ragged batch of pooled word vectors."""
    tensor = torch.tensor(
        [
            [
                [0.2, -0.1, 0.5],
                [0.7, 0.3, -0.4],
                [-0.2, 0.8, 0.1],
                [0.4, -0.6, 0.9],
            ],
            [
                [-0.5, 0.2, 0.6],
                [0.1, 0.9, -0.3],
                [0.8, -0.7, 0.4],
                [0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float64,
    )
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

    return Encoded(tensor, ("item", "word", "width"), mask)


def diagnose(subject: DeepInsideOutside) -> tuple[float, float, int, int]:
    """Return both chart scores and their eligible-span counts.

    Parameters
    ----------
    subject : DeepInsideOutside
        The real scorer to observe for one pass.
    """
    published = subject.site_map()
    split = next(site for site in published if site.role is Role.SPLIT_DISTRIBUTION)
    marginal = next(
        site for site in published if site.role is Role.CONSTITUENT_MARGINAL
    )
    selected = frozenset({split, marginal})

    with subject.instrument(selected, patches={}) as scope:
        subject.readouts(leaves())

    trace = Trace()

    for site, reading in scope.readings.items():
        trace.record(
            Observation(
                site,
                reading.tensor,
                axes=reading.axes,
                mask=reading.mask,
            )
        )

    trace.close()
    marginal_reading = trace[marginal]
    context = DiagnosticContext.observing(
        subject,
        span_index=SpanIndex(4),
        weighting=SpanWeighting(marginal_reading.tensor, axes=marginal_reading.axes),
    )
    entropy = (
        CutpointEntropy()
        .run(trace, context=context)
        .aggregate(split, "normalized-cutpoint-entropy")
    )
    length = (
        SpanLengthBias()
        .run(trace, context=context)
        .aggregate(marginal, "normalized-span-length")
    )

    return entropy.weighted, length.weighted, entropy.count, length.count


def test_real_chart_observations_compose_with_both_diagnostics() -> None:
    first = diagnose(scorer(7))
    changed = diagnose(scorer(19))

    assert first[2:] == (2, 7)
    assert changed[2:] == (2, 7)
    assert first[:2] != changed[:2]

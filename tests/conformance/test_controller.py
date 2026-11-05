from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.controller import (
    CONTROLLER_LAWS,
    ControllerConformance,
    ControllerSuite,
)

from lofi._checks import LawViolationError
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.probe import ensure_publishes

if TYPE_CHECKING:
    from collections.abc import Callable

JOIN: Final = Action("join", "concatenation")
APPLY: Final = Action("apply", "application")
KEEP: Final = Action("keep")

ACTIONS: Final = ActionSet((JOIN, KEEP, APPLY), KEEP)

COMPONENT: Final = ComponentId("gap-controller")

WEIGHING: Final = Site(COMPONENT, Role.COMBINATOR_LOGITS)

HIDDEN: Final = Site(COMPONENT, Role.RECURRENT_HIDDEN)

SCORES: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE | Access.PATCH,
)


def cells(pairs: int) -> Encoded:
    """Return the vectors of a run of chart cells.

    Parameters
    ----------
    pairs
        How many cells to build.

    Returns
    -------
    Encoded
        Rank 2 over the axes ``("cell", "width")``, at double precision.
    """
    return Encoded(
        torch.arange(pairs, dtype=torch.float64).unsqueeze(-1), ("cell", "width")
    )


class Gap:
    def __init__(self, actions: ActionSet) -> None:
        self._actions = actions

    @property
    def action_set(self) -> ActionSet:
        return self._actions

    @property
    def input_width(self) -> int:
        return 1

    @property
    def input(self) -> str:
        return "inside"

    def columns(self) -> int:
        return len(self.action_set.actions)

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        gap = (left.tensor - right.tensor).abs().sum(dim=-1)
        scale = torch.arange(1, self.columns() + 1, dtype=gap.dtype)

        scores = -gap.unsqueeze(-1) * scale

        return LogDistribution(
            scores.log_softmax(dim=-1),
            ("cell", "action"),
            normalized_over="action",
        )

    def site_map(self) -> SiteMap:
        return SiteMap(COMPONENT, ComponentKind.CONTROLLER, {WEIGHING: SCORES})

    def load(self, digest: str) -> None:
        """Read a checkpoint recorded under an action-set digest.

        Parameters
        ----------
        digest
            The digest the checkpoint stored.

        Raises
        ------
        ValueError
            If it is not this controller's own, since the columns of that checkpoint
            point at other actions.
        """
        if digest != self.action_set.digest:
            msg = (
                f"the checkpoint records the action set {digest}, and this "
                f"controller weighs {self.action_set.digest}"
            )
            raise ValueError(msg)


class Counting(Gap):
    def __init__(self, actions: ActionSet, columns: int = 2) -> None:
        super().__init__(actions)
        self._columns = columns

    @override
    def columns(self) -> int:
        return self._columns


class Sorting(Gap):
    def __init__(self, actions: ActionSet) -> None:
        super().__init__(
            ActionSet(
                tuple(sorted(actions.actions, key=lambda action: action.name)),
                actions.identity,
            )
        )


class Reserving(Gap):
    def __init__(self, actions: ActionSet) -> None:
        super().__init__(ActionSet(actions.actions, actions.actions[-1]))


class Trusting(Gap):
    @override
    def load(self, digest: str) -> None: ...


class Boasting(Gap):
    @override
    def site_map(self) -> SiteMap:
        return SiteMap(
            COMPONENT,
            ComponentKind.CONTROLLER,
            {WEIGHING: SCORES, HIDDEN: SCORES},
        )


def _against[T: Gap](build: Callable[[ActionSet], T]) -> ControllerConformance[T]:
    """Return the conformance the laws run one kind of controller through.

    Parameters
    ----------
    build
        The controller to build for an action set.

    Returns
    -------
    ControllerConformance
        The fixture, over the three-action set and the cells above.
    """
    return ControllerConformance(
        build=build,
        actions=ACTIONS,
        cells=cells,
        load=lambda subject, digest: subject.load(digest),
        sites=SiteObservation(
            build=lambda: build(ACTIONS),
            observe=lambda subject: {WEIGHING: subject.weigh(cells(3), right=cells(3))},
            read=lambda subject: (subject.weigh(cells(3), right=cells(3)).tensor,),
        ),
    )


def _red[T: Gap](conformance: ControllerConformance[T]) -> frozenset[str]:
    """Return the name of every law that fails against a subject.

    Parameters
    ----------
    conformance
        The subject and what the laws need to run against it.

    Returns
    -------
    frozenset of str
        The names of the failing laws.
    """
    suite = ControllerSuite()
    red: list[str] = []

    for law in CONTROLLER_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, Callable[[ActionSet], Gap]], ...]] = (
    ("test_the_column_count_is_the_size_of_the_action_set", Counting),
    ("test_the_column_order_is_the_action_order", Sorting),
    ("test_the_identity_is_a_column_rather_than_an_increment", Reserving),
    ("test_a_checkpoint_from_another_action_set_is_refused", Trusting),
    ("test_the_site_map_matches_what_instrumentation_yields", Boasting),
)


class TestAControllerThatReadsItsActionSet(ControllerSuite):
    @pytest.fixture
    def conformance(self) -> ControllerConformance[Gap]:
        return _against(Gap)


def test_every_law_holds_of_a_controller_that_reads_its_set() -> None:
    assert _red(_against(Gap)) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(
    law: str, build: Callable[[ActionSet], Gap]
) -> None:
    assert _red(_against(build)) == frozenset({law})


def test_the_published_point_describes_the_weights_the_controller_returns() -> None:
    controller = Gap(ACTIONS)
    published = controller.site_map()
    spec = published[WEIGHING]

    weights = controller.weigh(cells(3), right=cells(3))

    assert published.kind is ComponentKind.CONTROLLER
    assert ensure_publishes(published) is published
    assert spec.axes == weights.axes
    assert spec.dtype_class is DTypeClass.LOG_PROBABILITY
    assert weights.tensor.shape[spec.position("action")] == len(ACTIONS.actions)
    assert weights.normalized_over == "action"
    assert Access.OBSERVE in spec.access

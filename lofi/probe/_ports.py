"""Probing protocols."""

from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from lofi.probe._causal import CausalPair, JointCausalPair

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.nn.ports import InstrumentScope, Patch, Reading
    from lofi.nn.sites import Role, Site, SiteMap
    from lofi.probe._causal import (
        AlignmentPolicy,
        CounterfactualPair,
        HighLevelVariable,
        IIAResult,
        JointCounterfactualPair,
        Side,
    )
    from lofi.probe._circuits import Unit
    from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
    from lofi.probe._joint import JointReport
    from lofi.probe._selectivity import ControlTask, ProbeCapacity, SpanInventory
    from lofi.probe._values import Trace
    from lofi.terms import BatchEncoding

__all__ = [
    "Alignment",
    "Classifier",
    "Diagnostic",
    "JointAlignment",
    "JointSubject",
    "ModelReadyCausalPair",
    "ModelReadyJointCausalPair",
    "ProbeReport",
    "Probed",
    "Subject",
    "TermEncoder",
    "TermVectors",
]


@runtime_checkable
class ModelReadyCausalPair(CausalPair, Protocol):
    """A causal pair carrying exact inputs and tokenizer rows for a model."""

    @property
    def id(self) -> str:
        """Return the pair's stable identity."""

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the source tokens shared by both assignments."""

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the policy under which model rows were selected."""

    @property
    def tokenizer(self) -> str:
        """Return the tokenizer identity shared by both model sides."""

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        """Return a fresh snapshot of one side's named model inputs."""

    def rows(self, side: Side, /) -> tuple[int, ...]:
        """Return exact model rows selected for one side."""

    def live_rows(self, side: Side, /) -> int:
        """Return the live-row extent of one side's tokenizer input."""


@runtime_checkable
class ModelReadyJointCausalPair(JointCausalPair, Protocol):
    """A joint structural pair carrying exact inputs and per-variable model rows."""

    @property
    def id(self) -> str:
        """Return the pair's stable identity."""

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the source tokens shared by both assignments."""

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the policy under which model rows were selected."""

    @property
    def tokenizer(self) -> str:
        """Return the tokenizer identity shared by both model sides."""

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        """Return a fresh snapshot of one side's named model inputs."""

    def rows(self, side: Side, variable: HighLevelVariable, /) -> tuple[int, ...]:
        """Return exact model rows selected for one changed variable."""

    def selected_rows(self, side: Side, /) -> tuple[int, ...]:
        """Return the ordered union of both changed variables' model rows."""

    def live_rows(self, side: Side, /) -> int:
        """Return the live-row extent of one side's tokenizer input."""


@runtime_checkable
class Alignment[PairT: CausalPair = CounterfactualPair](Protocol):
    """A derivation decision's model alignment.

    Attributes
    ----------
    policy : AlignmentPolicy
        The span-to-token rules this alignment was fitted under.
    rank : int
        The dimension the featurizer is allowed to look in.
    """

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the span-to-token rules this alignment was fitted under."""

    @property
    def rank(self) -> int:
        """Return the dimension the featurizer is allowed to look in."""

    def fit(self, pairs: Sequence[PairT]) -> Self:
        """Return this alignment fitted to the decisions the pairs vary.

        Parameters
        ----------
        pairs : Sequence[PairT]
            The pairs to fit on.

        Returns
        -------
        Self
            A fitted alignment.
        """

    def apply(self, variable: HighLevelVariable) -> frozenset[Unit]:
        """Return the units where this alignment realizes the variable.

        Parameters
        ----------
        variable : HighLevelVariable
            The decision to locate.

        Returns
        -------
        frozenset[Unit]
            The units realizing it.
        """

    def iia(self, pairs: Sequence[PairT]) -> IIAResult:
        """Return how often intervening on those units produced the prediction.

        Parameters
        ----------
        pairs : Sequence[PairT]
            The pairs to test on, held out from the ones fitting saw.

        Returns
        -------
        IIAResult
            The accuracy, exact variable, rank, and null it was tested against.
        """


class Diagnostic(Protocol):
    """One measurement over a scope's readings.

    Attributes
    ----------
    tag : ClassVar[str]
        The name this diagnostic registered under, unique across the registry.
    requires : ClassVar[frozenset[Role]]
        The roles a model must publish for this diagnostic to apply.
    scores : ClassVar[frozenset[str]]
        The measurement names this diagnostic reports on the closed unit interval.
    """

    tag: ClassVar[str]
    requires: ClassVar[frozenset[Role]]
    scores: ClassVar[frozenset[str]]

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return what this diagnostic measured over one scope's readings.

        Parameters
        ----------
        trace : Trace
            The readings the scope took, keyed by the point each came from.
        context : DiagnosticContext
            What the components under study published, which is where a reading's axes
            and dtype class are read from.

        Returns
        -------
        DiagnosticResult
            The measurements, under this diagnostic's tag.

        Raises
        ------
        KeyError
            If the trace holds a reading at a point the context does not publish.
        """


class Probed[ReadingT: Reading](Protocol):
    """A component that supports instrumented regions."""

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this component.

        Returns
        -------
        SiteMap
            Each point published, under the specification of what is found there.
        """

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[ReadingT]]
    ) -> InstrumentScope[ReadingT]:
        """Return the region over which those points are observed and patched.

        Parameters
        ----------
        sites : frozenset[Site]
            The points to read, drawn from what this component publishes.
        patches : Mapping[Site, Patch[ReadingT]]
            What replaces the value at a point, keyed by the point.

        Returns
        -------
        InstrumentScope[ReadingT]
            The region, not yet entered.
        """


@runtime_checkable
class Subject[ReadingT: Reading, PairT: CausalPair = CounterfactualPair](
    Probed[ReadingT], Protocol
):
    """A component whose decisions at a high-level variable are read off its output.

    Attributes
    ----------
    policy : AlignmentPolicy
        The span-to-token rules the subject presents each pair's derivations under.
    """

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the span-to-token rules the subject presents each pair under."""

    def decisions(self, variable: HighLevelVariable, /) -> tuple[str | int, ...]:
        """Return the decisions the subject can make at a variable, in score order.

        Parameters
        ----------
        variable : HighLevelVariable
            The decision to enumerate the alternatives of.

        Returns
        -------
        tuple[str | int, ...]
            Every alternative, in the order the subject scores them.
        """

    def run(self, pairs: Sequence[PairT], side: Side, /) -> Tensor:
        """Run one side of each pair and score the decisions at its varied variable.

        Parameters
        ----------
        pairs : Sequence[PairT]
            The pairs to run, each varying the same variable.
        side : Side
            Which of each pair's two derivations to run.

        Returns
        -------
        Tensor
            Rank 2 over the pairs and then the alternatives at the varied variable,
            in the order ``decisions`` names them, higher scoring the more likely.
        """


@runtime_checkable
class JointSubject[
    ReadingT: Reading,
    PairT: JointCausalPair = JointCounterfactualPair,
](Probed[ReadingT], Protocol):
    """A component scoring both decisions changed by a joint causal pair."""

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the span-to-token rules the subject presents each pair under."""

    def decisions(self, variable: HighLevelVariable, /) -> tuple[str | int, ...]:
        """Return one changed variable's alternatives in score order."""

    def run(self, pairs: Sequence[PairT], side: Side, /) -> tuple[Tensor, Tensor]:
        """Run one side and score the dominant and dominated variables separately."""


@runtime_checkable
class JointAlignment[PairT: JointCausalPair = JointCounterfactualPair](Protocol):
    """A simultaneous alignment of two dominance-related variables."""

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the span-to-token policy used for fitting and evaluation."""

    @property
    def widths(self) -> tuple[int, int]:
        """Return the two disjoint mask widths in dominance order."""

    def fit(self, pairs: Sequence[PairT]) -> Self:
        """Return a new joint alignment fitted to the supplied pairs."""

    def matches(self, pairs: Sequence[PairT]) -> tuple[tuple[bool, bool], ...]:
        """Return one pair of marginal outcomes per joint counterfactual."""

    def report(self, pairs: Sequence[PairT]) -> JointReport:
        """Return both marginals, their product, joint accuracy, and its interval."""


@runtime_checkable
class TermVectors(Protocol):
    """A tensor whose axes are named and whose unfilled positions are marked.

    Attributes
    ----------
    tensor : Tensor
        The values.
    axes : tuple[str, ...]
        The name of each axis, outermost first.
    mask : Tensor | None
        Which positions of the outermost axes hold data, or None where all do.
    """

    @property
    def tensor(self) -> Tensor:
        """Return the values."""

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""

    @property
    def mask(self) -> Tensor | None:
        """Return which positions of the outermost axes hold data, or None."""


@runtime_checkable
class TermEncoder(Protocol):
    """A component encoding a batch of laid-out terms as one vector each."""

    def encode(self, terms: BatchEncoding, /) -> TermVectors:
        """Return one vector per term of a laid-out batch.

        Parameters
        ----------
        terms : BatchEncoding
            The terms as one codec laid them out, in batch order.

        Returns
        -------
        TermVectors
            One vector per term, over an axis for the terms and one for their
            coordinates, masked only where a row of the batch holds no term.
        """


class ProbeReport(Protocol):
    """What a classifier probe reports on one pair of tasks.

    Attributes
    ----------
    accuracy : float
        The share of held-out spans labeled correctly on the real task.
    control_accuracy : float
        The share labeled correctly on the control task.
    selectivity : float
        How far the real task ran ahead of the control task.
    control_seed : int
        The seed the control task drew its labels under.
    capacity : ProbeCapacity
        What both fits were run at.
    parameters : int
        How many weights each fit moved.
    """

    @property
    def accuracy(self) -> float:
        """Return the share of held-out spans labeled correctly on the real task."""

    @property
    def control_accuracy(self) -> float:
        """Return the share labeled correctly on the control task."""

    @property
    def selectivity(self) -> float:
        """Return how far the real task ran ahead of the control task."""

    @property
    def control_seed(self) -> int:
        """Return the seed the control task drew its labels under."""

    @property
    def capacity(self) -> ProbeCapacity:
        """Return what both fits were run at."""

    @property
    def parameters(self) -> int:
        """Return how many weights each fit moved."""


class Classifier(Protocol):
    """A span classifier scored against a control task of the same spans.

    Attributes
    ----------
    capacity : ProbeCapacity
        What both fits run at.
    control : ControlTask
        The labels the real task's accuracy is measured against.
    """

    @property
    def capacity(self) -> ProbeCapacity:
        """Return what both fits run at."""

    @property
    def control(self) -> ControlTask:
        """Return the labels the real task's accuracy is measured against."""

    def under(self, control: ControlTask) -> Self:
        """Return the same probe scored against another control task.

        Parameters
        ----------
        control : ControlTask
            The control task the new probe draws its labels from.

        Returns
        -------
        Self
            A probe of the same capacity under the given control task.
        """

    def run(self, fitting: SpanInventory, held_out: SpanInventory) -> ProbeReport:
        """Return what this probe reached on both tasks and the gap between them.

        Parameters
        ----------
        fitting : SpanInventory
            The spans both fits run on.
        held_out : SpanInventory
            The spans both fits are scored on, held out from the ones fitting saw.

        Returns
        -------
        ProbeReport
            The two accuracies, their gap, the seed the control labels were drawn
            under, the capacity both fits ran at, and how many weights each moved.
        """

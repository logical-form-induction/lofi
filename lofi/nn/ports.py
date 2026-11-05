"""Neural component protocols and forward-pass options."""

from enum import Enum, auto
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    Self,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_finite
from lofi.nn.sampling import KEEP_OFFERED, SamplingStrategy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.judge import ActionSet
    from lofi.nn._values import ChartReadouts, Encoded, LogDistribution, TokenEncoding
    from lofi.nn.sites import ReceiverSide, Site, SiteMap
    from lofi.terms import BatchEncoding, HoleContext, Sort, Term

__all__ = [
    "ActionDecoder",
    "ArityScheme",
    "ChartScorer",
    "Consumption",
    "Controller",
    "ForwardMode",
    "ForwardOptions",
    "FrontierDecoder",
    "InstrumentScope",
    "Instrumentable",
    "MarginalDecoder",
    "NoAdmissibleChoiceError",
    "Observable",
    "Patch",
    "Reading",
    "ReceiverEdge",
    "ReceiverLocalChartScorer",
    "RelationKind",
    "Relaxation",
    "SoftTermEncoder",
    "StandardDeclaration",
    "StructuralDeclaration",
    "StructuralRelation",
    "SubwordPooler",
    "SyntheticTreeEncoder",
    "TeacherForcedDecoder",
    "TermEncoder",
    "TokenEncoder",
    "describe_structure",
    "ensure_mode",
]


class NoAdmissibleChoiceError(ValueError):
    """A partial term whose declared decoder policy admits no next choice."""


class ForwardMode(Enum):
    """Specifies how a decoder obtains terms."""

    #: The terms are given, and the decoder scores them.
    OBSERVE = auto()

    #: The terms are drawn from the decoder's own distribution.
    SAMPLE = auto()

    #: No term is drawn, and the decoder sums over the ones it can reach.
    MARGINALIZE = auto()


#: The option names declared by each forward mode.
_DECLARED: Final[Mapping[ForwardMode, frozenset[str]]] = MappingProxyType(
    {
        ForwardMode.OBSERVE: frozenset(),
        ForwardMode.SAMPLE: frozenset({"sampling_rate", "sampling", "relaxation"}),
        ForwardMode.MARGINALIZE: frozenset(),
    }
)


def _undeclared(option: str, mode: ForwardMode) -> str:
    """Return the message for an option a mode does not declare.

    Parameters
    ----------
    option : str
        The option's name.
    mode : ForwardMode
        The mode that does not declare it.

    Returns
    -------
    str
        The message, naming what the mode does declare.
    """
    declared = _DECLARED[mode]
    takes = ", ".join(sorted(declared)) if declared else "no option"

    return f"{mode.name} declares {takes}, so it has no {option}"


def _checked_rate(value: float) -> float:
    """Return a sampling rate.

    Parameters
    ----------
    value : float
        The fraction of a batch to draw.

    Raises
    ------
    TypeError
        If it is not a number.
    ValueError
        If it is not above zero and at most one.
    """
    rate = ensure_finite("a sampling rate", value)

    if not 0.0 < rate <= 1.0:
        msg = f"a sampling rate lies in (0, 1], not at {rate}"
        raise ValueError(msg)

    return rate


def _checked_temperature(value: float) -> float:
    """Return a relaxation temperature.

    Parameters
    ----------
    value : float
        The temperature the relaxation is drawn at.

    Raises
    ------
    TypeError
        If it is not a number.
    ValueError
        If it is not above zero.
    """
    temperature = ensure_finite("a relaxation temperature", value)

    if temperature <= 0.0:
        msg = f"a relaxation temperature is above zero, not {temperature}"
        raise ValueError(msg)

    return temperature


class Relaxation(NamedTuple):
    """Training-time relaxation of a sampling choice."""

    gumbel_temperature: float
    straight_through: bool


class ForwardOptions:
    """Configures a decoder forward pass.

    Parameters
    ----------
    mode : ForwardMode
        Which pass this is.
    sampling_rate : float | None
        The fraction to draw, or None for the mode's own default.
    sampling : SamplingStrategy | None
        The sampling restriction, or None for the mode's default.
    relaxation : Relaxation | None
        The training relaxation, or None for the mode's default.

    Attributes
    ----------
    mode : ForwardMode
        Which pass this is.
    sampling_rate : float
        The fraction of a batch a sampling pass draws rather than takes as given, above
        zero and at most one.
    sampling : SamplingStrategy
        The restriction a sampling pass draws under.
    relaxation : Relaxation
        The separate training-time relaxation of the draw.

    Raises
    ------
    TypeError
        If the mode is not a mode, if the straight-through flag is not a boolean, or if
        a rate or a temperature is not a number.
    ValueError
        If an option is named that the mode does not declare, if the sampling rate falls
        outside ``(0, 1]``, or if the temperature is not above zero.
    """

    __slots__ = ("_mode", "_relaxation", "_sampling", "_sampling_rate")

    def __init__(
        self,
        mode: ForwardMode,
        *,
        sampling_rate: float | None = None,
        sampling: SamplingStrategy | None = None,
        relaxation: Relaxation | None = None,
    ) -> None:
        if not isinstance(mode, ForwardMode):
            msg = f"a pass runs in a ForwardMode, not {mode!r}"
            raise TypeError(msg)

        declared = _DECLARED[mode]
        named: tuple[tuple[str, object | None], ...] = (
            ("relaxation", relaxation),
            ("sampling", sampling),
            ("sampling_rate", sampling_rate),
        )

        for option, value in named:
            if value is not None and option not in declared:
                raise ValueError(_undeclared(option, mode))

        if sampling is not None and not isinstance(sampling, SamplingStrategy):
            msg = f"a sampling restriction is a SamplingStrategy, not {sampling!r}"
            raise TypeError(msg)

        if relaxation is not None and not isinstance(relaxation, Relaxation):
            msg = f"a relaxation is a Relaxation, not {relaxation!r}"
            raise TypeError(msg)

        self._mode = mode
        self._sampling_rate = (
            _checked_rate(1.0 if sampling_rate is None else sampling_rate)
            if "sampling_rate" in declared
            else None
        )
        if "sampling" in declared:
            self._sampling = (
                SamplingStrategy(KEEP_OFFERED) if sampling is None else sampling
            )
            held = (
                Relaxation(gumbel_temperature=1.0, straight_through=False)
                if relaxation is None
                else relaxation
            )
            temperature = _checked_temperature(held.gumbel_temperature)

            if not isinstance(held.straight_through, bool):
                msg = (
                    "a straight-through flag is a boolean, not "
                    f"{held.straight_through!r}"
                )
                raise TypeError(msg)

            self._relaxation = Relaxation(
                gumbel_temperature=temperature,
                straight_through=held.straight_through,
            )
        else:
            self._sampling = None
            self._relaxation = None

    @property
    def mode(self) -> ForwardMode:
        """Return the forward-pass mode."""
        return self._mode

    @property
    def sampling_rate(self) -> float:
        """Return the fraction of a batch drawn by a sampling pass.

        Raises
        ------
        ValueError
            If this pass's mode does not declare it.
        """
        if self._sampling_rate is None:
            raise ValueError(_undeclared("sampling_rate", self._mode))

        return self._sampling_rate

    @property
    def sampling(self) -> SamplingStrategy:
        """Return the restriction a sampling pass draws under.

        Raises
        ------
        ValueError
            If this pass's mode does not declare it.
        """
        if self._sampling is None:
            raise ValueError(_undeclared("sampling", self._mode))

        return self._sampling

    @property
    def relaxation(self) -> Relaxation:
        """Return the separate training-time relaxation of a sampling pass.

        Raises
        ------
        ValueError
            If this pass's mode does not declare it.
        """
        if self._relaxation is None:
            raise ValueError(_undeclared("relaxation", self._mode))

        return self._relaxation

    def declares(self, option: str) -> bool:
        """Return whether this pass's mode declares an option.

        Parameters
        ----------
        option : str
            The option's name.

        Returns
        -------
        bool
            Whether reading it returns a value rather than raising.
        """
        return option in _DECLARED[self._mode]

    def _identity(
        self,
    ) -> tuple[ForwardMode, float | None, SamplingStrategy | None, Relaxation | None]:
        """Return the fields these options are equal on."""
        return (
            self._mode,
            self._sampling_rate,
            self._sampling,
            self._relaxation,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value describes the same pass."""
        if type(other) is not ForwardOptions:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the mode and the settings it declares."""
        return hash((ForwardOptions, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the mode and every setting it declares, in keyword form."""
        settings = "".join(
            f", {option}={value!r}"
            for option, value in (
                ("sampling_rate", self._sampling_rate),
                ("sampling", self._sampling),
                ("relaxation", self._relaxation),
            )
            if value is not None
        )

        return f"ForwardOptions({self._mode}{settings})"


def ensure_mode(
    options: ForwardOptions, modes: frozenset[ForwardMode], /
) -> ForwardOptions:
    """Return the options a pass runs under.

    Parameters
    ----------
    options : ForwardOptions
        The pass to run.
    modes : frozenset[ForwardMode]
        The modes the decoder declares.

    Returns
    -------
    ForwardOptions
        The options, unchanged.

    Raises
    ------
    ValueError
        If the pass names a mode outside the declared set.
    """
    if options.mode not in modes:
        served = ", ".join(sorted(mode.name for mode in modes)) or "no mode"
        msg = f"this decoder runs in {served}, not in {options.mode.name}"
        raise ValueError(msg)

    return options


@runtime_checkable
class ActionDecoder(Protocol):
    """Decodes parent distributions for rows choosing one controller action."""

    @property
    def modes(self) -> frozenset[ForwardMode]:
        """Return the forward modes this decoder runs in."""

    def compose(
        self, left: Encoded, /, *, right: Encoded, options: ForwardOptions
    ) -> LogDistribution:
        """Return parent-symbol distributions for paired child vectors."""


class RelationKind(Enum):
    """Distinguishes a relation over single slots from one over slot pairs."""

    #: One value per slot.
    UNARY = auto()

    #: One value per ordered pair of slots.
    PAIRWISE = auto()


class StructuralRelation(Enum):
    """Identifies structure a term codec derives from the slots of its layout.

    A unary relation has one value per slot and a pairwise relation one value per
    ordered pair of slots, the first slot of the pair being the one that reads.
    """

    #: The index of a slot in the linearization.
    POSITION = ("position", RelationKind.UNARY)

    #: The number of edges between a slot and the root.
    DEPTH = ("depth", RelationKind.UNARY)

    #: The number of child slots hanging off a slot.
    ARITY = ("arity", RelationKind.UNARY)

    #: Whether the second slot is the parent of the first.
    PARENT = ("parent", RelationKind.PAIRWISE)

    #: Whether the second slot is the first or a slot the first descends from.
    ANCESTOR = ("ancestor", RelationKind.PAIRWISE)

    #: Whether the two slots are distinct and share a parent.
    SIBLING = ("sibling", RelationKind.PAIRWISE)

    #: The number of edges on the path between the two slots.
    TREE_DISTANCE = ("tree_distance", RelationKind.PAIRWISE)

    #: Whether the two slots belong to one operator application: the same slot, a
    #: slot and its parent, or two slots sharing a parent.
    SAME_CONSTITUENT = ("same_constituent", RelationKind.PAIRWISE)

    #: Whether the second slot lies between the lowest and the highest slot of the
    #: subtree rooted at the first.
    SPAN_CONTAINS = ("span_contains", RelationKind.PAIRWISE)

    #: Whether the relation is over single slots or over slot pairs.
    kind: RelationKind

    def __new__(cls, label: str, kind: RelationKind) -> Self:
        """Return the member whose value is its label.

        Parameters
        ----------
        label : str
            The name the relation answers to.
        kind : RelationKind
            Whether it is over single slots or over slot pairs.
        """
        member = object.__new__(cls)
        member._value_ = label
        member.kind = kind

        return member


class Consumption(Enum):
    """Identifies how a component uses a structural relation."""

    #: A hard attention mask: attention is refused where the relation does not
    #: hold.
    MASK = auto()

    #: An additive attention bias, learned per relation.
    BIAS = auto()

    #: An embedding of the relation's value, added to a slot's input.
    EMBEDDING = auto()

    #: A recurrence dependency: a slot's state is computed from the states at the
    #: slots the relation connects it to.
    RECURRENCE = auto()


type StructuralDeclaration = Mapping[StructuralRelation, Consumption]
"""The relations a component reads from its codec, each with how it uses it."""


def describe_structure(declaration: StructuralDeclaration, /) -> str:
    """Return a declaration in the form ``{relation: consumption, ...}``.

    Parameters
    ----------
    declaration : StructuralDeclaration
        The declaration to describe.

    Returns
    -------
    str
        The relations in declaration order, each with its consumption in lowercase,
        or ``{}`` for a component that reads no structure.
    """
    entries = ", ".join(
        f"{relation.value}: {consumption.name.lower()}"
        for relation, consumption in declaration.items()
    )

    return f"{{{entries}}}"


class StandardDeclaration(Enum):
    """Names the structural declarations the benchmark sweeps over.

    Each member names one model of the ablation. Every member reads the position
    of a slot as an embedding, since two slots that share every other relation are
    told apart by nothing else.
    """

    #: Positions only.
    POSITIONS = ((StructuralRelation.POSITION, Consumption.EMBEDDING),)

    #: Positions with depth.
    POSITIONS_WITH_DEPTH = (
        (StructuralRelation.POSITION, Consumption.EMBEDDING),
        (StructuralRelation.DEPTH, Consumption.EMBEDDING),
    )

    #: The ancestor relation as a hard attention mask.
    ANCESTOR_MASK = (
        (StructuralRelation.POSITION, Consumption.EMBEDDING),
        (StructuralRelation.ANCESTOR, Consumption.MASK),
    )

    #: The ancestor and sibling relations as additive attention biases.
    ANCESTOR_AND_SIBLING_BIAS = (
        (StructuralRelation.POSITION, Consumption.EMBEDDING),
        (StructuralRelation.ANCESTOR, Consumption.BIAS),
        (StructuralRelation.SIBLING, Consumption.BIAS),
    )

    #: The parent relation as a recurrence dependency.
    PARENT_RECURRENCE = (
        (StructuralRelation.POSITION, Consumption.EMBEDDING),
        (StructuralRelation.PARENT, Consumption.RECURRENCE),
    )

    @property
    def declaration(self) -> StructuralDeclaration:
        """Return the declaration this member names, as a read-only mapping."""
        return MappingProxyType(dict(self.value))


@runtime_checkable
class ArityScheme(Protocol):
    """Factors a decoder state into child states and combines them back."""

    @property
    def width(self) -> int:
        """Return the size of the feature axis of the states this scheme reads."""

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the operator arities this scheme holds modules for, ascending."""

    def factor(self, parent: Encoded, /, *, arity: int) -> tuple[Encoded, ...]:
        """Return one child state per argument position of an operator.

        Parameters
        ----------
        parent : Encoded
            The state at the node being expanded, over the feature axis innermost.
        arity : int
            How many arguments the chosen operator takes.

        Returns
        -------
        tuple[Encoded, ...]
            One state per argument position, each of the parent's own axes, shape and
            mask. A nullary operator returns nothing.

        Raises
        ------
        ArityUnsupportedError
            If the scheme holds no module at that arity.
        ValueError
            If the parent does not run over the feature axis innermost, or runs over it
            at another width than the scheme reads.
        """

    def combine(self, children: Sequence[Encoded], /) -> Encoded:
        """Return the parent state one operator's children make.

        Parameters
        ----------
        children : Sequence[Encoded]
            One state per argument position, in argument order.

        Returns
        -------
        Encoded
            The parent, of the children's own axes and shape, masked where every child
            holds data.

        Raises
        ------
        ArityUnsupportedError
            If the scheme holds no module at that rank.
        ValueError
            If a child does not run over the feature axis innermost, runs over it at
            another width than the scheme reads, or disagrees with its siblings.
        """


@runtime_checkable
class FrontierDecoder[S](Protocol):
    """Fill the next hole of a term frontier one choice at a time."""

    def start(self, inputs: Encoded, /) -> S:
        """Return the initial frontier state for one encoded input."""

    def weigh(self, state: S, /, *, context: HoleContext) -> LogDistribution:
        """Return the distribution over choices legal at the next hole."""

    def fill(self, state: S, /, *, choices: Tensor) -> S:
        """Return the frontier state after filling its next hole."""

    def admits(self, term: Term, /) -> bool:
        """Return whether one completed frontier term satisfies its policy."""


@runtime_checkable
class TeacherForcedDecoder(Protocol):
    """Report choice rows along supplied terms."""

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        """Return one choice distribution per occupied term slot."""


@runtime_checkable
class MarginalDecoder(Protocol):
    """Report choice rows after summing over preceding choices."""

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: Sort) -> LogDistribution:
        """Return one marginal choice distribution per root-sort layout slot."""


@runtime_checkable
class TermEncoder(Protocol):
    """Encodes terms and their nodes as vectors."""

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of everything this encoder returns."""

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            Rank 2 over the axes ``("term", "width")``, real valued, with a mask only
            where a row of the batch holds no term.
        """

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per node of each term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("term", "slot", "width")``, real valued, masked over
            the term and slot axes by the occupancy of the linearization.
        """

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this encoder."""


@runtime_checkable
class SyntheticTreeEncoder(Protocol):
    """Continue a synthetic operator tree from supplied live leaf vectors."""

    @property
    def output_width(self) -> int:
        """Return the width of each live leaf and completed root."""

    def continue_synthetic(
        self,
        structure: BatchEncoding,
        leaves: Encoded,
        /,
        *,
        leaf_slots: tuple[int, ...],
    ) -> Encoded:
        """Encode roots after installing one live leaf state in each row."""

    def validate_synthetic_continuation(self) -> None:
        """Refuse unless live vectors contain the complete continuation state."""


@runtime_checkable
class SoftTermEncoder(TermEncoder, Protocol):
    """Encode per-slot distributions over language choices."""

    def encode_soft(
        self,
        choices: LogDistribution,
        /,
        *,
        structure: BatchEncoding,
    ) -> Encoded:
        """Return one vector per term for a distribution over its choices.

        Parameters
        ----------
        choices : LogDistribution
            The log probability of each choice at each slot, over the axes ``("term",
            "slot", "choice")`` and normalized over the choice axis.
        structure : BatchEncoding
            The parent, depth, and occupancy channels aligned with the term and slot
            axes. Its symbol channel is ignored.

        Returns
        -------
        Encoded
            Rank 2 over the axes ``("term", "width")``, real valued, with a mask only
            where a row of the batch holds no term.

        Raises
        ------
        ValueError
            If the distribution has incompatible axes, term and slot shape, or
            occupancy mask.
        """


@runtime_checkable
class Controller(Protocol):
    """Scores actions for adjacent chart cells."""

    @property
    def input_width(self) -> int:
        """Return the size of the feature axis of the cells this controller reads."""

    @property
    def input(self) -> str:
        """Return whether this controller reads inside or span representations."""

    @property
    def action_set(self) -> ActionSet:
        """Return the actions this controller weighs, in the order its columns take."""

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        """Return the log weight of each action over a pair of adjacent cells.

        Parameters
        ----------
        left : Encoded
            The left cell of each pair, over the axes ``("cell", "width")``.
        right : Encoded
            The right cell of each pair, in the order of the cell axis.

        Returns
        -------
        LogDistribution
            Rank 2 over the axes ``("cell", "action")``, at floating precision and
            unnormalized.
        """

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this controller.

        Returns
        -------
        SiteMap
            Each point published, under the specification of what is found there.
        """


@runtime_checkable
class TokenEncoder(Protocol):
    """Encodes batch words as contextual rows."""

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of the rows this encoder produces."""

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        """Return the encoder's rows for a batch of items, with their alignment.

        Parameters
        ----------
        items : Sequence[Sequence[str]]
            The words of each item, in item order and then in word order.

        Returns
        -------
        TokenEncoding
            Rows over the axes ``("item", "row", "width")``, real valued and masked over
            the item and row axes, with one alignment per item.
        """

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this encoder.

        Returns
        -------
        SiteMap
            Each point published, under the specification of what is found there.
        """


@runtime_checkable
class SubwordPooler(Protocol):
    """Pools encoder rows into one vector per word."""

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of the vectors this pooler returns."""

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        """Return one vector per word of each item.

        Parameters
        ----------
        encoding : TokenEncoding
            The rows to pool and the alignment saying which rows hold which word.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("item", "word", "width")``, real valued, masked over
            the item and word axes, with a word no row encoded masked off rather than
            filled.
        """

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this pooler.

        Returns
        -------
        SiteMap
            Each point published, under the specification of what is found there.
        """


@runtime_checkable
class ChartScorer(Protocol):
    """Computes typed chart outputs, including any ordered action inventory."""

    @property
    def publishes(self) -> frozenset[str]:
        """Return the readouts this scorer fills, by the name each has on the record."""

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        """Return the quantities the chart computed for a batch.

        Parameters
        ----------
        leaves : Encoded
            One vector per word of each item, over the axes ``("item", "word",
            "width")``.

        Returns
        -------
        ChartReadouts
            The published readouts, each over the axes its own record declares, and
            every unpublished one absent. An action-bearing scorer pairs its complete
            action score tensor with the exact ordered inventory that names its
            columns.
        """

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this scorer.

        Returns
        -------
        SiteMap
            Each point published, under the specification of what is found there.
        """


class ReceiverEdge(NamedTuple):
    """Names one chart-cell activation at the receiver that consumes it.

    Attributes
    ----------
    position : int
        The coordinate of the edge on its site's edge axis.
    source : int
        The dense span identifier of the child cell.
    target : int
        The dense span identifier of the parent cell that receives it.
    split : int
        The target's zero-based split coordinate.
    side : ReceiverSide
        Which input of the binary receiver consumes the child.
    """

    position: int
    source: int
    target: int
    split: int
    side: ReceiverSide

    @property
    def coordinates(self) -> Mapping[str, int]:
        """Return the public site coordinate that names only this edge."""
        return MappingProxyType({"edge": self.position})


@runtime_checkable
class ReceiverLocalChartScorer(ChartScorer, Protocol):
    """Publishes the chart-cell activation consumed at each receiver edge."""

    @property
    def receiver_site(self) -> Site:
        """Return the observe-and-patch site holding receiver-edge activations."""

    def receiver_edges(self, index: SpanIndex, /) -> tuple[ReceiverEdge, ...]:
        """Return the edge-axis metadata for one chart index in axis order."""


@runtime_checkable
class Reading(Protocol):
    """A value observed at a model site."""

    @property
    def site(self) -> Site:
        """Return the point the value was read at."""

    @property
    def tensor(self) -> Tensor:
        """Return the observed tensor."""

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""

    @property
    def mask(self) -> Tensor | None:
        """Return the validity mask, or None if every position is valid."""


@runtime_checkable
class Patch[ReadingT: Reading](Protocol):
    """Replaces a model-site value during a forward pass."""

    def stands_in_for(self, reading: ReadingT, /) -> Tensor:
        """Return the replacement value for a site.

        Parameters
        ----------
        reading : ReadingT
            What was found there, which a patch may read the shape, the axes and the
            mask of.

        Returns
        -------
        Tensor
            The replacement, of the reading's own shape.
        """


@runtime_checkable
class InstrumentScope[ReadingT: Reading](Protocol):
    """Instrumentation state for one component call."""

    @property
    def readings(self) -> Mapping[Site, ReadingT]:
        """Return the values observed during the scope."""

    def __enter__(self) -> Self:
        """Return the scope, with its points observed and patched."""

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        """Leave the region and restore the component after success or failure."""


@runtime_checkable
class Observable(Protocol):
    """Publishes named instrumentation sites."""

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this component.

        Returns
        -------
        SiteMap
            Each point published, under the specification of what is found there.
        """


@runtime_checkable
class Instrumentable[ReadingT: Reading](Protocol):
    """An instrumentation context for published sites."""

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

        Raises
        ------
        ValueError
            If a point is one this component does not publish, or if a point named for
            patching does not permit it.
        """

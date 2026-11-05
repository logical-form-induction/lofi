"""The compiler from response specifications to response programs.

Compilation produces a design-independent :class:`ResponseProgram`. Binding
sizes its finite sets, builds its host columns, and lays out drawn sites in a
single parameter vector. The declared scale identification determines which
discriminations, linking slopes, or cutpoint variances are fixed. Every
backend scores the programs this module compiles.
"""

import math
from hashlib import sha256
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    cast,
    final,
    override,
)

import torch

from lofi.judgment._design import POSITION, _ensure_readout
from lofi.judgment._program import (
    Binary,
    Bind,
    Coordinate,
    Draw,
    Gather,
    Name,
    ObserveBounded,
    Operator,
    Prior,
    Step,
    TriangularTransform,
    ensure_identifier,
    product,
    total,
)
from lofi.judgment._program import Literal as ProgramLiteral
from lofi.judgment._spec import (
    AuxiliaryParameter,
    Bounded,
    BoundedResponseSpec,
    Choice,
    ConditionalLikelihood,
    CorrelatedRandomSlope,
    Cumulative,
    FactorTerm,
    FixedCutpointVariance,
    FixedDiscrimination,
    IdentityLinkOnLogit,
    InteractionTerm,
    LinkTerm,
    OffsetTerm,
    RandomIntercept,
    RandomSlope,
    TruncatedMarginalLikelihood,
    installed_linkings,
)
from lofi.judgment.bounded import TruncatedNormalFamily
from lofi.judgment.choice import ChoiceFamily
from lofi.judgment.cumulative import (
    CumulativeFamily,
    DiscriminationGroup,
    PredictedParameter,
    ThresholdGroup,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment._design import ResponseDesign
    from lofi.judgment._ports import LinkingCompiler
    from lofi.judgment._program import Expression, Observation
    from lofi.judgment._spec import (
        Predictor,
        PriorSpec,
        ResponseDeclaration,
        ResponseSpec,
        ThresholdPriors,
    )

__all__ = [
    "DEFAULT_COEFFICIENT_PRIOR",
    "DEFAULT_SCALE_PRIOR",
    "ITEM",
    "PARTICIPANT",
    "PLATE",
    "PREDICTOR",
    "RESPONSE",
    "UNIT_RESPONSE",
    "BoundProgram",
    "ConditionalPlan",
    "ResponseCompiler",
    "ResponseProgram",
    "Slot",
    "TruncatedMarginalPlan",
    "index_column",
    "predictor_site",
]

#: The host column holding each row's recorded response level.
RESPONSE: Final = "response"

#: The host column holding a normalized bounded response.
UNIT_RESPONSE: Final = "unit_response"

#: The finite set indexing the response rows.
PLATE: Final = "Resp"

#: The grouping factor naming a design's participant column.
PARTICIPANT: Final = "participant"

#: The grouping factor naming a design's item column.
ITEM: Final = "item"

#: The site holding the linear predictor.
PREDICTOR: Final = "predictor"

#: Prior of a fixed-effect coefficient.
DEFAULT_COEFFICIENT_PRIOR: Final = Prior("Normal", 0.0, 5.0)

#: Prior of the scale of a random effect's offsets.
DEFAULT_SCALE_PRIOR: Final = Prior("HalfNormal", 1.0)

_STANDARD_NORMAL: Final = Prior("Normal", 0.0, 1.0)
"""The distribution of a non-centered random effect's standardized offsets."""


class ConditionalPlan(NamedTuple):
    """The compiled ordinary conditional-likelihood treatment."""

    objective: Literal["conditional"] = "conditional"


class TruncatedMarginalPlan(NamedTuple):
    """The compiled controls for one truncated group-local integral."""

    objective: Literal["truncated_marginal"]
    group: str
    radius: float
    relative_tolerance: float
    absolute_tolerance: float
    quadrature_points: int
    quadrature_seed: int


type ObjectivePlan = ConditionalPlan | TruncatedMarginalPlan


def _objective_plan(spec: ResponseDeclaration) -> ObjectivePlan:
    """Lower a response declaration's objective to its runtime value."""
    if isinstance(spec, BoundedResponseSpec):
        return ConditionalPlan()

    match spec.objective:
        case ConditionalLikelihood():
            return ConditionalPlan()
        case TruncatedMarginalLikelihood() as declared:
            return TruncatedMarginalPlan(
                declared.objective,
                declared.group,
                declared.radius,
                declared.relative_tolerance,
                declared.absolute_tolerance,
                declared.quadrature_points,
                declared.quadrature_seed,
            )
        case unsupported:
            msg = f"the likelihood objective {unsupported!r} has no runtime plan"

            raise TypeError(msg)


def predictor_site(parameter: AuxiliaryParameter, /) -> str:
    """Return the site holding the logarithm of an auxiliary parameter.

    Parameters
    ----------
    parameter : AuxiliaryParameter
        Parameter whose linear predictor the site holds.

    Returns
    -------
    str
        Name of the site the parameter's formula binds its sum to.
    """
    return f"log_{parameter.value}"


def index_column(design: ResponseDesign, name: str) -> tuple[Tensor, int]:
    """Return a grouping factor's dense index column and its level count.

    The presentation position is a column of a forced-choice design: it runs
    over the alternatives of each row in the order they were shown, so it is
    read from the alternative count rather than stored.

    Parameters
    ----------
    design : ResponseDesign
        Design holding the column.
    name : str
        The participant column, the item column, the presentation position,
        or a declared factor.

    Raises
    ------
    ValueError
        If the design has no such column.
    """
    if name == POSITION:
        shown = design.alternatives_per_row

        if shown is None:
            msg = "the design shows no alternatives, and so no presentation position"

            raise ValueError(msg)

        return torch.arange(shown).repeat(design.n_rows), shown

    if name == PARTICIPANT:
        return design.participants, len(design.participant_levels)

    if name == ITEM:
        return design.items, len(design.item_levels)

    column = design.factors.get(name)

    if column is None:
        msg = (
            f"the design has no grouping factor {name!r}; it has "
            f"{(PARTICIPANT, ITEM, *design.factors)!r}"
        )

        raise ValueError(msg)

    return column, len(design.factor_levels[name])


def _object_of(factors: tuple[str, ...]) -> str:
    """Return the name of the finite set indexing a product of factors.

    Parameters
    ----------
    factors : tuple[str, ...]
        Factor names, at least one.
    """
    return "By".join(
        "".join(part.capitalize() for part in factor.split("_")) for factor in factors
    )


def _prior_of(declared: PriorSpec) -> Prior:
    """Return a declared prior as a program prior.

    Parameters
    ----------
    declared : PriorSpec
        Prior as the specification declares it.
    """
    return Prior(declared.distribution, *declared.parameters)


def _bindable(expression: Expression) -> Expression:
    """Return an expression a site can be bound to.

    A site is bound to a bare name only when that name is itself a site, so
    an expression that is one host column alone is written as one times the
    column.

    Parameters
    ----------
    expression : Expression
        Expression over sites and host columns.
    """
    if isinstance(expression, Name):
        return Binary(Operator.MUL, ProgramLiteral(1.0), expression)

    return expression


class _Effect(NamedTuple):
    """A fixed effect's name, the factors it is indexed by, and its link terms."""

    name: str
    factors: tuple[str, ...]
    links: tuple[LinkTerm, ...]


class _Scales(NamedTuple):
    """What the declared identification leaves free on the latent scale."""

    #: Prior of a drawn discrimination, the value it is held at, or None for one.
    discrimination: Prior | float | None

    #: Variance imposed on each row of cutpoints, or None to leave it free.
    cutpoint_variance: float | None

    #: Whether every link term enters at a slope held at one.
    unit_slope: bool

    #: Whether a fitted linking slope may cross zero.
    signed_slope: bool

    #: Linear predictor the discrimination follows, or None where it follows none.
    predicted: PredictedParameter | None = None


class _Shape(NamedTuple):
    """The sets and columns a response family adds to a compiled program."""

    #: Sets the family declares, beside the response plate.
    reserved: tuple[str, ...]

    #: Names the family binds, in binding order.
    bound: tuple[str, ...]

    #: Set indexing the thresholds, if the family observes any.
    cutpoint_object: str | None

    #: Grouping factor the thresholds vary over, if they vary over one.
    threshold_group: str | None

    #: One-element set a single draw is indexed by.
    scalar_object: str

    #: Set indexing an alternative's place, if rows show alternatives.
    alternative_object: str | None

    #: Number of alternatives every row shows, if rows show any.
    alternatives: int | None

    #: Host column weighting each place in a difference, if one is taken.
    contrast: str | None

    #: Weight of each place in that difference, in presentation order.
    contrast_weights: tuple[float, ...]


def _distinct(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return names in first-seen order without repeats.

    Parameters
    ----------
    names : tuple[str, ...]
        Names in declaration order, possibly repeated.
    """
    return tuple(dict.fromkeys(names))


def _fixed_effect(term: Predictor) -> _Effect:
    """Return the form of a fixed effect.

    Parameters
    ----------
    term : Predictor
        A link term, a factor term, or an interaction of fixed effects.

    Raises
    ------
    ValueError
        If the term is not a fixed effect.
    """
    match term:
        case LinkTerm():
            return _Effect(term.readout, (), (term,))
        case FactorTerm():
            return _Effect(term.name, (term.name,), ())
        case InteractionTerm():
            left = _fixed_effect(term.left)
            right = _fixed_effect(term.right)

            return _Effect(
                f"{left.name}_by_{right.name}",
                _distinct(left.factors + right.factors),
                left.links + right.links,
            )
        case _:
            msg = f"a fixed effect was expected and the {term.kind} term was given"

            raise ValueError(msg)


@final
class _Draft:
    """The steps of one program body as they accumulate.

    Parameters
    ----------
    unit : str
        Name of the one-element set a scalar draw is indexed by.
    unit_slope : bool
        Whether every link term enters at a slope held at one.
    signed_slope : bool
        Whether a fitted linking slope may cross zero.
    """

    __slots__ = (
        "bound",
        "indices",
        "readouts",
        "signed_slope",
        "steps",
        "terms",
        "unit",
        "unit_slope",
    )

    def __init__(self, unit: str, *, unit_slope: bool, signed_slope: bool) -> None:
        self.unit = unit
        self.unit_slope = unit_slope
        self.signed_slope = signed_slope
        self.steps: list[Step] = []
        self.bound: list[str] = []
        self.indices: dict[str, tuple[str, ...]] = {}
        self.readouts: list[str] = []
        self.terms: list[str] = []

    def sample(
        self,
        site: str,
        index: tuple[str, ...],
        prior: Prior,
        *,
        event_shape: tuple[int, ...] = (),
    ) -> None:
        """Append a draw of a site indexed by a product of finite sets.

        Parameters
        ----------
        site : str
            Site to bind.
        index : tuple[str, ...]
            Finite sets whose product the draw is indexed by.
        prior : Prior
            Distribution drawn from.
        """
        self.steps.append(Draw(site, index, prior, event_shape=event_shape))
        self.bound.append(site)

    def let(self, site: str, expression: Expression) -> None:
        """Append a deterministic binding.

        Parameters
        ----------
        site : str
            Site to bind.
        expression : Expression
            Expression bound to it.
        """
        self.steps.append(Bind(site, expression))
        self.bound.append(site)

    def index(self, factors: tuple[str, ...]) -> str:
        """Return the host column holding the row index into a factor product.

        Parameters
        ----------
        factors : tuple[str, ...]
            Factor names, at least one.

        Raises
        ------
        ValueError
            If the column's name is already the index of another product.
        """
        name = "_by_".join(factors)
        held = self.indices.setdefault(name, factors)

        if held != factors:
            msg = f"the host column {name!r} indexes both {held!r} and {factors!r}"

            raise ValueError(msg)

        return name


@final
class _Objects:
    """The finite sets a program declares, each with one meaning.

    Parameters
    ----------
    reserved : tuple[str, ...]
        The plate and the response family's sets.
    """

    __slots__ = ("_origins",)

    def __init__(self, reserved: tuple[str, ...]) -> None:
        self._origins: dict[str, tuple[str, ...] | None] = dict.fromkeys(reserved)

    def declare(self, factors: tuple[str, ...]) -> None:
        """Declare the set indexing a product of factors.

        Parameters
        ----------
        factors : tuple[str, ...]
            Factor names, at least one.

        Raises
        ------
        ValueError
            If the set's name is not an identifier, or already names a
            reserved set or the product of other factors.
        """
        name = ensure_identifier(_object_of(factors), "an object")
        held = self._origins.setdefault(name, factors)

        if held != factors:
            msg = f"the object {name!r} would index both {held!r} and {factors!r}"

            raise ValueError(msg)

    @property
    def declared(self) -> Mapping[str, tuple[str, ...]]:
        """Return the factors each factor-indexed set is the product of, in order."""
        return {
            name: factors
            for name, factors in self._origins.items()
            if factors is not None
        }


class Slot(NamedTuple):
    """One drawn site's place in the flat parameter vector."""

    site: str
    size: int
    shape: tuple[int, ...] = ()


@final
class ResponseProgram:
    """A response specification compiled to program steps.

    The program is independent of any design. Its finite sets are named
    and its host columns are known, but the sets are sized, the host data
    built and the drawn sites laid out only when it is bound to a design.

    Parameters
    ----------
    levels : int | None
        Number of ordered response levels, or None for a continuous response.
    steps : tuple[Step, ...]
        Draws and bindings of the program, then its one observation.
    objects : Mapping[str, tuple[str, ...]]
        Factors whose product each factor-indexed set indexes, by set, in
        declaration order.
    indices : Mapping[str, tuple[str, ...]]
        Factors whose product each host index column indexes, by column.
    readouts : tuple[str, ...]
        Readout columns the program reads, in first-use order.
    cutpoint_object : str | None
        Finite set indexing the thresholds, or None for a program that
        observes no thresholds.
    threshold_group : str | None
        Grouping factor whose levels each draw their own row of thresholds,
        or None for a program whose thresholds are shared by every row.
    scalar_object : str
        One-element set a single draw is indexed by.
    alternative_object : str | None
        Finite set indexing an alternative's place in presentation order,
        or None for a program whose rows show no alternatives.
    alternatives : int | None
        Number of alternatives every row shows, or None for rows judged on
        their own.
    contrast : str | None
        Host column weighting each place in the difference of two scores,
        or None for a program that takes no such difference.
    contrast_weights : tuple[float, ...]
        Weight of each place in that difference, in presentation order, and
        empty where there is no such column.

    Raises
    ------
    ValueError
        If the steps do not end with the observation, observe before the
        end, or name an alternative count without the set indexing it.
    """

    __slots__ = (
        "_alternative_object",
        "_alternatives",
        "_body",
        "_contrast",
        "_contrast_weights",
        "_cutpoint_object",
        "_groups",
        "_indices",
        "_levels",
        "_objective",
        "_objects",
        "_observation",
        "_readouts",
        "_scalar_object",
        "_threshold_group",
    )

    def __init__(
        self,
        *,
        levels: int | None,
        steps: tuple[Step, ...],
        objects: Mapping[str, tuple[str, ...]],
        indices: Mapping[str, tuple[str, ...]],
        readouts: tuple[str, ...],
        objective: ObjectivePlan,
        groups: tuple[str, ...],
        cutpoint_object: str | None,
        threshold_group: str | None,
        scalar_object: str,
        alternative_object: str | None = None,
        alternatives: int | None = None,
        contrast: str | None = None,
        contrast_weights: tuple[float, ...] = (),
    ) -> None:
        if not steps:
            msg = "a program ends with its observation, and this one has no step"

            raise ValueError(msg)

        if (alternative_object is None) != (alternatives is None):
            msg = (
                f"a program over alternatives names both the set indexing them and "
                f"their number, and this one names {alternative_object!r} and "
                f"{alternatives!r}"
            )

            raise ValueError(msg)

        if (contrast is None) != (not contrast_weights) or (
            contrast is not None and len(contrast_weights) != alternatives
        ):
            msg = (
                f"a program weighting the places of a difference weights each of "
                f"{alternatives!r} alternatives, and this one weights "
                f"{contrast_weights!r} in {contrast!r}"
            )

            raise ValueError(msg)

        last = steps[-1]

        if isinstance(last, Draw | Bind):
            msg = (
                f"a program ends with its observation, and this one ends with {last!r}"
            )

            raise TypeError(msg)

        body: list[Draw | Bind] = []

        for position, step in enumerate(steps[:-1]):
            if not isinstance(step, Draw | Bind):
                msg = (
                    f"a program observes once, at its end, and this one observes "
                    f"at step {position} of {len(steps)}"
                )

                raise TypeError(msg)

            body.append(step)

        self._levels = levels
        self._body = tuple(body)
        self._observation = last
        self._objects: Mapping[str, tuple[str, ...]] = MappingProxyType(dict(objects))
        self._indices: Mapping[str, tuple[str, ...]] = MappingProxyType(dict(indices))
        self._readouts = readouts
        self._objective = objective
        self._groups = groups
        self._cutpoint_object = cutpoint_object
        self._threshold_group = threshold_group
        self._scalar_object = scalar_object
        self._alternative_object = alternative_object
        self._alternatives = alternatives
        self._contrast = contrast
        self._contrast_weights = contrast_weights

    @property
    def levels(self) -> int | None:
        """Return the ordered level count, or None for a continuous response."""
        return self._levels

    @property
    def steps(self) -> tuple[Step, ...]:
        """Return every step of the program, the observation last."""
        return (*self._body, self._observation)

    @property
    def body(self) -> tuple[Draw | Bind, ...]:
        """Return the draws and bindings that precede the observation, in order."""
        return self._body

    @property
    def draws(self) -> tuple[Draw, ...]:
        """Return the drawn sites in draw order."""
        return tuple(step for step in self._body if isinstance(step, Draw))

    @property
    def observation(self) -> Observation:
        """Return the observation the program ends with."""
        return self._observation

    @property
    def threshold_group(self) -> str | None:
        """Return the grouping factor the thresholds vary over, or None."""
        return self._threshold_group

    @property
    def scalar_object(self) -> str:
        """Return the one-element set a single draw is indexed by."""
        return self._scalar_object

    @property
    def alternatives(self) -> int | None:
        """Return how many alternatives every row shows, or None if rows show none."""
        return self._alternatives

    @property
    def objects(self) -> Mapping[str, tuple[str, ...]]:
        """Return the factors each factor-indexed set is the product of."""
        return self._objects

    @property
    def indices(self) -> Mapping[str, tuple[str, ...]]:
        """Return the factors each host index column indexes the product of."""
        return self._indices

    @property
    def readouts(self) -> tuple[str, ...]:
        """Return the readout columns read, in first-use order."""
        return self._readouts

    @property
    def objective(self) -> ObjectivePlan:
        """Return the compiled likelihood treatment."""
        return self._objective

    @property
    def groups(self) -> tuple[str, ...]:
        """Return the declared random-effect grouping factors."""
        return self._groups

    @property
    def host_columns(self) -> tuple[str, ...]:
        """Return every host column the program reads, the response first."""
        weights = () if self._contrast is None else (self._contrast,)

        return (self._observation.response, *self._indices, *self._readouts, *weights)

    def _sized(self, rows: int, cardinalities: Mapping[str, int]) -> dict[str, int]:
        """Return the number of elements of each finite set, in declaration order.

        Parameters
        ----------
        rows : int
            Number of response rows.
        cardinalities : Mapping[str, int]
            Number of levels of every factor the program gathers over.
        """
        sizes = {PLATE: rows}

        if self._cutpoint_object is not None:
            if self._levels is None:
                msg = "a continuous response program cannot declare cutpoints"

                raise ValueError(msg)

            sizes[self._cutpoint_object] = self._levels - 1

        sizes[self._scalar_object] = 1

        if self._alternative_object is not None and self._alternatives is not None:
            sizes[self._alternative_object] = self._alternatives

        for name, factors in self._objects.items():
            size = 1

            for factor in factors:
                size *= cardinalities[factor]

            sizes[name] = size

        return sizes

    def _gathered(self, design: ResponseDesign, factor: str) -> tuple[Tensor, int]:
        """Return a factor's index column at the resolution the predictor runs at.

        A program whose rows show alternatives scores every alternative, so
        a column holding one level per row is repeated once per alternative
        of that row.

        Parameters
        ----------
        design : ResponseDesign
            Design holding the column.
        factor : str
            Factor to read.
        """
        column, count = index_column(design, factor)

        if self._alternatives is None or factor == POSITION:
            return column, count

        return column.repeat_interleave(self._alternatives), count

    def over(self, design: ResponseDesign, /) -> BoundProgram:
        """Return the program bound to a design's dimensions.

        Parameters
        ----------
        design : ResponseDesign
            Design supplying the row count, the level count of every
            grouping factor the program gathers over, and the readouts it
            reads.

        Raises
        ------
        ValueError
            If the design has no rows, another number of response levels or
            alternatives, or lacks a factor or a readout the program reads.
        """
        if design.n_rows == 0:
            msg = "a program is bound to at least one row, and the design has none"

            raise ValueError(msg)

        if design.response_levels != self._levels:
            msg = (
                f"a program over {self._levels!r} response levels was given a "
                f"design over {design.response_levels!r}"
            )

            raise ValueError(msg)

        if design.alternatives_per_row != self._alternatives:
            msg = (
                f"a program over {self._alternatives!r} alternatives per row was "
                f"given a design over {design.alternatives_per_row!r}"
            )

            raise ValueError(msg)

        cardinalities = {
            factor: index_column(design, factor)[1]
            for factors in self._indices.values()
            for factor in factors
        }
        sizes = self._sized(design.n_rows, cardinalities)
        response = design.responses

        if isinstance(self._observation, ObserveBounded):
            normalized = design.unit_responses

            if normalized is None:
                msg = "a bounded observation was given an ordinal response design"

                raise ValueError(msg)

            response = normalized

        data: dict[str, Tensor] = {self._observation.response: response}

        for name, factors in self._indices.items():
            combined = design.participants.new_zeros(design.n_readout_rows)

            for factor in factors:
                column, count = self._gathered(design, factor)
                combined = combined * count + column

            data[name] = combined

        if self._contrast is not None:
            data[self._contrast] = torch.tensor(
                self._contrast_weights,
                dtype=torch.float64,
                device=design.responses.device,
            )

        for name in self._readouts:
            column = design.readouts.get(name)

            if column is None:
                msg = (
                    f"a program reading the readout {name!r} was given a design "
                    f"with the readouts {tuple(design.readouts)!r}"
                )

                raise ValueError(msg)

            data[name] = column

        return BoundProgram(self, sizes=sizes, host_data=data)

    def _key(
        self,
    ) -> tuple[
        int | None,
        tuple[Step, ...],
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[str, ...],
        ObjectivePlan,
        tuple[str, ...],
        str | None,
        str | None,
        str,
        str | None,
        int | None,
        str | None,
        tuple[float, ...],
    ]:
        """Build the immutable content used for equality and hashing."""
        return (
            self._levels,
            self.steps,
            tuple(self._objects.items()),
            tuple(self._indices.items()),
            self._readouts,
            self._objective,
            self._groups,
            self._cutpoint_object,
            self._threshold_group,
            self._scalar_object,
            self._alternative_object,
            self._alternatives,
            self._contrast,
            self._contrast_weights,
        )

    @property
    def digest(self) -> str:
        """Return the content digest identifying this program in any process.

        The digest is over the reconstructing representation of the content
        equality compares, so equal programs share a digest and programs
        differing in any step or plan do not. Unlike the hash, it does not
        depend on the interpreter, so a stored digest identifies the program
        that wrote it.

        Returns
        -------
        str
            Hexadecimal SHA-256 of the program's content.
        """
        return sha256(repr(self._key()).encode("utf-8")).hexdigest()

    @override
    def __eq__(self, other: object) -> bool:
        """Compare compiled programs by content."""
        if type(other) is not ResponseProgram:
            return NotImplemented

        return self._key() == other._key()

    @override
    def __hash__(self) -> int:
        """Hash the compiled program's immutable content."""
        return hash((ResponseProgram, self._key()))

    @override
    def __repr__(self) -> str:
        """Return the level count and the host columns."""
        return (
            f"ResponseProgram(levels={self._levels}, "
            f"host_columns={self.host_columns!r})"
        )


@final
class BoundProgram:
    """A program bound to one design: its set sizes, host data and parameter layout.

    Parameters
    ----------
    program : ResponseProgram
        Program bound.
    sizes : Mapping[str, int]
        Number of elements of each finite set, in declaration order.
    host_data : Mapping[str, Tensor]
        Per-row tensor of each host column the program reads.
    """

    __slots__ = ("_host_data", "_layout", "_program", "_sizes")

    def __init__(
        self,
        program: ResponseProgram,
        *,
        sizes: Mapping[str, int],
        host_data: Mapping[str, Tensor],
    ) -> None:
        self._program = program
        self._sizes: Mapping[str, int] = MappingProxyType(dict(sizes))
        self._host_data: Mapping[str, Tensor] = MappingProxyType(dict(host_data))
        layout = []

        for draw in program.draws:
            indexed = math.prod(self._sizes[name] for name in draw.index)
            shape = (
                ()
                if not draw.event_shape
                else (
                    draw.event_shape
                    if draw.index == (self._program.scalar_object,)
                    else (indexed, *draw.event_shape)
                )
            )
            size = indexed if not shape else math.prod(shape)
            layout.append(Slot(draw.site, size, shape))

        self._layout = tuple(layout)

    @property
    def program(self) -> ResponseProgram:
        """Return the program bound."""
        return self._program

    @property
    def levels(self) -> int | None:
        """Return the ordered level count, or None for a continuous response."""
        return self._program.levels

    @property
    def rows(self) -> int:
        """Return the number of response rows."""
        return self._sizes[PLATE]

    @property
    def sizes(self) -> Mapping[str, int]:
        """Return the number of elements of each finite set."""
        return self._sizes

    @property
    def host_data(self) -> Mapping[str, Tensor]:
        """Return the per-row tensors the program reads, the response first."""
        return self._host_data

    @property
    def observation(self) -> Observation:
        """Return the observation the program ends with."""
        return self._program.observation

    def with_readouts(self, columns: Mapping[str, Tensor], /) -> BoundProgram:
        """Return the program bound to other values of some of its readouts.

        A readout column a torch module computed carries that module's
        graph, so a program bound to one is scored at the values the module
        gives its rows and differentiates back into the module.

        Parameters
        ----------
        columns : Mapping[str, Tensor]
            Value of each readout column to replace, one axis as long as
            the column it replaces.

        Returns
        -------
        BoundProgram
            The program over the same sizes, the same response and index
            columns, and the given values of the named readouts.

        Raises
        ------
        ValueError
            If a name is not a readout the program reads.
        DesignError
            If a column is not one axis as long as the column it replaces,
            or holds a nonfinite value.
        """
        readouts = self._program.readouts
        host = dict(self._host_data)

        for name, column in columns.items():
            if name not in readouts:
                msg = (
                    f"a program reading the readouts {readouts!r} was given a "
                    f"column for {name!r}"
                )

                raise ValueError(msg)

            _ensure_readout(name, column, host[name].numel())
            host[name] = column

        return BoundProgram(self._program, sizes=self._sizes, host_data=host)

    @property
    def layout(self) -> tuple[Slot, ...]:
        """Return each drawn site's size, in draw order.

        A site indexed by a product of sets is laid out flat, its first set
        varying slowest, which is how a table reads it back.
        """
        return self._layout

    @property
    def size(self) -> int:
        """Return the length of the flat parameter vector."""
        return sum(slot.size for slot in self._layout)

    def sites(self, params: Tensor, /) -> dict[str, Tensor]:
        """Return the value of each drawn site held in a flat parameter vector.

        Parameters
        ----------
        params : Tensor
            One floating-point value per element of every drawn site, in
            the layout's order.

        Raises
        ------
        TypeError
            If the vector holds values of a nonfloating kind.
        ValueError
            If the vector is not one axis of the layout's length.
        """
        if not params.is_floating_point():
            msg = (
                f"a parameter vector holds {params.dtype}, which is not a "
                f"floating-point kind"
            )

            raise TypeError(msg)

        layout = self._layout

        if params.ndim != 1 or params.numel() != self.size:
            msg = (
                f"a program over {self.size} parameters was given a vector of shape "
                f"{tuple(params.shape)}"
            )

            raise ValueError(msg)

        pieces = torch.split(params, [slot.size for slot in layout])

        return {
            slot.site: piece if not slot.shape else piece.reshape(slot.shape)
            for slot, piece in zip(layout, pieces, strict=True)
        }

    @override
    def __repr__(self) -> str:
        """Return the row count, the level count and the host columns."""
        return (
            f"BoundProgram(rows={self.rows}, levels={self.levels}, "
            f"host_columns={self._program.host_columns!r})"
        )


@final
class ResponseCompiler:
    """The compiler from response specifications to programs.

    Parameters
    ----------
    linkings : Mapping[str, LinkingCompiler]
        The program form of each linking family a link term may declare, by
        family name.
    """

    __slots__ = ("_linkings",)

    def __init__(self, linkings: Mapping[str, LinkingCompiler]) -> None:
        self._linkings: Mapping[str, LinkingCompiler] = MappingProxyType(dict(linkings))

    @classmethod
    def installed(cls) -> ResponseCompiler:
        """Return the compiler of every linking family installed in the environment.

        Each family is listed in the ``lofi.linking`` entry-point group under the
        family name a link term declares, naming the program form it compiles to.

        Returns
        -------
        ResponseCompiler
            A compiler holding every listed family, in family-name order.
        """
        linkings = installed_linkings()

        return cls({family: linkings.get(family) for family in linkings.available()})

    @property
    def families(self) -> tuple[str, ...]:
        """Return the linking families this compiler compiles."""
        return tuple(self._linkings)

    def _link(self, draft: _Draft, term: LinkTerm, prefix: str) -> Expression:
        """Append a link term's draws and return its expression.

        Parameters
        ----------
        draft : _Draft
            Program body being built.
        term : LinkTerm
            Link term to compile.
        prefix : str
            Site prefix of the term's parameter draws.

        Raises
        ------
        ValueError
            If the term declares a linking family this compiler does not
            compile.
        """
        compile_linking = self._linkings.get(term.linking.family)

        if compile_linking is None:
            msg = (
                f"the link term over {term.readout!r} declares the linking family "
                f"{term.linking.family!r}, and this compiler compiles "
                f"{self.families!r}"
            )

            raise ValueError(msg)

        steps = compile_linking(
            term,
            prefix,
            unit_slope=draft.unit_slope,
            signed_slope=draft.signed_slope,
        )

        for site, prior in steps.draws.items():
            draft.sample(site, (draft.unit,), prior)

        draft.readouts.extend(term.readouts)

        return _bindable(steps.expression)

    def _effect(
        self,
        draft: _Draft,
        effect: _Effect,
        *,
        prefix: str,
        centered: bool,
        varying: bool = False,
    ) -> None:
        """Append the steps of one effect and its term site.

        Parameters
        ----------
        draft : _Draft
            Program body being built.
        effect : _Effect
            Name, factors and link terms of the effect.
        prefix : str
            Prefix separating the sites of this effect's formula from those
            of every other formula of the program.
        centered : bool
            Whether the coefficients are drawn directly, rather than as a
            scale times standardized offsets.
        varying : bool
            Whether the effect varies a fixed effect the program has bound,
            so that each link term's site is read rather than compiled again.

        Raises
        ------
        ValueError
            If a varying effect reads a link term's site the program has not
            bound.
        """
        name = f"{prefix}{effect.name}"
        factors, links = effect.factors, effect.links
        parts: list[Expression] = []

        if factors:
            gathered = draft.index(factors)
            table = _object_of(factors)

            if centered:
                draft.sample(f"{name}_effects", (table,), DEFAULT_COEFFICIENT_PRIOR)
                parts.append(Gather(f"{name}_effects", gathered))
            else:
                draft.sample(f"{name}_scale", (draft.unit,), DEFAULT_SCALE_PRIOR)
                draft.sample(f"{name}_offsets", (table,), _STANDARD_NORMAL)
                parts.append(
                    Binary(
                        Operator.MUL,
                        Name(f"{name}_scale"),
                        Gather(f"{name}_offsets", gathered),
                    )
                )

        alone = not factors and len(links) == 1

        for link in links:
            site = f"{prefix}{link.readout}_term"

            if varying:
                if site not in draft.bound:
                    msg = (
                        f"the random slope {name!r} varies the link term over "
                        f"{link.readout!r}, which the predictor must declare "
                        f"before it"
                    )

                    raise ValueError(msg)

                parts.append(Name(site))

                continue

            prefix = name if alone else f"{name}_{link.readout}"
            expression = self._link(draft, link, prefix)

            if alone:
                parts.append(expression)
            else:
                draft.let(f"{prefix}_linked", expression)
                parts.append(Name(f"{prefix}_linked"))

        draft.let(f"{name}_term", product(parts))
        draft.terms.append(f"{name}_term")

    def _correlated_slope(
        self,
        draft: _Draft,
        term: CorrelatedRandomSlope,
        *,
        prefix: str,
    ) -> None:
        """Append one jointly varying intercept-and-slope block."""
        inner = _fixed_effect(term.term)
        fixed_site = f"{prefix}{inner.name}_term"

        if fixed_site not in draft.bound:
            msg = (
                f"the correlated random slope over {inner.name!r} varies a fixed "
                "effect the predictor must declare before it"
            )

            raise ValueError(msg)

        name = f"{prefix}{term.group}_{inner.name}_correlated"
        gathered = draft.index((term.group,))
        table = _object_of((term.group,))
        scales = f"{name}_scales"
        correlation = f"{name}_correlation"
        offsets = f"{name}_offsets"
        coefficients = f"{name}_coefficients"
        basis_parts: list[Expression] = []
        fixed_prefix = f"{prefix}{inner.name}"
        alone = len(inner.links) == 1

        for link in inner.links:
            linking = self._linkings.get(link.linking.family)

            if linking is None:
                msg = (
                    f"the link term over {link.readout!r} declares the linking family "
                    f"{link.linking.family!r}, and this compiler compiles "
                    f"{self.families!r}"
                )

                raise ValueError(msg)

            link_prefix = fixed_prefix if alone else f"{fixed_prefix}_{link.readout}"
            basis = linking(
                link,
                link_prefix,
                unit_slope=True,
                signed_slope=draft.signed_slope,
            )
            missing = tuple(site for site in basis.draws if site not in draft.bound)

            if missing:
                msg = (
                    f"the correlated random slope over {inner.name!r} reuses the "
                    "fixed effect's linking parameters, and these are not bound: "
                    f"{missing!r}"
                )

                raise ValueError(msg)

            basis_parts.append(basis.expression)

        basis_expression = product(basis_parts)

        draft.sample(
            scales,
            (draft.unit,),
            DEFAULT_SCALE_PRIOR,
            event_shape=(2,),
        )
        draft.sample(
            correlation,
            (draft.unit,),
            Prior("LKJCholesky", term.concentration),
            event_shape=(2, 2),
        )
        draft.sample(
            offsets,
            (table,),
            _STANDARD_NORMAL,
            event_shape=(2,),
        )
        draft.let(
            coefficients,
            Binary(
                Operator.MUL,
                TriangularTransform(Name(correlation), Name(offsets)),
                Name(scales),
            ),
        )
        row = Gather(coefficients, gathered)
        draft.let(
            f"{name}_term",
            Binary(
                Operator.ADD,
                Coordinate(row, 0),
                Binary(
                    Operator.MUL,
                    Coordinate(row, 1),
                    basis_expression,
                ),
            ),
        )
        draft.terms.append(f"{name}_term")

    def _term(self, draft: _Draft, term: Predictor, *, prefix: str) -> None:
        """Append the steps of one predictor term.

        Parameters
        ----------
        draft : _Draft
            Program body being built.
        term : Predictor
            Term to compile.
        prefix : str
            Prefix separating the sites of this term's formula from those of
            every other formula of the program.
        """
        match term:
            case OffsetTerm():
                draft.readouts.append(term.readout)
                draft.let(f"{prefix}{term.readout}_term", _bindable(Name(term.readout)))
                draft.terms.append(f"{prefix}{term.readout}_term")
            case RandomIntercept():
                self._effect(
                    draft,
                    _Effect(f"{term.group}_intercept", (term.group,), ()),
                    prefix=prefix,
                    centered=False,
                )
            case CorrelatedRandomSlope():
                self._correlated_slope(draft, term, prefix=prefix)
            case RandomSlope():
                inner = _fixed_effect(term.term)
                self._effect(
                    draft,
                    _Effect(
                        f"{term.group}_{inner.name}_slope",
                        _distinct((term.group, *inner.factors)),
                        inner.links,
                    ),
                    prefix=prefix,
                    centered=False,
                    varying=True,
                )
            case _:
                self._effect(draft, _fixed_effect(term), prefix=prefix, centered=True)

    def _formula(
        self,
        draft: _Draft,
        terms: tuple[Predictor, ...],
        *,
        site: str,
        prefix: str,
    ) -> None:
        """Append one formula's steps and bind the sum of its terms.

        Parameters
        ----------
        draft : _Draft
            Program body being built.
        terms : tuple[Predictor, ...]
            Terms whose sum the site holds.
        site : str
            Site the sum is bound to.
        prefix : str
            Prefix separating this formula's sites from those of every other
            formula of the program.

        Raises
        ------
        ValueError
            If a term is one this compiler cannot compile.
        """
        opened = len(draft.terms)

        for term in terms:
            self._term(draft, term, prefix=prefix)

        draft.let(site, total(Name(name) for name in draft.terms[opened:]))

    def _scales(self, spec: ResponseDeclaration, /) -> _Scales:
        """Return the scales the declared identification fixes.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification to read.

        Raises
        ------
        TypeError
            If the identification has no response-program form.
        """
        identification = spec.identification
        drawn = (
            None
            if spec.priors.discrimination is None
            else _prior_of(spec.priors.discrimination)
        )
        unit, signed = identification.unit_slope, identification.signed_slope

        match identification:
            case IdentityLinkOnLogit():
                return _Scales(drawn, None, unit_slope=unit, signed_slope=signed)
            case FixedDiscrimination(value=value):
                return _Scales(
                    None if value == 1.0 else value,
                    None,
                    unit_slope=unit,
                    signed_slope=signed,
                )
            case FixedCutpointVariance(value=value):
                predicted = (
                    PredictedParameter(
                        predictor_site(AuxiliaryParameter.DISCRIMINATION)
                    )
                    if AuxiliaryParameter.DISCRIMINATION in spec.predicted
                    else None
                )

                return _Scales(
                    drawn,
                    value,
                    unit_slope=unit,
                    signed_slope=signed,
                    predicted=predicted,
                )
            case unsupported:
                msg = (
                    "the scale identification "
                    f"{unsupported!r} has no response-program form"
                )

                raise TypeError(msg)

    def _predictor(
        self, spec: ResponseDeclaration, scales: _Scales, unit: str
    ) -> _Draft:
        """Return the body of every formula, each sum bound as it is reached.

        The latent response comes first, and then the formula of each
        auxiliary parameter that carries one, whose sites are prefixed by
        the parameter they predict so that two formulas over one factor do
        not name the same coefficient.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification whose terms are compiled.
        scales : _Scales
            Scales the declared identification fixes.
        unit : str
            Name of the one-element set a scalar draw is indexed by.

        Raises
        ------
        ValueError
            If the specification declares a term this compiler cannot compile.
        """
        draft = _Draft(
            unit,
            unit_slope=scales.unit_slope,
            signed_slope=scales.signed_slope,
        )

        self._formula(draft, spec.predictor, site=PREDICTOR, prefix="")

        for formula in spec.parameters:
            self._formula(
                draft,
                formula.predictor,
                site=predictor_site(formula.parameter),
                prefix=f"{formula.parameter.value}_",
            )

        return draft

    def _assembled(
        self,
        spec: ResponseDeclaration,
        draft: _Draft,
        observed: tuple[Step, ...],
        shape: _Shape,
    ) -> ResponseProgram:
        """Return the compiled program of a predictor and its observation.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification compiled.
        draft : _Draft
            Body of the linear predictor.
        observed : tuple[Step, ...]
            Steps the response family adds, its observation last.
        shape : _Shape
            Names and counts of the sets the response family indexes by.

        Raises
        ------
        ValueError
            If two names of the program coincide or a name is not an
            identifier.
        """
        readouts = _distinct(tuple(draft.readouts))
        objects = _Objects(tuple(dict.fromkeys((PLATE, *shape.reserved))))

        for factors in draft.indices.values():
            for factor in factors:
                objects.declare((factor,))

            if len(factors) > 1:
                objects.declare(factors)

        _ensure_distinct(
            "a name",
            (
                UNIT_RESPONSE if isinstance(spec, BoundedResponseSpec) else RESPONSE,
                *draft.indices,
                *readouts,
                *draft.bound,
                *shape.bound,
            ),
        )

        return ResponseProgram(
            levels=None if isinstance(spec, BoundedResponseSpec) else spec.levels,
            steps=(*draft.steps, *observed),
            objects=objects.declared,
            indices=draft.indices,
            readouts=readouts,
            objective=_objective_plan(spec),
            groups=spec.groups,
            cutpoint_object=shape.cutpoint_object,
            threshold_group=shape.threshold_group,
            scalar_object=shape.scalar_object,
            alternative_object=shape.alternative_object,
            alternatives=shape.alternatives,
            contrast=shape.contrast,
            contrast_weights=shape.contrast_weights,
        )

    def compile(self, spec: ResponseDeclaration, /) -> ResponseProgram:
        """Compile a response specification to a program.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification to compile.

        Raises
        ------
        TypeError
            If the specification declares a response family that compiles to
            no program.
        ValueError
            If the specification declares a term this compiler cannot
            compile, or a linking family it does not hold; or if two names of
            the program coincide or a name is not an identifier.
        """
        if isinstance(spec, BoundedResponseSpec):
            return self._bounded(spec, spec.family)

        family = spec.family

        if isinstance(family, Choice):
            return self._choice(spec, family)

        if not isinstance(family, Cumulative):
            msg = (
                f"the {family.family!r} family compiles to no response program; "
                f"it is fitted by its own head"
            )

            raise TypeError(msg)

        levels = spec.levels
        thresholds = cast("ThresholdPriors", spec.priors.thresholds)
        scales = self._scales(spec)
        discrimination = (
            scales.discrimination if scales.predicted is None else scales.predicted
        )
        cutpoint_variance = scales.cutpoint_variance
        group = None

        if any(
            term.kind == "random_intercept" and PARTICIPANT in term.groups
            for term in spec.predictor
        ):
            group = ThresholdGroup(PARTICIPANT, _object_of((PARTICIPANT,)))

        discrimination_group = None

        if family.discrimination_group is not None:
            discrimination_group = DiscriminationGroup(
                family.discrimination_group,
                _object_of((family.discrimination_group,)),
            )

        response_family = CumulativeFamily(
            levels,
            base=_prior_of(thresholds.base),
            increment=_prior_of(thresholds.increment),
            discrimination=discrimination,
            group=group,
            discrimination_group=discrimination_group,
            cutpoint_variance=cutpoint_variance,
        )
        draft = self._predictor(spec, scales, response_family.scalar_object)

        if discrimination_group is not None:
            draft.index((discrimination_group.index,))

        return self._assembled(
            spec,
            draft,
            response_family.steps(predictor=PREDICTOR, response=RESPONSE, plate=PLATE),
            _Shape(
                reserved=(
                    response_family.index_object,
                    response_family.scalar_object,
                ),
                bound=response_family.bound_names,
                cutpoint_object=response_family.index_object,
                threshold_group=None if group is None else group.index,
                scalar_object=response_family.scalar_object,
                alternative_object=None,
                alternatives=None,
                contrast=None,
                contrast_weights=(),
            ),
        )

    def _bounded(
        self, spec: BoundedResponseSpec, family: Bounded, /
    ) -> ResponseProgram:
        """Compile a bounded truncated-normal specification to a program.

        Parameters
        ----------
        spec : BoundedResponseSpec
            Specification to compile.
        family : Bounded
            Response family it declares.

        """
        scale = (
            _prior_of(cast("PriorSpec", spec.priors.scale))
            if family.scale is None
            else family.scale
        )

        scales = self._scales(spec)
        response_family = TruncatedNormalFamily(scale)
        draft = self._predictor(spec, scales, response_family.scalar_object)

        return self._assembled(
            spec,
            draft,
            response_family.steps(
                predictor=PREDICTOR,
                response=UNIT_RESPONSE,
                plate=PLATE,
            ),
            _Shape(
                reserved=(response_family.scalar_object,),
                bound=response_family.bound_names,
                cutpoint_object=None,
                threshold_group=None,
                scalar_object=response_family.scalar_object,
                alternative_object=None,
                alternatives=None,
                contrast=None,
                contrast_weights=(),
            ),
        )

    def _choice(self, spec: ResponseSpec, family: Choice, /) -> ResponseProgram:
        """Compile a forced-choice specification to a program.

        Parameters
        ----------
        spec : ResponseSpec
            Specification to compile.
        family : Choice
            Response family it declares.

        Raises
        ------
        ValueError
            If the specification declares a term this compiler cannot
            compile.
        """
        thresholds = spec.priors.thresholds
        scales = self._scales(spec)
        levels = spec.levels
        alternatives = family.alternatives(levels)
        discrimination_group = None

        if family.discrimination_group is not None:
            discrimination_group = DiscriminationGroup(
                family.discrimination_group,
                _object_of((family.discrimination_group,)),
            )

        ordered = (
            None
            if thresholds is None
            else CumulativeFamily(
                levels,
                base=_prior_of(thresholds.base),
                increment=_prior_of(thresholds.increment),
                cutpoint_variance=scales.cutpoint_variance,
            )
        )
        response_family = ChoiceFamily(
            alternatives,
            discrimination=scales.discrimination,
            discrimination_group=discrimination_group,
            ordered=ordered,
        )
        draft = self._predictor(spec, scales, response_family.scalar_object)

        if discrimination_group is not None:
            draft.index((discrimination_group.index,))

        return self._assembled(
            spec,
            draft,
            response_family.steps(predictor=PREDICTOR, response=RESPONSE, plate=PLATE),
            _Shape(
                reserved=(
                    *(
                        ()
                        if response_family.cutpoint_object is None
                        else (response_family.cutpoint_object,)
                    ),
                    response_family.scalar_object,
                    response_family.index_object,
                ),
                bound=response_family.bound_names,
                cutpoint_object=response_family.cutpoint_object,
                threshold_group=None,
                scalar_object=response_family.scalar_object,
                alternative_object=response_family.index_object,
                alternatives=alternatives,
                contrast=response_family.contrast_column,
                contrast_weights=(
                    ()
                    if response_family.contrast_column is None
                    else response_family.contrast
                ),
            ),
        )

    def emit(
        self, spec: ResponseDeclaration, design: ResponseDesign, /
    ) -> BoundProgram:
        """Compile a response specification and bind it to a design.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification to compile.
        design : ResponseDesign
            Design supplying the row count and the level count of every
            grouping factor the predictor reads.

        Raises
        ------
        TypeError
            If the specification declares a response family other than the
            cumulative one.
        ValueError
            If the specification cannot be compiled, or the design cannot
            bind the program.
        """
        return self.compile(spec).over(design)


def _ensure_distinct(role: str, names: tuple[str, ...]) -> None:
    """Check that names are identifiers and none repeats.

    Parameters
    ----------
    role : str
        What the names are, used in error messages.
    names : tuple[str, ...]
        Names to check.

    Raises
    ------
    ValueError
        If a name is not an identifier or two names coincide.
    """
    seen: set[str] = set()

    for name in names:
        ensure_identifier(name, role)

        if name in seen:
            msg = f"{role} is bound twice in the program: {name!r}"

            raise ValueError(msg)

        seen.add(name)

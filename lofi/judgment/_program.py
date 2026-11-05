"""The response-program language: its steps, its expressions and their value.

A program draws sites from priors, binds sites to expressions, and ends with
one observation: an ordered logistic over a location and its cutpoints, or a
multinomial logit over a row of alternatives. :func:`evaluate` computes an
expression's value over bound sites and host columns, and is the one
interpretation of the arithmetic every backend uses.
"""

import enum
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, assert_never, final, override

import torch
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch import Tensor

__all__ = [
    "Apply",
    "Binary",
    "Bind",
    "Coordinate",
    "Draw",
    "Expression",
    "Function",
    "Gather",
    "LinkingSteps",
    "Literal",
    "Name",
    "Observation",
    "ObserveBounded",
    "ObserveChoice",
    "ObserveOrdered",
    "Operator",
    "Prior",
    "Step",
    "Table",
    "TriangularTransform",
    "ensure_identifier",
    "evaluate",
    "product",
    "total",
]


def ensure_identifier(text: str, role: str) -> str:
    """Return a name a program may bind or read.

    Parameters
    ----------
    text : str
        Candidate identifier.
    role : str
        Identifier role used in error messages.

    Returns
    -------
    str
        The validated identifier.

    Raises
    ------
    ValueError
        If ``text`` is not a valid identifier.
    """
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) is None:
        msg = f"{role} must be an identifier: {text!r}"

        raise ValueError(msg)

    return text


@final
class Prior:
    """A distribution a site is drawn from, by family name and parameters.

    Parameters
    ----------
    family : str
        Distribution family, such as ``Normal`` or ``HalfNormal``.
    *arguments : float
        The family's parameters in its conventional order: location then
        scale, a scale alone, a rate alone, or a lower then an upper bound.

    Raises
    ------
    TypeError
        If an argument is not a number.
    ValueError
        If ``family`` is not an identifier or an argument is not finite.
    """

    __slots__ = ("_arguments", "_family")

    def __init__(self, family: str, *arguments: float) -> None:
        for argument in arguments:
            if not math.isfinite(argument):
                msg = f"prior argument must be finite: {argument!r}"

                raise ValueError(msg)

        self._family = ensure_identifier(family, "a prior's distribution family")
        self._arguments = tuple(float(argument) for argument in arguments)

    @property
    def family(self) -> str:
        """Return the distribution family."""
        return self._family

    @property
    def arguments(self) -> tuple[float, ...]:
        """Return the family's parameters in order."""
        return self._arguments

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another prior has the same family and arguments."""
        if type(other) is not Prior:
            return NotImplemented

        return (self._family, self._arguments) == (other._family, other._arguments)

    @override
    def __hash__(self) -> int:
        """Return a hash over the family and the arguments."""
        return hash((Prior, self._family, self._arguments))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        arguments = "".join(f", {argument!r}" for argument in self._arguments)

        return f"{type(self).__name__}({self._family!r}{arguments})"


class Operator(enum.StrEnum):
    """A binary arithmetic operator, valued by its symbol."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


class Function(enum.StrEnum):
    """A unary function an expression applies to a tensor."""

    SOFTPLUS = "softplus"
    CUMSUM = "cumsum"
    EXP = "exp"
    LOG = "log"
    MEAN = "mean"
    SQRT = "sqrt"
    SQUARE = "square"
    SIGMOID = "sigmoid"


@final
class Name:
    """A reference to a bound site or a host column.

    Parameters
    ----------
    identifier : str
        Name referred to.

    Raises
    ------
    ValueError
        If the name is not an identifier.
    """

    __slots__ = ("_identifier",)

    def __init__(self, identifier: str) -> None:
        self._identifier = ensure_identifier(identifier, "a name")

    @property
    def identifier(self) -> str:
        """Return the name referred to."""
        return self._identifier

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to."""
        return frozenset((self._identifier,))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another name refers to the same identifier."""
        if type(other) is not Name:
            return NotImplemented

        return self._identifier == other._identifier

    @override
    def __hash__(self) -> int:
        """Return a hash over the identifier."""
        return hash((Name, self._identifier))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Name({self._identifier!r})"


@final
class Literal:
    """A finite number written into the program.

    Parameters
    ----------
    value : float
        Number written.

    Raises
    ------
    ValueError
        If the number is not finite.
    """

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        if not math.isfinite(value):
            msg = f"a literal must be finite: {value!r}"

            raise ValueError(msg)

        self._value = float(value)

    @property
    def value(self) -> float:
        """Return the number written."""
        return self._value

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to, which are none."""
        return frozenset()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another literal writes the same number."""
        if type(other) is not Literal:
            return NotImplemented

        return self._value == other._value

    @override
    def __hash__(self) -> int:
        """Return a hash over the number."""
        return hash((Literal, self._value))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Literal({self._value!r})"


@final
class Gather:
    """The rows of a bound table selected by a host index column.

    Parameters
    ----------
    table : str
        Site holding one entry, or one row of entries, per level.
    index : str
        Host column holding each row's level.

    Raises
    ------
    ValueError
        If either name is not an identifier.
    """

    __slots__ = ("_index", "_table")

    def __init__(self, table: str, index: str) -> None:
        self._table = ensure_identifier(table, "a gathered table")
        self._index = ensure_identifier(index, "an index column")

    @property
    def table(self) -> str:
        """Return the site gathered from."""
        return self._table

    @property
    def index(self) -> str:
        """Return the host column gathered by."""
        return self._index

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to."""
        return frozenset((self._table, self._index))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another gather reads the same table by the same column."""
        if type(other) is not Gather:
            return NotImplemented

        return (self._table, self._index) == (other._table, other._index)

    @override
    def __hash__(self) -> int:
        """Return a hash over the table and the column."""
        return hash((Gather, self._table, self._index))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Gather({self._table!r}, {self._index!r})"


@final
class Table:
    """A flat site over a product of two sets, read as one row per level.

    Parameters
    ----------
    source : str
        Site holding one entry per element of the product, its first
        factor varying slowest.
    rows : str
        Finite set the rows index.
    columns : str
        Finite set the columns index.
    width : int
        Number of columns.

    Raises
    ------
    ValueError
        If a name is not an identifier or the width is not positive.
    """

    __slots__ = ("_columns", "_rows", "_source", "_width")

    def __init__(self, source: str, rows: str, columns: str, *, width: int) -> None:
        if width < 1:
            msg = f"a table has at least one column: {width!r}"

            raise ValueError(msg)

        self._source = ensure_identifier(source, "a table's source")
        self._rows = ensure_identifier(rows, "a table's row set")
        self._columns = ensure_identifier(columns, "a table's column set")
        self._width = width

    @property
    def source(self) -> str:
        """Return the flat site the table reads."""
        return self._source

    @property
    def rows(self) -> str:
        """Return the finite set the rows index."""
        return self._rows

    @property
    def columns(self) -> str:
        """Return the finite set the columns index."""
        return self._columns

    @property
    def width(self) -> int:
        """Return the number of columns."""
        return self._width

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to."""
        return frozenset((self._source,))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another table reads the same source at the same shape."""
        if type(other) is not Table:
            return NotImplemented

        return (self._source, self._rows, self._columns, self._width) == (
            other._source,
            other._rows,
            other._columns,
            other._width,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the source and the shape."""
        return hash((Table, self._source, self._rows, self._columns, self._width))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return (
            f"Table({self._source!r}, {self._rows!r}, {self._columns!r}, "
            f"width={self._width!r})"
        )


@final
class Apply:
    """A unary function applied to an expression.

    Parameters
    ----------
    function : Function
        Function applied.
    argument : Expression
        Expression applied to.
    """

    __slots__ = ("_argument", "_function")

    def __init__(self, function: Function, argument: Expression) -> None:
        self._function = function
        self._argument = argument

    @property
    def function(self) -> Function:
        """Return the function applied."""
        return self._function

    @property
    def argument(self) -> Expression:
        """Return the expression applied to."""
        return self._argument

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to."""
        return self._argument.names

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another application is the same call."""
        if type(other) is not Apply:
            return NotImplemented

        return (self._function, self._argument) == (other._function, other._argument)

    @override
    def __hash__(self) -> int:
        """Return a hash over the function and the argument."""
        return hash((Apply, self._function, self._argument))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Apply({self._function!r}, {self._argument!r})"


@final
class Binary:
    """Two expressions combined by an arithmetic operator.

    Parameters
    ----------
    operator : Operator
        Operator applied.
    left : Expression
        Left operand.
    right : Expression
        Right operand.
    """

    __slots__ = ("_left", "_operator", "_right")

    def __init__(self, operator: Operator, left: Expression, right: Expression) -> None:
        self._operator = operator
        self._left = left
        self._right = right

    @property
    def operator(self) -> Operator:
        """Return the operator applied."""
        return self._operator

    @property
    def left(self) -> Expression:
        """Return the left operand."""
        return self._left

    @property
    def right(self) -> Expression:
        """Return the right operand."""
        return self._right

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to."""
        return self._left.names | self._right.names

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another combination has the same operator and operands."""
        if type(other) is not Binary:
            return NotImplemented

        return (self._operator, self._left, self._right) == (
            other._operator,
            other._left,
            other._right,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the operator and the operands."""
        return hash((Binary, self._operator, self._left, self._right))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Binary({self._operator!r}, {self._left!r}, {self._right!r})"


@final
class Coordinate:
    """One trailing coordinate selected from an expression.

    Parameters
    ----------
    argument : Expression
        Expression whose final axis is selected.
    index : int
        Zero-based coordinate on the final axis.

    Raises
    ------
    TypeError
        If the coordinate is not an integer.
    ValueError
        If the coordinate is negative.
    """

    __slots__ = ("_argument", "_index")

    def __init__(self, argument: Expression, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            msg = f"an expression coordinate is an integer: {index!r}"

            raise TypeError(msg)

        if index < 0:
            msg = f"an expression coordinate is nonnegative: {index!r}"

            raise ValueError(msg)

        self._argument = argument
        self._index = index

    @property
    def argument(self) -> Expression:
        """Return the expression selected from."""
        return self._argument

    @property
    def index(self) -> int:
        """Return the selected coordinate."""
        return self._index

    @property
    def names(self) -> frozenset[str]:
        """Return the names the expression refers to."""
        return self._argument.names

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another selection has the same argument and index."""
        if type(other) is not Coordinate:
            return NotImplemented

        return (self._argument, self._index) == (other._argument, other._index)

    @override
    def __hash__(self) -> int:
        """Return a hash over the argument and index."""
        return hash((Coordinate, self._argument, self._index))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Coordinate({self._argument!r}, {self._index!r})"


@final
class TriangularTransform:
    """Row vectors transformed by a lower-triangular matrix.

    The result is ``vectors @ factor.T``. This is the row-vector form of
    transforming standard-normal coefficient vectors by a Cholesky factor.

    Parameters
    ----------
    factor : Expression
        Lower-triangular matrix expression.
    vectors : Expression
        Matrix whose rows are coefficient vectors.
    """

    __slots__ = ("_factor", "_vectors")

    def __init__(self, factor: Expression, vectors: Expression) -> None:
        self._factor = factor
        self._vectors = vectors

    @property
    def factor(self) -> Expression:
        """Return the lower-triangular factor expression."""
        return self._factor

    @property
    def vectors(self) -> Expression:
        """Return the row-vector expression."""
        return self._vectors

    @property
    def names(self) -> frozenset[str]:
        """Return the names both expressions refer to."""
        return self._factor.names | self._vectors.names

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another transform has the same factor and vectors."""
        if type(other) is not TriangularTransform:
            return NotImplemented

        return (self._factor, self._vectors) == (other._factor, other._vectors)

    @override
    def __hash__(self) -> int:
        """Return a hash over the factor and vectors."""
        return hash((TriangularTransform, self._factor, self._vectors))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"TriangularTransform({self._factor!r}, {self._vectors!r})"


type Expression = (
    Name | Literal | Gather | Table | Apply | Binary | Coordinate | TriangularTransform
)
"""A tree evaluating to one tensor."""


def total(operands: Iterable[Expression]) -> Expression:
    """Return the sum of expressions, associated to the left.

    Parameters
    ----------
    operands : Iterable[Expression]
        Expressions to add, at least one.

    Raises
    ------
    ValueError
        If no expression is given.
    """
    return _fold(Operator.ADD, tuple(operands))


def product(operands: Iterable[Expression]) -> Expression:
    """Return the product of expressions, associated to the left.

    Parameters
    ----------
    operands : Iterable[Expression]
        Expressions to multiply, at least one.

    Raises
    ------
    ValueError
        If no expression is given.
    """
    return _fold(Operator.MUL, tuple(operands))


def _fold(operator: Operator, operands: tuple[Expression, ...]) -> Expression:
    """Return operands combined by an operator, associated to the left.

    Parameters
    ----------
    operator : Operator
        Operator applied between adjacent operands.
    operands : tuple[Expression, ...]
        Expressions to combine, at least one.

    Raises
    ------
    ValueError
        If no expression is given.
    """
    if not operands:
        msg = f"combining by {operator.value!r} needs at least one expression"

        raise ValueError(msg)

    combined = operands[0]

    for operand in operands[1:]:
        combined = Binary(operator, combined, operand)

    return combined


def evaluate(
    expression: Expression,
    environment: Mapping[str, Tensor],
    /,
    *,
    like: Tensor,
) -> Tensor:
    """Return an expression's value over bound sites and host columns.

    A gather indexes the leading axis of its table by the index column, a
    table reshapes its flat source to one row per level, and a function
    of a row reduces or accumulates along the last axis. Operands combine
    under broadcasting.

    Parameters
    ----------
    expression : Expression
        Expression to evaluate.
    environment : Mapping[str, Tensor]
        Value of every name the expression refers to.
    like : Tensor
        Tensor whose floating-point kind and device a literal is written at.

    Raises
    ------
    KeyError
        If the expression refers to a name the environment does not bind.
    """
    match expression:
        case Name(identifier=identifier):
            result = environment[identifier]
        case Literal(value=value):
            result = like.new_tensor(value)
        case Gather(table=table, index=index):
            result = environment[table][environment[index]]
        case Table(source=source, width=width):
            result = environment[source].reshape(-1, width)
        case Apply(function=function, argument=argument):
            result = _apply(function, evaluate(argument, environment, like=like))
        case Binary(operator=operator, left=left, right=right):
            result = _combine(
                operator,
                evaluate(left, environment, like=like),
                evaluate(right, environment, like=like),
            )
        case Coordinate(argument=argument, index=index):
            result = evaluate(argument, environment, like=like)[..., index]
        case TriangularTransform(factor=factor, vectors=vectors):
            matrix = evaluate(factor, environment, like=like)
            rows = evaluate(vectors, environment, like=like)
            dtype = torch.promote_types(rows.dtype, matrix.dtype)

            result = torch.matmul(
                rows.to(dtype=dtype), matrix.to(dtype=dtype).transpose(-1, -2)
            )

    return result


def _apply(function: Function, argument: Tensor) -> Tensor:
    """Return a unary function's value.

    Parameters
    ----------
    function : Function
        Function applied.
    argument : Tensor
        Value applied to.
    """
    match function:
        case Function.SOFTPLUS:
            result = functional.softplus(argument)
        case Function.CUMSUM:
            result = torch.cumsum(argument, dim=-1)
        case Function.EXP:
            result = torch.exp(argument)
        case Function.LOG:
            result = torch.log(argument)
        case Function.MEAN:
            result = torch.mean(argument, dim=-1)
        case Function.SQRT:
            result = torch.sqrt(argument)
        case Function.SQUARE:
            result = torch.square(argument)
        case Function.SIGMOID:
            result = torch.sigmoid(argument)
        case _ as unreachable:
            assert_never(unreachable)

    return result


def _combine(operator: Operator, left: Tensor, right: Tensor) -> Tensor:
    """Return two values combined by an arithmetic operator.

    Parameters
    ----------
    operator : Operator
        Operator applied.
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.
    """
    match operator:
        case Operator.ADD:
            return left + right
        case Operator.SUB:
            return left - right
        case Operator.MUL:
            return left * right
        case Operator.DIV:
            return left / right


@final
class Draw:
    """A site drawn from a prior, indexed by a product of finite sets.

    Parameters
    ----------
    site : str
        Site bound.
    index : tuple[str, ...]
        Finite sets whose product indexes the draw, at least one.
    prior : Prior
        Distribution drawn from.
    event_shape : tuple[int, ...]
        Shape of one jointly carried value. Empty for a scalar draw.

    Raises
    ------
    TypeError
        If an event axis is not an integer.
    ValueError
        If a name is not an identifier or no set is given.
    """

    __slots__ = ("_event_shape", "_index", "_prior", "_site")

    def __init__(
        self,
        site: str,
        index: tuple[str, ...],
        prior: Prior,
        *,
        event_shape: tuple[int, ...] = (),
    ) -> None:
        if not index:
            msg = f"the draw of {site!r} is indexed by at least one finite set"

            raise ValueError(msg)

        self._site = ensure_identifier(site, "a drawn site")
        self._index = tuple(ensure_identifier(name, "an index set") for name in index)
        self._prior = prior

        if any(
            not isinstance(size, int) or isinstance(size, bool) for size in event_shape
        ):
            msg = f"a draw's event axes are integers: {event_shape!r}"

            raise TypeError(msg)

        if any(size < 1 for size in event_shape):
            msg = f"a draw's event axes are positive: {event_shape!r}"

            raise ValueError(msg)

        self._event_shape = tuple(event_shape)

    @property
    def site(self) -> str:
        """Return the site bound."""
        return self._site

    @property
    def index(self) -> tuple[str, ...]:
        """Return the finite sets whose product indexes the draw."""
        return self._index

    @property
    def prior(self) -> Prior:
        """Return the distribution drawn from."""
        return self._prior

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Return the shape of one jointly carried value."""
        return self._event_shape

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another draw binds the same site the same way."""
        if type(other) is not Draw:
            return NotImplemented

        return (self._site, self._index, self._prior, self._event_shape) == (
            other._site,
            other._index,
            other._prior,
            other._event_shape,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the site, the index and the prior."""
        return hash((Draw, self._site, self._index, self._prior, self._event_shape))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        event = "" if not self._event_shape else f", event_shape={self._event_shape!r}"

        return f"Draw({self._site!r}, {self._index!r}, {self._prior!r}{event})"


@final
class Bind:
    """A site bound to an expression.

    Parameters
    ----------
    site : str
        Site bound.
    expression : Expression
        Expression the site holds.

    Raises
    ------
    ValueError
        If the site is not an identifier.
    """

    __slots__ = ("_expression", "_site")

    def __init__(self, site: str, expression: Expression) -> None:
        self._site = ensure_identifier(site, "a bound site")
        self._expression = expression

    @property
    def site(self) -> str:
        """Return the site bound."""
        return self._site

    @property
    def expression(self) -> Expression:
        """Return the expression the site holds."""
        return self._expression

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another binding binds the same site to the same expression."""
        if type(other) is not Bind:
            return NotImplemented

        return (self._site, self._expression) == (other._site, other._expression)

    @override
    def __hash__(self) -> int:
        """Return a hash over the site and the expression."""
        return hash((Bind, self._site, self._expression))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"Bind({self._site!r}, {self._expression!r})"


@final
class ObserveBounded:
    """The observation of each row under a unit-interval truncated normal.

    Parameters
    ----------
    response : str
        Host column holding each normalized floating response.
    plate : str
        Finite set indexing the rows.
    location : str
        Site holding each row's location in the open unit interval.
    scale : str
        Site holding the positive standard deviation.

    Raises
    ------
    ValueError
        If a name is not an identifier.
    """

    __slots__ = ("_location", "_plate", "_response", "_scale")

    def __init__(
        self,
        response: str,
        plate: str,
        *,
        location: str,
        scale: str,
    ) -> None:
        self._response = ensure_identifier(response, "a response")
        self._plate = ensure_identifier(plate, "a response plate")
        self._location = ensure_identifier(location, "a location site")
        self._scale = ensure_identifier(scale, "a scale site")

    @property
    def response(self) -> str:
        """Return the host column holding each normalized response."""
        return self._response

    @property
    def plate(self) -> str:
        """Return the finite set indexing the rows."""
        return self._plate

    @property
    def location(self) -> str:
        """Return the site holding each row's location."""
        return self._location

    @property
    def scale(self) -> str:
        """Return the site holding the positive standard deviation."""
        return self._scale

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another observation scores the same column the same way."""
        if type(other) is not ObserveBounded:
            return NotImplemented

        return (
            self._response,
            self._plate,
            self._location,
            self._scale,
        ) == (
            other._response,
            other._plate,
            other._location,
            other._scale,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the column, plate and parameter sites."""
        return hash(
            (
                ObserveBounded,
                self._response,
                self._plate,
                self._location,
                self._scale,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return (
            f"ObserveBounded({self._response!r}, {self._plate!r}, "
            f"location={self._location!r}, scale={self._scale!r})"
        )


@final
class ObserveOrdered:
    """The observation of each row's level under an ordered logistic.

    Parameters
    ----------
    response : str
        Host column holding each row's recorded level.
    plate : str
        Finite set indexing the rows.
    location : str
        Site holding each row's location on the latent scale.
    thresholds : str
        Site holding the cutpoints, one row shared by every response row or
        one row per response row.

    Raises
    ------
    ValueError
        If a name is not an identifier.
    """

    __slots__ = ("_location", "_plate", "_response", "_thresholds")

    def __init__(
        self,
        response: str,
        plate: str,
        *,
        location: str,
        thresholds: str,
    ) -> None:
        self._response = ensure_identifier(response, "a response")
        self._plate = ensure_identifier(plate, "a response plate")
        self._location = ensure_identifier(location, "a location site")
        self._thresholds = ensure_identifier(thresholds, "a thresholds site")

    @property
    def response(self) -> str:
        """Return the host column holding each row's recorded level."""
        return self._response

    @property
    def plate(self) -> str:
        """Return the finite set indexing the rows."""
        return self._plate

    @property
    def location(self) -> str:
        """Return the site holding each row's location."""
        return self._location

    @property
    def thresholds(self) -> str:
        """Return the site holding the cutpoints."""
        return self._thresholds

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another observation scores the same column the same way."""
        if type(other) is not ObserveOrdered:
            return NotImplemented

        return (self._response, self._plate, self._location, self._thresholds) == (
            other._response,
            other._plate,
            other._location,
            other._thresholds,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the column, the plate and the argument sites."""
        return hash(
            (
                ObserveOrdered,
                self._response,
                self._plate,
                self._location,
                self._thresholds,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return (
            f"ObserveOrdered({self._response!r}, {self._plate!r}, "
            f"location={self._location!r}, thresholds={self._thresholds!r})"
        )


@final
class ObserveChoice:
    """The observation of each row's chosen alternative under a multinomial logit.

    Parameters
    ----------
    response : str
        Host column holding the place of each row's chosen alternative in
        presentation order.
    plate : str
        Finite set indexing the rows.
    logits : str
        Site holding one row of scores per response row, one column per
        alternative in presentation order.

    Raises
    ------
    ValueError
        If a name is not an identifier.
    """

    __slots__ = ("_logits", "_plate", "_response")

    def __init__(self, response: str, plate: str, *, logits: str) -> None:
        self._response = ensure_identifier(response, "a response")
        self._plate = ensure_identifier(plate, "a response plate")
        self._logits = ensure_identifier(logits, "a logits site")

    @property
    def response(self) -> str:
        """Return the host column holding each row's chosen place."""
        return self._response

    @property
    def plate(self) -> str:
        """Return the finite set indexing the rows."""
        return self._plate

    @property
    def logits(self) -> str:
        """Return the site holding each row's scores over the alternatives."""
        return self._logits

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another observation scores the same column the same way."""
        if type(other) is not ObserveChoice:
            return NotImplemented

        return (self._response, self._plate, self._logits) == (
            other._response,
            other._plate,
            other._logits,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the column, the plate and the scores."""
        return hash((ObserveChoice, self._response, self._plate, self._logits))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return (
            f"ObserveChoice({self._response!r}, {self._plate!r}, "
            f"logits={self._logits!r})"
        )


type Observation = ObserveBounded | ObserveOrdered | ObserveChoice
"""The one statement of a program that scores the recorded response."""


type Step = Draw | Bind | Observation
"""One statement of a program body."""


@final
class LinkingSteps:
    """The program form of one linking: its parameter draws and its expression.

    The expression is written over the draw sites and the readout columns
    the linking reads, and evaluates to one value per row. A linking whose
    parameters are all pinned draws nothing.

    Parameters
    ----------
    draws : Mapping[str, Prior]
        Prior of each fitted parameter, by site name, in draw order.
    expression : Expression
        Expression of the linking's contribution to the latent scale.

    Raises
    ------
    ValueError
        If a site is not an identifier.
    """

    __slots__ = ("_draws", "_expression")

    def __init__(self, draws: Mapping[str, Prior], expression: Expression) -> None:
        self._draws: Mapping[str, Prior] = MappingProxyType(
            {
                ensure_identifier(site, "a linking parameter site"): prior
                for site, prior in draws.items()
            }
        )
        self._expression = expression

    @property
    def draws(self) -> Mapping[str, Prior]:
        """Return the prior of each fitted parameter, by site, in draw order."""
        return self._draws

    @property
    def expression(self) -> Expression:
        """Return the expression of the contribution."""
        return self._expression

    @override
    def __repr__(self) -> str:
        """Return the draws and the expression."""
        return f"LinkingSteps({dict(self._draws)!r}, {self._expression!r})"

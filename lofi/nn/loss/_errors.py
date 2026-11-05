"""Objective-layer exceptions."""

__all__ = [
    "EveryItemAbstainedError",
    "ImpossibleNegativeBracketError",
    "ImproperMassError",
    "InvalidCutpointError",
    "MissingOutputError",
    "ObjectiveError",
    "SparseDivergenceError",
    "StochasticFactoringError",
    "UncomputedCellError",
    "UndeclaredOutputError",
]


class ObjectiveError(Exception):
    """Base class for objective-layer errors."""


class UndeclaredOutputError(ObjectiveError):
    """Reports access to an output that an objective did not declare.

    Parameters
    ----------
    objective : str
        The name of the objective that reached for the field.
    field : str
        The field it reached for.
    declared : frozenset[str]
        The fields it declared.
    """

    __slots__ = ("declared", "field", "objective")

    def __init__(
        self,
        *,
        objective: str,
        field: str,
        declared: frozenset[str],
    ) -> None:
        self.objective = objective
        self.field = field
        self.declared = declared

        wanted = ", ".join(sorted(declared)) or "nothing"
        msg = f"{objective} reads {field}, having declared {wanted}"

        super().__init__(msg)


class MissingOutputError(ObjectiveError):
    """Reports a required output missing from the configured graph.

    Parameters
    ----------
    objective : str
        The name of the objective that needs the outputs.
    missing : frozenset[str]
        The fields it needs that nothing publishes.
    published : frozenset[str]
        The fields the graph does publish.
    """

    __slots__ = ("missing", "objective", "published")

    def __init__(
        self,
        *,
        objective: str,
        missing: frozenset[str],
        published: frozenset[str],
    ) -> None:
        self.objective = objective
        self.missing = missing
        self.published = published

        wanted = ", ".join(sorted(missing))
        offered = ", ".join(sorted(published)) or "nothing"
        msg = f"{objective} reads {wanted}, and the graph publishes {offered}"

        super().__init__(msg)


class InvalidCutpointError(ObjectiveError):
    """Reports a cutpoint outside its span.

    Parameters
    ----------
    item : int
        Which item of the batch was read.
    span : tuple[int, int]
        The span it was read from, as a start and an exclusive end.
    cut : int
        The cutpoint that was asked for.
    """

    __slots__ = ("cut", "item", "span")

    def __init__(self, *, item: int, span: tuple[int, int], cut: int) -> None:
        self.item = item
        self.span = span
        self.cut = cut

        start, end = span
        msg = f"item {item} has no cutpoint {cut} in the span [{start}, {end})"

        super().__init__(msg)


class UncomputedCellError(ObjectiveError):
    """Reports a read of a chart cell a readout never computed.

    Parameters
    ----------
    item : int
        Which item of the batch was read.
    span : tuple[int, int]
        The span that was read, as a start and an exclusive end.
    """

    __slots__ = ("item", "span")

    def __init__(self, *, item: int, span: tuple[int, int]) -> None:
        self.item = item
        self.span = span

        start, end = span
        msg = f"the readout never computed the span [{start}, {end}) of item {item}"

        super().__init__(msg)


class ImproperMassError(ObjectiveError):
    """Reports a published log probability above zero, which no mass can have.

    Parameters
    ----------
    readout : str
        The id the mass was published under.
    item : int
        Which item of the batch was read.
    span : tuple[int, int]
        The span that was read, as a start and an exclusive end.
    value : float
        The log probability that was published there.
    """

    __slots__ = ("item", "readout", "span", "value")

    def __init__(
        self, *, readout: str, item: int, span: tuple[int, int], value: float
    ) -> None:
        self.readout = readout
        self.item = item
        self.span = span
        self.value = value

        start, end = span
        msg = (
            f"the readout {readout!r} published the log probability {value} at the "
            f"span [{start}, {end}) of item {item}, and a mass is at most one"
        )

        super().__init__(msg)


class ImpossibleNegativeBracketError(ObjectiveError):
    """Reports a negative bracket on a span the parser always builds.

    Its constituent probability is exactly one, so excluding it has no mass and its
    negative log is infinite.

    Parameters
    ----------
    objective : str
        The declared name of the objective that read the bracket.
    item : int
        Which item of the batch was read.
    span : tuple[int, int]
        The bracketed span, as a start and an exclusive end.
    """

    __slots__ = ("item", "objective", "span")

    def __init__(self, *, objective: str, item: int, span: tuple[int, int]) -> None:
        self.objective = objective
        self.item = item
        self.span = span

        start, end = span
        msg = (
            f"{objective} excludes the span [{start}, {end}) of item {item}, which "
            "every derivation the parser assigns that item builds"
        )

        super().__init__(msg)


class EveryItemAbstainedError(ObjectiveError):
    """Reports a batch the prover reached no answer about at any item.

    A mean over nothing is not a zero, and a zero here would train the decoder
    against the prover's clock and budget rather than against its own samples.

    Parameters
    ----------
    objective : str
        The name of the objective that scored the batch.
    items : int
        How many items the batch held, every one of which abstained.
    """

    __slots__ = ("items", "objective")

    def __init__(self, *, objective: str, items: int) -> None:
        self.objective = objective
        self.items = items

        msg = (
            f"{objective} means over the items the prover decided, and all "
            f"{items} of this batch abstained"
        )

        super().__init__(msg)


class StochasticFactoringError(ObjectiveError):
    """Reports a decoder whose factoring draws a sample, which the exact path refuses.

    Parameters
    ----------
    role : str
        Which side of the comparison the decoder was handed as.
    decoder : str
        The name of the decoder's type.
    """

    __slots__ = ("decoder", "role")

    def __init__(self, *, role: str, decoder: str) -> None:
        self.role = role
        self.decoder = decoder

        msg = (
            f"the {role} decoder {decoder} draws a sample when it factors a state, "
            "so its state at a position is not a function of the position and the "
            "exact recursion cannot run over it"
        )

        super().__init__(msg)


class SparseDivergenceError(ObjectiveError):
    """Reports a divergence that is infinite under the mapping a readout publishes.

    A mapping that leaves symbols at exactly zero mass makes any comparison that
    reads the log of that mass infinite wherever the other side has mass there, which
    is common early in a run. The pair is refused where it is configured rather than
    floored where it is computed.

    Parameters
    ----------
    mapping : str
        The readout mapping that leaves symbols at exactly zero mass.
    divergence : str
        The comparison that reads the log of that mass.
    replacement : str
        The finite comparison built on the same entropy.
    """

    __slots__ = ("divergence", "mapping", "replacement")

    def __init__(self, *, mapping: str, divergence: str, replacement: str) -> None:
        self.mapping = mapping
        self.divergence = divergence
        self.replacement = replacement

        msg = (
            f"{mapping} leaves symbols at exactly zero mass, and {divergence} is "
            f"infinite wherever the other side has mass there. {replacement} is the "
            "finite comparison built on the same entropy"
        )

        super().__init__(msg)

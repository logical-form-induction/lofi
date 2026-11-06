"""Rungs comparing decomposed column levels of a declared table against memorized ones.

A rung scores every declared row against every declared column, and two
rungs differ in the declared lift alone: whether a column reaches the score
as a label with nothing shared between columns, as a joint tensor over the
primitive types its argument slots hold, or through a row-side latent that
the primitive types are read against. Every rung that decomposes a column
reads the same primitive-type map, so what separates those rungs is the
lift rather than the map.

Primitive types and lexical-semantic primitives are two inventories. An
argument slot holds exactly one primitive type, and a row possesses any
number of lexical-semantic primitives at once, which is why membership in a
row class is read as a distribution and possession is read as one
independent probability each.
"""

from typing import TYPE_CHECKING, Literal, cast, final, override

import torch
from torch import nn
from torch.optim import Optimizer

from lofi._checks import ensure_non_negative, ensure_positive
from lofi.judgment._spec import (
    ColumnInventory,
    ColumnLabels,
    JointTypeTensor,
    LexicalPossession,
    MemorizedCells,
    PossessionAndClasses,
    RowClasses,
    TableAblationSpec,
)
from lofi.registry import plugin_for

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._spec import Predictor
    from lofi.judgment.ordinal import OrdinalHead
    from lofi.spec import ConfiguredSpec

__all__ = [
    "PinnedMap",
    "PrimitiveTypeMap",
    "TableRung",
    "fit_rung",
    "ladder_of",
    "mean_log_density",
    "slot_types",
]

type _PinTarget = Literal["semantic_primitives", "item_clusters"]
type _PinMode = Literal["independent_sigmoid", "normalized_softmax"]


def _drawn(
    shape: tuple[int, ...], generator: torch.Generator, *, scale: float
) -> Tensor:
    """Return a scaled standard normal draw at a shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the draw.
    generator : torch.Generator
        Source of the draws, whose device the result is placed on.
    """
    drawn = torch.randn(
        shape, generator=generator, dtype=torch.float64, device=generator.device
    )

    return drawn * scale


@final
class PinnedMap(nn.Module):
    """Rows over a latent inventory with a frozen identity prefix.

    Parameters
    ----------
    items : int
        Number of rows in the map.
    inventory : int
        Number of columns in the target inventory.
    target : _PinTarget
        Inventory the rows map into.
    mode : _PinMode
        Link applied to the learnable regular rows.
    cardinality : int
        Number of leading rows and columns forming the identity prefix.
    generator : torch.Generator
        Source of the initial draws, whose device the map is placed on.
    """

    _special: Tensor
    logits: nn.Parameter | None

    def __init__(
        self,
        items: int,
        inventory: int,
        *,
        target: _PinTarget,
        mode: _PinMode,
        cardinality: int,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()
        ensure_positive("pinned-map item cardinality", items)
        ensure_positive("pinned-map inventory cardinality", inventory)
        ensure_non_negative("pinned-prefix cardinality", cardinality)

        if target not in ("semantic_primitives", "item_clusters"):
            msg = f"the pinned-map target {target!r} is not declared"

            raise ValueError(msg)

        if mode not in ("independent_sigmoid", "normalized_softmax"):
            msg = f"the pinned-map mode {mode!r} is not declared"

            raise ValueError(msg)

        expected: _PinMode = (
            "independent_sigmoid"
            if target == "semantic_primitives"
            else "normalized_softmax"
        )

        if mode != expected:
            msg = (
                f"the pinned-map target {target!r} requires mode {expected!r}, "
                f"not {mode!r}"
            )

            raise ValueError(msg)

        if cardinality > items:
            msg = (
                f"the pinned-prefix cardinality {cardinality} exceeds the item "
                f"cardinality {items}"
            )

            raise ValueError(msg)

        if cardinality > inventory:
            msg = (
                f"the pinned-prefix cardinality {cardinality} exceeds the target "
                f"inventory cardinality {inventory}"
            )

            raise ValueError(msg)

        if mode == "normalized_softmax" and cardinality == inventory < items:
            msg = (
                f"the pinned-prefix cardinality {cardinality} reserves every "
                f"target column but leaves {items - cardinality} regular rows"
            )

            raise ValueError(msg)

        self._items = items
        self._inventory = inventory
        self._target = target
        self._mode = mode
        self._cardinality = cardinality
        special = torch.eye(
            cardinality,
            inventory,
            dtype=torch.float64,
            device=generator.device,
        )
        self.register_buffer("_special", special)
        regular = items - cardinality
        columns = (
            inventory if mode == "independent_sigmoid" else inventory - cardinality
        )

        if regular == 0:
            self.register_parameter("logits", None)
        else:
            self.logits = nn.Parameter(
                _drawn((regular, columns), generator, scale=initial_scale)
            )

    @property
    def items(self) -> int:
        """Return the number of item rows."""
        return self._items

    @property
    def inventory(self) -> int:
        """Return the number of target columns."""
        return self._inventory

    @property
    def target(self) -> _PinTarget:
        """Return the inventory the rows map into."""
        return self._target

    @property
    def mode(self) -> _PinMode:
        """Return the link applied to regular rows."""
        return self._mode

    @property
    def cardinality(self) -> int:
        """Return the size of the frozen identity prefix."""
        return self._cardinality

    @override
    def forward(self) -> Tensor:
        """Return the linked map, with the identity rows prepended."""
        logits = self.logits

        if logits is None:
            return self._special

        if self._mode == "independent_sigmoid":
            regular = torch.sigmoid(logits)
        else:
            padding = logits.new_full(
                (self._items - self._cardinality, self._cardinality),
                -torch.inf,
            )
            regular = torch.softmax(torch.cat((padding, logits), dim=1), dim=1)

        return torch.cat((self._special, regular), dim=0)


def slot_types(inventory: ColumnInventory, *, device: torch.device) -> Tensor:
    """Return the primitive type each argument slot of each column holds.

    Rows index columns in declaration order and columns index argument slots.
    A column taking fewer arguments than the widest one holds the null type
    in every slot beyond its own, so every row is the inventory's arity long.

    Parameters
    ----------
    inventory : ColumnInventory
        Columns and the primitive types their slots are drawn from.
    device : torch.device
        Device the result is placed on.
    """
    positions = {name: index for index, name in enumerate(inventory.primitive_types)}
    rows = [
        [positions[argument] for argument in column.arguments]
        + [inventory.null_index] * (inventory.arity - len(column.arguments))
        for column in inventory.levels
    ]

    return torch.tensor(rows, dtype=torch.long, device=device)


@final
class PrimitiveTypeMap(nn.Module):
    """One vector per primitive type, the null type included.

    Parameters
    ----------
    inventory : ColumnInventory
        Inventory whose types are mapped, and which assigns the null index.
    width : int
        Width of the vector a primitive type is mapped to.
    generator : torch.Generator
        Source of the initial draws, whose device the vectors are placed on.
    """

    def __init__(
        self,
        inventory: ColumnInventory,
        width: int,
        *,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self._inventory = inventory
        self._width = width
        self.vectors = nn.Parameter(
            _drawn(
                (inventory.type_count, width),
                generator,
                scale=initial_scale,
            )
        )

    @property
    def inventory(self) -> ColumnInventory:
        """Return the inventory whose types are mapped."""
        return self._inventory

    @property
    def width(self) -> int:
        """Return the width of the vector a primitive type is mapped to."""
        return self._width

    def summed(self, slots: Tensor) -> Tensor:
        """Return each column's vector as the sum over its argument slots.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column.
        """
        return self.vectors[slots].sum(dim=1)

    def joint(self, slots: Tensor) -> Tensor:
        """Return each column's joint tensor over its slots, flattened.

        The slots are lifted by outer product, so a column whose padding the
        null type filled carries that type's vector along the axis it fills
        and the result is one vector per column of the width raised to the
        arity.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column.
        """
        vectors = self.vectors[slots]
        product = vectors[:, 0, :]

        for slot in range(1, vectors.shape[1]):
            paired = product.unsqueeze(-1) * vectors[:, slot, :].unsqueeze(1)
            product = paired.flatten(start_dim=1)

        return product


@final
class _Memorized(nn.Module):
    """A free score for every row and column, which memorizes the table.

    Parameters
    ----------
    rows : int
        Number of rows under comparison.
    columns : int
        Number of columns under comparison.
    device : torch.device
        Device the scores are placed on.
    """

    def __init__(self, rows: int, columns: int, *, device: torch.device) -> None:
        super().__init__()
        self.cells = nn.Parameter(
            torch.zeros(rows, columns, dtype=torch.float64, device=device)
        )

    def table(self, slots: Tensor, /) -> Tensor:
        """Return the latent score of every row and column.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column, which a
            memorized table does not read.
        """
        del slots

        return self.cells


@final
class _Labels(nn.Module):
    """Column vectors read from a free table over column labels.

    The table is free, so two columns sharing a primitive type share nothing
    here. Each of the declared layers is affine and is followed by a
    hyperbolic tangent.

    Parameters
    ----------
    rows : int
        Number of rows under comparison.
    columns : int
        Number of columns under comparison.
    width : int
        Width of the vector a column is represented by.
    depth : int
        Number of affine layers between the label table and the row.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        width: int,
        depth: int,
        *,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self.labels = nn.Parameter(
            _drawn((columns, width), generator, scale=initial_scale)
        )
        self.weights = nn.Parameter(
            _drawn((depth, width, width), generator, scale=initial_scale)
        )
        self.biases = nn.Parameter(
            torch.zeros(depth, width, dtype=torch.float64, device=generator.device)
        )
        self.row_vectors = nn.Parameter(
            _drawn((rows, width), generator, scale=initial_scale)
        )

    def table(self, slots: Tensor, /) -> Tensor:
        """Return the latent score of every row and column.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column, which a label
            table does not read.
        """
        del slots
        columns = self.labels

        for layer in range(self.weights.shape[0]):
            columns = torch.tanh(columns @ self.weights[layer] + self.biases[layer])

        return self.row_vectors @ columns.T


@final
class _JointType(nn.Module):
    """Rows read against the joint tensor over a column's argument slots.

    Parameters
    ----------
    rows : int
        Number of rows under comparison.
    primitive_types : PrimitiveTypeMap
        The map from a primitive type to its vector.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.
    """

    def __init__(
        self,
        rows: int,
        primitive_types: PrimitiveTypeMap,
        *,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self.primitive_types = primitive_types
        joint = primitive_types.width**primitive_types.inventory.arity
        self.row_vectors = nn.Parameter(
            _drawn((rows, joint), generator, scale=initial_scale)
        )

    def table(self, slots: Tensor, /) -> Tensor:
        """Return the latent score of every row and column.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column.
        """
        return self.row_vectors @ self.primitive_types.joint(slots).T


@final
class _Possession(nn.Module):
    """Columns read through the lexical-semantic primitives a row possesses.

    Possession is many per row, so each lexical-semantic primitive gets its
    own probability, and each scores a column through the summed vectors of
    the primitive types its slots hold.

    Parameters
    ----------
    rows : int
        Number of rows under comparison.
    lexical_primitives : int
        Number of lexical-semantic primitives a row may possess.
    primitive_types : PrimitiveTypeMap
        The map from a primitive type to its vector.
    prefix : int
        Number of leading rows pinned to matching semantic primitives.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.
    """

    def __init__(
        self,
        rows: int,
        lexical_primitives: int,
        primitive_types: PrimitiveTypeMap,
        prefix: int,
        *,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self.primitive_types = primitive_types
        self.possession = PinnedMap(
            rows,
            lexical_primitives,
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=prefix,
            generator=generator,
            initial_scale=initial_scale,
        )
        self.readouts = nn.Parameter(
            _drawn(
                (lexical_primitives, primitive_types.width),
                generator,
                scale=initial_scale,
            )
        )

    def table(self, slots: Tensor, /) -> Tensor:
        """Return the latent score of every row and column.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column.
        """
        columns = self.primitive_types.summed(slots)

        return self.possession() @ self.readouts @ columns.T


@final
class _Classes(nn.Module):
    """Columns read through a learned row class and its entailments.

    Membership in a row class is single-label, so it is a distribution over
    classes. One class entails another's columns through a class-by-class
    relation, and a class scores a column through the summed vectors of the
    primitive types its slots hold.

    Parameters
    ----------
    rows : int
        Number of rows under comparison.
    classes : int
        Number of row classes the membership latent ranges over.
    primitive_types : PrimitiveTypeMap
        The map from a primitive type to its vector.
    prefix : int
        Number of leading rows pinned to matching item clusters.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.
    """

    def __init__(
        self,
        rows: int,
        classes: int,
        primitive_types: PrimitiveTypeMap,
        prefix: int,
        *,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self.primitive_types = primitive_types
        self.membership = PinnedMap(
            rows,
            classes,
            target="item_clusters",
            mode="normalized_softmax",
            cardinality=prefix,
            generator=generator,
            initial_scale=initial_scale,
        )
        self.entailment = nn.Parameter(
            _drawn((classes, classes), generator, scale=initial_scale)
        )
        self.readouts = nn.Parameter(
            _drawn(
                (classes, primitive_types.width),
                generator,
                scale=initial_scale,
            )
        )

    def table(self, slots: Tensor, /) -> Tensor:
        """Return the latent score of every row and column.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column.
        """
        columns = self.primitive_types.summed(slots)
        membership = self.membership()
        entailment = torch.sigmoid(self.entailment)

        return membership @ entailment @ self.readouts @ columns.T


@final
class _PossessionAndClasses(nn.Module):
    """Both row-side latents, each with the link its support takes.

    Parameters
    ----------
    possession : _Possession
        The possession latent over the lexical-semantic primitives.
    classes : _Classes
        The membership latent over the row classes.
    """

    def __init__(self, possession: _Possession, classes: _Classes) -> None:
        super().__init__()
        self.possession = possession
        self.classes = classes

    def table(self, slots: Tensor, /) -> Tensor:
        """Return the latent score of every row and column.

        Parameters
        ----------
        slots : Tensor
            The primitive type of each slot of each column.
        """
        return self.possession.table(slots) + self.classes.table(slots)


type _Lift = (
    _Memorized | _Labels | _JointType | _Possession | _Classes | _PossessionAndClasses
)


def _lift_of(
    spec: TableAblationSpec,
    primitive_types: PrimitiveTypeMap,
    *,
    generator: torch.Generator,
    initial_scale: float,
) -> _Lift:
    """Return the parameterization the declared lift names.

    Parameters
    ----------
    spec : TableAblationSpec
        The rung whose lift is parameterized.
    primitive_types : PrimitiveTypeMap
        The map every decomposing rung reads.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.

    Raises
    ------
    TypeError
        If the lift is outside the declared union.
    """
    rows = len(spec.rows.levels)
    columns = len(spec.columns.inventory.levels)
    lift = spec.lift

    match lift:
        case MemorizedCells():
            return _Memorized(rows, columns, device=generator.device)
        case ColumnLabels():
            return _Labels(
                rows,
                columns,
                spec.width,
                lift.depth,
                generator=generator,
                initial_scale=initial_scale,
            )
        case JointTypeTensor():
            return _JointType(
                rows,
                primitive_types,
                generator=generator,
                initial_scale=initial_scale,
            )
        case LexicalPossession():
            return _Possession(
                rows,
                lift.lexical_primitives.count,
                primitive_types,
                lift.prefix.cardinality,
                generator=generator,
                initial_scale=initial_scale,
            )
        case RowClasses():
            return _Classes(
                rows,
                lift.classes,
                primitive_types,
                lift.prefix.cardinality,
                generator=generator,
                initial_scale=initial_scale,
            )
        case PossessionAndClasses():
            return _PossessionAndClasses(
                _Possession(
                    rows,
                    lift.lexical_primitives.count,
                    primitive_types,
                    lift.primitive_prefix.cardinality,
                    generator=generator,
                    initial_scale=initial_scale,
                ),
                _Classes(
                    rows,
                    lift.classes,
                    primitive_types,
                    lift.cluster_prefix.cardinality,
                    generator=generator,
                    initial_scale=initial_scale,
                ),
            )

    msg = f"the lift {lift.lift!r} has no parameterization in this module"

    raise TypeError(msg)


@final
class TableRung(nn.Module):
    """Latent scores for row-by-column judgments under one declared lift.

    The rung holds a score for every declared row and column, and a judgment
    is scored by the row and the column its row records. A design recording
    a row or a column the comparison does not declare has no score here, so
    it is refused rather than given one.

    Parameters
    ----------
    spec : TableAblationSpec
        The rung: the rows, the columns, the width and the lift.
    primitive_types : PrimitiveTypeMap
        The map from a primitive type to its vector, shared across rungs.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.

    Raises
    ------
    TypeError
        If the lift is outside the declared union.
    ValueError
        If the map was built over other columns or at another width.
    """

    _slots: Tensor

    def __init__(
        self,
        spec: TableAblationSpec,
        primitive_types: PrimitiveTypeMap,
        *,
        generator: torch.Generator,
        initial_scale: float,
    ) -> None:
        super().__init__()

        if primitive_types.inventory != spec.columns.inventory:
            msg = (
                "a rung reads the primitive-type map of the columns it declares, "
                "and this map was built over other columns"
            )

            raise ValueError(msg)

        if primitive_types.width != spec.width:
            msg = (
                f"a rung of width {spec.width} reads a map of width "
                f"{primitive_types.width}"
            )

            raise ValueError(msg)

        self._spec = spec
        self._rows = {row: index for index, row in enumerate(spec.rows.levels)}
        self._columns = {
            label: index for index, label in enumerate(spec.columns.inventory.labels)
        }
        self.register_buffer(
            "_slots", slot_types(spec.columns.inventory, device=generator.device)
        )
        self._lift = _lift_of(
            spec,
            primitive_types,
            generator=generator,
            initial_scale=initial_scale,
        )

    @property
    def spec(self) -> TableAblationSpec:
        """Return the rung this evaluator was built from."""
        return self._spec

    @property
    def terms(self) -> tuple[Predictor, ...]:
        """Return the declared terms this evaluator implements."""
        return self._spec.terms

    def table(self) -> Tensor:
        """Return the latent score of every declared row and column.

        Rows index rows and columns index columns, both in the order the
        specification declares them.
        """
        return self._lift.table(self._slots)

    def _column(
        self,
        design: JudgmentDesign,
        factor: str,
        positions: dict[str, int],
    ) -> Tensor:
        """Return the rung's index of the level each row records.

        Parameters
        ----------
        design : JudgmentDesign
            Rows whose factor column is read.
        factor : str
            Name of the factor to read.
        positions : dict[str, int]
            Index the rung holds each level under.

        Raises
        ------
        ValueError
            If the design declares no such factor, or records a level the
            comparison does not declare.
        """
        column = design.factors.get(factor)

        if column is None:
            msg = (
                f"the comparison reads the factor {factor!r} and the design "
                f"declares {tuple(design.factors)!r}"
            )

            raise ValueError(msg)

        levels = design.factor_levels[factor]
        lookup: list[int] = []

        for level in levels:
            index = positions.get(level)

            if index is None:
                msg = (
                    f"the design records {level!r} under {factor!r}, which the "
                    f"comparison does not declare"
                )

                raise ValueError(msg)

            lookup.append(index)

        return torch.tensor(lookup, dtype=torch.long, device=column.device)[column]

    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        """Return one latent score per judgment row.

        Parameters
        ----------
        design : JudgmentDesign
            Rows and columns to evaluate.

        Raises
        ------
        ValueError
            If the design declares neither factor the comparison reads, or
            records a row or a column the comparison does not declare.
        """
        rows = self._column(design, self._spec.rows.attribute, self._rows)
        columns = self._column(design, self._spec.columns.attribute, self._columns)

        return self.table()[rows, columns]

    @override
    def __repr__(self) -> str:
        """Return the lift, the row count and the column count."""
        return (
            f"TableRung(lift={self._spec.lift.lift!r}, "
            f"rows={len(self._rows)}, columns={len(self._columns)})"
        )


def ladder_of(
    specs: Sequence[TableAblationSpec],
    /,
    *,
    generator: torch.Generator,
) -> tuple[TableRung, ...]:
    """Return one rung per specification, over one primitive-type map.

    The map is built once and handed to every rung, so the rungs that
    decompose a column read the same primitive types and differ in the lift
    alone.

    Parameters
    ----------
    specs : Sequence[TableAblationSpec]
        The rungs of the comparison, in ladder order.
    generator : torch.Generator
        Source of the initial draws, whose device the parameters are on.

    Raises
    ------
    ValueError
        If no specification is given, or two of them declare different
        rows, different columns or different widths.
    """
    if not specs:
        msg = "a ladder has at least one rung and none was given"

        raise ValueError(msg)

    first = specs[0]

    for spec in specs[1:]:
        if (
            spec.columns != first.columns
            or spec.rows != first.rows
            or spec.width != first.width
            or spec.schedule != first.schedule
        ):
            msg = (
                f"the rung {spec.lift.lift!r} declares other rows, columns, "
                f"width or fit schedule than the rung {first.lift.lift!r}, so "
                f"the two are not rungs of one ladder"
            )

            raise ValueError(msg)

    initial_scale = first.schedule.initial_scale
    primitive_types = PrimitiveTypeMap(
        first.columns.inventory,
        first.width,
        generator=generator,
        initial_scale=initial_scale,
    )

    return tuple(
        TableRung(
            spec,
            primitive_types,
            generator=generator,
            initial_scale=initial_scale,
        )
        for spec in specs
    )


def mean_log_density(
    rung: TableRung,
    head: OrdinalHead,
    design: JudgmentDesign,
    /,
) -> Tensor:
    """Return the mean log probability of the responses a design records.

    Parameters
    ----------
    rung : TableRung
        Evaluator of the latent score of each row.
    head : OrdinalHead
        Cutpoints between adjacent response levels.
    design : JudgmentDesign
        Rows whose recorded responses are scored.

    Raises
    ------
    ValueError
        If the head and the design disagree about the number of levels.
    """
    if design.response_levels != head.levels:
        msg = (
            f"a head over {head.levels} levels was given a design over "
            f"{design.response_levels}"
        )

        raise ValueError(msg)

    log_probabilities = head.log_probabilities(rung.latent_scores(design))
    recorded = design.responses.to(dtype=torch.long).unsqueeze(1)

    return log_probabilities.gather(1, recorded).squeeze(1).mean()


def fit_rung(
    rung: TableRung,
    head: OrdinalHead,
    design: JudgmentDesign,
    /,
) -> None:
    """Fit a rung and the head it is read through to a design.

    Each step descends the gradient of the mean negative log probability of
    every recorded response in the design, so the fit is a function of the
    design and of the parameters it starts from. The same decay applies to
    every parameter of every rung, so it shrinks a memorized cell and a
    primitive type's vector alike. The rung's schedule declares its optimizer,
    update count, initialization scale and whether each update consumes the
    full design or a deterministic minibatch.

    Parameters
    ----------
    rung : TableRung
        Evaluator whose parameters are fitted.
    head : OrdinalHead
        Cutpoints, fitted alongside the rung.
    design : JudgmentDesign
        Rows fitted to.
    """
    schedule = rung.spec.schedule
    parameters = [*rung.parameters(), *head.parameters()]
    declaration = cast("ConfiguredSpec[object]", schedule.optimizer)
    built = plugin_for(declaration).construct(declaration, parameters)

    if not isinstance(built, Optimizer):
        msg = (
            f"the optimizer plugin {schedule.optimizer.kind!r} built {built!r}, "
            "not a torch optimizer"
        )
        raise TypeError(msg)

    optimizer = built

    for step in range(schedule.steps):
        batch = design

        if schedule.batch_size is not None and schedule.batch_size < design.n_rows:
            start = step * schedule.batch_size % design.n_rows
            positions = tuple(
                (start + offset) % design.n_rows
                for offset in range(schedule.batch_size)
            )
            batch = design.select_rows(positions)

        optimizer.zero_grad()
        loss = -mean_log_density(rung, head, batch)
        loss.backward()
        optimizer.step()

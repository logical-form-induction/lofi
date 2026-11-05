import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._distance import TypeDistanceObjective, inverse_distance_weights
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.terms import (
    BatchEncoding,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
    TreeEditDistance,
    class_sensitive_relabel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import LossTerm
    from lofi.nn.ports import TermEncoder
    from lofi.terms import NodeLabel, RelabelCost, Term


SIGNATURE: Final = Signature(
    "regressed",
    ("T",),
    (
        Operator("A", "T", family="atom"),
        Operator("B", "T", family="atom"),
        Operator("f", "T", arg_sorts=("T", "T"), family="functor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CODEC: Final = TermCodec(
    INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3)
)
STORE: Final = TermStore(SIGNATURE)
CAPACITY: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
WIDTH: Final = 2
A: Final = STORE.op(SIGNATURE.by_name("A"))
B: Final = STORE.op(SIGNATURE.by_name("B"))


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


type Pair = tuple[Term, Term]


class Pairs(NamedTuple):
    pairs: tuple[Pair, ...]
    left: BatchEncoding
    right: BatchEncoding


class BagEncoder:
    """Encodes a type as the sum of its occupied symbols' embeddings."""

    def __init__(self) -> None:
        weight = torch.zeros(SYMBOLS, WIDTH, dtype=torch.float64)
        weight[INDEX.index(SIGNATURE.by_name("A"))] = torch.tensor([1.0, 0.0])
        weight[INDEX.index(SIGNATURE.by_name("B"))] = torch.tensor([0.0, 3.0])
        weight[INDEX.index(SIGNATURE.by_name("f"))] = torch.tensor([0.5, -0.5])
        self.weight = weight.requires_grad_()

    @property
    def output_width(self) -> int:
        return WIDTH

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        occupied = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        embedded = self.weight[symbols] * occupied.unsqueeze(-1).to(torch.float64)
        held = occupied.any(dim=1)

        return Encoded(
            embedded.sum(dim=1), ("term", "width"), None if bool(held.all()) else held
        )


ENCODER: Final = BagEncoder()


def objective(
    encoder: BagEncoder = ENCODER, *, relabel: RelabelCost = class_sensitive_relabel
) -> TypeDistanceObjective:
    return TypeDistanceObjective(
        cast("TermEncoder", encoder),
        inverse_distance_power=0.0,
        embedding_penalty=0.0,
        relabel=relabel,
    )


def dead_row() -> Encoding:
    """Return a row holding no term, whose slots read as some symbol anyway."""
    filled = CODEC.encode(f(f(A, B), f(B, A)))

    return Encoding(
        tuple(2 for _ in filled.symbols),
        filled.parents,
        filled.depths,
        tuple(False for _ in filled.occupied),
    )


def bundle_of(
    pairs: tuple[Pair, ...],
    left: tuple[Encoding, ...],
    right: tuple[Encoding, ...],
) -> ObjectiveOutputs[Pairs]:
    return ObjectiveOutputs(
        spans=SpanIndex(1),
        chart=CellChart(
            LogDistribution(torch.zeros(len(pairs), 1, dtype=torch.float64), SPAN_AXES)
        ),
        batch=Pairs(pairs, BatchEncoding(left), BatchEncoding(right)),
    )


def bundle_over(*pairs: Pair) -> ObjectiveOutputs[Pairs]:
    return bundle_of(
        pairs,
        CODEC.encode_batch([left for left, _ in pairs]).rows,
        CODEC.encode_batch([right for _, right in pairs]).rows,
    )


PAIRS: Final[tuple[Pair, ...]] = (
    (A, B),
    (A, f(A, B)),
    (f(A, B), f(B, A)),
    (f(f(A, B), A), B),
)


def bundle() -> ObjectiveOutputs[Pairs]:
    return bundle_over(*PAIRS)


def permuted(outputs: ObjectiveOutputs[Pairs]) -> ObjectiveOutputs[Pairs]:
    order = (3, 1, 0, 2)
    batch = outputs.batch

    return bundle_of(
        tuple(batch.pairs[position] for position in order),
        tuple(batch.left.rows[position] for position in order),
        tuple(batch.right.rows[position] for position in order),
    )


def refilled(outputs: ObjectiveOutputs[Pairs]) -> ObjectiveOutputs[Pairs]:
    batch = outputs.batch

    return bundle_of(
        (*batch.pairs, (A, B)),
        (*batch.left.rows, dead_row()),
        (*batch.right.rows, CODEC.encode(B)),
    )


def empty_bundle() -> ObjectiveOutputs[Pairs]:
    return bundle_of(((A, B),), (dead_row(),), (dead_row(),))


def inputs_of(outputs: ObjectiveOutputs[Pairs]) -> tuple[Tensor, ...]:
    del outputs

    return (ENCODER.weight,)


def bag(term: Term) -> list[float]:
    encoding = CODEC.encode(term)
    vector = [0.0] * WIDTH

    for slot, symbol in enumerate(encoding.symbols):
        if encoding.occupied[slot]:
            for axis in range(WIDTH):
                vector[axis] += ENCODER.weight[symbol, axis].item()

    return vector


def by_hand(outputs: ObjectiveOutputs[Pairs]) -> float:
    distance = TreeEditDistance()
    errors = []

    for (left, right), row in zip(
        outputs.batch.pairs, outputs.batch.left.rows, strict=True
    ):
        if not any(row.occupied):
            continue

        squared = sum(
            (first - second) ** 2
            for first, second in zip(bag(left), bag(right), strict=True)
        )
        errors.append((squared - distance(left, right) ** 2) ** 2)

    return sum(errors) / len(errors) if errors else 0.0


def models_built(call: Callable[[], LossTerm]) -> int:
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


def fixture() -> ObjectiveFixture[Pairs]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (bundle_over((A, B)), bundle_over((A, A), (B, B))),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
        tolerance=1e-9,
    )


class TestTypeDistanceObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Pairs]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that pairs whose encodings sit at the wrong distance score higher."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_value_is_the_mean_squared_error_between_the_squared_distances() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.count == len(PAIRS)
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)


def test_a_pair_of_identical_types_at_one_point_scores_zero() -> None:
    term = objective().evaluate(bundle_over((A, A), (f(A, B), f(A, B))))

    assert term.value.item() == 0.0
    assert term.diagnostics["absolute_error"].item() == 0.0


def test_the_absolute_error_is_read_in_edit_distance_units() -> None:
    term = objective().evaluate(bundle_over((A, B)))

    assert term.diagnostics["absolute_error"].item() == pytest.approx(
        abs(math.sqrt(10.0) - 1.0), rel=1e-12
    )
    assert term.value.item() == pytest.approx((10.0 - 1.0) ** 2, rel=1e-12)
    assert term.diagnostics["absolute_error"].grad_fn is None


def test_the_target_is_the_edit_distance_under_the_relabel_cost_held() -> None:
    def free(left: NodeLabel, right: NodeLabel, /) -> int:
        del left, right

        return 0

    built = objective(relabel=free)

    term = built.evaluate(bundle_over((A, B)))

    assert built.relabel is free
    assert term.value.item() == pytest.approx(10.0**2, rel=1e-12)


def test_a_pair_is_costed_once_per_step_and_nothing_is_kept_between_steps() -> None:
    costed: list[tuple[NodeLabel, NodeLabel]] = []

    def counted(left: NodeLabel, right: NodeLabel, /) -> int:
        costed.append((left, right))

        return class_sensitive_relabel(left, right)

    built = objective(relabel=counted)
    repeated = bundle_over((A, B), (A, B), (f(A, B), f(A, B)))

    built.evaluate(bundle_over((A, B), (f(A, B), f(A, B))))
    distinct = tuple(costed)
    costed.clear()
    built.evaluate(repeated)
    first = tuple(costed)
    costed.clear()
    built.evaluate(repeated)

    assert len(distinct) > 0
    assert first == distinct
    assert tuple(costed) == first


def test_a_pair_missing_its_second_type_is_neither_scored_nor_counted() -> None:
    complete = objective().evaluate(bundle())
    padded = objective().evaluate(
        bundle_of(
            (*PAIRS, (B, A)),
            (*bundle().batch.left.rows, CODEC.encode(B)),
            (*bundle().batch.right.rows, dead_row()),
        )
    )

    assert padded.count == len(PAIRS)
    assert padded.value.item() == pytest.approx(complete.value.item(), rel=1e-12)
    assert padded.diagnostics["absolute_error"].item() == pytest.approx(
        complete.diagnostics["absolute_error"].item(), rel=1e-12
    )


def test_moving_the_encodings_toward_the_edit_distance_lowers_the_value() -> None:
    encoder = BagEncoder()
    built = objective(encoder)
    outputs = bundle_over((A, B))

    before = built.evaluate(outputs)
    before.value.backward()

    assert encoder.weight.grad is not None

    with torch.no_grad():
        encoder.weight -= 0.01 * encoder.weight.grad

    after = built.evaluate(outputs)

    assert after.value.item() < before.value.item()


def test_a_pair_missing_a_type_is_neither_scored_nor_counted() -> None:
    complete = objective().evaluate(bundle())
    padded = objective().evaluate(refilled(bundle()))

    assert padded.count == len(PAIRS)
    assert padded.value.item() == pytest.approx(complete.value.item(), rel=1e-12)


def test_a_batch_holding_no_pair_is_inapplicable() -> None:
    term = objective().evaluate(empty_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


@pytest.mark.parametrize("dropped", ["pairs", "left", "right"])
def test_a_batch_whose_pairs_and_linearizations_disagree_is_refused(
    dropped: str,
) -> None:
    complete = bundle().batch
    shortened = Pairs(
        complete.pairs[1:] if dropped == "pairs" else complete.pairs,
        BatchEncoding(complete.left.rows[1:]) if dropped == "left" else complete.left,
        BatchEncoding(complete.right.rows[1:])
        if dropped == "right"
        else complete.right,
    )
    outputs: ObjectiveOutputs[Pairs] = ObjectiveOutputs(
        spans=SpanIndex(1),
        chart=CellChart(LogDistribution(torch.zeros(4, 1), SPAN_AXES)),
        batch=shortened,
    )

    with pytest.raises(ValueError, match="pairs of types"):
        objective().evaluate(outputs)


def test_the_objective_declares_what_it_reads() -> None:
    built = objective()

    assert built.requires == frozenset({"batch"})
    assert built.name == "type-distance"
    assert built.relabel is class_sensitive_relabel
    assert "BagEncoder" in repr(built)


def test_a_blank_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        TypeDistanceObjective(
            cast("TermEncoder", ENCODER),
            inverse_distance_power=0.0,
            embedding_penalty=0.0,
            name="",
        )


def test_inverse_distance_weights_are_normalized_and_monotone() -> None:
    distances = torch.tensor([0.0, 1.0, 3.0, 3.0], dtype=torch.float64)
    weighted = inverse_distance_weights(distances, 2.0)
    uniform = inverse_distance_weights(distances, 0.0)

    assert weighted.sum().item() == pytest.approx(1.0)
    assert torch.equal(uniform, torch.full_like(uniform, 0.25))
    assert weighted[0] > weighted[1] > weighted[2]
    assert weighted[2] == weighted[3]


def test_the_embedding_penalty_enters_pair_contributions_before_reduction() -> None:
    outputs = bundle()
    baseline = TypeDistanceObjective(
        cast("TermEncoder", ENCODER),
        inverse_distance_power=0.0,
        embedding_penalty=0.0,
    ).evaluate(outputs)
    penalized = TypeDistanceObjective(
        cast("TermEncoder", ENCODER),
        inverse_distance_power=1.5,
        embedding_penalty=0.25,
    ).evaluate(outputs)
    weighted_without_penalty = TypeDistanceObjective(
        cast("TermEncoder", ENCODER),
        inverse_distance_power=1.5,
        embedding_penalty=0.0,
    ).evaluate(outputs)

    assert baseline.count == penalized.count
    assert penalized.value > weighted_without_penalty.value
    assert (
        penalized.value.item() - weighted_without_penalty.value.item()
        == pytest.approx(0.25 * penalized.diagnostics["embedding_penalty"].item())
    )


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_distance_objective_coefficients_are_refused(value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        TypeDistanceObjective(
            cast("TermEncoder", ENCODER),
            inverse_distance_power=value,
            embedding_penalty=0.0,
        )


def test_the_term_reaches_the_suite_under_its_name() -> None:
    outputs = bundle()
    built = objective()

    result = ObjectiveSuite([(0.5, built)]).evaluate(outputs)

    assert result.total.item() == pytest.approx(
        0.5 * built.evaluate(outputs).value.item(), rel=1e-12
    )
    assert set(result.diagnostics) == {
        "type-distance/loss",
        "type-distance/absolute_error",
        "type-distance/embedding_penalty",
    }

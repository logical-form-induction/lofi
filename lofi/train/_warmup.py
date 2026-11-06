"""The token-encoder warmup step, and the row transfer that seeds it."""

from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_name, ensure_non_negative, refuse_reassignment
from lofi.nn.tokens.transfer import initialize_from_types
from lofi.train._optimizing import Optimizing, over_no_span, stepped_result
from lofi.train._pretraining import Trained
from lofi.train._steps import StepOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.lexicon import LexicalSource
    from lofi.nn.ports import TermEncoder
    from lofi.nn.tokens.embedder import TokenVocabulary
    from lofi.nn.tokens.transfer import AmbiguityPolicy, TransferReport
    from lofi.terms import BatchEncoding, Term, TermCodec
    from lofi.train._losses import LossAssembly
    from lofi.train._optimization import OptimizationResult
    from lofi.train._ports import OptimizationPolicy
    from lofi.train._steps import TrainResult

__all__ = ["TokenWarmupStep", "Utterances", "WarmupBatch", "transfer_rows"]


class Utterances:
    """The utterances and selected lexical candidates a warmup step reads.

    Parameters
    ----------
    items : Sequence[Sequence[str]]
        The words of each item, in item order and then in word order.
    candidates : Sequence[Sequence[Sequence[Term]]]
        The admitted types of each word, aligned to items and words.
    step : int
        The count of optimizer steps the run has taken before this one.

    Raises
    ------
    TypeError
        If the step is not an integer.
    ValueError
        If the step is negative.
    """

    __slots__ = ("_candidates", "_items", "_step")

    def __init__(
        self,
        items: Sequence[Sequence[str]],
        candidates: Sequence[Sequence[Sequence[Term]]],
        step: int,
    ) -> None:
        held_items = tuple(tuple(words) for words in items)
        held_candidates = tuple(
            tuple(tuple(types) for types in item) for item in candidates
        )

        if len(held_candidates) != len(held_items) or any(
            len(types) != len(words)
            for words, types in zip(held_items, held_candidates, strict=True)
        ):
            msg = "warm-up candidates do not align one row to every item and word"
            raise ValueError(msg)

        object.__setattr__(self, "_items", held_items)
        object.__setattr__(self, "_candidates", held_candidates)
        object.__setattr__(self, "_step", ensure_non_negative("step", step))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("warm-up request")

    @property
    def items(self) -> tuple[tuple[str, ...], ...]:
        """Return the words of each item."""
        return self._items

    @property
    def candidates(self) -> tuple[tuple[tuple[Term, ...], ...], ...]:
        """Return the selected types of each word."""
        return self._candidates

    @property
    def step(self) -> int:
        """Return the count of optimizer steps taken before this one."""
        return self._step

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another request holds the same items at the same position."""
        if type(other) is not Utterances:
            return NotImplemented

        return (
            self._items == other._items
            and self._candidates == other._candidates
            and self._step == other._step
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the items and the position."""
        return hash((Utterances, self._items, self._candidates, self._step))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this request."""
        return (
            f"Utterances({self._items!r}, candidates={self._candidates!r}, "
            f"step={self._step!r})"
        )


@final
class WarmupBatch:
    """A batch of items whose words each may take the types a source admits.

    Parameters
    ----------
    words : tuple[tuple[str, ...], ...]
        The words of each item, in item order and then in word order.
    types : BatchEncoding
        The linearized types the words may take, one row per candidate.
    positions : tuple[tuple[int, int], ...]
        The item and the word each candidate row belongs to, in row order.
    """

    __slots__ = ("_positions", "_types", "_words")

    def __init__(
        self,
        words: tuple[tuple[str, ...], ...],
        types: BatchEncoding,
        positions: tuple[tuple[int, int], ...],
    ) -> None:
        object.__setattr__(self, "_words", words)
        object.__setattr__(self, "_types", types)
        object.__setattr__(self, "_positions", positions)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("warm-up batch")

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return the words of each item."""
        return self._words

    @property
    def types(self) -> BatchEncoding:
        """Return the linearized candidate types, one row per candidate."""
        return self._types

    @property
    def positions(self) -> tuple[tuple[int, int], ...]:
        """Return the item and the word of each candidate row."""
        return self._positions

    def __len__(self) -> int:
        """Return how many items the batch holds."""
        return len(self._words)

    @override
    def __repr__(self) -> str:
        """Return the item and candidate counts."""
        return f"WarmupBatch(items={len(self)}, candidates={len(self._positions)})"


def _admitted[W](
    source: LexicalSource[Term, W], words: Sequence[str]
) -> tuple[tuple[int, Term], ...]:
    """Return the position and the type of each entry the source admits for the words.

    Parameters
    ----------
    source : LexicalSource[Term, W]
        What enumerates the entries.
    words : Sequence[str]
        The words, in position order.

    Returns
    -------
    tuple[tuple[int, Term], ...]
        Each admitted type once per position, in the order the source gave them.
    """
    seen: set[tuple[int, Term]] = set()
    admitted: list[tuple[int, Term]] = []

    for position, entry, _ in source.axioms(words):
        candidate = (position, entry.category)

        if candidate not in seen:
            seen.add(candidate)
            admitted.append(candidate)

    return tuple(admitted)


def transfer_rows[W](
    table: Tensor,
    source: LexicalSource[Term, W],
    *,
    encoder: TermEncoder,
    codec: TermCodec,
    vocabulary: TokenVocabulary,
    policy: AmbiguityPolicy,
) -> tuple[Tensor, TransferReport]:
    """Return the table with each token's row seeded from the types it may take.

    Every type the source admits for a token, read as a one-word item, is encoded
    through the term encoder, and the token's row is written from those vectors
    under the ambiguity policy. A token the source admits no type for keeps its row,
    as does one whose every admitted type the encoder holds no vector for.

    Parameters
    ----------
    table : Tensor
        The rows to write into, rank 2 over the axes ``("index", "width")`` at floating
        precision, with one row per index the vocabulary describes.
    source : LexicalSource[Term, W]
        What enumerates the types a token may take.
    encoder : TermEncoder
        What encodes each type to a vector as wide as the table's rows.
    codec : TermCodec
        What linearizes the types for the encoder.
    vocabulary : TokenVocabulary
        What the table's rows are indexed by, and whose tokens are seeded.
    policy : AmbiguityPolicy
        What to do with a token that may take several types.

    Returns
    -------
    tuple[Tensor, TransferReport]
        A copy of the table with the seeded rows replaced, and the counts of what the
        transfer did over the tokens the source admits a type for.
    """
    offered: dict[str, list[int]] = {}
    types: list[Term] = []

    for token in vocabulary.tokens:
        admitted = _admitted(source, (token,))

        if admitted:
            offered[token] = list(range(len(types), len(types) + len(admitted)))
            types.extend(term for _, term in admitted)

    if not types:
        return initialize_from_types(table, {}, vocabulary=vocabulary, policy=policy)

    encoded = encoder.encode(codec.encode_batch(types))
    held = encoded.mask
    tensor = encoded.tensor.detach()
    vectors = {
        token: [tensor[row] for row in rows if held is None or bool(held[row])]
        for token, rows in offered.items()
    }
    seeded = {token: found for token, found in vectors.items() if found}

    return initialize_from_types(table, seeded, vocabulary=vocabulary, policy=policy)


@final
class TokenWarmupStep:
    """Warm a token encoder toward selected occurrence-level candidates.

    Each request carries the candidates selected once at the data boundary. The
    step linearizes those candidates through the codec and takes one optimizer
    step on the assembled loss over them. An item with no selected candidate is
    counted as dropped.

    Parameters
    ----------
    codec : TermCodec
        What linearizes those types for the encoder.
    assembly : LossAssembly[WarmupBatch]
        The weighted objectives the token encoder trains under.
    policy : OptimizationPolicy[Tensor]
        What runs the backward pass and the optimizer.
    role : str
        The role the token encoder's state is checkpointed under.
    name : str | None
        The step's name, or None to name it after the role.

    Raises
    ------
    TypeError
        If the role or the name is not a string.
    ValueError
        If the role or the name is empty.
    """

    __slots__ = ("_codec", "_name", "_optimizing", "_role")

    def __init__(
        self,
        codec: TermCodec,
        assembly: LossAssembly[WarmupBatch],
        policy: OptimizationPolicy[Tensor],
        *,
        role: str,
        name: str | None = None,
    ) -> None:
        self._codec = codec
        self._optimizing: Optimizing[WarmupBatch] = Optimizing(assembly, policy)
        self._role = ensure_name("role", role)
        self._name = ensure_name("name", f"warmup:{role}" if name is None else name)

    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @property
    def requires(self) -> frozenset[str]:
        """Return required value names."""
        return frozenset({"candidates", "items", "step"})

    @property
    def produces(self) -> frozenset[str]:
        """Return produced value names."""
        return frozenset({self._role})

    @property
    def assembly(self) -> LossAssembly[WarmupBatch]:
        """Return the weighted objectives."""
        return self._optimizing.assembly

    def _batch(self, inputs: Utterances) -> tuple[WarmupBatch, int]:
        """Return selected candidates and how many items admit none.

        Parameters
        ----------
        inputs : Utterances
            The aligned words and selected candidates.
        """
        positions: list[tuple[int, int]] = []
        types: list[Term] = []
        dropped = 0
        items = inputs.items

        for item, candidates in enumerate(inputs.candidates):
            admitted = tuple(
                (position, term)
                for position, offered in enumerate(candidates)
                for term in dict.fromkeys(offered)
            )

            if not admitted:
                dropped += 1

            for position, term in admitted:
                positions.append((item, position))
                types.append(term)

        batch = WarmupBatch(items, self._codec.encode_batch(types), tuple(positions))

        return batch, dropped

    def run(self, inputs: Utterances, /) -> StepOutcome[Trained, TrainResult]:
        """Read the candidates for one batch of items and take one optimizer step.

        Parameters
        ----------
        inputs : Utterances
            The items and the position in the run.

        Returns
        -------
        StepOutcome[Trained, TrainResult]
            The checkpoint role, and the result over the items the source admitted a
            type for, with the candidate count beside the optimizer and loss scalars;
            a batch with no candidate takes no step.
        """
        items = inputs.items
        step = inputs.step
        batch, dropped = self._batch(inputs)
        metrics = {"warmup/candidates": float(len(batch.positions))}
        reported: OptimizationResult | None = None

        if batch.positions:
            reported, trained = self._optimizing.run(
                over_no_span(batch, len(batch)), step=step
            )
            metrics.update(trained)

        return StepOutcome(
            Trained(self._role),
            stepped_result(
                step, len(items) - dropped, reported, metrics, dropped=dropped
            ),
        )

    @override
    def __repr__(self) -> str:
        """Return the name, the role, the source and the assembly."""
        return (
            f"TokenWarmupStep(name={self._name!r}, role={self._role!r}, "
            f"assembly={self.assembly!r})"
        )

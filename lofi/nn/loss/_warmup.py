"""Token warm-up against the nearest of the types a token may take."""

from typing import TYPE_CHECKING, Final, NamedTuple, override

import torch

from lofi._checks import ensure_name
from lofi.nn._values import Encoded
from lofi.nn.loss._likelihood import negative_log_likelihood
from lofi.nn.loss._ports import LossTerm
from lofi.nn.strategies import score
from lofi.terms import BatchEncoding

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.loss._ports import AmbiguousTokens, PublishedOutputs
    from lofi.nn.ports import (
        SubwordPooler,
        TeacherForcedDecoder,
        TermEncoder,
        TokenEncoder,
    )
    from lofi.terms import ChoiceIndex

__all__ = ["Nearest", "TokenWarmupObjective", "nearest_candidates"]

#: The axes a readout reads one vector per term over.
_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")


class Nearest(NamedTuple):
    """The candidate each token is nearest to, over the tokens that have one.

    Attributes
    ----------
    tokens : Tensor
        The token each selected candidate belongs to, in ascending order, so the
        tokens with no live candidate are absent.
    rows : Tensor
        The row of the selected candidate for each of those tokens.
    """

    tokens: Tensor
    rows: Tensor


def nearest_candidates(distances: Tensor, *, owners: Tensor, tokens: int) -> Nearest:
    """Return the row nearest each token among the rows that belong to it.

    A token whose rows are all at an infinite distance selects none and is absent
    from the result. Among rows at the same distance, the earliest wins.

    Parameters
    ----------
    distances : Tensor
        One distance per row, at floating precision, infinite where a row is not to be
        selected.
    owners : Tensor
        The token each row belongs to, at integer precision, in the order of the rows.
    tokens : int
        How many tokens the rows are shared among.

    Returns
    -------
    Nearest
        The tokens that selected a row, ascending, and the row each selected.
    """
    best = distances.new_full((tokens,), torch.inf).scatter_reduce(
        0, owners, distances, reduce="amin", include_self=True
    )
    winning = (distances == best[owners]) & distances.isfinite()
    rows = torch.arange(distances.shape[0], device=distances.device)
    selected = torch.full(
        (tokens,), distances.shape[0], dtype=torch.long, device=distances.device
    ).scatter_reduce(
        0, owners[winning], rows[winning], reduce="amin", include_self=True
    )
    live = selected < distances.shape[0]

    return Nearest(torch.arange(tokens, device=distances.device)[live], selected[live])


def _positions(
    batch: AmbiguousTokens, *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Return where each candidate's token sits, and which token that is.

    Parameters
    ----------
    batch : AmbiguousTokens
        The tokens and the types each may take.
    device : torch.device
        Where the index tensors are allocated.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, int]
        The item and the word of each candidate's token, the token each candidate
        belongs to, numbered in order of first appearance, and how many tokens there
        are.

    Raises
    ------
    ValueError
        If there is not one position per candidate row.
    """
    positions = batch.positions
    rows = len(batch.types.rows)

    if len(positions) != rows:
        msg = f"the batch holds {rows} candidate types at {len(positions)} positions"
        raise ValueError(msg)

    numbered: dict[tuple[int, int], int] = {}
    owners = [numbered.setdefault(position, len(numbered)) for position in positions]
    items = torch.tensor(
        [item for item, _ in positions], dtype=torch.long, device=device
    )
    words = torch.tensor(
        [word for _, word in positions], dtype=torch.long, device=device
    )

    return (
        items,
        words,
        torch.tensor(owners, dtype=torch.long, device=device),
        len(numbered),
    )


class TokenWarmupObjective[BatchT: AmbiguousTokens]:
    """Pulls each token's vector toward the nearest of the types it may take.

    For each token, the objective selects the candidate with minimum mean squared
    distance and adds its distance and negative readout log weight. Tokens without a
    usable candidate are excluded.

    Parameters
    ----------
    tokens : TokenEncoder
        What encodes each item's words to rows.
    pooler : SubwordPooler
        What pools those rows to one vector per word.
    encoder : TermEncoder
        What encodes each candidate type to a vector.
    readout : TeacherForcedDecoder
        What weighs a type against a token's vector.
    index : ChoiceIndex
        The payload-aware choices used to select each supplied type's rows.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_encoder", "_index", "_name", "_pooler", "_readout", "_tokens")

    def __init__(
        self,
        tokens: TokenEncoder,
        pooler: SubwordPooler,
        encoder: TermEncoder,
        readout: TeacherForcedDecoder,
        index: ChoiceIndex,
        *,
        name: str = "token-warmup",
    ) -> None:
        self._tokens = tokens
        self._pooler = pooler
        self._encoder = encoder
        self._readout = readout
        self._index = index
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads, which is the batch."""
        return frozenset({"batch"})

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return the mean least difference plus the mean negative log weight.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the tokens of the least mean squared difference between a
            token's vector and its candidates, plus the mean over the tokens of the
            negative log weight of the selected candidate, counted in the tokens
            scored, reporting the two means detached as ``distance`` and
            ``likelihood``.

        Raises
        ------
        ValueError
            If there is not one position per candidate row, if a position lies beyond
            the items or the words the batch was encoded to, or if the readout does
            not weigh one type per token.
        """
        batch = outputs.batch
        leaves = self._pooler.pool(self._tokens.encode_words(batch.words))
        encoded = self._encoder.encode(batch.types)
        items, words, owners, tokens = _positions(batch, device=leaves.tensor.device)
        held, longest, _ = leaves.tensor.shape

        if bool((items >= held).any() or (words >= longest).any()):
            msg = (
                f"a position lies beyond the {held} items and {longest} words the "
                "batch was encoded to"
            )
            raise ValueError(msg)

        vectors = leaves.tensor[items, words]
        distances = (vectors - encoded.tensor).pow(2).mean(dim=-1)
        live = _live(leaves.mask, items, words, encoded.mask)

        if live is not None:
            distances = distances.masked_fill(~live, torch.inf)

        with torch.no_grad():
            nearest = nearest_candidates(distances, owners=owners, tokens=tokens)

        count = int(nearest.rows.shape[0])

        if count == 0:
            return LossTerm.inapplicable(distances)

        selected = BatchEncoding(
            tuple(batch.types.rows[row] for row in nearest.rows.tolist())
        )
        weights = score(
            self._readout,
            Encoded(vectors[nearest.rows], _TERM_AXES),
            selected,
            index=self._index,
        )
        total, weighed = negative_log_likelihood(
            weights, terms=count, what="the selected types"
        )

        if weighed != count:
            msg = f"the readout weighed {weighed} of the {count} selected types"
            raise ValueError(msg)

        distance = distances[nearest.rows].mean()
        likelihood = total / count

        return LossTerm(
            distance + likelihood,
            count,
            diagnostics={
                "distance": distance.detach(),
                "likelihood": likelihood.detach(),
            },
        )

    @override
    def __repr__(self) -> str:
        """Return the name and the four components' types."""
        return (
            f"TokenWarmupObjective(name={self._name!r}, "
            f"tokens={type(self._tokens).__name__}, "
            f"pooler={type(self._pooler).__name__}, "
            f"encoder={type(self._encoder).__name__}, "
            f"readout={type(self._readout).__name__})"
        )


def _live(
    leaves: Tensor | None, items: Tensor, words: Tensor, types: Tensor | None
) -> Tensor | None:
    """Return which candidates have both a token vector and a type.

    Parameters
    ----------
    leaves : Tensor | None
        Which words hold a vector, over the item and the word, or None where every
        word does.
    items : Tensor
        The item of each candidate's token.
    words : Tensor
        The word of each candidate's token.
    types : Tensor | None
        Which candidate rows hold a type, or None where every row does.

    Returns
    -------
    Tensor | None
        A boolean tensor over the candidates, or None where every candidate is live.
    """
    if leaves is None:
        return types

    encoded = leaves[items, words]

    return encoded if types is None else encoded & types

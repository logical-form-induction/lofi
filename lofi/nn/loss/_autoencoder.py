"""Reconstruction of a type from its own encoding."""

from typing import TYPE_CHECKING, override

from lofi._checks import ensure_name
from lofi.nn.loss._likelihood import negative_log_likelihood
from lofi.nn.loss._ports import LossTerm
from lofi.nn.strategies import exact_matches, score
from lofi.terms import BatchEncoding

if TYPE_CHECKING:
    from lofi.nn.loss._ports import PublishedOutputs, TypeItems
    from lofi.nn.ports import TeacherForcedDecoder, TermEncoder
    from lofi.terms import ChoiceIndex

__all__ = ["TypeAutoencoderObjective"]


class TypeAutoencoderObjective[BatchT: TypeItems]:
    """Scores how poorly a decoder reads each type back from its encoding.

    Each type of the batch is encoded to one vector and weighed by the decoder under
    a pass handed the type itself, so the value at a type is the negative of the log
    weight the decoder gives the type against its own encoding. The term is the mean
    over the rows that hold a type.

    Parameters
    ----------
    encoder : TermEncoder
        What encodes each type to the vector the decoder reads.
    decoder : TeacherForcedDecoder
        What weighs a type against a vector.
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

    __slots__ = ("_decoder", "_encoder", "_index", "_name")

    def __init__(
        self,
        encoder: TermEncoder,
        decoder: TeacherForcedDecoder,
        index: ChoiceIndex,
        *,
        name: str = "type-autoencoder",
    ) -> None:
        self._encoder = encoder
        self._decoder = decoder
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
        """Return the mean negative log probability of the step's types.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the rows holding a type of the type's negative log
            probability, counted in those rows, reporting the same mean per slot of
            the linearization as ``per_slot``.

        Raises
        ------
        ValueError
            If the decoder does not weigh one term per row of the batch.
        """
        supplied = outputs.batch.terms
        terms = BatchEncoding(tuple(row for row in supplied.rows if any(row.occupied)))
        encoded = self._encoder.encode(terms)
        weights = score(self._decoder, encoded, terms, index=self._index)
        total, count = negative_log_likelihood(
            weights, terms=len(terms.rows), what="the types"
        )

        if count == 0:
            return LossTerm.inapplicable(weights.tensor)

        value = total / count
        exact = exact_matches(self._decoder, encoded, terms, index=self._index)
        live = weights.mask

        if live is not None:
            exact = exact & live

        return LossTerm(
            value,
            count,
            diagnostics={
                "exact_match": exact.to(value.dtype).sum() / count,
                "per_slot": value.detach() / supplied.width,
            },
        )

    @override
    def __repr__(self) -> str:
        """Return the name and the two components' types."""
        return (
            f"TypeAutoencoderObjective(name={self._name!r}, "
            f"encoder={type(self._encoder).__name__}, "
            f"decoder={type(self._decoder).__name__})"
        )

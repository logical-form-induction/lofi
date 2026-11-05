"""Neural interfaces to term codecs."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.terms import BatchEncoding, Encoding, OperatorIndex, Term

__all__ = ["CodecMetadata", "TermCoding"]


class CodecMetadata(Protocol):
    """Finite alphabet and layout metadata required by neural tensor views."""

    @property
    def index(self) -> OperatorIndex:
        """Return the operator index the symbol codes are drawn from."""
        ...

    @property
    def n_symbols(self) -> int:
        """Return the alphabet size, padding included."""
        ...

    @property
    def capacity(self) -> int:
        """Return the number of slots every encoding this codec makes occupies."""
        ...

    @property
    def skeleton(self) -> Encoding | None:
        """Return the fixed layout structure, or None for term-specific structure."""
        ...

    @property
    def legal_symbols(self) -> tuple[tuple[bool, ...], ...]:
        """Return which symbols may occupy each slot, one row per slot."""
        ...


class TermCoding(CodecMetadata, Protocol):
    """Encode terms under one finite alphabet and declared layout."""

    def encode_batch(self, terms: Sequence[Term], /) -> BatchEncoding:
        """Encode equal-width terms as one batch."""
        ...

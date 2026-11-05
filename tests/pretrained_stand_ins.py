"""A randomly initialized pretrained-model loader that stands in for a checkpoint.

The loader satisfies the pretrained-loader port the token-encoder plugin
reads, so tests reach the pretrained route without downloading a model.
"""

import importlib
from math import sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override
from zlib import crc32

import torch
from torch import nn

from lofi._checks import ensure_positive
from lofi.nn.tokens.pretrained import (
    EncoderArchitecturePlugin,
    MissingPretrainedBackendError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._artifacts import PinnedCheckpoint
    from lofi.nn.tokens.pretrained import DecoderModel

__all__ = ["StandInLoader"]


class _StandInConfiguration:
    """Width of a stand-in stack, under the name each architecture reads."""

    __slots__ = ("d_model", "hidden_size", "num_hidden_layers", "num_layers")

    def __init__(self, width: int) -> None:
        self.hidden_size = width
        self.d_model = width
        self.num_hidden_layers = 1
        self.num_layers = 1


class _StandInOutput:
    """Rows a stand-in stack returns."""

    __slots__ = ("hidden_states", "last_hidden_state")

    def __init__(
        self, last_hidden_state: Tensor, hidden_states: tuple[Tensor, ...] | None
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _StandInStack(nn.Module):
    """A token embedding followed by one tanh layer."""

    def __init__(
        self, vocabulary_size: int, width: int, generator: torch.Generator
    ) -> None:
        super().__init__()
        self.config = _StandInConfiguration(width)
        self.embedding = nn.Parameter(
            torch.randn(vocabulary_size, width, generator=generator)
        )
        self.weight = nn.Parameter(
            torch.randn(width, width, generator=generator) / sqrt(width)
        )

    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> _StandInOutput:
        """Return contextless rows for the token indices, zero under padding."""
        del return_dict
        embedded = self.embedding[input_ids]
        rows = torch.tanh(embedded @ self.weight)
        mask = attention_mask.unsqueeze(-1).to(rows.dtype)

        return _StandInOutput(
            rows * mask,
            (embedded * mask, rows * mask) if output_hidden_states else None,
        )


class _StandInBatch:
    """Token indices, attention mask and character spans of a stand-in batch."""

    __slots__ = ("attention_mask", "input_ids", "offset_mapping")

    def __init__(
        self, input_ids: Tensor, attention_mask: Tensor, offset_mapping: Tensor
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping


class _WordTokenizer:
    """One token per space-separated word, indexed by a fixed vocabulary.

    The padding and unknown-word indices come first, and the vocabulary words
    take the indices after them in the order given.

    Parameters
    ----------
    words : Sequence[str]
        Words with their own index.

    Raises
    ------
    ValueError
        If ``words`` repeats a word.
    """

    pad_token_id = 0
    unknown_token_id = 1
    model_max_length = 512

    def __init__(self, words: Sequence[str]) -> None:
        if len(set(words)) != len(words):
            msg = "a stand-in vocabulary repeats a word"
            raise ValueError(msg)

        first = self.unknown_token_id + 1
        self._vocabulary: Mapping[str, int] = MappingProxyType(
            {word: first + index for index, word in enumerate(words)}
        )
        self.size = first + len(words)

    @property
    def all_special_ids(self) -> list[int]:
        """Return the padding and unknown-word indices."""
        return [self.pad_token_id, self.unknown_token_id]

    def get_vocab(self) -> dict[str, int]:
        """Return the index of every token, the padding and unknown-word ones first."""
        return {
            "<pad>": self.pad_token_id,
            "<unk>": self.unknown_token_id,
            **self._vocabulary,
        }

    def __len__(self) -> int:
        """Return how many indices the tokenizer assigns."""
        return self.size

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> _StandInBatch:
        """Return the padded word indices and word spans of a batch of texts.

        Parameters
        ----------
        texts : Sequence[str]
            Each item's words, separated by single spaces.
        padding : bool
            Accepted for the tokenizer interface; batches are always padded.
        return_tensors : str
            Accepted for the tokenizer interface; tensors are always torch.
        truncation : bool
            Accepted for the tokenizer interface; items are never truncated.
        return_offsets_mapping : bool
            Accepted for the tokenizer interface; spans are always returned.
        """
        del padding, return_tensors, truncation, return_offsets_mapping
        items = tuple(text.split(" ") if text else () for text in texts)
        longest = max(map(len, items), default=0)
        ids = torch.full((len(items), longest), self.pad_token_id, dtype=torch.long)
        offsets = torch.zeros((len(items), longest, 2), dtype=torch.long)

        for item, words in enumerate(items):
            at = 0

            for position, word in enumerate(words):
                ids[item, position] = self._vocabulary.get(word, self.unknown_token_id)
                offsets[item, position, 0] = at
                offsets[item, position, 1] = at + len(word)
                at += len(word) + 1

        mask = ids.ne(self.pad_token_id).to(torch.long)

        return _StandInBatch(ids, mask, offsets)


class StandInLoader:
    """Loads a small randomly initialized stack in place of a checkpoint.

    The encoder stack embeds each word of a fixed vocabulary and applies one
    tanh layer. Its configuration reports the declared width as both
    ``hidden_size`` and ``d_model``, so it stands in for a stack of either
    architecture. The decoder language model is a two-layer, single-head Llama
    decoder of the declared width over the same vocabulary, with its output
    head. The initial weights depend on the loader seed and on the checkpoint
    identifier, so two candidates start from different stand-ins and one
    candidate always starts from the same one.

    Parameters
    ----------
    vocabulary : Sequence[str]
        Words with their own embedding row.
    width : int
        Output width of the stack.
    seed : int
        Seed mixed with the checkpoint identifier to initialize the weights.

    Raises
    ------
    TypeError
        If ``width`` is not an integer.
    ValueError
        If ``width`` is not positive or the vocabulary repeats a word.
    """

    __slots__ = ("_seed", "_tokenizer", "_width")

    def __init__(self, vocabulary: Sequence[str], *, width: int, seed: int) -> None:
        self._tokenizer = _WordTokenizer(vocabulary)
        self._width = ensure_positive("a stand-in width", width)
        self._seed = seed

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> _StandInStack:
        """Return a freshly initialized stand-in stack for a checkpoint.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release the stack stands in for.
        architecture : EncoderArchitecturePlugin
            The stack of the release it stands in for, which the stand-in's
            configuration serves either way.
        dtype : torch.dtype
            The precision the parameters are held in.
        """
        del architecture
        generator = torch.Generator().manual_seed(
            crc32(f"{self._seed}:{checkpoint.identity}".encode())
        )

        return _StandInStack(self._tokenizer.size, self._width, generator).to(
            dtype=dtype
        )

    def load_decoder(
        self, checkpoint: PinnedCheckpoint, *, dtype: torch.dtype
    ) -> DecoderModel:
        """Return a freshly initialized stand-in decoder language model.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release the model stands in for.
        dtype : torch.dtype
            The precision the parameters are held in.

        Raises
        ------
        ValueError
            If the declared width is odd, which rotary positions cannot split.
        MissingPretrainedBackendError
            If Transformers is not installed.
        """
        if self._width % 2:
            msg = f"a stand-in decoder's width is even, not {self._width}"
            raise ValueError(msg)

        name = "transformers"

        try:
            backend = importlib.import_module(name)

        except ModuleNotFoundError as error:
            raise MissingPretrainedBackendError(name) from error

        configuration = backend.LlamaConfig(
            vocab_size=self._tokenizer.size,
            hidden_size=self._width,
            intermediate_size=2 * self._width,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=self._width,
            pad_token_id=self._tokenizer.pad_token_id,
            bos_token_id=None,
            eos_token_id=None,
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(crc32(f"{self._seed}:{checkpoint.identity}".encode()))
            model = cast(
                "nn.Module", backend.AutoModelForCausalLM.from_config(configuration)
            )

        return cast("DecoderModel", model.to(dtype=dtype))

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> _WordTokenizer:
        """Return the word-level tokenizer over the stand-in vocabulary.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release the tokenizer stands in for.
        """
        del checkpoint

        return self._tokenizer

    @override
    def __repr__(self) -> str:
        """Return a summary naming the vocabulary size, width and seed."""
        return (
            f"StandInLoader(<{self._tokenizer.size} indices>, width={self._width}, "
            f"seed={self._seed})"
        )

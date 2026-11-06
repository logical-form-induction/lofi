"""Pretrained multilingual token encoders."""

import importlib
from bisect import bisect_left, bisect_right
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Protocol,
    assert_never,
    cast,
    final,
    override,
    runtime_checkable,
)

import torch
from torch import nn

from lofi._alignment import Alignment
from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.nn._values import Encoded, TokenEncoding
from lofi.nn.policy import place
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.nn.tokens.window import (
    ContextOverflowError,
    ContextWindow,
    overflow_policy,
)
from lofi.registry import ComponentPlugin, Registry, RegistryError
from lofi.spec.tokens.pretrained import (
    ContextWindowSettings,
    FixedLayerReadoutSettings,
    LastLayerReadoutSettings,
    LayerReadoutSettings,
    PretrainedTokenEncoderSettings,
    PretrainedTokenEncoderSpec,
    ScalarMixReadoutSettings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from torch import Tensor

    from lofi._artifacts import ArtifactFetcher, PinnedCheckpoint
    from lofi.nn.build import BuildContext
    from lofi.nn.sites import ComponentId

__all__ = [
    "ENCODER_ONLY",
    "T5_ENCODER",
    "AbsentModelError",
    "CausalLoader",
    "CausalModel",
    "CausalOutput",
    "CheckpointAligner",
    "CrossingPieceError",
    "DecoderLoader",
    "DecoderModel",
    "DecoderTokenizer",
    "EncoderArchitecturePlugin",
    "HuggingFaceCausalLoader",
    "HuggingFacePretrainedLoader",
    "LocalPretrainedLoader",
    "MissingPretrainedBackendError",
    "PretrainedLoader",
    "PretrainedTokenEncoder",
    "SentenceTokenizer",
    "WordRows",
    "ZeroPieceWordError",
    "declared_window",
    "encoder_architecture",
    "load_aligner",
    "load_pretrained",
]


_ROW_AXES = ("item", "row", "width")
_MAX_PUBLISHED_CONTEXT: Final = 1_000_000_000
"""Values at or above this bound are tokenizer sentinels, not model capabilities."""


class _Configuration(Protocol):
    """A checkpoint configuration, read only by the model class it builds."""


class _HiddenSizeConfiguration(Protocol):
    """Configuration with an encoder hidden width."""

    hidden_size: int


class _DecoderConfiguration(_HiddenSizeConfiguration, Protocol):
    """Decoder configuration naming its width and hidden-transform count."""

    num_hidden_layers: int


class _T5Configuration(Protocol):
    """T5-family configuration with its model width."""

    d_model: int


class _EncoderOutput(Protocol):
    """Output from a pretrained encoder stack."""

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None


class _EncoderStack(Protocol):
    """Pretrained encoder operations used by the adapter."""

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over the encoder parameters."""
        ...

    def __call__(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> _EncoderOutput:
        """Encode token indices under an attention mask."""
        ...


class _HiddenSizeEncoder(_EncoderStack, Protocol):
    """Encoder whose configuration names ``hidden_size``."""

    @property
    def config(self) -> _HiddenSizeConfiguration:
        """Return the checkpoint configuration."""
        ...


class _T5Encoder(_EncoderStack, Protocol):
    """T5-family encoder stack whose configuration names ``d_model``."""

    @property
    def config(self) -> _T5Configuration:
        """Return the checkpoint configuration."""
        ...


class _TokenizedBatch(Protocol):
    """Tokenized batch fields used by the adapter."""

    input_ids: Tensor
    attention_mask: Tensor
    offset_mapping: Tensor


class _Tokenizer(Protocol):
    """Fast tokenizer operations used by the adapter."""

    pad_token_id: int
    model_max_length: int

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> _TokenizedBatch:
        """Tokenize a batch of texts, recording each row's character span."""
        ...


class CausalOutput(Protocol):
    """Output of a causal language model's forward pass."""

    @property
    def logits(self) -> Tensor:
        """Return next-token logits over the vocabulary at every position."""
        ...


class CausalModel(Protocol):
    """Causal language model operations a sentence scorer uses."""

    def eval(self) -> object:
        """Put the model in evaluation mode."""
        ...

    def __call__(self, *, input_ids: Tensor) -> CausalOutput:
        """Return the next-token logits of a batch of token indices."""
        ...


class SentenceTokenizer(Protocol):
    """Tokenizer operations a sentence scorer uses."""

    @property
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token index, or None if there is none."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Return the token indices of a text."""
        ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Return the token strings of token indices."""
        ...


class DecoderTokenizer(_Tokenizer, Protocol):
    """A word-aligned tokenizer that also names its vocabulary and special tokens."""

    @property
    def all_special_ids(self) -> list[int]:
        """Return the indices of every special token."""
        ...

    def get_vocab(self) -> dict[str, int]:
        """Return the index of every token string."""
        ...

    def __len__(self) -> int:
        """Return how many token indices the tokenizer assigns."""
        ...


class DecoderModel(CausalModel, Protocol):
    """A causal language model whose decoder stack and output head run apart."""

    @property
    def config(self) -> _DecoderConfiguration:
        """Return the checkpoint configuration."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over the model parameters."""
        ...

    def modules(self) -> Iterator[nn.Module]:
        """Iterate over the model and every module inside it."""
        ...

    def get_decoder(self) -> _EncoderStack:
        """Return the decoder stack, which returns rows without the output head."""
        ...

    def get_output_embeddings(self) -> nn.Module:
        """Return the output head mapping rows to next-token logits."""
        ...


class _HuggingFaceTokenizer(DecoderTokenizer, SentenceTokenizer, Protocol):
    """A fast tokenizer read either as word-aligned text or as running text."""

    eos_token_id: int | None


class PretrainedLoader(Protocol):
    """Loads an encoder stack and its tokenizer from one pinned checkpoint."""

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> _EncoderStack:
        """Load the stack of a checkpoint an architecture runs.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        architecture : EncoderArchitecturePlugin
            Which stack of the release to load.
        dtype : torch.dtype
            The precision the parameters are held in.
        """
        ...

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> _Tokenizer:
        """Load the fast tokenizer with word alignment.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        ...


@runtime_checkable
class DecoderLoader(Protocol):
    """Loads a decoder language model and its tokenizer from one pinned checkpoint."""

    def load_decoder(
        self, checkpoint: PinnedCheckpoint, *, dtype: torch.dtype
    ) -> DecoderModel:
        """Load the causal language model with its output head.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        dtype : torch.dtype
            The precision the parameters are held in.
        """
        ...

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> DecoderTokenizer:
        """Load the fast tokenizer with word alignment.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        ...


class CausalLoader(Protocol):
    """Loads a causal language model and its tokenizer from one pinned checkpoint."""

    @property
    def dtype(self) -> torch.dtype:
        """Return the precision the loaded model's parameters are held in."""
        ...

    def load_causal_model(self, checkpoint: PinnedCheckpoint) -> CausalModel:
        """Load the causal language model with its output head.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        ...

    def load_sentence_tokenizer(
        self, checkpoint: PinnedCheckpoint
    ) -> SentenceTokenizer:
        """Load the tokenizer that reads running text.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        ...


class MissingPretrainedBackendError(ImportError):
    """Unavailable pretrained-model client."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        message = (
            f"pretrained models, encoder-only or causal, require the {backend!r} "
            "package; install lofi[nn]"
        )
        super().__init__(message)


class AbsentModelError(FileNotFoundError):
    """Reports a pinned checkpoint the backend could not load.

    Parameters
    ----------
    checkpoint : PinnedCheckpoint
        The checkpoint that could not be loaded.
    reason : str
        What the backend reported.
    """

    __slots__ = ("checkpoint",)

    def __init__(self, checkpoint: PinnedCheckpoint, reason: str) -> None:
        self.checkpoint = checkpoint

        super().__init__(
            f"the checkpoint {checkpoint.identity} could not be loaded: {reason}"
        )


class _PinnedFactory[LoadedT](Protocol):
    """Hugging Face factory loading one repository at one commit."""

    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        dtype: torch.dtype,
    ) -> LoadedT:
        """Load one fetched release in a precision."""
        ...


class _TokenizerFactory(Protocol):
    """Hugging Face tokenizer factory."""

    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        use_fast: bool,
    ) -> _HuggingFaceTokenizer:
        """Load one tokenizer from a local directory."""
        ...


class _ConfigurationFactory(Protocol):
    """Hugging Face configuration factory."""

    def from_pretrained(
        self, checkpoint: str, *, local_files_only: bool
    ) -> _Configuration:
        """Load one model configuration."""


class _ConfiguredEncoderFactory(_PinnedFactory[_EncoderStack], Protocol):
    """Hugging Face encoder factory supporting weight-free construction."""

    def from_config(
        self, configuration: _Configuration, *, dtype: torch.dtype
    ) -> _EncoderStack:
        """Construct an encoder in a precision from configuration only."""


class _ConfiguredDecoderFactory(_PinnedFactory[DecoderModel], Protocol):
    """Hugging Face causal-model factory supporting weight-free construction."""

    def from_config(
        self, configuration: _Configuration, *, dtype: torch.dtype
    ) -> DecoderModel:
        """Construct a causal language model in a precision from configuration."""


class _EncoderModelClass(_PinnedFactory[_EncoderStack], Protocol):
    """Hugging Face model class whose constructor builds from configuration only."""

    def __call__(self, configuration: _Configuration) -> nn.Module:
        """Construct an uninitialized model from a configuration."""
        ...


class _Transformers(Protocol):
    """Hugging Face factories used by the loaders."""

    AutoConfig: _ConfigurationFactory
    AutoModel: _ConfiguredEncoderFactory
    AutoModelForCausalLM: _ConfiguredDecoderFactory
    AutoTokenizer: _TokenizerFactory
    MT5EncoderModel: _EncoderModelClass


class _FactorySelector(Protocol):
    """Select the checkpoint factory for one encoder architecture."""

    def __call__(self, backend: _Transformers, /) -> _PinnedFactory[_EncoderStack]:
        """Return the factory that loads this architecture."""
        ...


class _ArchitectureConstructor(Protocol):
    """Build one encoder architecture from local configuration."""

    def __call__(
        self,
        backend: _Transformers,
        configuration: _Configuration,
        dtype: torch.dtype,
        /,
    ) -> _EncoderStack:
        """Return an uninitialized stack in the requested precision."""
        ...


class _WidthReader(Protocol):
    """Read the output width of one encoder architecture."""

    def __call__(self, stack: _EncoderStack, /) -> int:
        """Return the positive output width declared by ``stack``."""
        ...


class _LayerCountReader(Protocol):
    """Read the number of hidden transforms in one encoder architecture."""

    def __call__(self, stack: _EncoderStack, /) -> int:
        """Return the nonnegative hidden-transform count."""
        ...


class EncoderArchitecturePlugin(FrozenValue):
    """Load, construct and measure one pretrained encoder architecture."""

    __slots__ = ("_construct", "_factory", "_layers", "_width")
    _noun = "encoder-architecture plugin"

    def __init__(
        self,
        *,
        factory: _FactorySelector,
        construct: _ArchitectureConstructor,
        width: _WidthReader,
        layers: _LayerCountReader,
    ) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_construct", construct)
        object.__setattr__(self, "_width", width)
        object.__setattr__(self, "_layers", layers)

    def factory(self, backend: _Transformers, /) -> _PinnedFactory[_EncoderStack]:
        """Return the checkpoint factory this architecture loads through."""
        return self._factory(backend)

    def construct(
        self,
        backend: _Transformers,
        configuration: _Configuration,
        dtype: torch.dtype,
        /,
    ) -> _EncoderStack:
        """Return an uninitialized stack in the requested precision."""
        return self._construct(backend, configuration, dtype)

    def output_width(self, stack: _EncoderStack, /) -> int:
        """Return the positive output width declared by ``stack``."""
        return self._width(stack)

    def layer_count(self, stack: _EncoderStack, /) -> int:
        """Return the count of hidden transforms in ``stack``."""
        return self._layers(stack)


def encoder_architecture(name: str, /) -> EncoderArchitecturePlugin:
    """Return the pretrained encoder architecture installed under ``name``."""
    member = Registry[object]("encoder_architecture").get(name)

    if not isinstance(member, EncoderArchitecturePlugin):
        msg = (
            f"the encoder_architecture entry {name!r} names {member!r}, "
            "not an encoder-architecture plugin"
        )
        raise RegistryError(msg, family="encoder_architecture")

    return member


def _encoder_only_factory(backend: _Transformers, /) -> _PinnedFactory[_EncoderStack]:
    """Return the backend's whole-encoder factory."""
    return backend.AutoModel


def _t5_encoder_factory(backend: _Transformers, /) -> _PinnedFactory[_EncoderStack]:
    """Return the backend's T5-family encoder-only factory."""
    return backend.MT5EncoderModel


def _construct_encoder_only(
    backend: _Transformers,
    configuration: _Configuration,
    dtype: torch.dtype,
    /,
) -> _EncoderStack:
    """Construct a whole encoder from local configuration."""
    return backend.AutoModel.from_config(configuration, dtype=dtype)


def _construct_t5_encoder(
    backend: _Transformers,
    configuration: _Configuration,
    dtype: torch.dtype,
    /,
) -> _EncoderStack:
    """Construct a T5-family encoder stack from local configuration."""
    stack = backend.MT5EncoderModel(configuration).to(dtype=dtype)
    return cast("_EncoderStack", stack)


def _hidden_size(stack: _EncoderStack, /) -> int:
    """Return ``config.hidden_size`` from a whole encoder stack."""
    configuration = cast("_HiddenSizeEncoder", stack).config
    return _width("config.hidden_size", configuration.hidden_size)


def _t5_width(stack: _EncoderStack, /) -> int:
    """Return ``config.d_model`` from a T5-family encoder stack."""
    configuration = cast("_T5Encoder", stack).config
    return _width("config.d_model", configuration.d_model)


class _EncoderOnlyLayers(Protocol):
    """An encoder-only stack configuration naming its hidden-layer count."""

    num_hidden_layers: int


class _T5Layers(Protocol):
    """A T5 configuration naming its encoder-layer count."""

    num_layers: int


def _encoder_only_layers(stack: _EncoderStack, /) -> int:
    """Return ``config.num_hidden_layers`` from an encoder-only stack."""
    configuration = cast("_HiddenSizeEncoder", stack).config
    return ensure_non_negative(
        "config.num_hidden_layers",
        cast("_EncoderOnlyLayers", configuration).num_hidden_layers,
    )


def _t5_layers(stack: _EncoderStack, /) -> int:
    """Return ``config.num_layers`` from a T5-family encoder stack."""
    configuration = cast("_T5Encoder", stack).config
    return ensure_non_negative(
        "config.num_layers", cast("_T5Layers", configuration).num_layers
    )


ENCODER_ONLY: Final = EncoderArchitecturePlugin(
    factory=_encoder_only_factory,
    construct=_construct_encoder_only,
    width=_hidden_size,
    layers=_encoder_only_layers,
)
T5_ENCODER: Final = EncoderArchitecturePlugin(
    factory=_t5_encoder_factory,
    construct=_construct_t5_encoder,
    width=_t5_width,
    layers=_t5_layers,
)


def _padded(tokenizer: _HuggingFaceTokenizer, /) -> _HuggingFaceTokenizer:
    """Return a tokenizer that pads, with its end-of-sequence token if it has no pad.

    A decoder language model's tokenizer often declares no padding token, and a
    padded batch needs one; the rows it fills are masked out wherever they are read.

    Parameters
    ----------
    tokenizer : _HuggingFaceTokenizer
        The loaded tokenizer, which is given a padding token in place if it lacks
        one and has an end-of-sequence token.
    """
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


class _HuggingFaceLoader:
    """The installed Transformers module every Hugging Face loader reaches."""

    def _backend(self) -> _Transformers:
        """Return the installed Transformers module.

        Raises
        ------
        MissingPretrainedBackendError
            If Transformers is not installed.
        """
        backend = "transformers"

        try:
            loaded = importlib.import_module(backend)

        except ModuleNotFoundError as error:
            raise MissingPretrainedBackendError(backend) from error

        return cast("_Transformers", loaded)


class _FetchedLoader(_HuggingFaceLoader):
    """Loads what a pinned release holds from the directory a fetcher returns.

    Every release is fetched through the one artifact fetcher, and the backend
    reads only the fetched directory, so no download bypasses the fetcher.

    Parameters
    ----------
    fetcher : ArtifactFetcher
        The fetcher every release is materialized through.
    """

    def __init__(self, fetcher: ArtifactFetcher, /) -> None:
        self._fetcher = fetcher

    def _directory(self, checkpoint: PinnedCheckpoint, /) -> str:
        """Return the local directory the fetcher materializes a release in.

        Raises
        ------
        AbsentModelError
            If the release cannot be fetched.
        """
        try:
            return str(self._fetcher.fetch(checkpoint))

        except OSError as error:
            raise AbsentModelError(checkpoint, str(error)) from error

    def _pinned[LoadedT](
        self,
        factory: _PinnedFactory[LoadedT],
        checkpoint: PinnedCheckpoint,
        dtype: torch.dtype,
    ) -> LoadedT:
        """Return what a factory loads from a fetched release in a precision.

        The precision is passed to the factory explicitly, so a checkpoint whose
        configuration declares another precision is still loaded in it.

        Raises
        ------
        AbsentModelError
            If the release cannot be fetched or the backend cannot load it.
        """
        directory = self._directory(checkpoint)

        try:
            return factory.from_pretrained(
                directory, local_files_only=True, dtype=dtype
            )

        except OSError as error:
            raise AbsentModelError(checkpoint, str(error)) from error

    def _tokenizer(self, checkpoint: PinnedCheckpoint) -> _HuggingFaceTokenizer:
        """Return the fast tokenizer of a fetched release.

        Raises
        ------
        AbsentModelError
            If the release cannot be fetched or the backend cannot load it.
        """
        directory = self._directory(checkpoint)

        try:
            loaded = self._backend().AutoTokenizer.from_pretrained(
                directory, local_files_only=True, use_fast=True
            )

        except OSError as error:
            raise AbsentModelError(checkpoint, str(error)) from error

        return _padded(loaded)

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> DecoderTokenizer:
        """Load a fast tokenizer with word alignment.

        A tokenizer declaring no padding token pads with its end-of-sequence token.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        return self._tokenizer(checkpoint)


class HuggingFacePretrainedLoader(_FetchedLoader):
    """Loads encoder stacks through Hugging Face.

    An encoder-only checkpoint loads through ``AutoModel``, and the encoder stack
    of a T5-family checkpoint through ``MT5EncoderModel``, which holds no decoder.

    Parameters
    ----------
    fetcher : ArtifactFetcher
        The fetcher every release is materialized through.
    """

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> _EncoderStack:
        """Load the stack of a checkpoint an architecture runs.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        architecture : EncoderArchitecturePlugin
            Which stack of the release to load.
        dtype : torch.dtype
            The precision the parameters are held in.
        """
        backend = self._backend()
        return self._pinned(architecture.factory(backend), checkpoint, dtype)

    def load_decoder(
        self, checkpoint: PinnedCheckpoint, *, dtype: torch.dtype
    ) -> DecoderModel:
        """Load a causal language model with its output head at its commit.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        dtype : torch.dtype
            The precision the parameters are held in.

        Raises
        ------
        AbsentModelError
            If the backend cannot load the checkpoint at its commit.
        """
        return self._pinned(self._backend().AutoModelForCausalLM, checkpoint, dtype)


class HuggingFaceCausalLoader(_FetchedLoader):
    """Loads causal language models with their output head, in single precision.

    Parameters
    ----------
    fetcher : ArtifactFetcher
        The fetcher every release is materialized through.
    """

    @property
    def dtype(self) -> torch.dtype:
        """Return the precision every model is loaded in."""
        return torch.float32

    def load_causal_model(self, checkpoint: PinnedCheckpoint) -> CausalModel:
        """Load a causal language model.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        return self._pinned(
            self._backend().AutoModelForCausalLM, checkpoint, self.dtype
        )

    def load_sentence_tokenizer(
        self, checkpoint: PinnedCheckpoint
    ) -> SentenceTokenizer:
        """Load the tokenizer that reads running text.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release to load.
        """
        return self._tokenizer(checkpoint)


class LocalPretrainedLoader(_HuggingFaceLoader):
    """Builds an encoder from verified local configuration and tokenizer assets.

    Parameters
    ----------
    source : Path
        Directory containing configuration and tokenizer files, but no weights.
    checkpoint : PinnedCheckpoint
        The release the run declaration must present when using these assets.
    """

    def __init__(self, source: Path, checkpoint: PinnedCheckpoint) -> None:
        self._source = source
        self._checkpoint = checkpoint

    def _require_checkpoint(self, checkpoint: PinnedCheckpoint, /) -> str:
        """Return the local source after checking the declared release."""
        if checkpoint != self._checkpoint:
            msg = (
                f"local construction resources are for checkpoint "
                f"{self._checkpoint.identity!r}, not {checkpoint.identity!r}"
            )
            raise ValueError(msg)

        return str(self._source)

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> _EncoderStack:
        """Construct an uninitialized encoder stack from local configuration.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release the run declares, which must be the one the assets are for.
        architecture : EncoderArchitecturePlugin
            Which stack of the release to construct.
        dtype : torch.dtype
            The precision the parameters are held in.

        Raises
        ------
        ValueError
            If the declared release is not the one the assets are for.
        """
        source = self._require_checkpoint(checkpoint)
        backend = self._backend()
        configuration = backend.AutoConfig.from_pretrained(
            source, local_files_only=True
        )

        return architecture.construct(backend, configuration, dtype)

    def load_decoder(
        self, checkpoint: PinnedCheckpoint, *, dtype: torch.dtype
    ) -> DecoderModel:
        """Construct an uninitialized causal language model from local configuration.

        Parameters
        ----------
        checkpoint : PinnedCheckpoint
            The release the run declares, which must be the one the assets are for.
        dtype : torch.dtype
            The precision the parameters are held in.

        Raises
        ------
        ValueError
            If the declared release is not the one the assets are for.
        """
        source = self._require_checkpoint(checkpoint)
        backend = self._backend()
        configuration = backend.AutoConfig.from_pretrained(
            source, local_files_only=True
        )

        return backend.AutoModelForCausalLM.from_config(configuration, dtype=dtype)

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> DecoderTokenizer:
        """Load a tokenizer from the local construction assets.

        A tokenizer declaring no padding token pads with its end-of-sequence token.
        """
        source = self._require_checkpoint(checkpoint)

        return _padded(
            self._backend().AutoTokenizer.from_pretrained(
                source, local_files_only=True, use_fast=True
            )
        )


def _width(name: str, value: int) -> int:
    """Return a positive checkpoint width.

    Parameters
    ----------
    name : str
        The configuration field.
    value : int
        The field value.

    Raises
    ------
    TypeError
        If the width is not an integer.
    ValueError
        If the width is not positive.
    """
    checked = ensure_non_negative(name, value)

    if checked == 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)

    return checked


class ZeroPieceWordError(ValueError):
    """Reports words a tokenizer produced no row for.

    Parameters
    ----------
    positions : tuple[int, ...]
        The positions of the words with no row, in word order.
    """

    __slots__ = ("_positions",)

    def __init__(self, positions: tuple[int, ...]) -> None:
        super().__init__(f"the tokenizer produced no row for the words at {positions}")

        self._positions = positions

    @property
    def positions(self) -> tuple[int, ...]:
        """Return the positions of the words with no row."""
        return self._positions


class CrossingPieceError(ValueError):
    """Reports a token row whose character span crosses two words.

    Parameters
    ----------
    text : str
        The joined text the tokenizer read.
    characters : tuple[int, int]
        The row's start and end character in that text.
    words : tuple[int, int]
        The positions of the first and last word the row covers.
    """

    __slots__ = ("_characters", "_words")

    def __init__(
        self, text: str, characters: tuple[int, int], words: tuple[int, int]
    ) -> None:
        start, end = characters
        first, last = words

        super().__init__(
            f"a token row covering characters {start} to {end} of {text!r} "
            f"crosses the words at {first} and {last}"
        )

        self._characters = characters
        self._words = words

    @property
    def characters(self) -> tuple[int, int]:
        """Return the row's start and end character."""
        return self._characters

    @property
    def words(self) -> tuple[int, int]:
        """Return the first and last word the row covers."""
        return self._words


def _text(words: Sequence[str], join: str = " ") -> str:
    """Return an item's words as the joined text a tokenizer reads.

    Parameters
    ----------
    words : Sequence[str]
        The input words.
    join : str
        The string the words are joined by.
    """
    return join.join(words)


def _word_ids(
    words: Sequence[str], offsets: Sequence[Sequence[int]], *, join: str = " "
) -> tuple[int | None, ...]:
    """Return the input word each token row covers, read off its character span.

    A row with an empty span is a special or padding row and covers no word. A row
    covering only the joining string before a word belongs to that word, as a byte-level
    tokenizer's word-initial marker does.

    Parameters
    ----------
    words : Sequence[str]
        The input words, joined by ``join`` into the tokenized text.
    offsets : Sequence[Sequence[int]]
        The start and end character of each token row in that text.
    join : str
        The string the words are joined by.

    Returns
    -------
    tuple[int | None, ...]
        One input-word index, or None for a row covering no word, per token row.

    Raises
    ------
    CrossingPieceError
        If a row's span crosses the boundary between two words.
    """
    starts: list[int] = []
    ends: list[int] = []
    at = 0

    for word in words:
        starts.append(at)
        ends.append(at + len(word))
        at += len(word) + len(join)

    covered: list[int | None] = []

    for start, end in offsets:
        if start == end:
            covered.append(None)
            continue

        first = bisect_right(ends, start)
        last = bisect_left(starts, end) - 1

        if first > last:
            covered.append(first if first < len(words) else None)
        elif first == last:
            covered.append(first)
        else:
            raise CrossingPieceError(_text(words, join), (start, end), (first, last))

    return tuple(covered)


def _positions(
    word_ids: Sequence[int | None],
    live: Sequence[int],
    word_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the live token rows assigned to each input word.

    Parameters
    ----------
    word_ids : Sequence[int | None]
        One input-word index or special-token marker per token row.
    live : Sequence[int]
        The unpadded token rows.
    word_count : int
        The number of input words.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Token rows grouped by input word.

    Raises
    ------
    ZeroPieceWordError
        If a word has no token row.
    """
    grouped: list[list[int]] = [[] for _ in range(word_count)]

    for row in live:
        word = word_ids[row]

        if word is not None:
            grouped[word].append(row)

    missing = tuple(word for word, rows in enumerate(grouped) if not rows)

    if missing:
        raise ZeroPieceWordError(missing)

    return tuple(tuple(rows) for rows in grouped)


def _trimmed(
    words: Sequence[str],
    word_ids: Sequence[int | None],
    live: Sequence[int],
    *,
    window: ContextWindow,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...] | None, ...]]:
    """Return selected token rows and each word's rows after packing.

    Parameters
    ----------
    words : Sequence[str]
        The input words.
    word_ids : Sequence[int | None]
        One input-word index or special-token marker per token row.
    live : Sequence[int]
        The unpadded token rows.
    window : ContextWindow
        The row budget and overflow policy.

    Returns
    -------
    tuple[tuple[int, ...], tuple[tuple[int, ...] | None, ...]]
        The selected source rows, and each word's packed rows in order, or None
        for a word the window dropped.

    Raises
    ------
    ContextOverflowError
        If the item exceeds a window that refuses overflow.
    """
    grouped = _positions(word_ids, live, len(words))

    if len(live) <= window.rows:
        selected = tuple(live)
        kept = len(words)
    else:
        specials = tuple(row for row in live if word_ids[row] is None)
        used = len(specials)
        kept = 0

        if used <= window.rows:
            for rows in grouped:
                if used + len(rows) > window.rows:
                    break

                used += len(rows)
                kept += 1

            selected = tuple(
                row
                for row in live
                if word_ids[row] is None or cast("int", word_ids[row]) < kept
            )
        else:
            selected = specials[: window.rows]

        window.resolve(words, needed=len(live), kept=kept)

    remapped = {source: target for target, source in enumerate(selected)}
    packed = tuple(
        tuple(remapped[row] for row in rows) if word < kept else None
        for word, rows in enumerate(grouped)
    )

    return selected, packed


def _interval(rows: tuple[int, ...] | None) -> tuple[int, int] | None:
    """Return the half-open interval from a word's first row to its last."""
    return None if rows is None else (rows[0], rows[-1] + 1)


def _refuse_blank(items: Sequence[Sequence[str]]) -> None:
    """Refuse an item holding a word with no character other than whitespace.

    Raises
    ------
    ValueError
        If a word is empty or only whitespace.
    """
    for item, words in enumerate(items):
        for position, word in enumerate(words):
            if not word.strip():
                msg = (
                    f"the word at {position} of item {item} is {word!r}; an "
                    "input word needs a character other than whitespace"
                )
                raise ValueError(msg)


def _tokenized(
    tokenizer: _Tokenizer, items: Sequence[Sequence[str]], join: str
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the token indices, attention mask and character offsets of items.

    Raises
    ------
    ValueError
        If the tokenizer returns inconsistent dimensions.
    """
    tokenized = tokenizer(
        [_text(words, join) for words in items],
        padding=True,
        return_tensors="pt",
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = tokenized.input_ids
    attention = tokenized.attention_mask
    offsets = tokenized.offset_mapping

    if input_ids.ndim != 2 or attention.shape != input_ids.shape:
        msg = (
            "a tokenizer returns rank-2 input ids and an attention mask of the "
            f"same shape, not {tuple(input_ids.shape)} and {tuple(attention.shape)}"
        )
        raise ValueError(msg)

    if input_ids.shape[0] != len(items):
        msg = (
            f"a tokenizer received {len(items)} items and returned {input_ids.shape[0]}"
        )
        raise ValueError(msg)

    if tuple(offsets.shape) != (*input_ids.shape, 2):
        msg = (
            "a tokenizer returns one start and end character per token row, "
            f"not offsets of shape {tuple(offsets.shape)} for input ids of "
            f"shape {tuple(input_ids.shape)}"
        )
        raise ValueError(msg)

    return input_ids, attention, offsets


def _live(attention: Tensor, item: int) -> tuple[int, ...]:
    """Return the unpadded token rows of one item."""
    return tuple(row for row in range(attention.shape[1]) if bool(attention[item, row]))


def _output_width(stack: _EncoderStack, architecture: EncoderArchitecturePlugin) -> int:
    """Return the width a stack's configuration declares under its architecture.

    Parameters
    ----------
    stack : _EncoderStack
        The loaded stack.
    architecture : EncoderArchitecturePlugin
        Which stack it is, and so which configuration field names its width.

    Raises
    ------
    TypeError
        If the width is not an integer.
    ValueError
        If the width is not positive.
    """
    return architecture.output_width(stack)


class _LastLayerReadout(nn.Module):
    """Read the final hidden layer directly."""

    @property
    def requires_hidden_states(self) -> bool:
        """Return false because the ordinary output already holds the final rows."""
        return False

    def rows(self, output: _EncoderOutput, /) -> Tensor:
        """Return the stack's final rows."""
        return output.last_hidden_state


class _FixedLayerReadout(nn.Module):
    """Read one declared position of the stack's hidden state tuple."""

    def __init__(self, layer: int, *, count: int) -> None:
        super().__init__()

        if layer >= count:
            msg = f"fixed layer {layer} is outside the checkpoint's {count} states"
            raise ValueError(msg)

        self._layer = layer

    @property
    def requires_hidden_states(self) -> bool:
        """Return true because the selected layer may not be final."""
        return True

    def rows(self, output: _EncoderOutput, /) -> Tensor:
        """Return the declared hidden state tuple position."""
        hidden = output.hidden_states

        if hidden is None:
            msg = "a fixed-layer readout requested hidden states but received none"
            raise ValueError(msg)

        return hidden[self._layer]


class _ScalarMixReadout(nn.Module):
    """Learn a normalized scalar mixture over the stack's hidden states."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(count, dtype=torch.float32))

    @property
    def requires_hidden_states(self) -> bool:
        """Return true because every hidden state enters the mixture."""
        return True

    def rows(self, output: _EncoderOutput, /) -> Tensor:
        """Return the trainable convex mixture of hidden states."""
        hidden = output.hidden_states

        if hidden is None:
            msg = "a scalar-mix readout requested hidden states but received none"
            raise ValueError(msg)

        if len(hidden) != self.logits.numel():
            msg = (
                f"the checkpoint declared {self.logits.numel()} hidden states "
                f"but returned {len(hidden)}"
            )
            raise ValueError(msg)

        weights = self.logits.softmax(dim=0).to(dtype=hidden[0].dtype)
        stacked = torch.stack(hidden, dim=0)
        return torch.einsum("l,lirw->irw", weights, stacked)


type _LayerReadout = _LastLayerReadout | _FixedLayerReadout | _ScalarMixReadout


def _layer_readout(
    settings: LayerReadoutSettings,
    *,
    state_count: int,
) -> _LayerReadout:
    """Build one typed pretrained-layer readout."""
    match settings:
        case LastLayerReadoutSettings():
            return _LastLayerReadout()
        case FixedLayerReadoutSettings(layer):
            return _FixedLayerReadout(layer, count=state_count)
        case ScalarMixReadoutSettings():
            return _ScalarMixReadout(state_count)
        case unreachable:
            assert_never(unreachable)


class PretrainedTokenEncoder(nn.Module):
    """Contextual token encoder backed by one stack of a pretrained checkpoint.

    Each item's words are joined by the declared string and tokenized whole, the rows
    are run through the stack, and each word is aligned to the rows whose
    character spans fall inside it.

    Parameters
    ----------
    stack : _EncoderStack
        The loaded encoder stack.
    tokenizer : _Tokenizer
        The matching fast tokenizer.
    architecture : EncoderArchitecturePlugin
        Which stack of the checkpoint was loaded.
    checkpoint : PinnedCheckpoint
        The release the stack and tokenizer were loaded from.
    component : ComponentId
        The handle assigned to this component.
    trainable : bool
        Whether the encoder parameters take gradients.
    window : ContextWindow
        The token-row limit and overflow policy.
    readout : LayerReadoutSettings
        Which hidden state layers the encoder publishes.
    join : str
        The string the words are joined by before tokenization.

    Raises
    ------
    TypeError
        If the stack is not a torch module, or its configured width is not an
        integer.
    ValueError
        If the configured width is not positive, or the tokenizer's padding
        index is negative.
    """

    def __init__(
        self,
        stack: _EncoderStack,
        tokenizer: _Tokenizer,
        *,
        architecture: EncoderArchitecturePlugin,
        checkpoint: PinnedCheckpoint,
        component: ComponentId,
        trainable: bool,
        window: ContextWindow,
        readout: LayerReadoutSettings,
        join: str,
    ) -> None:
        super().__init__()

        if not isinstance(stack, nn.Module):
            msg = f"an encoder stack is a torch module, not {stack!r}"
            raise TypeError(msg)

        pad = ensure_non_negative("the tokenizer padding index", tokenizer.pad_token_id)

        self._encoder = stack
        self._tokenizer = tokenizer
        self._architecture = architecture
        self._output_width = _output_width(stack, architecture)
        self._checkpoint = checkpoint
        self._component = component
        self._trainable = trainable
        self._window = window
        self._readout = _layer_readout(
            readout, state_count=architecture.layer_count(stack) + 1
        )
        self._join = join
        self._pad_token_id = pad

        for parameter in self._encoder.parameters():
            parameter.requires_grad_(trainable)

    @property
    def output_width(self) -> int:
        """Return the width recorded by the checkpoint configuration."""
        return self._output_width

    @property
    def architecture(self) -> EncoderArchitecturePlugin:
        """Return which stack of the checkpoint the encoder runs."""
        return self._architecture

    @property
    def checkpoint(self) -> PinnedCheckpoint:
        """Return the release the encoder was loaded from."""
        return self._checkpoint

    @property
    def trainable(self) -> bool:
        """Return whether the encoder parameters take gradients."""
        return self._trainable

    @property
    def window(self) -> ContextWindow:
        """Return the token-row limit."""
        return self._window

    def _empty(self, items: Sequence[Sequence[str]]) -> TokenEncoding:
        """Return an empty encoded batch."""
        parameter = next(self._encoder.parameters(), None)
        device = None if parameter is None else parameter.device
        dtype = torch.float32 if parameter is None else parameter.dtype
        tensor = torch.empty((0, 0, self._output_width), device=device, dtype=dtype)
        mask = torch.empty((0, 0), device=device, dtype=torch.bool)

        return TokenEncoding(Encoded(tensor, _ROW_AXES, mask), (), words=items)

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        """Return contextual subword rows and their input-word alignment.

        Parameters
        ----------
        items : Sequence[Sequence[str]]
            The words of each item, in item order and then in word order.

        Returns
        -------
        TokenEncoding
            Contextual rows with padding masked and one alignment per item.

        Raises
        ------
        ContextOverflowError
            If an item exceeds a window that refuses overflow.
        ZeroPieceWordError
            If the tokenizer produces no row for a word.
        CrossingPieceError
            If a token row crosses the boundary between two words.
        ValueError
            If a word is empty or only whitespace, or the tokenizer or encoder
            returns inconsistent dimensions.
        """
        if not items:
            return self._empty(items)

        _refuse_blank(items)
        input_ids, attention, offsets = _tokenized(self._tokenizer, items, self._join)
        selected: list[tuple[int, ...]] = []
        alignments: list[Alignment] = []

        for item, words in enumerate(items):
            word_ids = _word_ids(words, offsets[item].tolist(), join=self._join)
            rows, packed = _trimmed(
                words, word_ids, _live(attention, item), window=self._window
            )
            selected.append(rows)
            alignments.append(Alignment(tuple(map(_interval, packed))))

        longest = max(map(len, selected), default=0)
        packed_ids = input_ids.new_full((len(items), longest), self._pad_token_id)
        packed_attention = attention.new_zeros((len(items), longest))

        for item, rows in enumerate(selected):
            count = len(rows)

            if count:
                index = torch.tensor(rows, dtype=torch.long, device=input_ids.device)
                packed_ids[item, :count] = input_ids[item].index_select(0, index)
                packed_attention[item, :count] = 1

        parameter = next(self._encoder.parameters(), None)
        device = packed_ids.device if parameter is None else parameter.device
        packed_ids = packed_ids.to(device=device)
        packed_attention = packed_attention.to(device=device)
        encoded = self._encoder(
            input_ids=packed_ids,
            attention_mask=packed_attention,
            return_dict=True,
            output_hidden_states=self._readout.requires_hidden_states,
        )
        output = self._readout.rows(encoded)

        expected = (len(items), longest, self._output_width)

        if tuple(output.shape) != expected:
            msg = (
                f"the encoder configuration declares output shape {expected} "
                f"for this batch, not {tuple(output.shape)}"
            )
            raise ValueError(msg)

        mask = packed_attention.to(dtype=torch.bool)
        rows = output * mask.unsqueeze(-1).to(dtype=output.dtype)

        return TokenEncoding(
            Encoded(rows, _ROW_AXES, mask), tuple(alignments), words=items
        )

    def site_map(self) -> SiteMap:
        """Return the contextual rows under ``Role.TOKEN_EMBEDDING``."""
        return SiteMap(
            self._component,
            ComponentKind.TOKEN_ENCODER,
            {
                Site(self._component, Role.TOKEN_EMBEDDING): SiteSpec(
                    _ROW_AXES,
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                )
            },
        )


def declared_window(
    settings: ContextWindowSettings, tokenizer: _Tokenizer, /
) -> ContextWindow:
    """Return the context window a declaration bounds an item's rows by.

    Parameters
    ----------
    settings : ContextWindowSettings
        The declared row bound and overflow policy.

    Returns
    -------
    ContextWindow
        The resolved context window.
    """
    rows = settings.rows

    if rows is None:
        try:
            published = tokenizer.model_max_length
        except AttributeError as error:
            msg = "a tokenizer publishes no context-window capability"
            raise TypeError(msg) from error

        if isinstance(published, bool) or not isinstance(published, int):
            msg = (
                "a tokenizer publishes its context-window capability as an integer, "
                f"not {published!r}"
            )
            raise TypeError(msg)

        if published <= 0 or published >= _MAX_PUBLISHED_CONTEXT:
            msg = (
                "a tokenizer publishes a finite positive context-window capability, "
                f"not {published!r}"
            )
            raise ValueError(msg)

        rows = published

    return ContextWindow(rows, overflow=overflow_policy(settings.overflow))


@final
class WordRows(FrozenValue):
    """The exact live rows one tokenizer produces for each word of one item.

    Parameters
    ----------
    word_rows : tuple[tuple[int, ...], ...]
        The ordered rows of each word, in word order.
    special_rows : tuple[int, ...]
        The ordered live rows no word produced.
    live_rows : int
        How many live rows the item takes.
    """

    __slots__ = ("_live_rows", "_special_rows", "_word_rows")

    _noun: ClassVar[str] = "word-row alignment"

    def __init__(
        self,
        word_rows: tuple[tuple[int, ...], ...],
        special_rows: tuple[int, ...],
        live_rows: int,
    ) -> None:
        object.__setattr__(self, "_word_rows", word_rows)
        object.__setattr__(self, "_special_rows", special_rows)
        object.__setattr__(self, "_live_rows", live_rows)

    @property
    def word_rows(self) -> tuple[tuple[int, ...], ...]:
        """Return the ordered rows of each word, in word order."""
        return self._word_rows

    @property
    def special_rows(self) -> tuple[int, ...]:
        """Return the ordered live rows no word produced."""
        return self._special_rows

    @property
    def live_rows(self) -> int:
        """Return how many live rows the item takes."""
        return self._live_rows

    def _identity(self) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int]:
        """Return the three fields the rows are equal on."""
        return (self._word_rows, self._special_rows, self._live_rows)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same rows."""
        if not isinstance(other, WordRows):
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the rows."""
        return hash(self._identity())

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these rows."""
        return (
            f"WordRows({self._word_rows!r}, {self._special_rows!r}, "
            f"{self._live_rows!r})"
        )


@final
class CheckpointAligner:
    """Reads the exact rows a checkpoint's tokenizer gives each word of an item.

    The words are joined and tokenized as a pretrained token encoder of the same
    checkpoint, join and window tokenizes them, and each word owns the rows
    whose character spans fall inside it. Every row of every word is kept, so a
    word whose rows are interrupted by a row of no word keeps that gap.

    Parameters
    ----------
    checkpoint : PinnedCheckpoint
        The release the tokenizer was loaded from.
    tokenizer : _Tokenizer
        The checkpoint's fast tokenizer.
    join : str
        The string the words are joined by before tokenization.
    window : ContextWindow
        The row bound an item must fit whole.
    """

    __slots__ = ("_checkpoint", "_join", "_tokenizer", "_window")

    def __init__(
        self,
        checkpoint: PinnedCheckpoint,
        tokenizer: _Tokenizer,
        *,
        join: str,
        window: ContextWindow,
    ) -> None:
        self._checkpoint = checkpoint
        self._tokenizer = tokenizer
        self._join = join
        self._window = window

    @property
    def checkpoint(self) -> PinnedCheckpoint:
        """Return the release the tokenizer was loaded from."""
        return self._checkpoint

    @property
    def identity(self) -> str:
        """Return the checkpoint's ``repository@commit`` identity."""
        return self._checkpoint.identity

    def align(self, words: Sequence[str], /) -> WordRows:
        """Return every live row of every word and every row of no word.

        An exact alignment covers every word, so an item longer than the window
        is refused whatever the window's overflow policy.

        Parameters
        ----------
        words : Sequence[str]
            The item's words, in order.

        Raises
        ------
        ContextOverflowError
            If the item takes more rows than the window holds.
        ZeroPieceWordError
            If the tokenizer produces no row for a word.
        CrossingPieceError
            If a token row crosses the boundary between two words.
        ValueError
            If a word is empty or only whitespace, or the tokenizer returns
            inconsistent dimensions.
        """
        held = tuple(words)
        _refuse_blank((held,))
        _, attention, offsets = _tokenized(self._tokenizer, (held,), self._join)
        word_ids = _word_ids(held, offsets[0].tolist(), join=self._join)
        live = _live(attention, 0)

        if len(live) > self._window.rows:
            raise ContextOverflowError(held, rows=self._window.rows, needed=len(live))

        _, packed = _trimmed(held, word_ids, live, window=self._window)

        return WordRows(
            tuple(rows for rows in packed if rows is not None),
            tuple(target for target, row in enumerate(live) if word_ids[row] is None),
            len(live),
        )


def load_aligner(
    checkpoint: PinnedCheckpoint,
    *,
    loader: PretrainedLoader,
    join: str,
    window: ContextWindowSettings,
) -> CheckpointAligner:
    """Load a checkpoint's tokenizer as a word-row aligner.

    Parameters
    ----------
    checkpoint : PinnedCheckpoint
        The release the tokenizer is loaded from.
    loader : PretrainedLoader
        The loading backend.
    join : str
        The string the words are joined by before tokenization.
    window : ContextWindowSettings
        The declared row bound, or none for the tokenizer's published one.
    """
    tokenizer = loader.load_tokenizer(checkpoint)

    return CheckpointAligner(
        checkpoint,
        tokenizer,
        join=join,
        window=declared_window(window, tokenizer),
    )


def load_pretrained(
    checkpoint: PinnedCheckpoint,
    *,
    architecture: EncoderArchitecturePlugin,
    component: ComponentId,
    trainable: bool,
    window: ContextWindowSettings,
    readout: LayerReadoutSettings,
    join: str,
    loader: PretrainedLoader,
    dtype: torch.dtype,
) -> PretrainedTokenEncoder:
    """Load one stack of a checkpoint and its tokenizer as a token encoder.

    Parameters
    ----------
    checkpoint : PinnedCheckpoint
        The release the stack and tokenizer are loaded from.
    architecture : EncoderArchitecturePlugin
        Which stack of the release is loaded.
    component : ComponentId
        The handle assigned to this component.
    trainable : bool
        Whether the encoder parameters take gradients.
    window : ContextWindowSettings
        The token-row limit and overflow policy.
    readout : LayerReadoutSettings
        Which hidden state layers the encoder publishes.
    join : str
        The string the words are joined by before tokenization.
    loader : PretrainedLoader
        The loading backend.
    dtype : torch.dtype
        The precision the parameters are loaded in.
    """
    stack = loader.load_encoder(checkpoint, architecture=architecture, dtype=dtype)
    tokenizer = loader.load_tokenizer(checkpoint)

    return PretrainedTokenEncoder(
        stack,
        tokenizer,
        architecture=architecture,
        checkpoint=checkpoint,
        component=component,
        trainable=trainable,
        window=declared_window(window, tokenizer),
        readout=readout,
        join=join,
    )


def _build_pretrained(
    settings: PretrainedTokenEncoderSettings, context: BuildContext, /
) -> PretrainedTokenEncoder:
    """Build and place a pretrained token encoder through the run's loader.

    Parameters
    ----------
    settings : PretrainedTokenEncoderSettings
        The pinned checkpoint, which of its stacks runs, whether it trains, and
        its context window.
    context : BuildContext
        The loader the weights come through, the component handle and the
        device policy, whose parameter precision the weights are loaded in.

    Returns
    -------
    PretrainedTokenEncoder
        The loaded encoder.
    """
    return place(
        load_pretrained(
            settings.checkpoint,
            architecture=encoder_architecture(settings.architecture),
            component=context.component,
            trainable=settings.trainable,
            window=settings.window,
            readout=settings.readout,
            join=settings.join,
            loader=context.loader,
            dtype=context.policy.dtype,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(
    spec=PretrainedTokenEncoderSpec, build=_build_pretrained
)
"""The pretrained token encoder's plugin."""

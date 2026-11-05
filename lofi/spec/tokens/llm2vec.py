"""LLM2Vec token encoder declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.artifacts import PretrainedSourceSpec, source_artifact
from lofi.spec.components.model import (
    _DEFAULT_TOKEN_ENCODER_TRAINABLE,
    TokenEncoderSpec,
)
from lofi.spec.components.trainer import OptimizerSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSpec
from lofi.spec.tokens.pretrained import (
    _DEFAULT_CONTEXT_WINDOW,
    _DEFAULT_WORD_JOIN,
    ContextWindowSpec,
)

if TYPE_CHECKING:
    from lofi._artifacts import PinnedCheckpoint
    from lofi.spec.tokens.pretrained import ContextWindowSettings

__all__ = [
    "AdaptationName",
    "LLM2VecTokenEncoderSettings",
    "LLM2VecTokenEncoderSpec",
    "MaskReplacementName",
    "MaskedNextTokenSettings",
    "MaskedNextTokenSpec",
]

type AdaptationName = str
"""Which installed parameter adaptation mechanism an LLM2Vec run uses."""

type MaskReplacementName = str
"""What a token chosen for masked next-token prediction is replaced with.

Under ``mixed`` four in five chosen tokens become the mask token, one in ten a
token drawn uniformly from the vocabulary and one in ten stay as they are. Under
``mask`` every chosen token becomes the mask token.
"""


def _default_optimizer() -> AdamWOptimizerSpec:
    """Return the published masked-next-token optimizer declaration."""
    return AdamWOptimizerSpec(learning_rate=5e-5)


class MaskedNextTokenSettings(NamedTuple):
    """The settings masked next-token adaptation runs under.

    Attributes
    ----------
    corpus : str
        The absolute path of the text file adapted on, one item per line.
    mask : str
        The token string a chosen token is replaced with.
    mask_probability : float
        The probability each eligible token is chosen.
    replacement : MaskReplacementName
        What a chosen token is replaced with.
    steps : int
        How many optimizer steps adaptation takes.
    batch_size : int
        How many corpus lines each step reads.
    learning_rate : float
        The step size of the adaptation optimizer.
    """

    corpus: str
    mask: str
    mask_probability: float
    replacement: MaskReplacementName
    mask_replacement_fraction: float
    random_replacement_fraction: float
    steps: int
    batch_size: int
    optimizer: OptimizerSpec
    parameterization: AdaptationName
    rank: int
    alpha: float
    seed: int


class MaskedNextTokenSpec(dx.Model, extra="forbid"):
    """Declares the masked next-token adaptation run before the encoder is used.

    A fraction of each line's tokens is replaced, and the token at each replaced
    position is predicted from the bidirectional row one position before it,
    through the model's own output head.

    Attributes
    ----------
    corpus : str
        The text file adapted on, one item per line, or a packaged resource.
    mask : str
        The token string a chosen token is replaced with, which the tokenizer's
        vocabulary must hold.
    mask_probability : float
        The probability each eligible token is chosen, in (0, 1].
    replacement : MaskReplacementName
        What a chosen token is replaced with.
    steps : int
        How many optimizer steps adaptation takes, at least one.
    batch_size : int
        How many corpus lines each step reads, at least one.
    learning_rate : float
        The step size of the adaptation optimizer, positive.
    """

    corpus: str
    mask: str = "_"
    mask_probability: float = 0.2
    replacement: MaskReplacementName = "mixed"
    mask_replacement_fraction: float = 0.8
    random_replacement_fraction: float = 0.1
    steps: int = 1000
    batch_size: int = 32
    optimizer: OptimizerSpec = dx.field(default_factory=_default_optimizer)
    parameterization: str = "full"
    rank: int = 8
    alpha: float = 8.0
    seed: int = 0

    __axioms__ = (
        dx.axiom(
            "mask_probability > 0 and mask_probability <= 1",
            message="a mask probability lies in (0, 1]",
        ),
        dx.axiom(
            "mask_replacement_fraction >= 0 and random_replacement_fraction >= 0 "
            "and mask_replacement_fraction + random_replacement_fraction <= 1",
            message=(
                "mask and random replacement fractions are nonnegative and sum "
                "to at most one"
            ),
        ),
        dx.axiom("steps > 0", message="an adaptation of no steps adapts nothing"),
        dx.axiom("batch_size > 0", message="a batch of no lines reads nothing"),
    )

    @dx.validates("corpus")
    @classmethod
    def _corpus_is_named(cls, value: str) -> str:
        """Validate the corpus."""
        return ensure_name("an adaptation corpus", value)

    @dx.validates("mask")
    @classmethod
    def _mask_token_is_named(cls, value: str) -> str:
        """Validate the mask token."""
        return ensure_name("a mask token", value)

    @dx.validates("replacement")
    @classmethod
    def _replacement_is_named(cls, value: str) -> str:
        """Validate the installed replacement plugin name."""
        return ensure_name("a mask replacement", value)

    @dx.validates("parameterization")
    @classmethod
    def _parameterization_is_named(cls, value: str) -> str:
        """Validate the installed adaptation plugin name."""
        return ensure_name("an adaptation", value)

    @dx.validates("rank")
    @classmethod
    def _rank_is_positive(cls, value: int) -> int:
        """Validate the low-rank adaptation rank."""
        return ensure_positive("an adaptation rank", value)

    @dx.validates("alpha")
    @classmethod
    def _alpha_is_positive(cls, value: float) -> float:
        """Validate the low-rank adaptation scale."""
        checked = ensure_finite("an adaptation scale", value)

        if checked <= 0:
            msg = "an adaptation scale is positive"
            raise ValueError(msg)

        return checked

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int) -> int:
        """Validate the dedicated adaptation seed."""
        return ensure_non_negative("an adaptation seed", value)

    @dx.validates("mask_replacement_fraction", "random_replacement_fraction")
    @classmethod
    def _replacement_fraction_is_finite(cls, value: float) -> float:
        """Validate one declared replacement fraction."""
        return ensure_finite("a replacement fraction", value)

    def settings(self) -> MaskedNextTokenSettings:
        """Return the settings adaptation runs under."""
        return MaskedNextTokenSettings(
            corpus=lower_package_resource(self.corpus),
            mask=self.mask,
            mask_probability=self.mask_probability,
            replacement=self.replacement,
            mask_replacement_fraction=self.mask_replacement_fraction,
            random_replacement_fraction=self.random_replacement_fraction,
            steps=self.steps,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            parameterization=self.parameterization,
            rank=self.rank,
            alpha=self.alpha,
            seed=self.seed,
        )


class LLM2VecTokenEncoderSettings(NamedTuple):
    """The settings an LLM2Vec token encoder is built from.

    Attributes
    ----------
    checkpoint : PinnedCheckpoint
        The release the decoder language model and its tokenizer are fetched from.
    trainable : bool
        Whether the decoder's parameters take gradients during training.
    window : ContextWindowSettings
        The bound on one item's rows.
    adaptation : MaskedNextTokenSettings | None
        The masked next-token adaptation run before use, or None for none.
    """

    checkpoint: PinnedCheckpoint
    trainable: bool
    window: ContextWindowSettings
    adaptation: MaskedNextTokenSettings | None
    join: str


class LLM2VecTokenEncoderSpec(TokenEncoderSpec):
    """Declares a decoder language model converted into a bidirectional encoder.

    Attributes
    ----------
    kind : Literal["llm2vec"]
        Registry tag.
    checkpoint : PretrainedSourceSpec
        The hub repository and commit the decoder language model and its
        tokenizer are fetched from.
    trainable : bool
        Whether the decoder's parameters take gradients during training, that is,
        whether the encoder is fine-tuned or frozen.
    window : ContextWindowSpec
        The bound on one item's rows.
    adaptation : MaskedNextTokenSpec | None
        The masked next-token adaptation run before the encoder is used, or None
        where the checkpoint is used as loaded.
    """

    kind: Literal["llm2vec"] = "llm2vec"
    checkpoint: PretrainedSourceSpec
    trainable: bool = _DEFAULT_TOKEN_ENCODER_TRAINABLE
    window: ContextWindowSpec = _DEFAULT_CONTEXT_WINDOW
    adaptation: MaskedNextTokenSpec | None = None
    join: str = _DEFAULT_WORD_JOIN

    @dx.validates("join")
    @classmethod
    def _join_is_not_empty(cls, value: str) -> str:
        """Validate the string placed between input words."""
        if not value:
            msg = "a word join is not empty"
            raise ValueError(msg)

        return value

    @dx.model_validator()
    def _loads_a_hub_checkpoint(self) -> Self:
        """Refuse a source that is not a hub repository at one commit."""
        self._checkpoint()

        return self

    @override
    def pretrained_weights(self) -> str:
        """Return the decoder checkpoint and the revision it is pinned at."""
        return self._checkpoint().identity

    def _checkpoint(self) -> PinnedCheckpoint:
        """Return the pinned release the declared source names."""
        return source_artifact(self.checkpoint).hub_checkpoint(
            "an LLM2Vec token encoder"
        )

    def settings(self) -> LLM2VecTokenEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return LLM2VecTokenEncoderSettings(
            checkpoint=self._checkpoint(),
            trainable=self.trainable,
            window=self.window.settings(),
            adaptation=None if self.adaptation is None else self.adaptation.settings(),
            join=self.join,
        )

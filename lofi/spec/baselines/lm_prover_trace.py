"""Validated inputs and raw generations for LM-plus-prover baselines."""

import json
from typing import Literal, Self

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.lf import RELATION_LABELS, InferenceLabel

__all__ = [
    "GeneratedFormalization",
    "LmProverData",
    "LmProverGenerationItem",
    "LmProverGenerations",
    "LmProverPair",
    "read_lm_prover_data",
    "read_lm_prover_generations",
]

type _Split = Literal["dev", "test"]


class LmProverPair(dx.Model, extra="forbid"):
    """One quantifier-free inference pair and its gold logical forms."""

    item_id: str
    split: _Split
    premise_tokens: tuple[str, ...]
    hypothesis_tokens: tuple[str, ...]
    label: InferenceLabel
    gold_premise: str
    gold_hypothesis: str

    @dx.validates("item_id", "gold_premise", "gold_hypothesis")
    @classmethod
    def _text_is_nonempty(cls, value: str) -> str:
        """Return a nonempty identifier or logical form text."""
        return ensure_name("an LM-plus-prover pair field", value)

    @dx.validates("label")
    @classmethod
    def _label_is_a_logical_relation(cls, label: InferenceLabel) -> InferenceLabel:
        """Return a proof relation rather than a coarsened output label."""
        if label not in RELATION_LABELS:
            msg = f"an LM-plus-prover pair needs a logical relation, not {label!r}"
            raise ValueError(msg)
        return label

    @dx.validates("premise_tokens", "hypothesis_tokens")
    @classmethod
    def _tokens_are_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return a nonempty sequence of nonempty surface tokens."""
        if not values:
            msg = "an LM-plus-prover sentence has at least one token"
            raise ValueError(msg)
        return tuple(
            ensure_name(f"an LM-plus-prover token at {position}", token)
            for position, token in enumerate(values)
        )


class LmProverData(dx.Model, extra="forbid"):
    """The held-out pair population scored by every baseline pipeline."""

    pairs: tuple[LmProverPair, ...]

    @dx.model_validator()
    def _pairs_are_distinct_and_partitioned(self) -> Self:
        """Require distinct items and nonempty development and test partitions."""
        identifiers = tuple(pair.item_id for pair in self.pairs)
        if len(set(identifiers)) != len(identifiers):
            msg = "LM-plus-prover data repeats an item identifier"
            raise ValueError(msg)
        splits = {pair.split for pair in self.pairs}
        if splits != {"dev", "test"}:
            msg = "LM-plus-prover data must contain nonempty dev and test partitions"
            raise ValueError(msg)
        return self


class GeneratedFormalization(dx.Model, extra="forbid"):
    """One model-generated premise and hypothesis formalization attempt."""

    premise: str
    hypothesis: str
    feedback: str | None = None

    @dx.validates("premise", "hypothesis")
    @classmethod
    def _form_text_is_nonempty(cls, value: str) -> str:
        """Return nonempty generated logical form text."""
        return ensure_name("a generated logical form", value)

    @dx.validates("feedback")
    @classmethod
    def _feedback_is_nonempty_when_present(cls, value: str | None) -> str | None:
        """Return absent feedback or a nonempty solver message."""
        return None if value is None else ensure_name("solver feedback", value)


class LmProverGenerationItem(dx.Model, extra="forbid"):
    """The ordered formalization attempts recorded for one inference pair."""

    item_id: str
    candidates: tuple[GeneratedFormalization, ...]

    @dx.validates("item_id")
    @classmethod
    def _item_id_is_named(cls, value: str) -> str:
        """Return a nonempty pair identifier."""
        return ensure_name("an LM-plus-prover generation item", value)

    @dx.validates("candidates")
    @classmethod
    def _candidates_are_nonempty(
        cls, values: tuple[GeneratedFormalization, ...]
    ) -> tuple[GeneratedFormalization, ...]:
        """Return at least one model generation."""
        if not values:
            msg = "an LM-plus-prover generation item has at least one candidate"
            raise ValueError(msg)
        return values


class LmProverGenerations(dx.Model, extra="forbid"):
    """Model and prompt identity beside generated formalizations, never scores."""

    pipeline: str
    model: str
    prompt_digest: str
    items: tuple[LmProverGenerationItem, ...]

    @dx.validates("pipeline")
    @classmethod
    def _pipeline_is_named(cls, value: str) -> str:
        """Return the nonempty name of the pipeline that generated the trace."""
        return ensure_name("an LM-plus-prover pipeline name", value)

    @dx.validates("model")
    @classmethod
    def _model_is_named(cls, value: str) -> str:
        """Return a nonempty pinned model identity."""
        return ensure_name("an LM-plus-prover model identity", value)

    @dx.validates("prompt_digest")
    @classmethod
    def _prompt_has_a_content_identity(cls, value: str) -> str:
        """Return a lower-case SHA-256 prompt identity."""
        digest = ensure_name("an LM-plus-prover prompt digest", value)
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            msg = f"an LM-plus-prover prompt has no SHA-256 identity: {value!r}"
            raise ValueError(msg)
        return digest

    @dx.model_validator()
    def _items_are_distinct(self) -> Self:
        """Require one generation record per item identifier."""
        identifiers = tuple(item.item_id for item in self.items)
        if not identifiers:
            msg = "an LM-plus-prover generation trace contains at least one item"
            raise ValueError(msg)
        if len(set(identifiers)) != len(identifiers):
            msg = "an LM-plus-prover generation trace repeats an item identifier"
            raise ValueError(msg)
        return self


def read_lm_prover_data(text: str, /) -> LmProverData:
    """Validate one complete LM-plus-prover evaluation population."""
    try:
        return LmProverData.model_validate_json(text)
    except json.JSONDecodeError as error:
        msg = f"LM-plus-prover data is not JSON: {error}"
        raise ValueError(msg) from error
    except dx.ValidationError as error:
        msg = f"LM-plus-prover data does not validate: {error}"
        raise ValueError(msg) from error


def read_lm_prover_generations(text: str, /) -> LmProverGenerations:
    """Validate generated formalizations without accepting computed scores."""
    try:
        return LmProverGenerations.model_validate_json(text)
    except json.JSONDecodeError as error:
        msg = f"LM-plus-prover generations are not JSON: {error}"
        raise ValueError(msg) from error
    except dx.ValidationError as error:
        msg = f"LM-plus-prover generations do not validate: {error}"
        raise ValueError(msg) from error

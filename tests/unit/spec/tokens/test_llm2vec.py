from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi._artifacts import PinnedCheckpoint
from lofi.spec.artifacts import HubSourceSpec
from lofi.spec.components.model import TokenEncoderSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSpec
from lofi.spec.tokens.llm2vec import (
    LLM2VecTokenEncoderSpec,
    MaskedNextTokenSettings,
    MaskedNextTokenSpec,
)
from lofi.spec.tokens.pretrained import ContextWindowSettings, ContextWindowSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

REVISION = "0123456789abcdef" * 2 + "01234567"


def test_the_tag_reaches_the_llm2vec_variant_through_the_family_root() -> None:
    declared = TokenEncoderSpec.model_validate(
        {
            "kind": "llm2vec",
            "checkpoint": {
                "kind": "hub",
                "repository": "an-org/a-decoder",
                "revision": REVISION,
            },
            "trainable": True,
            "window": {"rows": 64, "overflow": "refuse"},
            "adaptation": {
                "corpus": "/data/corpus.txt",
                "mask": "<mask>",
                "mask_probability": 0.4,
                "replacement": "mask",
                "steps": 7,
                "batch_size": 3,
                "optimizer": {"kind": "adamw", "learning_rate": 0.001},
                "parameterization": "lora",
                "rank": 4,
                "alpha": 8.0,
                "seed": 19,
            },
        }
    )

    assert type(declared) is LLM2VecTokenEncoderSpec
    assert TokenEncoderSpec.__variants__["llm2vec"] is LLM2VecTokenEncoderSpec
    assert declared.settings() == (
        PinnedCheckpoint("an-org/a-decoder", REVISION),
        True,
        ContextWindowSettings(rows=64, overflow="refuse"),
        MaskedNextTokenSettings(
            corpus="/data/corpus.txt",
            mask="<mask>",
            mask_probability=0.4,
            replacement="mask",
            mask_replacement_fraction=0.8,
            random_replacement_fraction=0.1,
            steps=7,
            batch_size=3,
            optimizer=AdamWOptimizerSpec(learning_rate=0.001),
            parameterization="lora",
            rank=4,
            alpha=8.0,
            seed=19,
        ),
        " ",
    )


def test_an_encoder_is_frozen_unbounded_and_unadapted_unless_declared() -> None:
    declared = LLM2VecTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository="an-org/a-decoder", revision=REVISION)
    )
    settings = declared.settings()

    assert settings.trainable is False
    assert settings.window == ContextWindowSettings(None, "report")
    assert settings.adaptation is None
    assert declared.dependencies() == ()
    assert declared.pretrained_weights() == f"an-org/a-decoder@{REVISION}"


def test_adaptation_defaults_to_the_published_recipe() -> None:
    settings = MaskedNextTokenSpec(corpus="/data/corpus.txt").settings()

    assert settings == MaskedNextTokenSettings(
        corpus="/data/corpus.txt",
        mask="_",
        mask_probability=0.2,
        replacement="mixed",
        mask_replacement_fraction=0.8,
        random_replacement_fraction=0.1,
        steps=1000,
        batch_size=32,
        optimizer=AdamWOptimizerSpec(learning_rate=5e-5),
        parameterization="full",
        rank=8,
        alpha=8.0,
        seed=0,
    )


def test_a_packaged_corpus_resolves_to_an_absolute_path() -> None:
    settings = MaskedNextTokenSpec(
        corpus="lofi.resource:run/synthetic-parser.vec"
    ).settings()

    assert settings.corpus.startswith("/")
    assert settings.corpus.endswith("run/synthetic-parser.vec")


def _payload(**adaptation: JsonValue) -> dict[str, JsonValue]:
    return {
        "kind": "llm2vec",
        "checkpoint": {"kind": "hub", "repository": "m", "revision": REVISION},
        "adaptation": {"corpus": "c", **adaptation},
    }


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "llm2vec", "revision": REVISION}, "checkpoint"),
        (
            {"kind": "llm2vec", "checkpoint": {"kind": "hub", "repository": "m"}},
            "revision",
        ),
        (
            {
                "kind": "llm2vec",
                "checkpoint": {
                    "kind": "url",
                    "url": "https://example.org/model.zip",
                    "sha256": "0" * 64,
                },
            },
            "an LLM2Vec token encoder loads a hub repository",
        ),
        (
            {
                "kind": "llm2vec",
                "checkpoint": {"kind": "hub", "repository": "", "revision": REVISION},
            },
            "a checkpoint",
        ),
        (
            {
                "kind": "llm2vec",
                "checkpoint": {"kind": "hub", "repository": "m", "revision": "main"},
            },
            "full lowercase commit hash",
        ),
        (
            {
                "kind": "llm2vec",
                "checkpoint": {"kind": "hub", "repository": "m", "revision": REVISION},
                "window": {"rows": 0},
            },
            "a window of no rows holds no word",
        ),
        (
            {
                "kind": "llm2vec",
                "checkpoint": {"kind": "hub", "repository": "m", "revision": REVISION},
                "architecture": "encoder-only",
            },
            "architecture",
        ),
        (
            {
                "kind": "llm2vec",
                "checkpoint": {"kind": "hub", "repository": "m", "revision": REVISION},
                "adaptation": {},
            },
            "corpus",
        ),
        (_payload(corpus=""), "an adaptation corpus"),
        (_payload(mask=""), "a mask token"),
        (_payload(mask_probability=0.0), r"lies in \(0, 1\]"),
        (_payload(mask_probability=1.5), r"lies in \(0, 1\]"),
        (_payload(mask_replacement_fraction=-0.1), "replacement fractions"),
        (_payload(random_replacement_fraction=-0.1), "replacement fractions"),
        (
            _payload(mask_replacement_fraction=0.8, random_replacement_fraction=0.3),
            "sum to at most one",
        ),
        (_payload(steps=0), "adapts nothing"),
        (_payload(batch_size=0), "reads nothing"),
        (
            _payload(optimizer={"kind": "adamw", "learning_rate": 0.0}),
            "learning_rate",
        ),
    ],
)
def test_a_missing_empty_or_out_of_range_setting_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TokenEncoderSpec.model_validate(payload)


def test_replacement_names_are_open_to_installed_plugins() -> None:
    assert MaskedNextTokenSpec(corpus="c", replacement="random").replacement == (
        "random"
    )


def test_the_variant_round_trips_through_its_json_form() -> None:
    declared = LLM2VecTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository="an-org/a-decoder", revision=REVISION),
        trainable=True,
        window=ContextWindowSpec(rows=8),
        adaptation=MaskedNextTokenSpec(corpus="/data/corpus.txt", steps=5),
    )

    read = TokenEncoderSpec.model_validate_json(declared.model_dump_json())

    assert read == declared
    assert read.model_dump_json() == declared.model_dump_json()

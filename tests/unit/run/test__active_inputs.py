"""Tests for the inputs an active-learning run reads from its documents."""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from lofi.judgment import (
    DesignError,
    MaskedLMGeneratorSpec,
    MaskedLMLexiconGenerator,
)
from lofi.run._active_inputs import (
    pool_readouts,
    read_bead_judgments,
    read_encoding,
    read_pool,
)
from lofi.run._entrypoint import main
from lofi.spec.acquisition import PointReadoutSettings, TokenizerSettings

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec.lexicon_generators import LexiconGeneratorSpec

CONFIG: Final = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)
"""The packaged synthetic parser whose sentences a pool is built over."""

PREDICATES: Final = ("word-1", "word-2", "word-3", "word-5", "word-6", "word-7")
"""The predicates the fixture generator proposes, in preference order."""

LABELS: Final = ("low", "mid", "high")
"""The labels of the fixture acceptability question, in response order."""

WHITESPACE: Final = TokenizerSettings("whitespace", "en", None)

ROOT_MASS: Final = {
    "p_gram": PointReadoutSettings("sentences", "log-mass"),
    "p_gram_complement": PointReadoutSettings("sentences", "log-complement"),
}
"""The trained point's root mass pair as two predictor columns."""


def _template(name: str, text: str, slots: tuple[str, ...], index: int) -> JsonValue:
    """Return one inline bead template whose slots read their filler's role."""
    return {
        "id": f"6f1c3a52-6c55-4a8e-9a0b-0d7c3e2f1a0{index}",
        "name": name,
        "template_string": text,
        "language_code": "eng",
        "slots": {
            slot: {
                "name": slot,
                "constraints": [
                    {"expression": f"self.features.get('role') == '{slot}'"}
                ],
            }
            for slot in slots
        },
    }


def item_document() -> dict[str, JsonValue]:
    """Return a two-template construction over the synthetic parser's words."""
    return {
        "namespace": "d1547144-8a00-4c23-b341-6d87f38b0f65",
        "templates": [
            {
                "kind": "inline",
                "template": _template(
                    "transitive",
                    "{subject} {verb} {object}",
                    ("subject", "verb", "object"),
                    1,
                ),
            },
            {
                "kind": "inline",
                "template": _template(
                    "intransitive", "{subject} {verb}", ("subject", "verb"), 2
                ),
            },
        ],
        "lexicons": [
            {
                "kind": "inline",
                "lexicon": {
                    "name": "arguments",
                    "items": [
                        {
                            "id": "0b7e2d10-2f4c-4c55-8f0e-5d1f8a3c2b01",
                            "lemma": "word-0",
                            "language_code": "eng",
                            "features": {"role": "subject"},
                        },
                        {
                            "id": "0b7e2d10-2f4c-4c55-8f0e-5d1f8a3c2b02",
                            "lemma": "word-4",
                            "language_code": "eng",
                            "features": {"role": "object"},
                        },
                    ],
                },
            }
        ],
        "generators": [
            {
                "kind": "masked-lm",
                "lexicon": "verbs",
                "language_code": "eng",
                "features": {"role": "verb"},
                "checkpoint": {
                    "kind": "hub",
                    "repository": "fixture/masked-lm",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                },
                "artifacts": {
                    "root": str(Path("/var/cache/lofi/artifacts").absolute()),
                    "offline": True,
                },
                "mask": "[MASK]",
                "count": len(PREDICATES),
                "prompts": ["word-0 [MASK] word-4"],
            }
        ],
        "filling": {"strategy": "exhaustive"},
        "template_fillings": {},
        "constraints": [],
        "admit": None,
        "item_template": {
            "kind": "inline",
            "item_template": {
                "id": "9a3e5c71-1d2b-4f6a-8c4e-7b0d2e1f3a00",
                "name": "rated sentence",
                "judgment_type": "acceptability",
                "task_type": "ordinal_scale",
                "task_spec": {"prompt": "How natural is this sentence?"},
                "presentation_spec": {},
                "elements": [
                    {"element_type": "filled_template_ref", "element_name": "sentence"}
                ],
            },
        },
        "attributes": {"structure": "template.name", "verb": "slots.verb.lemma"},
    }


def encoding_document() -> dict[str, JsonValue]:
    """Return the bead encoding of the fixture acceptability question."""
    return {
        "name": "acceptability",
        "n_levels": len(LABELS),
        "scale_type": "ordinal",
        "labels": list(LABELS),
    }


class FixtureFill:
    """Complete every fixture prompt with the fixture predicates in order."""

    def __call__(self, prompt: str, /, *, top_k: int) -> tuple[dict[str, object], ...]:
        """Return the fixture predicates, best first."""
        assert prompt == "word-0 [MASK] word-4"
        return tuple(
            {"token_str": predicate, "score": 1.0 - rank / top_k}
            for rank, predicate in enumerate(PREDICATES[:top_k])
        )


def fixture_generator(declaration: LexiconGeneratorSpec, /) -> MaskedLMLexiconGenerator:
    """Build the masked-LM generator over the fixture's fill-mask pipeline."""
    assert isinstance(declaration, MaskedLMGeneratorSpec)
    settings = declaration.settings()
    assert settings.checkpoint.repository == "fixture/masked-lm"
    return MaskedLMLexiconGenerator(settings, FixtureFill())


def train_point(root: Path, /) -> Path:
    """Train the packaged synthetic parser and return its run directory."""
    status = main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--seed",
            "3",
            "--experiment-seed",
            "7",
            "--root",
            str(root),
        ]
    )
    (run,) = tuple((root / "runs").glob("[0-9a-f]" * 16))
    assert status == 0
    return run


def annotation_lines(responses: dict[str, int], annotators: tuple[str, ...], /) -> str:
    """Return bead annotation records answering each item with its response."""
    return "".join(
        json.dumps(
            {
                "annotator_id": annotator,
                "item_id": item,
                "question_name": "acceptability",
                "response_label": LABELS[response],
            }
        )
        + "\n"
        for item, response in responses.items()
        for annotator in annotators
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return one trained synthetic point shared by the module's cases."""
    return train_point(tmp_path_factory.mktemp("store"))


def _pool(tmp_path: Path, /) -> Path:
    path = tmp_path / "items.json"
    path.write_text(json.dumps(item_document()), encoding="utf-8", newline="\n")
    return path


def test_the_pool_is_every_filling_of_the_declared_templates(tmp_path: Path) -> None:
    read = read_pool(_pool(tmp_path), generator_for=fixture_generator)
    pool = read.items

    assert read.sources == (tmp_path / "items.json",)
    assert len(pool) == 2 * len(PREDICATES)
    assert {item.item_metadata["structure"] for item in pool} == {
        "transitive",
        "intransitive",
    }
    assert pool[0].rendered_elements == {"sentence": "word-0 word-1 word-4"}
    assert pool[len(PREDICATES)].rendered_elements == {"sentence": "word-0 word-1"}


def test_bead_records_are_read_as_judgments_of_the_encoded_question(
    tmp_path: Path,
) -> None:
    encoding = tmp_path / "encoding.json"
    encoding.write_text(json.dumps(encoding_document()), encoding="utf-8", newline="\n")
    responses = tmp_path / "responses.jsonl"
    other = json.dumps(
        {
            "annotator_id": "a",
            "item_id": "item-1",
            "question_name": "naturalness",
            "response_label": "yes",
        }
    )
    responses.write_text(
        annotation_lines({"item-1": 2, "item-2": 0}, ("a", "b")) + other + "\n",
        encoding="utf-8",
        newline="\n",
    )

    read = read_bead_judgments(responses, read_encoding(encoding))

    assert tuple(
        (judgment.participant_id, judgment.item_id, judgment.response)
        for judgment in read.judgments
    ) == (
        ("a", "item-1", 2),
        ("b", "item-1", 2),
        ("a", "item-2", 0),
        ("b", "item-2", 0),
    )
    assert read.skipped == 1


def test_a_label_outside_the_encoding_is_refused(tmp_path: Path) -> None:
    encoding = tmp_path / "encoding.json"
    encoding.write_text(json.dumps(encoding_document()), encoding="utf-8", newline="\n")
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps(
            {
                "annotator_id": "a",
                "item_id": "item-1",
                "question_name": "acceptability",
                "response_label": "superb",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(DesignError, match="superb"):
        read_bead_judgments(responses, read_encoding(encoding))


def test_each_item_takes_its_trained_points_root_mass_pair(
    tmp_path: Path, trained: Path
) -> None:
    pool = read_pool(_pool(tmp_path), generator_for=fixture_generator).items

    columns = pool_readouts(trained, pool, {"words": "sentence"}, WHITESPACE, ROOT_MASS)

    assert tuple(columns) == tuple(str(item.id) for item in pool)
    for held in columns.values():
        mass = held["p_gram"]
        complement = held["p_gram_complement"]
        assert mass <= 0.0
        assert math.exp(mass) + math.exp(complement) == pytest.approx(1.0, abs=1e-5)
    assert len({held["p_gram"] for held in columns.values()}) > 1


def test_an_element_an_item_does_not_render_is_refused(
    tmp_path: Path, trained: Path
) -> None:
    pool = read_pool(_pool(tmp_path), generator_for=fixture_generator).items

    with pytest.raises(ValueError, match="'question'"):
        pool_readouts(trained, pool, {"words": "question"}, WHITESPACE, ROOT_MASS)


def test_inputs_mapping_another_role_than_the_parser_reads_are_refused(
    tmp_path: Path, trained: Path
) -> None:
    pool = read_pool(_pool(tmp_path), generator_for=fixture_generator).items

    with pytest.raises(ValueError, match="reads the input roles"):
        pool_readouts(trained, pool, {"premise": "sentence"}, WHITESPACE, ROOT_MASS)


@pytest.mark.parametrize(
    ("readout", "reason"),
    [("absent", "publishes"), ("types", "mass readout")],
)
def test_a_readout_the_point_does_not_publish_as_a_mass_is_refused(
    tmp_path: Path, trained: Path, readout: str, reason: str
) -> None:
    pool = read_pool(_pool(tmp_path), generator_for=fixture_generator).items

    with pytest.raises(ValueError, match=reason):
        pool_readouts(
            trained,
            pool,
            {"words": "sentence"},
            WHITESPACE,
            {"p_gram": PointReadoutSettings(readout, "log-mass")},
        )

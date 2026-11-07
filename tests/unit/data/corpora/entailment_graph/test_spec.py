import json

import didactic.api as dx
import pytest

from lofi.data.corpora.entailment_graph import (
    EdgeListEntailmentGraphSpec,
    EntailmentGraphGeneratorSettings,
    EntailmentGraphGeneratorSpec,
)

DECLARATION = {
    "kind": "entailment-graph",
    "graph": {"kind": "edge-list", "source": "graph.tsv", "language": "en"},
    "lexicon": "nodes",
    "feature": "paired",
    "linked": "plausible",
    "sampled": "presumed-non-inference",
    "negative_ratio": 2,
    "seed": None,
}


def _declaration(**changed: object) -> EntailmentGraphGeneratorSpec:
    return EntailmentGraphGeneratorSpec.model_validate_json(
        json.dumps({**DECLARATION, **changed})
    )


def test_the_generator_declares_its_graph_feature_and_labels() -> None:
    declaration = _declaration()
    settings = declaration.settings()

    assert type(declaration.graph) is EdgeListEntailmentGraphSpec
    assert declaration.family() == "lexicon_generator"
    assert settings == EntailmentGraphGeneratorSettings(
        "entailment-graph",
        declaration.graph,
        "nodes",
        "paired",
        "plausible",
        "presumed-non-inference",
        2,
        None,
    )


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"linked": "same", "sampled": "same"}, "both labeled"),
        ({"negative_ratio": -1}, "negative ratio"),
        ({"feature": ""}, "generator field"),
        ({"seed": -3}, "seed"),
    ],
)
def test_an_incoherent_generator_is_refused(
    changed: dict[str, object], reason: str
) -> None:
    with pytest.raises(dx.ValidationError, match=reason):
        _declaration(**changed)

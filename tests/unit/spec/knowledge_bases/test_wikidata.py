import didactic.api as dx
import pytest

from lofi.lf.models import WikidataSettings
from lofi.spec.knowledge_bases import WikidataKnowledgeBaseSpec


@pytest.mark.parametrize("policy", ["undecided", "another-policy"])
def test_a_wikidata_declaration_discards_its_boundary_model(policy: str) -> None:
    settings = WikidataKnowledgeBaseSpec(
        snapshot="snapshot.jsonl",
        mapping="mapping.json",
        world="open",
        unmapped=policy,
    ).settings()

    assert isinstance(settings, WikidataSettings)
    assert settings.snapshot == "snapshot.jsonl"
    assert settings.mapping == "mapping.json"
    assert settings.world == "open"
    assert settings.unmapped == policy


def test_a_wikidata_declaration_refuses_an_empty_path() -> None:
    with pytest.raises(dx.ValidationError, match="knowledge-base path"):
        WikidataKnowledgeBaseSpec(
            snapshot="",
            mapping="mapping.json",
            world="open",
            unmapped="undecided",
        )

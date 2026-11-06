import didactic.api as dx
import pytest

from lofi.lf.models import GivenSettings, WikidataSettings
from lofi.spec.knowledge_bases import WikidataKnowledgeBaseSpec
from lofi.spec.structures import GivenStructureSourceSpec


def declaration() -> GivenStructureSourceSpec:
    return GivenStructureSourceSpec(
        knowledge_base=WikidataKnowledgeBaseSpec(
            snapshot="snapshot.jsonl",
            mapping="mapping.json",
            world="open",
            unmapped="undecided",
        ),
        full_function_space_cap=64,
        henkin_pool=8,
    )


def test_a_given_source_resolves_its_reader_and_frame_settings() -> None:
    settings = declaration().settings()

    assert isinstance(settings, GivenSettings)
    assert settings.knowledge_base_kind == "wikidata"
    assert isinstance(settings.knowledge_base, WikidataSettings)
    assert settings.full_function_space_cap == 64
    assert settings.henkin_pool == 8


@pytest.mark.parametrize("field", ["full_function_space_cap", "henkin_pool"])
def test_a_given_source_refuses_a_nonpositive_frame_limit(field: str) -> None:
    with pytest.raises(dx.ValidationError, match="positive"):
        declaration().with_(**{field: 0})

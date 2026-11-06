"""Tests for the knowledge-base declaration public surface."""

from lofi.spec.knowledge_bases import KnowledgeBaseSpec, WikidataKnowledgeBaseSpec
from lofi.spec.knowledge_bases.wikidata import (
    KnowledgeBaseSpec as DefinedKnowledgeBaseSpec,
)
from lofi.spec.knowledge_bases.wikidata import (
    WikidataKnowledgeBaseSpec as DefinedWikidataKnowledgeBaseSpec,
)


def test_the_knowledge_base_surface_publishes_its_declarations() -> None:
    assert KnowledgeBaseSpec is DefinedKnowledgeBaseSpec
    assert WikidataKnowledgeBaseSpec is DefinedWikidataKnowledgeBaseSpec

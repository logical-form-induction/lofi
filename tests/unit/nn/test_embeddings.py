import torch

from lofi._types import ComponentId
from lofi.nn.embeddings import ChoiceEmbedding
from lofi.nn.sites import Role
from lofi.terms import ChoiceIndex, Signature, Vocabulary


def test_choice_embedding_has_one_row_per_payload_aware_choice() -> None:
    signature = Signature(
        "embedding",
        ("T",),
        (),
        vocabularies=(Vocabulary("T", ("left", "right")),),
    )
    index = ChoiceIndex(signature, depth_bounds={"T": 1})
    embedding = ChoiceEmbedding(
        index=index,
        width=3,
        component=ComponentId("choices"),
    )
    left = index.index(signature.constant("T"), 0)
    right = index.index(signature.constant("T"), 1)

    assert embedding.index is index
    assert embedding.output_width == 3
    assert embedding(torch.tensor([left, right])).shape == (2, 3)
    assert not torch.equal(embedding.weight[left], embedding.weight[right])
    assert embedding.weight.shape == (index.width, 3)
    assert torch.equal(embedding.weight[index.PAD], torch.zeros(3))


def test_choice_embedding_publishes_its_table_as_one_parameter_site() -> None:
    signature = Signature("embedding", ("T",), ())
    index = ChoiceIndex(signature, depth_bounds={"T": 1})
    embedding = ChoiceEmbedding(
        index=index,
        width=2,
        component=ComponentId("choices"),
    )
    sites = embedding.site_map()

    assert len(sites) == 1
    site = next(iter(sites))
    assert site.component == ComponentId("choices")
    assert site.role is Role.PARAMETER
    assert site.key == ("choice-embedding",)
    assert sites[site].axes == ("choice", "width")

"""Packaged leaf signatures reach production training documents."""

from typing import TYPE_CHECKING

from tests.acceptance.driver import leaf, train

if TYPE_CHECKING:
    from pathlib import Path


def test_a_document_reads_a_packaged_leaf_lexicon_and_trains(tmp_path: Path) -> None:
    point = train("leaf-lexicon-point.yaml", tmp_path)

    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.language.leaf_lexicon.path",
        )
        == "extensional.json"
    )
    assert point.rows

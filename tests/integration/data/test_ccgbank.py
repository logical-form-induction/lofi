from typing import TYPE_CHECKING

import pytest

from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.corpus
@pytest.mark.integration
def test_a_real_ccgbank_section_round_trips(ccgbank_root: Path) -> None:
    paths = sorted(ccgbank_root.rglob("*.auto"))
    assert paths, "LOFI_CCGBANK contains no .auto files"

    section = tuple(path for path in paths if path.parent == paths[0].parent)
    codec = CcgbankAutoCodec()
    fragment = None

    for path in section:
        fragment = codec.decode(path.read_bytes(), into=fragment)

    assert fragment is not None
    assert fragment.records

    encoded = codec.encode(fragment.records)

    assert codec.decode(encoded) == fragment

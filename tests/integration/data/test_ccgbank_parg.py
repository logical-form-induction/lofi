from typing import TYPE_CHECKING

import pytest

from lofi.data.corpora.ccgbank._parg import PargCodec

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.corpus
@pytest.mark.integration
def test_real_ccgbank_parg_section_zero_round_trips(ccgbank_root: Path) -> None:
    paths = sorted(
        path for path in ccgbank_root.rglob("*.parg") if path.parent.name == "00"
    )
    assert paths, "LOFI_CCGBANK contains no Section 00 .parg files"

    codec = PargCodec()
    fragment = None

    for path in paths:
        fragment = codec.decode(path.read_bytes(), into=fragment)

    assert fragment is not None
    assert fragment.records

    assert codec.decode(codec.encode(fragment.records)) == fragment

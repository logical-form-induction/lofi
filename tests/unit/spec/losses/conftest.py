from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lofi.spec import RunSpecCodec, config_search_path
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec

if TYPE_CHECKING:
    from lofi.spec.components.model import ModelSection

PACKAGED_RUN = (
    Path(__file__).resolve().parents[4] / "lofi/configs/run/synthetic-parser.yaml"
)
"""The packaged run whose model section the loss declarations are checked against."""


@pytest.fixture(scope="module")
def packaged_model() -> ModelSection:
    """Return the packaged run's model with one span decoder named reconstruction."""
    model = (
        RunSpecCodec(search_path=config_search_path(PACKAGED_RUN.parents[3]))(
            PACKAGED_RUN
        )
    ).model

    return model.with_(span_decoders={"reconstruction": RnnSpanDecoderSpec()})

from lofi.spec._ports import Dependency
from lofi.spec.poolers._dependencies import ENCODER_DEPENDENCY


def test_a_pooler_takes_its_rows_from_the_token_encoder_role() -> None:
    assert (Dependency("token_encoder", "encoder"),) == ENCODER_DEPENDENCY

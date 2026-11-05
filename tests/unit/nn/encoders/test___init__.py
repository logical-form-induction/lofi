from lofi.nn.encoders import (
    HolographicCodebook,
    HolographicEncoderSettings,
    HrrTermEncoder,
    LinearRnnEncoder,
    SharedSubmodule,
    TransformerEncoderSettings,
    TransformerTermEncoder,
    TreeRnnEncoder,
)


def test_exports() -> None:
    assert HolographicCodebook is not None
    assert HolographicEncoderSettings is not None
    assert HrrTermEncoder is not None
    assert LinearRnnEncoder is not None
    assert TreeRnnEncoder is not None
    assert SharedSubmodule is not None
    assert TransformerEncoderSettings is not None
    assert TransformerTermEncoder is not None

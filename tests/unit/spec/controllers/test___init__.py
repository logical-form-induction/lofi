from lofi.spec import controllers
from lofi.spec.controllers import (
    BilinearControllerSpec,
    MlpControllerSpec,
    TransformerControllerSpec,
)


def test_the_package_publishes_its_declaration() -> None:
    assert "MlpControllerSpec" in controllers.__all__
    assert "BilinearControllerSpec" in controllers.__all__
    assert "TransformerControllerSpec" in controllers.__all__
    assert controllers.MlpControllerSpec is MlpControllerSpec
    assert controllers.BilinearControllerSpec is BilinearControllerSpec
    assert controllers.TransformerControllerSpec is TransformerControllerSpec

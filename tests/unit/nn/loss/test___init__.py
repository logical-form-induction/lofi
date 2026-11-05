from lofi.nn import loss


def test_the_package_initializer_does_not_publish_implementation_objects() -> None:
    assert not hasattr(loss, "__all__")
    assert not hasattr(loss, "LossComponentBindings")
    assert not hasattr(loss, "ObjectiveSuite")
    assert not hasattr(loss, "ReadoutId")

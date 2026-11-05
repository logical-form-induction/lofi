import lofi.nn


def test_the_package_initializer_does_not_publish_implementation_objects() -> None:
    assert not hasattr(lofi.nn, "__all__")
    assert not hasattr(lofi.nn, "Encoded")
    assert not hasattr(lofi.nn, "LogDistribution")
    assert not hasattr(lofi.nn, "TermHeads")

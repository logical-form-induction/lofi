from lofi.nn import grounding


def test_the_published_surface_names_every_exported_object() -> None:
    for name in grounding.__all__:
        assert getattr(grounding, name) is not None

    assert grounding.GradientPath.PATHWISE in grounding.GradientPath
    assert isinstance(
        grounding.Marginal(), grounding.Marginal | grounding.Best | grounding.Tempered
    )

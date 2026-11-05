"""Joint translator and registered-engine capabilities."""

from lofi.lf.provers import TranslatorCapability, joint_capability


def test_an_engine_cannot_widen_its_translator_fragment() -> None:
    translator = TranslatorCapability(
        order="first",
        quantifiers=True,
        functions=False,
        equality=False,
        world_types=(),
    )
    engine = TranslatorCapability(
        order="higher",
        quantifiers=True,
        functions=True,
        equality=True,
        world_types=("*",),
    )

    assert joint_capability(translator, engine) == translator


def test_joint_capability_intersects_named_world_resources() -> None:
    translator = TranslatorCapability(
        order="higher",
        quantifiers=True,
        functions=True,
        equality=True,
        world_types=("s", "w"),
    )
    engine = TranslatorCapability(
        order="higher",
        quantifiers=True,
        functions=True,
        equality=True,
        world_types=("w", "i"),
    )

    assert joint_capability(translator, engine).world_types == ("w",)

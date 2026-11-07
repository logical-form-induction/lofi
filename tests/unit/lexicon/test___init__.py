import pytest

from lofi import lexicon
from lofi.lexicon import (
    BeamSource,
    Clamp,
    ClampedSource,
    Entry,
    ErrorPolicy,
    ExhaustedBeamError,
    GrowingBeamSource,
    LexicalSource,
    LexiconError,
    StaticLexicon,
    UnsatisfiableClampError,
    _beam,
    _clamped,
    _errors,
    _ports,
    _static,
)


class Equality[T]:
    """Value equality used by the string-valued fixtures."""

    name = "equality"
    has_key = True

    def key(self, value: T, /) -> T:
        """Return the value as its representative."""
        return value

    def equivalent(self, left: T, right: T, /) -> bool:
        """Return whether two values compare equal."""
        return left == right


EQUALITY = Equality[str]()


def test_the_package_re_exports_its_ports_its_sources_and_its_errors() -> None:
    assert sorted(lexicon.__all__) == sorted(
        [
            *_ports.__all__,
            *_static.__all__,
            *_clamped.__all__,
            *_beam.__all__,
            *_errors.__all__,
        ]
    )
    assert lexicon.Clamp is _ports.Clamp
    assert lexicon.Entry is _ports.Entry
    assert lexicon.LexicalSource is _ports.LexicalSource
    assert lexicon.StaticLexicon is _static.StaticLexicon
    assert lexicon.ClampedSource is _clamped.ClampedSource
    assert lexicon.LexiconError is _errors.LexiconError
    assert lexicon.UnsatisfiableClampError is _errors.UnsatisfiableClampError
    assert lexicon.ExhaustedBeamError is _errors.ExhaustedBeamError
    assert lexicon.BeamSource is _beam.BeamSource
    assert lexicon.GrowingBeamSource is _beam.GrowingBeamSource


def test_the_re_exported_names_are_usable_through_the_package() -> None:
    entry = Entry("cat", "N", weight=-3.0)
    clamp: Clamp[str] = Clamp((0, 1), category={"N"})
    table = StaticLexicon((entry,))

    assert entry == Entry("cat", "N")
    assert clamp.covers(0)
    assert clamp.admits(entry, EQUALITY, EQUALITY)
    assert not isinstance(entry, LexicalSource)
    assert isinstance(table, LexicalSource)
    assert table.axioms(("cat",))[0][1].category == "N"


def test_the_re_exported_clamped_source_narrows_and_refuses() -> None:
    table = StaticLexicon((Entry("cat", "N"), Entry("cat", "NP")))
    clamp: Clamp[str] = Clamp((0, 1), category={"N"})
    source: ClampedSource[str] = ClampedSource(
        table, (clamp,), EQUALITY, EQUALITY, ErrorPolicy()
    )

    assert isinstance(source, LexicalSource)
    assert [entry.category for _, entry, _ in source.axioms(("cat",))] == ["N"]

    with pytest.raises(LexiconError):
        ClampedSource(
            table,
            (Clamp((0, 1), category={"S"}),),
            EQUALITY,
            EQUALITY,
            ErrorPolicy(),
        ).axioms(("cat",))

    with pytest.raises(UnsatisfiableClampError):
        ClampedSource(
            table,
            (Clamp((0, 1), category={"S"}),),
            EQUALITY,
            EQUALITY,
            ErrorPolicy(),
        ).axioms(("cat",))


def test_the_re_exported_beam_sources_seed_widen_and_refuse() -> None:
    beams = ((Entry("cat", "N", weight=-1.0), Entry("cat", "NP", weight=-2.0)),)
    source: BeamSource[str] = BeamSource(beams)
    seeded: GrowingBeamSource[str] = GrowingBeamSource(source)

    assert isinstance(source, LexicalSource)
    assert isinstance(seeded, LexicalSource)
    assert [entry.category for _, entry, _ in seeded.axioms(("cat",))] == ["N"]
    assert seeded.widen(0).widenings == 1

    with pytest.raises(LexiconError):
        seeded.widen(0).widen(0)

    with pytest.raises(ExhaustedBeamError):
        seeded.widen(0).widen(0)

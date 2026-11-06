import pytest

from lofi import formalism


def test_the_published_surface_checks_a_profile_against_a_bound() -> None:
    profile = formalism.DeductionProfile(
        1,
        frozenset({1, 2}),
        unary_closure=True,
        category_set="unknown",
        lexicalized=True,
    )

    bound = formalism.ProfileBound("a chart", 1, frozenset({2}), "a region index")

    with pytest.raises(formalism.ProfileUnsupportedError, match="arities"):
        bound.check(profile, "ccg")


def test_the_published_surface_includes_runtime_grammar_values() -> None:
    assert (
        formalism.Budget(
            depth=5,
            mass=1.0,
            beam=8,
            unary_rounds=1,
            threads=1,
            beta_fuel=1_000,
            max_category_arity=None,
            max_category_size=None,
        ).beam
        == 8
    )
    assert formalism.Formalism.__name__ == "Formalism"
    assert formalism.FormalismDeclaration.__name__ == "FormalismDeclaration"
    assert formalism.Grammar.__name__ == "Grammar"
    assert formalism.TheoryArtifact.__name__ == "TheoryArtifact"

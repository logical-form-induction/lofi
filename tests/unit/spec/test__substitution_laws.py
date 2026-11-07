from lofi.spec._substitution_laws import provide_rule_laws


def test_provider_declares_the_full_substitution_matrices() -> None:
    cases = provide_rule_laws()
    expected = {
        f"substitution/{view}/{role}/{order}/{variant}"
        for view in ("direct", "swapped", "undirected")
        for role in ("forward", "backward")
        for order in (1, 2)
        for variant in ("harmonic", "crossed")
    }

    assert cases.provider == "substitution"
    assert len(cases.ordinary) == 24
    assert len(cases.totality) == 2
    assert {case.name for case in cases.ordinary} == expected
    assert len({case.name for case in cases.totality}) == 2


def test_r3_substitution_cases_are_harmonic_and_bidirectional() -> None:
    names = tuple(case.name for case in provide_rule_laws().totality)

    assert names == (
        "r3/substitution/forward/1/harmonic",
        "r3/substitution/backward/1/harmonic",
    )
